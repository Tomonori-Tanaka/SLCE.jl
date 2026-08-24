# The penalty metric: the per-column scale `m` that makes a quadratic penalty
# independent of how the basis happens to normalize its columns. Everything here is
# estimator-agnostic — the type, its validation, the reference-ensemble generator, and
# the small numerical kernels the solvers share — so it sits below
# `fitting/estimators.jl`, which carries the metric on its structs, and below the two
# `penalty_metric` methods, which live with the bases they measure
# (`fitting/selection.jl` for `SLCEBasis`, `fitting/momentfit.jl` for `MomentBasis`).
#
# SCOPE: the metric is defined for the PURE-SPIN channels only. Its reference ensemble
# is uniform random spin directions, and a displacement factor `|u|^{2k}·R_{lm}(u)` has
# no value on one — the reference distribution of `u` is a modelling decision (which
# amplitude? which temperature? which sector?) that has to be written down before it
# can be sampled. Until that spec exists, `penalty_metric(::SLCEBasis)` refuses a basis
# with displacement content by name rather than measuring something arbitrary.

"""
    MetricProvenance

Where a penalty metric came from: the `channel` it was built for (`:energy` or
`:moment`), the `torque_weight` its reference norms were taken at (`0.0` on the moment
channel, which has no torque block), the reference-ensemble size `nconfig` and its
`seed`, and the `fingerprint` of the SALC basis it was built on. Carried on the
estimator so a fit can refuse a metric built for a different problem. The metric enters
every penalized coefficient, and a mismatch is invisible to the scale-invariance gates
— those hold for any `m ∝ c²`, right or wrong — so it has to be checked from outside.
"""
struct MetricProvenance
    channel::Symbol
    torque_weight::Float64
    nconfig::Int
    seed::Int
    fingerprint::UInt64

    function MetricProvenance(channel::Symbol, torque_weight::Real, nconfig::Integer,
                              seed::Integer, fingerprint::UInt64)
        channel in (:energy, :moment) || throw(ArgumentError(
            "metric channel must be :energy or :moment; got :$channel"))
        (isfinite(torque_weight) && 0 <= torque_weight <= 1) || throw(ArgumentError(
            "metric torque_weight must be in [0, 1]; got $torque_weight"))
        nconfig >= 2 ||
            throw(ArgumentError("metric nconfig must be ≥ 2; got $nconfig"))
        seed >= 0 || throw(ArgumentError("metric seed must be ≥ 0; got $seed"))
        return new(channel, Float64(torque_weight), Int(nconfig), Int(seed),
                   fingerprint)
    end
end

Base.show(io::IO, p::MetricProvenance) =
    print(io, "MetricProvenance(", p.channel, ", w=", p.torque_weight, ", K=",
          p.nconfig, ", seed=", p.seed, ")")

# --- penalty metric: shared validation and use ------------------------------------

# Validate a penalty metric at construction. `nothing` means "uniform" and is kept as a
# distinct value from an all-ones vector so `show`, the provenance check, and the
# byte-equality gates can tell "no metric" from "a metric that happens to be flat".
# Length is NOT checked here — a constructor does not see the design matrix, so that
# check lives at the `solve_coefficients` door.
function _validated_metric(metric)::Union{Nothing,Vector{Float64}}
    metric === nothing && return nothing
    m = Vector{Float64}(metric)
    isempty(m) &&
        throw(ArgumentError("metric must be nonempty (one entry per design column)"))
    all(x -> isfinite(x) && x >= 0, m) ||
        throw(ArgumentError("metric entries must be finite and ≥ 0"))
    any(>(0), m) || throw(ArgumentError(
        "metric is zero on every column, so no column would be penalized — that is " *
        "`OLS()`, not a penalized fit. Use `OLS()` if that is the intent."))
    return m
end

# The number of β-space columns a metric must cover. Under an ASR / freeze
# reparameterization the design handed to the solver lives in γ space (`size(X, 2) =
# q`), while the metric — like `column_groups`, and for the same reason — is indexed by
# the BASIS columns `β`. `size(nullspace, 1)` is that count.
_beta_columns(X::AbstractMatrix, nullspace)::Int =
    nullspace === nothing ? size(X, 2) : size(nullspace, 1)

# The per-column penalty scale the solvers see: the estimator's metric, or the uniform
# metric when none was supplied. `1.0 * x === x` for every Float64, so a `nothing`
# metric reproduces the unweighted penalty bit for bit and the solvers below need no
# separate no-metric branch.
function _metric_vector(metric::Union{Nothing,Vector{Float64}}, p::Int,
                        what::AbstractString)::Vector{Float64}
    metric === nothing && return ones(Float64, p)
    length(metric) == p || throw(DimensionMismatch(
        "$what metric length $(length(metric)) does not match design-matrix column " *
        "count $p; the metric was likely built on a different basis."))
    return metric
end

# `λ·D` in the coordinates the normal equations are solved in. `D === nothing` is the
# UNIFORM penalty and is kept as its own branch rather than folded into a vector of
# ones: under a reparameterization `Z` the uniform penalty is `λ·I` in γ space EXACTLY
# (`Z` is orthonormal), which is cheaper, more accurate, and — decisively — bit-for-bit
# what the pre-metric code did. `D` is always in β space; `Z'·D·Z` compresses it, which
# is where the weight map already lived before the metric existed.
_penalty_matrix(lambda::Float64, D::Union{Nothing,Vector{Float64}}, q::Int,
                nullspace::Union{Nothing,Matrix{Float64}}) =
    D === nothing ? lambda * I(q) :
    nullspace === nothing ? lambda * Diagonal(D) :
                            lambda * Symmetric(nullspace' * (D .* nullspace))

# Unpenalized columns (`mⱼ == 0`) leave the normal equations singular unless the
# directions they leave free are themselves determined by the data:
# `v'Av = ‖X̃v‖² + λ·Σⱼ mⱼ·(Zv)ⱼ²` vanishes only on the subspace where the second term
# does, so positive definiteness is exactly a rank condition on the design restricted
# to that subspace. A `Symmetric` solve on a singular matrix does not reliably throw —
# it can return numerical garbage instead (measured at ‖β‖ ~ 1e16 against OLS's 0.48 on
# a design with one duplicated column) — so refuse by name first.
#
# The test runs on the free block's GRAM, whose eigenvalues are the squared singular
# values of the design there, and cuts its condition number at `1/eps`. That is
# `κ ≲ 6.7e7`: the normal equations square the condition number, so past roughly `1e8`
# the unpenalized coefficients carry no significant digits whether or not the
# factorization reports failure. A Cholesky with `check = false` is not enough — on an
# exactly duplicated column the trailing pivot lands a rounding step above zero and
# succeeds — so the test is on the spectrum. The cut stays well inside `_rank_df`'s
# `min(size)·eps·σ₁`, which `_edof_free` applies to the same block, so a solve accepted
# here is never rejected by the diagnostic afterwards.
const _FREE_BLOCK_GRAM_RCOND = eps(Float64)

# The unpenalized subspace, as a basis of the coordinates `XtX` is expressed in.
# Without a reparameterization it is the coordinate axes `mⱼ == 0` — reported as
# indices, so the message can name columns. Under `Z` an unpenalized β column is no
# longer a γ coordinate, so the subspace is `null(diag(√m)·Z)`, taken from an SVD;
# for the selection `Z` a freeze produces this reduces to the index case exactly.
function _free_directions(metric::Vector{Float64},
                          nullspace::Union{Nothing,Matrix{Float64}})
    nullspace === nothing && return findall(iszero, metric)
    N = sqrt.(metric) .* nullspace
    F = svd(N)
    q = size(nullspace, 2)
    tolr = isempty(F.S) ? 0.0 : max(size(N)...) * eps(Float64) * F.S[1]
    k = count(>(tolr), F.S)
    return view(F.V, :, (k + 1):q)
end

function _check_free_block(XtX::AbstractMatrix, metric::Vector{Float64},
                           what::AbstractString;
                           columns::Union{Nothing,Vector{Int}} = nothing,
                           nullspace::Union{Nothing,Matrix{Float64}} = nothing)
    any(iszero, metric) || return nothing
    free = _free_directions(metric, nullspace)
    (free isa AbstractVector ? isempty(free) : size(free, 2) == 0) && return nothing
    G = free isa AbstractVector ? Matrix{Float64}(XtX[free, free]) :
        Matrix{Float64}(free' * XtX * free)
    ev = eigvals(Symmetric(G))
    if !(ev[end] > 0 && ev[1] > _FREE_BLOCK_GRAM_RCOND * ev[end])
        named = free isa AbstractVector ?
                (columns === nothing ? free : columns[free]) :
                "spanning $(size(free, 2)) direction(s) of the reparameterized design"
        kappa = ev[1] > 0 ? sqrt(ev[end] / ev[1]) : Inf
        throw(ArgumentError(
            "$what: the unpenalized columns $named (positions in the design as " *
            "SOLVED — a channel that freezes columns solves on a subset) are " *
            "linearly dependent (or nearly " *
            "so) on this design — their condition number is $kappa, against a limit " *
            "of $(sqrt(1 / _FREE_BLOCK_GRAM_RCOND)) — so the penalized normal " *
            "equations are singular and those coefficients are not identified. " *
            "Penalize the columns, or drop the dependent ones."))
    end
    return nothing
end

# Relative ∞-norm change of an IRLS iterate, measured in metric coordinates `√mⱼ·βⱼ`
# and over the PENALIZED columns only. `√mⱼ·βⱼ` is the invariant of a column rescaling
# (`Φⱼ → cⱼΦⱼ` sends `βⱼ → βⱼ/cⱼ` and `mⱼ → cⱼ²mⱼ`), so the stopping rule moves with the
# penalty instead of against it; restricting to penalized columns keeps one large
# unpenalized coefficient from turning the relative tolerance into an absolute one far
# coarser than the coefficients it is meant to converge. With a uniform metric and no
# unpenalized column this is exactly the plain relative ∞-norm rule the pre-metric code
# used, bit for bit. `smetric` is `sqrt.(metric)`, hoisted out of the iteration by the
# callers; both iterates are in β space (γ's ∞-norm is factorization-gauge-dependent
# under `Z`, which is why the callers lift before measuring).
function _irls_rel_change(beta_new::Vector{Float64}, beta::Vector{Float64},
                          smetric::Vector{Float64})::Float64
    num = 0.0
    den = 0.0
    @inbounds for j in eachindex(beta_new)
        smetric[j] > 0.0 || continue
        d = smetric[j] * abs(beta_new[j] - beta[j])
        d > num && (num = d)
        a = smetric[j] * abs(beta_new[j])
        a > den && (den = a)
    end
    return num / max(den, eps(Float64))
end

# One-line penalty-metric summary for `show`: how many columns it covers, how many of
# them it leaves unpenalized (the number a reader needs to sanity-check an intercept
# exemption without printing the whole vector), and what it was built from — the
# reproducibility record a user would otherwise have to dig out of the struct.
_metric_summary(::Nothing, ::Any) = "metric=uniform"
_metric_summary(m::Vector{Float64}, pv) =
    string("metric=", length(m), " columns, ", count(iszero, m), " unpenalized",
           pv === nothing ? "" : string(", from ", pv))

# The `metric` keyword of a basis-aware constructor: `:basis` builds the reference
# metric (handled by the caller), `nothing` is the uniform penalty, a vector is the
# caller's own. Any other symbol is a guess at the sentinel's spelling and is refused
# by name — falling through would die inside `Vector{Float64}(::Symbol)` with a bare
# `MethodError`.
function _checked_metric_keyword(metric)
    metric isa Symbol && throw(ArgumentError(
        "metric = :$metric is not a recognized sentinel. Use `:basis` to build the " *
        "reference metric from this basis, `nothing` for the unweighted penalty, or " *
        "pass a vector of per-column scales."))
    return metric
end

# The byte budget of one chunk of a chunked reference design. A metric is built inside
# a constructor, so the peak buffer has to be bounded by something that does not grow
# with the basis; this bounds it directly instead of bounding a row count.
const _METRIC_CHUNK_BYTES = 32 * 1024 * 1024

# SplitMix64 (Steele, Lea & Flood 2014), the reference-ensemble generator. Julia's
# `MersenneTwister` stream carries no cross-version stability guarantee, and this
# metric enters EVERY penalized coefficient — a stream that drifted between Julia
# releases would silently move every recorded penalized fit and every regression pin
# without a line of this package changing. SplitMix64 is fully specified by the four
# constants below, so the sequence is fixed for good.
@inline function _splitmix64(state::UInt64)::Tuple{UInt64,UInt64}
    s = state + 0x9e3779b97f4a7c15
    z = s
    z = (z ⊻ (z >> 30)) * 0xbf58476d1ce4e5b9
    z = (z ⊻ (z >> 27)) * 0x94d049bb133111eb
    return s, z ⊻ (z >> 31)
end

# `nconfig` uniform-random spin configurations on `nat` atoms — the reference ensemble
# the penalty metric averages over. Uniformity on the sphere is exact rather than
# approximate: `cosθ` uniform on [−1, 1) with `φ` uniform on [0, 2π) is the Archimedes
# construction, so there is no rejection loop and no normal deviate (either would make
# the stream depend on library internals the constants above are chosen to avoid).
function _reference_configs(nat::Int, nconfig::Int, seed::Int)::Vector{Matrix{Float64}}
    nat >= 1 || throw(ArgumentError("nat must be ≥ 1; got $nat"))
    nconfig >= 2 || throw(ArgumentError(
        "the reference ensemble needs ≥ 2 configurations to carry a variance; got " *
        "$nconfig"))
    seed >= 0 || throw(ArgumentError("seed must be ≥ 0; got $seed"))
    state = UInt64(seed)
    cfgs = Vector{Matrix{Float64}}(undef, nconfig)
    for c = 1:nconfig
        e = Matrix{Float64}(undef, 3, nat)
        for a = 1:nat
            state, z1 = _splitmix64(state)
            state, z2 = _splitmix64(state)
            ez = 2.0 * (Float64(z1 >> 11) * 0x1p-53) - 1.0
            phi = 2π * (Float64(z2 >> 11) * 0x1p-53)
            r = sqrt(max(0.0, 1.0 - ez * ez))
            e[1, a] = r * cos(phi)
            e[2, a] = r * sin(phi)
            e[3, a] = ez
        end
        cfgs[c] = e
    end
    return cfgs
end

# An accidental zero in a metric would silently unpenalize a column. Zero is reserved
# for a STRUCTURAL exemption — an intercept, an identically vanishing column — so a
# numerically-zero estimate is refused rather than floored. `exempt` names the columns
# whose zero IS structural; the report always uses original column indices, since a
# position inside a filtered vector is an index the reader cannot act on.
function _refuse_zero_metric(m::Vector{Float64}, what::AbstractString;
                             exempt::AbstractVector{Int} = Int[],
                             hint::AbstractString = "")
    bad = setdiff(findall(iszero, m), exempt)
    isempty(bad) && return nothing
    throw(ArgumentError(
        "$what: columns $bad have zero reference norm. Zero marks an unpenalized " *
        "column, so it is reserved for a structural exemption and never inferred " *
        "from a sample. A column that is identically zero on the reference ensemble " *
        "carries no information and should be removed from the basis." * hint))
end
