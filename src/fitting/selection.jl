# Model selection for the fit-accuracy-vs-Monte-Carlo-cost trade-off: group labels and
# a-priori MC costs over an `SLCEBasis` (the fixed weights of `GroupAdaptiveRidge`), GCV /
# effective degrees of freedom for linear estimators, and the λ-path driver with the
# cost-aware Pareto selection rule. Lives after `fit.jl`/`diagnostics.jl` in the include
# order because it needs `SLCEBasis`, `SLCEDataset`, `SLCEFit`, and `_assemble_problem`.

"""
    salc_groups(basis::SLCEBasis) -> Vector{Int}

Per-design-matrix-column group labels (contiguous `1:G`, one label per SALC in
`SALCKey` order): columns grouped by `(key.body, key.orbit_id, key.decors)`. This is
the granularity at which Monte-Carlo contraction entries vanish — all `L_S` / `Lf` /
`block` channels of one cluster orbit and decoration multiset share their entry
support, so an entry
disappears only when **every** coefficient of the group is zero. Feed the labels to
[`GroupAdaptiveRidge`](@ref) (or use the `GroupAdaptiveRidge(basis; ...)` convenience
constructor, which calls this for you).
"""
function salc_groups(basis::SLCEBasis)::Vector{Int}
    ks = basis.salc_basis.keys
    labels = Vector{Int}(undef, length(ks))
    g = 0
    for j in eachindex(ks)
        # keys are sorted by (body, orbit_id, decors, L_S, Lf, block), so equal
        # (body, orbit_id, decors) runs are contiguous — label at the change points
        if j == 1 || (ks[j].body, ks[j].orbit_id, ks[j].decors) !=
                     (ks[j-1].body, ks[j-1].orbit_id, ks[j-1].decors)
            g += 1
        end
        labels[j] = g
    end
    return labels
end

# One Monte-Carlo contraction entry: (member sites, per-slot factor labels, tensor
# index). The slot labels are `_slotkey` tuples `(channel, site, k, l)` rather than
# `Slot`s: the value is identical, and keying a `Set` on an explicit tuple is this
# package's convention where determinism matters.
#
# The slot vector is part of the key ON PURPOSE, and not only to separate channels.
# The MC pays a site program PER SLOT — `_push_term_programs!` emits one program per
# member site position, each of `|axes(q)|·nnz` entries — so an entry's sweep cost is
# `length(slots)`, readable off the key itself. Keying on the pure-spin per-site `ls`
# instead (what this did before) drops every displacement slot from the key, which
# both collapses distinct lattice groups onto one another and loses that factor.
const _SlotKey = Tuple{Channel,Int,Int,Int}
const _EntryKey = Tuple{Vector{Int},Vector{SVector{3,Int}},Vector{_SlotKey},Vector{Int}}

# A term's slot labels in its canonical slot order (SPIN axes before DISP, each by
# member site order) — the order `_canonicalize_members` fixes and the one
# SLCEMonteCarlo's `_align_reduced` reproduces, so equal terms give equal keys.
_slotkeys(t::SALCTerm)::Vector{_SlotKey} = _SlotKey[_slotkey(s) for s in t.slots]

# Function barrier over the rank-erased `folded::Array{Float64}` (runtime rank = slot
# count): pushes one `_EntryKey` per nonzero tensor element.
function _push_entries!(set::Set{_EntryKey}, atoms::Vector{Int},
                        shifts::Vector{SVector{3,Int}}, slotkeys::Vector{_SlotKey},
                        folded::Array{Float64})::Nothing
    for idx in CartesianIndices(folded)
        # exact != 0.0 mirrors the skip in `_push_term_programs!`. Note the MC's SITE
        # programs skip on `coef * folded[idx] == 0.0`, which a basis-level metric
        # cannot evaluate (there is no fitted coefficient yet); the divergence is
        # underflow-only and in the lower-bound direction. Do not soften to isapprox.
        if folded[idx] != 0.0
            push!(set, (atoms, shifts, slotkeys, collect(Tuple(idx))))
        end
    end
    return nothing
end

# Contiguity check shared by `group_costs` (the `GroupAdaptiveRidge` inner constructor
# performs the same validation on its own copy): labels cover 1:G with no gaps.
function _validate_labels(labels::AbstractVector{<:Integer}, n::Int, what::String)
    length(labels) == n ||
        throw(ArgumentError("$what length $(length(labels)) ≠ number of SALCs $n"))
    isempty(labels) && return 0
    minimum(labels) >= 1 ||
        throw(ArgumentError("$what labels must be ≥ 1; got $(minimum(labels))"))
    G = Int(maximum(labels))
    counts = zeros(Int, G)
    for g in labels
        counts[g] += 1
    end
    all(>(0), counts) ||
        throw(ArgumentError("$what labels must cover 1:$G with no gaps; " *
                            "empty group(s): $(findall(==(0), counts))"))
    return G
end

"""
    group_costs(basis::SLCEBasis,
                column_groups::AbstractVector{<:Integer} = salc_groups(basis);
                asr::Union{Nothing,ASRReparam} = nothing)
        -> Vector{Int}

Per-group Monte-Carlo **sweep** cost: over the distinct contraction entries — keys
`(member sites, per-slot factor labels, nonzero tensor index)` — in the union over the
group's SALCs (canonical members), the sum of each entry's slot count.

The slot count is the point. A single-site Metropolis proposal re-evaluates every
entry containing the moved site, and `SLCEMonteCarlo`'s `_push_term_programs!` emits
one **site program per member site position**, each of `|axes(q)|·nnz` entries — so
one instance costs `nnz · length(slots)` per sweep, not `nnz`. `nnz` alone is the size
of the separate **energy** program, which `total_energy` walks once per run. Pricing
`nnz` therefore mis-ranks across body orders (a 3-body group is priced at 2/3 of a
2-body group's real sweep cost) and, once displacement slots exist, across groups of
equal body order, since `length(slots)` can reach `2·body`.

This is an **a-priori proxy, and a lower bound**: an entry vanishes only when the whole
group is zero, so pricing groups by it is sound, but nothing downstream actually merges
two SALCs that touch one entry key (`decorated_terms` emits one term per
`(SALC, member, SALCTerm)`, and `TiledHamiltonian` tiles without merging), so the union
undercounts a group whose channels are realized separately. What it does **not** try to
model is the per-visit `O(nrows)` block (`fill!` plus the `delta_energy` row range):
`nrows` comes from `row_layout` over the basis *keys* and never shrinks when a group is
dropped, so that cost is selection-invariant and unattributable to any group. Sweep
*multiplicity* — that a site may be visited by the spin, overrelaxation and displacement
passes — is likewise not modelled here; it depends on run-time `UpdatePlan` counts and
on which other groups survive.

Costs are additive across the [`salc_groups`](@ref) partition (distinct
`(body, orbit_id, decors)` groups never share an entry key: distinct `orbit_id` gives
disjoint canonical `(atoms, shifts)`, and equal `orbit_id` with distinct `decors` gives
distinct sorted slot labels). `column_groups` may also be any coarser contiguous `1:G`
partition of the columns — the per-entry slot count keeps the sum additive under
coarsening, which a per-group multiplier would not.

Pass `asr` (a [`SLCEDataset`](@ref)'s `.asr`) to price **structurally infeasible**
groups at zero. A group every one of whose columns is annihilated by the constraint
([`group_freedom`](@ref) `s_g == 0`) can never be nonzero in a translation-invariant
model, so charging a Monte-Carlo sweep cost for it is a pure over-report — and not a
small one: on a `pmax = 1` spin+displacement truncation whose displacement content the
ASR kills outright, the dead group carries 94.7 % of `Σ_g c_g`. The discount is opt-in
so that a cost computed from the basis alone stays a property of the basis.
"""
function group_costs(basis::SLCEBasis,
                     column_groups::AbstractVector{<:Integer} =
                         salc_groups(basis);
                     asr::Union{Nothing,ASRReparam} = nothing)::Vector{Int}
    sl = basis.salc_basis.salcs
    G = _validate_labels(column_groups, length(sl), "group_costs")
    sets = [Set{_EntryKey}() for _ = 1:G]
    for j in eachindex(sl)
        set = sets[column_groups[j]]
        for m in sl[j].members, t in m.terms
            _push_entries!(set, m.atoms, m.shifts, _slotkeys(t), t.folded)
        end
    end
    # `length(k[3])` is the entry's slot count — the number of site programs the MC
    # emits for it. Reading it off the key (rather than off a per-group constant)
    # is what keeps the sum additive under a coarser `column_groups`.
    c = [sum(k -> length(k[3]), s; init = 0) for s in sets]
    asr === nothing && return c
    size(asr.Z, 1) == length(sl) || throw(DimensionMismatch(
        "asr covers $(size(asr.Z, 1)) columns but the basis has $(length(sl))"))
    # A group survives the discount if ANY of its columns is feasible — the discount
    # must not fire on a group the constraint merely restricts.
    live = falses(G)
    for j in eachindex(sl)
        norm(@view asr.Z[j, :]) < _ASR_DEAD_ROW && continue
        live[column_groups[j]] = true
    end
    return [live[g] ? c[g] : 0 for g = 1:G]
end

"""
    group_freedom(rep::ASRReparam, column_groups::AbstractVector{<:Integer})
        -> Vector{Float64}

How much freedom the ASR leaves each group: `s_g = ‖Z[g, :]‖_F²`, the squared Frobenius
norm of the group's row block of the null-space basis `Z`.

`Z` is orthonormal, so `Σ_g s_g == q = size(Z, 2)` **exactly** — `s_g` reads as "how
many of the `q` feasible directions this group holds", and `0 ≤ s_g ≤ p_g`. Despite
being written in terms of `Z`, it is gauge-invariant: it is `tr` of the orthogonal
projector `Z·Z'` onto `null(A)` restricted to the group's rows, so it does not depend
on the factorization that produced `Z` (which nothing may persist).

`s_g == 0` ⟺ every column of the group is structurally zeroed — no translation-
invariant model can carry the group at any support. That is the condition
[`group_costs`](@ref)`(...; asr)` prices at zero, and the group-resolved form of the
basis-level warning `build_asr` emits.

!!! note "Groups are not the ASR's own granularity"
    A small `s_g` is not a licence to drop the group independently. The constraint's
    all-or-nothing atoms are the *circuits* of `A`'s column matroid, which are small
    (2–3 columns measured) but cross group boundaries, so a single
    `(body, orbit_id, decors)` group generally has **no** feasible subspace on its own:
    `dim null(A[:, cols_g]) == 0` held for every displacement-touched group of every
    fixture measured. Read `s_g` as a diagnostic of what the truncation admits, not as
    a per-group cost that selection can independently recover.
"""
function group_freedom(rep::ASRReparam,
                       column_groups::AbstractVector{<:Integer})::Vector{Float64}
    Z = rep.Z
    G = _validate_labels(column_groups, size(Z, 1), "group_freedom")
    s = zeros(Float64, G)
    for j in axes(Z, 1)
        s[column_groups[j]] += sum(abs2, @view Z[j, :])
    end
    return s
end

"""
    cost_weights(basis::SLCEBasis; cost_exponent::Real = 1.0)
        -> (; column_groups::Vector{Int}, weights::Vector{Float64})

Fixed [`GroupAdaptiveRidge`](@ref) weights over the [`salc_groups`](@ref) partition:

    v_g = √p_g · (c_g / c̄)^θ,    θ = cost_exponent

with `p_g` the group's column count (the Yuan–Lin group-size factor), `c_g =`
[`group_costs`](@ref) and `c̄` their mean. `cost_exponent ∈ [0, 1]` — the derivation in
`docs/design-notes.md` §13 writes it `θ` — sets the cost-vs-accuracy tilt of the
penalty: `0` is cost-blind group selection, `1` penalizes each group in proportion to
its Monte-Carlo cost, so an expensive group must earn its keep with a correspondingly
larger error reduction. Sweeping it changes the *order* in which groups are eliminated
along a λ path, so the lower envelope over several values traces the (cost, error)
Pareto front (see [`select_fit`](@ref)).
"""
function cost_weights(basis::SLCEBasis; cost_exponent::Real = 1.0)
    (0 <= cost_exponent <= 1) || throw(ArgumentError(
        "cost_exponent must be in [0, 1]; got $cost_exponent"))
    column_groups = salc_groups(basis)
    c = group_costs(basis, column_groups)
    isempty(c) && return (; column_groups, weights = Float64[])
    all(>(0), c) && all(isfinite, c) ||
        error("internal: a canonical SALC group has no nonzero tensor entry")
    G = length(c)
    p = zeros(Int, G)
    for g in column_groups
        p[g] += 1
    end
    cbar = mean(c)
    t = Float64(cost_exponent)
    weights = [sqrt(p[g]) * (c[g] / cbar)^t for g = 1:G]
    return (; column_groups, weights)
end

"""
    GroupAdaptiveRidge(basis::SLCEBasis; lambda, cost_exponent = 1.0, epsilon = 1e-8,
                       max_iter = 50, tol = 1e-6)

Cost-weighted group estimator for `basis`: [`salc_groups`](@ref) column labels with the
fixed [`cost_weights`](@ref)`(basis; cost_exponent)` weights. See the primary
[`GroupAdaptiveRidge`](@ref) constructor for the estimator itself.
"""
function GroupAdaptiveRidge(basis::SLCEBasis; lambda::Real, cost_exponent::Real = 1.0,
                            epsilon::Real = 1e-8, max_iter::Integer = 50,
                            tol::Real = 1e-6)
    lw = cost_weights(basis; cost_exponent = cost_exponent)
    return GroupAdaptiveRidge(lw.column_groups, lw.weights; lambda = lambda,
                              epsilon = epsilon,
                              max_iter = max_iter, tol = tol)
end

# --- GCV / effective degrees of freedom -------------------------------------------

# The converged quadratic-penalty diagonal `(lambda, w)` of a linear estimator: the
# fitted values are `ŷ = X(X'X + λ·Diagonal(w))⁻¹X'y` with `w` frozen at the fitted
# coefficients. `w === nothing` ⇔ unpenalized (OLS, or λ = 0), where the effective dof
# is the design rank. The adaptive members recompute their converged diagonal from the
# fitted `beta` through the SAME weight formulas the solvers iterate (`AdaptiveRidge`'s
# `1/(β² + ε)`, `_gar_weights!` for the group form) — the formula lives in one place.
_penalty_diagonal(::OLS, beta::Vector{Float64}) = (0.0, nothing)
function _penalty_diagonal(est::Ridge, beta::Vector{Float64})
    est.lambda == 0.0 && return (0.0, nothing)
    return (est.lambda, ones(Float64, length(beta)))
end
function _penalty_diagonal(est::AdaptiveRidge, beta::Vector{Float64})
    est.lambda == 0.0 && return (0.0, nothing)
    return (est.lambda, @.(1.0 / (beta^2 + est.epsilon)))
end
function _penalty_diagonal(est::GroupAdaptiveRidge, beta::Vector{Float64})
    est.lambda == 0.0 && return (0.0, nothing)
    length(beta) == length(est.column_groups) || throw(DimensionMismatch(
        "coefficient length $(length(beta)) ≠ column_groups length " *
        "$(length(est.column_groups))"))
    w = Vector{Float64}(undef, length(beta))
    normsq = Vector{Float64}(undef, length(est.group_weights))
    _gar_weights!(w, beta, est.column_groups, est.group_weights, est.group_sizes,
                  est.epsilon, normsq)
    return (est.lambda, w)
end
_penalty_diagonal(est::AbstractEstimator, beta::Vector{Float64}) =
    throw(ArgumentError("gcv/effective_dof require a linear estimator " *
                        "(`islinear`); got $(typeof(est))"))

# Numerical-rank dof of the unpenalized smoother. The cut is `min(size(X))·eps·σ₁` —
# `LinearAlgebra.rank`'s actual default (its docstring: "n is the size of the SMALLEST
# dimension"), and the same convention `identifiability` (`fitting/diagnostics.jl`)
# picks with the reasoning written out there: the blocks are whitened by `1/√n`, so the
# singular values are sample-size-independent while a `max`-based cut GROWS with the row
# count, and merely adding data could flip a determined direction to "flat". This used
# to read `maximum(size(X))`, which undercounted the rank of any design with more rows
# than columns — i.e. every energy-only fit with many configurations, and every co-fit —
# making `gcv`'s `df` too small and its score optimistically biased.
function _rank_df(X::Matrix{Float64})::Float64
    s = svdvals(X)
    (isempty(s) || s[1] == 0.0) && return 0.0
    tolr = minimum(size(X)) * eps(Float64) * s[1]
    return Float64(count(>(tolr), s))
end

# Penalized effective dof `tr(X(X'X + λ·Diagonal(w))⁻¹X') = Σᵢ sᵢ/(sᵢ + λ)` over the
# eigenvalues `s` of the weighted Gram `X̃'X̃` (`p ≤ n`) or its dual `X̃X̃'` (`n < p`),
# `X̃ = X·D^{-1/2}` — always an eigenproblem on the smaller side, never an `n × p` SVD.
# A λ-path caller passes its cached `XtX` so the `p ≤ n` branch touches only `p × p`
# data per λ. Tiny negative eigenvalues from roundoff are clamped out.
function _edof(X::Matrix{Float64}, lambda::Float64, w::Vector{Float64};
               XtX::Union{Nothing,Matrix{Float64}} = nothing)::Float64
    n, p = size(X)
    M = if p <= n
        if XtX === nothing
            Xt = X ./ sqrt.(w)'
            Symmetric(Xt' * Xt)
        else
            isq = 1.0 ./ sqrt.(w)
            Symmetric(isq .* XtX .* isq')          # D^{-1/2}·X'X·D^{-1/2}
        end
    else
        Xt = X ./ sqrt.(w)'
        Symmetric(Xt * Xt')
    end
    df = 0.0
    for s in eigvals(M)
        s > 0.0 || continue
        df += s / (s + lambda)
    end
    return df
end

# ASR generalization of `_edof`: the β-space penalty diagonal `w` compresses to the
# dense SPD matrix `P = Z'·D·Z` in γ space, so the diagonal-whitening shortcut is
# silently wrong under `Z` — whiten by the Cholesky factor of `P` instead (design
# record §6 amendment 6). `X` here is the γ-space design `X̃ = X_β·Z` (q columns);
# q is small for joint bases, so the primal q × q eigenproblem suffices.
function _edof_ns(X::Matrix{Float64}, lambda::Float64, w::Vector{Float64},
                  Z::Matrix{Float64})::Float64
    all(>(0.0), w) ||
        throw(ArgumentError("effective_dof/gcv: the penalty diagonal has a " *
                            "nonpositive weight (a zero GroupAdaptiveRidge " *
                            "group weight?) — the compressed penalty Z'DZ must " *
                            "be positive definite"))
    P = Symmetric(Z' * (w .* Z))
    C = cholesky(P)
    G = X' * X
    W = (C.U' \ G) / C.U                      # U⁻ᵀ·(X̃'X̃)·U⁻¹, P = U'U
    df = 0.0
    for s in eigvals(Symmetric(Matrix(W)))
        s > 0.0 || continue
        df += s / (s + lambda)
    end
    return df
end

# Rows whose whitening weight is exactly zero (the energy block at
# w_T + w_F = 1) carry no information; GCV's `n` must not count them or the score
# is systematically optimistic. They stay in the SOLVE stack (bitwise compatibility
# of the fitted coefficients) — only the counting changes.
# The zero-weight condition is the SAME expression `_assemble_problem` scales
# with (`√(max(0, 1 − wT − wF)/n_E)`) — a `wT + wF >= 1` test can disagree with
# it by one rounding at sums that straddle 1 (e.g. 1/3 + 2/3).
_gcv_neff(f::SLCEFit)::Int =
    (max(0.0, 1 - f.torque_weight - f.force_weight) == 0.0 ? 0 :
     size(f.dataset.X_E, 1)) +
    (f.torque_weight > 0 ? length(f.dataset.y_T) : 0) +
    (f.force_weight > 0 ? length(f.dataset.y_F) : 0)

# The coefficient vector the estimator's own weight map was evaluated at: the solve
# runs on γ with `β = beta_p + Z·γ`, so the penalized quantity is `jphi − beta_p`
# (identical to `jphi` for every non-staged fit — `beta_p ≡ 0`).
#
# NOTE: keep this ABOVE the docstring below. Anything between a docstring and the
# function it documents steals it, and Documenter then reports "no docs found"
# while the unit suite stays green.
_penalty_beta(f::SLCEFit, rep::Union{Nothing,ASRReparam})::Vector{Float64} =
    rep === nothing ? f.jphi : f.jphi .- rep.beta_p

"""
    effective_dof(f::SLCEFit) -> Float64

Effective degrees of freedom of a linear-estimator fit: `tr(H) + 1`, where `H =
X(X'X + λ·Diagonal(w))⁻¹X'` is the hat matrix of the assembled (centered / whitened)
problem with the penalty diagonal frozen at the fitted coefficients, and the `+1`
counts the analytic intercept `j0`. For an unpenalized fit ([`OLS`](@ref), or
`lambda = 0`) this is the design rank `+1`. Distinct from [`dof`](@ref), the raw
parametric count. Linear estimators only ([`islinear`](@ref)); the adaptive members
([`AdaptiveRidge`](@ref) / [`GroupAdaptiveRidge`](@ref)) are handled in the standard
converged-weight sense. On an ASR-constrained fit the hat matrix lives in the
reparameterized (γ) space — the β-space penalty compresses to `Z'·D·Z` — so the
value is bounded by `p − rank(A) + 1`.
"""
function effective_dof(f::SLCEFit)::Float64
    islinear(f.estimator) || throw(ArgumentError(
        "effective_dof requires a linear estimator (`islinear`); " *
        "got $(typeof(f.estimator))"))
    rep = f.reparam                       # the STAGE's Z on a staged fit
    X, _, _, _, _ = _assemble_problem(f.dataset, f.torque_weight, f.force_weight, rep)
    # The weight map must be evaluated where the SOLVER evaluated it: at β = Z·γ,
    # i.e. `jphi − beta_p`. On an affine stage the offset is nonzero, and for a
    # GROUP penalty a frozen column's value would otherwise leak into a group norm
    # the solve never saw.
    lambda, w = _penalty_diagonal(f.estimator, _penalty_beta(f, rep))
    df = w === nothing ? _rank_df(X) :
         (rep === nothing ? _edof(X, lambda, w) : _edof_ns(X, lambda, w, rep.Z))
    return df + 1.0
end

# The GCV score (and the effective dof it used, intercept included) on an already-
# assembled problem; shared by `gcv(::SLCEFit)` and the `select_fit` λ-path driver
# (which passes its cached `XtX`). `w === nothing` ⇔ unpenalized. `beta` must be in
# the SAME coordinate space as `X`'s columns (γ under ASR — the caller compresses).
# `n_eff` is the informative row count (`size(X, 1)` minus zero-weight rows). The
# score is `Inf` when `df` approaches `n_eff` (the near-interpolating regime, where
# the GCV denominator loses meaning); the ≥ 1 slack keeps the score from exploding
# on rounding when `df ≈ n_eff`.
function _gcv_score(X::Matrix{Float64}, y::Vector{Float64}, beta::Vector{Float64},
                    lambda::Float64, w::Union{Nothing,Vector{Float64}};
                    XtX::Union{Nothing,Matrix{Float64}} = nothing,
                    Z::Union{Nothing,Matrix{Float64}} = nothing,
                    n_eff::Int = size(X, 1),
                    )::Tuple{Float64,Float64}
    n = n_eff
    df = (w === nothing ? _rank_df(X) :
          (Z === nothing ? _edof(X, lambda, w; XtX = XtX) :
           _edof_ns(X, lambda, w, Z))) + 1.0
    n - df < max(1.0, 1e-8 * n) && return (Inf, df)
    rss = sum(abs2, y .- X * beta)
    return (n * rss / (n - df)^2, df)
end

"""
    gcv(f::SLCEFit) -> Float64

Generalized cross-validation score of a linear-estimator fit:

    GCV = n·RSS / (n − df)²

over the `n` rows of the assembled (centered / whitened) problem, with `df =`
[`effective_dof`](@ref). Returns `Inf` in the near-interpolating regime `df → n`.
Linear estimators only ([`islinear`](@ref)).

!!! warning "Torque co-fits"
    With `torque_weight > 0` the energy row and the torque-component rows of one
    configuration are correlated, but GCV treats all rows as exchangeable — the score
    is then optimistic (the same leak configuration-grouped CV folds avoid). For a
    co-fit, prefer the grouped cross-validation criterion
    ([`select_fit`](@ref)`(...; criterion = :cv)`); use this GCV as a fast reference.
"""
function gcv(f::SLCEFit)::Float64
    islinear(f.estimator) || throw(ArgumentError(
        "gcv requires a linear estimator (`islinear`); got $(typeof(f.estimator))"))
    rep = f.reparam                       # the STAGE's Z on a staged fit
    X, y, _, _, _ = _assemble_problem(f.dataset, f.torque_weight, f.force_weight, rep)
    lambda, w = _penalty_diagonal(f.estimator, _penalty_beta(f, rep))
    # under ASR the assembled design is γ-space: compress β (orthonormal Z ⇒ γ = Z'β)
    beta = rep === nothing ? f.jphi : rep.Z' * _penalty_beta(f, rep)
    return first(_gcv_score(X, y, beta, lambda, w;
                            Z = rep === nothing ? nothing : rep.Z,
                            n_eff = _gcv_neff(f)))
end

# --- λ-path driver with the cost-aware Pareto selection rule ----------------------

# Deterministic, balanced, seed-controlled fold assignment (no RNG dependency): rank
# the distinct resampling units by a seeded hash, deal ranks round-robin into `nf`
# folds, then map each row to its unit's fold. Rows sharing a unit label never split
# across the train/holdout boundary. A core port of the GLMNet extension's
# `_make_folds` (the extension cannot be referenced from here); same seed ⇒ identical
# folds within a Julia session/version (`hash` is version-dependent).
#
# `strata` (optional, one label in `0:nstrata-1` per unit in `unique(units)` order)
# stratifies the deal: classes are dealt in DESCENDING label order, each continuing
# the round-robin from where the previous stopped — so the per-fold unit counts AND
# the per-fold count of every class differ by at most one across folds. Used by
# `cross_validate` on a mixed dataset, where the label packs the derivative channels
# a config carries (`2·torque + force`, scarcest channel in the high bit): without
# it, an unstratified deal routinely produces channel-free folds when the
# channel-bearing configs are the minority, and a channel-free TRAINING split is a
# hard error under the corresponding weight. `strata = nothing` is bit-identical to
# the unstratified deal, and a `Bool` stratum with the default `nstrata = 2` is
# bit-identical to the pre-multi-class deal (`true` = 1 is dealt first) — which is
# what keeps every recorded seed reproducing its folds.
function _grouped_folds(units::AbstractVector, nf::Int, seed::Int;
                        strata::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                        nstrata::Int = 2)::Vector{Int}
    uniq = unique(units)
    order = sortperm([hash((seed, u)) for u in uniq])
    foldof = Dict{eltype(uniq),Int}()
    if strata === nothing
        @inbounds for rank in eachindex(order)
            foldof[uniq[order[rank]]] = mod1(rank, nf)
        end
    else
        length(strata) == length(uniq) ||
            throw(DimensionMismatch("strata has $(length(strata)) entries for " *
                                    "$(length(uniq)) distinct units"))
        # strata is indexed by POSITION in unique(units); require the identity
        # (units already unique, in order) so a future caller with duplicated
        # units cannot silently misalign strata to unit values.
        units == uniq ||
            throw(ArgumentError("stratified folds require `units` to be the " *
                                "distinct resampling units themselves (strata is " *
                                "positional); deduplicate before calling"))
        all(s -> 0 <= s < nstrata, strata) || throw(ArgumentError(
            "strata labels must lie in 0:$(nstrata - 1); got extremes " *
            "$(minimum(strata)) and $(maximum(strata))"))
        # DESCENDING class order, and one running counter across classes — both are
        # load-bearing. Descending keeps the Bool case (`true = 1` dealt first) exactly
        # as it was, so every existing seed reproduces its folds; the shared counter is
        # what makes the per-fold count of EACH class differ by at most one, which is
        # the whole point of stratifying. Callers encode several presence bits as one
        # integer with the scarcest channel in the high bit (`cross_validate`:
        # `2·torque + force`), so the most constrained class is dealt first.
        dealt = 0
        for pass = (nstrata - 1):-1:0
            @inbounds for rank in eachindex(order)
                u = order[rank]
                strata[u] == pass || continue
                dealt += 1
                foldof[uniq[u]] = mod1(dealt, nf)
            end
        end
    end
    return [foldof[u] for u in units]
end

# The default relative alive floor: a group counts as alive at a given λ when one of
# its columns' scaled magnitude |βⱼ|·‖X[:,j]‖ exceeds this fraction of the largest
# scaled magnitude at that λ. The adaptive-ridge alive/dead gap is many orders of
# magnitude, so the exact ratio is uncritical — any value inside the gap partitions
# identically.
const _ALIVE_RTOL = 1e-6

# The cost-aware selection rule: among the path points whose score is within
# `(1 + score_rtol)` of the finite minimum, pick the one of smallest predicted cost; ties
# go to the earlier index (the larger λ, i.e. the more-regularized fit). `Inf` scores
# (near-interpolating fits) are never eligible.
function _select_pareto(scores::Vector{Float64}, costs::Vector{Float64},
                        score_rtol::Float64)::Int
    score_rtol >= 0 || throw(ArgumentError("score_rtol must be ≥ 0; got $score_rtol"))
    length(scores) == length(costs) || throw(DimensionMismatch(
        "scores length $(length(scores)) ≠ costs length $(length(costs))"))
    emin = Inf
    for s in scores
        isfinite(s) && s < emin && (emin = s)
    end
    isfinite(emin) || throw(ArgumentError(
        "no finite score on the λ path (every fit is near-interpolating); " *
        "extend `lambdas` toward larger values"))
    best = 0
    for i in eachindex(scores)
        scores[i] <= (1 + score_rtol) * emin || continue
        if best == 0 || costs[i] < costs[best]
            best = i
        end
    end
    return best
end

"""
    LambdaPath

Result of [`select_fit`](@ref) — the regularization path, named for the knob it
sweeps so it does not claim the generic word "selection" that this whole layer
(`select_fit`, `select_support`, [`SupportPath`](@ref)) shares.

The descending λ path with the per-λ selection score
(GCV, or grouped-CV mean squared error), effective dof (`NaN` under `criterion = :cv`,
where it is not computed), alive-group count, and predicted Monte-Carlo cost
`Σ_{g alive} c_g`; plus the selection tolerance `score_rtol`, the effective absolute alive
`threshold` at the selected λ, the `selected` index, and the selected `fit` (re-solved
cold at the selected λ, so `fit(SLCEFit, dataset, estimator)` reproduces it — and its
row of the table is re-derived from that cold solve, so `fit` / `threshold` /
`n_alive[selected]` / `cost[selected]` are mutually consistent). De-bias with
`refit(path.fit; threshold = path.threshold)`, which reproduces exactly the reported
alive support. A Tables.jl source with one row per λ (columns `lambda`, `score`,
`edof`, `n_alive`, `cost`, `selected`).
"""
struct LambdaPath
    lambda::Vector{Float64}       # descending
    score::Vector{Float64}
    criterion::Symbol             # :gcv | :cv
    edof::Vector{Float64}         # NaN per entry when criterion == :cv
    n_alive::Vector{Int}
    cost::Vector{Float64}
    score_rtol::Float64
    threshold::Float64            # effective absolute alive threshold at `selected`
    selected::Int
    fit::SLCEFit
end

Tables.istable(::Type{LambdaPath}) = true
Tables.columnaccess(::Type{LambdaPath}) = true
Tables.columns(p::LambdaPath) =
    (; lambda = p.lambda, score = p.score, edof = p.edof, n_alive = p.n_alive,
       cost = p.cost, selected = [i == p.selected for i in eachindex(p.lambda)])

function Base.show(io::IO, ::MIME"text/plain", p::LambdaPath)
    print(io, "LambdaPath (criterion = :", p.criterion, ", score_rtol = ", p.score_rtol,
          "; ", length(p.lambda), " λ):")
    for i in eachindex(p.lambda)
        print(io, "\n  λ = ", round(p.lambda[i]; sigdigits = 4),
              "  score = ", round(p.score[i]; sigdigits = 5),
              "  n_alive = ", p.n_alive[i],
              "  cost = ", round(p.cost[i]; sigdigits = 5))
        i == p.selected && print(io, "   ← selected")
    end
end

"""
    select_fit(dataset::SLCEDataset, est::GroupAdaptiveRidge;
               lambdas, torque_weight = 0.0, force_weight = 0.0, criterion = :gcv,
               score_rtol = 0.05, costs = nothing, threshold = nothing, nfolds = 5,
               seed = 1, asr = true)
        -> LambdaPath

Fit `est` along the descending λ path `lambdas` (each solve warm-started from the
previous λ's coefficients), score every fit, and select the **cheapest** λ whose score
is within `(1 + score_rtol)` of the path minimum — the cost-aware generalization of the
conventional `:lambda_1se` rule. The returned [`LambdaPath`](@ref) carries the full
per-λ table and the selected fit (re-solved cold, so it is reproducible by a plain
[`fit`](@ref) call); follow with [`refit`](@ref) to de-bias the surviving groups.

`est` supplies the column groups, fixed weights, and IRLS controls; **its own `lambda`
is ignored** (the path is `lambdas`). Scoring:

- `criterion = :gcv` (default) — the [`gcv`](@ref) score from the closed-form hat
  matrix; fast, no refitting. See the co-fit caveat in the [`gcv`](@ref) docstring:
  with `torque_weight > 0` prefer `:cv`.
- `criterion = :cv` — `nfolds`-fold configuration-grouped cross-validation (folds
  never split a configuration's energy/torque rows; deterministic seeded fold
  assignment). The centering/whitening constants stay global — the score ranks λ, it
  is not an unbiased error estimate.

A group is **alive** at a given λ when any of its columns clears the
scaled-magnitude rule `|jϕⱼ|·‖X[:, j]‖ > threshold` on the assembled design (the same
support rule as [`refit`](@ref)). `GroupAdaptiveRidge` crushes dead groups to
tiny-but-**nonzero** values (the `epsilon` floor), so the default
`threshold = nothing` uses a per-λ **relative** floor, `1e-6` of that λ's largest
scaled magnitude — the alive/dead gap spans many orders of magnitude, so any ratio
inside the gap gives the same partition; an absolute number reproduces `refit`'s rule
verbatim (and `threshold = 0` counts every group alive — the cost column is then
flat). The effective absolute threshold at the selected λ is returned as
`path.threshold`; de-bias with `refit(path.fit; threshold = path.threshold)` to
realize exactly the reported support. The predicted Monte-Carlo cost of a fit is
`Σ_{g alive} c_g` with `c_g` from `costs` (default: `SLCE.group_costs` of the
dataset's basis under `est`'s column partition). `score_rtol` sets the accuracy tolerance
of the cost–error trade; sweep the `cost_exponent` of `SLCE.cost_weights` to tilt the
penalty itself and trace a Pareto front over both knobs.

`force_weight` joins `torque_weight` in the fit objective, the GCV row count and the
`:cv` fold strata, exactly as in [`cross_validate`](@ref); the selected λ is re-solved
cold with both weights.

!!! note "Joint bases and the ASR"
    The λ path solves on a cached **unconstrained** Gram and decides alive groups by a
    β-indexed magnitude rule, so it is only faithful to a fit that is itself
    unconstrained. `asr` is threaded to [`fit`](@ref) exactly as it is by
    [`cross_validate`](@ref): the default `true` refuses a displacement-decorated
    dataset (whose reparameterization the path would ignore), and `asr = false`
    selects a deliberately unconstrained joint model end to end — the selected fit is
    re-solved cold with the same `asr`, so `fit` reproduces it verbatim. Selection
    *under* the constraint is separate work.
"""
function select_fit(dataset::SLCEDataset, est::GroupAdaptiveRidge;
                    lambdas::AbstractVector{<:Real}, torque_weight::Real = 0.0,
                    criterion::Symbol = :gcv, score_rtol::Real = 0.05,
                    costs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                    threshold::Union{Nothing,Real} = nothing, nfolds::Integer = 5,
                    seed::Integer = 1, asr::Bool = true,
                    force_weight::Real = 0.0)::LambdaPath
    isempty(dataset.y_E) && throw(ArgumentError("dataset has no observations"))
    # The λ path below solves UNCONSTRAINED (direct `_solve_gar` on the cached Gram)
    # and its alive rule is β-indexed, so it is correct exactly when the fit it
    # reproduces is also unconstrained. `_resolve_asr_rep` answers that with `fit`'s
    # own rules (including the refusal when `asr = true` meets a displacement basis
    # that carries no reparameterization), so `asr = false` selects a deliberately
    # unconstrained joint model here just as it fits one there.
    rep = _resolve_asr_rep(dataset, asr)
    rep === nothing ||
        throw(ArgumentError("select_fit under an ASR reparameterization is not " *
                            "implemented yet: the λ path solves on a cached " *
                            "unconstrained Gram and its alive rule is β-indexed, " *
                            "so running it here would report a support the " *
                            "constrained solve never chose. Pass asr = false to " *
                            "select a deliberately unconstrained model, or use " *
                            "fit/cross_validate, which are constrained"))
    w = Float64(torque_weight)
    wF = Float64(force_weight)
    (0.0 <= w <= 1.0) || throw(ArgumentError("torque_weight must be in [0, 1]; got $w"))
    (0.0 <= wF <= 1.0) ||
        throw(ArgumentError("force_weight must be in [0, 1]; got $wF"))
    (w + wF <= 1.0) || throw(ArgumentError(
        "torque_weight + force_weight must be ≤ 1; got $(w + wF)"))
    if w > 0 && !has_torque(dataset)
        throw(ArgumentError("torque_weight = $w but the dataset has no torque data"))
    end
    if wF > 0 && !has_force(dataset)
        throw(ArgumentError("force_weight = $wF but the dataset has no force data"))
    end
    isempty(lambdas) && throw(ArgumentError("lambdas must be nonempty"))
    all(l -> isfinite(l) && l >= 0, lambdas) ||
        throw(ArgumentError("lambdas must be finite and ≥ 0"))
    criterion in (:gcv, :cv) ||
        throw(ArgumentError("criterion must be :gcv or :cv; got :$criterion"))
    score_rtol >= 0 || throw(ArgumentError("score_rtol must be ≥ 0; got $score_rtol"))
    threshold === nothing || threshold >= 0 ||
        throw(ArgumentError("threshold must be ≥ 0 (or nothing for the relative " *
                            "default); got $threshold"))
    nfolds >= 2 || throw(ArgumentError("nfolds must be ≥ 2; got $nfolds"))
    G = length(est.group_weights)
    length(est.column_groups) == n_salcs(dataset.basis) || throw(DimensionMismatch(
        "estimator column_groups length $(length(est.column_groups)) ≠ basis column " *
        "count $(n_salcs(dataset.basis))"))
    cg = costs === nothing ?
         Float64.(group_costs(dataset.basis, est.column_groups)) : Float64.(costs)
    length(cg) == G ||
        throw(ArgumentError("costs length $(length(cg)) ≠ number of groups $G"))

    lams = sort!(unique(Float64.(lambdas)); rev = true)
    nl = length(lams)
    X, y, _, _, rowgroups = _assemble_problem(dataset, w, wF)
    n = size(X, 1)
    XtX = Matrix{Float64}(X' * X)
    Xty = Vector{Float64}(X' * y)
    colnorms = [norm(view(X, :, j)) for j = 1:size(X, 2)]

    # Warm-started descending path: each IRLS is seeded with the previous (more
    # regularized, already group-sparse) λ's solution.
    betas = Vector{Vector{Float64}}(undef, nl)
    prev = nothing
    for i = 1:nl
        b = lams[i] == 0.0 ? (X \ y) :
            _solve_gar(XtX, Xty, lams[i], est.column_groups, est.group_weights,
                       est.group_sizes, est.epsilon, est.max_iter, est.tol;
                       beta0 = prev)
        betas[i] = b
        prev = b
    end

    # Alive groups (the refit support rule, per group) and the predicted MC cost. The
    # adaptive-ridge iteration never produces exact zeros (the ε floor), so the
    # `threshold === nothing` default applies a per-λ *relative* floor on the scaled
    # magnitudes — the alive/dead gap spans many orders, so any ratio inside it gives
    # the same partition. Returns (alive count, predicted cost, absolute threshold).
    thr_abs = threshold === nothing ? nothing : Float64(threshold)
    function alive_stats!(alive::BitVector, b::Vector{Float64})
        smax = 0.0
        for j in eachindex(b)
            m = abs(b[j]) * colnorms[j]
            m > smax && (smax = m)
        end
        t = thr_abs === nothing ? _ALIVE_RTOL * smax : thr_abs
        fill!(alive, false)
        for j in eachindex(b)
            abs(b[j]) * colnorms[j] > t && (alive[est.column_groups[j]] = true)
        end
        c = sum(cg[g] for g = 1:G if alive[g]; init = 0.0)
        return count(alive), c, t
    end
    n_alive = Vector{Int}(undef, nl)
    cost = Vector{Float64}(undef, nl)
    alive = falses(G)
    for i = 1:nl
        n_alive[i], cost[i], _ = alive_stats!(alive, betas[i])
    end

    edof = fill(NaN, nl)
    score = Vector{Float64}(undef, nl)
    # w = 1 leaves the energy rows in the stack with exactly zero weight — they
    # carry no information and must not inflate GCV's n (same zero test as the
    # assembly's scale expression).
    # Zero-weight blocks carry no information and must not inflate GCV's `n`; the
    # shared definition is `_gcv_neff`, and the energy block's weight is
    # `1 - w - wF`, not `1 - w`.
    neff = n - (max(0.0, 1 - w - wF) == 0.0 ? length(dataset.y_E) : 0)
    if criterion === :gcv
        wv = Vector{Float64}(undef, length(Xty))
        normsq = Vector{Float64}(undef, G)
        for i = 1:nl
            if lams[i] == 0.0
                score[i], edof[i] = _gcv_score(X, y, betas[i], 0.0, nothing;
                                               n_eff = neff)
            else
                _gar_weights!(wv, betas[i], est.column_groups, est.group_weights,
                              est.group_sizes, est.epsilon, normsq)
                score[i], edof[i] = _gcv_score(X, y, betas[i], lams[i], wv;
                                               XtX = XtX, n_eff = neff)
            end
        end
    else
        units = rowgroups === nothing ? collect(1:n) : rowgroups
        nunits = length(unique(units))
        nf = min(Int(nfolds), div(nunits, 3))
        nf >= 2 || throw(ArgumentError(
            "cross-validation needs at least 6 resampling units for ≥ 2 folds; got " *
            "$nunits. Use criterion = :gcv or pass more data."))
        nf < Int(nfolds) &&
            @warn "select_fit: reducing CV folds so every fold keeps ≥ 3 resampling " *
                  "units" requested = Int(nfolds) effective = nf units = nunits
        # Stratify by which derivative channels each config carries on a mixed dataset
        # (same rationale, label packing and descending deal order as cross_validate):
        # the λ path must not be ranked on folds with wildly uneven channel content.
        # Units here are ROW labels; the fold assignment is per distinct unit, so build
        # the per-unit strata and deal over the distinct units.
        nc_u = length(dataset)
        ragged_tq = w > 0 && has_torque(dataset) &&
            length(dataset.torque_config) < 3 * n_atoms(dataset.basis.crystal) * nc_u
        ragged_fr = wF > 0 && has_force(dataset) &&
            length(unique(dataset.force_config)) < nc_u
        strata = nothing
        if ragged_tq || ragged_fr
            tqp = falses(nc_u)
            ragged_tq && (tqp[unique(dataset.torque_config)] .= true)
            frp = falses(nc_u)
            ragged_fr && (frp[unique(dataset.force_config)] .= true)
            strata = [2 * tqp[i] + frp[i] for i = 1:nc_u]
        end
        ufolds = _grouped_folds(collect(1:nc_u), nf, seed;
                                strata = strata, nstrata = 4)
        folds = [ufolds[u] for u in units]
        sse = zeros(Float64, nl)
        for k = 1:nf
            ho = findall(==(k), folds)
            Xho = X[ho, :]
            yho = y[ho]
            # training Gram by downdating the cached full Gram — no per-fold X pass
            XtX_tr = XtX .- Xho' * Xho
            Xty_tr = Xty .- Xho' * yho
            prevf = nothing
            for i = 1:nl
                bf = if lams[i] == 0.0
                    tr_rows = findall(!=(k), folds)
                    X[tr_rows, :] \ y[tr_rows]
                else
                    _solve_gar(XtX_tr, Xty_tr, lams[i], est.column_groups,
                               est.group_weights, est.group_sizes, est.epsilon,
                               est.max_iter, est.tol; beta0 = prevf)
                end
                prevf = bf
                sse[i] += sum(abs2, yho .- Xho * bf)
            end
        end
        score .= sse ./ n
    end

    sel = _select_pareto(score, cost, Float64(score_rtol))
    est_sel = GroupAdaptiveRidge(lams[sel], est.column_groups, est.group_weights,
                                 est.epsilon, est.max_iter, est.tol)
    fsel = fit(SLCEFit, dataset, est_sel; torque_weight = w, force_weight = wF,
               asr = asr)
    # Re-derive the selected row from the cold re-solve, so `fit` / `threshold` /
    # `n_alive[selected]` / `cost[selected]` are mutually consistent (warm and cold
    # agree only within the IRLS tol — a knife-edge coefficient could differ). Under
    # :gcv the score/edof are functions of the returned fit, so re-derive them too
    # (`path.score[selected] == gcv(path.fit)` exactly); the :cv score aggregates
    # fold models and does not depend on the full-data solve — leave it.
    n_alive[sel], cost[sel], t_sel = alive_stats!(alive, fsel.jphi)
    if criterion === :gcv
        if lams[sel] == 0.0
            score[sel], edof[sel] = _gcv_score(X, y, fsel.jphi, 0.0, nothing;
                                               n_eff = neff)
        else
            wv = Vector{Float64}(undef, length(Xty))
            normsq = Vector{Float64}(undef, G)
            _gar_weights!(wv, fsel.jphi, est.column_groups, est.group_weights,
                          est.group_sizes, est.epsilon, normsq)
            score[sel], edof[sel] = _gcv_score(X, y, fsel.jphi, lams[sel], wv;
                                               XtX = XtX, n_eff = neff)
        end
    end
    return LambdaPath(lams, score, criterion, edof, n_alive, cost, Float64(score_rtol),
                      t_sel, sel, fsel)
end

# --- threshold sweep: the (cost, error) front of de-biased refits -----------------

# The auto threshold grid for `select_support`: at most `n` log-rank-spaced points on
# the sorted per-group scaled magnitudes, each threshold the midpoint between
# consecutive *distinct* magnitudes (a midpoint at an exact tie would equal the tied
# value and, under the strict `>` rule, exclude the whole tie — tied groups die
# together instead), plus `0.0` for the full-support anchor. Returned descending
# (sparsest refit first); duplicates and degenerate ranks collapse, so fewer than `n`
# points can come back (always ≥ 1: the anchor).
function _support_thresholds(n::Integer, m_g::Vector{Float64})::Vector{Float64}
    n >= 2 || throw(ArgumentError("npoints must be ≥ 2; got $n"))
    ms = sort(m_g; rev = true)
    G = length(ms)
    thr = Float64[]
    for r in unique(round.(Int, exp.(range(log(1), log(G); length = n))))
        if r == G
            push!(thr, 0.0)
        elseif ms[r] > ms[r+1]
            push!(thr, (ms[r] + ms[r+1]) / 2)
        end
    end
    return sort!(unique(thr); rev = true)
end

"""
    SupportPath

Result of [`select_support`](@ref): the descending threshold sweep with, per point,
the alive-group count, predicted Monte-Carlo cost `Σ_{g alive} c_g`, the selection
`score` (the fit's own `(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F` objective on the
evaluation dataset), and the energy / torque / force RMSEs (`rmse_torque` and
`rmse_force` are `NaN` when the evaluation dataset carries no data for that channel;
the three are in **different units** — eV, eV, eV/Å); plus the tolerance `score_rtol`,
the `selected` index, and the de-biased refit `fit` at the selected threshold. A
Tables.jl source with one row per threshold (columns `threshold`, `n_alive`, `cost`,
`score`, `rmse_energy`, `rmse_torque`, `rmse_force`, `selected`).
"""
struct SupportPath
    threshold::Vector{Float64}    # descending (sparsest first)
    n_alive::Vector{Int}
    cost::Vector{Float64}
    score::Vector{Float64}
    rmse_energy::Vector{Float64}
    rmse_torque::Vector{Float64}  # NaN per entry when the evalset has no torque
    rmse_force::Vector{Float64}   # NaN per entry when the evalset has no forces
    score_rtol::Float64
    selected::Int
    fit::SLCEFit
end

Tables.istable(::Type{SupportPath}) = true
Tables.columnaccess(::Type{SupportPath}) = true
Tables.columns(p::SupportPath) =
    (; threshold = p.threshold, n_alive = p.n_alive, cost = p.cost, score = p.score,
       rmse_energy = p.rmse_energy, rmse_torque = p.rmse_torque,
       rmse_force = p.rmse_force,
       selected = [i == p.selected for i in eachindex(p.threshold)])

function Base.show(io::IO, ::MIME"text/plain", p::SupportPath)
    print(io, "SupportPath (score_rtol = ", p.score_rtol, "; ", length(p.threshold),
          " thresholds):")
    for i in eachindex(p.threshold)
        print(io, "\n  thr = ", round(p.threshold[i]; sigdigits = 4),
              "  score = ", round(p.score[i]; sigdigits = 5),
              "  n_alive = ", p.n_alive[i],
              "  cost = ", round(p.cost[i]; sigdigits = 5))
        i == p.selected && print(io, "   ← selected")
    end
end

"""
    select_support(f::SLCEFit; npoints = 25, thresholds = nothing, score_rtol = 0.05,
                   column_groups = nothing, costs = nothing, evalset = f.dataset,
                   estimator = OLS())
        -> SupportPath

The weights come from `f` (`torque_weight` / `force_weight`); there is nothing to pass
here.

Trace the (predicted Monte-Carlo cost, error) front of **de-biased refits** of `f`
over a sweep of alive thresholds, and select the cheapest point whose score is within
`(1 + score_rtol)` of the front minimum — the same Pareto rule as [`select_fit`](@ref).
This is the second knob of the cost-aware workflow: [`select_fit`](@ref) sweeps the
penalty λ, while this sweeps the support directly. On real data the group-magnitude
spectrum is typically continuous (no clean alive/dead gap), so most of the
cost–error trade lives here.

Per threshold `t` the point's fit is `refit(f, estimator; threshold = t)`, keeping the
*columns* with `|jϕⱼ|·‖X[:, j]‖ > t` on `f`'s assembled design (the [`refit`](@ref)
rule). The reported `n_alive` and `cost` are then read back off that refit — the alive
groups are those holding a nonzero de-biased coefficient, and the predicted cost is
`Σ_{g alive} c_g`. Reading them off the *pre*-threshold magnitudes instead would agree
without a constraint but over-report under one (below). Note that a weak column of an
alive group may still be dropped; the group's cost is paid either way. The `score` is
the fit's own objective
`(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F` (the weights taken from `f`) evaluated
on `evalset` — pass a held-out `SLCEDataset` (built on the same basis; see dataset
slicing) for an honest error axis; the default is the in-sample training set.

!!! warning "Derivative-only fits price pure-spin groups at zero"
    At `torque_weight + force_weight == 1` the energy block gets weight zero, so every
    column absent from the derivative blocks has norm zero in the assembled design
    (`fit` warns about this by name). Such groups are then never alive at any
    threshold and contribute no cost — correct, since a derivative-only fit genuinely
    cannot see them, but it means the front is a front over the *derivative-visible*
    model only. The relative alive floor also spans blocks of different units there;
    pass an absolute `thresholds` vector if that matters.

!!! warning "Under an ASR, the displacement groups move as one"
    A joint fit carries the acoustic sum rule, and the constraint's rows tie columns
    **across** the [`salc_groups`](@ref) partition. Measured on every fixture tried
    (G from 2 to 20): not one displacement-touched group has a feasible subspace on
    its own (`dim null(A[:, cols_g]) == 0`), and closing the groups under the
    constraint's connected components collapses them to **two** clusters regardless of
    `G`. So the cost axis is real on the pure-spin groups — which `build_asr` leaves
    untouched, identity blocks in `Z` — and close to binary on the displacement side.
    The front is still honest, because `cost` is derived from the returned refit and
    [`group_costs`](@ref) is given the fit's constraint; it simply has fewer reachable
    values there than `G` suggests. [`group_freedom`](@ref) reports what the constraint
    leaves each group.

The sweep is either automatic or explicit, and the two are **separate keywords** on
purpose: `npoints` is a point count for the automatic grid (**at most** that many
points, log-rank-spaced on the per-group magnitude spectrum plus the full-support
anchor — duplicate ranks and exact magnitude ties collapse), while `thresholds` is an
explicit vector of absolute thresholds and overrides it. One keyword carrying both
meanings turned `thresholds = 10` — "sweep down to a magnitude of 10" — into a
silent ten-point grid, distinguishable from `thresholds = [10.0]` only by the literal's
type. `column_groups`/`costs` default to `SLCE.salc_groups` / `SLCE.group_costs` of
the training basis.
"""
function select_support(f::SLCEFit;
                        npoints::Integer = 25,
                        thresholds::Union{Nothing,AbstractVector{<:Real}} = nothing,
                        score_rtol::Real = 0.05,
                        column_groups::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                        costs::Union{Nothing,AbstractVector{<:Real}} = nothing,
                        evalset::SLCEDataset = f.dataset,
                        estimator::AbstractEstimator = OLS())::SupportPath
    score_rtol >= 0 || throw(ArgumentError("score_rtol must be ≥ 0; got $score_rtol"))
    _reject_fixed_coefficients(estimator)
    isempty(evalset.y_E) && throw(ArgumentError("evalset has no observations"))
    basis = f.dataset.basis
    lab = column_groups === nothing ? salc_groups(basis) : Vector{Int}(column_groups)
    G = _validate_labels(lab, length(f.jphi), "select_support")
    # The default cost discounts groups the fit's own constraint makes infeasible: on a
    # plain ASR fit `f.reparam === f.dataset.asr` (staged fits are refused below), so
    # this is the constraint the reported front actually lives under.
    cg = costs === nothing ?
         Float64.(group_costs(basis, lab; asr = f.reparam)) : Float64.(costs)
    length(cg) == G ||
        throw(ArgumentError("costs length $(length(cg)) ≠ number of groups $G"))
    evalset.basis.salc_basis.fingerprint == basis.salc_basis.fingerprint ||
        throw(ArgumentError("evalset was built on a different SLCEBasis than the fit"))
    w = f.torque_weight
    if w > 0 && !has_torque(evalset)
        throw(ArgumentError("f is a torque co-fit (torque_weight = $w) but evalset " *
                            "has no torque data"))
    end
    wF = f.force_weight
    if wF > 0 && !has_force(evalset)
        throw(ArgumentError("f is a force co-fit (force_weight = $wF) but evalset " *
                            "has no force data"))
    end
    # Three states, two of them refused, and the order matters: a plain joint fit
    # carries `dataset.asr` as its reparameterization, so keying the staged check on
    # `reparam !== nothing` told unstaged callers their fit was staged. `_is_staged`
    # (fit.jl) is the one definition of the distinction.
    if _is_staged(f)
        # A staged fit's frozen columns are not candidates for thresholding (they were
        # never fitted), so the per-group alive/cost accounting below would report a
        # support the stage cannot choose.
        throw(ArgumentError("select_support on a staged fit (frozen / sector_mask) " *
                            "is not implemented yet — its frozen columns are not " *
                            "selectable, so the group cost front would be wrong; " *
                            "use refit, which stays inside the stage"))
    end

    # Per-group max scaled magnitude on the assembled training design — and it must be
    # `refit`'s design, weights included. `_assemble_problem`'s energy whitening is
    # `√((1 − w_T − w_F)/n_E)`, so dropping `wF` here would put `m_g` (and every
    # threshold derived from it) in different units from the `threshold` handed to
    # `refit` below: the reported alive/cost columns and the realized support would
    # silently disagree.
    X, _, _, _, _ = _assemble_problem(f.dataset, w, wF)
    m_g = zeros(Float64, G)
    for j in eachindex(f.jphi)
        m = abs(f.jphi[j]) * norm(view(X, :, j))
        g = lab[j]
        m > m_g[g] && (m_g[g] = m)
    end
    thr = if thresholds === nothing
        _support_thresholds(npoints, m_g)
    else
        isempty(thresholds) && throw(ArgumentError("thresholds must be nonempty"))
        all(t -> isfinite(t) && t >= 0, thresholds) ||
            throw(ArgumentError("thresholds must be finite and ≥ 0"))
        sort!(unique(Float64.(thresholds)); rev = true)
    end

    nt = length(thr)
    n_alive = Vector{Int}(undef, nt)
    cost = Vector{Float64}(undef, nt)
    score = Vector{Float64}(undef, nt)
    rmseE = Vector{Float64}(undef, nt)
    rmseT = fill(NaN, nt)
    rmseF = fill(NaN, nt)
    fits = Vector{SLCEFit}(undef, nt)
    for i = 1:nt
        fr = refit(f, estimator; threshold = thr[i])
        fits[i] = fr
        # Alive groups are read off the REFIT, not off the pre-threshold magnitudes
        # `m_g > thr[i]`. The two agree without a constraint, and this is already the
        # convention the front is pinned to; under an ASR they do not, because `refit`
        # re-derives the null space on the support and a support that splits a
        # constraint-coupled column set structurally zeroes some survivors. Deriving
        # the row from `fr.jphi` is what makes the reported cost the cost of the model
        # actually returned, rather than an upper bound on it.
        alive = falses(G)
        for j in eachindex(fr.jphi)
            fr.jphi[j] != 0.0 && (alive[lab[j]] = true)
        end
        n_alive[i] = count(alive)
        cost[i] = sum(cg[g] for g = 1:G if alive[g]; init = 0.0)
        mseE = mean(abs2, evalset.y_E .- (fr.j0 .+ evalset.X_E * fr.jphi))
        rmseE[i] = sqrt(mseE)
        # The three-block objective, in the same prediction-space convention
        # `cross_validate` uses. A channel absent from the evalset is omitted rather
        # than counted as zero; with both derivative weights zero the objective IS the
        # energy MSE, so the historical `score = mseE` is kept verbatim.
        sc = (1 - w - wF) * mseE
        anyderiv = false
        if has_torque(evalset)
            mseT = mean(abs2, evalset.y_T .- evalset.X_T * fr.jphi)
            rmseT[i] = sqrt(mseT)
            if w > 0
                sc += w * mseT
                anyderiv = true
            end
        end
        if has_force(evalset)
            mseF = mean(abs2, evalset.y_F .- evalset.X_F * fr.jphi[evalset.force_cols])
            rmseF[i] = sqrt(mseF)
            if wF > 0
                sc += wF * mseF
                anyderiv = true
            end
        end
        score[i] = anyderiv ? sc : mseE
    end
    sel = _select_pareto(score, cost, Float64(score_rtol))
    return SupportPath(thr, n_alive, cost, score, rmseE, rmseT, rmseF,
                       Float64(score_rtol), sel, fits[sel])
end

# --- configuration-grouped K-fold cross-validation (generic) -----------------------

"""
    CVResult

Result of [`cross_validate`](@ref): per-fold holdout scores plus the pooled
out-of-fold error. `score` is the fit's own objective
`(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F` on each fold's held-out
configurations; `rmse_energy`, `rmse_torque` and `rmse_force` report the three error
axes separately whenever the dataset carries the data, independent of the weights
(each is `NaN` per entry when that channel is absent from the dataset or from that
fold's holdout). They are in **different units** (eV, eV, eV/Å) — compare each against
itself across folds, never against another. The `pooled_*` fields aggregate the
out-of-fold residuals of all folds — every configuration is held out exactly once, so
they are single whole-dataset numbers, not means of the per-fold columns. A Tables.jl
source with one row per fold (columns `fold`, `n_holdout`, `score`, `rmse_energy`,
`rmse_torque`, `rmse_force`).
"""
struct CVResult
    nfolds::Int                   # effective fold count (see cross_validate)
    seed::Int
    torque_weight::Float64
    force_weight::Float64
    n_holdout::Vector{Int}        # held-out configurations per fold
    score::Vector{Float64}
    rmse_energy::Vector{Float64}
    rmse_torque::Vector{Float64}  # NaN: energy-only dataset, or torque-free fold
    rmse_force::Vector{Float64}   # NaN: force-free dataset, or force-free fold
    pooled_score::Float64
    pooled_rmse_energy::Float64
    pooled_rmse_torque::Float64   # NaN for a torque-free dataset
    pooled_rmse_force::Float64    # NaN for a force-free dataset
end

Tables.istable(::Type{CVResult}) = true
Tables.columnaccess(::Type{CVResult}) = true
Tables.columns(r::CVResult) =
    (; fold = collect(eachindex(r.score)), n_holdout = r.n_holdout, score = r.score,
       rmse_energy = r.rmse_energy, rmse_torque = r.rmse_torque,
       rmse_force = r.rmse_force)

function Base.show(io::IO, ::MIME"text/plain", r::CVResult)
    print(io, "CVResult (", r.nfolds, " folds, seed = ", r.seed,
          ", torque_weight = ", r.torque_weight,
          ", force_weight = ", r.force_weight, "):")
    for k in eachindex(r.score)
        print(io, "\n  fold ", k, ": n = ", r.n_holdout[k],
              "  score = ", round(r.score[k]; sigdigits = 5),
              "  rmse_E = ", round(r.rmse_energy[k]; sigdigits = 5))
        isnan(r.rmse_torque[k]) ||
            print(io, "  rmse_T = ", round(r.rmse_torque[k]; sigdigits = 5))
        isnan(r.rmse_force[k]) ||
            print(io, "  rmse_F = ", round(r.rmse_force[k]; sigdigits = 5))
    end
    print(io, "\n  pooled: score = ", round(r.pooled_score; sigdigits = 5),
          "  rmse_E = ", round(r.pooled_rmse_energy; sigdigits = 5))
    isnan(r.pooled_rmse_torque) ||
        print(io, "  rmse_T = ", round(r.pooled_rmse_torque; sigdigits = 5))
    isnan(r.pooled_rmse_force) ||
        print(io, "  rmse_F = ", round(r.pooled_rmse_force; sigdigits = 5))
end

"""
    cross_validate(dataset::SLCEDataset, estimator::AbstractEstimator;
                   torque_weight = 0.0, force_weight = 0.0, nfolds = 5, seed = 1,
                   asr = true, frozen = nothing, sector_mask = :all) -> CVResult

Configuration-grouped `nfolds`-fold cross-validation of
`fit(SLCEFit, dataset, estimator; torque_weight)`: each fold's model is fit from
scratch on the training configurations only (energy centering and torque whitening
included — nothing leaks across the split), then scored on the held-out
configurations in prediction space. A configuration is one resampling unit, so its
energy row and all of its torque rows (none, for a torque-free configuration of a
mixed dataset) always land on the same side of the split. Fold assignment is
deterministic in `seed` (seeded-hash round-robin, the same rule as
[`select_fit`](@ref)'s `:cv`) and, on a mixed dataset, **stratified by torque
presence** so the torque-bearing minority spreads evenly across folds — a
torque-free holdout fold scores energy-only (its `rmse_torque` entry is `NaN`),
and `torque_weight > 0` additionally requires ≥ 2 torque-bearing configurations
and caps the fold count by their number so every training split keeps torque
rows. (Fold assignments on a mixed dataset therefore differ from an unstratified
deal at the same `seed`; uniform-torque and energy-only datasets are unaffected.)
When the dataset has fewer than `3·nfolds` configurations the fold count is
reduced (with a warning) so every fold keeps ≥ 3 configurations, and fewer than
6 configurations is an error.

Use it to compare estimators, `torque_weight` settings, or a [`refit`](@ref)-style
support on an equal footing: `rmse_energy` and `rmse_torque` report both error axes
regardless of `torque_weight` (a `torque_weight = 0` fit still gets its torque
error measured when the dataset carries torque data). This differs from
`select_fit(...; criterion = :cv)`, which ranks a λ path under **global**
centering/whitening for speed — `cross_validate` is the honest
generalization-error estimate.

`asr`, `frozen` and `sector_mask` are threaded to every fold's [`fit`](@ref), so a
staged fitting plan is cross-validated exactly as it will be run. The frozen model
is held fixed across folds by construction — it is an input, not something the
folds re-estimate — so a plan whose earlier stage was fitted on *these* same
configurations reports an optimistic score; cross-validate the whole chain (or fit
the frozen stage on separate data) if that matters.

A `FixedCoefficients` (or an `AdaptiveLasso` carrying one) is rejected: its fixed,
full-data coefficient vector does not depend on the training fold, so the holdout
score would leak the held-out data.

`force_weight` behaves exactly like `torque_weight`: the score becomes the fit's own
three-block objective `(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F`, the fold deal is
stratified on both derivative channels at once, and `rmse_force` is reported whenever
the dataset carries forces. The three RMSE columns are in **different units**
(eV, eV, eV/Å), so compare each against itself across folds, not against each other.
"""
function cross_validate(dataset::SLCEDataset, estimator::AbstractEstimator;
                        torque_weight::Real = 0.0, force_weight::Real = 0.0,
                        nfolds::Integer = 5,
                        seed::Integer = 1, asr::Bool = true,
                        frozen::Union{Nothing,SLCEModel} = nothing,
                        sector_mask = :all)::CVResult
    w = Float64(torque_weight)
    wF = Float64(force_weight)
    (0.0 <= w <= 1.0) || throw(ArgumentError("torque_weight must be in [0, 1]; got $w"))
    (0.0 <= wF <= 1.0) ||
        throw(ArgumentError("force_weight must be in [0, 1]; got $wF"))
    (w + wF <= 1.0) || throw(ArgumentError(
        "torque_weight + force_weight must be ≤ 1; got $(w + wF)"))
    if w > 0 && !has_torque(dataset)
        throw(ArgumentError("torque_weight = $w but the dataset has no torque data"))
    end
    if wF > 0 && !has_force(dataset)
        throw(ArgumentError("force_weight = $wF but the dataset has no force data"))
    end
    nfolds >= 2 || throw(ArgumentError("nfolds must be ≥ 2; got $nfolds"))
    if _carries_fixed_coefficients(estimator)
        throw(ArgumentError("cross_validate does not accept a FixedCoefficients (or " *
            "an AdaptiveLasso carrying one): its fixed full-data coefficient vector " *
            "does not depend on the training fold, so the holdout score would leak. " *
            "Pass the estimator that produced the pilot instead."))
    end
    nc = length(dataset)
    nf = min(Int(nfolds), div(nc, 3))
    hastq = has_torque(dataset)
    hasfr = has_force(dataset)
    # Stratify the fold deal by which DERIVATIVE CHANNELS each config carries (a mixed
    # dataset has torque rows for some configs and force rows for others, and the two
    # selections are independent). Without stratification, an unstratified deal
    # routinely produces channel-free folds when the channel-bearing configs are the
    # minority, and a channel-free TRAINING split is a hard error under that channel's
    # weight. The label packs both bits, torque high — `_grouped_folds` deals classes
    # in descending order, so on a force-free dataset it degenerates to the
    # torque-only `(true, false)` deal bit-identically and every recorded seed keeps
    # its folds.
    tqpresent = falses(nc)
    hastq && (tqpresent[unique(dataset.torque_config)] .= true)
    # Force presence enters the strata ONLY under `wF > 0`. Torque's unconditional
    # stratification is grandfathered; adding a second unconditional one would change
    # the fold deal — and therefore the score — of every force-carrying dataset at
    # `force_weight = 0`, invalidating recorded results to buy nothing but a prettier
    # `rmse_force` spread. At `wF = 0` the deal is bit-identical to the torque-only
    # one, which is the "cross_validate ignores the force block" contract.
    frpresent = falses(nc)
    wF > 0 && hasfr && (frpresent[unique(dataset.force_config)] .= true)
    ntq = count(tqpresent)
    nfr = count(frpresent)
    if w > 0
        # Every fold must hold ≥ 1 torque-bearing config (so every training split
        # keeps some): cap the fold count by the number of torque-bearing configs.
        ntq >= 2 || throw(ArgumentError(
            "torque_weight = $w > 0 needs ≥ 2 torque-bearing configurations for " *
            "cross-validation (every training split must keep torque rows); got $ntq"))
        nf = min(nf, ntq)
    end
    if wF > 0
        nfr >= 2 || throw(ArgumentError(
            "force_weight = $wF > 0 needs ≥ 2 force-bearing configurations for " *
            "cross-validation (every training split must keep force rows); got $nfr"))
        nf = min(nf, nfr)
    end
    nf >= 2 || throw(ArgumentError(
        "cross-validation needs at least 6 configurations for ≥ 2 folds; got $nc"))
    nf < Int(nfolds) &&
        @warn "cross_validate: reducing CV folds so every fold keeps ≥ 3 " *
              "configurations and (for torque_weight / force_weight > 0) ≥ 1 " *
              "configuration bearing that channel" requested = Int(nfolds) effective = nf configs = nc

    strata = hastq || (wF > 0 && hasfr) ?
             [2 * tqpresent[i] + frpresent[i] for i = 1:nc] : nothing
    folds = _grouped_folds(collect(1:nc), nf, seed; strata = strata, nstrata = 4)
    n_holdout = Vector{Int}(undef, nf)
    score = Vector{Float64}(undef, nf)
    rmseE = Vector{Float64}(undef, nf)
    rmseT = fill(NaN, nf)
    rmseF = fill(NaN, nf)
    sseE = 0.0
    sseT = 0.0
    sseF = 0.0
    nE = 0
    nT = 0
    nF = 0
    for k = 1:nf
        ho = findall(==(k), folds)
        tr = findall(!=(k), folds)
        f = fit(SLCEFit, dataset[tr], estimator; torque_weight = w,
                force_weight = wF, asr = asr,
                frozen = frozen, sector_mask = sector_mask)
        hset = dataset[ho]
        residE = hset.y_E .- (f.j0 .+ hset.X_E * f.jphi)
        mseE = mean(abs2, residE)
        n_holdout[k] = length(ho)
        rmseE[k] = sqrt(mseE)
        sseE += sum(abs2, residE)
        nE += length(residE)
        # A holdout fold of a mixed dataset can hold no rows of a given derivative
        # channel (only when that channel's weight is 0 — stratification plus the fold
        # cap guarantee rows in every fold under a positive weight). Its RMSE entry
        # stays NaN and the score simply omits the block, never `0 * NaN`; the
        # remaining weights are NOT renormalized, so a fold that lost a block scores
        # lower than one that kept it, which is why the cap exists.
        sc = mseE * (1 - w - wF)
        anyderiv = false
        if hastq && !isempty(hset.y_T)
            residT = hset.y_T .- hset.X_T * f.jphi
            mseT = mean(abs2, residT)
            rmseT[k] = sqrt(mseT)
            sseT += sum(abs2, residT)
            nT += length(residT)
            if w > 0
                sc += w * mseT
                anyderiv = true
            end
        end
        if hasfr && !isempty(hset.y_F)
            residF = hset.y_F .- hset.X_F * f.jphi[hset.force_cols]
            mseF = mean(abs2, residF)
            rmseF[k] = sqrt(mseF)
            sseF += sum(abs2, residF)
            nF += length(residF)
            if wF > 0
                sc += wF * mseF
                anyderiv = true
            end
        end
        # With both derivative weights zero the objective IS the energy MSE — keep the
        # historical `score = mseE` rather than the vacuously rescaled `1 · mseE`.
        score[k] = anyderiv ? sc : mseE
    end
    pE = sseE / nE
    pT = nT > 0 ? sseT / nT : NaN
    pF = nF > 0 ? sseF / nF : NaN
    pooled = if w > 0 || wF > 0
        (1 - w - wF) * pE + (w > 0 ? w * pT : 0.0) + (wF > 0 ? wF * pF : 0.0)
    else
        pE
    end
    return CVResult(nf, Int(seed), w, wF, n_holdout, score, rmseE, rmseT, rmseF,
                    pooled, sqrt(pE), sqrt(pT), sqrt(pF))
end
