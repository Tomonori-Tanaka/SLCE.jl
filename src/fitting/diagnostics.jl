# Fit diagnostics: coefficients, intercept, counts, residuals, and the energy /
# torque goodness-of-fit (R^2, RMSE, RSS). coef/nobs/dof extend StatsAPI.

"""
    coef(fit_or_model) -> Vector{Float64}

The fitted SALC coefficients `Jϕ`, one per design-matrix column (in [`SALCKey`](@ref)
order). The reference energy `j0` is separate; read it with [`intercept`](@ref).
"""
coef(f::SLCEFit) = f.jphi
coef(m::SLCEModel) = m.jphi

"""
    intercept(fit_or_model) -> Float64

The reference energy `j0` (the SCE intercept), recovered analytically in [`fit`](@ref).
"""
intercept(f::SLCEFit) = f.j0
intercept(m::SLCEModel) = m.j0

"""
    nobs(f::SLCEFit) -> Int

The number of energy observations used in the fit.
"""
nobs(f::SLCEFit) = length(f.dataset.y_E)

"""
    dof(f::SLCEFit) -> Int

Degrees of freedom consumed by the fit: the number of FREE parameters plus the
intercept `j0`. Unconstrained, that is `length(coef(f)) + 1`; under the ASR
constraint (`fit(...; asr = true)` on a joint basis) the `rank(A)` exactly-enforced
equalities are not free, so `dof = p − rank(A) + 1`; a staged fit
(`frozen` / `sector_mask`) counts only the columns that stage fits, minus the
constraints binding them. All three are one expression — the column count of the
reparameterization the fit actually solved under. This is the parametric count,
not an effective / selected count, even for a sparse estimator.
"""
dof(f::SLCEFit)::Int =
    (f.reparam === nothing ? length(f.jphi) : size((f.reparam::ASRReparam).Z, 2)) + 1

"""
    residuals_energy(f::SLCEFit) -> Vector{Float64}

The per-configuration energy residuals `y_E − (j0 + X_E·jϕ)` (eV), one per training
configuration — the same residuals the energy `R²` / RMSE are built from.
"""
residuals_energy(f::SLCEFit)::Vector{Float64} = f.residuals

"""
    residuals_torque(f::SLCEFit) -> Vector{Float64}

The torque residuals `y_T − X_T·jϕ` over the flattened (config-major, then atom-major,
then `xyz`) per-atom torque components (eV). Requires a torque-carrying dataset; the
torque prediction has no intercept (`j0` does not enter it).
"""
function residuals_torque(f::SLCEFit)::Vector{Float64}
    has_torque(f.dataset) || throw(ArgumentError("dataset has no torque data"))
    return f.dataset.y_T .- f.dataset.X_T * f.jphi
end

"""
    rss_energy(f::SLCEFit) -> Float64

Energy residual sum of squares `Σ (y_E − ŷ_E)²` (eV²).
"""
rss_energy(f::SLCEFit)::Float64 = sum(abs2, residuals_energy(f))

"""
    rss_torque(f::SLCEFit) -> Float64

Torque residual sum of squares `Σ (y_T − X_T·jϕ)²` (eV²). Requires a torque-carrying
dataset.
"""
rss_torque(f::SLCEFit)::Float64 = sum(abs2, residuals_torque(f))

"""
    r2_energy(f::SLCEFit) -> Float64

In-sample energy coefficient of determination `R²` (1 = the SALC span reproduces every
training energy). A degenerate target (all training energies equal ⇒ zero total variance)
returns `1.0` by convention.
"""
function r2_energy(f::SLCEFit)::Float64
    y = f.dataset.y_E
    ss_tot = sum(abs2, y .- mean(y))
    return ss_tot == 0 ? 1.0 : 1 - rss_energy(f) / ss_tot
end

"""
    rmse_energy(f::SLCEFit) -> Float64

In-sample energy root-mean-square error (same unit as the training energies, typically eV).
"""
rmse_energy(f::SLCEFit)::Float64 = sqrt(rss_energy(f) / nobs(f))

"""
    r2_torque(f::SLCEFit) -> Float64

In-sample torque `R²`, computed over the flattened per-atom torque components. Requires a
torque-carrying dataset. The torque prediction `τ = X_T·jϕ` has no intercept (`j0` does
not enter it), so the `R²` baseline is the uncentered total sum of squares `Σ τ²` (the
error of the zero predictor), not the mean-centered one.
"""
function r2_torque(f::SLCEFit)::Float64
    ss_res = rss_torque(f)          # validates the dataset has torque (throws if not)
    ss_tot = sum(abs2, f.dataset.y_T)  # no-intercept model ⇒ uncentered baseline
    return ss_tot == 0 ? 1.0 : 1 - ss_res / ss_tot
end

"""
    rmse_torque(f::SLCEFit) -> Float64

In-sample torque root-mean-square error over the flattened per-atom torque components
(eV). Requires a torque-carrying dataset.
"""
rmse_torque(f::SLCEFit)::Float64 = sqrt(rss_torque(f) / length(f.dataset.y_T))

"""
    residuals_force(f::SLCEFit) -> Vector{Float64}

The force residuals `y_F − X_F·jϕ[force_cols]` over the flattened (config-major, then
displacement-referenced-atom-major, then `xyz`) per-atom force components (eV/Å).
Requires a force-carrying dataset; the force prediction has no intercept (`j0` does not
enter it), and only the displacement-active SALC columns contribute (the stored `X_F`
is compact — see [`SLCEDataset`](@ref)).
"""
function residuals_force(f::SLCEFit)::Vector{Float64}
    d = f.dataset
    has_force(d) || throw(ArgumentError("dataset has no force data"))
    return d.y_F .- d.X_F * f.jphi[d.force_cols]
end

"""
    rss_force(f::SLCEFit) -> Float64

Force residual sum of squares `Σ (y_F − ŷ_F)²` ((eV/Å)²). Requires a force-carrying
dataset.
"""
rss_force(f::SLCEFit)::Float64 = sum(abs2, residuals_force(f))

"""
    r2_force(f::SLCEFit) -> Float64

In-sample force `R²`, computed over the flattened per-atom force components. Requires a
force-carrying dataset. Like the torque block, the force prediction has no intercept, so
the `R²` baseline is the uncentered total sum of squares `Σ f²` (the error of the zero
predictor).
"""
function r2_force(f::SLCEFit)::Float64
    ss_res = rss_force(f)           # validates the dataset has force (throws if not)
    ss_tot = sum(abs2, f.dataset.y_F)  # no-intercept model ⇒ uncentered baseline
    return ss_tot == 0 ? 1.0 : 1 - ss_res / ss_tot
end

"""
    rmse_force(f::SLCEFit) -> Float64

In-sample force root-mean-square error over the flattened per-atom force components
(eV/Å). Requires a force-carrying dataset.
"""
rmse_force(f::SLCEFit)::Float64 = sqrt(rss_force(f) / length(f.dataset.y_F))

# --- identifiability of the assembled design ---------------------------------------

# Declared BEFORE the docstring below: a `const` between a docstring and the
# function it documents silently steals the docstring (Documenter strict then
# fails on the missing `@docs` entry — the unit suite cannot see it).
const IdentifiabilityReport = @NamedTuple{ncols::Int, rank::Int, nullity::Int,
                                          sigma_min::Float64, sigma_max::Float64,
                                          sigma_cut::Float64, gap::Float64}

"""
    identifiability(f::SLCEFit; rtol = nothing) -> NamedTuple
    identifiability(dataset::SLCEDataset; torque_weight = 0.0, force_weight = 0.0,
                    asr = true, rtol = nothing) -> NamedTuple

Rank diagnosis of the design the fit actually solves: how many coefficient
directions the training data determine, and how many they leave **flat**. Returns

    (; ncols, rank, nullity, sigma_min, sigma_max, sigma_cut, gap)

where `ncols` is the number of free parameters the estimator sees (`p` unconstrained,
`q = p − rank(A)` under the ASR reparameterization — the diagnosis is in the same
coordinates the solve uses), `rank` is the numerical rank of the assembled
(centered / whitened, stacked) design at `sigma_cut = rtol·σ_max`, and
`nullity = ncols − rank` counts the directions along which the objective is exactly
flat. `nullity > 0` means the data do **not** determine the model: an unpenalized
solve returns the estimator's own null-space choice (`OLS`'s minimum-norm
solution), and a penalized one returns whatever the penalty prefers — in both cases
a number that is not an estimate. Different fitted coefficients can then reproduce
the training observables to machine precision.

`rtol` is **relative** and `sigma_cut` is the **absolute** singular value it produced,
so the two are not interchangeable: feeding a report's `sigma_cut` back in as `rtol`
would square the cut and declare a rank-deficient design identified. That asymmetry is
why the field is not called `tol`.

`rtol` defaults to `min(size(X))·eps` (`LinearAlgebra.rank`'s convention;
deliberately **not** `max(size(X))·eps` — the blocks are whitened by `1/√n`, so the
singular values are sample-size-independent while a `max`-based cut grows with the
row count, and merely adding data could flip a determined direction to "flat").
`gap` is the ratio of the smallest kept to the largest dropped singular value, i.e.
how clean the rank decision was: `Inf` when nothing was dropped, and a value within
a few orders of 1 means the spectrum has no clear break and the rank is
**ambiguous** — treat such a report as inconclusive rather than as a count. Note
that the ASR decides `rank(A)` at its own fixed `1e-10` relative cut (with a
forbidden-band refusal, `fitting/asr.jl`), and that `OLS`'s `\\` makes a third rank
decision inside its pivoted QR; on a well-posed design all three agree with orders
of margin, and `gap` is what tells you whether that holds here.

The fit method reassembles the problem exactly as [`fit`](@ref) did (same weights,
same `asr` setting); the dataset method answers the same question **before** fitting,
for a `(torque_weight, force_weight, asr)` combination one is considering. The
report is weight-dependent by construction: a *small* nonzero `torque_weight` /
`force_weight` scales that block by `√(w/n)`, so directions only that block sees can
fall under the `σ_max`-relative cut and read out as flat when they are merely
downweighted.

On a [`refit`](@ref) result the report still covers the **full** design: a refit's
support is not recorded on `SLCEFit`, and the numbers then describe the space the
de-biasing happened inside. That is a bound in the safe direction — dropping
columns never raises the nullity — so `nullity == 0` on a refit result still means
its support sub-design is determined too.

This is the exact counterpart of the dead-column warning `fit` always emits: that
one is structural (columns the data do not touch at all), while a flat direction is
generally a *combination* of columns. The canonical example is a derivative-only
fit on center-of-mass-free displacements — the physically natural DFT sampling —
where translation-violating directions become invisible: which ones depends on the
channel, and for a basis whose displacement content is all spin-decorated the
torque channel is blind to exactly the violating subspace, so such a fit is
rank-deficient without the ASR and of full rank `q` with it. A basis carrying
spin-free (lattice-only) sectors is not covered by that statement — torque is blind
to all spin-independent content, so its deficiency exceeds `rank(A)` and survives
the constraint. Both cases are gated in `test/unit/test_identifiability.jl`.

Cost is a full SVD of the assembled design, `O(n·q²)`; this is a diagnostic to call
deliberately, not a per-fit check.
"""
function identifiability(f::SLCEFit;
                         rtol::Union{Nothing,Real} = nothing)::IdentifiabilityReport
    rep = f.reparam                       # the STAGE's Z on a staged fit
    X, _, _, _, _ = _assemble_problem(f.dataset, f.torque_weight, f.force_weight, rep)
    return _identifiability(X, rtol)
end

function identifiability(dataset::SLCEDataset; torque_weight::Real = 0.0,
                         force_weight::Real = 0.0, asr::Bool = true,
                         rtol::Union{Nothing,Real} = nothing)::IdentifiabilityReport
    w = Float64(torque_weight)
    wF = Float64(force_weight)
    _validate_fit_request(dataset, w, wF)
    rep = _resolve_asr_rep(dataset, asr)
    X, _, _, _, _ = _assemble_problem(dataset, w, wF, rep)
    return _identifiability(X, rtol)
end

function _identifiability(X::AbstractMatrix,
                          rtol::Union{Nothing,Real})::IdentifiabilityReport
    q = size(X, 2)
    s = svdvals(X)
    smax = isempty(s) ? 0.0 : s[1]
    rt = rtol === nothing ? minimum(size(X)) * eps(Float64) : Float64(rtol)
    (rt >= 0 && isfinite(rt)) ||
        throw(ArgumentError("rtol must be finite and ≥ 0; got $rtol"))
    scut = rt * smax                      # ABSOLUTE cut; `rt` is the relative one
    r = count(>(scut), s)
    # `svdvals` returns min(n, q) values: with fewer rows than columns the smallest
    # SINGULAR value of the q-dimensional operator is exactly 0, not `s[end]`.
    smin = length(s) < q ? 0.0 : (isempty(s) ? 0.0 : s[end])
    # How clean the cut was. A dropped singular value of exactly 0 (or nothing
    # dropped at all — including the structural zeros when n < q, which never enter
    # `s`) is an unambiguous decision.
    kept = r >= 1 ? s[r] : 0.0
    dropped = r < length(s) ? s[r + 1] : 0.0
    gap = dropped == 0.0 ? Inf : kept / dropped
    return (; ncols = q, rank = r, nullity = q - r, sigma_min = smin,
            sigma_max = smax, sigma_cut = scut, gap = gap)
end

# --- StatsAPI generics defaulting to the energy block ------------------------------
# An SCE fit has two observable blocks (energy / torque); the unqualified StatsAPI
# generics resolve to the **energy** block by convention (the torque block keeps its
# explicit `*_torque` accessors). Extending — not shadowing — the generics keeps
# `using GLM` / `using StatsBase` collision-free.

"""
    residuals(f::SLCEFit) -> Vector{Float64}

The energy residuals — [`residuals_energy`](@ref). Extends `StatsAPI.residuals`; the
torque block is [`residuals_torque`](@ref).
"""
residuals(f::SLCEFit)::Vector{Float64} = residuals_energy(f)

"""
    r2(f::SLCEFit) -> Float64

The energy `R²` — [`r2_energy`](@ref). Extends `StatsAPI.r2`; the torque block is
[`r2_torque`](@ref).
"""
r2(f::SLCEFit)::Float64 = r2_energy(f)

"""
    predict(model_or_fit, configs) -> Float64 or Vector{Float64}

The predicted energy — [`predict_energy`](@ref). Extends `StatsAPI.predict`; the
torque observable is [`predict_torque`](@ref).
"""
predict(m::SLCEModel, data) = predict_energy(m, data)
predict(f::SLCEFit, data) = predict_energy(f, data)
