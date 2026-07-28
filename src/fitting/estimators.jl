"""
    AbstractEstimator

Regression strategy. Implement one by subtyping this and defining

    solve_coefficients(est, X, y) -> Vector{Float64}

where `(X, y)` is already **column-centered** (so the solver adds no intercept;
the reference energy `j0` is recovered analytically downstream). In-tree
estimators: [`OLS`](@ref), [`Ridge`](@ref). GLMNet-backed estimators (Lasso /
ElasticNet) are provided by a package extension.
"""
abstract type AbstractEstimator end

"""
    OLS()

Ordinary least squares, solved by pivoted QR (`X \\ y`) — **not** the normal
equations, which would square the condition number. On a rank-deficient design this
returns the minimum-norm solution rather than an arbitrary one, which is why every
other closed-form estimator routes here at zero penalty.
"""
struct OLS <: AbstractEstimator end

"""
    Ridge(; lambda = 0.0)

L2-penalized least squares with penalty `lambda ≥ 0`. `lambda = 0` routes to
[`OLS`](@ref)'s QR path, exactly (not within a tolerance).

For `lambda > 0` this solves the normal equations `(XᵀX + λI)β = Xᵀy`, which is safe
because the penalty makes the matrix positive definite, but which does square the
condition number of `X`. On an ill-conditioned design (the ASR forbidden band admits
singular-value ratios down to `1e-12`) prefer a larger `lambda`, or `OLS` if the
intent was no penalty at all.
"""
struct Ridge <: AbstractEstimator
    lambda::Float64

    function Ridge(lambda::Real)
        (lambda >= 0 && isfinite(lambda)) ||
            throw(ArgumentError("lambda must be finite and ≥ 0; got $lambda"))
        return new(Float64(lambda))
    end
end
Ridge(; lambda::Real = 0.0) = Ridge(lambda)

"""
    ElasticNet(; alpha = 1.0, lambda = nothing, standardize = true, nfolds = 10,
               select = :lambda_min, seed = 1, nlambda = 100)

Elastic-net (mixed L1/L2) regression, backed by GLMNet. The method lights up only
when GLMNet is loaded (`using GLMNet` activates the extension); the type itself
lives in the core package so it can be named and dispatched on without the heavy
dependency.

`alpha ∈ [0, 1]` is the elastic-net mixing — `alpha = 1` is the Lasso (pure L1),
`alpha = 0` is ridge-like (pure L2). GLMNet minimizes

    (1 / 2n)·‖y − Xβ‖² + λ·[ (1 − α)/2·‖β‖₂² + α·‖β‖₁ ]

over the **column-centered** `(X, y)` it is handed (see [`AbstractEstimator`](@ref));
no intercept is fitted, so `j0` is still recovered analytically downstream. Note the
`1/2n` and the L1 term: this `lambda` is on a different scale than [`Ridge`](@ref)'s
plain `λ‖β‖₂²`, and is **not** directly comparable.

`standardize = true` (the conventional Lasso default) scales each column to unit
variance before fitting, so with it the penalty effectively acts as `λ·std(colⱼ)` per
column — the L1 term treats differently-scaled SALC columns evenhandedly. Coefficients
are always returned on the original scale.

Penalty strength:
- `lambda = nothing` (default): chosen by `nfolds`-fold cross-validation over
  GLMNet's automatic λ path (the fold count is capped at `n ÷ 3` for small samples).
  `select` picks which λ — `:lambda_min` (lowest CV error) or `:lambda_1se` (the
  sparsest model within one standard error of the minimum, the conventional
  parsimonious choice). `seed` fixes the fold assignment so the fit is reproducible
  within a Julia session/version. `nlambda` sizes the λ grid.
- `lambda = ⟨number⟩`: fit at exactly that penalty, skipping CV.

For an energy+torque co-fit (see [`fit`](@ref)) the CV folds are **grouped by
configuration** — a configuration's energy row and all its torque-component rows go
into the same fold — so resampling never splits one configuration. Because the design
is the whitened, stacked energy+torque matrix, the per-column standardization (hence
the effective penalty on each SALC) depends on `torque_weight`; `lambda` is therefore
not directly comparable between an energy-only fit and a co-fit.
"""
struct ElasticNet <: AbstractEstimator
    alpha::Float64
    lambda::Union{Nothing,Float64}
    standardize::Bool
    nfolds::Int
    select::Symbol
    seed::Int
    nlambda::Int
end
function ElasticNet(; alpha::Real = 1.0, lambda::Union{Nothing,Real} = nothing,
                    standardize::Bool = true, nfolds::Integer = 10,
                    select::Symbol = :lambda_min, seed::Integer = 1,
                    nlambda::Integer = 100)
    (0 <= alpha <= 1) || throw(ArgumentError("alpha must be in [0, 1]; got $alpha"))
    (lambda === nothing || (lambda >= 0 && isfinite(lambda))) ||
        throw(ArgumentError("lambda must be ≥ 0 (or nothing for CV); got $lambda"))
    select in (:lambda_min, :lambda_1se) ||
        throw(ArgumentError("select must be :lambda_min or :lambda_1se; got :$select"))
    nfolds >= 2 || throw(ArgumentError("nfolds must be ≥ 2; got $nfolds"))
    nlambda >= 1 || throw(ArgumentError("nlambda must be ≥ 1; got $nlambda"))
    return ElasticNet(Float64(alpha), lambda === nothing ? nothing : Float64(lambda),
                      standardize, Int(nfolds), select, Int(seed), Int(nlambda))
end

"""
    Lasso(; kwargs...)

The Lasso (L1-penalized) estimator: [`ElasticNet`](@ref) with `alpha = 1`. Takes
the same keyword arguments except `alpha`.
"""
Lasso(; kwargs...) = ElasticNet(; alpha = 1.0, kwargs...)

"""
    AdaptiveLasso(; pilot = OLS(), lambda = nothing, gamma = 1.0,
                  epsilon = eps(Float64), standardize = true, nfolds = 10,
                  select = :lambda_min, seed = 1, nlambda = 100)

One-shot Adaptive Lasso (Zou 2006), backed by GLMNet — the solve lights up only
when GLMNet is loaded (`using GLMNet` activates the extension); the type lives in
the core package so it can be named and dispatched on without the heavy dependency.

The fit is two-stage. First `pilot` (any [`AbstractEstimator`](@ref)) is run on the
same centered `(X, y)` to get a pilot coefficient vector `β̂`. Then a **weighted**
Lasso is solved with per-column penalty factors

    wⱼ = 1 / max(|β̂ⱼ|, epsilon)^gamma

so columns the pilot found large are penalized little and columns it found small are
penalized heavily — the data-driven reweighting that gives the Adaptive Lasso its
oracle selection property. `gamma = 0` reduces to a plain Lasso (uniform weights);
`gamma = 1` (the default) is the Zou 2006 / ALAMODE recipe. `epsilon > 0` floors the
pilot magnitude so a numerically-zero pilot coefficient gives a large-but-finite
penalty rather than `Inf`. GLMNet internally rescales the weights to sum to the
column count, so `lambda` is **not** directly comparable to a plain [`Lasso`](@ref)'s
once `gamma > 0`. The pilot is fit on the un-standardized design and `standardize =
true` then penalizes in the standardized coordinate, so the effective per-column weight
carries an extra `std(colⱼ)` factor on top of `1/|β̂ⱼ|^γ` — the ALAMODE convention,
consistent with [`ElasticNet`](@ref) / [`Lasso`](@ref) here (set `standardize = false`
for the textbook Zou form).

`pilot = OLS()` matches Zou 2006. For a rank-deficient or near-collinear design (e.g.
`n_salcs ≥ nconfig` at `torque_weight = 0`) prefer `pilot = Ridge(lambda = small)`: the
OLS minimum-norm solution puts ~1e-10 noise in the null-space directions, which is
not floored by `epsilon = eps(Float64)` and miscalibrates those weights. Any pilot is
allowed, including a cross-validating [`ElasticNet`](@ref) or a [`PrecomputedPilot`](@ref)
reusing a prior fit's coefficients; the pilot's own solve receives the co-fit `row_groups`.

Penalty strength and the remaining keywords (`lambda`, `standardize`, `nfolds`,
`select`, `seed`, `nlambda`) behave exactly as in [`ElasticNet`](@ref): `lambda =
nothing` (default) selects λ by configuration-grouped cross-validation with the
adaptive weights held fixed across the path; a number fits at that fixed penalty.
"""
struct AdaptiveLasso <: AbstractEstimator
    pilot::AbstractEstimator
    lambda::Union{Nothing,Float64}
    gamma::Float64
    epsilon::Float64
    standardize::Bool
    nfolds::Int
    select::Symbol
    seed::Int
    nlambda::Int
end
function AdaptiveLasso(; pilot::AbstractEstimator = OLS(),
                       lambda::Union{Nothing,Real} = nothing, gamma::Real = 1.0,
                       epsilon::Real = eps(Float64), standardize::Bool = true,
                       nfolds::Integer = 10, select::Symbol = :lambda_min,
                       seed::Integer = 1, nlambda::Integer = 100)
    (lambda === nothing || (lambda >= 0 && isfinite(lambda))) ||
        throw(ArgumentError("lambda must be ≥ 0 (or nothing for CV); got $lambda"))
    gamma >= 0 || throw(ArgumentError("gamma must be ≥ 0; got $gamma"))
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0; got $epsilon"))
    select in (:lambda_min, :lambda_1se) ||
        throw(ArgumentError("select must be :lambda_min or :lambda_1se; got :$select"))
    nfolds >= 2 || throw(ArgumentError("nfolds must be ≥ 2; got $nfolds"))
    nlambda >= 1 || throw(ArgumentError("nlambda must be ≥ 1; got $nlambda"))
    return AdaptiveLasso(pilot, lambda === nothing ? nothing : Float64(lambda),
                         Float64(gamma), Float64(epsilon), standardize, Int(nfolds),
                         select, Int(seed), Int(nlambda))
end

"""
    PrecomputedPilot(beta)

Estimator adapter that returns a fixed coefficient vector from `solve_coefficients`,
ignoring `(X, y)` except for a length check against `size(X, 2)`. Intended as an
[`AdaptiveLasso`](@ref) `pilot`: it lets the adaptive reweighting reuse coefficients
from a previous fit (`coef(f)`) instead of running a fresh pilot regression. The
vector is copied at construction, so later mutation of the caller's storage does not
leak in.
"""
struct PrecomputedPilot <: AbstractEstimator
    beta::Vector{Float64}
    PrecomputedPilot(b::AbstractVector{<:Real}) = new(Vector{Float64}(b))
end

"""
    AdaptiveRidge(; lambda, epsilon = 1e-8, max_iter = 50, tol = 1e-6)

Iterative Adaptive Ridge (Frommlet & Nuel 2016). Approximates an L0-penalized fit by
repeatedly refitting a per-coefficient weighted ridge problem

    min_b ‖y − Xb‖² + lambda·Σⱼ wⱼ·bⱼ²

and updating the weights `wⱼ = 1 / (bⱼ² + epsilon)` between iterations. Iteration zero
is a plain ridge solve (uniform weights); each subsequent step rebuilds the weights
from the current coefficients, so large coefficients get a light penalty and small
ones a heavy penalty — iterating drives the small ones toward zero (an L0
approximation). Each subproblem is the analytic weighted ridge
`b = (X'X + lambda·Diagonal(w)) \\ (X'y)`, the same closed-form family as
[`Ridge`](@ref); no GLMNet is involved, so this estimator works in the core package
without the extension.

The iteration stops when the relative infinity-norm change in the coefficient vector
drops below `tol`, or after `max_iter` reweighting steps. `lambda = 0` reduces to
[`OLS`](@ref). The penalty acts directly on the coefficients (no `standardize`
keyword): the per-cluster `(4π)^(N/2)` basis scale already places them on the
conventional spin-model scale, so a single `epsilon` is a roughly uniform magnitude
floor across clusters.
"""
struct AdaptiveRidge <: AbstractEstimator
    lambda::Float64
    epsilon::Float64
    max_iter::Int
    tol::Float64
end
function AdaptiveRidge(; lambda::Real, epsilon::Real = 1e-8, max_iter::Integer = 50,
                       tol::Real = 1e-6)
    (lambda >= 0 && isfinite(lambda)) ||
        throw(ArgumentError("lambda must be finite and ≥ 0; got $lambda"))
    epsilon > 0 || throw(ArgumentError("epsilon must be > 0; got $epsilon"))
    max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1; got $max_iter"))
    tol > 0 || throw(ArgumentError("tol must be > 0; got $tol"))
    return AdaptiveRidge(Float64(lambda), Float64(epsilon), Int(max_iter), Float64(tol))
end

"""
    GroupAdaptiveRidge(column_groups, group_weights; lambda, epsilon = 1e-8,
                       max_iter = 50, tol = 1e-6)

Group extension of [`AdaptiveRidge`](@ref): iterative reweighted ridge approximating the
weighted **group-L0** penalty `lambda·Σ_g v_g·1{β_g ≠ 0}`. Each reweighting step solves
the analytic weighted ridge

    min_b ‖y − Xb‖² + lambda·Σⱼ wⱼ·bⱼ²,    wⱼ = v_g / (‖β_g‖² + p_g·epsilon)

where `g = column_groups[j]`, `β_g` is the current coefficient sub-vector of group `g`,
`p_g` its column count, and `v_g = group_weights[g]` a **fixed** positive per-group
weight. At a converged fixed point the penalty contribution of a surviving group tends
to `lambda·v_g` (a constant per alive group), so `v_g` is exactly the group-L0 weight —
e.g. a Monte-Carlo cost weight from `SLCE.cost_weights`. The denominator equals
`p_g·(mean_j βⱼ² + epsilon)`, so `epsilon` is a per-coefficient magnitude floor
independent of the group size (the same calibration as [`AdaptiveRidge`](@ref), whose
update this reproduces exactly for singleton groups with unit weights).

`column_groups` labels the design-matrix **columns** with contiguous group ids `1:G`
(every label present; for an SLCE fit use `SLCE.salc_groups`); it is unrelated to
the per-**row** `row_groups` keyword of [`solve_coefficients`](@ref), which this estimator
ignores. `group_weights` has length `G`. Both vectors are copied at construction.
`lambda = 0` reduces to [`OLS`](@ref). Like `AdaptiveRidge`, the converged fit is a
linear smoother in the fixed-weight sense ([`islinear`](@ref) is `true`), so
[`gcv`](@ref) applies. Iterating to convergence drives whole groups toward zero; follow
with [`refit`](@ref) to select the support and de-bias the survivors, or drive the λ
path with [`select_fit`](@ref).
"""
struct GroupAdaptiveRidge <: AbstractEstimator
    lambda::Float64
    column_groups::Vector{Int}
    group_weights::Vector{Float64}
    epsilon::Float64
    max_iter::Int
    tol::Float64
    group_sizes::Vector{Int}     # p_g, derived and cached by the inner constructor

    function GroupAdaptiveRidge(lambda::Real, column_groups::AbstractVector{<:Integer},
                                group_weights::AbstractVector{<:Real}, epsilon::Real,
                                max_iter::Integer, tol::Real)
        (lambda >= 0 && isfinite(lambda)) ||
            throw(ArgumentError("lambda must be finite and ≥ 0; got $lambda"))
        epsilon > 0 || throw(ArgumentError("epsilon must be > 0; got $epsilon"))
        max_iter >= 1 || throw(ArgumentError("max_iter must be ≥ 1; got $max_iter"))
        tol > 0 || throw(ArgumentError("tol must be > 0; got $tol"))
        isempty(column_groups) &&
            throw(ArgumentError("column_groups must be nonempty (one label per column)"))
        minimum(column_groups) >= 1 ||
            throw(ArgumentError("column_groups labels must be ≥ 1; got " *
                                "$(minimum(column_groups))"))
        G = Int(maximum(column_groups))
        sizes = zeros(Int, G)
        for g in column_groups
            sizes[g] += 1
        end
        all(>(0), sizes) ||
            throw(ArgumentError("column_groups labels must cover 1:$G with no gaps; " *
                                "empty group(s): $(findall(==(0), sizes))"))
        length(group_weights) == G ||
            throw(ArgumentError("group_weights length $(length(group_weights)) ≠ " *
                                "number of groups $G"))
        all(v -> isfinite(v) && v > 0, group_weights) ||
            throw(ArgumentError("group_weights must be finite and > 0"))
        return new(Float64(lambda), Vector{Int}(column_groups),
                   Vector{Float64}(group_weights), Float64(epsilon), Int(max_iter),
                   Float64(tol), sizes)
    end
end
function GroupAdaptiveRidge(column_groups::AbstractVector{<:Integer},
                            group_weights::AbstractVector{<:Real};
                            lambda::Real, epsilon::Real = 1e-8, max_iter::Integer = 50,
                            tol::Real = 1e-6)
    return GroupAdaptiveRidge(lambda, column_groups, group_weights, epsilon, max_iter, tol)
end

# Compact display: the default struct printer would dump the whole `beta` vector
# (hundreds–thousands of entries) and recurse into the nested pilot.
Base.show(io::IO, p::PrecomputedPilot) =
    print(io, "PrecomputedPilot(", length(p.beta), " coefficients)")
Base.show(io::IO, e::AdaptiveLasso) =
    print(io, "AdaptiveLasso(pilot=", e.pilot, ", lambda=", e.lambda, ", gamma=",
          e.gamma, ", epsilon=", e.epsilon, ", standardize=", e.standardize, ")")
Base.show(io::IO, e::GroupAdaptiveRidge) =
    print(io, "GroupAdaptiveRidge(", length(e.column_groups), " columns in ",
          length(e.group_weights), " groups, lambda=", e.lambda,
          ", epsilon=", e.epsilon, ")")

"""
    islinear(est) -> Bool

Whether `est` is a linear (closed-form) estimator — one whose fitted values are
`ŷ = H·y` with a hat matrix `H` independent of `y`. Gates closed-form diagnostics
(e.g. [`gcv`](@ref) / [`effective_dof`](@ref)). [`OLS`](@ref), [`Ridge`](@ref),
[`AdaptiveRidge`](@ref), and [`GroupAdaptiveRidge`](@ref) (the latter two linear in
the converged-weight sense, the standard adaptive-ridge approximation) return `true`;
the penalty-path / pilot estimators ([`ElasticNet`](@ref) / [`Lasso`](@ref) /
[`AdaptiveLasso`](@ref)) are not linear. Extends `StatsAPI.islinear`.
"""
islinear(::AbstractEstimator) = false
islinear(::OLS) = true
islinear(::Ridge) = true
islinear(::AdaptiveRidge) = true
islinear(::GroupAdaptiveRidge) = true

"""
    solve_coefficients(est, X, y; row_groups = nothing, nullspace = nothing) -> Vector{Float64}

Solve for the SALC coefficients from the (already centered) design matrix `X` and
target `y`. See [`AbstractEstimator`](@ref) for the `(X, y)` contract.

`row_groups` is an optional per-row label vector (one entry per row of `X`) marking rows
that derive from the same physical sample — in an energy+torque co-fit, a
configuration's energy row and its torque-component rows share a label. Estimators
that resample rows (cross-validating [`ElasticNet`](@ref) / [`Lasso`](@ref)) keep
same-label rows together so the resampling does not leak within-sample structure;
estimators with a closed-form fit ([`OLS`](@ref) / [`Ridge`](@ref)) ignore it.

`nullspace` (ASR-constrained fits) is the **orthonormal** basis `Z` of the
constraint null space: the caller passes the reparameterized design `X̃ = X_β·Z`
and receives γ with `β = Z·γ`. Penalties stay defined on β: quadratic estimators
whose weights depend on β ([`AdaptiveRidge`](@ref) / [`GroupAdaptiveRidge`](@ref))
evaluate them at `β = Z·γ` each iteration and solve the compressed SPD system
`X̃'X̃ + λ·Z'·D(β)·Z`; for [`OLS`](@ref)/[`Ridge`](@ref) orthonormality makes the
plain γ-space solve already the β-penalized one (`‖β‖ = ‖γ‖`). L1 estimators
(GLMNet extension) reject it — L1 under `Z` is a generalized lasso GLMNet cannot
solve, and L1 on γ is factorization-gauge-dependent.
"""
function solve_coefficients(est::AbstractEstimator, X::AbstractMatrix, y::AbstractVector;
                            row_groups = nothing, nullspace = nothing)
    error("solve_coefficients has no method for $(typeof(est)); load the backend " *
          "package (e.g. `using GLMNet` for Lasso/ElasticNet).")
end

function solve_coefficients(::OLS, X::AbstractMatrix, y::AbstractVector;
                            row_groups = nothing, nullspace = nothing)::Vector{Float64}
    # `nullspace` is inert: the QR min-norm γ maps to the min-norm feasible β
    # (orthonormal Z), so the γ-space OLS IS the constrained OLS.
    return X \ y   # QR-based least squares (more robust than the normal equations)
end

function solve_coefficients(est::Ridge, X::AbstractMatrix, y::AbstractVector;
                            row_groups = nothing, nullspace = nothing)::Vector{Float64}
    # `nullspace` is inert: with orthonormal Z, λ‖γ‖² = λ‖Z·γ‖² = λ‖β‖² — the
    # γ-space ridge is verbatim the β-penalized constrained ridge.
    #
    # Exact zero only: at lambda = 0 the penalty is gone and `XtX` may be singular, so
    # route to the QR (min-norm) OLS path — the same guard `AdaptiveRidge` and
    # `GroupAdaptiveRidge` carry, and for the same reason. Do NOT widen this to a
    # near-zero tolerance. Without it a rank-deficient design does not throw: the
    # symmetric factorization of a singular matrix returns numerical garbage, measured
    # at ‖β‖ ~ 1e16 against OLS's 0.48 on a design with one duplicated column — silently,
    # from an estimator whose docstring promises OLS at lambda = 0.
    est.lambda == 0.0 && return solve_coefficients(OLS(), X, y)
    n = size(X, 2)
    return Symmetric(X' * X + est.lambda * I(n)) \ (X' * y)
end

function solve_coefficients(est::AdaptiveRidge, X::AbstractMatrix, y::AbstractVector;
                            row_groups = nothing,
                            nullspace::Union{Nothing,Matrix{Float64}} = nothing)::Vector{Float64}
    # Exact zero only: at lambda = 0 the penalty diagonal is gone and `XtX` may be
    # singular, so route to the QR (min-norm) OLS path. Any lambda > 0 makes
    # `XtX + lambda·Diagonal(w)` SPD below, so the symmetric solve is safe — do NOT
    # widen this to a near-zero tolerance.
    est.lambda == 0.0 && return solve_coefficients(OLS(), X, y)
    XtX = X' * X
    Xty = X' * y
    q = size(XtX, 2)
    nullspace === nothing || size(nullspace, 2) == q || throw(DimensionMismatch(
        "nullspace has $(size(nullspace, 2)) columns but the design has $q"))
    # Iteration 0: uniform-weight ridge (numerically Ridge(lambda) — with Z
    # orthonormal, Z'·I·Z = I, so the same seed serves the constrained loop). Only
    # the penalty changes between iterations; the Gram matrix `XtX` is formed once.
    # With lambda > 0 and every wⱼ > 0 (epsilon > 0), the shifted matrix is SPD
    # throughout (Z full column rank).
    coefs = Symmetric(XtX + est.lambda * I(q)) \ Xty
    p = nullspace === nothing ? q : size(nullspace, 1)
    w = Vector{Float64}(undef, p)
    for _ = 1:est.max_iter
        beta = nullspace === nothing ? coefs : nullspace * coefs
        @. w = 1.0 / (beta^2 + est.epsilon)     # the β-space weight map, unchanged
        P = nullspace === nothing ? est.lambda * Diagonal(w) :
            est.lambda * Symmetric(nullspace' * (w .* nullspace))
        coefs_new = Symmetric(XtX + P) \ Xty
        # Relative ∞-norm change ON β (γ's ∞-norm is factorization-gauge-
        # dependent under Z); the eps floor only guards the all-zero case.
        beta_new = nullspace === nothing ? coefs_new : nullspace * coefs_new
        rel = mapreduce((a, b) -> abs(a - b), max, beta_new, beta) /
              max(maximum(abs, beta_new), eps(Float64))
        coefs = coefs_new
        rel < est.tol && break
    end
    return coefs
end

# The group-adaptive-ridge weight map — the ONLY definition of the update
# `wⱼ = v_g / (‖β_g‖² + p_g·ε)`. The GCV / effective-dof diagnostics recompute the
# converged penalty diagonal through this same function (`_penalty_diagonal` in
# fitting/selection.jl); change the formula here and nowhere else. `normsq` is a
# reusable length-`G` accumulator; fills `w` in place and returns it.
function _gar_weights!(w::Vector{Float64}, beta::Vector{Float64},
                       column_groups::Vector{Int}, group_weights::Vector{Float64},
                       group_sizes::Vector{Int}, epsilon::Float64,
                       normsq::Vector{Float64})::Vector{Float64}
    fill!(normsq, 0.0)
    @inbounds for j in eachindex(beta)
        normsq[column_groups[j]] += abs2(beta[j])
    end
    @inbounds for j in eachindex(beta)
        g = column_groups[j]
        w[j] = group_weights[g] / (normsq[g] + group_sizes[g] * epsilon)
    end
    return w
end

# The reweighted-ridge iteration on a precomputed Gram system, shared by
# `solve_coefficients(::GroupAdaptiveRidge, …)` (`beta0 = nothing`, cold start) and the
# `select_fit` λ-path driver (`beta0` = the previous λ's solution, warm start; the
# caller forms `XtX`/`Xty` once for the whole path). Requires `lambda > 0` — the
# callers route the exact `lambda == 0` case to OLS.
function _solve_gar(XtX::Matrix{Float64}, Xty::Vector{Float64}, lambda::Float64,
                    column_groups::Vector{Int}, group_weights::Vector{Float64},
                    group_sizes::Vector{Int}, epsilon::Float64, max_iter::Int,
                    tol::Float64;
                    beta0::Union{Nothing,Vector{Float64}} = nothing,
                    nullspace::Union{Nothing,Matrix{Float64}} = nothing)::Vector{Float64}
    q = length(Xty)
    p = nullspace === nothing ? q : size(nullspace, 1)
    w = Vector{Float64}(undef, p)
    normsq = Vector{Float64}(undef, length(group_weights))
    # Compressed penalty: Diagonal(w) unconstrained, Z'·D·Z under ASR (the weight
    # map stays in β space — the "group penalties on β unchanged" pin).
    penalty(wv) = nullspace === nothing ? lambda * Diagonal(wv) :
                  lambda * Symmetric(nullspace' * (wv .* nullspace))
    # Iteration 0 (cold start): the fixed-weight ridge `wⱼ = v_g` — with unit weights
    # this is numerically Ridge(lambda), matching the AdaptiveRidge initialization. A
    # warm start replaces it with the caller's `beta0` (γ0 under ASR) and enters the
    # loop directly. With lambda > 0 and every wⱼ > 0 (epsilon > 0, Z full column
    # rank), the shifted matrix is SPD throughout — do NOT widen the callers'
    # `lambda == 0` routing to a tolerance.
    coefs = if beta0 === nothing
        @inbounds for j = 1:p
            w[j] = group_weights[column_groups[j]]
        end
        Symmetric(XtX + penalty(w)) \ Xty
    else
        copy(beta0)
    end
    for _ = 1:max_iter
        beta = nullspace === nothing ? coefs : nullspace * coefs
        _gar_weights!(w, beta, column_groups, group_weights, group_sizes, epsilon, normsq)
        coefs_new = Symmetric(XtX + penalty(w)) \ Xty
        # Relative ∞-norm change ON β (γ's ∞-norm is factorization-gauge-
        # dependent under Z); the eps floor only guards the all-zero case.
        beta_new = nullspace === nothing ? coefs_new : nullspace * coefs_new
        rel = mapreduce((a, b) -> abs(a - b), max, beta_new, beta) /
              max(maximum(abs, beta_new), eps(Float64))
        coefs = coefs_new
        rel < tol && break
    end
    return coefs
end

function solve_coefficients(est::GroupAdaptiveRidge, X::AbstractMatrix, y::AbstractVector;
                            row_groups = nothing,
                            nullspace::Union{Nothing,Matrix{Float64}} = nothing)::Vector{Float64}
    ncols_beta = nullspace === nothing ? size(X, 2) : size(nullspace, 1)
    length(est.column_groups) == ncols_beta || throw(DimensionMismatch(
        "GroupAdaptiveRidge column_groups length $(length(est.column_groups)) does not " *
        "match the (β-space) design column count $ncols_beta; the labels were likely " *
        "built on a different SLCEBasis."))
    nullspace === nothing || size(nullspace, 2) == size(X, 2) ||
        throw(DimensionMismatch("nullspace has $(size(nullspace, 2)) columns but " *
                                "the design has $(size(X, 2))"))
    # Exact zero only (see the AdaptiveRidge method above): at lambda = 0 the penalty
    # diagonal is gone and the Gram matrix may be singular — route to QR (min-norm) OLS.
    est.lambda == 0.0 && return solve_coefficients(OLS(), X, y)
    return _solve_gar(Matrix{Float64}(X' * X), Vector{Float64}(X' * y), est.lambda,
                      est.column_groups, est.group_weights, est.group_sizes,
                      est.epsilon, est.max_iter, est.tol; nullspace = nullspace)
end

function solve_coefficients(est::PrecomputedPilot, X::AbstractMatrix, y::AbstractVector;
                            row_groups = nothing, nullspace = nothing)::Vector{Float64}
    nullspace === nothing ||
        throw(ArgumentError("PrecomputedPilot carries a fixed β-space coefficient " *
                            "vector — an ASR-constrained (γ-space) solve is " *
                            "undefined for it. Fit with asr = false, or use a " *
                            "fresh estimator."))
    length(est.beta) == size(X, 2) || throw(DimensionMismatch(
        "PrecomputedPilot coefficient length $(length(est.beta)) does not match " *
        "design-matrix column count $(size(X, 2)); the pilot was likely fit on a " *
        "different SLCEBasis."))
    return copy(est.beta)
end
