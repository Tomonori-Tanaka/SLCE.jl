module SLCEGLMNetExt

# Provides `solve_coefficients(::ElasticNet, …)` and `solve_coefficients(::AdaptiveLasso, …)`
# when GLMNet is loaded. The `ElasticNet` / `Lasso` / `AdaptiveLasso` types live in the
# core package (so they can be named and dispatched on without this dependency); only the
# actual GLMNet solve lights up here, mirroring the Spglib and Sunny extensions.
#
# Contract (see `AbstractEstimator`): `X` is already column-centered and `y` centered, so
# we fit with `intercept = false` — GLMNet adds no intercept and `j0` is recovered
# analytically downstream in `fit`. GLMNet minimizes
#   (1/2n)·‖y − Xβ‖² + λ·[(1−α)/2·‖β‖₂² + α·‖β‖₁]
# and returns β on the original (un-standardized) scale.

using SLCE: ElasticNet, AdaptiveLasso
import SLCE: solve_coefficients
using GLMNet: glmnet, glmnetcv

function solve_coefficients(est::ElasticNet, X::AbstractMatrix, y::AbstractVector;
                            groups = nothing)::Vector{Float64}
    Xf = Matrix{Float64}(X)
    yf = Vector{Float64}(y)
    return _glmnet_solve(Xf, yf, est.alpha, est.lambda, est.standardize, est.nfolds,
                         est.select, est.seed, est.nlambda, groups, nothing)
end

function solve_coefficients(est::AdaptiveLasso, X::AbstractMatrix, y::AbstractVector;
                            groups = nothing)::Vector{Float64}
    Xf = Matrix{Float64}(X)
    yf = Vector{Float64}(y)
    # First stage: the pilot fit (any estimator) supplies the adaptive weights. Its own
    # solve honors `groups`, so e.g. an ElasticNet pilot CV-groups by configuration too.
    beta_pilot = solve_coefficients(est.pilot, Xf, yf; groups = groups)
    # Per-column penalty factor wⱼ = 1 / max(|β̂_pilotⱼ|, ε)^γ (Zou 2006): GLMNet rescales
    # the vector to sum to nvars internally. The weights are independent of λ, so they are
    # held fixed across the whole (fixed-λ or CV) path below.
    pf = inv.(max.(abs.(beta_pilot), est.epsilon) .^ est.gamma)
    return _glmnet_solve(Xf, yf, 1.0, est.lambda, est.standardize, est.nfolds,
                         est.select, est.seed, est.nlambda, groups, pf)
end

# Shared GLMNet plumbing for ElasticNet and AdaptiveLasso. `pf` is the optional
# per-column penalty factor (AdaptiveLasso's adaptive weights), forwarded to GLMNet only
# when given so the ElasticNet call stays identical to its pre-AdaptiveLasso form.
# `lambda === nothing` selects λ by configuration-grouped CV; a number fits at exactly
# that penalty (GLMNet warm-starts a path, so a one-element λ vector is a valid request).
function _glmnet_solve(Xf::Matrix{Float64}, yf::Vector{Float64}, alpha::Float64,
                       lambda::Union{Nothing,Float64}, standardize::Bool, nfolds::Int,
                       select::Symbol, seed::Int, nlambda::Int, groups,
                       pf::Union{Nothing,Vector{Float64}})::Vector{Float64}
    if lambda === nothing
        return _solve_cv(Xf, yf, alpha, standardize, nfolds, select, seed, nlambda,
                         groups, pf)
    end
    path = pf === nothing ?
        glmnet(Xf, yf; alpha = alpha, lambda = [lambda], standardize = standardize,
               intercept = false) :
        glmnet(Xf, yf; alpha = alpha, lambda = [lambda], standardize = standardize,
               intercept = false, penalty_factor = pf)
    return collect(Float64, path.betas[:, 1])
end

# Cross-validated penalty selection over GLMNet's automatic λ path. Folds are over
# resampling units — whole `groups` if given (so a co-fit configuration's energy and
# torque rows stay together), otherwise individual rows.
function _solve_cv(Xf::Matrix{Float64}, yf::Vector{Float64}, alpha::Float64,
                   standardize::Bool, nfolds::Int, select::Symbol, seed::Int,
                   nlambda::Int, groups,
                   pf::Union{Nothing,Vector{Float64}})::Vector{Float64}
    n = length(yf)
    units = groups === nothing ? collect(1:n) : collect(groups)
    nunits = length(unique(units))
    # GLMNet's own default cap; each fold needs a few resampling units to fit.
    nf = min(nfolds, div(nunits, 3))
    nf >= 2 || error("cross-validation needs at least 6 resampling units for ≥ 2 " *
                     "folds; got $nunits. Pass an explicit `lambda` to skip CV.")
    folds = _make_folds(units, nf, seed)
    cv = pf === nothing ?
        glmnetcv(Xf, yf; alpha = alpha, folds = folds, standardize = standardize,
                 intercept = false, nlambda = nlambda) :
        glmnetcv(Xf, yf; alpha = alpha, folds = folds, standardize = standardize,
                 intercept = false, nlambda = nlambda, penalty_factor = pf)
    idx = _select_lambda(cv, select)
    return collect(Float64, cv.path.betas[:, idx])
end

# Deterministic, balanced, seed-controlled fold assignment (no RNG dependency): rank
# the distinct resampling units by a seeded hash, deal ranks round-robin into `nf`
# folds, then map each row to its unit's fold. Rows sharing a unit label never split
# across the train/holdout boundary. Same seed ⇒ identical folds ⇒ reproducible CV
# (within a Julia session/version, since `hash` is version-dependent).
function _make_folds(units::AbstractVector, nf::Int, seed::Int)::Vector{Int}
    uniq = unique(units)
    order = sortperm([hash((seed, u)) for u in uniq])
    foldof = Dict{eltype(uniq),Int}()
    @inbounds for rank in eachindex(order)
        foldof[uniq[order[rank]]] = mod1(rank, nf)
    end
    return [foldof[u] for u in units]
end

# Index into the descending CV λ path. `:lambda_min` is the lowest-CV-error λ;
# `:lambda_1se` is the most-regularized λ (largest λ ⇒ smallest index ⇒ sparsest)
# whose mean CV loss is still within one standard error of that minimum.
function _select_lambda(cv, select::Symbol)::Int
    imin = argmin(cv.meanloss)
    select === :lambda_min && return imin
    thresh = cv.meanloss[imin] + cv.stdloss[imin]
    for i = 1:imin
        cv.meanloss[i] <= thresh && return i
    end
    return imin
end

end # module SLCEGLMNetExt
