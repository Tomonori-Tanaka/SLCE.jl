# Fitting and prediction: assemble the regression problem, run the estimator
# (fit/refit), and evaluate the fitted surface (predict_energy/predict_torque).

# Assemble the regression problem the estimator actually sees: the column-centered
# energy-only design (`w == 0`), or the whitened, stacked energy+torque design
# (`w > 0`), together with the centering constants `xbar`/`ybar` (for the analytic `j0`)
# and the per-row resampling-unit `groups`. Shared by `fit` and `refit` so the two build
# byte-identical designs.
function _assemble_problem(dataset::SLCEDataset, w::Float64)
    X_E = dataset.X_E
    y_E = dataset.y_E
    xbar = vec(mean(X_E; dims = 1))
    ybar = mean(y_E)
    if w > 0
        n_E = length(y_E)
        n_T = length(dataset.y_T)
        # A mixed dataset can be sliced down to a torque-free part (e.g. a CV training
        # fold that happens to hold no torque-bearing config); √(w/0) would silently
        # produce a degenerate objective, so fail loudly here.
        n_T > 0 || throw(ArgumentError(
            "torque_weight = $w > 0 but this dataset (or data split) has no torque " *
            "rows — stratify the split by torque presence or use torque_weight = 0"))
        # On a mixed dataset the torque-bearing MINORITY carries the whole weight
        # w (the objective is a pair of per-block means). That is the intended
        # semantics, but it is qualitatively different from the all-torque case, so
        # say so once instead of silently redefining what torque_weight weighs.
        nat3 = 3 * n_atoms(dataset.basis.crystal)
        if n_T < nat3 * n_E
            @warn "mixed dataset: torque rows cover only part of the configurations" *
                  " — torque_weight = $w concentrates on the torque-bearing subset" coverage =
                round(n_T / (nat3 * n_E); digits = 3) maxlog = 1
        end
        se = sqrt((1 - w) / n_E)
        sm = sqrt(w / n_T)
        X = vcat((X_E .- xbar') .* se, dataset.X_T .* sm)
        y = vcat((y_E .- ybar) .* se, dataset.y_T .* sm)
        # Resampling-unit labels for grouped cross-validation: a configuration's energy
        # row and all its torque-component rows share its index, so a CV-based
        # estimator never splits one configuration across folds. The torque-row labels
        # are READ from the dataset's stored per-row config index (`torque_config`) —
        # never re-derived from a uniform-block assumption, which a mixed (ragged)
        # torque design violates.
        groups = vcat(collect(1:n_E), dataset.torque_config)
    else
        X = X_E .- xbar'
        y = y_E .- ybar
        groups = nothing                            # each row is its own configuration
    end
    return X, y, xbar, ybar, groups
end

"""
    fit(SLCEFit, dataset, estimator; torque_weight = 0.0) -> SLCEFit

Fit the SCE coefficients. The energy design matrix is column-centered, so the
reference energy `j0` is recovered analytically as `mean(y_E − X_E·jϕ)`
(independent of the estimator), and the centered problem is handed to
`solve_coefficients`.

With `torque_weight = w ∈ (0, 1]` and a torque-carrying `dataset` (see
[`SLCEDataset`](@ref)), the fit minimizes the per-sample-normalized objective

    L = (1 − w)·MSE_energy + w·MSE_torque

by whitening: the centered energy block is row-scaled by `√((1−w)/n_E)` and the
torque block by `√(w/n_T)`, then stacked. `j0` does not enter the torque block, so
it is still recovered from the energy data alone. `w = 0` (the default) is a
pure-energy fit; requesting `w > 0` without torque data is an error.

A resampling estimator (a cross-validating [`ElasticNet`](@ref) / [`Lasso`](@ref))
receives per-row group labels so its folds are **grouped by configuration** — a
configuration's energy row and its torque-component rows are never split across the
train/holdout boundary, which would otherwise leak within-configuration structure
into the CV estimate and bias λ selection.
"""
function fit(::Type{SLCEFit}, dataset::SLCEDataset, estimator::AbstractEstimator;
             torque_weight::Real = 0.0)::SLCEFit
    isempty(dataset.y_E) && throw(ArgumentError("dataset has no observations"))
    w = Float64(torque_weight)
    (0.0 <= w <= 1.0) || throw(ArgumentError("torque_weight must be in [0, 1]; got $w"))
    if w > 0 && !has_torque(dataset)
        throw(ArgumentError("torque_weight = $w but the dataset has no torque data; " *
                            "build it with SLCEDataset(basis, configs, energies, torques)"))
    end
    X, y, xbar, ybar, groups = _assemble_problem(dataset, w)
    jphi = solve_coefficients(estimator, X, y; groups = groups)
    j0 = ybar - dot(xbar, jphi)
    residuals = dataset.y_E .- (j0 .+ dataset.X_E * jphi)
    return SLCEFit(dataset, j0, jphi, estimator, residuals, w)
end

"""
    refit(f::SLCEFit, estimator = OLS(); threshold = 0.0) -> SLCEFit

Re-fit on the **support** of an existing fit `f` — the classic de-biasing step that
follows a sparse fit: select the basis functions a regularized estimator kept, then
re-solve on just those columns (by default with [`OLS`](@ref)) to remove the shrinkage
bias on the survivors. The dataset and `torque_weight` of `f` are reused, so the design is
assembled exactly as in [`fit`](@ref).

A column `j` is in the support when its **scaled-magnitude contribution** exceeds
`threshold`:

    |coef(f)[j]| · ‖X[:, j]‖ > threshold

where `X` is the assembled (centered / whitened) design. `threshold = 0` (the default)
keeps the columns of nonzero scaled magnitude — the nonzero support of `f`, minus any
column the centering annihilates to a zero column. Coefficients off the support are set to
zero; `j0` is recovered analytically as in `fit`. If the support is empty, an all-zero `jϕ`
is returned (with a warning) and `j0` falls back to `mean(y_E)`.

A [`PrecomputedPilot`](@ref) (or an [`AdaptiveLasso`](@ref) whose pilot is one) is
rejected: its fixed coefficient vector has the original column count, not the refit
support length, and is meaningless once a support has been chosen.
"""
function refit(f::SLCEFit, estimator::AbstractEstimator = OLS();
               threshold::Real = 0.0)::SLCEFit
    threshold >= 0 || throw(ArgumentError("refit threshold must be ≥ 0; got $threshold"))
    _reject_precomputed_pilot(estimator)
    dataset = f.dataset
    w = f.torque_weight
    X, y, xbar, ybar, groups = _assemble_problem(dataset, w)
    jphi_in = f.jphi
    # Scaled-magnitude support on the assembled design: a column survives when its
    # contribution magnitude |jϕ_j|·‖X[:, j]‖ clears the threshold.
    support = findall(j -> abs(jphi_in[j]) * norm(@view X[:, j]) > threshold,
                      eachindex(jphi_in))
    jphi = zeros(Float64, length(jphi_in))
    if isempty(support)
        # Short-circuit before `solve_coefficients`: a GLMNet-backed estimator would error
        # on a zero-column design, and the all-zero `jϕ` is a well-defined (degenerate) fit
        # whose `j0` falls back to `mean(y_E)`.
        @warn "refit: empty support — every coefficient is at or below the " *
              "scaled-magnitude threshold. Returning an all-zero jϕ; " *
              "j0 falls back to mean(y_E)." threshold
    else
        jphi[support] .= solve_coefficients(estimator, view(X, :, support), y;
                                            groups = groups)
    end
    j0 = ybar - dot(xbar, jphi)
    residuals = dataset.y_E .- (j0 .+ dataset.X_E * jphi)
    return SLCEFit(dataset, j0, jphi, estimator, residuals, w)
end

# A `PrecomputedPilot` carries a fixed, full-design coefficient vector. `refit` and
# `cross_validate` must both reject it (with rationales of their own), so the predicate
# lives here as the single definition both rejection sites share.
_carries_precomputed_pilot(e::AbstractEstimator)::Bool =
    e isa PrecomputedPilot || (e isa AdaptiveLasso && e.pilot isa PrecomputedPilot)

# `refit` re-solves on a column sub-matrix `X[:, support]`, so a `PrecomputedPilot` —
# whose fixed coefficient vector carries the original full column count — would throw a
# `DimensionMismatch` deep in `solve_coefficients`, and is meaningless once a support has
# been chosen. Reject it (and an `AdaptiveLasso` carrying one) upfront with a clear message.
function _reject_precomputed_pilot(e::AbstractEstimator)
    if e isa PrecomputedPilot
        throw(ArgumentError("refit does not accept a PrecomputedPilot: its fixed " *
            "coefficient vector has the original column count, not the refit support " *
            "length. Pass a fresh estimator such as OLS()."))
    elseif e isa AdaptiveLasso && e.pilot isa PrecomputedPilot
        throw(ArgumentError("refit does not accept an AdaptiveLasso whose pilot is a " *
            "PrecomputedPilot: the pilot's fixed coefficient vector has the original " *
            "column count, not the refit support length. Use a fresh pilot such as OLS()."))
    end
    return nothing
end

"""
    SLCEModel(f::SLCEFit) -> SLCEModel

Extract the lightweight predictor from a fit.
"""
SLCEModel(f::SLCEFit) = SLCEModel(f.dataset.basis, f.j0, f.jphi, f.dataset.basis.salc_basis.keys)

"""
    predict_energy(model, data) -> Float64 or Vector{Float64}

Predict the energy of a spin configuration (`3 × n_atoms` matrix) or a vector of
configurations. The vector form is evaluated in parallel over `Threads.nthreads()`
threads (set `julia -t` / `JULIA_NUM_THREADS`); the result is thread-count-independent.
"""
function predict_energy(model::SLCEModel, config::AbstractMatrix{<:Real})::Float64
    _validate_config(config, n_atoms(model.basis.crystal))
    salcs = model.basis.salc_basis.salcs
    e = model.j0
    scratch = SALCScratch()                 # reused workspace (dnPl + harmonic tables)
    @inbounds for k in eachindex(model.jphi)
        e += model.jphi[k] * evaluate_salc(salcs[k], config, scratch)
    end
    return e
end
function predict_energy(model::SLCEModel, configs::AbstractVector)::Vector{Float64}
    nat = n_atoms(model.basis.crystal)
    for (i, c) in enumerate(configs)                   # serial: clean errors, not wrapped
        _validate_config(c, nat; label = "config $i")
    end
    out = Vector{Float64}(undef, length(configs))
    Threads.@threads for i in eachindex(configs, out)  # independent slots
        out[i] = predict_energy(model, configs[i])
    end
    return out
end
predict_energy(f::SLCEFit, data) = predict_energy(SLCEModel(f), data)

"""
    predict_torque(model, config) -> Matrix{Float64}

Predict the per-atom torque `τ_a = −e_a × ∂E/∂e_a` of a spin configuration
(`3 × n_atoms`), returned as a `3 × n_atoms` matrix. A vector of configurations
returns a vector of such matrices. This is the Landau–Lifshitz / physical torque
`m_a × B_eff,a` (the negative rotation-gradient of the energy), the analytic
derivative of the same surface [`predict_energy`](@ref) evaluates, so the two are
consistent by construction (`τ = −e × ∇E`). The vector form is evaluated in parallel
over `Threads.nthreads()` threads; the result is thread-count-independent.
"""
function predict_torque(model::SLCEModel, config::AbstractMatrix{<:Real})::Matrix{Float64}
    nat = n_atoms(model.basis.crystal)
    _validate_config(config, nat)
    salcs = model.basis.salc_basis.salcs
    G = zeros(Float64, 3, nat)
    scratch = SALCScratch()                 # reused workspace (dnPl + harmonic tables)
    @inbounds for k in eachindex(model.jphi)
        accumulate_grad!(G, salcs[k], config, model.jphi[k], scratch)
    end
    T = Matrix{Float64}(undef, 3, nat)
    @inbounds for a = 1:nat
        ea = SVector{3,Float64}(config[1, a], config[2, a], config[3, a])
        ga = SVector{3,Float64}(G[1, a], G[2, a], G[3, a])
        t = cross(ga, ea)                # τ = ∇E × e = −e × ∇E  (physical / LL torque)
        T[1, a] = t[1]
        T[2, a] = t[2]
        T[3, a] = t[3]
    end
    return T
end
function predict_torque(model::SLCEModel, configs::AbstractVector)::Vector{Matrix{Float64}}
    nat = n_atoms(model.basis.crystal)
    for (i, c) in enumerate(configs)                   # serial: clean errors, not wrapped
        _validate_config(c, nat; label = "config $i")
    end
    out = Vector{Matrix{Float64}}(undef, length(configs))
    Threads.@threads for i in eachindex(configs, out)  # independent slots
        out[i] = predict_torque(model, configs[i])
    end
    return out
end
predict_torque(f::SLCEFit, data) = predict_torque(SLCEModel(f), data)
