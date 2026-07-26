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

Degrees of freedom consumed by the fit: `length(coef(f)) + 1` — one per SCE coefficient
plus the intercept `j0`. This is the full parametric count, not an effective / selected
count, even for a sparse estimator.
"""
dof(f::SLCEFit)::Int = length(f.jphi) + 1

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
