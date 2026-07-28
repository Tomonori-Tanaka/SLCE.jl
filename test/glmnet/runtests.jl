# GLMNet-export validation: fit SLCE coefficients with the Lasso / ElasticNet
# estimators and confirm the penalized solver honors the column-centered contract,
# recovers a sparse signal, and selects the penalty by cross-validation. Runs in a
# separate environment (heavy GLMNet / Fortran binary dependency), mirroring
# `test/sunny/` and `test/oracle/`. The `ElasticNet` / `Lasso` types and their
# argument validation are covered Estimator-free in `test/unit/test_fit.jl`; here we
# exercise the actual GLMNet solve through the `SLCEGLMNetExt` extension.

using Test
using Random
using LinearAlgebra
import SLCE
import GLMNet   # triggers SLCEGLMNetExt

const MR = SLCE

_rdir(rng) = (v = randn(rng, 3); v / norm(v))
randcfg(rng, nat) = reduce(hcat, (_rdir(rng) for _ = 1:nat))
avg(x) = sum(x) / length(x)

@testset "GLMNet estimators" begin
    lat = MR.Lattice(Matrix(3.0 * I(3)))
    crystal = MR.Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    inter = MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = true)
    basis = MR.SLCEBasis(crystal, inter)          # NoSymmetry ⇒ a wide (44-column) basis
    m = MR.n_salcs(basis)
    @test m > 10
    keys = basis.salc_basis.keys
    kiso = findfirst(k -> sort(SLCE.spin_ls(k)) == [1, 1] && k.Lf == 0, keys)
    @test kiso !== nothing                       # the isotropic (Heisenberg) channel

    rng = MersenneTwister(7)
    configs = [randcfg(rng, 2) for _ = 1:80]

    # A dense in-span target (no noise) and the design matrix it lives in.
    ds0 = MR.SLCEDataset(basis, configs, zeros(length(configs)))
    βdense = randn(rng, m)
    ydense = 0.4 .+ ds0.X_E * βdense

    @testset "tiny λ ≈ OLS, and the analytic-j0 contract holds" begin
        ds = MR.SLCEDataset(basis, configs, ydense)
        fols = MR.fit(MR.SLCEFit, ds, MR.OLS())
        fl = MR.fit(MR.SLCEFit, ds, MR.Lasso(lambda = 1e-6))
        @test isapprox(MR.predict_energy(fols, configs), ydense; atol = 1e-8)   # OLS interpolates
        @test isapprox(MR.predict_energy(fl, configs), ydense; rtol = 2e-2)     # near-unpenalized
        # j0 is recovered analytically downstream regardless of the estimator, so the
        # in-sample mean prediction always equals the mean target.
        @test avg(MR.predict_energy(fl, configs)) ≈ avg(ydense)
    end

    # A sparse signal (only the isotropic column) plus noise: the regime the Lasso is for.
    ynoisy = 0.4 .+ 1.0 .* ds0.X_E[:, kiso] .+ 0.01 .* randn(rng, length(configs))
    dssparse = MR.SLCEDataset(basis, configs, ynoisy)

    @testset "CV Lasso recovers the support and is sparse" begin
        f = MR.fit(MR.SLCEFit, dssparse, MR.Lasso())              # CV, :lambda_min
        @test argmax(abs.(MR.coef(f))) == kiso                    # signal column dominates
        @test MR.r2_energy(f) > 0.95                              # and still fits
    end

    @testset ":lambda_1se shrinks more than :lambda_min and is sparse" begin
        fmin = MR.fit(MR.SLCEFit, dssparse, MR.Lasso(select = :lambda_min))
        f1se = MR.fit(MR.SLCEFit, dssparse, MR.Lasso(select = :lambda_1se))
        # λ_1se ≥ λ_min ⇒ more shrinkage; ‖β‖₁ is monotone non-increasing in λ.
        @test norm(MR.coef(f1se), 1) <= norm(MR.coef(fmin), 1) + 1e-8
        # With 44 columns and a 1-sparse signal, the parsimonious 1-SE model is sparse.
        @test count(iszero, MR.coef(f1se)) >= 1
    end

    @testset "seeded CV is reproducible" begin
        a = MR.coef(MR.fit(MR.SLCEFit, dssparse, MR.Lasso(seed = 3)))
        b = MR.coef(MR.fit(MR.SLCEFit, dssparse, MR.Lasso(seed = 3)))
        @test a == b
    end

    @testset "ElasticNet (α<1) differs from the Lasso and is less sparse" begin
        λ = 0.02
        fl = MR.fit(MR.SLCEFit, dssparse, MR.Lasso(lambda = λ))
        fe = MR.fit(MR.SLCEFit, dssparse, MR.ElasticNet(alpha = 0.3, lambda = λ))
        @test !isapprox(MR.coef(fe), MR.coef(fl); atol = 1e-8)
        @test count(iszero, MR.coef(fe)) <= count(iszero, MR.coef(fl))
    end

    @testset "energy+torque co-fit flows through GLMNet" begin
        truem = MR.SLCEModel(basis, 0.4, βdense, keys)
        torqs = MR.predict_torque(truem, configs)
        dst = MR.SLCEDataset(basis, configs, ydense, torqs)
        # explicit λ: both observables recovered
        f = MR.fit(MR.SLCEFit, dst, MR.Lasso(lambda = 1e-5); torque_weight = 0.3)
        @test MR.r2_energy(f) > 0.9
        @test MR.r2_torque(f) > 0.9
        # CV path: exercises configuration-grouped folds (energy + torque rows of one
        # config stay together) and still fits both observables.
        fcv = MR.fit(MR.SLCEFit, dst, MR.Lasso(); torque_weight = 0.3)
        @test MR.r2_energy(fcv) > 0.9
        @test MR.r2_torque(fcv) > 0.9
    end

    @testset "AdaptiveLasso recovers the support and is sparse" begin
        # Fixed λ, OLS pilot: the weighted L1 should select the signal column and fit it.
        fa = MR.fit(MR.SLCEFit, dssparse, MR.AdaptiveLasso(lambda = 1e-3))
        @test argmax(abs.(MR.coef(fa))) == kiso
        @test MR.r2_energy(fa) > 0.95
        @test count(iszero, MR.coef(fa)) >= 1                     # genuinely sparse
        # CV-selected λ (the default) likewise selects the signal column.
        facv = MR.fit(MR.SLCEFit, dssparse, MR.AdaptiveLasso())
        @test argmax(abs.(MR.coef(facv))) == kiso
        @test MR.r2_energy(facv) > 0.95
    end

    @testset "gamma = 0 reduces to plain Lasso" begin
        # With γ = 0 the weights are uniform (≡ 1 after GLMNet's sum-to-nvars rescale),
        # so an OLS-pilot AdaptiveLasso is byte-for-byte a plain Lasso at the same λ.
        λ = 0.02
        fa0 = MR.fit(MR.SLCEFit, dssparse, MR.AdaptiveLasso(lambda = λ, gamma = 0.0))
        fl = MR.fit(MR.SLCEFit, dssparse, MR.Lasso(lambda = λ))
        @test isapprox(MR.coef(fa0), MR.coef(fl); atol = 1e-10)
    end

    @testset "the adaptive reweighting changes the fit, and the pilot is pluggable" begin
        λ = 0.02
        fl = MR.fit(MR.SLCEFit, dssparse, MR.Lasso(lambda = λ))
        fa = MR.fit(MR.SLCEFit, dssparse, MR.AdaptiveLasso(lambda = λ, gamma = 1.0))
        @test !isapprox(MR.coef(fa), MR.coef(fl); atol = 1e-8)    # γ > 0 reweights
        # A Ridge pilot (recommended for rank-deficient designs) also fits.
        far = MR.fit(MR.SLCEFit, dssparse, MR.AdaptiveLasso(pilot = MR.Ridge(lambda = 1e-4),
                                                           lambda = 1e-3))
        @test MR.r2_energy(far) > 0.95
        # PrecomputedPilot reuses a prior fit's coefficients instead of a fresh pilot run;
        # with the OLS pilot's own coefficients it matches the OLS-pilot AdaptiveLasso.
        prior = MR.fit(MR.SLCEFit, dssparse, MR.OLS())
        fpp = MR.fit(MR.SLCEFit, dssparse,
                     MR.AdaptiveLasso(pilot = MR.PrecomputedPilot(MR.coef(prior)), lambda = 1e-3))
        fos = MR.fit(MR.SLCEFit, dssparse, MR.AdaptiveLasso(pilot = MR.OLS(), lambda = 1e-3))
        @test isapprox(MR.coef(fpp), MR.coef(fos); atol = 1e-8)
    end

    @testset "AdaptiveLasso energy+torque co-fit flows through grouped CV" begin
        truem = MR.SLCEModel(basis, 0.4, βdense, keys)
        torqs = MR.predict_torque(truem, configs)
        dst = MR.SLCEDataset(basis, configs, ydense, torqs)
        fco = MR.fit(MR.SLCEFit, dst, MR.AdaptiveLasso(); torque_weight = 0.3)
        @test MR.r2_energy(fco) > 0.9
        @test MR.r2_torque(fco) > 0.9
    end
end
