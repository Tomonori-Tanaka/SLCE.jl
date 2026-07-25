using Test
using SLCE
using LinearAlgebra
using Random

struct _DummyEstimator <: SLCE.AbstractEstimator end

@testset "fit / predict" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    spec = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = true)
    basis = SLCEBasis(crystal, spec)                 # NoSymmetry backend (no Spglib needed)
    m = length(basis.salc_basis)
    @test m > 0

    @testset "BasisSpec / Ridge constructor validation" begin
        @test_throws ArgumentError BasisSpec(; nbody = 0, cutoff = 1.5, lmax = [2])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = -1.0, lmax = [2])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = 1.5, lmax = Int[])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [-1])
        @test_throws ArgumentError Ridge(; lambda = -1.0)
        @test_throws ArgumentError Ridge(Inf)
        @test Ridge(; lambda = 0.0).lambda == 0.0
        @test basis.spec === spec                   # the spec rides along on the basis
        @test salcs(basis) === basis.salc_basis.salcs   # the public accessor
        @test length(salcs(basis)) == m
    end

    @testset "SLCEModel(basis, j0, jphi) fills keys from the basis" begin
        jphi3 = randn(MersenneTwister(7), m)
        p = SLCEModel(basis, 0.25, jphi3)
        @test p.keys == basis.salc_basis.keys
        @test p.j0 == 0.25 && p.jphi == jphi3
        @test_throws DimensionMismatch SLCEModel(basis, 0.0, zeros(m + 1))
    end

    rng = MersenneTwister(1)
    configs = [randcfg(rng, 2) for _ = 1:40]

    # synthetic energies that lie exactly in the SALC span
    ds0 = SLCEDataset(basis, configs, zeros(length(configs)))
    true_jphi = randn(rng, m)
    true_j0 = 0.7
    y = true_j0 .+ ds0.X_E * true_jphi
    ds = SLCEDataset(basis, configs, y)

    @testset "OLS recovers an in-span target" begin
        f = fit(SLCEFit, ds, OLS())
        @test r2_energy(f) ≈ 1.0 atol = 1e-9
        @test rmse_energy(f) < 1e-6
        @test isapprox(predict_energy(f, configs), y; atol = 1e-7)
        @test nobs(f) == 40
        @test length(coef(f)) == m
    end

    @testset "Ridge(0) ≈ OLS; Ridge(λ) shrinks" begin
        f = fit(SLCEFit, ds, OLS())
        f0 = fit(SLCEFit, ds, Ridge(lambda = 0.0))
        @test isapprox(coef(f0), coef(f); atol = 1e-6)
        fλ = fit(SLCEFit, ds, Ridge(lambda = 10.0))
        @test norm(coef(fλ)) <= norm(coef(f)) + 1e-9
    end

    @testset "unknown estimator → friendly error" begin
        @test_throws ErrorException solve_coefficients(_DummyEstimator(), ds.X_E, ds.y_E)
    end

    # The GLMNet-backed estimators are constructed and validated in the core (the
    # type is dependency-free); the actual solve needs `using GLMNet` and is exercised
    # in test/glmnet/. Without the extension loaded, a friendly error points there.
    @testset "ElasticNet / Lasso: construction, validation, deferred backend" begin
        @test Lasso() === ElasticNet(; alpha = 1.0)
        @test Lasso(lambda = 0.1).alpha == 1.0
        en = ElasticNet(; alpha = 0.4, lambda = 0.2, select = :lambda_1se, nfolds = 5, seed = 9)
        @test en.alpha == 0.4 && en.lambda == 0.2 && en.select === :lambda_1se
        @test ElasticNet().lambda === nothing                       # default: choose by CV
        @test !SLCE.islinear(Lasso())                     # not a closed-form estimator

        @test_throws ArgumentError ElasticNet(; alpha = 1.5)
        @test_throws ArgumentError ElasticNet(; alpha = -0.1)
        @test_throws ArgumentError ElasticNet(; lambda = -1.0)
        @test_throws ArgumentError ElasticNet(; select = :bogus)
        @test_throws ArgumentError ElasticNet(; nfolds = 1)

        # GLMNet is not loaded in this environment ⇒ the fallback fires.
        @test_throws ErrorException solve_coefficients(Lasso(), ds.X_E, ds.y_E)
    end
end

@testset "adaptive estimators" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    # isotropy = false ⇒ a wide (44-column) basis, so concentration / selection is meaningful
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = false)
    basis = SLCEBasis(crystal, interaction)
    m = length(basis.salc_basis)
    @test m > 10

    rng = MersenneTwister(11)
    configs = [randcfg(rng, 2) for _ = 1:80]
    ds0 = SLCEDataset(basis, configs, zeros(length(configs)))

    @testset "AdaptiveRidge: validation, islinear, lambda = 0 ⇒ OLS" begin
        @test SLCE.islinear(AdaptiveRidge(lambda = 1e-3))   # linear smoother (GCV-eligible)
        ar = AdaptiveRidge(lambda = 1e-3, epsilon = 1e-6, max_iter = 100, tol = 1e-8)
        @test ar.lambda == 1e-3 && ar.epsilon == 1e-6 && ar.max_iter == 100 && ar.tol == 1e-8
        @test_throws ArgumentError AdaptiveRidge(lambda = -1.0)
        @test_throws ArgumentError AdaptiveRidge(lambda = 1.0, epsilon = 0.0)
        @test_throws ArgumentError AdaptiveRidge(lambda = 1.0, max_iter = 0)
        @test_throws ArgumentError AdaptiveRidge(lambda = 1.0, tol = 0.0)

        dense = randn(rng, m)
        y = 0.5 .+ ds0.X_E * dense
        ds = SLCEDataset(basis, configs, y)
        # The penalty vanishes at lambda = 0, so the reweighting is a no-op ⇒ plain OLS.
        f0 = fit(SLCEFit, ds, AdaptiveRidge(lambda = 0.0))
        fols = fit(SLCEFit, ds, OLS())
        @test isapprox(coef(f0), coef(fols); atol = 1e-7)
        @test isapprox(intercept(f0), intercept(fols); atol = 1e-9)
        # A tiny penalty still recovers an in-span target essentially exactly.
        ftiny = fit(SLCEFit, ds, AdaptiveRidge(lambda = 1e-10))
        @test r2_energy(ftiny) > 1 - 1e-6
    end

    @testset "AdaptiveRidge approximates L0 selection on a synthetic centered design" begin
        # A controlled sparse signal (no SCE column correlations to confound the test):
        # the reweighting must drive the genuinely-zero coordinates far below the active ones.
        rng2 = MersenneTwister(2024)
        n, p = 80, 10
        Xs = randn(rng2, n, p)
        btrue = zeros(p); btrue[3] = 2.0; btrue[7] = -1.5
        ys = Xs * btrue + 0.01 * randn(rng2, n)
        Xc = Xs .- transpose(vec(sum(Xs; dims = 1) ./ n))   # column-centered, the solver's contract
        yc = ys .- sum(ys) / n
        b = solve_coefficients(AdaptiveRidge(lambda = 1e-2), Xc, yc)
        active = abs.(b[[3, 7]])
        inactive = abs.(b[setdiff(1:p, [3, 7])])
        @test minimum(active) > 50 * maximum(inactive)      # zeros driven far down
        @test isapprox(b[3], 2.0; atol = 0.05) && isapprox(b[7], -1.5; atol = 0.05)
        # max_iter = 1 stops after one reweight: no more concentrated than the converged fit.
        b1 = solve_coefficients(AdaptiveRidge(lambda = 1e-2, max_iter = 1), Xc, yc)
        @test maximum(abs.(b1[setdiff(1:p, [3, 7])])) >= maximum(inactive) - 1e-12
    end

    @testset "PrecomputedPilot: returns fixed coefficients, flows through fit" begin
        dense = randn(rng, m)
        y = 0.3 .+ ds0.X_E * dense
        ds = SLCEDataset(basis, configs, y)
        beta = randn(rng, m)
        @test solve_coefficients(PrecomputedPilot(beta), ds.X_E, ds.y_E) == beta
        @test_throws DimensionMismatch solve_coefficients(PrecomputedPilot([1.0]), ds.X_E, ds.y_E)
        # The stored vector is copied at construction, so caller mutation does not leak in.
        src = copy(beta); pp = PrecomputedPilot(src); src[1] += 100.0
        @test solve_coefficients(pp, ds.X_E, ds.y_E)[1] == beta[1]
        # Used as the estimator itself: jphi is the stored vector, j0 recovered analytically.
        f = fit(SLCEFit, ds, PrecomputedPilot(beta))
        @test coef(f) == beta
        @test sprint(show, PrecomputedPilot(beta)) == "PrecomputedPilot($m coefficients)"
    end

    @testset "AdaptiveLasso: construction, validation, deferred backend" begin
        al = AdaptiveLasso()
        @test al.pilot isa OLS && al.lambda === nothing && al.gamma == 1.0 && al.standardize
        @test AdaptiveLasso(pilot = Ridge(lambda = 1e-4), gamma = 0.5, lambda = 1e-3).gamma == 0.5
        @test !SLCE.islinear(AdaptiveLasso())             # pilot + L1 path: not a smoother
        @test_throws ArgumentError AdaptiveLasso(gamma = -1.0)
        @test_throws ArgumentError AdaptiveLasso(epsilon = 0.0)
        @test_throws ArgumentError AdaptiveLasso(select = :bogus)
        @test_throws ArgumentError AdaptiveLasso(lambda = -1.0)
        @test_throws ArgumentError AdaptiveLasso(nfolds = 1)
        @test occursin("AdaptiveLasso(pilot=", sprint(show, AdaptiveLasso()))
        # GLMNet absent in this environment ⇒ the deferred solve points at the extension.
        @test_throws ErrorException solve_coefficients(AdaptiveLasso(lambda = 1e-3), ds0.X_E, ds0.y_E)
    end
end

@testset "refit and accessors" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = false)
    basis = SLCEBasis(crystal, interaction)
    m = length(basis.salc_basis)
    @test m > 10

    rng = MersenneTwister(11)
    configs = [randcfg(rng, 2) for _ = 1:80]
    ds0 = SLCEDataset(basis, configs, zeros(length(configs)))

    # A sparse in-span signal on two columns + noise (the regime selection is for).
    ktrue = [3, 7]
    beta = zeros(m); beta[ktrue] .= [1.0, -0.7]
    y = 0.4 .+ ds0.X_E * beta .+ 0.005 .* randn(rng, length(configs))
    ds = SLCEDataset(basis, configs, y)
    truem = SLCEModel(basis, 0.4, beta, basis.salc_basis.keys)
    dst = SLCEDataset(basis, configs, 0.4 .+ ds0.X_E * beta, predict_torque(truem, configs))

    @testset "energy accessors: dof, residuals, rss, and metric consistency" begin
        f = fit(SLCEFit, ds, OLS())
        @test dof(f) == m + 1                                   # coefficients + intercept
        @test residuals_energy(f) === f.residuals               # the stored energy residuals
        @test length(residuals_energy(f)) == nobs(f)
        # from-scratch reference (y − prediction), not the accessors' own definitions —
        # a typo'd rss/rmse implementation must fail here, not be re-executed verbatim
        r_ref = y .- predict_energy(f, configs)
        @test residuals_energy(f) ≈ r_ref atol = 1e-9
        @test rss_energy(f) ≈ sum(abs2, r_ref) atol = 1e-9
        @test rmse_energy(f) ≈ sqrt(sum(abs2, r_ref) / length(y)) atol = 1e-9
        ss_tot = sum(abs2, y .- sum(y) / length(y))
        @test r2_energy(f) ≈ 1 - sum(abs2, r_ref) / ss_tot atol = 1e-9
        # torque accessors require a torque-carrying dataset
        @test_throws ArgumentError residuals_torque(f)
        @test_throws ArgumentError rss_torque(f)
        @test_throws ArgumentError r2_torque(f)
    end

    @testset "torque accessors on a co-fit are mutually consistent" begin
        fco = fit(SLCEFit, dst, OLS(); torque_weight = 0.3)
        @test length(residuals_torque(fco)) == length(dst.y_T)
        # from-scratch reference: the flattened torque prediction of the fitted model
        t_ref = dst.y_T .- reduce(vcat, vec.(predict_torque(SLCEModel(fco), configs)))
        @test residuals_torque(fco) ≈ t_ref atol = 1e-9
        @test rss_torque(fco) ≈ sum(abs2, t_ref) atol = 1e-9
        @test rmse_torque(fco) ≈ sqrt(sum(abs2, t_ref) / length(dst.y_T)) atol = 1e-9
        @test r2_torque(fco) ≈ 1 - sum(abs2, t_ref) / sum(abs2, dst.y_T)  # uncentered baseline
    end

    @testset "StatsAPI generics default to the energy block" begin
        f = fit(SLCEFit, ds, OLS())
        @test residuals(f) === residuals_energy(f)
        @test r2(f) == r2_energy(f)
        @test predict(f, configs) ≈ predict_energy(f, configs)
        model = SLCEModel(f)
        @test predict(model, configs) ≈ predict_energy(model, configs)
        @test predict(model, configs[1]) ≈ predict_energy(model, configs[1])
        # islinear (StatsAPI): closed-form estimators vs penalty-path / unknown ones
        @test SLCE.islinear(OLS()) && SLCE.islinear(Ridge())
        @test SLCE.islinear(AdaptiveRidge(lambda = 1e-3))
        @test !SLCE.islinear(Lasso()) && !SLCE.islinear(ElasticNet())
        @test !SLCE.islinear(_DummyEstimator())
    end

    @testset "refit(threshold = 0) on a dense OLS fit is (near) identity" begin
        f = fit(SLCEFit, ds, OLS())
        fr = refit(f)                                           # default OLS, keep nonzero support
        @test isapprox(coef(fr), coef(f); atol = 1e-7)
        @test isapprox(intercept(fr), intercept(f); atol = 1e-9)
    end

    @testset "refit de-biases a selected support" begin
        # AdaptiveRidge concentrates onto the true columns; a scaled-magnitude threshold
        # then keeps the dominant ones and OLS re-solves on just those.
        fa = fit(SLCEFit, ds, AdaptiveRidge(lambda = 1e-2))
        Xc = ds.X_E .- sum(ds.X_E; dims = 1) ./ size(ds.X_E, 1)
        contrib = [abs(coef(fa)[j]) * norm(@view Xc[:, j]) for j = 1:m]
        fr = refit(fa; threshold = 0.05 * maximum(contrib))
        support = findall(!iszero, coef(fr))
        @test issubset(ktrue, support)                          # the true columns survive
        @test length(support) < m                               # and it is genuinely sparse
        @test all(iszero, coef(fr)[setdiff(1:m, support)])      # off-support is exactly zero
        @test r2_energy(fr) > 0.95
    end

    @testset "refit edge cases: empty support, validation, pilot rejection" begin
        f = fit(SLCEFit, ds, OLS())
        # An unreachable threshold drops every column ⇒ all-zero jϕ, j0 = mean(y_E).
        fr = @test_logs (:warn,) refit(f; threshold = 1e12)
        @test all(iszero, coef(fr))
        @test isapprox(intercept(fr), sum(y) / length(y); atol = 1e-9)
        @test_throws ArgumentError refit(f; threshold = -1.0)
        # A PrecomputedPilot (its fixed vector has the full column count) is rejected, as is
        # an AdaptiveLasso carrying one.
        @test_throws ArgumentError refit(f, PrecomputedPilot(coef(f)))
        @test_throws ArgumentError refit(f, AdaptiveLasso(pilot = PrecomputedPilot(coef(f)),
                                                          lambda = 1e-3))
    end
end
