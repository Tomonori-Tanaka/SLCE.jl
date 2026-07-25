using Test
using SLCE
using LinearAlgebra
using Random

# The design-matrix assembly (`_design_energy` / `_design_torque`) and the vector
# `predict_energy` / `predict_torque` forms are multithreaded over independent
# columns / slots. These checks pin the threaded output to a race-free serial
# reference (and to the independent scalar predict path): they hold at any thread
# count and would flag a data race when run with `julia -t N>1`.

@testset "threaded assembly / prediction equals serial" begin
    @info "running with $(Threads.nthreads()) thread(s)"
    Threads.nthreads() == 1 &&
        @warn "threading tests run serial; launch `julia -t N>1` to exercise the parallel path"

    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = true)
    basis = SLCEBasis(crystal, interaction)
    salcs = basis.salc_basis.salcs
    m = length(salcs)
    nat = 2

    rng = MersenneTwister(7)
    configs = [randcfg(rng, nat) for _ = 1:64]
    cfgs = [Matrix{Float64}(c) for c in configs]

    @testset "_design_energy: threaded == serial double loop" begin
        X = SLCE._design_energy(basis, cfgs)             # threaded
        Xref = Matrix{Float64}(undef, length(cfgs), m)
        for j = 1:m, i in eachindex(cfgs)                      # race-free serial
            Xref[i, j] = SLCE.evaluate_salc(salcs[j], cfgs[i])
        end
        @test X == Xref                                        # exact: same kernel, deterministic
        @test SLCE._design_energy(basis, cfgs) == X      # idempotent across calls
    end

    # Fit a model so the columns carry real coefficients for the cross-checks.
    y = 0.7 .+ SLCE._design_energy(basis, cfgs) * randn(MersenneTwister(3), m)
    ds = SLCEDataset(basis, configs, y)
    model = SLCEModel(fit(SLCEFit, ds, OLS()))
    jphi = coef(model)

    @testset "_design_torque: threaded X_T·jϕ == scalar predict_torque" begin
        X_T = SLCE._design_torque(basis, cfgs)           # threaded
        @test X_T == SLCE._design_torque(basis, cfgs)    # idempotent
        # Independent path: assemble the full-model torque from the scalar kernel,
        # flattened config-major / atom-major / xyz, and compare to X_T·jϕ.
        block = 3 * nat
        ref = Vector{Float64}(undef, length(cfgs) * block)
        for ci in eachindex(cfgs)
            τ = predict_torque(model, cfgs[ci])                # scalar (all SALCs)
            rb = block * (ci - 1)
            for a = 1:nat, d = 1:3
                ref[rb + 3 * (a - 1) + d] = τ[d, a]
            end
        end
        @test isapprox(X_T * jphi, ref; atol = 1e-12, rtol = 0)
    end

    @testset "vector predict_* (threaded) == scalar map (serial)" begin
        @test predict_energy(model, configs) == [predict_energy(model, c) for c in configs]
        @test predict_torque(model, configs) == [predict_torque(model, c) for c in configs]
    end
end
