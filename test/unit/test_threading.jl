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
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = false)
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

    # The joint (displacement-decorated) builders are a SECOND set of threaded loops
    # with a heavier per-task state — a `SALCScratch` plus the two gradient buffers
    # `Ge`/`Gu` — and they are not reached by the pure-spin checks above. Hoisting any
    # of those three out of the loop body would be a silent cross-column race, so each
    # joint design gets the same race-free serial reference.
    @testset "joint designs (disp-decorated) == serial reference" begin
        latj = Lattice(Matrix(3.0 * I(3)))
        crj = Crystal(latj, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        bj = SLCEBasis(crj, BasisSpec(crj; lmax = 1, pmax = 1, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        sj = bj.salc_basis.salcs
        mj = length(sj)
        natj = n_atoms(crj)
        rngj = MersenneTwister(19)
        cj = [randcfg(rngj, natj) for _ = 1:24]
        uj = [0.03 .* randn(rngj, 3, natj) for _ = 1:24]
        cols = SLCE._disp_active_cols(bj)
        atoms = findall(SLCE._disp_referenced_atoms(bj))
        @test !isempty(cols) && length(cols) < mj      # both column kinds present

        XE = SLCE._design_energy(bj, cj, uj)                      # threaded
        XEref = Matrix{Float64}(undef, length(cj), mj)
        scr = SLCE.SALCScratch()
        for j = 1:mj, i in eachindex(cj)                          # race-free serial
            XEref[i, j] = SLCE.evaluate_salc(sj[j], cj[i], uj[i], scr)
        end
        @test XE == XEref
        @test SLCE._design_energy(bj, cj, uj) == XE               # idempotent

        XF = SLCE._design_force(bj, cj, uj, cols, atoms)          # threaded
        XT = SLCE._design_torque(bj, cj, uj, atoms)               # threaded
        blk = 3 * length(atoms)
        XFref = Matrix{Float64}(undef, length(cj) * blk, length(cols))
        XTref = Matrix{Float64}(undef, length(cj) * blk, mj)
        Ge = Matrix{Float64}(undef, 3, natj)
        Gu = Matrix{Float64}(undef, 3, natj)
        scr2 = SLCE.SALCScratch()
        for j = 1:mj, ci in eachindex(cj)                         # race-free serial
            fill!(Ge, 0.0)
            fill!(Gu, 0.0)
            SLCE.accumulate_grad!(Ge, Gu, sj[j], cj[ci], uj[ci], 1.0, scr2)
            jj = findfirst(==(j), cols)
            rb = blk * (ci - 1)
            for (k, a) in enumerate(atoms)
                e_a = view(cj[ci], :, a)
                t = cross(view(Ge, :, a), e_a)       # τ = ∇Φ × e (the design's sign)
                for d = 1:3
                    XTref[rb + 3 * (k - 1) + d, j] = t[d]
                    jj === nothing ||
                        (XFref[rb + 3 * (k - 1) + d, jj] = -Gu[d, a])
                end
            end
        end
        @test XF == XFref
        @test XT == XTref
        @test SLCE._design_force(bj, cj, uj, cols, atoms) == XF   # idempotent
        @test SLCE._design_torque(bj, cj, uj, atoms) == XT
    end
end
