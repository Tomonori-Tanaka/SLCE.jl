# SLCEDataset views (src/slce/model.jl): length / getindex slicing / vcat. The rows
# of X_E / X_T must follow the selected configs exactly — a mis-sliced torque block
# would silently corrupt a co-fit — so every slice is compared against the rows of
# the full matrices, and vcat of a partition must rebuild the original bit-for-bit.

using Test
using SLCE
using LinearAlgebra
using Random

@testset "SLCEDataset slicing / vcat" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    basis = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                                        soc = false))
    rng = MersenneTwister(31)
    n = 6
    nat = 2
    configs = [randcfg(rng, nat) for _ = 1:n]
    energies = collect(1.0:n)
    torques = [randn(rng, 3, nat) for _ = 1:n]
    ds = SLCEDataset(basis, configs, energies, torques)
    dse = SLCEDataset(basis, configs, energies)              # energy-only twin

    trows(i) = (3 * nat * (i - 1) + 1):(3 * nat * i)        # config i's torque rows

    @testset "length / firstindex / lastindex" begin
        @test length(ds) == n
        @test firstindex(ds) == 1
        @test lastindex(ds) == n
    end

    @testset "range slice carries matching design rows" begin
        s = ds[2:4]
        @test s.basis === basis
        @test length(s) == 3
        @test s.configs == configs[2:4]
        @test s.y_E == energies[2:4]
        @test s.X_E == ds.X_E[2:4, :]
        @test s.X_T == ds.X_T[reduce(vcat, trows.(2:4)), :]
        @test s.y_T == ds.y_T[reduce(vcat, trows.(2:4))]
    end

    @testset "mask / colon / duplicates / end" begin
        s = ds[[true, false, true, false, false, true]]
        @test s.y_E == energies[[1, 3, 6]]
        @test s.X_T == ds.X_T[reduce(vcat, trows.([1, 3, 6])), :]
        c = ds[:]
        @test c.X_E == ds.X_E && c.y_T == ds.y_T
        dup = ds[[3, 3]]                                     # bootstrap-style reuse
        @test length(dup) == 2 && dup.y_E == [3.0, 3.0]
        @test ds[2:end].y_E == energies[2:end]
        se = dse[5:6]
        @test !has_torque(se) && se.y_E == [5.0, 6.0]
    end

    @testset "slice validation" begin
        @test_throws ArgumentError ds[Int[]]
        @test_throws DimensionMismatch ds[[true, false]]
        @test_throws BoundsError ds[[0]]
        @test_throws BoundsError ds[[n + 1]]
    end

    @testset "vcat of a partition rebuilds the original" begin
        w = vcat(ds[1:2], ds[3:3], ds[4:6])
        @test w.configs == ds.configs
        @test w.X_E == ds.X_E && w.y_E == ds.y_E
        @test w.X_T == ds.X_T && w.y_T == ds.y_T
        @test vcat(ds) === ds
        we = vcat(dse[1:3], dse[4:6])
        @test we.X_E == dse.X_E && !has_torque(we)
    end

    @testset "vcat validation and mixed concatenation" begin
        basis2 = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                             soc = false))
        ds2 = SLCEDataset(basis2, configs, energies)
        @test_throws ArgumentError vcat(dse, ds2)            # fingerprint mismatch
        # torque-bearing + energy-only concatenate into a MIXED dataset: the torque
        # rows keep their configurations through the re-offset torque_config
        mx = vcat(ds, dse)
        @test length(mx) == 2n && has_torque(mx)
        @test mx.X_T == ds.X_T && mx.y_T == ds.y_T           # only ds contributes rows
        @test mx.torque_config == ds.torque_config           # configs 1:n unchanged
        mx2 = vcat(dse, ds)                                  # reversed order: offset by n
        @test mx2.torque_config == ds.torque_config .+ n
        @test mx2.X_T == ds.X_T
    end

    @testset "ragged (mixed-presence) torque bookkeeping" begin
        # torque data only for configs 2 and 5 — built via the explicit mixed form
        sel = [2, 5]
        dm = SLCEDataset(basis, configs, energies, torques[sel], sel)
        @test has_torque(dm)
        @test length(dm.y_T) == 2 * 3 * nat
        @test dm.torque_config == repeat(sel; inner = 3 * nat)
        # torque design rows are bit-identical to the all-torque dataset's rows for
        # the same configs (the design only depends on the config)
        @test dm.X_T == ds.X_T[reduce(vcat, trows.(sel)), :]
        @test dm.y_T == reduce(vcat, [vec(torques[i]) for i in sel])
        @test dm.X_E == ds.X_E                               # energy block unaffected
        # slicing keeps exactly the surviving rows and relabels them
        s = dm[[5, 1, 2]]                                    # unsorted, mixed presence
        @test s.torque_config == vcat(fill(1, 3 * nat), fill(3, 3 * nat))
        @test s.X_T == vcat(dm.X_T[(3 * nat + 1):(6 * nat), :],
                            dm.X_T[1:(3 * nat), :])
        se = dm[[1, 3]]                                      # no torque-bearing config
        @test !has_torque(se) && isempty(se.torque_config)
        # a partition rebuilds the mixed dataset bit-for-bit
        w = vcat(dm[1:3], dm[4:6])
        @test w.X_T == dm.X_T && w.y_T == dm.y_T && w.torque_config == dm.torque_config
        # duplicate slicing duplicates the rows with fresh labels
        dup = dm[[2, 2]]
        @test dup.torque_config == vcat(fill(1, 3 * nat), fill(2, 3 * nat))
        @test dup.X_T[1:(3 * nat), :] == dup.X_T[(3 * nat + 1):(6 * nat), :]
        # invariant surface of the mixed constructor
        @test_throws ArgumentError SLCEDataset(basis, configs, energies,
                                              torques[[2, 5]], [5, 2])   # unsorted sel
        @test_throws ArgumentError SLCEDataset(basis, configs, energies,
                                              torques[[2, 2]], [2, 2])   # duplicate sel
        @test_throws ArgumentError SLCEDataset(basis, configs, energies,
                                              torques[sel], [2, 7])      # out of range
        @test_throws ArgumentError SLCEDataset(basis, configs, energies,
                                              torques[sel], Int[])       # empty sel
        @test_throws ArgumentError SLCEDataset(basis, configs, energies,
                                              [torques[2]], sel)         # block count
    end

    @testset "mixed dataset: grouped labels, co-fit, stratified CV" begin
        rng2 = MersenneTwister(77)
        jphi = randn(rng2, length(basis.salc_basis))
        sel = [1, 4]
        y = 0.3 .+ ds.X_E * jphi
        τs = [reshape(ds.X_T[trows(i), :] * jphi, 3, nat) for i in sel]
        dm = SLCEDataset(basis, configs, y, τs, sel)
        # grouped-CV labels come from the stored per-row config index — the old
        # uniform-block derivation div(n_T, n_E) would silently mislabel here
        X, _, _, _, groups = SLCE._assemble_problem(dm, 0.5)
        @test groups == vcat(1:n, repeat(sel; inner = 3 * nat))
        @test size(X, 1) == n + 2 * 3 * nat
        # the co-fit recovers the in-span target through the ragged torque block
        f = fit(SLCEFit, dm, OLS(); torque_weight = 0.5)
        @test isapprox(f.jphi, jphi; atol = 1e-8)
        # a torque-free training slice under w > 0 fails loudly at fit's
        # has_torque pre-check (the √(w/0) guard in _assemble_problem is
        # defense-in-depth behind it)
        @test_throws ArgumentError fit(SLCEFit, dm[[2, 3]], OLS(); torque_weight = 0.5)
        # stratified cross_validate: torque-bearing configs are a minority, yet
        # every fold/score stays finite (needs ≥ 2 torque configs for w > 0)
        cv = cross_validate(dm, OLS(); torque_weight = 0.5, nfolds = 2)
        @test all(isfinite, cv.score) && isfinite(cv.pooled_score)
        # w = 0 on a mixed dataset: score is never 0·NaN even for torque-free folds
        cv0 = cross_validate(dm, OLS(); torque_weight = 0.0, nfolds = 2)
        @test all(isfinite, cv0.score)
        # a single torque-bearing config cannot co-fit-CV (training folds would starve)
        dm1 = SLCEDataset(basis, configs, y, [τs[1]], [1])
        @test_throws ArgumentError cross_validate(dm1, OLS(); torque_weight = 0.5,
                                                  nfolds = 2)
    end

    @testset "a slice fits and predicts the held-out part" begin
        jphi = randn(rng, length(basis.salc_basis))
        y = 0.3 .+ ds.X_E * jphi                             # in-span target
        dsy = SLCEDataset(basis, configs, y)
        f = fit(SLCEFit, dsy[1:4], OLS())
        @test isapprox(predict_energy(f, configs[5:6]), y[5:6]; atol = 1e-6)
    end

    @testset "a basis with no columns cannot carry training data" begin
        # A zero-column basis is reachable from an ordinary spec — the pair a
        # minimum-image convention cannot express is the same-atom one, so a
        # cation-cation superexchange spec on a cell with one cation has nothing
        # to build — and everything downstream then reports on the intercept:
        # r2_energy is exactly 0.0 and predict_energy answers with one number for
        # every configuration. The refusal belongs at this boundary because
        # `restrict(model, :spin)` builds an empty basis deliberately (a
        # lattice-only model's clamped-ion sub-model) and must keep working.
        empty_b = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 0.05, lmax = [2],
                                               soc = false))
        @test n_salcs(empty_b) == 0
        @test_throws ArgumentError SLCEDataset(empty_b, configs, energies)
        err = try
            SLCEDataset(empty_b, configs, energies)
        catch e
            e
        end
        @test occursin("no SALC columns", err.msg)
        @test occursin("image of ITSELF", err.msg)
    end
end

@testset "the atol cap binds the dataset door (review 2026-08-11 M2)" begin
    # `_validate_config` validates WITHOUT projecting — the raw columns enter the
    # design matrix — so past `_DIRECTION_ATOL_MAX` a moment-scaled vector is not
    # merely silently projected (the projecting doors' complaint): it enters raw and
    # biases every fitted jϕ by C_l·δ. This door used to accept ANY atol
    # (`atol = 0.5` admitted ‖e‖ = 0.6 columns into a ~98 %-biased design).
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    basis = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                                        soc = false))
    rng = MersenneTwister(11)
    configs = [randcfg(rng, 2) for _ = 1:4]
    energies = collect(1.0:4)
    scaled = [c .* 0.6 for c in configs]
    err = try
        SLCEDataset(basis, scaled, energies; atol = 0.5)
        nothing
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("exceeds the hard cap", err.msg)
    # a widened-but-capped band still serves the 4-decimal-MAGMOM demand …
    near = [c .* (1 + 2e-5) for c in configs]
    @test length(SLCEDataset(basis, near, energies; atol = 1e-4)) == 4
    # … and just past the cap is refused regardless of the data
    @test_throws ArgumentError SLCEDataset(basis, near, energies; atol = 2e-2)
end
