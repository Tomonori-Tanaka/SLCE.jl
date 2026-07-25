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

    @testset "vcat validation" begin
        basis2 = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                             soc = false))
        ds2 = SLCEDataset(basis2, configs, energies)
        @test_throws ArgumentError vcat(dse, ds2)            # fingerprint mismatch
        @test_throws ArgumentError vcat(ds, dse)             # torque vs energy-only
    end

    @testset "a slice fits and predicts the held-out part" begin
        jphi = randn(rng, length(basis.salc_basis))
        y = 0.3 .+ ds.X_E * jphi                             # in-span target
        dsy = SLCEDataset(basis, configs, y)
        f = fit(SLCEFit, dsy[1:4], OLS())
        @test isapprox(predict_energy(f, configs[5:6]), y[5:6]; atol = 1e-6)
    end
end
