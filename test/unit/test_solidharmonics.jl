# Displacement-channel solid harmonics: 4π-free (Racah-type) normalization,
# polynomial regularity at u = 0, and the Euclidean (non-tangent) gradient.

using SLCE: Harmonics, SolidHarmonics
using StaticArrays
using LinearAlgebra: norm
using Random: MersenneTwister

@testset "SolidHarmonics" begin
    rng = MersenneTwister(0x51ce)
    randunit(rng) = (v = randn(rng, 3); v ./ norm(v))

    @testset "index and validation" begin
        @test SolidHarmonics.solid_harmonic_index(0, 0) == 1
        for l = 0:6, m = -l:l
            @test SolidHarmonics.solid_harmonic_index(l, m) == Harmonics.lm_index(l, m)
        end
        @test SolidHarmonics.num_solid_harmonics(4) == 25
        @test_throws ArgumentError SolidHarmonics.solid_harmonics(-1, [0.0, 0.0, 0.0])
        @test_throws ArgumentError SolidHarmonics.solid_harmonics(2, [1.0, 0.0])
        @test_throws ArgumentError SolidHarmonics.Rlm(2, 3, [1.0, 0.0, 0.0])
        @test_throws ArgumentError SolidHarmonics.grad_Rlm(1, -2, [1.0, 0.0, 0.0])
        vals = zeros(3)
        @test_throws ArgumentError SolidHarmonics.solid_harmonics!(vals, 2, [1.0, 0.0, 0.0])
        grads = zeros(3, 3)
        @test_throws ArgumentError SolidHarmonics.solid_harmonics_grad!(zeros(9), grads,
            2, [1.0, 0.0, 0.0])
    end

    # The M1 normalization gate: the displacement kernel is 4π-free (Racah-type),
    # so on the unit sphere it differs from the spin-side tesseral kernel by the
    # explicit factor √(4π/(2l+1)) — asserted here up to l = 16.
    @testset "Zlm cross-check with explicit √(4π/(2l+1)) factor, l ≤ 16" begin
        lmax = 16
        for _ = 1:8
            u = randunit(rng)
            vals = SolidHarmonics.solid_harmonics(lmax, u)
            for l = 0:lmax
                factor = sqrt(4π / (2l + 1))
                for m = -l:l
                    idx = SolidHarmonics.solid_harmonic_index(l, m)
                    zref = factor * Harmonics.Zlm(l, m, u)
                    @test vals[idx] ≈ zref atol = 1e-11 rtol = 1e-11
                end
            end
        end
    end

    @testset "rank-1 factors are the Cartesian components exactly" begin
        for _ = 1:4
            u = randn(rng, 3) * 0.7
            @test SolidHarmonics.Rlm(1, -1, u) == u[2]
            @test SolidHarmonics.Rlm(1, 0, u) == u[3]
            @test SolidHarmonics.Rlm(1, 1, u) == u[1]
        end
        @test SolidHarmonics.Rlm(0, 0, randn(rng, 3)) == 1.0
    end

    @testset "homogeneity R(λu) = λ^l R(u)" begin
        u = randn(rng, 3)
        λ = 0.37
        vals1 = SolidHarmonics.solid_harmonics(6, u)
        vals2 = SolidHarmonics.solid_harmonics(6, λ * u)
        for l = 0:6, m = -l:l
            idx = SolidHarmonics.solid_harmonic_index(l, m)
            @test vals2[idx] ≈ λ^l * vals1[idx] atol = 1e-12 rtol = 1e-12
        end
    end

    @testset "u = 0 exactness" begin
        z3 = [0.0, 0.0, 0.0]
        vals, grads = SolidHarmonics.solid_harmonics_grad(5, z3)
        @test vals[1] == 1.0
        for l = 1:5, m = -l:l
            @test vals[SolidHarmonics.solid_harmonic_index(l, m)] == 0.0
        end
        # Gradient at u = 0: exactly zero except the rank-1 block, whose gradients
        # are the exact Cartesian unit vectors (R₁₋₁, R₁₀, R₁₁ = y, z, x).
        for l = 0:5, m = -l:l
            idx = SolidHarmonics.solid_harmonic_index(l, m)
            g = SVector{3, Float64}(grads[1, idx], grads[2, idx], grads[3, idx])
            if l == 1
                expected = m == -1 ? SVector(0.0, 1.0, 0.0) :
                           m == 0 ? SVector(0.0, 0.0, 1.0) : SVector(1.0, 0.0, 0.0)
                @test g == expected
            else
                @test g == SVector(0.0, 0.0, 0.0)
            end
        end
    end

    @testset "Euclidean gradient vs central differences" begin
        h = 1e-6
        for _ = 1:4
            u = randn(rng, 3) * 0.8
            vals, grads = SolidHarmonics.solid_harmonics_grad(6, u)
            @test vals ≈ SolidHarmonics.solid_harmonics(6, u) atol = 0 rtol = 0
            for l = 0:6, m = -l:l
                idx = SolidHarmonics.solid_harmonic_index(l, m)
                for α = 1:3
                    up = copy(u); up[α] += h
                    um = copy(u); um[α] -= h
                    fd = (SolidHarmonics.Rlm(l, m, up) - SolidHarmonics.Rlm(l, m, um)) /
                         (2h)
                    @test grads[α, idx] ≈ fd atol = 1e-6 rtol = 1e-6
                end
            end
        end
    end

    @testset "in-place variants match and are allocation-free" begin
        u = randn(rng, 3)
        n = SolidHarmonics.num_solid_harmonics(8)
        vals = Vector{Float64}(undef, n)
        grads = Matrix{Float64}(undef, 3, n)
        SolidHarmonics.solid_harmonics!(vals, 8, u)
        @test vals == SolidHarmonics.solid_harmonics(8, u)
        SolidHarmonics.solid_harmonics_grad!(vals, grads, 8, u)
        vref, gref = SolidHarmonics.solid_harmonics_grad(8, u)
        @test vals == vref
        @test grads == gref
        su = SVector{3, Float64}(u)
        alloc = @allocated SolidHarmonics.solid_harmonics!(vals, 8, su)
        @test alloc == 0
        alloc = @allocated SolidHarmonics.solid_harmonics_grad!(vals, grads, 8, su)
        @test alloc == 0
    end

    @testset "single-(l,m) accessors agree with the batch evaluator" begin
        u = randn(rng, 3) * 1.3
        vals, grads = SolidHarmonics.solid_harmonics_grad(4, u)
        for l = 0:4, m = -l:l
            idx = SolidHarmonics.solid_harmonic_index(l, m)
            @test SolidHarmonics.Rlm(l, m, u) == vals[idx]
            @test SolidHarmonics.grad_Rlm(l, m, u) ==
                  SVector{3, Float64}(grads[1, idx], grads[2, idx], grads[3, idx])
        end
    end
end
