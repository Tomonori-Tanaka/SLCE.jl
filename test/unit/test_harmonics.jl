using Test
using SLCE: Harmonics
using StaticArrays
using LinearAlgebra
using Random

const Z = Harmonics.Zlm
const gZ = Harmonics.grad_Zlm

@testset "harmonics" begin
    rng = MersenneTwister(20260627)

    # Convention-independent anchor #1: closed-form standard real solid harmonics
    # on the unit sphere. These pin the absolute normalization and signs (the
    # gauge-invariant / oracle tests cannot — a self-consistent wrong sign would
    # pass those).
    @testset "closed-form low-l values" begin
        c1 = sqrt(3 / (4π))
        c2 = sqrt(15 / (4π))
        c20 = sqrt(5 / (16π))
        c22 = sqrt(15 / (16π))
        for _ = 1:50
            u = rand_unit(rng)
            x, y, z = u
            @test Z(0, 0, u) ≈ 1 / (2 * sqrt(π))
            @test Z(1, -1, u) ≈ c1 * y
            @test Z(1, 0, u) ≈ c1 * z
            @test Z(1, 1, u) ≈ c1 * x
            @test Z(2, -2, u) ≈ c2 * x * y
            @test Z(2, -1, u) ≈ c2 * y * z
            @test Z(2, 0, u) ≈ c20 * (3z^2 - 1)
            @test Z(2, 1, u) ≈ c2 * x * z
            @test Z(2, 2, u) ≈ c22 * (x^2 - y^2)
        end
    end

    # Convention-independent anchor #2: the gradient is tangent (u·∇Z = 0) and
    # agrees with an on-sphere central difference. The FD perturbation is
    # renormalized onto the sphere (the harmonics are not degree-0 homogeneous),
    # which is exactly what the torque self-consistency relies on.
    @testset "gradient tangency + on-sphere central difference" begin
        ε = 1e-6
        for l = 0:4, m = -l:l
            for _ = 1:8
                u = rand_unit(rng)
                g = gZ(l, m, u)
                @test abs(dot(g, u)) < 1e-7
                a = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
                t1 = a - dot(a, u) * u
                t1 = t1 / norm(t1)
                t2 = cross(u, t1)
                for t in (t1, t2)
                    up = u + ε * t
                    up = up / norm(up)
                    um = u - ε * t
                    um = um / norm(um)
                    fd = (Z(l, m, up) - Z(l, m, um)) / (2ε)
                    @test isapprox(fd, dot(g, t); atol = 1e-5, rtol = 1e-5)
                end
            end
        end
    end

    # Convention-independent anchor #3: tangency is an IDENTITY in the input, not a
    # consequence of the input being nearly unit. The projection subtracts
    # `û(û·∂Z)`, so `û·∇Z = 0` analytically for every `u ≠ 0` — scaling `u` scales
    # `∂Z` and `û` is unchanged, so the removed component is the whole radial part
    # at any radius. Nothing here is read off the implementation: the expected
    # value is exact zero from that identity, and the bound is the float rounding
    # floor with headroom.
    #
    # This is the gate on the `/ r²` in `_grad_zlm_assemble`. Dropping it (the
    # `u(u·∂Z)` form) leaves a radial residue `≈ 2·C_l·δ` with
    # `C_l = √((2l+1)/4π)·l(l+1)/2`, i.e. 1.7e-7 at `δ = 1e-8` and 1.7e-5 at
    # `δ = 1e-6` — the mutation is resolved by 7 to 10 orders against this bound,
    # and the `δ = 0` row alone would NOT resolve it (both forms agree there).
    #
    # `δ` is bounded by the OTHER precondition rather than by the projection: the
    # polar recursion reaches `dnPl`, whose domain is `|z| ≤ 1`, so a scaled
    # direction must keep `|z|(1 + δ) ≤ 1`. The fixture therefore holds `|z|` away
    # from the pole instead of sampling the sphere uniformly.
    @testset "gradient tangency is independent of ‖u‖" begin
        dirs = SVector{3,Float64}[]
        while length(dirs) < 12
            v = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
            abs(v[3]) <= 0.9 && push!(dirs, v)   # room for the largest δ below
        end
        for δ in (0.0, 1e-12, 1e-8, 1e-6, 1e-4, 1e-2), u0 in dirs
            u = (1 + δ) * u0
            for l = 0:4, m = -l:l
                @test abs(dot(u0, Harmonics.grad_Zlm_unsafe(l, m, u))) < 1e-12
            end
        end
    end

    @testset "validation" begin
        u = SVector{3,Float64}(0.0, 0.0, 1.0)
        @test_throws ArgumentError Z(-1, 0, u)
        @test_throws ArgumentError Z(1, 2, u)
        @test_throws ArgumentError Z(1, 0, SVector{3,Float64}(1.0, 1.0, 1.0))
        @test_throws ArgumentError gZ(2, -3, u)
    end

    @testset "lm_index contiguous and unique" begin
        idxs = [Harmonics.lm_index(l, m) for l = 0:4 for m = -l:l]
        @test sort(idxs) == collect(1:Harmonics.num_lm(4))
    end

    @testset "cache-threaded variants: bitwise-identical and allocation-free" begin
        # The 4-arg methods reuse `cache` as the dnPl recursion workspace; the value
        # must be identical bit for bit (the recursion overwrites every entry it
        # reads — cache contents must be irrelevant, hence the NaN poisoning).
        rng = MersenneTwister(11)
        lmax = 6
        cache = Vector{Float64}(undef, lmax + 1)
        for _ = 1:5
            u = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
            for l = 0:lmax, m = -l:l
                fill!(cache, NaN)
                @test Harmonics.Zlm_unsafe(l, m, u, cache) ==
                      Harmonics.Zlm_unsafe(l, m, u)
                fill!(cache, NaN)
                @test Harmonics.grad_Zlm_unsafe(l, m, u, cache) ==
                      Harmonics.grad_Zlm_unsafe(l, m, u)
            end
        end
        # measure through a function barrier — a global-scope call boxes its result
        zallocs(l, m, u, c) = @allocations Harmonics.Zlm_unsafe(l, m, u, c)
        gallocs(l, m, u, c) = @allocations Harmonics.grad_Zlm_unsafe(l, m, u, c)
        u = normalize(SVector{3,Float64}(0.3, -0.5, 0.81))
        zallocs(4, 2, u, cache)                       # warm up
        gallocs(4, 2, u, cache)
        @test zallocs(4, 2, u, cache) == 0
        @test gallocs(4, 2, u, cache) == 0
    end
end
