# The unit-direction types (src/direction.jl). What is being gated is a CONTRACT, not
# a computation, so every expected value here comes from the contract's own statement —
# "the result is a unit vector", "these bits are preserved", "this is refused" — and
# never from running the constructor and recording what it produced.

using Test
using SLCE
using SLCE: UnitVector3, SpinConfiguration, Trusted, _DIRECTION_ATOL,
            _DIRECTION_ATOL_MAX
using LinearAlgebra
using Random
using StaticArrays

@testset "unit directions" begin
    rng = MersenneTwister(31)

    @testset "the plain constructor establishes the invariant" begin
        # The promise is `‖u‖ = 1`, so the oracle is the number 1 — not whatever the
        # projection happens to emit. `4eps()` is rounding headroom, not a fitted bound.
        for _ = 1:200
            v = randn(rng, 3)
            for scale in (1.0, 1 + 5e-7, 1 - 5e-7)
                u = UnitVector3(normalize(v) .* scale)
                @test abs(norm(u) - 1) < 4eps()
            end
        end
        # ... including for an input that is already exactly unit, where projection is
        # a no-op in value but not necessarily in bits
        u = UnitVector3(SVector(0.0, 0.0, 1.0))
        @test u[1] == 0.0 && u[2] == 0.0 && u[3] == 1.0
    end

    @testset "Trusted preserves the caller's bits" begin
        # This is the constructor SLCEDynamics' checkpoint restore needs: an
        # already-on-sphere state must come back bit-identical, because re-projecting
        # applies an operation the uninterrupted run never applied and `normalize` is
        # not bitwise idempotent. The oracle is bit equality with the input.
        moved = 0
        for _ = 1:2000
            v = normalize(SVector{3,Float64}(randn(rng), randn(rng), randn(rng)))
            t = UnitVector3(v, Trusted())
            @test t[1] === v[1] && t[2] === v[2] && t[3] === v[3]
            p = UnitVector3(v)
            (p[1] === v[1] && p[2] === v[2] && p[3] === v[3]) || (moved += 1)
        end
        # ... and the reason the two constructors must both exist: re-projecting an
        # already-unit direction is NOT the identity on bits. Measured ~38 %; the gate
        # only asserts it is common enough that a resume path cannot ignore it.
        @test moved > 200
    end

    @testset "validation: what is refused, and why the component bound is separate" begin
        @test_throws ArgumentError UnitVector3([0.0, 0.0, 0.0])          # not unit
        @test_throws ArgumentError UnitVector3([1.7, 0.0, 0.0])          # a magnitude
        @test_throws ArgumentError UnitVector3([NaN, 0.0, 0.0])          # not finite
        @test_throws DimensionMismatch UnitVector3([1.0, 0.0])
        # The pole hazard. `‖e‖ − 1 = 5e-9` clears any sane band, so no tolerance can
        # establish the harmonic kernel's `|z| ≤ 1` precondition; only the component
        # bound rejects this, and the error must be ours rather than a bare DomainError
        # thrown later from inside an accumulation.
        pole = [0.0, 0.0, 1 + 5e-9]
        @test abs(norm(pole) - 1) < 1e-8
        @test_throws ArgumentError UnitVector3(pole, Trusted())
        # ... and it rejects nothing harmless: off norm with every component inside
        @test UnitVector3([0.6, 0.8, 0.0] .* (1 + 1e-7), Trusted()) isa UnitVector3
        # The plain constructor's projection makes the pole case representable, because
        # projection lands the component exactly on 1.0 rather than past it.
        @test UnitVector3(pole)[3] == 1.0
    end

    @testset "atol is a band, and it is capped" begin
        v = normalize(SVector(1.0, 1.0, 1.0)) .* (1 + 2e-5)   # a 4-decimal MAGMOM
        @test_throws ArgumentError UnitVector3(v)                       # default 1e-6
        @test UnitVector3(v; atol = 1e-4) isa UnitVector3
        # The cap is what keeps "widen the tolerance" from becoming "accept a different
        # vector and silently project it onto the sphere".
        @test_throws ArgumentError UnitVector3(v; atol = 2 * _DIRECTION_ATOL_MAX)
        @test_throws ArgumentError UnitVector3(v; atol = -1.0)
        @test _DIRECTION_ATOL < _DIRECTION_ATOL_MAX
    end

    @testset "SpinConfiguration: every column, by construction" begin
        m = reduce(hcat, [normalize(randn(rng, 3)) .* (1 + 4e-7) for _ = 1:5])
        c = SpinConfiguration(m)
        @test size(c) == (3, 5) && SLCE.n_atoms(c) == 5
        for a = 1:5
            @test abs(norm(c[:, a]) - 1) < 4eps()
        end
        @test isapprox(Matrix(c), m; rtol = 1e-6)
        # one bad column is enough, and the message names which
        bad = copy(m)
        bad[:, 3] .*= 1.7
        err = try
            SpinConfiguration(bad)
            nothing
        catch ex
            ex
        end
        @test err isa ArgumentError && occursin("column 3", err.msg)
        @test_throws ArgumentError SpinConfiguration(zeros(3, 2))
        @test_throws ArgumentError SpinConfiguration(ones(2, 2))
        # idempotent on its own type: already-established, so nothing is re-projected
        @test SpinConfiguration(c) === c
    end
end
