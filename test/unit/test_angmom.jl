using Test
using SLCE: AngularMomentum, Harmonics
using StaticArrays
using LinearAlgebra
using Random

const CG = AngularMomentum.clebsch_gordan
const Dreal = AngularMomentum.wignerD_real

@testset "angular momentum" begin
    @testset "Clebsch-Gordan known values" begin
        @test CG(1, 0, 1, 0, 0, 0) ≈ -1 / sqrt(3)
        @test CG(1, 1, 1, -1, 0, 0) ≈ 1 / sqrt(3)
        @test CG(1, 1, 1, -1, 1, 0) ≈ 1 / sqrt(2)
        @test CG(1, 1, 1, -1, 2, 0) ≈ 1 / sqrt(6)
        @test CG(2, 0, 2, 0, 0, 0) ≈ 1 / sqrt(5)
        @test CG(1, 1, 1, 1, 2, 2) ≈ 1.0
        # selection rules
        @test CG(1, 1, 1, 1, 0, 0) == 0.0      # M = 2 ≠ 0
        @test CG(1, 0, 1, 0, 3, 0) == 0.0      # J > j1+j2
        @test CG(1, 0, 1, 0, 1, 0) == 0.0      # ⟨1010|10⟩ vanishes
        # Float64 factorial range: overflow throws instead of returning Inf/NaN
        @test_throws ArgumentError CG(60, 0, 60, 0, 60, 0)   # 3l + 1 = 181 > 170
        @test isfinite(CG(20, 0, 20, 0, 20, 0))              # comfortably in range
        @test CG(60, 0, 60, 0, 200, 0) == 0.0                # rule fails first: 0, no throw
    end

    @testset "CG orthonormality" begin
        for j1 = 0:2, j2 = 0:2
            for J = abs(j1 - j2):(j1+j2), Jp = abs(j1 - j2):(j1+j2)
                for M = -J:J
                    Mp = M
                    s = 0.0
                    for m1 = -j1:j1, m2 = -j2:j2
                        s += CG(j1, m1, j2, m2, J, M) * CG(j1, m1, j2, m2, Jp, Mp)
                    end
                    @test isapprox(s, J == Jp ? 1.0 : 0.0; atol = 1e-12)
                end
            end
        end
    end

    @testset "real Wigner-D: functional identity (own Zₗₘ, fresh points)" begin
        rng = MersenneTwister(7)
        for _ = 1:20
            R = rand_rotation(rng)
            for l = 0:4
                Δ = Dreal(l, R)
                @test size(Δ) == (2l + 1, 2l + 1)
                # orthogonality (rotation preserves the L² inner product)
                @test isapprox(Δ * Δ', Matrix(I, 2l + 1, 2l + 1); atol = 1e-9)
                # identity on FRESH directions not used in the fit
                for _ = 1:5
                    u = rand_unit(rng)
                    ru = R * u
                    lhs = [Harmonics.Zlm(l, m, ru) for m = -l:l]
                    rhs = Δ * [Harmonics.Zlm(l, m, u) for m = -l:l]
                    @test isapprox(lhs, rhs; atol = 1e-9)
                end
            end
        end
    end

    @testset "real Wigner-D: special cases" begin
        @test Dreal(0, Matrix{Float64}(I, 3, 3)) == fill(1.0, 1, 1)
        # identity rotation → identity matrix
        for l = 1:3
            @test isapprox(Dreal(l, Matrix{Float64}(I, 3, 3)),
                           Matrix(I, 2l + 1, 2l + 1); atol = 1e-9)
        end
        # inversion: Zₗₘ(−u) = (−1)^l Zₗₘ(u) ⇒ Δ = (−1)^l I
        for l = 1:4
            Δ = Dreal(l, Matrix{Float64}(-I, 3, 3))
            @test isapprox(Δ, (-1.0)^l * Matrix(I, 2l + 1, 2l + 1); atol = 1e-9)
        end
        # a reflection (improper) is handled too: σ_z = diag(1,1,-1)
        rng = MersenneTwister(99)
        σz = SMatrix{3,3,Float64}(diagm([1.0, 1.0, -1.0]))
        for l = 1:3
            Δ = Dreal(l, σz)
            @test isapprox(Δ * Δ', Matrix(I, 2l + 1, 2l + 1); atol = 1e-9)
            u = rand_unit(rng)
            lhs = [Harmonics.Zlm(l, m, σz * u) for m = -l:l]
            rhs = Δ * [Harmonics.Zlm(l, m, u) for m = -l:l]
            @test isapprox(lhs, rhs; atol = 1e-9)
        end
    end

    @testset "complex→real unitary matches the closed form (paper Eq. B2)" begin
        # U^{(l)}: Z_l^m = Σ_{m'} U[m,m'] Y_l^{m'}, indexed row/col = m+l+1.
        # Convention-independent anchor (does not use the Magesty oracle).
        for l = 0:3
            U = SLCE.AngularMomentum.c2r_matrix(l)
            ix(m) = m + l + 1
            ref = zeros(ComplexF64, 2l + 1, 2l + 1)
            ref[ix(0), ix(0)] = 1
            for m = 1:l
                s = 1 / sqrt(2.0)
                ref[ix(m), ix(m)] = (-1.0)^m * s
                ref[ix(m), ix(-m)] = s
                ref[ix(-m), ix(-m)] = im * s
                ref[ix(-m), ix(m)] = -im * (-1.0)^m * s
            end
            @test isapprox(U, ref; atol = 1e-12)
            @test isapprox(U' * U, Matrix(I, 2l + 1, 2l + 1); atol = 1e-12)  # unitary
        end
    end
end
