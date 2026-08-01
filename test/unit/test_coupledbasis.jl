using Test
using SLCE: Harmonics, AngularMomentum
using SLCE
using StaticArrays
using LinearAlgebra
using Random

const coupled_bases = SLCE.coupled_bases
const Dreal = AngularMomentum.wignerD_real

# Coupled field components f_Mf(e₁,…,e_N) = Σ tensor[m…, Mf] ∏ Zₗᵢmᵢ(eᵢ).
function coupled_value(cb, es)
    ls, Lf, T = cb.ls, cb.Lf, cb.tensor
    N = length(ls)
    Zs = [[Harmonics.Zlm(ls[i], m, es[i]) for m = -ls[i]:ls[i]] for i = 1:N]
    f = zeros(2Lf + 1)
    for index in CartesianIndices(ntuple(i -> 2ls[i] + 1, N))
        w = 1.0
        for i = 1:N
            w *= Zs[i][index[i]]
        end
        for q = 1:(2Lf+1)
            f[q] += T[Tuple(index)..., q] * w
        end
    end
    return f
end

@testset "coupled bases" begin
    @testset "realness, shape, Frobenius norm² = 2Lf+1" begin
        for ls in ([1], [2], [1, 1], [1, 2], [2, 2])
            for cb in coupled_bases(ls)
                @test eltype(cb.tensor) == Float64
                @test ndims(cb.tensor) == length(ls) + 1
                @test size(cb.tensor, ndims(cb.tensor)) == 2cb.Lf + 1
                @test isapprox(sum(abs2, cb.tensor), 2cb.Lf + 1; rtol = 1e-10)
            end
        end
    end

    @testset "pair coupling enumerates Lf = |l1-l2|:(l1+l2)" begin
        cbs = coupled_bases([1, 2])
        @test sort(getfield.(cbs, :Lf)) == [1, 2, 3]
        @test all(cb -> length(cb.Lseq) == 0, cbs)   # N=2 ⇒ no intermediate
        # scalar_only keeps only Lf == 0 (none here for (1,2))
        @test isempty(coupled_bases([1, 2]; scalar_only = true))
        @test length(coupled_bases([1, 1]; scalar_only = true)) == 1   # Lf=0 exists for (1,1)
    end

    # The defining property, tying M4 back to M2 (Zₗₘ) and M3 (wignerD_real) with
    # no oracle: under a common rotation R of all site directions, the coupled
    # components transform as the Lf irrep.
    @testset "rotational equivariance: f(R·e) = Δ^Lf(R) · f(e)" begin
        rng = MersenneTwister(2024)
        for ls in ([1], [2], [1, 1], [1, 2], [2, 2])
            cbs = coupled_bases(ls)
            for _ = 1:10
                R = rand_rotation(rng)
                es = [rand_unit(rng) for _ = 1:length(ls)]
                Res = [R * e for e in es]
                for cb in cbs
                    f = coupled_value(cb, es)
                    fR = coupled_value(cb, Res)
                    Δ = Dreal(cb.Lf, R)
                    @test isapprox(fR, Δ * f; atol = 1e-9, rtol = 1e-8)
                end
            end
        end
    end
end
