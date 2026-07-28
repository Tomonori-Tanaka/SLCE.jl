# Magnon–phonon vertices (src/slce/magnonphonon.jl) — the mixed `∂²E/∂u ∂e`, design
# record §7's third strain-tier deliverable.
#
# The claim is exactness, so the gate is a finite difference of the production evaluator:
# fold the vertices over lattice shifts (the Γ-restricted picture `predict_energy` works
# in) and compare against a mixed central difference in one displacement component and one
# tangential spin direction. Two further properties are structural rather than numerical
# and are gated as such: the spin derivative is TANGENTIAL (`V·ê_b ≡ 0`, so no local-frame
# convention is smuggled in), and for a model satisfying the acoustic sum rule a rigid
# translation cannot change any spin derivative either.

using Test
using SLCE
using SLCE: build_asr, _lift_gamma
using LinearAlgebra
using Random

_mp_unit(rng) = normalize(randn(rng, 3))
_mp_cfg(rng, n) = reduce(hcat, [_mp_unit(rng) for _ = 1:n])

# A dimer chain whose bilinear exchange is dressed by a relative displacement — the
# magnetoelastic content a magnon–phonon vertex is made of — plus a pure-lattice
# `degree = 2` sector so the ASR constraint is nontrivial. `degree` selects which
# displacement degree the SPIN-carrying sector uses, which is how the "declared at
# degree 2 ⇒ no vertices" case is reached.
function _mp_model(rng; degree = 1, spin = true, j0 = 0.0)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    sectors = Sector[Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]
    spin && push!(sectors, Sector(spin = [1, 1], disp = (degree = degree,), sites = 1:2,
                                  cutoff = 1.1))
    basis = SLCEBasis(cr, BasisSpec(cr; lmax = spin ? 2 : 0, pmax = 2, sectors = sectors))
    rep = build_asr(basis)
    beta = rep === nothing ? randn(rng, n_salcs(basis)) :
           _lift_gamma(rep, randn(rng, size(rep.Z, 2)))
    return cr, basis, SLCEModel(basis, j0, beta)
end

# The Γ-restricted fold: `predict_energy` applies the same displacement in every cell, so
# every lattice shift lands on the same (a, b) pair.
function _mp_gamma(v, nat)
    G = [zeros(3, 3) for _ = 1:nat, _ = 1:nat]
    for ((a, b, _), T) in v.vertices
        G[a, b] .+= T
    end
    return G
end

@testset "magnon–phonon vertices" begin

    @testset "exact against a mixed finite difference" begin
        rng = MersenneTwister(0x6d70)
        _, _, model = _mp_model(rng)
        nat = 2
        e = _mp_cfg(rng, nat)
        v = magnon_phonon_vertices(model; spins = e)
        @test length(v) > 0
        @test v.spins == e
        G = _mp_gamma(v, nat)
        h, s = 1e-5, 1e-5
        for a = 1:nat, b = 1:nat, α = 1:3
            f = _mp_unit(rng)
            function E(du, ds)
                u = zeros(3, nat)
                u[α, a] = du
                ec = copy(e)
                ec[:, b] = normalize(e[:, b] + ds * f)
                return predict_energy(model, ec, u)
            end
            fd = (E(h, s) - E(h, -s) - E(-h, s) + E(-h, -s)) / (4h * s)
            exact = dot(G[a, b][α, :], f)
            @test fd ≈ exact atol = 1e-6 * max(1.0, abs(exact))
        end
    end

    @testset "the spin derivative is tangential, and the translation sum rule holds" begin
        rng = MersenneTwister(0x6d71)
        _, _, model = _mp_model(rng)
        nat = 2
        e = _mp_cfg(rng, nat)
        v = magnon_phonon_vertices(model; spins = e)
        # no local-frame convention is invented: the radial direction is projected out, so
        # any (f₁, f₂) the caller picks sees the same physics
        for ((_, b, _), T) in v.vertices
            @test norm(T * e[:, b]) < 1e-12 * max(1.0, norm(T))
        end
        # translating the crystal rigidly cannot change the energy, hence cannot change its
        # derivative with respect to any spin — this is the ASR one index further out
        @test asr_residual(model) < 1e-12
        for b = 1:nat
            S = zeros(3, 3)
            for ((_, bb, _), T) in v.vertices
                bb == b && (S .+= T)
            end
            @test norm(S) < 1e-10 * maximum(norm(T) for (_, T) in v.vertices)
        end
    end

    @testset "which content produces a vertex, and which does not" begin
        rng = MersenneTwister(0x6d72)
        # a magnetoelastic sector declared at degree 2 feeds the spin-dependent FORCE
        # CONSTANTS, not the vertices: a first derivative in u sees nothing there
        _, _, deg2 = _mp_model(rng; degree = 2)
        @test length(magnon_phonon_vertices(deg2; spins = _mp_cfg(rng, 2))) == 0
        @test length(force_constants(deg2; spins = _mp_cfg(rng, 2))) > 0
        # ...and a lattice-only model has no spin to differentiate at all
        _, _, latt = _mp_model(rng; spin = false)
        @test_throws ArgumentError magnon_phonon_vertices(latt)
    end

    @testset "error surface and show" begin
        rng = MersenneTwister(0x6d73)
        _, _, model = _mp_model(rng)
        # the spin state is not optional for a spin-carrying basis — the shared
        # `_resolve_spins` rule, same as the force constants and the strain path
        @test_throws ArgumentError magnon_phonon_vertices(model)
        @test_throws DimensionMismatch magnon_phonon_vertices(model; spins = zeros(3, 5))
        v = magnon_phonon_vertices(model; spins = _mp_cfg(rng, 2))
        @test occursin("MagnonPhononVertices", sprint(show, v))
        @test occursin("pairs", sprint(show, v))
    end
end
