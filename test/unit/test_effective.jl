# Effective models at a displaced reference (src/slce/effective.jl) — M5-1, design
# record §9d. The claim is an EXACT change of expansion point, so every gate here is
# against the production evaluator: the identity `E_eff(δu) ≡ E(u0 + δu)` at small and
# large `u0`, the analytic force at `u0` reproduced by the re-expanded surface, the
# degree structure the shift is supposed to have (lower triangular, and generating
# degree 0/1 that the reference expansion does not carry), and the refusal surface.
#
# Note on how "1e-13 relative" is measured. A random spin configuration can sit at a
# zero crossing of the energy, where NO method has pointwise relative accuracy — on
# this fixture |E| ranges over four orders of magnitude and the pointwise ratio reaches
# 7e-13 while the absolute error never exceeds 6e-14. The honest measure is the error
# against the ENERGY SCALE of the sample, which is what these gates use.

using Test
using SLCE
using SLCE: salcs
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_em_cfg(rng, n) = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:n])

# total δu degree of one effective term
_em_deg(t) = sum(sum(p) for (_, p) in t.disps; init = 0)
_em_degrees(em) = sort(unique(_em_deg(t) for t in em.terms))

# worst |E_eff(δu) − E(u0 + δu)| over random samples, together with the energy scale
# it should be judged against
function _em_mismatch(em, model, u0, rng, nat; ntrials = 400, du = 0.05)
    worst, scale = 0.0, 0.0
    for _ = 1:ntrials
        e = _em_cfg(rng, nat)
        d = du .* randn(rng, 3, nat)
        ref = predict_energy(model, e, u0 .+ d)
        worst = max(worst, abs(predict_energy(em, e, d) - ref))
        scale = max(scale, abs(ref))
    end
    return worst / scale
end

@testset "effective models at a displaced reference" begin
    rng = MersenneTwister(5)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    nat = 2
    # pure-spin + coupled + lattice-only, so the re-expansion has to carry spin-dressed
    # AND bare displacement content through the shift
    b = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (nbody = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1),
        Sector(disp = (degree = 2,), nbody = 1:2, cutoff = 1.1)]))
    model = SLCEModel(b, 0.3, randn(rng, n_salcs(b)))
    u0 = 0.07 .* randn(rng, 3, nat)
    em = effective_model(model; u0)

    @testset "the exactness identity E_eff(δu) ≡ E(u0 + δu)" begin
        @test _em_mismatch(em, model, u0, rng, nat) < 1e-13
        # The map is exact, not perturbative: a displacement comparable to a bond
        # length and an increment half as large again must agree to the same roundoff.
        big = 0.9 .* randn(rng, 3, nat)
        embig = effective_model(model; u0 = big)
        @test _em_mismatch(embig, model, big, rng, nat; du = 0.4) < 1e-13
        # u0 = 0 is the original surface back again
        em0 = effective_model(model; u0 = zeros(3, nat))
        @test _em_mismatch(em0, model, zeros(3, nat), rng, nat) < 1e-13
    end

    @testset "δu = 0 is the u0-frozen point, not the clamped-ion one" begin
        # Design record §9d (ii): this difference is the whole purpose of the utility.
        # If they agreed, the re-expansion would have generated no degree-0 content and
        # would not be a model of the displaced structure at all.
        e = _em_cfg(rng, nat)
        z = zeros(3, nat)
        @test isapprox(predict_energy(em, e, z), predict_energy(model, e, u0);
                       rtol = 1e-13)
        @test abs(predict_energy(em, e, z) - predict_energy(model, e, z)) > 1e-3
    end

    @testset "the re-expansion reproduces the analytic force at u0" begin
        # The degree-1 content of the effective model IS the reference force of the
        # displaced structure. Compare a central difference of the RE-EXPANDED surface
        # against the model's own analytic `predict_force` at u0 — two independent code
        # paths through two different expansion points.
        e = _em_cfg(rng, nat)
        h = 1e-5
        F = zeros(3, nat)
        for a = 1:nat, α = 1:3
            d = zeros(3, nat)
            d[α, a] = h
            F[α, a] = -(predict_energy(em, e, d) - predict_energy(em, e, -d)) / (2h)
        end
        @test isapprox(F, predict_force(model, e, u0); atol = 1e-8)
        # and it is a real force, not a coincidence of zeros
        @test maximum(abs, F) > 1e-3
    end

    @testset "degree structure: lower triangular, and degree 1 is the u0 signature" begin
        em0 = effective_model(model; u0 = zeros(3, nat))
        # The reference expansion of this fixture carries displacement degrees 0 and 2
        # only (degree 0 = its pure-spin sectors). Shifting generates degree 1 and more
        # degree 0 — and generates NOTHING above the original maximum, which is the
        # lower-triangular claim.
        @test _em_degrees(em0) == [0, 2]
        @test _em_degrees(em) == [0, 1, 2]
        @test maximum(_em_degrees(em)) == maximum(_em_degrees(em0))
        # The generated degree-0 content does NOT add spin-only terms — it lands on
        # the keys that already exist and RENORMALIZES their coefficients. That
        # renormalization is exactly the content a frozen-structure (and later a
        # thermally renormalized) spin model is supposed to carry, so assert it on the
        # coefficients rather than on a count.
        spin_only(x) = Dict(t.spins => t.coef for t in x.terms if isempty(t.disps))
        @test keys(spin_only(em)) == keys(spin_only(em0))
        @test any(!isapprox(spin_only(em)[k], spin_only(em0)[k]; rtol = 1e-8)
                  for k in keys(spin_only(em)))
        # the spin-free part of that constant lands in j0 — and at u0 = 0 nothing is
        # generated at all, because every displacement factor is homogeneous
        @test em0.j0 == model.j0
        @test abs(em.j0 - model.j0) > 1e-6
    end

    @testset "terms are canonical, sorted and reproducible" begin
        em2 = effective_model(model; u0)
        @test length(em2) == length(em)
        @test [(t.coef, t.spins, t.disps) for t in em2.terms] ==
              [(t.coef, t.spins, t.disps) for t in em.terms]
        @test issorted(em.terms; by = t -> (t.disps, t.spins))
        # one entry per (spins, monomial): no duplicate keys survive the accumulation
        @test length(unique((t.spins, t.disps) for t in em.terms)) == length(em.terms)
        # a stored monomial never carries the zero exponent triple, and never repeats
        # an atom (the per-atom merge)
        for t in em.terms
            @test all(p -> p != (0, 0, 0), (p for (_, p) in t.disps))
            @test allunique(a for (a, _) in t.disps)
            @test issorted([a for (a, _) in t.disps])
        end
    end

    @testset "two displacement slots on the SAME atom merge into one variable" begin
        # An AllImages self-bond on a one-atom cell puts both slots of a pair term on
        # atom 1. Displacements are cell-periodic, so the two are the same variable and
        # their exponents must add; this is the only fixture that exercises the merge.
        lat1 = Lattice(Matrix(2.0 * I(3)))
        xt1 = Crystal(lat1, zeros(3, 1), [1], ["Fe"])
        b1 = SLCEBasis(xt1, BasisSpec(xt1; lmax = 1, pmax = 1, sectors = [
                Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 2.1)]);
            images = AllImages())
        m1 = SLCEModel(b1, 0.0, randn(rng, n_salcs(b1)))
        u01 = 0.11 .* randn(rng, 3, 1)
        em1 = effective_model(m1; u0 = u01)
        @test _em_mismatch(em1, m1, u01, rng, 1; du = 0.06) < 1e-13
        # both slots landed on atom 1, so a degree-2 term is one atom carrying
        # exponent sum 2 — never two entries of exponent sum 1
        @test 2 in _em_degrees(em1)
        for t in em1.terms
            @test length(t.disps) <= 1
        end
    end

    @testset "atol prunes, and says so by breaking exactness" begin
        # Pick the cut from the data (the median coefficient magnitude) rather than a
        # magic number, so the gate keeps meaning if the fixture's scale changes.
        mags = sort!([abs(t.coef) for t in em.terms])
        emc = effective_model(model; u0, atol = mags[div(end, 2)])
        @test length(emc) < length(em)
        # still close, but no longer the identity — the docstring's warning, gated
        @test _em_mismatch(emc, model, u0, rng, nat) > 1e-13
        # Loose on purpose, and it only claims the pruned model is still on the scale
        # of the energy rather than divergent: dropping the smaller half of the
        # coefficients has no tight a-priori error bound, so a tighter number here
        # would be tuned to the fixture rather than derived.
        @test _em_mismatch(emc, model, u0, rng, nat) < 1.0
        @test length(effective_model(model; u0, atol = 0.0)) == length(em)
    end

    @testset "the refusal surface" begin
        # a strain is not a displacement field: the affine pattern is not cell-periodic
        @test_throws ArgumentError effective_model(model; u0, strain = 0.01 * I(3))
        # a pure-spin model has nothing to re-expand
        bs = SLCEBasis(cr, BasisSpec(cr; lmax = 2, nbody = 2, cutoff = 1.1))
        ms = SLCEModel(bs, 0.0, randn(rng, n_salcs(bs)))
        @test_throws ArgumentError effective_model(ms; u0)
        @test_throws DimensionMismatch effective_model(model; u0 = zeros(3, nat + 1))
        @test_throws DimensionMismatch effective_model(model; u0 = zeros(2, nat))
        @test_throws ArgumentError effective_model(model; u0 = fill(NaN, 3, nat))
        @test_throws ArgumentError effective_model(model; u0, atol = -1.0)
        @test_throws ArgumentError effective_model(model; u0, atol = Inf)
        # evaluation validates its own arguments
        e = _em_cfg(rng, nat)
        @test_throws DimensionMismatch predict_energy(em, zeros(3, nat + 1),
                                                      zeros(3, nat + 1))
        @test_throws ArgumentError predict_energy(em, e, zeros(3, nat + 1))
    end

    @testset "show reports the shape a reader needs" begin
        s = sprint(show, em)
        @test occursin("EffectiveModel", s)
        @test occursin(string(length(em)), s)
        @test occursin("2 atoms", s)
    end
end
