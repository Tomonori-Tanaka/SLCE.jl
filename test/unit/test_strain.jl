# Homogeneous-strain response — `strain_derivatives`, the exact ε-derivatives of the
# cell energy at the model's own reference (design record §7's strain deliverable tier,
# §12 gate (t)).
#
# The acceptance gate is the Taylor identity against `affine_energy`: the strain
# derivatives are read off `solid_harmonic_poly`'s monomial coefficients, while
# `affine_energy` runs the production kernel `_eval_term_mixed`, so agreement is between
# two independent implementations of the same fact — and because a term of displacement
# degree `n` is EXACTLY a degree-`n` polynomial in `ε`, the identity is exact at finite
# strain, not asymptotic. That is what makes a wrong factor visible; a small-ε finite
# difference would hide one inside the truncation error.
#
# Gate (t) is the other half: an origin shift adds a uniform translation to the affine
# field, so the answer is origin-independent exactly when `Σ_i ∇_i E = 0`. The ASR is
# therefore enforced as a THROW, and both sides of that are pinned here.

using Test
using SLCE
using SLCE: build_asr, _lift_gamma
using LinearAlgebra
using Random

_strain_unit(rng) = normalize(randn(rng, 3))
_strain_cfg(rng, nat) = reduce(hcat, [_strain_unit(rng) for _ = 1:nat])

# A dimer chain with displacement content, and an ASR-feasible random coefficient
# vector for it. `degrees` selects which displacement degrees the basis carries, so a
# test can isolate the order-1 (magnetoelastic) content from the order-2 (elastic) one.
function _strain_model(rng; degrees = (1, 2), spin = true, seed_j0 = 0.31)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    sectors = Sector[]
    spin && push!(sectors, Sector(spin = (sites = 1:2,), cutoff = 1.1))
    for d in degrees
        push!(sectors, spin ?
              Sector(spin = [2], disp = (degree = d,), sites = 1:2, cutoff = 1.1) :
              Sector(disp = (degree = d,), sites = 1:2, cutoff = 1.1))
    end
    # `pmax` bounds the displacement capacity of the SPEC, and `_basis_has_disp` reads
    # the spec — so a fixture that means "no displacement content" must zero it, not
    # merely leave the disp sectors out.
    pmax = isempty(degrees) ? 0 : 2
    basis = SLCEBasis(cr, BasisSpec(cr; lmax = 2, pmax = pmax, sectors = sectors))
    rep = build_asr(basis)
    beta = rep === nothing ? randn(rng, n_salcs(basis)) :
           _lift_gamma(rep, randn(rng, size(rep.Z, 2)))
    return cr, basis, SLCEModel(basis, seed_j0, beta)
end

# The Taylor sum `E(0) + D¹:M + ½ D²:M:M` for an unsymmetrized (general affine map)
# pair of derivative tensors. Written out longhand — never through the implementation's
# own contraction — because the index layout `[α₁, β₁, α₂, β₂]` is part of the contract.
function _strain_taylor(E0, D1, D2, M)
    acc = E0
    for a = 1:3, b = 1:3
        acc += D1[a, b] * M[a, b]
    end
    for a = 1:3, b = 1:3, c = 1:3, d = 1:3
        acc += 0.5 * D2[a, b, c, d] * M[a, b] * M[c, d]
    end
    return acc
end

@testset "strain_derivatives ≡ the affine energy, exactly, at finite strain" begin
    rng = MersenneTwister(2718)
    cr, _, model = _strain_model(rng)
    e = _strain_cfg(rng, n_atoms(cr))
    D1 = strain_derivatives(model; spins = e, order = 1, symmetrize = false)
    D2 = strain_derivatives(model; spins = e, order = 2, symmetrize = false)
    @test size(D1) == (3, 3)
    @test size(D2) == (3, 3, 3, 3)
    E0 = affine_energy(model, e, zeros(3, 3))

    # The basis carries displacement degrees 1 and 2 and nothing higher, so the
    # two-term Taylor sum is the WHOLE energy — exact at t = 2, a 100%-scale affine
    # map, not merely at small t. Any missed factor, dropped image shift or mislabelled
    # index would show up here at once.
    M = 0.02 * randn(rng, 3, 3)
    for t in (0.25, 1.0, 2.0, -3.0)
        @test _strain_taylor(E0, D1, D2, t * M) ≈ affine_energy(model, e, t * M) atol =
            1e-12 * max(1.0, abs(E0))
    end

    # ... and the symmetrized tensors reproduce it on SYMMETRIC maps, which is the only
    # place `ε` is defined. (On an asymmetric map they must NOT: that difference is the
    # rotational content, `rotation_transfer_residual`'s subject.)
    S1 = strain_derivatives(model; spins = e, order = 1)
    S2 = strain_derivatives(model; spins = e, order = 2)
    eps = (M + M') / 2
    @test _strain_taylor(E0, S1, S2, eps) ≈ affine_energy(model, e, eps) atol = 1e-12
    @test S1 ≈ (D1 + D1') / 2
    @test !isapprox(S1, D1; atol = 1e-8)                # the fixture is not symmetric
    for a = 1:3, b = 1:3, c = 1:3, d = 1:3
        @test S2[a, b, c, d] ≈ S2[b, a, c, d]
        @test S2[a, b, c, d] ≈ S2[a, b, d, c]
        @test S2[a, b, c, d] ≈ S2[c, d, a, b]           # mixed partials, not imposed
    end
end

@testset "gate (t): the ASR earns origin independence, and is enforced" begin
    rng = MersenneTwister(31415)
    cr, basis, model = _strain_model(rng)
    e = _strain_cfg(rng, n_atoms(cr))
    @test asr_residual(model) < 1e-13

    ref = strain_derivatives(model; spins = e, order = 1)
    scale = norm(ref)
    @test scale > 1e-6
    for o in ([1.7, -0.4, 2.1], [-5.0, 3.0, 0.5], [11.0, 11.0, 11.0])
        @test norm(strain_derivatives(model; spins = e, order = 1, origin = o) - ref) <
              1e-11 * scale
    end
    ref2 = strain_derivatives(model; spins = e, order = 2)
    @test norm(strain_derivatives(model; spins = e, order = 2,
                                  origin = [2.5, -1.0, 0.75]) - ref2) < 1e-11 * norm(ref2)

    # An unconstrained model is refused, loudly and by name — never silently answered
    # with an origin-dependent number. The quantity is undefined for it, not inaccurate.
    bad = SLCEModel(basis, 0.0, randn(rng, n_salcs(basis)))
    @test asr_residual(bad) > 1e-6
    err = try
        strain_derivatives(bad; spins = e, order = 1)
        nothing
    catch ex
        ex
    end
    @test err isa ArgumentError
    @test occursin("acoustic sum rule", err.msg)
    @test occursin("asr_residual", err.msg)
    # ... and the reason it matters is visible on that same model: the affine energy
    # itself moves with the origin, which is the thing the throw is protecting against.
    eps = [0.02 0.01 0.0; 0.01 -0.03 0.0; 0.0 0.0 0.015]
    @test !isapprox(affine_energy(bad, e, eps),
                    affine_energy(bad, e, eps; origin = [3.0, 1.0, -2.0]); rtol = 1e-6)
end

@testset "order ≥ 2 needs the home-image gauge, and measures it" begin
    # THE FINDING. The ASR is `Σ_a ∂E/∂u_a ≡ 0` in the ATOM variables — an identity on
    # cell-periodic fields. At ε = 0 the affine field IS periodic, so order 1 is
    # origin-independent unconditionally. One order out it is not, and the atom-level
    # constraint no longer suffices: content anchored at an atom's home representative
    # cannot cancel against partners the clusters reach at a DIFFERENT image.
    #
    # The two crystals below are the same physical dimer chain. `1/3` puts atom 2 at
    # 1.0 Å — the bonded partner is the home representative; `2/3` puts it at 2.0 Å, so
    # the bond crosses the cell edge. They fit identical periodic data equally well.
    rng = MersenneTwister(8128)
    function _gauge_model(frac2, seed)
        cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 frac2; 0.0 0.0; 0.0 0.0],
                     [1, 1], ["Fe"])
        basis = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        rep = build_asr(basis)
        return SLCEModel(basis, 0.0,
                         _lift_gamma(rep, randn(MersenneTwister(seed),
                                                size(rep.Z, 2))))
    end
    good = _gauge_model(1 / 3, 4)
    bad = _gauge_model(2 / 3, 4)
    e = zeros(3, 2)
    @test asr_residual(good) < 1e-13 && asr_residual(bad) < 1e-13   # both conform

    # order 1: identical treatment, and the gauge cannot reach it
    @test strain_derivatives(good; spins = e, order = 1) ==
          strain_derivatives(good; spins = e, order = 1, origin = [4.0, 1.0, -2.0])
    @test strain_derivatives(bad; spins = e, order = 1) ==
          strain_derivatives(bad; spins = e, order = 1, origin = [4.0, 1.0, -2.0])

    # order 2: the consistent description answers, the split one refuses
    D2 = strain_derivatives(good; spins = e, order = 2)
    @test norm(D2) > 1e-6
    err = try
        strain_derivatives(bad; spins = e, order = 2)
        nothing
    catch ex
        ex
    end
    @test err isa ArgumentError
    @test occursin("origin-independent", err.msg)
    @test occursin("home-image gauge", err.msg)

    # ... and the size of what it refused: not a small correction, an O(1) different
    # answer. The escape hatch exists only to make that measurable.
    raw = strain_derivatives(bad; spins = e, order = 2, check_origin = false)
    shifted = strain_derivatives(bad; spins = e, order = 2, origin = [4.0, 1.0, -2.0],
                                 check_origin = false)
    @test norm(shifted - raw) > 0.1 * norm(raw)
    # the check is not vacuous on the good model either — it runs and passes
    @test strain_derivatives(good; spins = e, order = 2, check_origin = false) ≈ D2
end

@testset "strain_derivatives: only degree-n displacement content contributes" begin
    rng = MersenneTwister(999)

    # degree 2 only ⇒ no ε-linear response at all (the reference is stress-free in the
    # displacement channel), but a nonzero elastic one
    _, _, quad = _strain_model(MersenneTwister(5); degrees = (2,))
    e = _strain_cfg(rng, 2)
    @test strain_derivatives(quad; spins = e, order = 1) == zeros(3, 3)
    @test norm(strain_derivatives(quad; spins = e, order = 2)) > 1e-6

    # degree 1 only ⇒ a stress, and no clamped-ion elastic constants
    _, _, lin = _strain_model(MersenneTwister(5); degrees = (1,))
    @test norm(strain_derivatives(lin; spins = e, order = 1)) > 1e-6
    @test strain_derivatives(lin; spins = e, order = 2) == zeros(3, 3, 3, 3)

    # a pure-spin model has no affine response whatever the order
    _, sbasis, spinonly = _strain_model(MersenneTwister(5); degrees = ())
    @test !SLCE._basis_has_disp(sbasis)
    @test strain_derivatives(spinonly; spins = e, order = 1) == zeros(3, 3)
    @test strain_derivatives(spinonly; spins = e, order = 2) == zeros(3, 3, 3, 3)
end

@testset "strain_derivatives depends on the magnetic state — or says it does not" begin
    rng = MersenneTwister(24601)
    cr, _, model = _strain_model(rng; degrees = (1,))
    nat = n_atoms(cr)
    e1 = _strain_cfg(rng, nat)
    e2 = _strain_cfg(rng, nat)
    D1 = strain_derivatives(model; spins = e1, order = 1)
    D2 = strain_derivatives(model; spins = e2, order = 1)
    # the whole point of a spin–lattice expansion: the reference stress is a function of
    # the magnetic state, which is what magnetoelasticity IS
    @test norm(D1 - D2) > 1e-6 * max(norm(D1), norm(D2))

    # ... and a basis whose degree-1 displacement terms carry no spin factor is the
    # silent trap: identical for every state, warned about rather than left to be
    # discovered downstream
    blind = let cr2 = cr
        b = SLCEBasis(cr2, BasisSpec(cr2; lmax = 2, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(disp = (degree = 1,), sites = 1:2, cutoff = 1.1)]))
        rep = build_asr(b)
        SLCEModel(b, 0.0, _lift_gamma(rep, randn(rng, size(rep.Z, 2))))
    end
    @test_logs (:warn, r"does not depend on `spins`") strain_derivatives(blind;
                                                                        spins = e1)
    @test strain_derivatives(blind; spins = e1, order = 1) ==
          strain_derivatives(blind; spins = e2, order = 1)
end

@testset "strain_derivatives: error surface" begin
    rng = MersenneTwister(77)
    cr, _, model = _strain_model(rng)
    e = _strain_cfg(rng, n_atoms(cr))
    @test_throws ArgumentError strain_derivatives(model; spins = e, order = 0)
    @test_throws ArgumentError strain_derivatives(model; spins = e, order = -1)
    @test_throws DimensionMismatch strain_derivatives(model; spins = e,
                                                      origin = [1.0, 2.0])
    @test_throws ArgumentError strain_derivatives(model; spins = e,
                                                  origin = [NaN, 0.0, 0.0])
    @test_throws DimensionMismatch strain_derivatives(model; spins = zeros(3, 5))
    # a spin-carrying basis refuses the omitted state, exactly as `force_constants` does
    err = try
        strain_derivatives(model)
        nothing
    catch ex
        ex
    end
    @test err isa ArgumentError
    @test occursin("`spins` is required", err.msg)

    # a lattice-only model may omit it, and the answer is the same one it gets by
    # passing the zeros marker explicitly
    _, _, latt = _strain_model(MersenneTwister(5); degrees = (1, 2), spin = false)
    @test strain_derivatives(latt; order = 1) ==
          strain_derivatives(latt; spins = zeros(3, 2), order = 1)
end
