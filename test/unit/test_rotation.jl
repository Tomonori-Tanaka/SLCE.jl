# Rotational invariance — the affine-field evaluation path (`affine_energy`) and the
# two diagnostics built on it: gate (q), a rigid rotation of the lattice at FINITE ω,
# and gate (r), the SOC rotation law `𝓡_U E = −𝓡_S E` as a two-sided identity.
#
# Neither is a constraint: the package imposes translation invariance (the ASR) and
# nothing else affine. What these gates pin is that the diagnostics measure what they
# claim to — that the affine path reduces exactly to the periodic one, that a
# linearized rotation is blind where the finite one is not, that a rotationally
# invariant potential fits to a near-invariant model, and that a random
# ASR-feasible model does not.

using Test
using SLCE
using SLCE: build_asr, salcs, disp_degree
using LinearAlgebra
using Random

_rot_unit(rng) = normalize(randn(rng, 3))
_rot_cfg(rng, nat) = reduce(hcat, [_rot_unit(rng) for _ = 1:nat])

# The generator of a rotation about ẑ, and the finite rotation itself. Written here
# independently of the implementation's own `_rotation_generator` — the point of the
# gates below is to compare against hand-derived values, never against the
# evaluator's.
const _WZ = [0.0 -1.0 0.0; 1.0 0.0 0.0; 0.0 0.0 0.0]
_rotz(om) = Matrix(1.0I, 3, 3) + sin(om) * _WZ + (1 - cos(om)) * (_WZ * _WZ)

# A dimer chain: two atoms 1.0 Å apart along x̂ in a 3.0 Å cubic cell, so that with a
# 1.1 Å cutoff each cell carries exactly one bond. `frac2` selects WHICH periodic
# image of atom 2 is the home-cell representative — 1/3 puts it at 1.0 Å (the bonded
# one), 2/3 at 2.0 Å (the same crystal, the same periodic training data, a different
# home image). The two are not interchangeable for an affine field; see the
# image-gauge testset.
function _dimer_basis(frac2, degree; sites = 1:2)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 frac2; 0.0 0.0; 0.0 0.0],
                 [1, 1], ["Fe"])
    spec = BasisSpec(cr; lmax = 1, pmax = 2,
                     sectors = [Sector(disp = (degree = degree,), sites = sites,
                                       cutoff = 1.1)])
    return cr, SLCEBasis(cr, spec)
end

@testset "affine_energy reduces to the periodic evaluator" begin
    rng = MersenneTwister(4242)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    basis = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)]))
    m = n_salcs(basis)
    nat = n_atoms(cr)
    model = SLCEModel(basis, 0.37, randn(rng, m))

    # `M = 0` makes the affine field the periodic one, and the two paths then run the
    # same kernel over the same members in the same order — so this is an EXACT
    # identity, not an approximate one. It is the gate that keeps the per-site
    # re-indexing in `affine_energy` from drifting into a second evaluator.
    for _ = 1:5
        e = _rot_cfg(rng, nat)
        u = 0.05 * randn(rng, 3, nat)
        @test affine_energy(model, e, zeros(3, 3); base = u) ===
              predict_energy(model, e, u)
    end
    e = _rot_cfg(rng, nat)
    @test affine_energy(model, e, zeros(3, 3)) === predict_energy(model, e,
                                                                  zeros(3, nat))
    # at M = 0 the origin multiplies zero, so it cannot enter
    @test affine_energy(model, e, zeros(3, 3); origin = [1.7, -0.3, 2.2]) ===
          predict_energy(model, e, zeros(3, nat))

    # error surface
    @test_throws DimensionMismatch affine_energy(model, e, zeros(2, 3))
    @test_throws ArgumentError affine_energy(model, e, fill(NaN, 3, 3))
    @test_throws DimensionMismatch affine_energy(model, e, zeros(3, 3);
                                                 origin = [1.0, 2.0])
    @test_throws ArgumentError affine_energy(model, e, zeros(3, 3);
                                             origin = [Inf, 0.0, 0.0])
    @test_throws ArgumentError rotational_residual(model, e; axis = [0.0, 0.0, 0.0])
    @test_throws DimensionMismatch rotational_residual(model, e; axis = (1.0, 0.0))
    @test_throws ArgumentError rotational_residual(model, e; omega = NaN)
    # a spin-carrying basis still refuses the `nothing` placeholder
    @test_throws ArgumentError affine_energy(model, nothing, zeros(3, 3))
end

@testset "the ASR earns origin independence of the affine field" begin
    rng = MersenneTwister(99)
    cr, basis = _dimer_basis(1 / 3, 1:2)
    m = n_salcs(basis)
    rep = build_asr(basis)
    M = 0.03 * [0.4 -1.0 0.2; 1.0 0.1 -0.5; -0.2 0.5 0.3]   # no symmetry assumed

    # A feasible (translation-invariant) model: shifting the rotation/strain centre is
    # a rigid translation of the whole field, which such a model cannot see.
    feas = SLCEModel(basis, 0.0, rep.Z * (rep.Z' * randn(rng, m)))
    @test asr_residual(feas) < 1e-13
    a = affine_energy(feas, nothing, M)
    b = affine_energy(feas, nothing, M; origin = [1.3, -0.4, 0.9])
    @test abs(a - b) <= 1e-11 * max(abs(a), 1.0)

    # An unconstrained model of the same basis: the same shift MOVES the answer. This
    # is what makes the test above an assertion about the ASR rather than about the
    # arithmetic.
    viol = SLCEModel(basis, 0.0, randn(rng, m))
    @test asr_residual(viol) > 1e-8
    av = affine_energy(viol, nothing, M)
    bv = affine_energy(viol, nothing, M; origin = [1.3, -0.4, 0.9])
    @test abs(av - bv) > 1e-3 * max(abs(av), 1.0)
end

@testset "gate (q): the linearized rotation is blind, the finite one is not" begin
    # A model whose only displacement content is degree 1 — a pure force field — with
    # the forces along the chain axis. Then, EXACTLY:
    #
    #   ΔE(M) = Σ_a (∂E/∂u_a)·(M R_a) = λ x̂·M(R₁ − R₂) = −λ (x̂ᵀ M x̂),
    #
    # because R₁ = 0 and R₂ = 1.0 x̂ and ∂E/∂u₂ = −∂E/∂u₁ (the ASR). For the
    # linearized rotation M = ωW that is x̂ᵀWx̂ = 0 — identically zero at every ω,
    # however non-invariant the model is. For the true rotation it is 1 − cos ω.
    rng = MersenneTwister(1234)
    cr, basis = _dimer_basis(1 / 3, 1:2)
    ss = salcs(basis)
    rep = build_asr(basis)
    lin = [j for j in eachindex(ss) if sum(disp_degree, ss[j].decors) == 1]
    @test length(lin) == 6                       # 3 components × 2 atoms
    # The ASR does not mix displacement degrees, so the feasible directions of the
    # degree-1 stratum are the null space of A restricted to those columns.
    Z1 = nullspace(rep.A[:, lin])
    @test size(Z1, 2) == 3                       # the three difference directions

    function _linmodel(c)
        beta = zeros(n_salcs(basis))
        beta[lin] .= Z1 * c
        return SLCEModel(basis, 0.0, beta)
    end
    # Solve for the coefficient vector whose force field is x̂ on atom 1.
    G = reduce(hcat, [-predict_force(_linmodel(Matrix(1.0I, 3, 3)[:, k]), nothing,
                                     zeros(3, 2))[:, 1] for k = 1:3])
    model = _linmodel(G \ [1.0, 0.0, 0.0])
    dEdu = -predict_force(model, nothing, zeros(3, 2))
    @test dEdu[:, 1] ≈ [1.0, 0.0, 0.0] atol = 1e-12
    @test dEdu[:, 2] ≈ [-1.0, 0.0, 0.0] atol = 1e-12          # the ASR supplies this
    @test asr_residual(model) < 1e-13

    E0 = affine_energy(model, nothing, zeros(3, 3))
    for om in (0.4, 0.2, 0.1, 0.05)
        dlin = affine_energy(model, nothing, om * _WZ) - E0
        dfin = affine_energy(model, nothing, _rotz(om) - I) - E0
        @test abs(dlin) < 1e-14                  # the linearized test is blind: ≡ 0
        @test dfin ≈ 1 - cos(om) rtol = 1e-11    # hand-derived, not the evaluator's
        @test rotational_residual(model, nothing; omega = om) > 0.01
    end
end

@testset "gate (q): a central potential fits to a near-invariant model" begin
    # The positive control the diagnostic needs to be falsifiable. A central pair
    # potential V(|r|) is exactly rotationally invariant, so a model fitted to it can
    # violate invariance only through its own truncation — and the signature of that
    # is a residual that VANISHES with ω, where a generic model's tends to a nonzero
    # constant. Both models live in the same basis and satisfy the same ASR.
    V(x) = 0.35 * (x - 1.0)^2 - 0.35 * (x - 1.0)
    Vp(x) = 0.7 * (x - 1.0) - 0.35            # V′(1) ≠ 0: a stressed bond
    d0 = [1.0, 0.0, 0.0]

    function _central_fit(cr, basis, bond, rng)
        ener(u) = V(norm(bond + u[:, 2] - u[:, 1]))
        function forc(u)
            r = bond + u[:, 2] - u[:, 1]
            g = Vp(norm(r)) * (r / norm(r))
            return hcat(g, -g)
        end
        data = [begin
                    u = 0.005 * randn(rng, 3, 2)
                    lattice_datum(ener(u); displacements = u, forces = forc(u),
                                  reference = cr)
                end for _ = 1:400]
        return fit(SLCEFit, SLCEDataset(basis, data), OLS(); force_weight = 1.0)
    end

    rng = MersenneTwister(2718)
    cr, basis = _dimer_basis(1 / 3, 1:2)
    f = _central_fit(cr, basis, d0, rng)
    @test r2_energy(f) > 1 - 1e-6
    central = SLCEModel(f)
    @test asr_residual(central) < 1e-12
    rep = build_asr(basis)
    generic = SLCEModel(basis, 0.0, rep.Z * (rep.Z' * randn(rng, n_salcs(basis))))

    rc = [rotational_residual(central, nothing; omega = om) for om in (0.1, 0.0125)]
    rg = [rotational_residual(generic, nothing; omega = om) for om in (0.1, 0.0125)]
    @test all(<(1e-2), rc)                    # invariant up to the truncation floor
    @test all(>(0.5), rg)                     # a generic feasible model is not
    @test rc[2] < 0.5 * rc[1]                 # and the floor closes as ω → 0
    @test rg[2] > 0.8 * rg[1]                 # while a real violation does not

    # The image gauge. The SAME physical crystal described with atom 2's home
    # representative at 2.0 Å instead of 1.0 Å fits the SAME periodic data equally
    # well — and is NOT rotationally invariant, because the on-site force content is
    # then attached to an atom that is not the one the bond acts on. Periodic
    # training data cannot distinguish the two models; this diagnostic can.
    rng2 = MersenneTwister(2718)
    cr2, basis2 = _dimer_basis(2 / 3, 1:2)
    f2 = _central_fit(cr2, basis2, -d0, rng2)
    @test abs(r2_energy(f2) - r2_energy(f)) < 1e-6      # an equally good fit ...
    shifted = SLCEModel(f2)
    @test rotational_residual(shifted, nothing; omega = 0.05) >
          50 * rotational_residual(central, nothing; omega = 0.05)   # ... not equally
                                                                     # invariant
end

@testset "gate (r): the SOC rotation law as a two-sided identity" begin
    # A pseudo-dipolar dimer, E = A·(ê₁·r̂)(ê₂·r̂): invariant under a rotation of the
    # spins and the lattice TOGETHER and under neither alone. A model fitted to it
    # must therefore show a LARGE lattice-only residual and a vanishing joint one —
    # which is `𝓡_U E = −𝓡_S E` with nothing else mixed in. The O_h fixtures
    # elsewhere in the suite are blind to this: cubic symmetry makes both sides
    # vanish for a reason that does not generalize.
    rng = MersenneTwister(31)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 1/3; 0.0 0.0; 0.0 0.0],
                 [1, 1], ["Fe"])
    bond = [1.0, 0.0, 0.0]
    basis = SLCEBasis(cr, BasisSpec(cr; lmax = 2, pmax = 2, sectors = [
        Sector(spin = [1, 1], disp = (degree = 0:2,), sites = 2, cutoff = 1.1)]))
    fp = crystal_fingerprint(cr)
    function datum(e, u)
        r = bond + u[:, 2] - u[:, 1]
        rh = r / norm(r)
        return TrainingDatum(; energy = 0.4 * dot(e[:, 1], rh) * dot(e[:, 2], rh),
                             directions = e, magmoms = ones(2), displacements = u,
                             provenance = DatumProvenance(; reference_id = "r",
                                                          reference_fingerprint = fp,
                                                          setup_id = "s"))
    end
    data = [datum(_rot_cfg(rng, 2), 0.01 * randn(rng, 3, 2)) for _ = 1:800]
    f = fit(SLCEFit, SLCEDataset(basis, data; use_torque = false, use_force = false),
            OLS())
    @test r2_energy(f) > 1 - 1e-8
    model = SLCEModel(f)
    @test asr_residual(model) < 1e-12
    generic = SLCEModel(basis, 0.0,
                        let rep = build_asr(basis)
                            rep.Z * (rep.Z' * randn(rng, n_salcs(basis)))
                        end)

    e0 = _rot_cfg(rng, 2)
    for om in (0.05, 0.025)
        # the lattice half alone is NOT small — in a SOC sector zero is the wrong
        # expectation for it, and reading it as a violation is the trap gate (r) exists
        # to close
        @test rotational_residual(model, e0; omega = om) > 1.0
        # ... yet the two halves cancel
        @test rotation_transfer_residual(model, e0; omega = om) < 1e-2
        # while a random ASR-feasible model of the same basis shows no cancellation
        @test rotation_transfer_residual(generic, e0; omega = om) > 0.5
    end
    # the cancellation is the ω → 0 limit of a truncated expansion, so it improves
    @test rotation_transfer_residual(model, e0; omega = 0.025) <
          0.5 * rotation_transfer_residual(model, e0; omega = 0.05)

    # `u0` moves the reference configuration the rotation acts on; a rigid rotation of
    # an ALREADY displaced crystal is still a rigid rotation, so the identity holds
    # there too (this is what makes `u0` more than an unused knob).
    @test rotation_transfer_residual(model, e0; omega = 0.05,
                                     u0 = 0.004 * randn(rng, 3, 2)) < 5e-2
end

@testset "no displacement content: the documented zero convention" begin
    rng = MersenneTwister(7)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0],
                 [1, 1], ["Fe"])
    basis = SLCEBasis(cr, BasisSpec(cr; lmax = 2, pmax = 0,
                                    sectors = [Sector(spin = (sites = 1:2,),
                                                      cutoff = 1.1)]))
    model = SLCEModel(basis, 0.1, randn(rng, n_salcs(basis)))
    e = _rot_cfg(rng, 2)
    # no affine response at all — neither the rotation nor the reference scale moves
    @test rotational_residual(model, e; omega = 0.1) == 0.0
    # the spin half is still there, and it has nothing to cancel against
    @test rotation_transfer_residual(model, e; omega = 0.1) == 1.0
end
