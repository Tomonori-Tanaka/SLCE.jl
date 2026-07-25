# The independent counting oracle: slot validation, the cycle-wise character
# formula vs explicit projector ranks (the gate-(g) infrastructure), the
# permuting-pair swap-sign regression, the axial-vs-polar kill-shot, and the
# gate-(o) representation pins + mutation teeth against `rep_scale`.

include("../support/countingoracle.jl")

using .CountingOracle
using SLCE
using SLCE: SolidHarmonics, rep_scale
using LinearAlgebra: norm, rank, I, cross, dot, normalize, det
using Random: MersenneTwister
using StaticArrays

# All 48 signed permutation matrices = the O_h point group in Cartesian axes.
function _oh_matrices()
    mats = Matrix{Float64}[]
    for p in [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        for sx in (-1, 1), sy in (-1, 1), sz in (-1, 1)
            M = zeros(3, 3)
            signs = (sx, sy, sz)
            for i = 1:3
                M[i, p[i]] = signs[i]
            end
            push!(mats, M)
        end
    end
    return mats
end

# The 24 proper rotations of the octahedral group.
_o_matrices() = [M for M in _oh_matrices() if det(M) > 0]

# O_h operations that map the given site list onto itself.
_site_ops(sites) = [cluster_op(M, sites) for M in _oh_matrices()
                    if all(any(norm(M * sites[i] - sites[j]) < 1e-8
                               for j = 1:length(sites)) for i = 1:length(sites))]

_rz(θ) = [cos(θ) -sin(θ) 0.0; sin(θ) cos(θ) 0.0; 0.0 0.0 1.0]

@testset "CountingOracle" begin
    rng = MersenneTwister(0xc0de)
    origin = [SVector(0.0, 0.0, 0.0)]

    @testset "slot validation" begin
        @test_throws ArgumentError SpinSlot(0, 1)
        @test_throws ArgumentError SpinSlot(1, 0)          # spin l = 0 unconstructable
        @test_throws ArgumentError DispSlot(1, 0, 0)       # degree 2k + L ≥ 1
        @test_throws ArgumentError DispSlot(1, -1, 1)
        @test DispSlot(1, 1, 0).k == 1                     # |u|² trace channel is legal
        @test_throws ArgumentError DispSymSlot(1, 0)
        @test_throws ArgumentError SpinPairSlot(1, 1, 1, 1, 0)  # site_a == site_b
        @test_throws ArgumentError SpinPairSlot(1, 2, 1, 1, 3)  # triangle rule
        @test_throws ArgumentError SpinPairSlot(1, 2, 0, 1, 1)
        # Repeated same-(site, channel) slots are rejected at every entry point.
        ops = [identity_op(1)]
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 1), SpinSlot(1, 2)]
        @test_throws ArgumentError count_invariants(slots, ops)
        slots = CountingOracle.AbstractSlot[DispSlot(1, 0, 1), DispSlot(1, 0, 2)]
        @test_throws ArgumentError representation_matrix(slots, ops[1])
        slots = CountingOracle.AbstractSlot[DispSlot(1, 0, 1), DispSymSlot(1, 2)]
        @test_throws ArgumentError stabilizer_ops(slots, ops)
        # A spin factor AND a displacement factor on one site is legal.
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 2), DispSlot(1, 0, 2)]
        @test count_invariants(slots, ops) == (2 * 2 + 1) * (2 * 2 + 1)
    end

    @testset "polynomial layer matches the production kernel" begin
        for _ = 1:3
            u = randn(rng, 3) * 0.9
            for l = 0:5, m = -l:l
                q = solid_harmonic_polynomial(l, m)
                @test q(u) ≈ SolidHarmonics.Rlm(l, m, u) atol = 1e-12 rtol = 1e-10
            end
        end
        # Gate (h): harmonic-block dimension identities Sym² = 5 + 1, Sym³ = 7 + 3.
        b2 = harmonic_blocks(2)
        @test [(b.k, b.L, length(b.polys)) for b in b2] == [(0, 2, 5), (1, 0, 1)]
        b3 = harmonic_blocks(3)
        @test [(b.k, b.L, length(b.polys)) for b in b3] == [(0, 3, 7), (1, 1, 3)]
        @test sym_dimension(2) == 6 && sym_dimension(3) == 10
        # Plethysm character at the identity is the Sym^p dimension.
        for p = 1:4
            @test CountingOracle._chi_sym_power(p, Matrix(1.0I, 3, 3)) ≈
                  sym_dimension(p) atol = 1e-10
        end
        # Defensive branches of the polynomial layer.
        q1 = solid_harmonic_polynomial(1, 0)
        q2 = solid_harmonic_polynomial(2, 0)
        @test_throws ArgumentError q1 + q2
        @test_throws ArgumentError q1 - q2
        @test_throws ArgumentError CountingOracle._compose_linear(q1, zeros(2, 2))
        @test_throws ArgumentError HomogeneousPolynomial(1, [1.0])
        # Explicit call: a literal `q^-1` would lower through literal_pow/inv.
        @test_throws ArgumentError Base.:^(q1, -1)
    end

    @testset "group closure guard" begin
        oh = [cluster_op(M, origin) for M in _oh_matrices()]
        assert_group_closure(oh)   # closed: no throw
        @test_throws ArgumentError assert_group_closure(oh[2:end])  # identity dropped
        broken = [cluster_op(_rz(2π / 3), origin)]  # C3z alone: no identity
        @test_throws ArgumentError assert_group_closure(broken)
        # Non-integer character sum from a non-group list is caught.
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 2)]
        notagroup = [identity_op(1), cluster_op(_rz(0.7), origin)]
        @test_throws ArgumentError count_invariants(slots, notagroup)
        # The same non-group average is asymmetric — invariant_basis's symmetry
        # guard catches it.
        @test_throws ErrorException invariant_basis(slots, notagroup)
        # {E, C4z, C4z⁻¹} (missing C2z) averages to a symmetric but
        # non-idempotent matrix (eigenvalues 1, 1/3, 1/3 on l = 1): this hits
        # the eigenvalue-neither-0-nor-1 guard specifically.
        slots1 = CountingOracle.AbstractSlot[SpinSlot(1, 1)]
        c4open = [identity_op(1), cluster_op(_rz(π / 2), origin),
                  cluster_op(_rz(3π / 2), origin)]
        @test_throws ErrorException invariant_basis(slots1, c4open)
    end

    @testset "gate (e): cubic single-site l=2 × p=2 count" begin
        oh = [cluster_op(M, origin) for M in _oh_matrices()]
        # Traceless (k=0, L=2) block: E_g⊗E_g ⊕ T_2g⊗T_2g ⇒ exactly 2 invariants.
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 2), DispSlot(1, 0, 2)]
        @test count_invariants(slots, oh) == 2
        @test rank(invariant_projector(slots, oh), atol = 1e-6) == 2
        @test size(invariant_basis(slots, oh), 2) == 2
        # |u|² trace channel: rank-2 spin ⊗ scalar ⇒ no invariant.
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 2), DispSlot(1, 1, 0)]
        @test count_invariants(slots, oh) == 0
        @test rank(invariant_projector(slots, oh), atol = 1e-6) == 0
        # Full Sym² slot = the two blocks together.
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 2), DispSymSlot(1, 2)]
        @test count_invariants(slots, oh) == 2
        @test rank(invariant_projector(slots, oh), atol = 1e-6) == 2
    end

    @testset "gate (g) kill-shot: axial spin × polar displacement" begin
        # Single site with inversion in the stabilizer: ê is axial (inversion-even),
        # u is polar (inversion-odd), so ê·u is killed — 0 invariants. Σl_spin is
        # odd here: this decoration is reachable only by explicit oracle slots
        # (production's T-parity filter never enumerates it).
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 1), DispSlot(1, 0, 1)]
        oh = [cluster_op(M, origin) for M in _oh_matrices()]
        @test count_invariants(slots, oh) == 0
        @test rank(invariant_projector(slots, oh), atol = 1e-6) == 0
        # Under the proper rotations alone, ê·u survives — the kill is inversion's.
        o24 = [cluster_op(M, origin) for M in _o_matrices()]
        @test count_invariants(slots, o24) == 1
        @test rank(invariant_projector(slots, o24), atol = 1e-6) == 1
    end

    @testset "no-SOC averaging via spin_only_op" begin
        o_spin = [spin_only_op(M, 1) for M in _o_matrices()]
        assert_group_closure(o_spin)
        for l = 1:3
            @test count_invariants(CountingOracle.AbstractSlot[SpinSlot(1, l)],
                o_spin) == 0
        end
        # The octahedral rotation group first admits an invariant at l = 4.
        @test count_invariants(CountingOracle.AbstractSlot[SpinSlot(1, 4)], o_spin) == 1
    end

    # Bond cluster ±x: the site-permuting case where the naive per-slot character
    # product is wrong and the cycle formula is required.
    @testset "permuting spin pair on a bond: cycle formula vs naive product" begin
        sites = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
        d4h = _site_ops(sites)
        @test length(d4h) == 16
        assert_group_closure(d4h)
        slots = CountingOracle.AbstractSlot[SpinSlot(1, 1), SpinSlot(2, 1)]
        n = count_invariants(slots, d4h)
        @test n == rank(invariant_projector(slots, d4h), atol = 1e-6)
        # Naive per-slot factorization (Π_slots χ(g) × χ_perm(g)) is wrong for a
        # site-exchanging op: the correct trace is χ(S²) on the 2-cycle. The bond
        # inversion (perm = [2,1], S = +1) makes the failure exact: naive gives
        # χ(S)² × 0 = 0, the true trace is χ(E) = 3.
        inv_op = only(op for op in d4h
                      if op.perm == [2, 1] && norm(op.spin_rotation - I) < 1e-10)
        στ = CountingOracle._slot_permutation(slots, inv_op)
        correct = CountingOracle._op_character(slots, inv_op, στ[1], στ[2])
        @test correct ≈ 3.0
        naive = CountingOracle._chi_rotation(1, inv_op.spin_rotation)^2 *
                count(i -> inv_op.perm[i] == i, 1:2)
        @test naive == 0.0
        @test !(naive ≈ correct)

        # Aggregate counterexample (the paper's Table row): on the 4-slot mixed
        # cluster the group-averaged naive formula no longer coincides with the
        # correct count — 9 (correct = projector rank) vs 11 (naive).
        slots4 = CountingOracle.AbstractSlot[SpinSlot(1, 1), SpinSlot(2, 1),
                                             DispSlot(1, 0, 1), DispSlot(2, 0, 1)]
        correct4 = count_invariants(slots4, d4h)
        @test correct4 == 9
        @test correct4 == rank(invariant_projector(slots4, d4h), atol = 1e-6)
        naive4 = sum(CountingOracle._chi_rotation(1, op.spin_rotation)^2 *
                     CountingOracle._chi_rotation(1, op.rotation)^2 *
                     count(i -> op.perm[i] == i, 1:2) for op in d4h) / length(d4h)
        @test naive4 ≈ 11.0
        @test !(naive4 ≈ correct4)
    end

    # The §12 _pair_swapped regression: two coupled pair slots exchanged by C4z.
    # The slot cycle has length 2 and one orientation-swapped mapping; the
    # pre-port prototype errored on any pair-slot cycle of length > 1 and read a
    # wrong single-slot swap flag.
    @testset "regression: 4-site / 2-pair cluster under ⟨C4z⟩" begin
        sites = [SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0),
                 SVector(-1.0, 0.0, 0.0), SVector(0.0, -1.0, 0.0)]
        c4 = [cluster_op(_rz(k * π / 2), sites) for k = 0:3]
        assert_group_closure(c4)
        # Pairs (1,3) and (2,4); C4z maps (1,3) → (2,4) directly and (2,4) → (3,1),
        # the swapped orientation of (1,3).
        slots = CountingOracle.AbstractSlot[SpinPairSlot(1, 3, 1, 1, 1),
                                            SpinPairSlot(2, 4, 1, 1, 1)]
        n = count_invariants(slots, c4)
        P = invariant_projector(slots, c4)
        @test n == rank(P, atol = 1e-6)

        # Independent physical check: realize the slots as v_A = ê₁ × ê₃ and
        # v_B = ê₂ × ê₄ in tesseral order (y, z, x); every invariant-basis column
        # must be numerically invariant under the group action on random spin
        # configurations, and a vector orthogonal to the invariant space must not
        # be.
        tess(v) = SVector(v[2], v[3], v[1])
        function fvec(es)
            vA = tess(cross(es[1], es[3]))
            vB = tess(cross(es[2], es[4]))
            return [vA[μ] * vB[ν] for ν = 1:3 for μ = 1:3]  # slot 1 fastest
        end
        B = invariant_basis(slots, c4)
        @test size(B, 2) == n
        for _ = 1:6
            es = [normalize(randn(rng, 3)) for _ = 1:4]
            for op in c4
                inv_perm = invperm(op.perm)
                es_g = [Matrix(op.spin_rotation) * es[inv_perm[i]] for i = 1:4]
                for c = 1:size(B, 2)
                    @test B[:, c]' * fvec(es_g) ≈ B[:, c]' * fvec(es) atol = 1e-9
                end
            end
            if n < 9
                # A direction outside the invariant space moves under some op.
                w = randn(rng, 9)
                w -= B * (B' * w)
                w /= norm(w)
                moved = any(abs(w' * fvec([Matrix(g.spin_rotation) *
                                           es[invperm(g.perm)[i]] for i = 1:4]) -
                                w' * fvec(es)) > 1e-6 for g in c4)
                @test moved
            end
        end
    end

    # SOC-less pair + bent ligand, p = 1 (the gate-(f) shape at oracle level):
    # exactly one ligand invariant (v_c ∥ ŷ), none for the collinear geometry.
    @testset "pair + ligand p=1: bent vs collinear superexchange path" begin
        bent = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0)]
        stab = _site_ops(bent)
        assert_group_closure(stab)
        slots = CountingOracle.AbstractSlot[SpinPairSlot(1, 2, 1, 1, 0),
                                            DispSlot(3, 0, 1)]
        n = count_invariants(slots, stab)
        @test n == 1
        @test rank(invariant_projector(slots, stab), atol = 1e-6) == 1
        # Collinear path: ligand at the bond's inversion center kills the term.
        # Model it as the ligand exactly between two magnetic sites on one line;
        # u_c is inversion-odd while the L_S = 0 spin scalar is even.
        line = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0), SVector(0.0, 0.0, 0.0)]
        stab2 = _site_ops(line)
        slots2 = CountingOracle.AbstractSlot[SpinPairSlot(1, 2, 1, 1, 0),
                                             DispSlot(3, 0, 1)]
        @test count_invariants(slots2, stab2) == 0
        @test rank(invariant_projector(slots2, stab2), atol = 1e-6) == 0
        # Free spin slots on the bent shape (production's enumeration; Σl_spin
        # even): L_S-resolved counts 1 (the superexchange path) / 2 (DMI-like)
        # / 4, total 7. Pinned here per pair-coupled sector AND as the free
        # total; the production gate (f) (test_sectorbasis.jl) consumes both.
        for (LS, nLS) in ((0, 1), (1, 2), (2, 4))
            slotsL = CountingOracle.AbstractSlot[SpinPairSlot(1, 2, 1, 1, LS),
                                                 DispSlot(3, 0, 1)]
            @test count_invariants(slotsL, stab) == nLS
            @test rank(invariant_projector(slotsL, stab), atol = 1e-6) == nLS
        end
        slots_f = CountingOracle.AbstractSlot[SpinSlot(1, 1), SpinSlot(2, 1),
                                              DispSlot(3, 0, 1)]
        @test count_invariants(slots_f, stab) == 7
        @test rank(invariant_projector(slots_f, stab), atol = 1e-6) == 7
        # dJ/dr half of gate (f): the L_S = 0 pair with the displacement on ONE
        # magnetic site (the bond-stretch decoration; disp on Fe breaks the Fe
        # swap, so the decorated stabilizer is {E, σz} and Frobenius gives the
        # orbit count). Exactly 2 invariants: (ê₁·ê₂)(u₂)_x (stretch) and
        # (ê₁·ê₂)(u₂)_y (sway) — production realizes them as the symmetrized
        # (u₁ − u₂)_x / (u₁ + u₂)_y combinations (test_sectorbasis.jl).
        slots_dj = CountingOracle.AbstractSlot[SpinPairSlot(1, 2, 1, 1, 0),
                                               DispSlot(2, 0, 1)]
        stab_dj = stabilizer_ops(slots_dj, stab)
        @test length(stab_dj) == 2
        @test count_invariants(slots_dj, stab_dj) == 2
        @test rank(invariant_projector(slots_dj, stab_dj), atol = 1e-6) == 2
    end

    # Gate (o): the trait function `rep_scale` (basis/decor.jl) declares the
    # per-channel O(3) action relative to the polar real Wigner matrix,
    # D_channel(l, R) = rep_scale(channel, det R, l) · D_polar(l, R). The oracle
    # derives both channel matrices independently by polynomial composition
    # (`_slot_matrix`; spin slots see the axial `det(R)·R` via `cluster_op`), so
    # the identity is an op-by-op cross-check of the declaration — production
    # itself never applies `rep_scale` (the even-Σl_spin screen makes
    # det^{Σl_spin} ≡ +1; design record §4), which is exactly what the teeth
    # below probe.
    @testset "gate (o): rep_scale ≡ the derived channel action + mutation teeth" begin
        oh = [cluster_op(M, origin) for M in _oh_matrices()]
        for op in oh
            d = det(op.rotation)
            for l = 1:3
                Dp = CountingOracle._slot_matrix(DispSlot(1, 0, l), op)
                Ds = CountingOracle._slot_matrix(SpinSlot(1, l), op)
                @test Ds ≈ rep_scale(SLCE.SPIN, d, l) .* Dp atol = 1e-9
                @test rep_scale(SLCE.DISP, d, l) == 1.0
            end
        end
        # Inversion specialization (the pin as stated in §12 o): axial spin
        # +I for every l, polar displacement (−1)^l I.
        inv_op = only(op for op in oh if norm(op.rotation + I) < 1e-12)
        for l = 1:4
            n = 2 * l + 1
            @test CountingOracle._slot_matrix(SpinSlot(1, l), inv_op) ≈
                  Matrix(1.0I, n, n) atol = 1e-12
            @test CountingOracle._slot_matrix(DispSlot(1, 0, l), inv_op) ≈
                  (-1)^l .* Matrix(1.0I, n, n) atol = 1e-12
        end

        # -- mutation teeth. Each multiplies the oracle's TRUE representation
        # (= det^{Σl_spin}·⊗D_polar) by a ±1 character built from rep_scale
        # with wrong channel arguments; the Reynolds average then counts a
        # different isotypic component, so a changed rank is the tooth firing.
        # Because the base is the true rep, the EFFECTIVE mutated rule relative
        # to the polar product carries an extra det^{Σl_spin}:
        #   fac_all  (character det^{Σl_all})  ⇒ effective det^{Σl_disp}·⊗D_polar
        #     — spin treated POLAR, disp treated AXIAL;
        #   fac_disp (character det^{Σl_disp}) ⇒ effective det^{Σl_all}·⊗D_polar
        #     — every slot axial. Since det^{Σl_spin}·det^{Σl_disp} ≡
        #     det^{Σl_all}, "reinstate a global det^{Σl_all} factor on the
        #     polar cache" and "keep spin axial but treat disp as axial" are
        #     the SAME wrong rule — this one tooth fences both prose mistakes.
        # On even-Σl_spin content the two effective rules coincide with each
        # other (det^{Σl_spin} ≡ +1 — the screen theorem that makes
        # production's no-det polar cache exact) and differ from production by
        # det^{Σl_disp}: odd-Σl_disp content carries the teeth.
        slotl(s) = s isa SpinSlot ? s.l : s.L
        mut_rank(slots, ops, factor) = rank(
            sum(factor(op) .* representation_matrix(slots, op) for op in ops) ./
            length(ops); atol = 1e-6)
        fac_all(slots) =
            op -> prod(rep_scale(SLCE.SPIN, det(op.rotation), slotl(s))
                       for s in slots)
        fac_disp(slots) =
            op -> prod(rep_scale(SLCE.SPIN, det(op.rotation), s.L)
                       for s in slots if s isa DispSlot; init = 1.0)

        # Gate-(f) shape (bent pair + ligand, Σl_all = 3 odd): both teeth fire —
        # the correct count 7 collapses to 5 — and they agree with each other
        # (Σl_spin = 2 even).
        bent = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0),
                SVector(0.0, 1.0, 0.0)]
        stabb = _site_ops(bent)
        slots_f = CountingOracle.AbstractSlot[SpinSlot(1, 1), SpinSlot(2, 1),
                                              DispSlot(3, 0, 1)]
        @test count_invariants(slots_f, stabb) == 7
        @test mut_rank(slots_f, stabb, fac_all(slots_f)) == 5
        @test mut_rank(slots_f, stabb, fac_disp(slots_f)) == 5
        # Kill-shot cluster (Σl_spin odd — an oracle-only input): the all-axial
        # rule (fac_disp; ≡ the global det^{Σl_all} rule ≡ disp-as-axial)
        # RESURRECTS ê·u under inversion (0 → 1), while the spin-polar ×
        # disp-axial rule (fac_all) stays blind (its twist over the true rep is
        # det^{Σl_all} = det², even here). This is why the two teeth are
        # distinct mutations even though they coincide on production-reachable
        # content.
        ks = CountingOracle.AbstractSlot[SpinSlot(1, 1), DispSlot(1, 0, 1)]
        @test mut_rank(ks, oh, fac_disp(ks)) == 1
        @test mut_rank(ks, oh, fac_all(ks)) == 0
        # The gate-(g) 9-count bond (Σl_spin AND Σl_disp both even) is blind to
        # BOTH teeth — the reason gates (f) and the kill-shot, not the 9-count,
        # carry them.
        sites2 = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
        d4h2 = _site_ops(sites2)
        slots9 = CountingOracle.AbstractSlot[SpinSlot(1, 1), SpinSlot(2, 1),
                                             DispSlot(1, 0, 1), DispSlot(2, 0, 1)]
        @test mut_rank(slots9, d4h2, fac_all(slots9)) == 9
        @test mut_rank(slots9, d4h2, fac_disp(slots9)) == 9
    end

    # Mixed-channel permuting cluster: the chirality-twist decoration
    # (ê_a × ê_b)·(u_a × u_b) on a centrosymmetric bond — swap flags interact
    # with displacement slot cycles.
    @testset "chirality-twist decoration on a centrosymmetric bond" begin
        sites = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0)]
        d4h = _site_ops(sites)
        slots = CountingOracle.AbstractSlot[SpinPairSlot(1, 2, 1, 1, 1),
                                            DispSlot(1, 0, 1), DispSlot(2, 0, 1)]
        n = count_invariants(slots, d4h)
        @test n == rank(invariant_projector(slots, d4h), atol = 1e-6)
        @test n >= 1   # the twist invariant survives inversion + swap
        # Operational check of the oracle's representation (this is what catches
        # a wrong CG exchange sign — n == rank alone is blind to it, both routes
        # sharing the sign code): the twist F = (ê₁ × ê₂)·(u₁ × u₂) has the
        # explicit coefficient vector c over the slot 1-fastest tensor basis
        # (pair slot realized as tess(ê₁ × ê₂); tesseral order (y, z, x) maps
        # position α to Cartesian component (2, 3, 1)[α], so c is the Levi-Civita
        # tensor reindexed). It must be fixed by every D(g) and lie in the span
        # of the invariant basis. Dropping the exchange sign moves the residual
        # from ~1e-16 to ‖c‖.
        eps3(i, j, k) = ((j - i) * (k - i) * (k - j)) / 2
        tessidx = (2, 3, 1)
        c = zeros(27)
        for α = 1:3, β = 1:3, γ = 1:3
            c[α + 3 * (β - 1) + 9 * (γ - 1)] =
                eps3(tessidx[α], tessidx[β], tessidx[γ])
        end
        for op in d4h
            @test norm(representation_matrix(slots, op) * c - c) < 1e-10
        end
        B = invariant_basis(slots, d4h)
        @test size(B, 2) == n
        @test norm(c - B * (B' * c)) < 1e-10
    end
end
