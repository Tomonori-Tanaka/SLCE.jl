# Mixed-channel (decor) SALC engine — joint M2b-2/M2d. Gates at engine level:
# pure-spin agreement with the production path (anti-drift), the cubic
# single-site count (gate e), production ≡ CountingOracle counts on a decorated
# bond (gate g flavor, incl. the chirality-twist L_S = 1 sector, gate g2), the
# u = 0 degeneracy (gate i core), mixed-config space-group invariance, the
# production-side channel-inversion pin (gate o), and L_S block-diagonality of
# the full grey projector (gate p).

using Test
using SLCE
using SLCE: SiteDecor, SiteFactor, SlotRef, SPIN, DISP, spin_decors, spin_ls,
            _orbit_salcs_decors, _orbit_salcs, _build_wig_cache, evaluate_salc,
            build_clusters, build_neighbor_list, _assemble_spacegroup, salcs
using LinearAlgebra
using StaticArrays
using Random

# shared helpers (same_members, ...) — included once by runtests.jl; standalone
# runs pull them in here
isdefined(@__MODULE__, :same_members) || include("testutils.jl")

# Molecule-in-a-box triangles (local copies of the test_nbody.jl fixtures):
# Cs isosceles (2 ordering orbits for ls = [1,1,2]) and C3v equilateral
# (one orbit of 3 assignments) — the review-blocker shapes for the anti-drift
# gate (block-index and gauge order must match the pure-spin engine exactly).
function _mx_triangle_cs(; L = 8.0)
    c = SVector{3,Float64}(0.5, 0.5, 0.5)
    offs = [SVector{3,Float64}(0.0, 1.5, 0.0), SVector{3,Float64}(-1.0, 0.0, 0.0),
            SVector{3,Float64}(1.0, 0.0, 0.0)]
    frac = reduce(hcat, [c + o / L for o in offs])
    crystal = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 1, 1], ["Fe"])
    σ = SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 1])
    rots = [SMatrix{3,3,Float64}(I), σ]
    trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
    return crystal, _assemble_spacegroup(crystal, rots, trans, "Cs", 0; tol = 1e-6)
end

function _mx_triangle_c3v(; L = 6.0, r = 1.2)
    c = SVector{3,Float64}(0.5, 0.5, 0.5)
    ang = deg2rad.([90.0, 210.0, 330.0])
    offs = [SVector{3,Float64}(r * cos(a), r * sin(a), 0.0) for a in ang]
    frac = reduce(hcat, [c + o / L for o in offs])
    crystal = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 1, 1], ["Fe"])
    cz(θ) = SMatrix{3,3,Float64}([cos(θ) -sin(θ) 0; sin(θ) cos(θ) 0; 0 0 1])
    M1 = SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 1])
    rots = [SMatrix{3,3,Float64}(I), cz(2π / 3), cz(4π / 3), M1, cz(2π / 3) * M1,
            cz(4π / 3) * M1]
    trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
    return crystal, _assemble_spacegroup(crystal, rots, trans, "C3v", 0; tol = 1e-6)
end

# Gate (p) helper: assemble the FULL grey-projector matrix over EVERY
# (L_S, Lf) coupled-path block of one decoration label — the same loop as
# `_project_and_fold_decors`, but without the per-block filter and without the
# small-entry drop — and return `(maxoff, ranktotal)` summed over the
# permutation orbits of the site assignments: `maxoff` is the largest
# cross-L_S matrix element over every per-op representation matrix D(g) AND
# the Reynolds average, `ranktotal` the summed projector rank.
function _ls_block_stats(crystal, sg, O, label, wcache)
    stab = SLCE._stabilizer(crystal, sg, O.representative)
    perms = unique([perm for (_, perm) in stab])
    maxoff = 0.0
    ranktotal = 0
    seen = Set{Vector{SLCE.SiteDecor}}()
    for tarr in SLCE._multiset_arrangements(label)
        tarr in seen && continue
        orbit = unique([tarr[p] for p in perms])
        foreach(o -> push!(seen, o), orbit)
        t = minimum(orbit)
        assignments = unique([t[p] for p in perms])
        slotlists = [SLCE._assignment_slots(a) for a in assignments]
        cbs = [SLCE._decor_coupled_bases(sl) for sl in slotlists]
        cols = Tuple{Int,Int,Int}[]              # (assignment, path, Mf)
        tags = Int[]                             # the path's L_S per column
        for ai in eachindex(assignments), ci in eachindex(cbs[ai])
            (LS, Lf, _) = cbs[ai][ci]
            for Mf = 1:(2 * Lf + 1)
                push!(cols, (ai, ci, Mf))
                push!(tags, LS)
            end
        end
        D = length(cols)
        colidx = Dict(cols[k] => k for k = 1:D)
        P = zeros(D, D)
        for (g, perm) in stab
            Dg = zeros(D, D)
            for k = 1:D
                (ai, ci, Mf) = cols[k]
                slots = slotlists[ai]
                v = SLCE._mfslice(cbs[ai][ci][3], Mf)
                for j in eachindex(slots)
                    slots[j].factor.l == 0 && continue
                    v = SLCE.AngularMomentum.nmode_mul(
                        v, SLCE._wig(wcache, slots[j].factor.l, g), j)
                end
                a2 = SLCE._assignment_image(assignments[ai], perm)
                ai2 = findfirst(==(a2), assignments)
                σ = SLCE._slot_sigma(slots, slotlists[ai2], perm)
                v = permutedims(v, invperm(σ))
                for ci2 in eachindex(cbs[ai2])
                    Lf2 = cbs[ai2][ci2][2]
                    for Mf2 = 1:(2 * Lf2 + 1)
                        Dg[colidx[(ai2, ci2, Mf2)], k] =
                            SLCE._frob(SLCE._mfslice(cbs[ai2][ci2][3], Mf2), v)
                    end
                end
            end
            for k1 = 1:D, k2 = 1:D
                tags[k1] == tags[k2] ||
                    (maxoff = max(maxoff, abs(Dg[k1, k2])))
            end
            P .+= Dg
        end
        P ./= length(stab)
        for k1 = 1:D, k2 = 1:D
            tags[k1] == tags[k2] || (maxoff = max(maxoff, abs(P[k1, k2])))
        end
        ranktotal += rank(P; atol = 1e-6)
    end
    return maxoff, ranktotal
end

@testset "mixed-channel SALC engine (M2b-2)" begin
    rng = MersenneTwister(0x51ce)

    # -- fixture A: 1-atom sc crystal with the full hand-assembled O_h group --
    # (a 1-atom cell hosts no MinimumImage pair — self-images are dropped — so
    # this crystal serves the single-site gates only)
    lat = Lattice(Matrix(2.0 * I(3)))
    xtal = Crystal(lat, zeros(3, 1), [1], ["Fe"])
    rots = oh48_matrices()
    trs = [SVector{3,Float64}(0, 0, 0) for _ in rots]
    sg = _assemble_spacegroup(xtal, rots, trs, "Pm-3m", 221; tol = 1e-5)
    wc = _build_wig_cache(sg, 4)
    cs1 = build_clusters(xtal, build_neighbor_list(xtal, 2.1), sg; nbody = 1)
    O1 = cs1.by_body[1][1]

    # -- fixture B: two Fe at cart ±(0.5, 0, 0) in a 3.0 cube, D4h ops about the
    # origin (= the bond midpoint): the x-reversing half SWAPS the two sites, so
    # the bond stabilizer is the full 16-op d4h of the CountingOracle bond gates.
    latB = Lattice(Matrix(3.0 * I(3)))
    xtalB = Crystal(latB, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    rotsB = [R for R in oh48_matrices() if abs(R[1, 1]) == 1.0]     # D4h about x
    @test length(rotsB) == 16
    trsB = [SVector{3,Float64}(0, 0, 0) for _ in rotsB]
    sgB = _assemble_spacegroup(xtalB, rotsB, trsB, "P4/mmm", 123; tol = 1e-5)
    wcB = _build_wig_cache(sgB, 4)
    csB = build_clusters(xtalB, build_neighbor_list(xtalB, 1.1), sgB; nbody = 2)
    O2 = csB.by_body[2][1]                      # the 1.0-long x bond orbit

    @testset "engines agree on pure spin (anti-drift gate)" begin
        # 1-body l = 2 and the bond pair (1,1) / (2,2) labels, on the O_h crystal.
        old1 = _orbit_salcs(xtal, sg, 1, 1, O1, [4], SLCE.LSUM_UNCAPPED, false, wc)
        new1 = _orbit_salcs_decors(xtal, sg, 1, 1, O1,
                                   [spin_decors([l]) for l = [2, 4]], true, wc)
        # pure-spin enumeration gives l = 2, 4 (l = 1, 3 die under O_h; Σl even)
        @test [spin_ls(s.key) for s in old1] == [spin_ls(s.key) for s in new1]
        @test all(s.key.L_S == s.key.Lf for s in new1)
        for (a, b) in zip(old1, new1)
            @test a.key == b.key
            same_members(a.members, b.members)
        end
        old2 = _orbit_salcs(xtalB, sgB, 2, 1, O2, [2], SLCE.LSUM_UNCAPPED, false, wcB)
        new2 = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2,
                                   [spin_decors([1, 1]), spin_decors([2, 2])],
                                   true, wcB)
        @test [s.key for s in old2] == [s.key for s in new2]
        for (a, b) in zip(old2, new2)
            same_members(a.members, b.members)
        end
        # Review-blocker shapes: ls = [1,1,2] on the Cs triangle SPLITS into two
        # ordering orbits sharing one sorted label (block indices must match the
        # pure-spin emission order), and on the C3v triangle one orbit carries
        # 3 assignments (the gauge is column-order dependent — bitwise match
        # required, not just span equality).
        for (xt, st) in (_mx_triangle_cs(), _mx_triangle_c3v())
            wc3 = SLCE._build_wig_cache(st, 2)
            cs3 = build_clusters(xt, build_neighbor_list(xt, 2.2), st; nbody = 3)
            O3 = cs3.by_body[3][1]
            old3 = _orbit_salcs(xt, st, 3, 1, O3, [2], SLCE.LSUM_UNCAPPED, false,
                                wc3)
            new3 = _orbit_salcs_decors(xt, st, 3, 1, O3, [spin_decors([1, 1, 2])],
                                       true, wc3)
            old112 = [s for s in old3 if SLCE.spin_ls(s.key) == [1, 1, 2]]
            @test length(new3) == length(old112) > 0
            @test [s.key for s in old112] == [s.key for s in new3]
            for (a, b) in zip(old112, new3)
                same_members(a.members, b.members)
            end
        end
    end

    @testset "gate (e): cubic single-site l=2 × p=2 blocks" begin
        # Traceless (k=0, L=2) block: exactly 2 invariants (E_g ⊕ T_2g squares).
        lab_e = [SiteDecor(; spin = 2, disp = (0, 2))]
        se = _orbit_salcs_decors(xtal, sg, 1, 1, O1, [lab_e], true, wc)
        @test length(se) == 2
        @test all(s.key.L_S == 2 for s in se)
        # |u|² trace channel: rank-2 spin ⊗ scalar ⇒ 0 invariants.
        lab_t = [SiteDecor(; spin = 2, disp = (1, 0))]
        @test isempty(_orbit_salcs_decors(xtal, sg, 1, 1, O1, [lab_t], true, wc))
        # soc = false kills the whole L_S = 2 sector.
        @test isempty(_orbit_salcs_decors(xtal, sg, 1, 1, O1, [lab_e], false, wc))
    end

    @testset "gate (g)/(g2): decorated bond count ≡ oracle, twist sector" begin
        # Both bond sites carry spin l=1 AND disp (0,1): the 4-slot cluster whose
        # oracle stabilizer count is 9 (test_countingoracle.jl pins it; the same
        # number fills the paper's counting table).
        lab = [SiteDecor(; spin = 1, disp = (0, 1)),
               SiteDecor(; spin = 1, disp = (0, 1))]
        sall = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [lab], true, wcB)
        @test length(sall) == 9
        # Gate (g2): the chirality twist (ê₁×ê₂)·(u₁×u₂) lives in L_S = 1 — the
        # SOC-carrying sector survives the centrosymmetric bond...
        @test any(s.key.L_S == 1 for s in sall)
        # ...and soc = false (L_S = 0 only) removes it.
        s0 = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [lab], false, wcB)
        @test all(s.key.L_S == 0 for s in s0)
        # soc = false is BITWISE the L_S = 0 subset of the soc = true build
        subset = [s for s in sall if s.key.L_S == 0]
        @test [s.key for s in s0] == [s.key for s in subset]
        for (a, b) in zip(s0, subset)
            same_members(a.members, b.members)
        end
        @test length(s0) < length(sall)
        # the (L_S = 1, Lf = 0) SALC IS the chirality twist: Φ ∝ (ê₁×ê₂)·(u₁×u₂)
        tw = only(s for s in sall if s.key.L_S == 1 && s.key.Lf == 0)
        rng2 = MersenneTwister(0x7715)
        ratios = Float64[]
        for _ = 1:6
            e = reduce(hcat, [normalize(randn(rng2, 3)) for _ = 1:2])
            u = randn(rng2, 3, 2) * 0.4
            F = dot(cross(e[:, 1], e[:, 2]), cross(u[:, 1], u[:, 2]))
            abs(F) < 1e-3 && continue
            push!(ratios, evaluate_salc(tw, e, u) / F)
        end
        @test !isempty(ratios)
        @test all(r -> isapprox(r, ratios[1]; rtol = 1e-9), ratios)
    end

    @testset "gate (i) core: u = 0 degeneracy and pure-spin consistency" begin
        lab = [SiteDecor(; spin = 1, disp = (0, 1)),
               SiteDecor(; spin = 1, disp = (0, 1))]
        sall = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [lab], true, wcB)
        e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:2])
        u0 = zeros(3, 2)
        for s in sall
            @test evaluate_salc(s, e, u0) == 0.0     # exact: homogeneous ≥ 1
        end
        # pure-spin SALCs evaluate identically through the joint form, ∀u
        pure = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [spin_decors([1, 1])],
                                   true, wcB)
        u = randn(rng, 3, 2) * 0.3
        for s in pure
            @test evaluate_salc(s, e, u) === evaluate_salc(s, e)
        end
        # the spin-only forms refuse a decorated SALC instead of mis-scaling
        mixed = first(sall)
        @test_throws ArgumentError evaluate_salc(mixed, e)
        @test_throws ArgumentError SLCE.accumulate_grad!(zeros(3, 2), mixed, e, 1.0)
        @test_throws ArgumentError evaluate_salc(mixed, e, zeros(3, 1))  # size
        # duplicate labels are rejected (collinear-column guard)
        lab2 = [SiteDecor(; spin = 1, disp = (0, 1)),
                SiteDecor(; spin = 1, disp = (0, 1))]
        @test_throws ArgumentError _orbit_salcs_decors(xtalB, sgB, 2, 1, O2,
                                                       [lab2, lab2], true, wcB)
    end

    @testset "mixed space-group invariance + time reversal" begin
        lab = [SiteDecor(; spin = 1, disp = (0, 1)),
               SiteDecor(; spin = 1, disp = (0, 1))]
        sall = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [lab], true, wcB)
        lab1 = [SiteDecor(; spin = 2, disp = (0, 2))]
        s1 = _orbit_salcs_decors(xtal, sg, 1, 1, O1, [lab1], true, wc)
        for _ = 1:6
            # bond crystal: ops with x → −x swap the two atom columns
            e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:2])
            u = randn(rng, 3, 2) * 0.4
            for s in sall
                v0 = evaluate_salc(s, e, u)
                for R in rotsB
                    p = R[1, 1] > 0 ? (1:2) : (2:-1:1)
                    eg = det(R) * R * e[:, p]      # axial spin action + site map
                    ug = R * u[:, p]               # polar displacement action
                    @test evaluate_salc(s, Matrix(eg), Matrix(ug)) ≈ v0 atol = 1e-10
                end
                @test evaluate_salc(s, -e, u) ≈ v0 atol = 1e-12
            end
            # single-site crystal: all 48 ops fix the lone atom
            e1 = reshape(normalize(randn(rng, 3)), 3, 1)
            u1 = randn(rng, 3, 1) * 0.4
            for s in s1
                v0 = evaluate_salc(s, e1, u1)
                for R in oh48_matrices()
                    @test evaluate_salc(s, Matrix(det(R) * R * e1),
                                        Matrix(R * u1)) ≈ v0 atol = 1e-10
                end
                @test evaluate_salc(s, -e1, u1) ≈ v0 atol = 1e-12
            end
        end
    end

    @testset "gate (o): channel-inversion pin on the production Wigner" begin
        # Production's single polar cache gives D_polar(l, −I) = (−1)^l I; the
        # channel actions follow from `rep_scale`: SPIN +I for every l (axial),
        # DISP (−1)^l I (polar). Production never applies rep_scale — the
        # even-Σl_spin screen makes det^{Σl_spin} ≡ +1 (design record §4), so
        # the polar cache alone is exact for both channels; this pin guards the
        # CLAIM the screen rests on, against the production Wigner kernel. The
        # oracle-side identity + mutation teeth live in test_countingoracle.jl.
        inv3 = SMatrix{3,3,Float64}(-1.0 * I)
        for l = 1:4
            n = 2 * l + 1
            W = SLCE.AngularMomentum.wignerD_real(l, inv3)
            @test W ≈ (-1)^l .* Matrix(1.0I, n, n) atol = 1e-12
            @test SLCE.rep_scale(SLCE.SPIN, -1.0, l) .* W ≈
                  Matrix(1.0I, n, n) atol = 1e-12
            @test SLCE.rep_scale(SLCE.DISP, -1.0, l) .* W ≈
                  (-1)^l .* Matrix(1.0I, n, n) atol = 1e-12
        end
    end

    @testset "gate (p): L_S block-diagonality of the grey projector" begin
        # Every per-op representation matrix — hence the Reynolds average — is
        # block-diagonal in the coupled-path label L_S: rotations act within a
        # path (the coupled basis carries them irrep-wise), and stabilizer site
        # permutations map spin slots to spin slots, preserving the total spin
        # rank. Production projects per (L_S, Lf) block and would silently MISS
        # cross-block weight if this broke — gate (p) is the fence under every
        # L_S claim (per-sector soc, sector masks, hierarchy).
        lab = [SiteDecor(; spin = 1, disp = (0, 1)),
               SiteDecor(; spin = 1, disp = (0, 1))]
        off, r = _ls_block_stats(xtalB, sgB, O2, lab, wcB)
        @test off < 1e-9
        # the full-space rank equals the engine's emitted SALC count
        @test r == 9
        @test r == length(_orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [lab], true,
                                              wcB))
        # 3-body mixed label on the Cs triangle: two assignment orbits, a
        # nontrivial site permutation, and L_S ∈ {0, 1, 2} paths
        xt, st = _mx_triangle_cs()
        wc3 = SLCE._build_wig_cache(st, 2)
        cs3 = build_clusters(xt, build_neighbor_list(xt, 2.2), st; nbody = 3)
        O3 = cs3.by_body[3][1]
        lab3 = sort([SiteDecor(; spin = 1), SiteDecor(; spin = 1),
                     SiteDecor(; disp = (0, 1))])
        off3, r3 = _ls_block_stats(xt, st, O3, lab3, wc3)
        @test off3 < 1e-9
        @test r3 == length(_orbit_salcs_decors(xt, st, 3, 1, O3, [lab3], true,
                                               wc3))
        @test r3 > 0
    end

    @testset "sector-driven build reproduces the engine (M2b-3b)" begin
        # The gate-(g) content expressed as a sector: spin = [1, 1] with a
        # degree-2 displacement budget under pmax = 1 realizes exactly the
        # doubly-decorated bond label, so the 4-arg builder must reproduce the
        # engine's 9 SALCs bitwise.
        nlB = build_neighbor_list(xtalB, 1.1)
        lab = [SiteDecor(; spin = 1, disp = (0, 1)),
               SiteDecor(; spin = 1, disp = (0, 1))]
        engine = _orbit_salcs_decors(xtalB, sgB, 2, 1, O2, [lab], true, wcB)
        spec = BasisSpec(["Fe"]; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1)])
        sb = SLCE.build_salc_basis(xtalB, sgB, csB, spec; neighbors = nlB)
        @test [s.key for s in sb.salcs] == [s.key for s in engine]
        for (a, b) in zip(sb.salcs, engine)
            @test same_members(a.members, b.members)
        end
        # Key-union invariant: adding an overlapping soc = false sector with the
        # same content changes NOTHING (per-label soc is OR'd, keys stay unique).
        spec2 = BasisSpec(["Fe"]; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1,
                   soc = false)])
        sb2 = SLCE.build_salc_basis(xtalB, sgB, csB, spec2; neighbors = nlB)
        @test [s.key for s in sb2.salcs] == [s.key for s in sb.salcs]
        # soc = false alone = the bitwise L_S = 0 subset of the soc = true build.
        spec0 = BasisSpec(["Fe"]; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1,
                   soc = false)])
        sb0 = SLCE.build_salc_basis(xtalB, sgB, csB, spec0; neighbors = nlB)
        subset = [s for s in sb.salcs if s.key.L_S == 0]
        @test [s.key for s in sb0.salcs] == [s.key for s in subset]
        for (a, b) in zip(sb0.salcs, subset)
            @test same_members(a.members, b.members)
        end
    end

    @testset "label validation" begin
        @test_throws ArgumentError _orbit_salcs_decors(xtalB, sgB, 2, 1, O2,
            [[SiteDecor(; spin = 1), SiteDecor(; spin = 2)]], true, wcB)  # Σl odd
        @test_throws ArgumentError _orbit_salcs_decors(xtalB, sgB, 2, 1, O2,
            [[SiteDecor(; spin = 1)]], true, wcB)                         # wrong N
        @test_throws ArgumentError _orbit_salcs_decors(xtalB, sgB, 2, 1, O2,
            [[SiteDecor(; spin = 2), SiteDecor(; spin = 1, disp = (0, 1))]],
            true, wcB)                                                    # unsorted
    end
end
