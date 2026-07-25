# Mixed-channel (decor) SALC engine — joint M2b-2. Gates at engine level:
# pure-spin agreement with the production path (anti-drift), the cubic
# single-site count (gate e), production ≡ CountingOracle counts on a decorated
# bond (gate g flavor, incl. the chirality-twist L_S = 1 sector, gate g2), the
# u = 0 degeneracy (gate i core), and mixed-config space-group invariance.

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

# All 48 signed permutation matrices = O_h in Cartesian (= fractional for sc).
function _oh48()
    mats = SMatrix{3,3,Float64,9}[]
    for p in [[1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]]
        for sx in (-1, 1), sy in (-1, 1), sz in (-1, 1)
            M = zeros(3, 3)
            signs = (sx, sy, sz)
            for i = 1:3
                M[i, p[i]] = signs[i]
            end
            push!(mats, SMatrix{3,3,Float64,9}(M))
        end
    end
    return mats
end

@testset "mixed-channel SALC engine (M2b-2)" begin
    rng = MersenneTwister(0x51ce)

    # -- fixture A: 1-atom sc crystal with the full hand-assembled O_h group --
    # (a 1-atom cell hosts no MinimumImage pair — self-images are dropped — so
    # this crystal serves the single-site gates only)
    lat = Lattice(Matrix(2.0 * I(3)))
    xtal = Crystal(lat, zeros(3, 1), [1], ["Fe"])
    rots = _oh48()
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
    rotsB = [R for R in _oh48() if abs(R[1, 1]) == 1.0]     # D4h about x
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
        @test length(s0) == count(s -> s.key.L_S == 0, sall)
        @test length(s0) < length(sall)
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
                for R in _oh48()
                    @test evaluate_salc(s, Matrix(det(R) * R * e1),
                                        Matrix(R * u1)) ≈ v0 atol = 1e-10
                end
                @test evaluate_salc(s, -e1, u1) ≈ v0 atol = 1e-12
            end
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
