# Sector-driven basis construction (M2b-3b) at the public-API level: the label
# generators, the sector-expressed pure-spin ≡ legacy-dense bit-identity,
# per-sector cutoff re-admission, per-species lmax/pmax engine caps, u = 0
# exactness, and persistence of a mixed basis — plus the M2d spec-level gates:
# (f) SOC-less pair+ligand+bond-stretch with no DMI on a mixed spec, (e2) the
# ε-linear two-constant cubic magnetoelastic form (B₁/B₂), (n) sector-mask ≡
# soc-false build, (d)/(a) dense ≡ p = 0-restricted sector build down to the
# design matrices and the MC consumption surface, and (i) full u = 0 bitwise
# degeneracy against the dense spin basis. Engine-level gates (counts vs the
# CountingOracle, invariance, soc subsets) live in test_mixedsalc.jl.

using Test
using SLCE
using SLCE: SiteDecor, evaluate_salc, salcs, build_neighbor_list, build_clusters,
    _spin_multisets, _disp_multisets, _marry_multisets, _sector_orbit_labels,
    LSUM_UNCAPPED, is_pure_spin, SPIN, DISP, _assemble_spacegroup
using LinearAlgebra
using Random

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_sb_cfg(rng) = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:2])

# Deep bit-exact equality of two bases' SALC content (keys + members).
function _sector_basis_identical(a, b)
    ka = [s.key for s in salcs(a)]
    kb = [s.key for s in salcs(b)]
    ka == kb || return false
    for (sa, sb) in zip(salcs(a), salcs(b))
        same_members(sa.members, sb.members) || return false
    end
    return true
end

@testset "sector-driven basis (M2b-3b)" begin
    @testset "label generators" begin
        # spin multisets: nondecreasing, Σ even, caps respected
        @test _spin_multisets(2, 2, LSUM_UNCAPPED) == [[1, 1], [2, 2]]
        @test _spin_multisets(2, 3, 4) == [[1, 1], [1, 3], [2, 2]]
        @test _spin_multisets(1, 2, LSUM_UNCAPPED) == [[2]]      # odd l alone drops
        @test _spin_multisets(0, 3, LSUM_UNCAPPED) == [Int[]]
        @test isempty(_spin_multisets(2, 0, LSUM_UNCAPPED))
        # disp multisets: the (k, l) decomposition per total degree
        @test _disp_multisets(1, 2) == [[(0, 1)]]
        @test sort(_disp_multisets(2, 2)) ==
              sort([[(0, 1), (0, 1)], [(0, 2)], [(1, 0)]])
        @test _disp_multisets(2, 1) == [[(0, 1), (0, 1)]]        # pcap kills singles
        @test _disp_multisets(0, 2) == [Tuple{Int,Int}[]]
        # marriage onto N sites (shared sites carry both channels)
        @test _marry_multisets([1, 1], [(0, 1)], 2) ==
              [sort([SiteDecor(; spin = 1), SiteDecor(; spin = 1, disp = (0, 1))])]
        m12 = _marry_multisets([1, 2], [(0, 1)], 2)
        @test length(m12) == 2                                   # disp on l=1 or l=2
        @test _marry_multisets([1, 1], [(0, 1), (0, 1)], 2) ==
              [[SiteDecor(; spin = 1, disp = (0, 1)),
                SiteDecor(; spin = 1, disp = (0, 1))]]
        # 3 sites: nothing shared
        m3 = _marry_multisets([1, 1], [(0, 1)], 3)
        @test m3 == [sort([SiteDecor(; spin = 1), SiteDecor(; spin = 1),
                           SiteDecor(; disp = (0, 1))])]
        @test isempty(_marry_multisets([1, 1], [(0, 1)], 5))     # too many sites
        @test isempty(_marry_multisets([1, 1], Tuple{Int,Int}[], 1))  # too few
    end

    @testset "sector-expressed pure spin ≡ legacy dense (bitwise)" begin
        # P1 fixture (the 44-column standard cell)
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        legacy = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2]))
        sect = SLCEBasis(cr, BasisSpec(cr; lmax = 2,
            sectors = [Sector(spin = (sites = 1:2,), cutoff = 1.5)]))
        @test _sector_basis_identical(legacy, sect)
        @test sect.salc_basis.fingerprint == legacy.salc_basis.fingerprint
        # symmetric chain with a Σl cap, soc = false variant
        latc = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        chain = Crystal(latc, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75],
                        [1, 1, 1, 1], ["Fe"])
        for soc in (true, false)
            lg = SLCEBasis(chain, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [3],
                                            lsum = 4, soc = soc))
            # global per-body lsum caps the sector content identically
            sc = SLCEBasis(chain, BasisSpec(chain; lmax = 3, lsum = 4,
                sectors = [Sector(spin = (sites = 1:2,), cutoff = 2.6,
                                  soc = soc)]))
            @test _sector_basis_identical(lg, sc)
            # equivalently: the sector-local lsum
            sc2 = SLCEBasis(chain, BasisSpec(chain; lmax = 3,
                sectors = [Sector(spin = (sites = 1:2, lsum = 4), cutoff = 2.6,
                                  soc = soc)]))
            @test _sector_basis_identical(lg, sc2)
        end
    end

    @testset "per-sector cutoff re-admission" begin
        # 4-atom chain: NN bond 2.5 Å, NNN 5.0 Å. A spin sector confined to the
        # NN shell + a displacement sector reaching the NNN shell: spin-decorated
        # pair SALCs exist only on NN orbits, displacement-only pairs on both.
        latc = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        chain = Crystal(latc, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75],
                        [1, 1, 1, 1], ["Fe"])
        spec = BasisSpec(chain; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], cutoff = 2.6),
            Sector(disp = (degree = 2,), sites = 2, cutoff = 5.2)])
        b = SLCEBasis(chain, spec)
        # classify each 2-body SALC by its bond length via the member atoms
        cart = SLCE.cartesian_positions(chain)
        A = chain.lattice.vectors
        function bondlen(s)
            m = s.members[1]
            p1 = cart[:, m.atoms[1]] .+ A * Float64.(m.shifts[1])
            p2 = cart[:, m.atoms[2]] .+ A * Float64.(m.shifts[2])
            return norm(p2 - p1)
        end
        spin2 = [s for s in salcs(b) if s.body == 2 && any(SLCE.has_spin, s.key.decors)]
        disp2 = [s for s in salcs(b)
                 if s.body == 2 && all(d -> !SLCE.has_spin(d), s.key.decors)]
        @test !isempty(spin2) && !isempty(disp2)
        @test all(s -> bondlen(s) < 2.6, spin2)          # spin capped at NN
        @test any(s -> bondlen(s) > 4.0, disp2)          # disp reaches NNN
        # single-radius reference: the spin sector at 5.2 would have NNN SALCs
        spec_wide = BasisSpec(chain; lmax = 1,
                              sectors = [Sector(spin = [1, 1], cutoff = 5.2)])
        bw = SLCEBasis(chain, spec_wide)
        @test any(s -> bondlen(s) > 4.0, [s for s in salcs(bw) if s.body == 2])
    end

    @testset "per-species caps: ligand pmax and the engine filter" begin
        # Fe carries spin (lmax 2) and small displacements (pmax 1); O is a
        # spin-clamped ligand (lmax 0, pmax 2). The engine's per-site filter
        # must keep spin factors on Fe and degree-2 displacement factors on O.
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 2], ["Fe", "O"])
        spec = BasisSpec(cr; lmax = ["Fe" => 2, "O" => 0],
                         pmax = ["Fe" => 1, "O" => 2], sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.5),
            Sector(spin = [2], disp = (degree = 1:2,), cutoff = 1.5)])
        b = SLCEBasis(cr, spec)
        @test n_salcs(b) > 0
        sp = cr.species
        # pure-spin content survives (Fe on-site l = 2) but never a spin pair
        # ([1,1]/[2,2] would need two spin-capable sites; the pair is Fe-O)
        @test any(s -> all(is_pure_spin, s.key.decors), salcs(b))
        @test all(s -> count(SLCE.has_spin, s.key.decors) <= 1, salcs(b))
        # the mixed sector reaches the ligand with degree-2 factors
        @test any(salcs(b)) do s
            any(m -> any(t -> any(sl -> sl.factor.channel == DISP &&
                                        2 * sl.factor.k + sl.factor.l == 2 &&
                                        sp[m.atoms[sl.site]] == 2, t.slots),
                         m.terms), s.members)
        end
        for s in salcs(b), m in s.members, t in m.terms, sl in t.slots
            at = m.atoms[sl.site]
            if sl.factor.channel == SPIN
                @test sp[at] == 1                        # spin on Fe only
            else
                # per-species pmax: Fe ≤ 1, O ≤ 2
                @test 2 * sl.factor.k + sl.factor.l <= (sp[at] == 1 ? 1 : 2)
            end
        end
    end

    @testset "u = 0 exactness and dataset guard on a mixed basis" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 2,), cutoff = 1.5),
            Sector(spin = [1, 1], disp = (degree = 1:2,), sites = 2, cutoff = 1.5),
            Sector(disp = (degree = 2,), cutoff = 1.5)])
        b = SLCEBasis(cr, spec)
        @test n_salcs(b) > 0
        @test allunique(b.salc_basis.keys)               # key-union invariant
        rng = MersenneTwister(0x53c7)
        e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:2])
        u = randn(rng, 3, 2) * 0.3
        u0 = zeros(3, 2)
        for s in salcs(b)
            v0 = evaluate_salc(s, e, u0)
            if all(is_pure_spin, s.key.decors)
                @test v0 == evaluate_salc(s, e)          # bitwise p = 0 degeneracy
            else
                @test v0 == 0.0                          # disp factors vanish at u = 0
            end
            # time-reversal evenness of the joint SALC (Σl_spin even)
            @test evaluate_salc(s, -e, u) ≈ evaluate_salc(s, e, u) atol = 1e-12
        end
        # the spin-only dataset path refuses displacement-decorated SALCs (M3
        # brings the joint data layer)
        cfg = [_sb_cfg(rng) for _ = 1:3]
        @test_throws ArgumentError SLCEDataset(b, cfg, randn(rng, 3))
    end

    @testset "gate (f): SOC-less pair + ligand p = 1 on a mixed spec — no DMI" begin
        # Bent Fe–O–Fe superexchange unit in a box (the CountingOracle gate-(f)
        # shape: Fe at ±x, O on +y), stabilizer C2v = {E, σz, C2y, σx} with the
        # x-reversing half swapping the two Fe. The spec is MIXED on purpose
        # (design record §13 risk 4): the 3-body ligand sector and the 2-body
        # bond-stretch (dJ/dr) sector are soc = false; a coexisting 2-body
        # Fe–Fe sector with a degree-2 displacement budget (the twist-carrying
        # doubly-decorated content) is soc = true.
        L = 8.0
        c = SVector{3,Float64}(0.5, 0.5, 0.5)
        offs = [SVector{3,Float64}(1.0, 0.0, 0.0), SVector{3,Float64}(-1.0, 0.0, 0.0),
                SVector{3,Float64}(0.0, 1.0, 0.0)]
        frac = reduce(hcat, [c + o / L for o in offs])
        cr = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 1, 2], ["Fe", "O"])
        rots = [SMatrix{3,3,Float64}(I),
                SMatrix{3,3,Float64}([1.0 0 0; 0 1 0; 0 0 -1]),    # σz
                SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 -1]),   # C2y (swaps Fe)
                SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 1])]    # σx (swaps Fe)
        trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
        sg = _assemble_spacegroup(cr, rots, trans, "C2v", 0; tol = 1e-6)
        nl = build_neighbor_list(cr, 2.3)                # Fe–Fe 2.0, Fe–O √2
        cs = build_clusters(cr, nl, sg; nbody = 3)
        mkspec(soc3) = BasisSpec(cr; lmax = ["Fe" => 1, "O" => 0],
                                 pmax = ["Fe" => 1, "O" => 1], sectors = [
            Sector(spin = [1, 1], disp = (degree = 1,), sites = 3, cutoff = 2.3,
                   soc = soc3),
            Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = 2.1,
                   soc = false),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 2.1)])
        b = SLCE.build_salc_basis(cr, sg, cs, mkspec(false); neighbors = nl)
        b3 = [s for s in b.salcs if s.body == 3]
        # the soc = false ligand sector emits ONLY its L_S = 0 block: exactly
        # the 1 superexchange-path invariant the oracle pins (L_S-resolved
        # bent counts 1/2/4 in test_countingoracle.jl) — and none of the
        # DMI-like L_S = 1 content
        @test length(b3) == 1
        @test all(s -> s.key.L_S == 0, b3)
        # operational no-DMI/no-SOC statement: every L_S = 0 SALC is invariant
        # under a GLOBAL spin rotation with the lattice held fixed; every
        # L_S ≥ 1 SALC (here: the soc = true Fe–Fe sector, incl. the twist)
        # must violate it
        rngf = MersenneTwister(0xf00d)
        @test any(s -> s.key.L_S >= 1, b.salcs)
        for s in b.salcs
            vmax = 0.0
            for _ = 1:4
                e = reduce(hcat, [normalize(randn(rngf, 3)) for _ = 1:3])
                u = randn(rngf, 3, 3) * 0.4
                v0 = evaluate_salc(s, e, u)
                for _ = 1:4
                    R = Matrix(rand_rotation(rngf))
                    vmax = max(vmax, abs(evaluate_salc(s, R * e, u) - v0))
                end
            end
            if s.key.L_S == 0
                @test vmax < 1e-9
            else
                @test vmax > 1e-4
            end
        end
        # the single ligand SALC IS the superexchange-path term
        # Φ ∝ (ê₁·ê₂)(u_O)_y — the C2v-fixed direction — and reads u_O only
        dj = only(b3)
        rats = Float64[]
        for _ = 1:8
            e = reduce(hcat, [normalize(randn(rngf, 3)) for _ = 1:3])
            uy = randn(rngf)
            u = zeros(3, 3)
            u[2, 3] = uy
            d = dot(e[:, 1], e[:, 2]) * uy
            abs(d) < 1e-2 && continue
            push!(rats, evaluate_salc(dj, e, u) / d)
        end
        @test length(rats) >= 2
        @test all(r -> isapprox(r, rats[1]; rtol = 1e-9), rats)
        e = reduce(hcat, [normalize(randn(rngf, 3)) for _ = 1:3])
        u = randn(rngf, 3, 3)
        u[2, 3] = 0.0                                    # kill (u_O)_y only
        @test abs(evaluate_salc(dj, e, u)) < 1e-12
        # the dJ/dr half of gate (f): the 2-body bond-stretch sector (degree-1
        # displacement on the magnetic pair itself, soc = false) emits exactly
        # the 2 SALCs the oracle pins on the decorated bond
        # (test_countingoracle.jl: {E, σz} Frobenius count 2) — spanning
        # (ê₁·ê₂)(u₁ − u₂)_x (the bond stretch, dJ/dr) and
        # (ê₁·ê₂)(u₁ + u₂)_y (the pair sway against the ligand)
        stretch = [s for s in b.salcs
                   if s.body == 2 &&
                      sum(SLCE.disp_degree(dd) for dd in s.key.decors) == 1]
        @test length(stretch) == 2
        @test all(s -> s.key.L_S == 0, stretch)
        # fresh stream: keeps the fit samples independent of how many SALCs
        # the invariance loop above happened to draw for
        rngs = MersenneTwister(0xd10d)
        As = zeros(10, 2)
        Ys = zeros(10, 2)
        for m = 1:10
            e = reduce(hcat, [normalize(randn(rngs, 3)) for _ = 1:3])
            u = randn(rngs, 3, 3) * 0.4
            sc = dot(e[:, 1], e[:, 2])
            As[m, 1] = sc * (u[1, 1] - u[1, 2])
            As[m, 2] = sc * (u[2, 1] + u[2, 2])
            for (k, s) in enumerate(stretch)
                Ys[m, k] = evaluate_salc(s, e, u)
            end
        end
        Cs = As \ Ys
        @test norm(As * Cs .- Ys) < 1e-9 * norm(Ys)
        @test abs(det(Cs)) > 1e-6              # both forms realized independently
        # flipping the ligand sector to soc = true ADDS exactly the DMI-like
        # L_S ≥ 1 ligand blocks (total = the oracle's free count 7 = 1 + 2 + 4)
        # and leaves the L_S = 0 part bitwise unchanged
        bs = SLCE.build_salc_basis(cr, sg, cs, mkspec(true); neighbors = nl)
        c3 = [s for s in bs.salcs if s.body == 3]
        @test length(c3) == 7
        @test count(s -> s.key.L_S == 1, c3) == 2
        @test count(s -> s.key.L_S == 2, c3) == 4
        c30 = [s for s in c3 if s.key.L_S == 0]
        @test [s.key for s in c30] == [s.key for s in b3]
        for (x, y) in zip(c30, b3)
            @test same_members(x.members, y.members)
        end
    end

    @testset "gate (e2): cubic l=2 × shell p=1 — the two-constant magnetoelastic form" begin
        # THE B₁/B₂ gate (design record §12; the 2026-07-25 correction: the
        # ε-linear magnetoelastic constants live in the l = 2 spin ×
        # neighbor-shell p = 1 relative-displacement sectors, NOT in the
        # single-site p = 2 block). Fixture: an octahedral Fe(O)₆ unit in a
        # box with the full 48-op O_h group — a spin-carrying center (l = 2)
        # and a p = 1 displacement shell.
        L = 10.0
        c = SVector{3,Float64}(0.5, 0.5, 0.5)
        dsh = 2.0
        shell = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0),
                 SVector(0.0, 1.0, 0.0), SVector(0.0, -1.0, 0.0),
                 SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0)]
        offs = vcat([SVector{3,Float64}(0, 0, 0)], [dsh * v for v in shell])
        frac = reduce(hcat, [c + o / L for o in offs])
        cr = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 2, 2, 2, 2, 2, 2],
                     ["Fe", "O"])
        rots = oh48_matrices()
        trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
        sg = _assemble_spacegroup(cr, rots, trans, "Pm-3m", 0; tol = 1e-6)
        nl = build_neighbor_list(cr, 2.1)                # center–shell bonds only
        cs = build_clusters(cr, nl, sg; nbody = 2)
        # a spin-only-times-p=1 spec has odd leading displacement degree — the
        # boundedness warning must fire (and is the expected shape here: e2 is
        # a subblock gate, not a standalone model)
        spec = @test_logs (:warn, r"odd") BasisSpec(cr;
            lmax = ["Fe" => 2, "O" => 0], pmax = ["Fe" => 0, "O" => 1],
            sectors = [Sector(spin = [2], disp = (degree = 1,), sites = 2,
                              cutoff = 2.1)])
        b = SLCE.build_salc_basis(cr, sg, cs, spec; neighbors = nl)
        # exactly TWO invariants on the bond orbit (the C4v count 2 = the two
        # cubic magnetoelastic constants), both in the SOC-carrying L_S = 2
        # sector, folded onto 6 physical bonds each
        @test length(b.salcs) == 2
        @test all(s -> s.key.L_S == 2, b.salcs)
        @test sort([s.key.Lf for s in b.salcs]) == [1, 3]
        @test all(s -> length(s.members) == 6, b.salcs)
        # ε-linear contraction: substitute the affine shell displacement
        # u_j = ε·d_j and fit against the two-constant cubic magnetoelastic
        # closed forms f_B1 = Σ_i ε_ii (n_i² − 1/3), f_B2 = Σ_{i≠j} ε_ij n_i n_j
        # — the substituted SALCs must lie EXACTLY in their span (the O_h
        # count: Y₂ = E_g ⊕ T_2g against ε_sym = A_1g ⊕ E_g ⊕ T_2g pairs to
        # exactly E_g·E_g ⊕ T_2g·T_2g, no third invariant) and realize both
        # invariant directions independently. The substitution is exact only
        # because every bond member is intracell AND the center's own
        # displacement is clamped (pmax Fe = 0): an image-crossing member
        # would need u = ε·(d_j + R), which the per-atom `u` layout cannot
        # express — homogeneous strain on periodic cells is the K(ε)
        # channel's job (M5), not this gate's.
        rvec = [Vector(dsh * v) for v in shell]
        function ucfg(eps)
            u = zeros(3, 7)
            for j = 1:6
                u[:, j + 1] = eps * rvec[j]
            end
            return u
        end
        fB1(n, eps) = sum(eps[i, i] * (n[i]^2 - 1 / 3) for i = 1:3)
        fB2(n, eps) = sum(eps[i, j] * n[i] * n[j] for i = 1:3 for j = 1:3 if i != j)
        rnge = MersenneTwister(0xe2)
        M = 14
        A2 = zeros(M, 2)
        Y2 = zeros(M, 2)
        for m = 1:M
            n = normalize(randn(rnge, 3))
            e = reduce(hcat, [normalize(randn(rnge, 3)) for _ = 1:7])
            e[:, 1] = n
            S = randn(rnge, 3, 3) * 0.1
            eps = (S + transpose(S)) / 2
            A2[m, 1] = fB1(n, eps)
            A2[m, 2] = fB2(n, eps)
            for (k, s) in enumerate(b.salcs)
                Y2[m, k] = evaluate_salc(s, e, ucfg(eps))
            end
        end
        C2 = A2 \ Y2
        @test norm(A2 * C2 .- Y2) < 1e-10 * norm(Y2)
        # both invariant directions realized independently (C2 is the fit
        # gauge — no B₁/B₂ normalization or sign convention is pinned here)
        @test abs(det(C2)) > 1e-6
        # origin-independence / strain-only content of the shell sum. Both
        # kills are exact O_h statements (NOT the SO(3) "2 ⊗ 1 has no scalar"
        # argument, which O_h ⊄ SO(3) would not support): a uniform shell
        # translation dies by parity (Y₂(n) is inversion-even, t polar-odd,
        # and the would-be invariant is t-linear), a rigid rotation because
        # ω ~ T_1g and E_g⊗T_1g ⊕ T_2g⊗T_1g contains no A_1g
        n = normalize(randn(rnge, 3))
        e = reduce(hcat, [normalize(randn(rnge, 3)) for _ = 1:7])
        e[:, 1] = n
        t = randn(rnge, 3)
        ut = zeros(3, 7)
        for j = 2:7
            ut[:, j] = t
        end
        w = randn(rnge, 3)
        W = [0.0 -w[3] w[2]; w[3] 0.0 -w[1]; -w[2] w[1] 0.0]
        for s in b.salcs
            @test abs(evaluate_salc(s, e, ut)) < 1e-12
            @test abs(evaluate_salc(s, e, ucfg(W))) < 1e-12
        end
    end

    @testset "gate (n): sector-mask ≡ soc-false basis on a mixed spec" begin
        # Two selection routes over one mixed multi-sector spec: build with
        # every sector soc = true and MASK the result to L_S = 0, or rebuild
        # with every sector soc = false. They must agree bitwise — the
        # soc-vs-sector_mask drift fence (design record §13 risk 4) on a spec
        # that exercises pure-spin, mixed, and pure-displacement sectors plus
        # the global lsum cap at once.
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        mk(soc) = BasisSpec(cr; lmax = 2, pmax = 2, lsum = 4, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.5, soc = soc),
            Sector(spin = [1, 1], disp = (degree = 1:2,), sites = 2, cutoff = 1.5,
                   soc = soc),
            Sector(disp = (degree = 2,), cutoff = 1.5, soc = soc)])
        full = SLCEBasis(cr, mk(true))
        masked = [s for s in salcs(full) if s.key.L_S == 0]
        direct = SLCEBasis(cr, mk(false))
        @test 0 < length(masked) < n_salcs(full)         # the mask bites
        @test [s.key for s in masked] == [s.key for s in salcs(direct)]
        for (a, b) in zip(masked, salcs(direct))
            @test same_members(a.members, b.members)
        end
    end

    @testset "gates (d)/(a): dense ≡ p = 0-restricted sector build, span + MC surface" begin
        # Gate (d), span equivalence at spec level: the joint machinery
        # restricted to p = 0 (spin-only sectors) reproduces the legacy dense
        # pure-spin basis — bitwise (stronger than span), and down to the fit
        # layer's span carriers, the energy AND torque design matrices.
        latc = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        chain = Crystal(latc, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75],
                        [1, 1, 1, 1], ["Fe"])
        dense = SLCEBasis(chain, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [2]))
        sect = SLCEBasis(chain, BasisSpec(chain; lmax = 2,
            sectors = [Sector(spin = (sites = 1:2,), cutoff = 2.6)]))
        @test _sector_basis_identical(dense, sect)
        rngd = MersenneTwister(0xd00d)
        cfgs = [randcfg(rngd, 4) for _ = 1:5]
        E = randn(rngd, 5)
        T = [randn(rngd, 3, 4) for _ = 1:5]
        da = SLCEDataset(dense, cfgs, E, T)
        db = SLCEDataset(sect, cfgs, E, T)
        @test da.X_E == db.X_E
        @test da.X_T == db.X_T
        # Gate (a), spec-level re-run of the relabel bit-identity: the MC
        # consumption surface (`multipole_terms` — the unmigrated
        # SLCEMonteCarlo's pure-spin path) is field-for-field identical
        # between the two builds under identical coefficients.
        jphi = randn(rngd, n_salcs(dense))
        ta = multipole_terms(SLCEModel(dense, 0.7, jphi))
        tb = multipole_terms(SLCEModel(sect, 0.7, jphi))
        @test length(ta) == length(tb) > 0
        for (x, y) in zip(ta, tb)
            @test x.coef === y.coef && x.body == y.body && x.atoms == y.atoms &&
                  x.shifts == y.shifts && x.ls == y.ls && x.folded == y.folded
        end
    end

    @testset "gate (i) full: u = 0 bitwise degeneracy vs the dense spin basis" begin
        # The joint model at u = 0 degenerates to the spin model BITWISE: the
        # pure-spin subset of a mixed build is the dense spin-only build
        # (keys + members), and joint evaluation at u = 0 returns the very
        # same floats (===) the dense basis returns, with every
        # displacement-decorated SALC exactly zero.
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        dense = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2]))
        mixed = SLCEBasis(cr, BasisSpec(cr; lmax = 2, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.5),
            Sector(spin = [1, 1], disp = (degree = 1:2,), sites = 2,
                   cutoff = 1.5)]))
        pure = [s for s in salcs(mixed) if all(is_pure_spin, s.key.decors)]
        @test n_salcs(dense) == 44               # the standard-cell column count
        @test !isempty(pure)
        @test length(pure) < n_salcs(mixed)
        @test [s.key for s in pure] == [s.key for s in salcs(dense)]
        for (a, b) in zip(pure, salcs(dense))
            @test same_members(a.members, b.members)
        end
        rngi = MersenneTwister(0x1f)
        u0 = zeros(3, 2)
        for _ = 1:4
            e = randcfg(rngi, 2)
            vals = Dict(s.key => evaluate_salc(s, e, u0) for s in salcs(mixed))
            for s in salcs(dense)
                @test vals[s.key] === evaluate_salc(s, e)
            end
            for s in salcs(mixed)
                all(is_pure_spin, s.key.decors) || @test vals[s.key] == 0.0
            end
        end
    end

    @testset "mixed basis persists and reloads verbatim" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        spec = BasisSpec(cr; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 0:2,), sites = 2, cutoff = 1.5)])
        b = SLCEBasis(cr, spec)
        path = joinpath(mktempdir(), "mixed.toml")
        SLCE.save(path, b)
        b2 = SLCE.load(SLCEBasis, path)
        @test b2.spec == spec
        @test b2.salc_basis.fingerprint == b.salc_basis.fingerprint
        @test _sector_basis_identical(b, b2)
    end
end
