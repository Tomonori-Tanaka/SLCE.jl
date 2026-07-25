# Sector-driven basis construction (M2b-3b) at the public-API level: the label
# generators, the sector-expressed pure-spin ≡ legacy-dense bit-identity,
# per-sector cutoff re-admission, per-species lmax/pmax engine caps, u = 0
# exactness, and persistence of a mixed basis. Engine-level gates (counts vs
# the CountingOracle, invariance, soc subsets) live in test_mixedsalc.jl.

using Test
using SLCE
using SLCE: SiteDecor, evaluate_salc, salcs, build_neighbor_list, build_clusters,
    _spin_multisets, _disp_multisets, _marry_multisets, _sector_orbit_labels,
    LSUM_UNCAPPED, is_pure_spin, SPIN, DISP
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
            sectors = [Sector(spin = (nbody = 1:2,), cutoff = 1.5)]))
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
                sectors = [Sector(spin = (nbody = 1:2,), cutoff = 2.6,
                                  soc = soc)]))
            @test _sector_basis_identical(lg, sc)
            # equivalently: the sector-local lsum
            sc2 = SLCEBasis(chain, BasisSpec(chain; lmax = 3,
                sectors = [Sector(spin = (nbody = 1:2, lsum = 4), cutoff = 2.6,
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
            Sector(disp = (degree = 2,), nbody = 2, cutoff = 5.2)])
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
            Sector(spin = (nbody = 1:2,), cutoff = 1.5),
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
            Sector(spin = (nbody = 2,), cutoff = 1.5),
            Sector(spin = [1, 1], disp = (degree = 1:2,), nbody = 2, cutoff = 1.5),
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

    @testset "mixed basis persists and reloads verbatim" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        spec = BasisSpec(cr; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 0:2,), nbody = 2, cutoff = 1.5)])
        b = SLCEBasis(cr, spec)
        path = joinpath(mktempdir(), "mixed.toml")
        SLCE.save(path, b)
        b2 = SLCE.load(SLCEBasis, path)
        @test b2.spec == spec
        @test b2.salc_basis.fingerprint == b.salc_basis.fingerprint
        @test _sector_basis_identical(b, b2)
    end
end
