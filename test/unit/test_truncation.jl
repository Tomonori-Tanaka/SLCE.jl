# BasisSpec truncation: sugar → canonical resolution (labels, wildcards,
# specificity, error paths), the per-body lsum filter on the SALC enumeration,
# and per-body / per-species-pair cutoffs in the neighbor list and cluster
# admission (checked against an independent brute force).

using Test
using SLCE
using SLCE: build_neighbor_list, candidate_clusters, cartesian_positions,
    n_salcs, read_setup, salcs
using StaticArrays: SVector
using LinearAlgebra: I, norm
using TOML

const _TRUNC_LABELS = ["Nd", "Fe", "B"]

# A cubic 3-species toy crystal: Nd at the corner, Fe at the body center, B at a
# face center — three distinct interatomic distances (Nd-Fe √3/2·a, Nd-B a/2,
# Fe-B a/2 offsets differ), convenient for per-pair cutoff assertions.
function _trunc_crystal(; a = 4.0)
    lat = Lattice(Matrix(a * I(3)))
    frac = [0.0 0.5 0.5; 0.0 0.5 0.5; 0.0 0.5 0.0]
    return Crystal(lat, frac, [1, 2, 3], _TRUNC_LABELS)
end

@testset "BasisSpec truncation" begin
    @testset "sugar resolution: lmax forms" begin
        sp = BasisSpec(_TRUNC_LABELS; nbody = 2, cutoff = Inf,
                       lmax = ["*" => 3, "B" => 0])
        @test sp.lmax == [3, 3, 0]
        @test sp.species_labels == _TRUNC_LABELS
        @test BasisSpec(_TRUNC_LABELS; nbody = 2, cutoff = Inf, lmax = 2).lmax ==
              [2, 2, 2]
        @test BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1, 2, 0]).lmax == [1, 2, 0]
        # dict form and pair-vector form agree
        @test BasisSpec(_TRUNC_LABELS; nbody = 2, cutoff = Inf,
                        lmax = Dict("*" => 1, "Nd" => 2)).lmax == [2, 1, 1]
        # crystal-first constructor picks up the labels
        cr = _trunc_crystal()
        @test BasisSpec(cr; nbody = 2, cutoff = Inf, lmax = ["*" => 1]).species_labels ==
              _TRUNC_LABELS
        # errors: unknown label, no coverage, duplicate key, label-keyed w/o labels
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, cutoff = Inf,
                                             lmax = ["*" => 1, "Xx" => 2])
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, cutoff = Inf,
                                             lmax = ["Nd" => 1])
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, cutoff = Inf,
                                             lmax = ["*" => 1, "*" => 2])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = Inf,
                                             lmax = ["*" => 1])
        # the removed keyword points at the replacement
        @test_throws ArgumentError BasisSpec(; nbody = 2, pair_cutoff = 3.0,
                                             cutoff = 3.0, lmax = [1])
    end

    @testset "sugar resolution: lsum forms" begin
        sp = BasisSpec(_TRUNC_LABELS; nbody = 3, cutoff = Inf, lmax = ["*" => 3],
                       lsum = [1 => 0, 2 => 4, 3 => 4])
        @test sp.lsum == [0, 4, 4]
        @test BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1], lsum = 2).lsum == [2, 2]
        sp2 = BasisSpec(; nbody = 3, cutoff = Inf, lmax = [1], lsum = [2 => 4])
        @test sp2.lsum == [SLCE.LSUM_UNCAPPED, 4, SLCE.LSUM_UNCAPPED]
        @test BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1]).lsum ==
              fill(SLCE.LSUM_UNCAPPED, 2)
        # body order outside 1:nbody / duplicates / negatives error
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1],
                                             lsum = [3 => 4])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1],
                                             lsum = [2 => 4, 2 => 6])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1],
                                             lsum = -1)
    end

    @testset "sugar resolution: cutoff forms and specificity" begin
        # scalar broadcast
        sp = BasisSpec(_TRUNC_LABELS; nbody = 3, cutoff = 5.0, lmax = ["*" => 1])
        @test length(sp.cutoff) == 2 && all(M -> M == fill(5.0, 3, 3), sp.cutoff)
        # one pair table for all body orders; specificity concrete > "A-*" > "*-*"
        sp = BasisSpec(_TRUNC_LABELS; nbody = 3, lmax = ["*" => 1],
                       cutoff = ["*-*" => 8.0, "Fe-*" => 6.0, "Fe-Fe" => 4.0])
        @test sp.cutoff[1] == sp.cutoff[2]
        M = sp.cutoff[1]
        @test M[2, 2] == 4.0                      # Fe-Fe concrete
        @test M[1, 2] == M[2, 1] == 6.0           # Nd-Fe via Fe-*
        @test M[1, 1] == M[1, 3] == M[3, 3] == 8.0
        # body-keyed: scalar and pair table per order
        sp = BasisSpec(_TRUNC_LABELS; nbody = 3, lmax = ["*" => 1],
                       cutoff = [2 => Inf, 3 => ["B-*" => 0.0, "*-*" => 5.0]])
        @test all(isinf, sp.cutoff[1])
        @test sp.cutoff[2][3, 1] == 0.0 && sp.cutoff[2][1, 2] == 5.0
        # unordered pair keys: "Nd-Fe" ≡ "Fe-Nd" (duplicate errors)
        @test BasisSpec(_TRUNC_LABELS; nbody = 2, lmax = ["*" => 1],
                        cutoff = ["Nd-Fe" => 3.0, "*-*" => 5.0]).cutoff[1][2, 1] == 3.0
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, lmax = ["*" => 1],
                                             cutoff = ["Nd-Fe" => 3.0, "Fe-Nd" => 3.0,
                                                       "*-*" => 5.0])
        # equal-specificity conflict / uncovered pair / body outside 2:nbody / -1
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, lmax = ["*" => 1],
                                             cutoff = ["Fe-*" => 4.0, "*-Nd" => 6.0,
                                                       "*-*" => 8.0])
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, lmax = ["*" => 1],
                                             cutoff = ["Fe-Fe" => 4.0])
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 2, lmax = ["*" => 1],
                                             cutoff = [1 => 4.0, 2 => 5.0])
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; nbody = 3, lmax = ["*" => 1],
                                             cutoff = [2 => 4.0])   # body3 uncovered
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = -1, lmax = [1])
        # canonical Vector{Matrix} round-trips through the constructor
        sp2 = BasisSpec(_TRUNC_LABELS; nbody = 3, lmax = ["*" => 1], cutoff = sp.cutoff)
        @test sp2.cutoff == sp.cutoff && sp2 == sp
    end

    @testset "_enumerate_ls honors the per-body Σl budget" begin
        perms2 = [[1, 2], [2, 1]]
        ls = SLCE._enumerate_ls(2, [1, 1], [3], 4, perms2)
        @test sort(ls) == [[1, 1], [1, 3], [2, 2]]        # Σl ≤ 4, even, canonical
        # an over-generous lmax is tightened by the budget, not enumerated
        @test sort(SLCE._enumerate_ls(2, [1, 1], [6], 4, perms2)) ==
              [[1, 1], [1, 3], [2, 2]]
        # uncapped reproduces the plain lmax product
        @test sort(SLCE._enumerate_ls(2, [1, 1], [2], SLCE.LSUM_UNCAPPED,
                                            perms2)) == [[1, 1], [2, 2]]
        # N = 1: lsum caps the single site
        @test SLCE._enumerate_ls(1, [1], [4], 0, [[1]]) == Vector{Int}[]
        @test SLCE._enumerate_ls(1, [1], [4], 2, [[1]]) == [[2]]
    end

    @testset "lsum gates on a real basis (bcc, one species)" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        build(spec) = SLCEBasis(cr, spec)
        # lsum = (0, 2) with lmax = 3 ≡ lmax = 1 unrestricted (the l02 equivalence;
        # body1 must be budgeted too — lmax = 1 kills it by parity, lsum by cap)
        b_l1 = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1], soc = false))
        b_s2 = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [3],
                               lsum = [1 => 0, 2 => 2], soc = false))
        @test [s.key for s in salcs(b_s2)] == [s.key for s in salcs(b_l1)]
        # a big cap ≡ no cap
        b_free = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [2]))
        b_big = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [2], lsum = 100))
        @test [s.key for s in salcs(b_big)] == [s.key for s in salcs(b_free)]
        # the budget strictly trims an l044-style basis
        b_all = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [3]))
        b_044 = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [3], lsum = [2 => 4]))
        @test 0 < n_salcs(b_044) < n_salcs(b_all)
        @test all(k -> sum(SLCE.spin_ls(k)) <= 4, (s.key for s in salcs(b_044)))
        @test any(k -> sum(SLCE.spin_ls(k)) > 4, (s.key for s in salcs(b_all)))
        # lsum[1] = 0 removes single-site terms without touching pairs
        b_no1 = build(BasisSpec(; nbody = 2, cutoff = Inf, lmax = [2],
                                lsum = [1 => 0]))
        @test all(k -> k.body == 2, (s.key for s in salcs(b_no1)))
        @test [k for k in (s.key for s in salcs(b_no1))] ==
              [k for k in (s.key for s in salcs(b_free)) if k.body == 2]
    end

    @testset "per-pair neighbor list" begin
        cr = _trunc_crystal(; a = 4.0)
        # a matrix filled with one radius ≡ the scalar method (identical pair sets)
        for images in (MinimumImage(), AllImages())
            nls = build_neighbor_list(cr, 3.0, images)
            nlm = build_neighbor_list(cr, fill(3.0, 3, 3), images)
            key(p) = (p.i, p.j, Tuple(p.shift))
            @test sort!(map(key, nlm.pairs)) == sort!(map(key, nls.pairs))
        end
        # zero radius excludes exactly that species pair
        cut = fill(6.0, 3, 3)
        cut[1, 3] = cut[3, 1] = 0.0                        # kill Nd-B
        nl = build_neighbor_list(cr, cut, MinimumImage())
        sp = cr.species
        @test !any(p -> Set((sp[p.i], sp[p.j])) == Set((1, 3)), nl.pairs)
        @test any(p -> Set((sp[p.i], sp[p.j])) == Set((1, 2)), nl.pairs)
        # every admitted pair obeys its own species-pair radius
        cut2 = [2.5 3.5 0.0; 3.5 2.5 6.0; 0.0 6.0 2.5]
        nl2 = build_neighbor_list(cr, cut2, MinimumImage())
        @test all(p -> p.distance <= cut2[sp[p.i], sp[p.j]] * (1 + 1e-8), nl2.pairs)
        # validation: asymmetric / negative / wrong size
        bad = fill(3.0, 3, 3); bad[1, 2] = 4.0
        @test_throws ArgumentError build_neighbor_list(cr, bad, MinimumImage())
        @test_throws ArgumentError build_neighbor_list(cr, fill(-1.0, 3, 3),
                                                       MinimumImage())
        @test_throws ArgumentError build_neighbor_list(cr, fill(3.0, 2, 2),
                                                       MinimumImage())
        # AllImages rejects Inf entries
        infm = fill(3.0, 3, 3); infm[2, 2] = Inf
        @test_throws ArgumentError build_neighbor_list(cr, infm, AllImages())
    end

    @testset "per-body per-pair cluster admission (vs brute force)" begin
        # An X-Y-X line (open boundaries): X-Y = 1.0 Å each, X-X = 2.0 Å.
        lat = Lattice(Matrix(10.0 * I(3)); pbc = (false, false, false))
        frac = [0.1 0.2 0.3; 0.0 0.0 0.0; 0.0 0.0 0.0]
        cr = Crystal(lat, frac, [1, 2, 1], ["X", "Y"])
        cutmat(xx, xy, yy) = [xx xy; xy yy]
        nl = build_neighbor_list(cr, fill(3.0, 2, 2), MinimumImage())
        # X-X edge (2.0 Å) beyond its own radius ⇒ no triplet, even though both
        # X-Y edges (1.0 Å) are admissible
        c3a = candidate_clusters(cr, nl, 3;
                                 cutoff = [cutmat(3.0, 3.0, 3.0),
                                           cutmat(1.5, 1.2, 3.0)])
        @test isempty(c3a[3])
        @test !isempty(c3a[2])                     # pairs survive under body2 = 3.0 Å
        # widen X-X for body 3 ⇒ exactly the one X-Y-X triplet appears (× the 3!
        # ordered anchored variants the enumeration emits before orbit reduction)
        c3b = candidate_clusters(cr, nl, 3;
                                 cutoff = [cutmat(3.0, 3.0, 3.0),
                                           cutmat(2.5, 1.2, 3.0)])
        @test length(c3b[3]) == 6
        @test all(sort(m.atoms) == [1, 2, 3] for m in c3b[3])
        # body-2 radii trim pairs independently of body-3 radii
        c2 = candidate_clusters(cr, nl, 3;
                                cutoff = [cutmat(1.5, 1.2, 3.0),
                                          cutmat(2.5, 1.2, 3.0)])
        @test all(Set((m.atoms[1], m.atoms[2])) != Set((1, 3)) for m in c2[2])
        @test length(c2[3]) == 6                   # body3 unaffected by body2 trim
        # brute force: every ordered anchored N-tuple whose edges all fit their own radii
        function brute(cutoffs)
            cart = SLCE.cartesian_positions(cr)
            pos(a) = SVector{3,Float64}(cart[1, a], cart[2, a], cart[3, a])
            sp = cr.species
            found = Dict(2 => 0, 3 => 0)
            for N = 2:3, tup in Iterators.product(fill(1:3, N)...)
                allunique(tup) || continue
                ok = all(norm(pos(tup[x]) - pos(tup[y])) <=
                             cutoffs[N - 1][sp[tup[x]], sp[tup[y]]]
                         for x = 1:N for y = (x + 1):N)
                ok && (found[N] += 1)
            end
            return found
        end
        # both image selections must agree with the brute force (the open cell has
        # no periodic images, so AllImages sees the same geometry — the point is to
        # pin its own percut2 branch)
        nl_ai = build_neighbor_list(cr, fill(3.0, 2, 2), AllImages())
        for cutoffs in ([cutmat(3.0, 3.0, 3.0), cutmat(1.5, 1.2, 3.0)],
                        [cutmat(3.0, 3.0, 3.0), cutmat(2.5, 1.2, 3.0)],
                        [cutmat(1.5, 1.2, 3.0), cutmat(2.5, 1.2, 3.0)],
                        [cutmat(1.5, 0.0, 3.0), cutmat(2.5, 1.2, 0.0)])
            b = brute(cutoffs)
            c = candidate_clusters(cr, nl, 3; cutoff = cutoffs)
            @test length(c[2]) == b[2] && length(c[3]) == b[3]
            c_ai = candidate_clusters(cr, nl_ai, 3; selection = AllImages(),
                                      cutoff = cutoffs)
            @test length(c_ai[2]) == b[2] && length(c_ai[3]) == b[3]
        end
    end

    @testset "SLCEBasis end-to-end with per-body cutoffs" begin
        cr = _trunc_crystal(; a = 4.0)
        # body2 keeps everything, body3 nothing: identical body-2 SALCs to a plain
        # nbody = 2 build, and no 3-body SALCs at all
        b2 = SLCEBasis(cr, BasisSpec(cr; nbody = 2, cutoff = Inf, lmax = ["*" => 1],
                                    soc = false))
        b23 = SLCEBasis(cr, BasisSpec(cr; nbody = 3, lmax = ["*" => 1], soc = false,
                                     cutoff = [2 => Inf, 3 => 0.0]))
        @test [s.key for s in salcs(b23)] == [s.key for s in salcs(b2)]
        # per-pair exclusion at the basis level: no SALC touches an excluded pair
        bx = SLCEBasis(cr, BasisSpec(cr; nbody = 2, lmax = ["*" => 1], soc = false,
                                    cutoff = ["Nd-B" => 0.0, "*-*" => Inf]))
        sp = cr.species
        @test all(s -> Set(sp[a] for a in s.members[1].atoms) != Set((1, 3)),
                  salcs(bx))
        @test n_salcs(bx) < n_salcs(b2)
    end

    @testset "persistence round-trip (v3) and legacy v2 read" begin
        cr = _trunc_crystal(; a = 4.0)
        spec = BasisSpec(cr; nbody = 3, soc = false,
                         lmax = ["*" => 1],
                         lsum = [2 => 4],
                         cutoff = [2 => Inf, 3 => ["B-*" => 0.0, "*-*" => 3.7]])
        b = SLCEBasis(cr, spec)
        path = joinpath(mktempdir(), "trunc.toml")
        SLCE.save(path, b)
        b2 = SLCE.load(SLCEBasis, path)
        @test b2.spec == spec                       # Inf + LSUM_UNCAPPED + labels intact
        @test n_salcs(b2) == n_salcs(b)
        # a legacy v2 document (scalar pair_cutoff) still loads, expanded
        doc = SLCE._to_doc(SLCEBasis(cr, BasisSpec(cr; nbody = 2, cutoff = 3.7,
                                                        lmax = ["*" => 1])))
        doc["schema_version"] = 2
        doc["spec"] = Dict{String,Any}("nbody" => 2, "pair_cutoff" => 3.7,
                                       "lmax" => [1, 1, 1], "isotropy" => false)
        p2 = joinpath(mktempdir(), "legacy.toml")
        open(p2, "w") do io
            TOML.print(io, doc)
        end
        bl = SLCE.load(SLCEBasis, p2)
        @test bl.spec.cutoff == [fill(3.7, 3, 3)]
        @test bl.spec.lsum == fill(SLCE.LSUM_UNCAPPED, 2)
    end

    @testset "input.toml: new interaction schema" begin
        dir = mktempdir()
        body = """
        [structure]
        lattice        = [[4.0,0,0],[0,4.0,0],[0,0,4.0]]
        positions      = [[0,0,0],[0.5,0.5,0.5],[0.5,0.5,0.0]]
        species        = [1,2,3]
        species_labels = ["Nd","Fe","B"]

        [interaction]
        nbody = 3
        soc = false

        [interaction.lmax]
        "*" = 1
        B   = 0

        [interaction.lsum]
        1 = 0
        2 = 4
        3 = 4

        [interaction.cutoff]
        2 = 8.0

        [interaction.cutoff.3]
        "Fe-*" = 6.0
        "*-*"  = 8.0
        """
        p = joinpath(dir, "new.toml")
        write(p, body)
        inp = read_setup(p)
        @test inp.spec.lmax == [1, 1, 0]
        @test inp.spec.lsum == [0, 4, 4]
        @test inp.spec.cutoff[1] == fill(8.0, 3, 3)
        @test inp.spec.cutoff[2][2, 1] == 6.0 && inp.spec.cutoff[2][1, 1] == 8.0
        @test inp.spec.species_labels == ["Nd", "Fe", "B"]
        # the removed key errors with the migration hint (placed inside
        # [interaction] itself, before the first sub-table header)
        p2 = joinpath(dir, "old.toml")
        write(p2, replace(body, "nbody = 3\nsoc = false" =>
                                "nbody = 3\nsoc = false\npair_cutoff = 8.0"))
        err = try
            read_setup(p2)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("replaced by `cutoff`", err.msg)
    end

    @testset "sector table: Sector sugar → SectorRule resolution" begin
        # explicit spin multiset + displacement degree budget (Int/range forms)
        sp = BasisSpec(_TRUNC_LABELS; lmax = 2, pmax = ["*" => 0, "Fe" => 2],
                       sectors = [Sector(spin = [1, 1], disp = 1:2, cutoff = 5.0)])
        r = sp.sectors[1]
        @test r.spin_mode == :explicit && r.spin_ls == [1, 1]
        @test r.spin_nsites == (2, 2) && r.disp_degree == (1, 2)
        @test r.sites == (2, 4)          # spin pair + up to two degree-1 disp sites
        @test r.soc && r.cutoff == fill(5.0, 3, 3)
        @test sp.nbody == 4 && sp.pmax == [0, 2, 0] && sp.soc
        @test sp.disp_scale == 1.0 && sp.lsum == fill(SLCE.LSUM_UNCAPPED, 4)
        @test all(M -> M == fill(5.0, 3, 3), sp.cutoff)      # derived envelope 2:4
        # single-Int spin, NamedTuple disp, explicit sites intersection
        s2 = BasisSpec(_TRUNC_LABELS; lmax = 2, pmax = 2,
                       sectors = Sector(spin = 2, disp = (degree = 1:2,),
                                        sites = 1, cutoff = 4.0)).sectors[1]
        @test s2.spin_mode == :explicit && s2.spin_ls == [2]
        @test s2.sites == (1, 1)         # both factors forced onto one site
        # general-truncation spin sector + sector-local caps
        s3 = BasisSpec(_TRUNC_LABELS; lmax = 3,
                       sectors = Sector(spin = (sites = 2:3, lmax = 2, lsum = 4),
                                        cutoff = 8.0)).sectors[1]
        @test s3.spin_mode == :any && s3.spin_nsites == (2, 3)
        @test s3.spin_lmax == 2 && s3.spin_lsum == 4 && s3.sites == (2, 3)
        @test s3.disp_degree == (0, 0)
        # lattice-only sector; pure-disp needs pmax
        s4 = BasisSpec(_TRUNC_LABELS; lmax = 0, pmax = 4,
                       sectors = Sector(disp = (degree = 2:4,),
                                        cutoff = 6.0)).sectors[1]
        @test s4.spin_mode == :none && s4.spin_nsites == (0, 0)
        @test s4.disp_degree == (2, 4) && s4.sites == (1, 4)
        # per-sector pair-table cutoff resolves with the labels
        s5 = BasisSpec(_TRUNC_LABELS; lmax = 1,
                       sectors = Sector(spin = (sites = 2,),
                                        cutoff = ["Fe-*" => 6.0,
                                                  "*-*" => 8.0])).sectors[1]
        @test s5.cutoff[2, 2] == 6.0 && s5.cutoff[1, 2] == 6.0
        @test s5.cutoff[1, 1] == 8.0 && s5.cutoff == s5.cutoff'
        @test s5.soc                     # default
        # soc = false is per sector
        s6 = BasisSpec(_TRUNC_LABELS; lmax = 1,
                       sectors = Sector(spin = [1, 1], soc = false,
                                        cutoff = 5.0)).sectors[1]
        @test !s6.soc
    end

    @testset "sector table: envelope, nbody, and validation errors" begin
        # envelope = per-body elementwise max over the admitting sectors
        sp = BasisSpec(_TRUNC_LABELS; lmax = 2, pmax = 2, sectors = [
            Sector(spin = (sites = 2,), cutoff = 8.0),
            Sector(spin = [1, 1], disp = 1:2, cutoff = 5.0),   # bodies 2:4
            Sector(disp = (degree = 2,), cutoff = 6.0)])       # bodies 1:2
        @test sp.nbody == 4
        @test sp.cutoff[1] == fill(8.0, 3, 3)                  # body 2: max(8,5,6)
        @test sp.cutoff[2] == fill(5.0, 3, 3)                  # body 3: sector 2 only
        @test sp.cutoff[3] == fill(5.0, 3, 3)                  # body 4: sector 2 only
        # explicit global nbody caps the table; a fully-excluded sector errors
        @test BasisSpec(_TRUNC_LABELS; lmax = 2, pmax = 2, nbody = 2,
                        sectors = [Sector(spin = [1, 1], disp = 1:2,
                                          cutoff = 5.0)]).nbody == 2
        @test_throws ArgumentError BasisSpec(_TRUNC_LABELS; lmax = 2, nbody = 2,
            sectors = [Sector(spin = (sites = 3:4,), cutoff = 5.0)])
        # sugar / consistency errors
        E = ArgumentError
        # `Sector()` itself constructs (sugar is resolved lazily) — the empty
        # content is rejected at spec resolution
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, sectors = [Sector()])
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1,
                                 sectors = [Sector(spin = [1, 2], cutoff = 5.0)])  # Σl odd
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1,
                                 sectors = [Sector(spin = (lmax = 2,), cutoff = 5.0)])  # no sites
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1,
                                 sectors = [Sector(spin = (sites = 2, bogus = 1),
                                                   cutoff = 5.0)])
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, pmax = 2,
                                 sectors = [Sector(disp = (degree = 0:0,),
                                                   cutoff = 5.0)])   # degree hi < 1
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, pmax = 2,
                                 sectors = [Sector(spin = [1, 1], disp = 1:2,
                                                   sites = 5, cutoff = 5.0)])
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, sectors = Sector[])
        # sector-mode keyword rules
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, cutoff = 5.0,
                                 sectors = [Sector(spin = [1, 1])])   # cutoff is per sector
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, soc = false,
                                 sectors = [Sector(spin = [1, 1])])   # soc is per sector
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1,
                                 sectors = [Sector(disp = 1:2, cutoff = 5.0)])  # pmax missing
        # dense-form keyword rules
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, cutoff = Inf, nbody = 2,
                                 pmax = 2)                            # pmax needs sectors
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, cutoff = Inf)  # nbody required
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, nbody = 2)     # cutoff required
        # the deprecated keyword points at soc (inversion documented in the message)
        err = try
            BasisSpec(_TRUNC_LABELS; lmax = 1, nbody = 2, cutoff = Inf,
                      isotropy = true)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("soc = false", err.msg)
        # disp_scale validation (canonical constructor); ≠ 1 is refused until
        # the kernels apply it (M3) — a declared-but-inert unit convention lies
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, nbody = 2, cutoff = Inf,
                                 disp_scale = 0.0)
        @test_throws E BasisSpec(_TRUNC_LABELS; lmax = 1, nbody = 2, cutoff = Inf,
                                 disp_scale = 0.05)
        @test BasisSpec(_TRUNC_LABELS; lmax = 1, nbody = 2, cutoff = Inf,
                        disp_scale = 1.0).disp_scale == 1.0
        # an explicit spin multiset no species can carry is loud, not empty
        err2 = try
            BasisSpec(_TRUNC_LABELS; lmax = 1, sectors = [Sector(spin = [2, 2],
                                                                 cutoff = 5.0)])
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError && occursin("contribute nothing", err2.msg)
        # odd maximum total displacement degree warns (boundedness heuristic)
        @test_logs (:warn, r"odd") BasisSpec(_TRUNC_LABELS; lmax = 0, pmax = 3,
            sectors = [Sector(disp = (degree = 2:3,), cutoff = 6.0)])
        # == and show smoke
        s = Sector(spin = [1, 1], disp = 1:2, soc = false, cutoff = 5.0)
        @test s == Sector(spin = [1, 1], disp = 1:2, soc = false, cutoff = 5.0)
        @test occursin("soc = false", sprint(show, s))
        sp2 = BasisSpec(_TRUNC_LABELS; lmax = 2, pmax = 2,
                        sectors = [Sector(spin = [1, 1], disp = 2, cutoff = 5.0)])
        @test sp2 == BasisSpec(_TRUNC_LABELS; lmax = 2, pmax = 2,
                               sectors = [Sector(spin = [1, 1], disp = 2,
                                                 cutoff = 5.0)])
        @test occursin("SectorRule", sprint(show, sp2.sectors[1]))
        @test occursin("sector 1", sprint(show, MIME("text/plain"), sp2))
        # sector specs build (M2b-3b); the sector-driven content is gated in
        # test_sectorbasis.jl / test_mixedsalc.jl
        cr = _trunc_crystal()
        @test SLCEBasis(cr, sp2) isa SLCEBasis
    end
end
