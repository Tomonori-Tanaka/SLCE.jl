using Test
using SLCE
using SLCE: candidate_clusters, read_setup, _assemble_spacegroup, _design_energy
using StaticArrays
using LinearAlgebra
using Random

# The 8 diagonal sign operations diag(±1,±1,±1): the point group that maps the eight
# (±L/2,±L/2,±L/2) WS corners of a body-centred cubic cell into a single orbit.
function _cubic_sign_group(crystal)
    rots = SMatrix{3,3,Float64}[]
    for sx in (1.0, -1.0), sy in (1.0, -1.0), sz in (1.0, -1.0)
        push!(rots, SMatrix{3,3,Float64}(Diagonal([sx, sy, sz])))
    end
    trans = [SVector{3,Float64}(0, 0, 0) for _ in rots]
    return _assemble_spacegroup(crystal, rots, trans, "manual-cubic-signs", 0; tol = 1e-8)
end

# Brute-force minimum-image distance for atoms (i, j) over a generous image box —
# an independent oracle for the in-source search.
function _brute_min_dist(crystal, i, j; box = 4)
    A = crystal.lattice.vectors
    cart = SLCE.cartesian_positions(crystal)
    ri = SVector{3,Float64}(cart[:, i]...)
    rj = SVector{3,Float64}(cart[:, j]...)
    best = Inf
    pbc = crystal.lattice.pbc
    r = ntuple(d -> pbc[d] ? box : 0, 3)
    for n1 = -r[1]:r[1], n2 = -r[2]:r[2], n3 = -r[3]:r[3]
        (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
        d = norm(rj + A * SVector{3,Float64}(n1, n2, n3) - ri)
        d < best && (best = d)
    end
    return best
end

pairset(nl) = Set((p.i, p.j, Tuple(p.shift)) for p in nl.pairs)

# Canonical identity of a cluster instance: sites sorted by (atom, shift), shifts
# re-anchored to the first — the same class the orbit builder groups by, so two
# enumerations of the same physical cluster compare equal whatever anchor found it.
function _canon_cluster(atoms, shifts)
    perm = sortperm(1:length(atoms); by = i -> (atoms[i], Tuple(shifts[i])))
    s0 = shifts[perm[1]]
    return (Int[atoms[k] for k in perm], [Tuple(shifts[k] - s0) for k in perm])
end

@testset "image selection (minimum image / Wigner–Seitz)" begin
    @testset "cutoff < L/2: MinimumImage ≡ AllImages (the resolvable regime)" begin
        # 4-atom chain (c = 10, NN = 2.5, L/2 = 4 in the short x/y dirs), cutoff 2.6
        lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
        a = build_neighbor_list(cr, 2.6, AllImages())
        m = build_neighbor_list(cr, 2.6, MinimumImage())
        @test pairset(a) == pairset(m)
        @test !isempty(m.pairs)
    end

    @testset "cubic body-center: (L/2,L/2,L/2) corner is an 8-fold WS tie" begin
        L = 3.0
        lat = Lattice(Matrix(L * I(3)))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        nl = build_neighbor_list(cr, Inf, MinimumImage())     # full Wigner–Seitz cell
        corner = sqrt(3) * L / 2
        d12 = [p.distance for p in nl.pairs if (p.i, p.j) == (1, 2)]
        @test length(d12) == 8                                  # 8 equidistant corner images
        @test all(d -> isapprox(d, corner; rtol = 1e-10), d12)
        # no aliased farther images of a cross pair: its WS-cell max distance is the
        # corner distance (i==i self-pairs, the nearest self-image at a full lattice
        # vector, are excluded — they are not aliases of a shorter bond)
        @test maximum(p.distance for p in nl.pairs if p.i != p.j) ≤ corner * (1 + 1e-10)
    end

    @testset "L/2 face is a 2-fold tie; both equidistant images kept" begin
        lat = Lattice([2.0 0 0; 0 10 0; 0 0 10]; pbc = (true, false, false))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])     # pair at L/2 = 1
        nl = build_neighbor_list(cr, Inf, MinimumImage())
        d12 = [p.distance for p in nl.pairs if (p.i, p.j) == (1, 2)]
        @test length(d12) == 2
        @test all(d -> isapprox(d, 1.0; rtol = 1e-12), d12)
    end

    @testset "cutoff > L/2: MinimumImage drops the periodic alias, AllImages keeps it" begin
        # period c = 6, pair at 2.0; its alias (the other wrap) is at 4.0 = 6 − 2.
        lat = Lattice([10.0 0 0; 0 10 0; 0 0 6.0]; pbc = (false, false, true))
        cr = Crystal(lat, [0.0 0.0; 0.0 0.0; 0.0 2.0 / 6], [1, 1], ["Fe"])
        da = sort([p.distance for p in build_neighbor_list(cr, 5.0, AllImages()).pairs
                   if (p.i, p.j) == (1, 2)])
        dm = sort([p.distance for p in build_neighbor_list(cr, 5.0, MinimumImage()).pairs
                   if (p.i, p.j) == (1, 2)])
        @test isapprox(da, [2.0, 4.0]; atol = 1e-10)            # minimum + its alias
        @test isapprox(dm, [2.0]; atol = 1e-10)                 # minimum image only
    end

    @testset "AllImages requires a finite cutoff" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, reshape([0.0, 0, 0], 3, 1), [1], ["Fe"])
        @test_throws ArgumentError build_neighbor_list(cr, Inf, AllImages())
    end

    @testset "search box is large enough (hexagonal): s=2 and s=3 agree" begin
        a, c = 2.5, 4.0
        lat = Lattice([a -a/2 0; 0 a*sqrt(3)/2 0; 0 0 c])      # 120° hexagonal
        cr = Crystal(lat, [0.0 0.4; 0.0 0.3; 0.0 0.2], [1, 1], ["Fe"])
        s2 = build_neighbor_list(cr, Inf, MinimumImage(); search = 2)
        s3 = build_neighbor_list(cr, Inf, MinimumImage(); search = 3)
        @test pairset(s2) == pairset(s3)
        # and the minimum image matches the brute-force oracle
        d12 = minimum(p.distance for p in s2.pairs if (p.i, p.j) == (1, 2))
        @test isapprox(d12, _brute_min_dist(cr, 1, 2); rtol = 1e-12)
    end

    @testset "N-body cliques are minimum-image-consistent and use distinct atoms" begin
        L = 3.0
        lat = Lattice(Matrix(L * I(3)))
        cr = Crystal(lat, [0.0 0.5 0.5; 0.0 0.5 0.0; 0.0 0.0 0.5], [1, 1, 1], ["Fe"])
        # full WS, 3-body: each cluster edge at the atom-pair minimum image, and all
        # three atoms distinct (a repeated atom would alias a lower-body term).
        cl = candidate_clusters(cr, build_neighbor_list(cr, Inf, MinimumImage()), 3;
                                selection = MinimumImage())
        A = cr.lattice.vectors
        cart = SLCE.cartesian_positions(cr)
        sitepos(b, R) = SVector{3,Float64}(cart[:, b]...) + A * SVector{3,Float64}(R...)
        @test !isempty(cl[3])
        for m in cl[3]
            @test length(unique(m.atoms)) == 3                  # distinct atoms
            for x = 1:3, y = (x + 1):3
                d = norm(sitepos(m.atoms[x], m.shifts[x]) - sitepos(m.atoms[y], m.shifts[y]))
                @test d ≤ _brute_min_dist(cr, m.atoms[x], m.atoms[y]) * (1 + 1e-8)
            end
        end
    end

    @testset "MinimumImage forbids repeated-atom clusters; AllImages allows them" begin
        # tiny z-period (2.0) so a cutoff of 3.0 reaches an atom's own farther images;
        # AllImages can then build a 3-body clique reusing an atom's images (atoms
        # {1,2,2}), whereas MinimumImage (only 2 distinct atoms) cannot form any.
        lat = Lattice([10.0 0 0; 0 10 0; 0 0 2.0]; pbc = (false, false, true))
        cr = Crystal(lat, [0.0 0.0; 0.0 0.0; 0.0 0.45], [1, 1], ["Fe"])
        mi = candidate_clusters(cr, build_neighbor_list(cr, Inf, MinimumImage()), 3;
                                selection = MinimumImage())
        al = candidate_clusters(cr, build_neighbor_list(cr, 3.0, AllImages()), 3;
                                selection = AllImages())
        @test isempty(mi[3])
        @test !isempty(al[3])                                  # repeated-atom clique
        @test any(m -> length(unique(m.atoms)) < 3, al[3])     # exactly the repeated-atom kind
    end

    @testset "AllImages: every clique edge obeys its own species-pair radius" begin
        # An `N ≥ 3` clique is grown from one anchor's neighbor list and its remaining
        # edges are re-checked by `candidate_clusters` itself. Both steps must decide an
        # edge by the SAME rule: while the list kept only the largest of its per-species
        # radii, the anchor's edges were gated by the species radii and the rest by that
        # maximum, so which sites a cluster happened to be reachable from decided it.
        a = 3.0
        crystal = Crystal(Lattice([a 0 0; 0 a 0; 0 0 a]),
                          [0.0 0.5; 0.0 0.0; 0.0 0.0], [1, 2], ["A", "B"])
        # A–A 3.2 Å admits an atom's own 3.0 Å images, B–B 2.9 Å does not, A–B 1.6 Å
        # admits only the 1.5 Å bond. No interatomic distance lies within 0.1 Å of any
        # radius, so the same-distance band plays no part in the verdicts.
        M = [3.2 1.6; 1.6 2.9]
        nl = SLCE.build_neighbor_list(crystal, M, AllImages())
        got = Set(_canon_cluster(m.atoms, m.shifts)
                  for m in candidate_clusters(crystal, nl, 3; selection = AllImages())[3])

        # Oracle: the declared rule written out directly — a triple of distinct
        # (atom, image) sites is a cluster iff every one of its three edges is within
        # its own species-pair radius — enumerated by brute force over an image box
        # wide enough for the largest radius (ceil(3.2 / 3.0) = 2). Every canonical
        # class has a representative with one site in the home cell, so fixing one
        # site there loses nothing.
        cart = SLCE.cartesian_positions(crystal)
        pos(b, R) = SVector{3,Float64}(cart[:, b]) + a * SVector{3,Float64}(R)
        sites = Tuple{Int,SVector{3,Int}}[]
        for b = 1:2, n1 = -2:2, n2 = -2:2, n3 = -2:2
            push!(sites, (b, SVector{3,Int}(n1, n2, n3)))
        end
        z3 = SVector{3,Int}(0, 0, 0)
        want = Set{Any}()
        for b0 = 1:2, x = 1:length(sites), y = (x + 1):length(sites)
            tri = ((b0, z3), sites[x], sites[y])
            length(unique(tri)) == 3 || continue
            ok = true
            for q = 1:3, r = (q + 1):3
                d = norm(pos(tri[r]...) - pos(tri[q]...))
                d <= M[crystal.species[tri[q][1]], crystal.species[tri[r][1]]] ||
                    (ok = false; break)
            end
            ok && push!(want, _canon_cluster([t[1] for t in tri],
                                             [t[2] for t in tri]))
        end
        @test !isempty(want)
        @test got == want

        # Hand-checked instance of the same rule. Fe and two Te in a cell far larger
        # than every radius, so only the home-cell triangle exists: d(Fe,Te) = 2.5 Å,
        # d(Te,Te) = 4.0 Å. A Te–Te radius of 3.0 Å forbids the triangle outright; a
        # Te–Te radius of 6.0 Å admits it, and the enumeration reaches it once per
        # (anchor, ordering) = 3! = 6 times, the same count every 3-body orbit gets.
        big = Lattice(Matrix(20.0 * I(3)))
        fete = Crystal(big, [10.0 8.0 12.0; 10.0 11.5 11.5; 10.0 10.0 10.0] ./ 20.0,
                       [1, 2, 2], ["Fe", "Te"])
        tri3(M) = candidate_clusters(fete, SLCE.build_neighbor_list(fete, M, AllImages()),
                                     3; selection = AllImages())[3]
        @test isempty(tri3([6.0 6.0; 6.0 3.0]))
        @test length(tri3([6.0 6.0; 6.0 6.0])) == 6

        # Per-order radii may only trim the list: one above the list's own for that
        # species pair reaches the re-checked edges and not the anchor's, so it is
        # refused rather than silently under-generating.
        narrow = SLCE.build_neighbor_list(fete, [3.0 3.0; 3.0 3.0], AllImages())
        wide = [6.0 6.0; 6.0 6.0]
        @test_throws ArgumentError candidate_clusters(fete, narrow, 3;
                                                      selection = AllImages(),
                                                      cutoff = [wide, wide])
        @test_throws ArgumentError candidate_clusters(fete, narrow, 3;
                                                      selection = MinimumImage(),
                                                      cutoff = [wide, wide])
    end

    @testset "no self-pairs under MinimumImage (same spin ⇒ not an independent pair)" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        nl = build_neighbor_list(cr, Inf, MinimumImage())
        @test all(p -> p.i != p.j, nl.pairs)                    # no (i,i,R) self-pairs
        cl = candidate_clusters(cr, nl, 2; selection = MinimumImage())
        @test all(m -> m.atoms[1] != m.atoms[2], cl[2])
    end

    @testset "skewed / non-reduced cell: adaptive search finds the true minimum" begin
        # a grossly non-reduced basis (a₂ ≈ 3.5·a₁) where a fixed ±2 image box would
        # miss the minimum image; the adaptive range must still find it.
        lat = Lattice([1.0 3.5 0.0; 0.0 0.4 0.0; 0.0 0.0 10.0]; pbc = (true, true, false))
        cr = Crystal(lat, [0.0 0.37; 0.0 0.41; 0.0 0.0], [1, 1], ["Fe"])
        nl = build_neighbor_list(cr, Inf, MinimumImage())
        d12 = minimum(p.distance for p in nl.pairs if (p.i, p.j) == (1, 2))
        @test isapprox(d12, _brute_min_dist(cr, 1, 2; box = 8); rtol = 1e-12)
    end

    @testset "fully non-periodic cell (pbc all false): one image per pair" begin
        lat = Lattice(Matrix(5.0 * I(3)); pbc = (false, false, false))
        cr = Crystal(lat, [0.0 0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])    # 1.0 Å apart
        nl = build_neighbor_list(cr, Inf, MinimumImage())
        @test count(p -> (p.i, p.j) == (1, 2), nl.pairs) == 1
        @test isapprox(only(p.distance for p in nl.pairs if (p.i, p.j) == (1, 2)), 1.0)
    end

    @testset "BasisSpec accepts Inf cutoff; rejects ≤ 0 / NaN" begin
        @test BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1]).cutoff[1][1, 1] == Inf
        @test BasisSpec(; nbody = 2, cutoff = 0.0, lmax = [1]).cutoff[1][1, 1] == 0.0  # 0 = excluded pair
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = -1.0, lmax = [1])
        @test_throws ArgumentError BasisSpec(; nbody = 2, cutoff = NaN, lmax = [1])
    end

    @testset "full-WS basis builds; under symmetry it has no collinear columns" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        inter = BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1], soc = false)
        @test n_salcs(SLCEBasis(cr, inter)) ≥ 1                    # default (NoSymmetry) build succeeds
        # with the cubic sign symmetry the 8 WS-corner ties collapse to one orbit;
        # the centered energy design matrix must then have full column rank — i.e. the
        # minimum-image enumeration left no aliased, collinear pair columns behind.
        sg = _cubic_sign_group(cr)
        cls = build_clusters(cr, build_neighbor_list(cr, Inf, MinimumImage()), sg;
                             nbody = 2, selection = MinimumImage())
        salcs = build_salc_basis(cr, sg, cls; lmax_by_species = [1], scalar_only = true)
        b = SLCEBasis(cr, sg, salcs, inter)
        rng = MersenneTwister(11)
        randcol() = (v = randn(rng, 3); v / norm(v))
        configs = [hcat(randcol(), randcol()) for _ = 1:40]
        Xc = let X = _design_energy(b, configs); X .- sum(X; dims = 1) ./ size(X, 1) end
        @test rank(Xc; rtol = 1e-9) == n_salcs(b)
    end

    @testset "input.toml: cutoff = inf and images key" begin
        dir = mktempdir()
        body = """
        [structure]
        lattice        = [[3.0,0,0],[0,3.0,0],[0,0,3.0]]
        positions      = [[0,0,0],[0.5,0.5,0.5]]
        species        = [1,1]
        species_labels = ["Fe"]

        [interaction]
        nbody       = 2
        cutoff = inf
        lmax        = [1]
        soc    = false
        """
        # default images = MinimumImage
        p1 = joinpath(dir, "mi.toml"); write(p1, body)
        inp = read_setup(p1)
        @test inp.spec.cutoff[1][1, 1] == Inf
        @test inp.images isa MinimumImage
        # explicit images = "all_images"
        p2 = joinpath(dir, "all.toml"); write(p2, body * "images = \"all_images\"\n")
        @test read_setup(p2).images isa AllImages
        # unknown selection errors
        p3 = joinpath(dir, "bad.toml"); write(p3, body * "images = \"nope\"\n")
        @test_throws ArgumentError read_setup(p3)
        # SLCEBasis(path) builds (MinimumImage + Inf), keyword override to AllImages errors on Inf
        @test n_salcs(SLCEBasis(p1)) ≥ 1
        @test_throws ArgumentError SLCEBasis(p1; images = AllImages())
    end

    @testset "SLCEDataset refuses a self-image (AllImages) basis" begin
        # One atom in a cubic cell: every admitted pair joins the atom to its own
        # periodic image, so BOTH ends carry e₁ and the l = (1,1), Lf = 0 function is
        # e₁·e₁ ≡ 1 — a constant, derived by hand and independent of whatever the
        # builder emits. A constant column says nothing about the configuration, so the
        # design is rank deficient by construction and a fit would return one arbitrary
        # representative of a non-unique solution. `asr = false` never fixed that (it
        # drops the constraint, not the degeneracy), so the door refuses instead;
        # building the same basis as a tiling template stays legal.
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 3.2, lmax = [1], soc = false);
                      images = AllImages())
        @test n_salcs(b) ≥ 1
        @test all(s -> any(m -> !allunique(m.atoms), s.members), SLCE.salcs(b))

        Random.seed!(20260824)
        cfgs = [(d = randn(3, 1); d ./ norm(d)) for _ = 1:8]
        X = _design_energy(b, cfgs)                     # building/evaluating is allowed
        for j in axes(X, 2)                             # every column is that constant
            col = view(X, :, j)
            @test maximum(col) - minimum(col) ≤ 1e-12 * max(1.0, abs(col[1]))
        end

        @test_throws SLCE.UnclassifiableBasis SLCEDataset(b, cfgs, zeros(8))
        msg = try
            SLCEDataset(b, cfgs, zeros(8))
            ""
        catch e
            sprint(showerror, e)
        end
        @test occursin("self-image", msg)               # names the cause
        @test occursin("MinimumImage", msg)             # and the way out
        @test occursin("asr = false does not help", msg)  # the withdrawn escape hatch

        # The joint door is a different constructor and must refuse too — before this
        # gate it was the loud-ish one (ASR unavailable → warn → `asr = false` advice).
        latj = Lattice(Matrix(2.0 * I(3)))
        crj = Crystal(latj, zeros(3, 1), [1], ["Fe"])
        bj = SLCEBasis(crj, BasisSpec(crj; lmax = 1, pmax = 1, sectors = [
                           Sector(spin = [1, 1], disp = (degree = 2,), sites = 2,
                                  cutoff = 2.1)]);
                       images = AllImages())
        dj = [TrainingDatum(; energy = 0.0, directions = reshape([0.0, 0.0, 1.0], 3, 1),
                            magmoms = [2.0], displacements = zeros(3, 1))]
        @test_throws SLCE.UnclassifiableBasis SLCEDataset(bj, dj)

        # Control: distinct-atom neighbors (bcc, 8 NN corner↔centre) are untouched.
        cr2 = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        b2 = SLCEBasis(cr2, BasisSpec(; nbody = 2, cutoff = 2.7, lmax = [1],
                                      soc = false))
        cfg2 = [(d = randn(3, 2); d ./ sqrt.(sum(abs2, d; dims = 1))) for _ = 1:8]
        @test SLCEDataset(b2, cfg2, zeros(8)) isa SLCEDataset
    end

    @testset "persistence round-trips a cutoff = Inf basis" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = Inf, lmax = [1], soc = false))
        path = joinpath(mktempdir(), "wsbasis.toml")
        SLCE.save(path, b)
        b2 = SLCE.load(SLCEBasis, path)
        @test b2.spec.cutoff[1][1, 1] == Inf
        @test n_salcs(b2) == n_salcs(b)
    end
end
