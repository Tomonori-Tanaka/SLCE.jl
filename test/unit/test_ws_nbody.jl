# Counting N-body clusters (N ≥ 3) on the Wigner–Seitz boundary.
#
# The 2-body WS-boundary ties (faces 2-fold, edges 4-fold, corners 8-fold) are covered
# in test_imageselection.jl. At N ≥ 3 the new difficulty is the *third edge*: a triangle
# {i, j, k} is admissible only if ALL three pairwise edges sit at their atom-pair
# minimum-image distance simultaneously — having each pair individually within the
# minimum-image set is NOT enough (the images that make i–j and i–k minimal may force
# j–k onto a longer, non-minimum image). These tests pin the counting against an
# independent brute force and guard the orbit reduction at the boundary.

using Test
using SLCE
using SLCE: candidate_clusters, build_clusters, n_atoms, cartesian_positions,
                      _assemble_spacegroup, _site_image, ClusterMember
using StaticArrays
using LinearAlgebra

# Independent brute-force compact-cluster enumeration: every anchored (shifts[1] = 0),
# ordered, distinct-atom N-tuple of sites whose every pairwise edge sits at the
# atom-pair minimum-image distance (within rtol) and within the cutoff. Written as
# plainly as possible — exactly the definition the production code must realize.
function _wsnb_brute(cr, cutoff, N; rtol = 1e-8, box = 3)
    A = SMatrix{3,3,Float64}(cr.lattice.vectors)
    nat = n_atoms(cr)
    cart = cartesian_positions(cr)
    pbc = cr.lattice.pbc
    rng = ntuple(d -> pbc[d] ? box : 0, 3)
    shifts = SVector{3,Int}[]
    for n1 = -rng[1]:rng[1], n2 = -rng[2]:rng[2], n3 = -rng[3]:rng[3]
        push!(shifts, SVector{3,Int}(n1, n2, n3))
    end
    z = SVector{3,Int}(0, 0, 0)
    pos(b, R) = SVector{3,Float64}(cart[1, b], cart[2, b], cart[3, b]) +
                A * SVector{3,Float64}(R[1], R[2], R[3])
    dmin = fill(Inf, nat, nat)
    for i = 1:nat, j = 1:nat
        i == j && continue
        for R in shifts
            d = norm(pos(j, R) - pos(i, z))
            d < dmin[i, j] && (dmin[i, j] = d)
        end
    end
    edge_ok(ax, Rx, ay, Ry) = begin
        d = norm(pos(ay, Ry) - pos(ax, Rx))
        isfinite(dmin[ax, ay]) && d <= dmin[ax, ay] * (1 + rtol) && d <= cutoff * (1 + rtol)
    end
    out = Set{Tuple}()
    member(sites) = (Tuple(s[1] for s in sites),
                     Tuple((s[2][1], s[2][2], s[2][3]) for s in sites))
    function extend!(chosen)
        if length(chosen) == N
            push!(out, member(chosen))
            return
        end
        for b = 1:nat, R in shifts
            any(s -> s[1] == b, chosen) && continue
            all(s -> edge_ok(s[1], s[2], b, R), chosen) && extend!(vcat(chosen, [(b, R)]))
        end
    end
    for a = 1:nat
        extend!([(a, z)])
    end
    return out
end

# Independent brute-force POINTED STAR enumeration: every anchored (shifts[1] = 0),
# ordered N-tuple whose first site is a marked-species atom and whose N−1 spokes from
# it each sit at the atom-pair minimum-image distance (within rtol) and within
# `cutoff_star` for that species pair. The environment-environment edges are FREE —
# a pointed star is pinned by its spokes, and it has a distinguished centre, so the
# compact-cluster rule the energy side needs does not apply.
#
# Two rules are deliberately NOT inherited from `_wsnb_brute`: the "distinct atoms"
# filter (a pointed star keeps two different minimum IMAGES of one neighbour and only
# drops an exact `(atom, shift)` repeat), and the anchor loop over every atom (only a
# marked species can carry the mark). Written as the definition, so the N! multiplicity
# falls out of "ordered tuple" rather than being asserted.
# `box = 2` is a cost as well as a coverage choice: the walk is
# `n_marked · (1 + z + … + z^(N-2)) · nat · (2box+1)^3` spoke tests, so a denser cell,
# a wider cutoff or an `N = 5` row would grow it sharply. The fixtures below are 3–4
# atoms with `z <= 6`, which keeps it at ~1e4–1e5 operations per case.
function _ptstar_brute(cr, spec, N; rtol = SLCE._SAME_DIST_RTOL, box = 2)
    A = SMatrix{3,3,Float64}(cr.lattice.vectors)
    nat = n_atoms(cr)
    cart = cartesian_positions(cr)
    pbc = cr.lattice.pbc
    rng = ntuple(d -> pbc[d] ? box : 0, 3)
    shifts = SVector{3,Int}[]
    for n1 = -rng[1]:rng[1], n2 = -rng[2]:rng[2], n3 = -rng[3]:rng[3]
        push!(shifts, SVector{3,Int}(n1, n2, n3))
    end
    z = SVector{3,Int}(0, 0, 0)
    pos(b, R) = SVector{3,Float64}(cart[1, b], cart[2, b], cart[3, b]) +
                A * SVector{3,Float64}(R[1], R[2], R[3])
    dmin = fill(Inf, nat, nat)
    for i = 1:nat, j = 1:nat, R in shifts
        (i == j && R == z) && continue
        d = norm(pos(j, R) - pos(i, z))
        d < dmin[i, j] && (dmin[i, j] = d)
    end
    sp = cr.species
    # `cutoff_star` is stored one matrix per STAR order, body `N` at index `N - 2`.
    # Spelled out here rather than through the production accessor: this oracle reads
    # the layout from the definition, not from the code it is checking.
    cut = spec.cutoff_star[N - 2]
    spoke_ok(a, b, R) = begin
        d = norm(pos(b, R) - pos(a, z))
        d <= dmin[a, b] * (1 + rtol) && d <= cut[sp[a], sp[b]] * (1 + rtol)
    end
    out = Set{Tuple}()
    member(sites) = (Tuple(s[1] for s in sites),
                     Tuple((s[2][1], s[2][2], s[2][3]) for s in sites))
    # every ordered N-tuple: the mark may sit at any position, and the whole tuple is
    # re-anchored so that its FIRST site is at the origin (the production convention)
    perms = _ptstar_perms(N)          # hoisted: rebuilt per emission otherwise
    function emit!(a, env)
        sites = vcat([(a, z)], env)
        for p in perms
            s1 = sites[p[1]][2]
            push!(out, member([(sites[q][1], sites[q][2] - s1) for q in p]))
        end
    end
    function extend!(a, env)
        if length(env) == N - 1
            emit!(a, env)
            return
        end
        for b = 1:nat, R in shifts
            # An atom is not a neighbour of ITSELF at any image: `_build_nl_minimage`
            # drops `i == j` outright, so `nbrs[i]` never contains atom `i` and a mark
            # can never be its own environment. Dropping only `(a, z)` here would let
            # the oracle admit `(a, R != 0)`, which production cannot produce — and
            # would matter, because `_design_moment` substitutes the marked column of
            # `e`, so such an environment factor would silently read the evaluation
            # axis instead of a spin.
            b == a && continue
            # only an EXACT (atom, shift) repeat is excluded among the environment
            # sites — two different minimum images of one neighbour are distinct
            any(s -> s == (b, R), env) && continue
            spoke_ok(a, b, R) && extend!(a, vcat(env, [(b, R)]))
        end
    end
    for a = 1:nat
        spec.marked[sp[a]] || continue
        extend!(a, Tuple{Int,SVector{3,Int}}[])
    end
    return out
end

# all permutations of 1:n, written out rather than borrowed from the code under test
function _ptstar_perms(n)
    n == 1 && return [[1]]
    out = Vector{Int}[]
    for p in _ptstar_perms(n - 1), i = 1:n
        push!(out, vcat(p[1:(i - 1)], [n], p[i:end]))
    end
    return out
end

# production candidate members → the same comparable representation
_wsnb_prodset(ms) =
    Set((Tuple(m.atoms), Tuple((s[1], s[2], s[3]) for s in m.shifts)) for m in ms)

# every edge of every production member is at the atom-pair minimum image
function _wsnb_edges_minimage(cr, ms; rtol = 1e-8, box = 3)
    A = SMatrix{3,3,Float64}(cr.lattice.vectors)
    cart = cartesian_positions(cr)
    pos(b, R) = SVector{3,Float64}(cart[1, b], cart[2, b], cart[3, b]) +
                A * SVector{3,Float64}(R[1], R[2], R[3])
    function bmin(i, j)
        best = Inf
        for n1 = -box:box, n2 = -box:box, n3 = -box:box
            (i == j && n1 == n2 == n3 == 0) && continue
            d = norm(pos(j, SVector{3,Int}(n1, n2, n3)) - pos(i, SVector{3,Int}(0, 0, 0)))
            d < best && (best = d)
        end
        return best
    end
    for m in ms
        N = length(m.atoms)
        for x = 1:N, y = (x + 1):N
            d = norm(pos(m.atoms[x], m.shifts[x]) - pos(m.atoms[y], m.shifts[y]))
            d <= bmin(m.atoms[x], m.atoms[y]) * (1 + rtol) || return false
        end
    end
    return true
end

# diag(±1,±1,±1) cubic sign group (8 ops): maps the (±L/2,±L/2,±L/2) WS corners into
# one orbit — enough symmetry to test the boundary orbit reduction without Spglib.
function _wsnb_cubic_signs(cr)
    rots = SMatrix{3,3,Float64}[]
    for sx in (1.0, -1.0), sy in (1.0, -1.0), sz in (1.0, -1.0)
        push!(rots, SMatrix{3,3,Float64}(Diagonal([sx, sy, sz])))
    end
    trans = [SVector{3,Float64}(0, 0, 0) for _ in rots]
    return _assemble_spacegroup(cr, rots, trans, "manual-cubic-signs", 0; tol = 1e-8)
end

# translation-normalized site-set of member `m` after op `g`, re-anchored at site `s`
function _wsnb_imgset(cr, sg, g, m, s)
    N = length(m.atoms)
    imgs = [_site_image(cr, sg, g, m.atoms[k], m.shifts[k]) for k = 1:N]
    R0 = imgs[s][2]
    return sort([(imgs[k][1], imgs[k][2][1] - R0[1], imgs[k][2][2] - R0[2],
                  imgs[k][2][3] - R0[3]) for k = 1:N])
end
_wsnb_siteset(m) = sort([(m.atoms[k], m.shifts[k][1], m.shifts[k][2], m.shifts[k][3])
                         for k = 1:length(m.atoms)])

@testset "Wigner–Seitz N-body cluster counting" begin
    L = 3.0
    cubic = Lattice(Matrix(L * I(3)))
    # face-positioned 3 atoms: A–B and A–C are 2-fold L/2 ties, B–C a 4-fold edge tie
    faces = Crystal(cubic, [0.0 0.5 0.0; 0.0 0.0 0.5; 0.0 0.0 0.0], [1, 1, 1], ["Fe"])
    # fcc conventional 4 atoms — 4-body clusters span the WS boundary
    fcc = Crystal(cubic, [0.0 0.5 0.5 0.0; 0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.5],
                  [1, 1, 1, 1], ["Fe"])
    # generic interior positions (a control: few/no ties)
    generic = Crystal(cubic, [0.1 0.55 0.2; 0.15 0.0 0.62; 0.0 0.3 0.4], [1, 1, 1], ["Fe"])
    # skewed hexagonal cell with face/edge atoms
    hexlat = Lattice([2.5 -1.25 0; 0 2.5*sqrt(3)/2 0; 0 0 4.0])
    hex = Crystal(hexlat, [0.0 0.5 0.5; 0.0 0.5 0.0; 0.0 0.0 0.5], [1, 1, 1], ["Fe"])

    @testset "candidate set == independent brute force (all N, incl. full WS)" begin
        cases = [("faces", faces), ("fcc", fcc), ("generic", generic), ("hex", hex)]
        for (nm, cr) in cases, cutoff in (Inf, 2.0, 1.6)
            cand = candidate_clusters(cr, build_neighbor_list(cr, cutoff, MinimumImage()),
                                      4; selection = MinimumImage())
            for N = 2:min(4, n_atoms(cr))
                prod = get(cand, N, ClusterMember[])
                @test _wsnb_prodset(prod) == _wsnb_brute(cr, cutoff, N)   # exact counting
                @test _wsnb_edges_minimage(cr, prod)                     # no spurious edges
            end
        end
    end

    @testset "pointed star candidates == independent brute force (N = 3, 4)" begin
        # The same discipline as the compact-cluster gate above, for the OTHER
        # enumeration in the package. The cells carry Wigner-Seitz ties on purpose:
        # a pointed star keeps two different minimum images of one neighbour, and a
        # brute force that copied the energy side's distinct-atom rule would go red
        # here — which is the point of writing it from the definition instead.
        cases = [("faces", faces), ("fcc", fcc), ("generic", generic), ("hex", hex)]
        # `cut = 3.2` exceeds the 3.0 cubic lattice constant, so an atom's own
        # periodic image is inside the star radius. Production still cannot use it (a
        # minimum-image neighbour list has no self-pairs); an oracle that dropped only
        # the mark's zero-shift site would admit it and go red here. Below 3.0 the
        # branch is unreachable and the gate says nothing about that rule.
        for (nm, cr) in cases, cut in (3.2, 2.0, 1.6, 1.3)
            spec = MomentSpec(; lmax_env = [1], sampled = [true], lmax_mark = 1,
                              nbody = 4, cutoff_pair = cut, cutoff_star = cut,
                              lsum = 4)
            nl = build_neighbor_list(cr, cut, MinimumImage())
            for N = 3:4
                prod = SLCE._pointed_star_candidates(cr, nl, spec, N)
                @test _wsnb_prodset(prod) == _ptstar_brute(cr, spec, N)
                # No member is emitted twice. The set comparison above cannot see a
                # duplicate, and a duplicated member is a silently double-counted
                # cluster in the design matrix.
                @test length(prod) == length(_wsnb_prodset(prod))
                # The N! ordering convention: each translation class contributes
                # exactly N! DISTINCT members. Counting emissions would be true by
                # construction (the loop pushes one per permutation); counting
                # distinct members is not — it also asserts that the N! orderings of
                # a class never collide, i.e. that the class has N distinct sites.
                classes = Dict{Any,Set{Any}}()
                for m in prod
                    sites = sort([(m.atoms[i], Tuple(m.shifts[i]))
                                  for i in eachindex(m.atoms)])
                    R0 = sites[1][2]           # anchor on the SORTED first site, so
                    k = [(a, s .- R0) for (a, s) in sites]   # the key is ordering-free
                    push!(get!(() -> Set{Any}(), classes, k),
                          (Tuple(m.atoms), Tuple(Tuple(v) for v in m.shifts)))
                end
                @test isempty(classes) || all(==(factorial(N)), length.(values(classes)))
            end
        end
    end

    @testset "exact boundary multiplicities (oracle-locked regression)" begin
        cand_f = candidate_clusters(faces, build_neighbor_list(faces, Inf, MinimumImage()),
                                    3; selection = MinimumImage())
        cand_c = candidate_clusters(fcc, build_neighbor_list(fcc, Inf, MinimumImage()),
                                    4; selection = MinimumImage())
        @test length(cand_f[3]) == 24                       # 3-body on the WS boundary
        @test length(cand_c[4]) == 192                      # 4-body on the WS boundary
        @test length(cand_c[3]) == 192
    end

    @testset "third-edge consistency: pairwise-minimal ≠ a compact triangle" begin
        # An equal-spaced 3-atom ring along z (period 3, atoms at 0, 1, 2): every pair's
        # minimum-image distance is 1 (≤ any cutoff), so a naive "all pairwise distances
        # in range" rule would admit a triangle — but no choice of images puts all three
        # edges at distance 1 at once (you cannot embed an equilateral triangle on a 1-D
        # ring), so the correct count of 3-body clusters is ZERO.
        ring = Crystal(Lattice([10.0 0 0; 0 10 0; 0 0 3.0]; pbc = (false, false, true)),
                       [0.0 0.0 0.0; 0.0 0.0 0.0; 0.0 1/3 2/3], [1, 1, 1], ["Fe"])
        nl = build_neighbor_list(ring, Inf, MinimumImage())
        # all three pairwise minimum images are present and equal (≤ cutoff)
        d(i, j) = minimum(p.distance for p in nl.pairs if (p.i, p.j) == (i, j))
        @test d(1, 2) ≈ 1.0 && d(1, 3) ≈ 1.0 && d(2, 3) ≈ 1.0
        cand = candidate_clusters(ring, nl, 3; selection = MinimumImage())
        @test !isempty(cand[2])                              # the pairs exist …
        @test isempty(cand[3])                               # … but no compact triangle
    end

    @testset "orbit reduction neither over- nor under-merges boundary clusters" begin
        for cr in (faces, fcc)
            sg = _wsnb_cubic_signs(cr)
            nl = build_neighbor_list(cr, Inf, MinimumImage())
            cs = build_clusters(cr, nl, sg; nbody = n_atoms(cr), selection = MinimumImage())
            for (body, orbits) in cs.by_body
                body >= 2 || continue
                ncand = length(get(candidate_clusters(cr, nl, body;
                                                      selection = MinimumImage()), body,
                                   ClusterMember[]))
                @test sum(o -> o.multiplicity, orbits; init = 0) == ncand   # partition

                # no over-merge: every member is genuinely equivalent to its representative
                for o in orbits
                    reps = Set(_wsnb_imgset(cr, sg, g, o.representative, s)
                               for g = 1:n_ops(sg), s = 1:body)
                    @test all(m -> _wsnb_siteset(m) in reps, o.members)
                end
                # no under-merge: each orbit is closed under the group (an op sends every
                # member to another member of the SAME orbit)
                for o in orbits
                    here = Set(_wsnb_siteset(m) for m in o.members)
                    @test all(m -> all(g -> _wsnb_imgset(cr, sg, g, m, 1) in here,
                                       1:n_ops(sg)), o.members)
                end
            end
        end
    end
end
