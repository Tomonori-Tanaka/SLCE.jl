# The pointed (site-marked) moment basis (src/basis/momentbasis.jl): MomentSpec
# validation, the FeGe B20 oracles ported from the design-record prototype (star
# closed form against an INDEPENDENT geometric enumeration, the 2√3 shell-sum
# normalization, G_i covariance including the evaluation axes, time reversal,
# marked-column substitution locality), and the pointed resolvability gate
# (signature rank ≡ independent random-design rank; null combinations annihilate
# the actual design; the repeated-image refusal).

using Test
using SLCE
using SLCE: _assemble_spacegroup, n_ops, cartesian_positions, _design_moment,
            salcs, n_salcs, has_disp, _orbit_salcs_decors, _build_wig_cache,
            SiteDecor, UnclassifiableBasis, SpaceGroup
using LinearAlgebra
using StaticArrays
using Random

# ── FeGe B20 primitive with the handwritten P2₁3 group (no Spglib in the core) ──
const _MB_WS = [[1 0 0; 0 1 0; 0 0 1], [-1 0 0; 0 -1 0; 0 0 1],
                [-1 0 0; 0 1 0; 0 0 -1], [1 0 0; 0 -1 0; 0 0 -1],
                [0 0 1; 1 0 0; 0 1 0], [0 0 1; -1 0 0; 0 -1 0],
                [0 0 -1; -1 0 0; 0 1 0], [0 0 -1; 1 0 0; 0 -1 0],
                [0 1 0; 0 0 1; 1 0 0], [0 -1 0; 0 0 1; -1 0 0],
                [0 1 0; 0 0 -1; -1 0 0], [0 -1 0; 0 0 -1; 1 0 0]]
const _MB_TS = [[0, 0, 0], [0.5, 0, 0.5], [0, 0.5, 0.5], [0.5, 0.5, 0],
                [0, 0, 0], [0.5, 0.5, 0], [0.5, 0, 0.5], [0, 0.5, 0.5],
                [0, 0, 0], [0, 0.5, 0.5], [0.5, 0.5, 0], [0.5, 0, 0.5]]

_mb_wrap01(x) = (y = mod(x, 1.0); y >= 1.0 - 1e-12 ? 0.0 : y)

function _mb_orbit4a(x::Float64)
    out = Vector{Vector{Float64}}()
    for (W, t) in zip(_MB_WS, _MB_TS)
        p = _mb_wrap01.(W * [x, x, x] .+ t)
        any(q -> maximum(abs.(q .- p)) < 1e-9, out) || push!(out, p)
    end
    sort!(out)
    return hcat(out...)
end

function _mb_fege()
    pos = hcat(_mb_orbit4a(0.1352), _mb_orbit4a(0.8414))
    xt = Crystal(Lattice(Matrix(4.7 * I(3))), pos, [1, 1, 1, 1, 2, 2, 2, 2],
                 ["Fe", "Ge"])
    sg = _assemble_spacegroup(xt,
        [SMatrix{3,3,Float64}(Float64.(W)) for W in _MB_WS],
        [SVector{3,Float64}(Float64.(t)) for t in _MB_TS], "P2_13", 198; tol = 1e-5)
    return xt, sg
end

struct _MBFixedSG <: SLCE.AbstractSymmetryBackend
    sg::SpaceGroup
end
SLCE.analyze_symmetry(b::_MBFixedSG, ::Crystal; tol::Real = 1e-5) = b.sg

_mb_unit(rng, nat) = (m = randn(rng, 3, nat);
                      for a = 1:nat; m[:, a] ./= norm(m[:, a]); end; m)

@testset "pointed moment basis" begin
    xt, sg = _mb_fege()
    bk = _MBFixedSG(sg)
    nat = 8
    cart = Matrix(cartesian_positions(xt))
    A = Matrix(xt.lattice.vectors)

    @testset "MomentSpec validation" begin
        ok = MomentSpec(; lmax_env = [2, 0], sampled = [true, false],
                        cutoff_pair = 3.0)
        @test ok.nbody == 3 && ok.cutoff_star == ok.cutoff_pair
        # the M3-1 assert: environment spin factors only on sampled species
        @test_throws ArgumentError MomentSpec(; lmax_env = [2, 1],
                                              sampled = [true, false],
                                              cutoff_pair = 3.0)
        @test_throws ArgumentError MomentSpec(; lmax_env = [2], sampled = [true],
                                              cutoff_pair = 3.0, lmax_mark = -1)
        @test_throws ArgumentError MomentSpec(; lmax_env = [2], sampled = [true],
                                              cutoff_pair = 3.0, nbody = 4)
        @test_throws ArgumentError MomentSpec(; lmax_env = [2], sampled = [true],
                                              cutoff_pair = 3.0,
                                              marked = [false])
        @test_throws ArgumentError MomentSpec(; lmax_env = [2, 2],
                                              sampled = [true, true],
                                              cutoff_pair = [3.0 2.0; 2.5 3.0])
        # species-count mismatch dies at the basis door
        @test_throws ArgumentError MomentBasis(xt,
            MomentSpec(; lmax_env = [2], sampled = [true], cutoff_pair = 3.0);
            backend = bk)
        # a cutoff below every shell leaves exactly the 1-body content (the
        # per-orbit intercepts μ₀ and single-site ê invariants never need a bond)
        tiny = MomentBasis(xt, MomentSpec(; lmax_env = [1, 1],
                                          sampled = [true, true],
                                          cutoff_pair = 0.5); backend = bk)
        @test all(k -> k.body == 1, tiny.salc_basis.keys)
        # caps and an admit predicate are mutually exclusive at the engine door
        wc = _build_wig_cache(sg, 1)
        nl = SLCE.build_neighbor_list(xt, 3.0)
        cs = SLCE.build_clusters(xt, nl, sg; nbody = 2)
        O = cs.by_body[2][1]
        lab = [sort([SiteDecor(spin = 1), SiteDecor(spin = 1, disp = (1, 0))])]
        @test_throws ArgumentError _orbit_salcs_decors(xt, sg, 2, 1, O, lab, false,
                                                       wc; lmax_by_species = [1, 1],
                                                       pmax_by_species = [2, 2],
                                                       admit = _ -> true)
    end

    spec = MomentSpec(; lmax_env = [2, 2], sampled = [true, true], lmax_mark = 2,
                      nbody = 3, cutoff_pair = 3.3, cutoff_star = 3.3)
    mb = MomentBasis(xt, spec; backend = bk)
    ks = mb.salc_basis.keys
    mark_l(k) = (i = findfirst(has_disp, k.decors); k.decors[i].spin_l)
    env_ls(k) = sort([d.spin_l for d in k.decors if !has_disp(d)])

    rng = MersenneTwister(0x20260819)
    e = _mb_unit(rng, nat)

    # independent geometric references (ported from the design-record prototype 01;
    # nothing here touches the SALC machinery)
    imgs = [A * Float64.([i, j, k]) for i = -1:1, j = -1:1, k = -1:1]
    function nn_neighbors(a; d0 = 2.881, tol = 0.05)
        nbrs = Tuple{Int,Vector{Float64}}[]
        for j = 1:4, R in imgs
            p = cart[:, j] + R
            abs(norm(p - cart[:, a]) - d0) < tol && push!(nbrs, (j, p))
        end
        return nbrs
    end
    star_ref(a) = begin
        nbrs = nn_neighbors(a)
        acc = 0.0
        for i = 1:length(nbrs), j = (i + 1):length(nbrs)
            (jj, pj) = nbrs[i]
            (kk, pk) = nbrs[j]
            abs(norm(pj - pk) - 2.881) < 0.05 || continue
            acc += dot(e[:, jj], e[:, kk])
        end
        acc
    end
    p1_ref(a) = sum(dot(e[:, a], e[:, j]) for (j, _) in nn_neighbors(a))

    @testset "FeGe oracles: closed forms vs independent geometry" begin
        @test mb.marked_atoms == collect(1:8)
        X1 = _design_moment(mb, [e], [copy(e)])          # mode-4 identity axes
        # star (0,1,1) on the closed nn Fe₃ triangle: SALC = 6 · Σ e_j·e_k over the
        # triangles at atom a — the prototype's headline geometric oracle
        jstars = [j for j in 1:length(ks) if ks[j].body == 3 && mark_l(ks[j]) == 0 &&
                  env_ls(ks[j]) == [1, 1] && all(==(1), mb.records[j].species) &&
                  all(abs(x - 2.881) < 0.05 for x in mb.records[j].edges)]
        @test length(jstars) == 1
        for a = 1:4
            @test X1[a, jstars[1]] ≈ 6.0 * star_ref(a) rtol = 1e-10
        end
        # nn Fe–Fe pointed (1,1): exactly 2 SALC blocks (the C₃ split of the 6-shell
        # into 3+3), whose SUM is the 2√3-normalized shell sum
        jnn = [j for j in 1:length(ks) if ks[j].body == 2 && mark_l(ks[j]) == 1 &&
               env_ls(ks[j]) == [1] && all(==(1), mb.records[j].species) &&
               abs(mb.records[j].edges[1] - 2.881) < 0.05]
        @test length(jnn) == 2
        for a = 1:4
            @test sum(X1[a, j] for j in jnn) ≈ 2 * sqrt(3) * p1_ref(a) rtol = 1e-10
        end
        # the marked-multiplicity structure of the nn pointed pair: 12 members
        # (6 bonds × 2 orderings), mark histogram 3 per Fe atom of the split
        s = salcs(mb)[jnn[1]]
        @test length(s.members) == 12
        marks = Int[]
        for m in s.members, t in m.terms, sl in t.slots
            sl.factor.channel == SLCE.DISP && push!(marks, m.atoms[sl.site])
        end
        @test sort(unique(marks)) == [1, 2, 3, 4]
        @test all(count(==(a), marks) == length(marks) ÷ 4 for a = 1:4)
    end

    @testset "covariance / TR / substitution locality" begin
        axr = _mb_unit(rng, nat)
        Xr = _design_moment(mb, [e], [axr])
        # G_i covariance including the axes: Φ(g·a; g∘e, g∘ê) = Φ(a; e, ê)
        dev = 0.0
        for g = 1:n_ops(sg)
            R = sg.ops[g].rotation_cart
            eg = similar(e)
            axg = similar(axr)
            for a = 1:nat
                eg[:, sg.map_sym[a, g]] = R * e[:, a]
                axg[:, sg.map_sym[a, g]] = R * axr[:, a]
            end
            Xg = _design_moment(mb, [eg], [axg])
            for a = 1:nat, j = 1:size(Xr, 2)
                dev = max(dev, abs(Xg[sg.map_sym[a, g], j] - Xr[a, j]))
            end
        end
        @test dev < 1e-12
        # time reversal is bitwise (every label has even total spin rank)
        @test _design_moment(mb, [-e], [-axr]) == Xr
        # marked-column substitution locality: changing the axis of atom b changes
        # rows of b and nothing else (the exactness of the substitution)
        ax2 = copy(axr)
        ax2[:, 3] = normalize(randn(rng, 3))
        X2 = _design_moment(mb, [e], [ax2])
        @test findall([any(X2[a, :] .!= Xr[a, :]) for a = 1:nat]) == [3]
        # mode-4 identity: substituting e's own columns is a no-op
        @test _design_moment(mb, [e], [copy(e)]) ==
              _design_moment(mb, [e], [Matrix(e)])
        # deterministic rebuild
        mb2 = MomentBasis(xt, spec; backend = bk)
        @test mb2.salc_basis.keys == mb.salc_basis.keys
    end

    @testset "pointed resolvability gate" begin
        # pairs-only basis: no repeated-image members, so the gate classifies —
        # and its symbolic rank must equal the rank of an INDEPENDENT random
        # design (two implementations of one question)
        psp = MomentSpec(; lmax_env = [2, 2], sampled = [true, true],
                         lmax_mark = 2, nbody = 2, cutoff_pair = 3.3)
        pmb = MomentBasis(xt, psp; backend = bk)
        res = moment_resolvability(pmb)
        cfgs = [_mb_unit(rng, nat) for _ = 1:60]
        axes = [_mb_unit(rng, nat) for _ = 1:60]
        X = _design_moment(pmb, cfgs, axes)
        sv = svd(X).S
        @test count(>(1e-9 * sv[1]), sv) == res.rank
        # every symbolic null combination annihilates the actual design
        for comb in res.null_combinations
            v = zeros(n_salcs(pmb))
            for (j, w) in comb
                v[j] = w
            end
            @test norm(X * v) < 1e-10 * norm(X) * norm(v)
        end
        # the null report NAMES columns (nonempty combinations) when rank-deficient
        res.rank < length(res.kept) &&
            @test all(!isempty, res.null_combinations)
        # census: every pointed pair orbit on this cell carries ≥ 2 mark classes
        # (both ends markable) — the face-(b) hazard preregistration
        @test all(c -> c.n_mark_atoms >= 2, res.census)
        # the star basis on the PRIMITIVE cell folds two images of one neighbor
        # into a single environment sphere — the gate must refuse loudly, exactly
        # like the energy side's UnclassifiableBasis, never overcount
        @test_throws UnclassifiableBasis moment_resolvability(mb)
    end
end
