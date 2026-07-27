using Test
using SLCE
using SLCE: _assemble_spacegroup, evaluate_salc
using StaticArrays
using LinearAlgebra
using Random

# An equilateral triangle of three equivalent atoms with a manual C3v site group
# (no Spglib): identity + two 3-fold rotations + three mirrors, permuting the three
# sites as S₃. This exercises the N ≥ 3 machinery — coupling-path mixing and, for
# unequal-l multisets like (1,1,2), l-ordering mixing (multi-term SALCs).
#
# The cell is HEXAGONAL. It used to be a cubic box, but a 3-fold rotation is not an
# operation of a cubic lattice — its fractional matrix is not integral — so that
# fixture declared a group that maps the lattice off itself. It happened to be
# harmless (every cluster here sits inside one cell, so no lattice shift is ever
# rotated), but `_validate_ops` refuses it, correctly. In the hexagonal basis the
# same geometric C3v is exactly integral, and the triangle, its side r√3, and the
# neighbour shells are all unchanged.
function _triangle_c3v(; a = 6.0, c = 6.0, r = 1.2)
    lat = Lattice(SMatrix{3,3,Float64}([a -a/2 0; 0 a*√3/2 0; 0 0 c]))
    ctr = SVector{3,Float64}(0.5, 0.5, 0.5)             # the C3 axis passes here
    ang = deg2rad.([90.0, 210.0, 330.0])
    offs = [SVector{3,Float64}(r * cos(t), r * sin(t), 0.0) for t in ang]
    frac = hcat([ctr + lat.reciprocal * o for o in offs]...)
    crystal = Crystal(lat, frac, [1, 1, 1], ["Fe"])
    C3 = SMatrix{3,3,Float64}([0 -1 0; 1 -1 0; 0 0 1])  # +120° about z, exact here
    M1 = SMatrix{3,3,Float64}([-1 1 0; 0 1 0; 0 0 1])   # mirror x → −x, fixes atom 1
    rots = [SMatrix{3,3,Float64}(I), C3, C3 * C3, M1, C3 * M1, C3 * C3 * M1]
    trans = [(SMatrix{3,3,Float64}(I) - R) * ctr for R in rots]
    sg = _assemble_spacegroup(crystal, rots, trans, "C3v", 0; tol = 1e-6)
    return crystal, sg
end

# (g·e)_a = R_g · e_{preimage(a)}.
function _act3(sg, g, e)
    nat = size(e, 2)
    R = Matrix(sg.ops[g].rotation_cart)
    out = similar(e)
    for a = 1:nat
        b = findfirst(==(a), @view sg.map_sym[:, g])
        out[:, a] = R * e[:, b]
    end
    return out
end

function _rcfg3(rng, nat)
    M = Matrix{Float64}(undef, 3, nat)
    for a = 1:nat
        v = randn(rng, 3)
        M[:, a] = v / norm(v)
    end
    return M
end

# τ_a = −e_a × ∇_tan E (physical / Landau–Lifshitz torque) by renormalized on-sphere
# central differences.
function _torque_fd3(model, config; δ = 1e-6)
    nat = size(config, 2)
    T = Matrix{Float64}(undef, 3, nat)
    for a = 1:nat
        e = config[:, a]
        g = zeros(3)
        for d = 1:3
            ep = copy(config); em = copy(config)
            step = zeros(3); step[d] = δ
            ep[:, a] = (e .+ step) ./ norm(e .+ step)
            em[:, a] = (e .- step) ./ norm(e .- step)
            g[d] = (predict_energy(model, ep) - predict_energy(model, em)) / (2δ)
        end
        T[:, a] = cross(SVector{3}(g), SVector{3}(e))   # τ = ∇E × e = −e × ∇E
    end
    return T
end

# Isosceles triangle: atom 1 is the apex, atoms 2,3 are mirror-equivalent. The site
# group is just {E, σ} (perms {e, (23)}) — a proper subgroup of S₃, so a degenerate
# multiset like (1,1,2) splits into two ordering orbits (l=2 on the apex vs on a base
# site), both labeled ls=[1,1,2] but distinct SALCs. Exercises SALCKey injectivity.
function _triangle_cs(; L = 8.0)
    c = SVector{3,Float64}(0.5, 0.5, 0.5)
    offs = [SVector{3,Float64}(0.0, 1.5, 0.0), SVector{3,Float64}(-1.0, 0.0, 0.0),
            SVector{3,Float64}(1.0, 0.0, 0.0)]
    frac = hcat([c + o / L for o in offs]...)
    crystal = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 1, 1], ["Fe"])
    σ = SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 1])
    rots = [SMatrix{3,3,Float64}(I), σ]
    trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
    return crystal, _assemble_spacegroup(crystal, rots, trans, "Cs", 0; tol = 1e-6)
end

# Two-species isosceles triangle: apex (atom 1) is species 2, the mirror-equivalent
# base pair (atoms 2,3) is species 1. With `lmax_by_species = [2, 1]` the base sites
# enumerate l ∈ {1,2} while the apex is capped at l = 1, so `maxl = maximum(lmax) = 2`
# but not every site reaches it — the exact configuration that stresses the Wigner
# cache bound, and (base↔base mixing) the per-orbit `blockcount` / multi-term path.
function _triangle_cs2(; L = 8.0)
    c = SVector{3,Float64}(0.5, 0.5, 0.5)
    offs = [SVector{3,Float64}(0.0, 1.5, 0.0), SVector{3,Float64}(-1.0, 0.0, 0.0),
            SVector{3,Float64}(1.0, 0.0, 0.0)]
    frac = hcat([c + o / L for o in offs]...)
    crystal = Crystal(Lattice(Matrix(L * I(3))), frac, [2, 1, 1], ["Fe", "Co"])
    σ = SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 1])
    rots = [SMatrix{3,3,Float64}(I), σ]
    trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
    return crystal, _assemble_spacegroup(crystal, rots, trans, "Cs", 0; tol = 1e-6)
end

@testset "N-body (3-body triangle)" begin
    crystal, sg = _triangle_c3v()
    nl = build_neighbor_list(crystal, 2.2)        # triangle side r√3 ≈ 2.08
    clusters = build_clusters(crystal, nl, sg; nbody = 3)
    basis = build_salc_basis(crystal, sg, clusters; lmax_by_species = [2])

    @testset "enumeration: a single 3-body orbit of the triangle" begin
        @test haskey(clusters.by_body, 3)
        @test length(clusters.by_body[3]) == 1
        @test clusters.by_body[3][1].species == [1, 1, 1]
        @test all(o -> o.body == 3, clusters.by_body[3])
    end

    @testset "multi-term SALCs exist (unequal l on equivalent sites)" begin
        @test any(s -> SLCE.spin_ls(s.key) == [1, 1, 2], basis.salcs)
        # an (1,1,2) channel must combine the 3 orderings → 3 terms per member
        s112 = first(s for s in basis.salcs if SLCE.spin_ls(s.key) == [1, 1, 2])
        @test length(s112.members[1].terms) == 3
        @test any(s -> length(s.members[1].terms) > 1, basis.salcs)
    end

    @testset "members are canonical and fold exactly (3-body)" begin
        perms = Dict(1 => ([1], [1]), 2 => ([2, 1], [1, 2]),
                     3 => ([2, 3, 1], [1, 3, 2]))
        for s in basis.salcs
            check_canonical_members(s)
            split_roundtrip_exact(s, perms)
        end
    end

    @testset "ground-truth invariance Φ(g·e)=Φ(e) and Φ(-e)=Φ(e)" begin
        rng = MersenneTwister(3)
        for _ = 1:25
            e = _rcfg3(rng, 3)
            for s in basis.salcs
                base = evaluate_salc(s, e)
                for g = 1:length(sg.ops)
                    @test isapprox(evaluate_salc(s, _act3(sg, g, e)), base; atol = 1e-9, rtol = 1e-8)
                end
                @test isapprox(evaluate_salc(s, -e), base; atol = 1e-9, rtol = 1e-8)
            end
        end
    end

    @testset "time reversal: only even-Σl channels" begin
        @test all(s -> iseven(sum(SLCE.spin_ls(s.key))), basis.salcs)
    end

    @testset "SALCs are linearly independent (no spurious channels)" begin
        rng = MersenneTwister(5)
        m = length(basis.salcs)
        configs = [_rcfg3(rng, 3) for _ = 1:6m]
        X = [evaluate_salc(basis.salcs[j], configs[i]) for i = 1:6m, j = 1:m]
        @test rank(X; atol = 1e-8) == m
    end

    @testset "SALCKey is injective when permutations split a degenerate multiset" begin
        xtal_cs, sg_cs = _triangle_cs()
        nl_cs = build_neighbor_list(xtal_cs, 2.2)
        cs_cs = build_clusters(xtal_cs, nl_cs, sg_cs; nbody = 3)
        basis_cs = build_salc_basis(xtal_cs, sg_cs, cs_cs; lmax_by_species = [2])
        @test allunique(basis_cs.keys)                       # the regression the fix guards
        @test issorted(basis_cs.keys)
        @test basis_cs.fingerprint == hash(basis_cs.keys)
        # ls=[1,1,2] splits into two distinct SALCs (apex vs base carries l=2)
        @test count(s -> SLCE.spin_ls(s.key) == [1, 1, 2] && s.Lf == 0, basis_cs.salcs) == 2
        # and they are still individually space-group invariant
        rng = MersenneTwister(13)
        for _ = 1:15
            e = _rcfg3(rng, 3)
            for s in basis_cs.salcs
                base = evaluate_salc(s, e)
                for g = 1:length(sg_cs.ops)
                    @test isapprox(evaluate_salc(s, _act3(sg_cs, g, e)), base; atol = 1e-9, rtol = 1e-8)
                end
            end
        end
    end

    @testset "end-to-end: energy+torque co-fit recovers an in-span 3-body model" begin
        # wrap the manual-symmetry basis in an SLCEBasis (default constructor)
        sceb = SLCEBasis(crystal, sg, basis, BasisSpec(; nbody = 3, cutoff = 2.2, lmax = [2]))
        m = n_salcs(sceb)
        rng = MersenneTwister(9)
        configs = [_rcfg3(rng, 3) for _ = 1:120]
        skel = SLCEDataset(sceb, configs, zeros(length(configs)))
        true_jphi = randn(rng, m)
        true_j0 = 0.25
        energies = true_j0 .+ skel.X_E * true_jphi
        model0 = SLCEModel(sceb, true_j0, true_jphi, sceb.salc_basis.keys)
        torques = [predict_torque(model0, c) for c in configs]

        ds = SLCEDataset(sceb, configs, energies, torques)
        f = fit(SLCEFit, ds, OLS(); torque_weight = 0.3)
        @test isapprox(coef(f), true_jphi; atol = 1e-6)
        @test isapprox(intercept(f), true_j0; atol = 1e-6)
        @test r2_energy(f) ≈ 1.0 atol = 1e-9
        @test r2_torque(f) ≈ 1.0 atol = 1e-9

        # torque is the exact gradient of the fitted energy surface (multi-term path)
        for c in configs[1:5]
            @test isapprox(predict_torque(f, c), _torque_fd3(SLCEModel(f), c); atol = 1e-5)
        end
    end

    @testset "build is deterministic / thread-safe (3-body, multi-species)" begin
        # The orbit loop in `build_salc_basis` runs under `Threads.@threads`; a rebuild
        # must reproduce keys *and* folded tensors byte-for-byte at any thread count.
        # This 3-body, two-species, multi-term orbit (`lmax_by_species = [2, 1]`, base↔
        # base degenerate-multiset split) is the path where the Wigner cache bound and
        # the per-orbit `blockcount` matter most — run with `julia -t N>1` to exercise
        # the parallel completion order (a shared-state race fails the exact checks).
        Threads.nthreads() == 1 &&
            @warn "determinism test runs serial; launch `julia -t N>1` to exercise the threaded path"
        xtal, sgcs = _triangle_cs2()
        nl = build_neighbor_list(xtal, 2.2)
        cl = build_clusters(xtal, nl, sgcs; nbody = 3)
        b1 = build_salc_basis(xtal, sgcs, cl; lmax_by_species = [2, 1])
        b2 = build_salc_basis(xtal, sgcs, cl; lmax_by_species = [2, 1])
        @test length(b1) > 0
        @test issorted(b1.keys)
        @test allunique(b1.keys)
        @test any(s -> s.Lf > 0, b1.salcs)         # anisotropic channels present
        @test b2.keys == b1.keys
        @test b2.fingerprint == b1.fingerprint
        @test length(b2) == length(b1)
        for (s1, s2) in zip(b1.salcs, b2.salcs)
            @test s1.key == s2.key
            @test length(s1.members) == length(s2.members)
            for (m1, m2) in zip(s1.members, s2.members)
                for (t1, t2) in zip(m1.terms, m2.terms)
                    @test t1.slots == t2.slots
                    @test t1.folded == t2.folded   # exact equality, not isapprox
                end
            end
        end
    end
end
