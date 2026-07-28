# Joint gradient kernel (M3 slice 1) — gate (j) at engine level: the two-buffer
# accumulate_grad!(Ge, Gu, salc, e, u, weight) against central finite
# differences (Euclidean in u; tangent-directional in e), bit-identity with the
# spin-only gradient on pure-spin SALCs, exact u = 0 behavior, and the
# error/fast-path surface. Model-level force rows (X_F) land with the M3 data
# layer; this fences the kernel they will share with evaluate_salc.

using Test
using SLCE
using SLCE: evaluate_salc, accumulate_grad!, build_neighbor_list, build_clusters,
    _assemble_spacegroup, SALCScratch
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_jg_unit(rng) = normalize(randn(rng, 3))

# Central-difference displacement gradient of one SALC (per column and
# component). Every displacement factor here has per-site degree ≤ 2, for
# which the central difference is polynomial-exact — the tolerance is pure
# roundoff.
function _jg_fd_u(s, e, u; h = 1e-5)
    G = zeros(3, size(u, 2))
    for a = 1:size(u, 2), μ = 1:3
        up = copy(u)
        um = copy(u)
        up[μ, a] += h
        um[μ, a] -= h
        G[μ, a] = (evaluate_salc(s, e, up) - evaluate_salc(s, e, um)) / (2 * h)
    end
    return G
end

# 5-point central difference — exact for per-column polynomial degree ≤ 4
# (the error term carries the 5th derivative), used for the quartic
# displacement fixture where the 3-point stencil is only O(h²).
function _jg_fd_u5(s, e, u; h = 1e-2)
    G = zeros(3, size(u, 2))
    for a = 1:size(u, 2), μ = 1:3
        f = function (δ)
            v = copy(u)
            v[μ, a] += δ
            return evaluate_salc(s, e, v)
        end
        G[μ, a] = (-f(2 * h) + 8 * f(h) - 8 * f(-h) + f(-2 * h)) / (12 * h)
    end
    return G
end

# Directional on-sphere finite difference along a tangent `t ⊥ e[:, a]` —
# equals `dot(Ge[:, a], t)` for the tangent-projected spin gradient.
function _jg_fd_e(s, e, u, a, t; h = 1e-6)
    ep = copy(e)
    em = copy(e)
    ep[:, a] = normalize(e[:, a] .+ h .* t)
    em[:, a] = normalize(e[:, a] .- h .* t)
    return (evaluate_salc(s, ep, u) - evaluate_salc(s, em, u)) / (2 * h)
end

@testset "joint gradient kernel (gate j, engine level)" begin
    # -- fixture 1: the D4h Fe–Fe bond crystal; mixed 9-SALC family, pure-disp
    # degree-2 content (incl. the |u|² trace channel k = 1), and the dense
    # pure-spin basis for the bit-identity gate.
    latB = Lattice(Matrix(3.0 * I(3)))
    xtalB = Crystal(latB, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    rotsB = [R for R in oh48_matrices() if abs(R[1, 1]) == 1.0]
    trsB = [SVector{3,Float64}(0, 0, 0) for _ in rotsB]
    sgB = _assemble_spacegroup(xtalB, rotsB, trsB, "P4/mmm", 123; tol = 1e-5)
    nlB = build_neighbor_list(xtalB, 1.1)
    csB = build_clusters(xtalB, nlB, sgB; nbody = 2)
    bmix = SLCE.build_salc_basis(xtalB, sgB, csB,
        BasisSpec(xtalB; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)]);
        neighbors = nlB)
    @test length(bmix.salcs) == 9
    bdisp = SLCE.build_salc_basis(xtalB, sgB, csB,
        BasisSpec(xtalB; lmax = 1, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), cutoff = 1.1)]);
        neighbors = nlB)
    @test any(s -> any(d -> d.disp_k == 1, s.key.decors), bdisp.salcs)  # trace k=1
    bspin = SLCEBasis(xtalB, BasisSpec(; nbody = 2, cutoff = 1.1, lmax = [2]))
    # quartic pure-disp content: BOTH halves of the displacement product rule
    # `r^{2k}∇R + 2k r^{2(k−1)}R·u` live simultaneously only for k ≥ 1 AND
    # l ≥ 1 factors (at l = 0 the ∇R half is exactly zero; at k = 0 the radial
    # half is skipped), and k = 2 exercises the r^{2(k−1)} exponent — degree 4
    # supplies (1, 2), (2, 0), and the paired (1, 1)
    bdisp4 = SLCE.build_salc_basis(xtalB, sgB, csB,
        BasisSpec(xtalB; lmax = 1, pmax = 4, sectors = [
            Sector(disp = (degree = 4,), cutoff = 1.1)]);
        neighbors = nlB)
    @test any(s -> any(d -> d.disp_k >= 1 && d.disp_l >= 1, s.key.decors),
              bdisp4.salcs)
    @test any(s -> any(d -> d.disp_k == 2, s.key.decors), bdisp4.salcs)
    # self-bond via AllImages on a 1-atom cell: BOTH slots of each channel fold
    # onto the SAME atom column — the repeated-column half of the chain rule
    lat1 = Lattice(Matrix(2.0 * I(3)))
    xt1 = Crystal(lat1, zeros(3, 1), [1], ["Fe"])
    rots1 = oh48_matrices()
    trs1 = [SVector{3,Float64}(0, 0, 0) for _ in rots1]
    sg1 = _assemble_spacegroup(xt1, rots1, trs1, "Pm-3m", 221; tol = 1e-5)
    nl1 = build_neighbor_list(xt1, 2.1, AllImages())
    cs1 = build_clusters(xt1, nl1, sg1; nbody = 2, selection = AllImages())
    bself = SLCE.build_salc_basis(xt1, sg1, cs1,
        BasisSpec(xt1; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2,
                   cutoff = 2.1)]);
        neighbors = nl1, selection = AllImages())
    @test !isempty(bself.salcs)
    @test all(s -> all(m -> m.atoms == [1, 1], s.members), bself.salcs)

    # -- fixture 2: the octahedral Fe(O)₆ unit (spin l = 2 × shell p = 1) — a
    # 6-member orbit exercises gradient transport across members.
    L = 10.0
    c = SVector{3,Float64}(0.5, 0.5, 0.5)
    shell = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0),
             SVector(0.0, 1.0, 0.0), SVector(0.0, -1.0, 0.0),
             SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0)]
    offs = vcat([SVector{3,Float64}(0, 0, 0)], [2.0 * v for v in shell])
    frac = reduce(hcat, [c + o / L for o in offs])
    crO = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 2, 2, 2, 2, 2, 2],
                  ["Fe", "O"])
    rots = oh48_matrices()
    trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
    sgO = _assemble_spacegroup(crO, rots, trans, "Pm-3m", 0; tol = 1e-6)
    nlO = build_neighbor_list(crO, 2.1)
    csO = build_clusters(crO, nlO, sgO; nbody = 2)
    specO = @test_logs (:warn, r"odd") BasisSpec(crO;
        lmax = ["Fe" => 2, "O" => 0], pmax = ["Fe" => 0, "O" => 1],
        sectors = [Sector(spin = [2], disp = (degree = 1,), sites = 2,
                          cutoff = 2.1)])
    bO = SLCE.build_salc_basis(crO, sgO, csO, specO; neighbors = nlO)
    @test all(s -> length(s.members) == 6, bO.salcs)

    rng = MersenneTwister(0x9ad)

    @testset "finite-difference gate (j): ∂Φ/∂u and tangent ∂Φ/∂e" begin
        for (cases, nat) in ((bmix.salcs, 2), (bdisp.salcs, 2), (bO.salcs, 7),
                             (bself.salcs, 1))
            for s in cases
                e = reduce(hcat, [_jg_unit(rng) for _ = 1:nat])
                u = randn(rng, 3, nat) * 0.4
                Ge = zeros(3, nat)
                Gu = zeros(3, nat)
                accumulate_grad!(Ge, Gu, s, e, u, 1.0)
                @test Gu ≈ _jg_fd_u(s, e, u) atol = 1e-8 rtol = 1e-7
                for a = 1:nat
                    # torque convention made explicit: Ge is tangent-projected
                    @test abs(dot(Ge[:, a], e[:, a])) < 1e-12
                    v = randn(rng, 3)
                    t = v .- dot(v, e[:, a]) .* e[:, a]
                    norm(t) < 1e-6 && continue
                    t ./= norm(t)
                    @test dot(Ge[:, a], t) ≈ _jg_fd_e(s, e, u, a, t) atol = 1e-6
                end
            end
        end
        # quartic content needs the degree-4-exact stencil (3-point is O(h²))
        for s in bdisp4.salcs
            e = reduce(hcat, [_jg_unit(rng) for _ = 1:2])
            u = randn(rng, 3, 2) * 0.4
            Ge = zeros(3, 2)
            Gu = zeros(3, 2)
            accumulate_grad!(Ge, Gu, s, e, u, 1.0)
            @test Gu ≈ _jg_fd_u5(s, e, u) atol = 1e-9 rtol = 1e-7
            @test all(==(0.0), Ge)                     # no spin content
        end
    end

    @testset "pure-spin bit-identity and untouched Gu" begin
        e = reduce(hcat, [_jg_unit(rng) for _ = 1:2])
        u = randn(rng, 3, 2) * 0.3
        for s in SLCE.salcs(bspin)
            Gref = zeros(3, 2)
            accumulate_grad!(Gref, s, e, 0.7)          # spin-only path
            Ge = zeros(3, 2)
            Gu = zeros(3, 2)
            accumulate_grad!(Ge, Gu, s, e, u, 0.7)     # joint path, same weight
            @test Ge == Gref                           # bitwise
            @test all(==(0.0), Gu)
        end
    end

    @testset "u = 0 exactness" begin
        e = reduce(hcat, [_jg_unit(rng) for _ = 1:2])
        u0 = zeros(3, 2)
        # every mixed bond SALC carries a degree-1 disp factor on BOTH sites:
        # the leave-one-out product vanishes at u = 0, so Gu(u = 0) ≡ 0 —
        # and so does Ge (each e-derivative still carries a disp factor)
        for s in bmix.salcs
            Ge = zeros(3, 2)
            Gu = zeros(3, 2)
            accumulate_grad!(Ge, Gu, s, e, u0, 1.0)
            @test all(==(0.0), Gu)
            @test all(==(0.0), Ge)
        end
        # a SINGLE degree-1 factor (Fe(O)₆) has a constant, nonzero gradient
        # at u = 0 that the finite difference reproduces exactly
        eO = reduce(hcat, [_jg_unit(rng) for _ = 1:7])
        u0O = zeros(3, 7)
        hit = false
        for s in bO.salcs
            Ge = zeros(3, 7)
            Gu = zeros(3, 7)
            accumulate_grad!(Ge, Gu, s, eO, u0O, 1.0)
            @test Gu ≈ _jg_fd_u(s, eO, u0O) atol = 1e-9
            hit |= any(!=(0.0), Gu)
        end
        @test hit
    end

    @testset "fast path, accumulation, scratch reuse, errors" begin
        s9 = bmix.salcs[1]
        e = reduce(hcat, [_jg_unit(rng) for _ = 1:2])
        u = randn(rng, 3, 2) * 0.3
        Ge = fill(0.125, 3, 2)
        Gu = fill(-0.25, 3, 2)
        @test accumulate_grad!(Ge, Gu, s9, e, u, 0.0) === (Ge, Gu)
        @test all(==(0.125), Ge) && all(==(-0.25), Gu)   # weight-0 fast path
        # accumulation over calls ≈ one call with the summed weight
        Ge1 = zeros(3, 2)
        Gu1 = zeros(3, 2)
        accumulate_grad!(Ge1, Gu1, s9, e, u, 0.3)
        accumulate_grad!(Ge1, Gu1, s9, e, u, 0.5)
        Ge2 = zeros(3, 2)
        Gu2 = zeros(3, 2)
        accumulate_grad!(Ge2, Gu2, s9, e, u, 0.8)
        @test Ge1 ≈ Ge2 rtol = 1e-12
        @test Gu1 ≈ Gu2 rtol = 1e-12
        # a shared scratch across different SALCs is bit-identical to fresh
        scratch = SALCScratch()
        for s in (bmix.salcs[1], bdisp.salcs[1], bmix.salcs[end])
            Ga = zeros(3, 2)
            Gb = zeros(3, 2)
            accumulate_grad!(Ga, Gb, s, e, u, 1.0, scratch)
            Gc = zeros(3, 2)
            Gd = zeros(3, 2)
            accumulate_grad!(Gc, Gd, s, e, u, 1.0)
            @test Ga == Gc && Gb == Gd
        end
        # error surface: buffer/field size mismatches, and the spin-only path
        # still refuses decorated SALCs by naming the joint form
        @test_throws ArgumentError accumulate_grad!(zeros(3, 2), zeros(3, 2), s9,
                                                    e, zeros(3, 1), 1.0)
        @test_throws ArgumentError accumulate_grad!(zeros(3, 1), zeros(3, 2), s9,
                                                    e, u, 1.0)
        @test_throws ArgumentError accumulate_grad!(zeros(3, 2), zeros(3, 1), s9,
                                                    e, u, 1.0)
        B = zeros(3, 2)
        @test_throws ArgumentError accumulate_grad!(B, B, s9, e, u, 1.0)
        @test_throws ArgumentError accumulate_grad!(zeros(3, 2), s9, e, 1.0)
    end
end
