using Test
using SLCE
using SLCE: _assemble_spacegroup, evaluate_salc, spin_ls
using StaticArrays
using LinearAlgebra
using Random

# ---------------------------------------------------------------------------
# ABSOLUTE NORMALIZATION OF THE SALC CHAIN.
#
# Why this file exists.  Two stages of the numeric chain are pinned to closed
# forms elsewhere -- the real solid harmonics (`test_harmonics.jl`, "closed-form
# low-l values") and the coupled-tensor norm (`test_coupledbasis.jl`,
# `sum(abs2, cb.tensor) == 2Lf+1`) -- and the stages AFTER them were not:
# the Reynolds projector, `_canonical_basis`, `_canonicalize_members`, the
# `folded` contraction, and `evaluate_salc`.  Everything that did cover them was
# gauge- or scale-INVARIANT, so a uniform rescaling of the whole basis passed
# every gate in the suite.  Measured 2026-08-21 on the C3v triangle: multiplying
# every `folded` tensor by 2 left both the orbit-sum invariance check and a
# member-resolved invariance check at 1e-14, i.e. green.  The four oracles below
# are the ones that see it.
#
# Shared conventions used by all four derivations, each pinned independently:
#   (H)  Z_lm are the STANDARD real solid harmonics on the unit sphere, e.g.
#        Z_1m(u) = sqrt(3/4pi) * (y, z, x) for m = -1, 0, 1
#        [test_harmonics.jl].  Hence the addition theorem
#            sum_m Z_lm(a) Z_lm(b) = ((2l+1)/4pi) * P_l(a.b).
#   (C)  For each (Lf, Mf) the coupled tensor is a unit vector in the m1 (x) m2
#        ... space: `sum(abs2, tensor) == 2Lf+1` over the (2Lf+1) Mf slices
#        [test_coupledbasis.jl].  Over all Lf and Mf the set is an ORTHONORMAL
#        BASIS of that space (Clebsch-Gordan completeness).
#   (S)  A body-N SALC carries the scale (4pi)^(N/2), which is exactly what
#        cancels the 1/sqrt(4pi) each harmonic carries.  This factor is what the
#        oracles below measure; nothing else in the suite does.
#   (F)  Orderings of one cluster are folded with the N! ordering sum, so a
#        2-body channel that survives the fold comes out multiplied by 2.
#
# What these oracles do NOT cover: the sign/gauge of an individual column
# (deliberately -- `_sign_canon!` is allowed to change it, and pinning a signed
# value would make a legitimate gauge change look like a regression), and any
# l > 4 (O2b is measured up to l = 4 only).
#
# TEETH.  Measured 2026-08-21 by mutating the source and re-running this file;
# every oracle has a mutation that kills it and leaves the other three green, so
# none of the four is carrying another's weight:
#
#   mutation of the builder / evaluator                     O1   O2a  O2b  O3
#   ------------------------------------------------------  ---  ---  ---  ---
#   the (4pi)^(N/2) scale halved                            DIE  DIE  DIE  DIE
#   the (4pi)^(1/2) of a 1-body SALC dropped                 ok  DIE   ok   ok
#   a 1.001 mis-scale applied only when some l >= 3          ok   ok  DIE   ok
#   the orbit sum skips the last member                      ok   ok   ok  DIE
#
# The first row is the point of the file: the same mutation left BOTH the
# orbit-sum invariance check in `test_salc.jl` AND a member-resolved invariance
# check green at 1e-14, because every invariance statement is scale-blind.
# ---------------------------------------------------------------------------

# The trivial space group (P1): identity only, so every stabilizer is trivial and
# the Reynolds projector is the identity.  That is what makes the closed forms
# below exhaustive rather than a subspace of themselves.
# The per-site spin ranks of a SALC, in decor order.
_ls(s) = spin_ls(s.key)

_p1(cr) = _assemble_spacegroup(cr, [SMatrix{3,3,Float64}(I)],
                               [SVector{3,Float64}(0, 0, 0)], "P1(manual)", 1;
                               tol = 1e-6)

# An equilateral triangle of three equivalent atoms with a manual C3v group, in a
# HEXAGONAL cell so the 3-fold rotation is integral in the lattice basis (the same
# fixture, and the same reason for the cell shape, as `test_nbody.jl`).
function _norm_triangle_c3v(; a = 6.0, c = 6.0, r = 1.2)
    lat = Lattice(SMatrix{3,3,Float64}([a -a/2 0; 0 a*√3/2 0; 0 0 c]))
    ctr = SVector{3,Float64}(0.5, 0.5, 0.5)
    offs = [SVector{3,Float64}(r * cos(t), r * sin(t), 0.0)
            for t in deg2rad.([90.0, 210.0, 330.0])]
    crystal = Crystal(lat, hcat([ctr + lat.reciprocal * o for o in offs]...),
                      [1, 1, 1], ["Fe"])
    C3 = SMatrix{3,3,Float64}([0 -1 0; 1 -1 0; 0 0 1])
    M1 = SMatrix{3,3,Float64}([-1 1 0; 0 1 0; 0 0 1])
    rots = [SMatrix{3,3,Float64}(I), C3, C3 * C3, M1, C3 * M1, C3 * C3 * M1]
    trans = [(SMatrix{3,3,Float64}(I) - R) * ctr for R in rots]
    return crystal, _assemble_spacegroup(crystal, rots, trans, "C3v", 0; tol = 1e-6)
end

@testset "absolute normalization" begin

    # -- O1 -----------------------------------------------------------------
    # Two sites, one pair member, P1, l1 = l2 = 1.  The Lf = 0 channel is the
    # Heisenberg invariant and its CONSTANT is fixed:
    #   C^{00}_{m1 m2} = delta_{m1 m2} / sqrt(3)      (the only scalar of two
    #                                                 vectors; norm 1 = sqrt(2*0+1))
    #   sum_m Z_1m(a) Z_1m(b) = (3/4pi) (a.b)         (H, with P_1(x) = x)
    #   (S): x (4pi)^(2/2) = 4pi   ==>  sqrt(3) (a.b)
    #   (F): x 2                   ==>  Phi(e) = 2 sqrt(3) (e1.e2)
    @testset "O1  2-body Heisenberg constant is 2*sqrt(3)" begin
        cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.2 -0.2; 0.0 0.0; 0.0 0.0],
                     [1, 1], ["Fe"])
        sg = _p1(cr)
        cl = build_clusters(cr, build_neighbor_list(cr, 1.5), sg; nbody = 2)
        b = build_salc_basis(cr, sg, cl; lmax_by_species = [1])
        pair0 = [s for s in b.salcs if s.body == 2 && s.Lf == 0]
        @test length(pair0) == 1                       # one scalar bilinear, no more
        @test length(pair0[1].members) == 1            # one pair member in this cell
        rng = MersenneTwister(20260821)
        for _ = 1:20
            e = randcfg(rng, 2)
            @test evaluate_salc(pair0[1], e) ≈ 2 * sqrt(3) * dot(e[:, 1], e[:, 2]) rtol = 1e-12
        end
    end

    # -- O2a ----------------------------------------------------------------
    # One site, l = 2, P1.  The projector is I_5, so the five SALCs' folded
    # vectors w^(b) are an ORTHONORMAL basis of R^5 -- which one, and with which
    # signs, is gauge.  The gauge-free statement is the Gram matrix:
    #   sum_b Phi_b(e) Phi_b(e')
    #     = 4pi * sum_{m m'} (sum_b w_m w_m') Z_2m(e) Z_2m'(e')
    #     = 4pi * sum_m Z_2m(e) Z_2m(e')        (completeness of the w's)
    #     = 4pi * (5/4pi) * P_2(e.e')           (H)
    #     = 5 * P_2(e.e').
    # Invariant under any orthogonal remixing X -> XR of the five columns, so a
    # legitimate change of `_canonical_basis`'s gauge cannot break it, while a
    # wrong (4pi)^(1/2) or a wrong harmonic normalization does.
    @testset "O2a 1-body l=2 obeys the addition theorem" begin
        cr = Crystal(Lattice(Matrix(3.0 * I(3))), reshape([0.0, 0.0, 0.0], 3, 1),
                     [1], ["Fe"])
        sg = _p1(cr)
        cl = build_clusters(cr, build_neighbor_list(cr, 1.5), sg; nbody = 1)
        b = build_salc_basis(cr, sg, cl; lmax_by_species = [2])
        onsite = [s for s in b.salcs if s.body == 1 && s.Lf == 2]
        @test length(onsite) == 5
        rng = MersenneTwister(20260821)
        for _ = 1:20
            e = randcfg(rng, 1)
            ep = randcfg(rng, 1)
            lhs = sum(evaluate_salc(s, e) * evaluate_salc(s, ep) for s in onsite)
            c = dot(e[:, 1], ep[:, 1])
            @test lhs ≈ 5 * (3c^2 - 1) / 2 atol = 1e-12
        end
    end

    # -- O2b ----------------------------------------------------------------
    # The same idea one body order up, and the ONLY gate in the suite that reaches
    # l >= 3.  One pair member, P1, l1 = l2 = l, `lsum` opened to 2l:
    #   count: the coupled tensors run over Lf = 0..2l with (2Lf+1) each, all with
    #          a trivial stabilizer, so exactly sum_{Lf} (2Lf+1) = (2l+1)^2 SALCs;
    #   value: by (C) they are an orthonormal basis of the m1 (x) m2 space, so
    #          sum_all Phi^2 = 2^2 (4pi)^2 sum_{m1 m2} Z_lm1(e1)^2 Z_lm2(e2)^2
    #                        = 4 (4pi)^2 ((2l+1)/4pi)^2 = 4 (2l+1)^2,
    #          independent of the configuration.  The leading 4 is (F) squared.
    @testset "O2b 2-body completeness: (2l+1)^2 columns, sum Phi^2 = 4(2l+1)^2" begin
        cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.2 -0.2; 0.0 0.0; 0.0 0.0],
                     [1, 1], ["Fe"])
        sg = _p1(cr)
        cl = build_clusters(cr, build_neighbor_list(cr, 1.5), sg; nbody = 2)
        rng = MersenneTwister(20260821)
        for l = 1:4
            b = build_salc_basis(cr, sg, cl;
                                 lmax_by_species = [l], lsum_by_body = [l, 2l])
            pair = [s for s in b.salcs if s.body == 2 && all(==(l), _ls(s))]
            @test length(pair) == (2l + 1)^2
            @test all(s -> length(s.members) == 1, pair)
            for _ = 1:5
                e = randcfg(rng, 2)
                @test sum(evaluate_salc(s, e)^2 for s in pair) ≈ 4 * (2l + 1)^2 rtol = 1e-12
            end
        end
    end

    # -- O3 -----------------------------------------------------------------
    # MEMBER COUNT.  O1/O2 cannot see it: they compare one SALC against a closed
    # form built from the same single member, so a builder that emitted each member
    # N times would scale both sides.  Here the reference side is built from the
    # THREE GEOMETRIC BONDS of the triangle, read off the configuration and never
    # from `s.members`, so the member multiplicity is under test.
    #   C3v makes the three bonds one orbit and leaves exactly one Lf = 0 pair
    #   channel; each member contributes O1's constant on its own bond, so
    #     Phi(e) = 2 sqrt(3) [ (e1.e2) + (e1.e3) + (e2.e3) ].
    @testset "O3  member multiplicity: the C3v triangle sums three bonds" begin
        cr, sg = _norm_triangle_c3v()
        cl = build_clusters(cr, build_neighbor_list(cr, 3.0), sg; nbody = 2)
        b = build_salc_basis(cr, sg, cl; lmax_by_species = [1])
        @test length(unique(k.orbit_id for k in b.keys if k.body == 2)) == 1
        pair0 = [s for s in b.salcs if s.body == 2 && s.Lf == 0]
        @test length(pair0) == 1
        @test length(pair0[1].members) == 3
        rng = MersenneTwister(20260821)
        for _ = 1:20
            e = randcfg(rng, 3)
            bonds = dot(e[:, 1], e[:, 2]) + dot(e[:, 1], e[:, 3]) + dot(e[:, 2], e[:, 3])
            @test evaluate_salc(pair0[1], e) ≈ 2 * sqrt(3) * bonds rtol = 1e-12
        end
    end
end
