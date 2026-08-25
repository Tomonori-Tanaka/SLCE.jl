using Test
using SLCE
using SLCE: _assemble_spacegroup, candidate_clusters, _canonical_key
using StaticArrays
using LinearAlgebra
using Random: Xoshiro, randn

# A backend returning a fixed, hand-assembled space group, so the SLCEBasis door
# (and its `tie_tol` plumbing) can be exercised with the exact manual groups the
# fixtures below construct — the extension seam `analyze_symmetry` documents.
struct _FixedGroupBackend <: SLCE.AbstractSymmetryBackend
    sg::SpaceGroup
end
SLCE.analyze_symmetry(b::_FixedGroupBackend, ::Crystal; tol::Real = 1e-5) = b.sg

@testset "clusters" begin
    lat = Lattice(Matrix(3.0 * I(3)))

    @testset "NoSymmetry: 1-body unreduced; 2-body merges directed reverses" begin
        crystal = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        nl = build_neighbor_list(crystal, 2.7)
        cs = build_clusters(crystal, nl, analyze_symmetry(NoSymmetry(), crystal); nbody = 2)
        cand = candidate_clusters(crystal, nl, 2)
        # 1-body: each atom is its own orbit (no anchoring freedom for N=1)
        @test length(cs.by_body[1]) == length(cand[1])
        @test all(o -> o.multiplicity == 1, cs.by_body[1])
        # 2-body: a directed pair (i,j,R) and its reverse (j,i,-R) are the same
        # unordered cluster, so each orbit merges exactly those two candidates.
        @test sum(o -> o.multiplicity, cs.by_body[2]) == length(cand[2])
        @test all(o -> o.multiplicity == 2, cs.by_body[2])
    end

    @testset "inversion reduces a centrosymmetric pair to one 1-body orbit" begin
        crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        rots = [SMatrix{3,3,Float64}(I), SMatrix{3,3,Float64}(-I)]
        trans = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)]
        sg = _assemble_spacegroup(crystal, rots, trans, "manual", 0; tol = 1e-6)
        nl = build_neighbor_list(crystal, 2.0)
        cs = build_clusters(crystal, nl, sg; nbody = 2)
        @test length(cs.by_body[1]) == 1            # the two atoms are equivalent
        @test cs.by_body[1][1].multiplicity == 2
        # partition: members sum to the candidate count
        cand = candidate_clusters(crystal, nl, 2)
        @test sum(o -> o.multiplicity, cs.by_body[1]) == length(cand[1])
        @test sum(o -> o.multiplicity, cs.by_body[2]) == length(cand[2])
    end

    @testset "orbit closure: image of a member shares the orbit key" begin
        crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        rots = [SMatrix{3,3,Float64}(I), SMatrix{3,3,Float64}(-I)]
        trans = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)]
        sg = _assemble_spacegroup(crystal, rots, trans, "manual", 0; tol = 1e-6)
        nl = build_neighbor_list(crystal, 2.0)
        cs = build_clusters(crystal, nl, sg; nbody = 2)
        for orbit in vcat(cs.by_body[1], cs.by_body[2])
            key = _canonical_key(crystal, sg, orbit.representative)
            for m in orbit.members
                @test _canonical_key(crystal, sg, m) == key   # all members share the key
            end
        end
    end

    @testset "the same-distance band travels on the list, not as a constant" begin
        # `build_neighbor_list`'s `tol` used to be accepted and then contradicted:
        # every downstream "is this edge inside the radius" decision read a hard-coded
        # 1e-8 instead, so a caller who widened the band got a list built one way and
        # clusters admitted another. The band now lives on the list.
        crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        nl = build_neighbor_list(crystal, 2.0, MinimumImage())
        @test nl.tol == 1e-8                              # the documented default
        d = nl.pairs[1].distance                          # the minimum-image bond
        @test d ≈ 1.2
        # a per-body radius that misses the bond by a relative 1e-4 — far above the
        # default band and far below a widened one
        M = fill(d / (1 + 1e-4), 1, 1)
        tight = candidate_clusters(crystal, nl, 2; cutoff = [M])
        @test isempty(tight[2])
        wide = build_neighbor_list(crystal, 2.0, MinimumImage(); tol = 1e-3)
        @test wide.tol == 1e-3
        @test length(wide.pairs) == length(nl.pairs)      # the list itself is unchanged
        @test !isempty(candidate_clusters(crystal, wide, 2; cutoff = [M])[2])
        # a relative band outside [0, 1) is not a tolerance
        @test_throws ArgumentError SLCE.NeighborList(fill(2.0, 1, 1), nl.pairs, 1.5)
        @test_throws ArgumentError SLCE.NeighborList(fill(2.0, 1, 1), nl.pairs, -1e-9)
    end

    @testset "first site anchored in the home cell" begin
        # single atom: its 2-body "bonds" are self-pairs (i == j), kept only under
        # AllImages — MinimumImage would (correctly) drop them, leaving nothing to test.
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        nl = build_neighbor_list(crystal, 3.1, AllImages())
        cand = candidate_clusters(crystal, nl, 2; selection = AllImages())
        @test !isempty(cand[2])
        @test all(m -> m.shifts[1] == SVector{3,Int}(0, 0, 0), cand[2])
    end

    # The orbit builder must REFUSE a candidate list that is not closed under the
    # space group, never skip the missing image: a skipped image leaves the orbit
    # short — a biased design column — and the SALCs projected on a short orbit are
    # not actually invariant. The failure is REACHABLE, not theoretical: coordinates
    # that are symmetric only approximately (relaxed DFT output; the symmetry
    # analysis at tol ~1e-3 still reports the ideal group) split minimum-image
    # distance ties beyond the neighbor band (1e-8), so symmetry-partner pairs keep
    # DIFFERENT tie images and the candidate set loses closure. On the MnTe(0001)
    # 3×3 slab this silently produced 54 spurious Lf ≠ 0 SALCs whose model broke
    # its own space group by ~4 meV (found 2026-08-12, in the spin-only
    # SCEFitting.jl; the gate itself originated here as d4660a7 — this fixture and
    # the `tie_tol` remedy are back-ported from the SCEFitting fix).
    @testset "orbit build refuses a non-group-closed candidate list; tie_tol heals" begin
        a = 1.0
        hexlat = Matrix([3a 0 0; -3a/2 3a*sqrt(3)/2 0; 0 0 8.0]')
        fr = Float64[]
        sp = Int[]
        for (s, (u, v)) in enumerate([(0.0, 0.0), (1 / 3, 2 / 3)]), i = 0:2, j = 0:2
            append!(fr, [(u + i) / 3, (v + j) / 3, 0.0])
            push!(sp, s)
        end
        frm = reshape(fr, 3, :)
        # exact C3-about-origin × the nine supercell translations (27 ops); the
        # honeycomb B sublattice at (1/3, 2/3) maps onto itself under this group
        E3 = SMatrix{3,3,Float64}(I)
        C3 = SMatrix{3,3,Float64}([0 -1 0; 1 -1 0; 0 0 1])
        rots = SMatrix{3,3,Float64}[]
        trans = SVector{3,Float64}[]
        for W in (E3, C3, C3 * C3), i = 0:2, j = 0:2
            push!(rots, W)
            push!(trans, SVector{3,Float64}(i / 3, j / 3, 0))
        end
        spec_hc = BasisSpec(["Mn", "Te"]; nbody = 2, lmax = [2, 2], cutoff = Inf,
                            lsum = [1 => 2, 2 => 2], soc = true)
        function build_hc(frx; tie = nothing)
            cr = Crystal(Lattice(hexlat; pbc = (true, true, false)), frx, sp,
                         ["Mn", "Te"])
            sg = _assemble_spacegroup(cr, rots, trans, "P3(manual)", 143; tol = 1e-3)
            nl = tie === nothing ?
                 build_neighbor_list(cr, SLCE._superset_cutoff(spec_hc),
                                     MinimumImage()) :
                 build_neighbor_list(cr, SLCE._superset_cutoff(spec_hc),
                                     MinimumImage(); tol = tie)
            return SLCE.build_clusters(cr, nl, sg; nbody = 2,
                                       selection = MinimumImage(),
                                       cutoff = spec_hc.cutoff)
        end
        # the ideal coordinates build (the fixture itself is legal)…
        cl = build_hc(frm)
        @test sum(length, values(cl.by_body)) == 10
        # …and the 1e-5-perturbed ones — same group, split ties — are refused loudly
        frp = frm .+ 1.0e-5 .* randn(Xoshiro(1), size(frm))
        err = try
            build_hc(frp)
            nothing
        catch e
            e
        end
        @test err isa ErrorException
        @test occursin("not closed under the space group", err.msg)
        @test occursin("tie_tol", err.msg)   # the message names the remedy
        # The remedy works: a tie band wider than the split the perturbation causes
        # (frac noise 1e-5 on bonds of a few Å ⇒ relative splits ≲ 1e-4; genuine
        # shell spacings are percents) re-merges the ties, the candidate set closes,
        # and the orbit STRUCTURE (per-body multiplicities and species, not just the
        # total count) is the ideal one again.
        clw = build_hc(frp; tie = 1e-3)
        orbit_shape(cl) = sort([(N, o.multiplicity, o.species)
                                for (N, os) in cl.by_body for o in os])
        @test orbit_shape(clw) == orbit_shape(cl)
        # And end to end through the SLCEBasis door — the assertion that fails if the
        # `tie_tol` keyword were accepted but dropped on the floor: the perturbed
        # crystal refuses at the default band and builds at the widened one, with
        # the SAME SALC keys as the ideal-coordinate build (tensors depend only on
        # the group operations, which the fixed backend pins).
        cr_ideal = Crystal(Lattice(hexlat; pbc = (true, true, false)), frm, sp,
                           ["Mn", "Te"])
        cr_pert = Crystal(Lattice(hexlat; pbc = (true, true, false)), frp, sp,
                          ["Mn", "Te"])
        be_ideal = _FixedGroupBackend(_assemble_spacegroup(cr_ideal, rots, trans,
                                                           "P3(manual)", 143;
                                                           tol = 1e-3))
        be_pert = _FixedGroupBackend(_assemble_spacegroup(cr_pert, rots, trans,
                                                          "P3(manual)", 143;
                                                          tol = 1e-3))
        b_ideal = SLCEBasis(cr_ideal, spec_hc; backend = be_ideal)
        @test_throws ErrorException SLCEBasis(cr_pert, spec_hc; backend = be_pert)
        b_healed = SLCEBasis(cr_pert, spec_hc; backend = be_pert, tie_tol = 1e-3)
        @test b_healed.salc_basis.keys == b_ideal.salc_basis.keys
    end

    # `SLCEBasis(...; tie_tol)` must validate its range — a band at/above the hard
    # cap merges genuinely distinct shells — and thread the value to the neighbor
    # list (candidate_clusters reads it back from `NeighborList.tol`, so one value
    # governs both sides).
    @testset "SLCEBasis tie_tol: validation" begin
        cr = Crystal(Lattice(Matrix(3.0 * I(3))),
                     [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        spec = BasisSpec(["Fe"]; nbody = 2, lmax = [1], cutoff = 1.5,
                         lsum = [1 => 2, 2 => 2], soc = false)
        @test_throws ArgumentError SLCEBasis(cr, spec; tie_tol = -1e-9)
        @test_throws ArgumentError SLCEBasis(cr, spec; tie_tol = Inf)
        @test_throws ArgumentError SLCEBasis(cr, spec; tie_tol = 1e-2)  # at the cap
        b1 = SLCEBasis(cr, spec)
        b2 = SLCEBasis(cr, spec; tie_tol = 1e-6)  # legal; same basis on clean coords
        @test b2.salc_basis.keys == b1.salc_basis.keys
    end
end
