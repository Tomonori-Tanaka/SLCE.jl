using Test
using SLCE
using SLCE: _assemble_spacegroup, candidate_clusters, _canonical_key
using StaticArrays
using LinearAlgebra

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

    @testset "first site anchored in the home cell" begin
        # single atom: its 2-body "bonds" are self-pairs (i == j), kept only under
        # AllImages — MinimumImage would (correctly) drop them, leaving nothing to test.
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        nl = build_neighbor_list(crystal, 3.1, AllImages())
        cand = candidate_clusters(crystal, nl, 2; selection = AllImages())
        @test !isempty(cand[2])
        @test all(m -> m.shifts[1] == SVector{3,Int}(0, 0, 0), cand[2])
    end
end
