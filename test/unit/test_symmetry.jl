using Test
using SLCE
using SLCE: _assemble_spacegroup, n_ops
using StaticArrays
using LinearAlgebra

@testset "symmetry" begin
    lat = Lattice(Matrix(3.0 * I(3)))

    @testset "NoSymmetry → P1" begin
        crystal = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        sg = analyze_symmetry(NoSymmetry(), crystal)
        @test sg.symbol == "P1"
        @test sg.number == 1
        @test length(sg.ops) == 1
        @test sg.ops[1].is_translation
        @test sg.ops[1].is_proper
        @test sg.map_sym == reshape([1, 2], 2, 1)
        @test sg.translation_ops == [1]
    end

    @testset "assembler: identity + inversion on a centrosymmetric pair" begin
        cryst = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        rots = [SMatrix{3,3,Float64}(I), SMatrix{3,3,Float64}(-I)]
        trans = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)]
        sg = _assemble_spacegroup(cryst, rots, trans, "manual", 0; tol = 1e-6)
        @test sg.ops[1].is_proper            # identity proper
        @test sg.ops[2].is_proper == false   # inversion improper
        @test isapprox(sg.ops[2].rotation_cart, SMatrix{3,3,Float64}(-I); atol = 1e-12)
        @test sg.map_sym[:, 1] == [1, 2]     # identity
        @test sg.map_sym[:, 2] == [2, 1]     # inversion swaps the pair
        for o = 1:n_ops(sg)
            @test sort(sg.map_sym[:, o]) == [1, 2]   # each op is a permutation
        end
    end

    @testset "unloaded backend → friendly error" begin
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        # Spglib is not loaded in the core test env, so SpglibBackend falls through
        # to the abstract-supertype method, which errors clearly.
        @test_throws ErrorException analyze_symmetry(SpglibBackend(), crystal)
    end
end
