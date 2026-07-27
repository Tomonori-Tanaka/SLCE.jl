using Test
using SLCE
using SLCE: _assemble_spacegroup, n_ops, SpaceGroup
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

    @testset "the assembler refuses a set that is not a space group" begin
        # Downstream code consumes a `SpaceGroup` AS a group — orbits are reduced by
        # stabilizer counting and SALCs are projected with (1/|G|) Σ_g — so a merely
        # plausible list of matrices does not fail, it silently produces wrong
        # multiplicities and a non-idempotent projector. Each of these used to load.
        cryst = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        E = SMatrix{3,3,Float64}(I)
        z0 = SVector{3,Float64}(0, 0, 0)
        asm(rots, trans; tol = 1e-6) =
            _assemble_spacegroup(cryst, rots, trans, "bad", 0; tol = tol)

        # a 3-fold rotation is not an operation of a cubic lattice: integral it is not
        cz = SMatrix{3,3,Float64}([cos(2π/3) -sin(2π/3) 0; sin(2π/3) cos(2π/3) 0; 0 0 1])
        @test_throws ArgumentError asm([E, cz, cz * cz], fill(z0, 3))
        # integral and det = 1, but a shear: it does not preserve the lattice metric,
        # which integrality alone never implies
        sh = SMatrix{3,3,Float64}([1 1 0; 0 1 0; 0 0 1])
        @test_throws ArgumentError asm([E, sh], fill(z0, 2))
        # not a bijection of the lattice
        @test_throws ArgumentError asm([E, SMatrix{3,3,Float64}(2I)], fill(z0, 2))
        # the identity is what every projector and every orbit is anchored on
        @test_throws ArgumentError asm([SMatrix{3,3,Float64}(-I)], [z0])
        # duplicates inflate every orbit multiplicity
        @test_throws ArgumentError asm([E, E], fill(z0, 2))
        @test_throws ArgumentError asm([E, E], [z0, SVector{3,Float64}(1, 0, 0)])
        # closed under inverses (each is an involution) but not under composition:
        # diag(-1,1,1)·diag(1,-1,1) = diag(-1,-1,1) is absent
        mx = SMatrix{3,3,Float64}(Diagonal([-1.0, 1, 1]))
        my = SMatrix{3,3,Float64}(Diagonal([1.0, -1, 1]))
        @test_throws ArgumentError asm([E, mx, my], fill(z0, 3))
        # rotation parts closed AND every inverse present, yet (W|t) is not:
        # (C2z|0) ∘ (I|½x) = (C2z|−½x) is absent
        c2z = SMatrix{3,3,Float64}(Diagonal([-1.0, -1, 1]))
        hx = SVector{3,Float64}(0.5, 0, 0)
        @test_throws ArgumentError asm([E, E, c2z], [z0, hx, z0])
        # ... and the honest group of that shape loads, on a crystal it maps onto
        halved = Crystal(lat, [0.25 0.75; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        sg = _assemble_spacegroup(halved, [E, E, c2z, c2z], [z0, hx, z0, hx],
                                  "P2/m-like", 0; tol = 1e-6)
        @test n_ops(sg) == 4

        # `map_sym` columns are documented as permutations; two atoms sharing an image
        # means `tol` is too loose for this structure, which the orbit code would then
        # miscount rather than reject
        near = Crystal(lat, [0.0 1e-3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        @test _assemble_spacegroup(near, [E], [z0], "P1", 1; tol = 1e-6) isa SpaceGroup
        @test_throws ErrorException _assemble_spacegroup(near, [E], [z0], "P1", 1;
                                                         tol = 1e-2)
    end

    @testset "unloaded backend → friendly error" begin
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        # Spglib is not loaded in the core test env, so SpglibBackend falls through
        # to the abstract-supertype method, which errors clearly.
        @test_throws ErrorException analyze_symmetry(SpglibBackend(), crystal)
    end
end
