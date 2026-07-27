using Test
using SLCE
using SLCE: cartesian_positions, interplanar_spacing
using StaticArrays
using LinearAlgebra

# Independent brute-force neighbor enumeration over a deliberately over-large image
# shell. This is the *permanent* reference for `build_neighbor_list` — the
# production path (tight `N_d`) is checked against it, never against itself, so a
# future cell-list/k-d-tree upgrade can be validated the same way.
function brute_neighbors(crystal, cutoff; margin = 2)
    lat = crystal.lattice
    A = lat.vectors
    nat = SLCE.n_atoms(crystal)
    cart = cartesian_positions(crystal)
    rng = ntuple(d -> lat.pbc[d] ?
                 (ceil(Int, cutoff * norm(lat.reciprocal[d, :])) + margin) : 0, 3)
    out = Set{NTuple{5,Int}}()
    for i = 1:nat, j = 1:nat
        ri = SVector{3,Float64}(cart[:, i])
        for n1 = -rng[1]:rng[1], n2 = -rng[2]:rng[2], n3 = -rng[3]:rng[3]
            (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
            rj = SVector{3,Float64}(cart[:, j]) + A * SVector{3,Float64}(n1, n2, n3)
            if norm(rj - ri) <= cutoff
                push!(out, (i, j, n1, n2, n3))
            end
        end
    end
    return out
end

keyset(nl) = Set((p.i, p.j, p.shift[1], p.shift[2], p.shift[3]) for p in nl.pairs)

@testset "geometry" begin
    @testset "Lattice reciprocal & interplanar spacing (cubic)" begin
        a = 3.0
        lat = Lattice(Matrix(a * I(3)))
        for d = 1:3
            @test interplanar_spacing(lat, d) ≈ a
        end
    end

    @testset "neighbor list vs brute force (cubic, cutoff > lattice constant)" begin
        a = 2.5
        lat = Lattice(Matrix(a * I(3)))
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        for cutoff in (1.0, 2.6, 5.1, 7.75)   # span several coordination shells
            nl = build_neighbor_list(crystal, cutoff)
            @test keyset(nl) == brute_neighbors(crystal, cutoff)
            @test all(p -> p.distance <= cutoff + 1e-9, nl.pairs)
        end
    end

    @testset "strongly sheared triclinic (1/‖b_i‖ ≠ ‖a_i‖)" begin
        A = [1.0 0.3 0.25;
             0.0 1.0 0.28;
             0.0 0.0 1.0] .* 3.0
        lat = Lattice(A)
        # the cutoff-driven range must use spacing, which is smaller than ‖a_i‖
        @test interplanar_spacing(lat, 1) < norm(A[:, 1]) - 1e-9 ||
              interplanar_spacing(lat, 2) < norm(A[:, 2]) - 1e-9
        frac = [0.0 0.4; 0.0 0.6; 0.0 0.1]
        crystal = Crystal(lat, frac, [1, 1], ["Fe"])
        for cutoff in (2.0, 4.5, 7.0)
            nl = build_neighbor_list(crystal, cutoff)
            @test keyset(nl) == brute_neighbors(crystal, cutoff)
        end
    end

    @testset "non-periodic axis uses no images" begin
        lat = Lattice(Matrix(3.0 * I(3)); pbc = (true, true, false))
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        nl = build_neighbor_list(crystal, 10.0)
        @test all(p -> p.shift[3] == 0, nl.pairs)
    end

    @testset "fractional coords wrapped to [0,1) on periodic axes" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        crystal = Crystal(lat, reshape([1.4, -0.2, 2.5], 3, 1), [1], ["Fe"])
        @test all(0.0 .<= crystal.frac_positions .< 1.0)
        # The half-open end is where `mod` alone fails: a tiny negative input rounds up
        # to exactly 1.0 (`mod(-1e-18, 1.0) === 1.0`), and the AllImages image box is
        # derived assuming |Δf| < 1 strictly. Clean coordinates in every other fixture
        # is why nothing caught it.
        for x in (-1e-18, -1e-17, -5e-16, -0.0)
            c = Crystal(lat, reshape([x, 0.5, 0.5], 3, 1), [1], ["Fe"])
            @test 0.0 <= c.frac_positions[1, 1] < 1.0
            @test mod(x, 1.0) >= 1.0 || c.frac_positions[1, 1] == mod(x, 1.0)
        end
        @test mod(-1e-18, 1.0) === 1.0        # pins WHY the snap is needed, not a typo
    end

    @testset "directed list contains both (i,j,R) and (j,i,-R)" begin
        a = 2.0
        lat = Lattice(Matrix(a * I(3)))
        frac = [0.0 0.5; 0.0 0.0; 0.0 0.0]
        crystal = Crystal(lat, frac, [1, 1], ["Fe"])
        nl = build_neighbor_list(crystal, 1.5)
        ks = keyset(nl)
        for (i, j, n1, n2, n3) in ks
            @test (j, i, -n1, -n2, -n3) in ks
        end
    end
end
