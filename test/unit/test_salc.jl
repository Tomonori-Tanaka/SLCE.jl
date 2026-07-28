using Test
using SLCE
using SLCE: _assemble_spacegroup, evaluate_salc, _canonicalize_members
using StaticArrays
using LinearAlgebra
using Random

# Apply a space-group op to a spin config: (g·e)_a = R_g · e_{preimage(a)}.
function act_on_config(sg, g, e)
    nat = size(e, 2)
    R = Matrix(sg.ops[g].rotation_cart)
    out = similar(e)
    for a = 1:nat
        b = findfirst(==(a), @view sg.map_sym[:, g])   # preimage of a under g
        out[:, a] = R * e[:, b]
    end
    return out
end

@testset "SALC" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    # manual Z₂ space group: identity + inversion (swaps the centrosymmetric pair)
    rots = [SMatrix{3,3,Float64}(I), SMatrix{3,3,Float64}(-I)]
    trans = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)]
    sg = _assemble_spacegroup(crystal, rots, trans, "manual", 0; tol = 1e-6)
    nl = build_neighbor_list(crystal, 1.5)
    clusters = build_clusters(crystal, nl, sg; nbody = 2)
    basis = build_salc_basis(crystal, sg, clusters; lmax_by_species = [2])

    @testset "basis is nonempty and keys are sorted/unique" begin
        @test length(basis) > 0
        @test issorted(basis.keys)
        @test allunique(basis.keys)
        @test basis.fingerprint == hash(basis.keys)
        @test any(s -> s.Lf > 0, basis.salcs)   # anisotropic channels present & tested
    end

    @testset "every SALC is space-group invariant: Φ(g·e) = Φ(e)" begin
        rng = MersenneTwister(7)
        for _ = 1:20
            e = Matrix(randcfg(rng, 2))
            for g = 1:length(sg.ops)
                ge = act_on_config(sg, g, e)
                for s in basis.salcs
                    @test isapprox(evaluate_salc(s, ge), evaluate_salc(s, e); atol = 1e-9, rtol = 1e-8)
                end
            end
        end
    end

    @testset "every SALC is time-reversal even: Φ(-e) = Φ(e)" begin
        rng = MersenneTwister(11)
        for _ = 1:20
            e = Matrix(randcfg(rng, 2))
            for s in basis.salcs
                @test isapprox(evaluate_salc(s, -e), evaluate_salc(s, e); atol = 1e-9, rtol = 1e-8)
            end
        end
    end

    @testset "no odd-Σl channels survive (time reversal)" begin
        for s in basis.salcs
            @test iseven(sum(SLCE.spin_ls(s.key)))
        end
    end

    @testset "members are canonical and fold exactly" begin
        perms = Dict(1 => ([1], [1]), 2 => ([2, 1], [1, 2]))
        for s in basis.salcs
            check_canonical_members(s)
            @test same_members(_canonicalize_members(s.members), s.members)  # idempotent
            split_roundtrip_exact(s, perms)
        end
    end

    @testset "scalar_only keeps only Lf = 0" begin
        iso = build_salc_basis(crystal, sg, clusters; lmax_by_species = [2], scalar_only = true)
        @test all(s -> s.Lf == 0, iso.salcs)
        @test length(iso) > 0
    end

    @testset "build is deterministic / thread-safe" begin
        # `build_salc_basis` processes orbits in parallel (`Threads.@threads`), writing
        # disjoint per-orbit results then sorting by key, so the output is byte-for-byte
        # independent of thread count and of the parallel completion order. A rebuild
        # must reproduce keys *and* the folded tensors exactly; run the suite with
        # `julia -t N>1` to exercise the threaded path (a shared-state race fails here).
        Threads.nthreads() == 1 &&
            @warn "determinism test runs serial; launch `julia -t N>1` to exercise the threaded path"
        b2 = build_salc_basis(crystal, sg, clusters; lmax_by_species = [2])
        @test b2.keys == basis.keys
        @test b2.fingerprint == basis.fingerprint
        @test length(b2) == length(basis)
        for (s1, s2) in zip(basis.salcs, b2.salcs)
            @test s1.key == s2.key
            @test length(s1.members) == length(s2.members)
            for (m1, m2) in zip(s1.members, s2.members)
                for (t1, t2) in zip(m1.terms, m2.terms)
                    @test t1.slots == t2.slots
                    @test t1.folded == t2.folded     # exact equality, not isapprox
                end
            end
        end
    end
end
