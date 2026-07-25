using Test
using SLCE
using Tables
using LinearAlgebra
using Random

@testset "coeftable (Tables.jl source)" begin
    rng = MersenneTwister(5)
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    basis = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = false))
    m = n_salcs(basis)
    configs = [(E = randn(rng, 3, 2); E ./ sqrt.(sum(abs2, E; dims = 1))) for _ = 1:40]
    f = fit(SLCEFit, SLCEDataset(basis, configs, randn(rng, 40)), OLS())
    c = coeftable(f)

    @testset "Tables.jl interface + schema" begin
        @test c isa SCECoefficients
        @test Tables.istable(typeof(c))
        @test Tables.rowaccess(typeof(c))
        sch = Tables.schema(c)
        @test sch.names == (:body, :orbit_id, :decors, :L_S, :Lf, :block, :J)
        @test sch.types == (Int, Int, String, Int, Int, Int, Float64)
        @test length(c) == m
    end

    @testset "columns carry the right data and types" begin
        ct = Tables.columntable(c)
        @test keys(ct) == (:body, :orbit_id, :decors, :L_S, :Lf, :block, :J)
        @test collect(ct.J) == f.jphi                         # J column == fitted coefficients, in order
        @test eltype(ct.body) == Int
        @test eltype(ct.Lf) == Int
        @test eltype(ct.decors) == String
        # columns agree with the basis keys, positionally
        ks = basis.salc_basis.keys
        @test collect(ct.body) == [k.body for k in ks]
        @test collect(ct.orbit_id) == [k.orbit_id for k in ks]
        @test collect(ct.L_S) == [k.L_S for k in ks]
        @test collect(ct.Lf) == [k.Lf for k in ks]
        @test collect(ct.block) == [k.block for k in ks]
    end

    @testset "decors renders pure-spin labels as the comma ls string" begin
        ks = basis.salc_basis.keys
        @test all(c[i].decors == join(SLCE.spin_ls(ks[i]), ",") for i = 1:m)
        # this 2-body lmax=[2] basis has both single-l (body 1) and multi-l (body 2) labels
        ls_vals = [c[i].decors for i = 1:m]
        @test "2" in ls_vals          # a body-1 l=2 term
        @test "1,1" in ls_vals        # a 2-body (1,1) term, comma-joined
        # mixed decors render the displacement factor explicitly
        mixed = [SLCE.SiteDecor(; spin = 2, disp = (0, 1)), SLCE.SiteDecor(; disp = (1, 0))]
        @test SLCE._decor_string(mixed) == "2+u(0,1),u(1,0)"
    end

    @testset "rows, indexing, iteration, accessors" begin
        @test eltype(c) == NamedTuple{(:body, :orbit_id, :decors, :L_S, :Lf, :block, :J),
                                      Tuple{Int,Int,String,Int,Int,Int,Float64}}
        k1 = basis.salc_basis.keys[1]
        @test c[1] == (body = k1.body, orbit_id = k1.orbit_id,
                       decors = join(SLCE.spin_ls(k1), ","), L_S = k1.L_S, Lf = k1.Lf,
                       block = k1.block, J = f.jphi[1])
        @test length(collect(c)) == m
        @test Tables.rowtable(c) == collect(c)
        @test coef(c) == f.jphi
        @test intercept(c) === f.j0
        # fit and its model give the same table
        cm = coeftable(SLCEModel(f))
        @test Tables.columntable(cm) == Tables.columntable(c)
    end

    @testset "empty model" begin
        eb = SLCEBasis(crystal, BasisSpec(; nbody = 1, cutoff = 0.1, lmax = [0]))
        ce = coeftable(SLCEModel(eb, 0.5, Float64[], eb.salc_basis.keys))
        @test length(ce) == 0
        @test Tables.istable(typeof(ce))
        ct = Tables.columntable(ce)
        @test keys(ct) == (:body, :orbit_id, :decors, :L_S, :Lf, :block, :J)
        @test isempty(ct.J)
        @test intercept(ce) === 0.5
    end

    @testset "display" begin
        @test m > 20                                  # this basis exercises the row cap
        s = sprint(show, MIME("text/plain"), c)
        @test occursin("$m terms", s)
        @test occursin("body", s) && occursin("decors", s)   # header
        @test occursin("⋮ ($(m - 20) more)", s)          # truncation count
        one = repr(c)
        @test occursin("$m terms", one) && occursin("j0=", one)
        # a short table prints no truncation marker
        short = coeftable(SLCEModel(basis, 0.0, zeros(m)[1:1], basis.salc_basis.keys[1:1]))
        @test !occursin("more", sprint(show, MIME("text/plain"), short))
    end

    @testset "errors" begin
        @test_throws ArgumentError SCECoefficients(basis.salc_basis.keys, Float64[], 0.0)  # length mismatch
    end
end
