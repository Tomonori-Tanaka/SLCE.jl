using Test
using SLCE
using LinearAlgebra
using StaticArrays
using Random

const MR = SLCE

# Fresh-draw unit config (so anisotropic / Lf > 0 channels do not vanish).
function _pcfg(rng, nat)
    M = Matrix{Float64}(undef, 3, nat)
    for a = 1:nat
        v = randn(rng, 3)
        M[:, a] = v / norm(v)
    end
    return M
end

# Deep, bit-exact structural equality of two bases' SALC content.
function _basis_identical(a::SLCEBasis, b::SLCEBasis)
    a.salc_basis.keys == b.salc_basis.keys || return false
    length(a.salc_basis.salcs) == length(b.salc_basis.salcs) || return false
    for (sa, sb) in zip(a.salc_basis.salcs, b.salc_basis.salcs)
        (sa.key == sb.key && sa.body == sb.body && sa.decors == sb.decors &&
         sa.L_S == sb.L_S && sa.Lf == sb.Lf) || return false
        length(sa.members) == length(sb.members) || return false
        for (ma, mb) in zip(sa.members, sb.members)
            (ma.atoms == mb.atoms && ma.shifts == mb.shifts) || return false
            length(ma.terms) == length(mb.terms) || return false
            for (ta, tb) in zip(ma.terms, mb.terms)
                (ta.slots == tb.slots && ta.folded == tb.folded) || return false
            end
        end
    end
    return true
end

@testset "persistence (TOML)" begin
    rng = MersenneTwister(11)
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = false)
    basis = SLCEBasis(crystal, interaction)   # NoSymmetry (P1); small but exercises Lf > 0
    m = length(basis.salc_basis)
    @test m > 0
    model = SLCEModel(basis, 0.37, randn(rng, m), basis.salc_basis.keys)
    testcfgs = [_pcfg(rng, 2) for _ = 1:6]

    @testset "model round-trip via the format-agnostic document (no file)" begin
        m2 = MR._model_from_doc(MR._to_doc(model))
        @test m2.basis.salc_basis.keys == model.basis.salc_basis.keys
        @test m2.j0 === model.j0
        @test m2.jphi == model.jphi                       # bit-exact
        @test _basis_identical(m2.basis, model.basis)
        @test predict_energy(m2, testcfgs) == predict_energy(model, testcfgs)
        @test predict_torque(m2, testcfgs[1]) == predict_torque(model, testcfgs[1])
    end

    @testset "model round-trip through a TOML file" begin
        path = tempname() * ".toml"
        MR.save(path, model)
        @test isfile(path)
        m2 = MR.load(SLCEModel, path)
        @test m2.j0 === model.j0
        @test m2.jphi == model.jphi
        @test _basis_identical(m2.basis, model.basis)
        @test predict_energy(m2, testcfgs) == predict_energy(model, testcfgs)
        # a basis can be read from a model file too (coefficients ignored)
        b2 = MR.load(SLCEBasis, path)
        @test _basis_identical(b2, model.basis)
        rm(path)
    end

    @testset "basis round-trip through a TOML file" begin
        path = tempname() * ".toml"
        MR.save(path, basis)
        b2 = MR.load(SLCEBasis, path)
        @test _basis_identical(b2, basis)
        @test b2.salc_basis.fingerprint == basis.salc_basis.fingerprint   # recomputed, same session
        @test b2.spec.nbody == basis.spec.nbody
        @test b2.spec.lmax == basis.spec.lmax
        @test b2.crystal.frac_positions == basis.crystal.frac_positions
        rm(path)
    end

    @testset "empty basis (0 SALCs) round-trips" begin
        eb = SLCEBasis(crystal, BasisSpec(; nbody = 1, cutoff = 0.1, lmax = [0]))
        @test length(eb.salc_basis) == 0
        path = tempname() * ".toml"
        MR.save(path, eb)
        eb2 = MR.load(SLCEBasis, path)
        @test length(eb2.salc_basis) == 0
        @test eb2.salc_basis.keys == eb.salc_basis.keys
        em = SLCEModel(eb, 1.25, Float64[], eb.salc_basis.keys)   # 0 coefficients, nonzero j0
        MR.save(path, em)
        em2 = MR.load(SLCEModel, path)
        @test em2.j0 === 1.25
        @test isempty(em2.jphi)
        @test predict_energy(em2, _pcfg(rng, 2)) === 1.25
        rm(path)
    end

    @testset "saving a fit persists its model" begin
        configs = [_pcfg(rng, 2) for _ = 1:40]
        energies = randn(rng, 40)
        ds = SLCEDataset(basis, configs, energies)
        f = fit(SLCEFit, ds, OLS())
        path = tempname() * ".toml"
        MR.save(path, f)                       # SLCEFit saved as its model
        m2 = MR.load(SLCEModel, path)
        @test m2.j0 === f.j0
        @test m2.jphi == f.jphi
        @test predict_energy(m2, testcfgs) == predict_energy(f, testcfgs)
        rm(path)
    end

    @testset "coefficients re-pair to the basis by key, not by position" begin
        doc = MR._to_doc(model)
        reverse!(doc["couplings"])             # scramble the on-disk coefficient order
        m2 = MR._model_from_doc(doc)
        @test m2.jphi == model.jphi            # still correct after re-pairing by key
        @test predict_energy(m2, testcfgs) == predict_energy(model, testcfgs)
    end

    @testset "pre-v4 docs fold to canonical members on load" begin
        # Fabricate a v3-style document: every member expanded into two permuted,
        # de-anchored, half-weight ordered images (the pre-v4 redundancy). Loading
        # must fold it back to the canonical members bit-exactly (0.5·T + 0.5·T).
        doc = MR._to_doc(model)
        doc["schema_version"] = 3
        R0 = SVector(1, 0, -1)
        red = map(model.basis.salc_basis.salcs) do s
            members = MR.SALCMember[]
            for m in s.members
                N = length(m.atoms)
                p = collect(N:-1:1)
                push!(members, MR.SALCMember(m.atoms, m.shifts,
                    [MR.SALCTerm(t.slots, 0.5 .* t.folded) for t in m.terms]))
                push!(members, MR.SALCMember(m.atoms[p], [sh + R0 for sh in m.shifts[p]],
                    [MR.SALCTerm(MR.spin_slots(MR._term_spin_ls(t)[p]),
                                 0.5 .* permutedims(t.folded, p))
                     for t in m.terms]))
            end
            MR.SALC(s.key, s.body, s.decors, s.L_S, s.Lf, members)
        end
        doc["salcs"] = [MR._salc_doc(s) for s in red]
        b2 = MR._basis_from_doc(doc)
        @test _basis_identical(b2, model.basis)
        # a v4+ document is NOT re-folded (loaded verbatim) — and is already canonical
        @test Int(MR._to_doc(model)["schema_version"]) == 5
    end

    @testset "v4 docs back-read through the pure-spin relabel (gate a)" begin
        # Fabricate a v4 document: keys carry the legacy sorted "ls" (no decors,
        # no L_S). Loading must apply the total, value-preserving v4→v5 map
        # (per-site l → pure-spin decor, L_S := Lf) with bit-identical numerics.
        doc = MR._to_doc(model)
        doc["schema_version"] = 4
        function _v4key!(kd)
            kd["ls"] = Int[t[1] for t in kd["decors"]]   # pure-spin: spin_l list
            delete!(kd, "decors")
            delete!(kd, "L_S")
            return kd
        end
        for sd in doc["salcs"]
            _v4key!(sd["key"])
            for md in sd["members"], td in md["terms"]
                td["ls"] = Int[t[4] for t in td["slots"]]   # identity spin slots
                delete!(td, "slots")
            end
        end
        for cd in doc["couplings"]
            _v4key!(cd["key"])
        end
        m2 = MR._model_from_doc(doc)
        @test m2.basis.salc_basis.keys == model.basis.salc_basis.keys
        @test all(k.L_S == k.Lf && SLCE.is_pure_spin(k)
                  for k in m2.basis.salc_basis.keys)
        @test m2.basis.salc_basis.fingerprint == model.basis.salc_basis.fingerprint
        @test m2.jphi == model.jphi
        @test _basis_identical(m2.basis, model.basis)
        @test predict_energy(m2, testcfgs) == predict_energy(model, testcfgs)
        @test predict_torque(m2, testcfgs[1]) == predict_torque(model, testcfgs[1])
        # MC program-array proxy: the introspection dump the downstream adjacency
        # build consumes is identical term-by-term.
        ta = multipole_terms(model)
        tb = multipole_terms(m2)
        @test length(ta) == length(tb)
        @test all(a.coef === b.coef && a.body == b.body && a.atoms == b.atoms &&
                  a.shifts == b.shifts && a.ls == b.ls && a.folded == b.folded
                  for (a, b) in zip(ta, tb))
    end

    @testset "space-group ops round-trip (multi-op P-1)" begin
        # the 2-atom cell is inversion-symmetric (inversion swaps the two sites)
        ops_rot = [SMatrix{3,3,Float64}(I), SMatrix{3,3,Float64}(-1.0 * I)]
        ops_tr = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)]
        sg = MR._assemble_spacegroup(crystal, ops_rot, ops_tr, "P-1", 2; tol = 1e-5)
        @test n_ops(sg) == 2
        sg2 = MR._symmetry_from(crystal, MR._symmetry_doc(sg))
        @test sg2.symbol == sg.symbol && sg2.number == sg.number && sg2.tol == sg.tol
        @test n_ops(sg2) == n_ops(sg)
        @test sg2.map_sym == sg.map_sym
        @test sg2.translation_ops == sg.translation_ops
        for (o1, o2) in zip(sg.ops, sg2.ops)
            @test o2.rotation_frac == o1.rotation_frac
            @test o2.rotation_cart == o1.rotation_cart
            @test o2.translation_frac == o1.translation_frac
            @test o2.is_proper == o1.is_proper
            @test o2.is_translation == o1.is_translation
        end
    end

    @testset "-0.0 is normalized to +0.0 on write" begin
        @test MR._jnum(-0.0) === 0.0
        @test MR._jnum(0.0) === 0.0
        @test MR._jnum(-1.5) === -1.5
    end

    @testset "error paths" begin
        # wrong / unsupported schema
        @test_throws ArgumentError MR._basis_from_doc(
            Dict{String,Any}("schema" => "bogus", "schema_version" => 1))
        bad = MR._to_doc(basis)
        bad["schema_version"] = 999
        @test_throws ArgumentError MR._basis_from_doc(bad)
        # a missing schema_version key must error cleanly (not a MethodError)
        nosv = MR._to_doc(basis)
        delete!(nosv, "schema_version")
        @test_throws ArgumentError MR._basis_from_doc(nosv)
        # a model cannot be loaded from a basis document (no coefficients)
        @test_throws ArgumentError MR._model_from_doc(MR._to_doc(basis))
        # coefficient count mismatch
        short = MR._to_doc(model)
        pop!(short["couplings"])
        @test_throws ArgumentError MR._model_from_doc(short)
        # a coefficient whose key is absent from the basis
        wrongkey = MR._to_doc(model)
        wrongkey["couplings"][1]["key"]["block"] = 9999
        @test_throws ArgumentError MR._model_from_doc(wrongkey)
        # non-injective SALC keys (duplicate design-matrix column)
        dup = MR._to_doc(basis)
        push!(dup["salcs"], deepcopy(dup["salcs"][1]))
        @test_throws ArgumentError MR._basis_from_doc(dup)
    end
end
