using Test
using SLCE
using LinearAlgebra
using StaticArrays
using Random
using TOML

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
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = true)
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

    # The whole model-level reload contract above runs on a PURE-SPIN basis, whose
    # v5 keys are the value-preserving `spin_decors(ls)` / `L_S = Lf` map of the old
    # v4 ones. A displacement-decorated key exercises the parts that map represents:
    # a real `SiteDecor` list, `L_S ≠ Lf`, slot-based terms — and the joint predicts
    # that read them. Structure and fingerprint are gated for a mixed SLCEBasis in
    # test_sectorbasis.jl; what is gated here is the coefficient re-pairing and the
    # numbers that come out afterwards.
    @testset "decorated model round-trips (keys, coefficients, joint predicts)" begin
        crm = Crystal(Lattice(Matrix(3.0 * I(3))),
                      [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        bm = SLCEBasis(crm, BasisSpec(crm; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        km = bm.salc_basis.keys
        @test any(k -> any(MR.has_disp, k.decors), km)      # the fixture is decorated
        @test any(k -> k.L_S != k.Lf, km)                   # and L_S is a real field
        mm = SLCEModel(bm, 0.31, randn(rng, length(km)), km)
        pm = tempname() * ".toml"
        MR.save(pm, mm)
        m2 = MR.load(SLCEModel, pm)
        @test m2.keys == mm.keys
        @test m2.jphi == mm.jphi                            # bitwise, re-paired by key
        es = [_pcfg(rng, 2) for _ = 1:8]
        us = [0.05 .* randn(rng, 3, 2) for _ = 1:8]
        for (e, u) in zip(es, us)                           # joint predicts, all three
            @test predict_energy(m2, e, u) == predict_energy(mm, e, u)
            @test predict_force(m2, e, u) == predict_force(mm, e, u)
            @test predict_torque(m2, e, u) == predict_torque(mm, e, u)
        end
        # scrambling the on-disk order must not move a decorated model either
        docm = MR._to_doc(mm)
        reverse!(docm["couplings"])
        @test MR._model_from_doc(docm).jphi == mm.jphi
        # `asr_residual` is RECOMPUTED from the basis on demand (never persisted),
        # so a feasible model must still read as feasible after a reload
        rep = MR.build_asr(bm)
        mfeas = SLCEModel(bm, 0.0, rep.Z * randn(rng, size(rep.Z, 2)), km)
        pf = tempname() * ".toml"
        MR.save(pf, mfeas)
        @test asr_residual(mfeas) < 1e-13
        @test asr_residual(MR.load(SLCEModel, pf)) == asr_residual(mfeas)
        rm(pm)
        rm(pf)
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
        @test Int(MR._to_doc(model)["schema_version"]) == MR.PERSIST_SCHEMA_VERSION
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

    @testset "sector-table spec doc round-trips (TOML) and legacy layouts read" begin
        labels = ["Nd", "Fe", "B"]
        sp = BasisSpec(labels; lmax = 2, pmax = ["*" => 0, "Fe" => 2],
                       sectors = [
            Sector(spin = (sites = 2:3, lmax = 2, lsum = 4), cutoff = 8.0),
            Sector(spin = [1, 1], disp = 1:2, soc = false,
                   cutoff = ["Fe-*" => 6.0, "*-*" => Inf])])
        d = MR._spec_doc(sp)
        # exercise the actual serializer (Symbols → Strings, Inf, typemax sentinels)
        buf = IOBuffer()
        TOML.print(buf, Dict("spec" => d))
        d2 = TOML.parse(String(take!(buf)))["spec"]
        sp2 = MR._spec_from(d2)
        @test sp2 == sp
        @test sp2.sectors[2].soc == false && isinf(sp2.sectors[2].cutoff[1, 1])
        @test sp2.sectors[1].spin_lmax == 2 && sp2.sectors[1].spin_lsum == 4
        @test sp2.sectors[2].spin_lmax == SLCE.LSUM_UNCAPPED
        @test sp2.disp_scale == 1.0 && sp2.pmax == [0, 2, 0]
        # a legacy "isotropy"-keyed spec doc (v3/v4 layout) reads with the inversion
        dl = MR._spec_doc(BasisSpec(labels; nbody = 2, cutoff = 3.7, lmax = 1))
        @test dl["soc"] === true && !haskey(dl, "isotropy")
        delete!(dl, "soc")
        delete!(dl, "pmax")
        delete!(dl, "sectors")
        delete!(dl, "disp_scale")
        dl["isotropy"] = true
        spl = MR._spec_from(dl)
        @test spl.soc == false && isempty(spl.sectors)
        @test spl.pmax == [0, 0, 0] && spl.disp_scale == 1.0
    end

    # The v6 rename (schema tags "scefitting/sce-*" → "slce/*", sector key "nbody" →
    # "sites") deliberately kept the READ path compatible: breaking an API costs a
    # call-site edit, breaking the on-disk format strands every model already saved.
    # Without this test that promise is untested code, and the next refactor drops it.
    @testset "pre-rename documents still load (schema tags + the v5 sector key)" begin
        # A SECTOR-CARRYING fixture: the shared `model` above is the dense form, whose
        # `spec.sectors` is empty — downgrading its (nonexistent) sector table would
        # make the second half of this test silently vacuous. The `@test` below is the
        # guard against that happening again.
        crv = Crystal(Lattice(Matrix(3.0 * I(3))),
                      [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        bv = SLCEBasis(crv, BasisSpec(crv; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)]))
        mv = SLCEModel(bv, 0.29, randn(rng, n_salcs(bv)), bv.salc_basis.keys)
        @test length(bv.spec.sectors) == 2          # the downgrade below is not a no-op

        doc = MR._to_doc(mv)
        @test doc["schema"] == "slce/model" && Int(doc["schema_version"]) == 6
        @test all(sec -> haskey(sec, "sites"), doc["spec"]["sectors"])

        # Downgrade to exactly what this package used to write.
        old = deepcopy(doc)
        old["schema"] = "scefitting/sce-model"
        old["schema_version"] = 5
        for sec in old["spec"]["sectors"]
            sec["nbody"] = sec["sites"]             # the v5 spelling
            delete!(sec, "sites")
        end
        @test all(sec -> haskey(sec, "nbody") && !haskey(sec, "sites"),
                  old["spec"]["sectors"])

        m2 = MR._model_from_doc(old)
        @test m2.jphi == mv.jphi && m2.j0 == mv.j0 && m2.keys == mv.keys
        b2 = MR._basis_from_doc(old)
        @test _basis_identical(b2, bv)
        @test b2.spec == bv.spec                    # sector table survives the key rename
        @test [r.sites for r in b2.spec.sectors] == [r.sites for r in bv.spec.sectors]

        # the basis tag is accepted the same way
        oldb = deepcopy(MR._to_doc(bv))
        oldb["schema"] = "scefitting/sce-basis"
        oldb["schema_version"] = 5
        @test _basis_identical(MR._basis_from_doc(oldb), bv)
        # ...and a tag that was never ours is still refused
        bogus = deepcopy(doc); bogus["schema"] = "scefitting/sce-notathing"
        @test_throws ArgumentError MR._basis_from_doc(bogus)
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
