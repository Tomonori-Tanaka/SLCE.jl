# Fitted-model introspection (src/slce/introspect.jl): the code-neutral `spin_multipole_terms` /
# `bilinear_terms` view downstream packages (e.g. the SLCETools.jl samplers) read instead of
# the SALC-basis internals. The gates are energy reconstructions: summing the per-term
# tesseral contraction must reproduce `predict_energy − j0`.

using Test
using SLCE
using LinearAlgebra
using Random
using StaticArrays

const Zlm = SLCE.Harmonics.Zlm

# A 2-body lmax=[2] noncollinear model carries [1,1] bilinear, [1,2]/[2,2] biquadratic, and
# [2] single-ion channels — every body order / l the introspection must round-trip.
function _multichannel_basis()
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    return SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = true))
end

# Random unit-column spin configuration (3 × n).
function _rand_config(rng, n)
    e = randn(rng, 3, n)
    for a = 1:n
        e[:, a] ./= norm(@view e[:, a])
    end
    return e
end

# Energy from the neutral multipole terms: Σ_term coef·(4π)^(N/2)·Σ_μ folded[μ] ∏ Z.
function _energy_from_terms(terms, e)
    tot = 0.0
    for t in terms
        N = t.body
        s = 0.0
        for idx in CartesianIndices(t.folded)
            w = t.folded[idx]
            w == 0.0 && continue
            for i = 1:N
                μ = idx[i] - t.ls[i] - 1
                ei = SVector{3,Float64}(e[1, t.atoms[i]], e[2, t.atoms[i]], e[3, t.atoms[i]])
                w *= Zlm(t.ls[i], μ, ei)
            end
            s += w
        end
        tot += t.coef * (4π)^(N / 2) * s
    end
    return tot
end

# Energy from the general decorated terms, evaluated slot by slot from the PUBLISHED
# fields only (channel, site, k, l, folded, scale) — the independent reconstruction a
# downstream consumer would write. Note `slots[i].site` indexes the term's `atoms`, and
# the scale comes from the field, never from `body`.
function _energy_from_decorated(terms, e, u)
    tot = 0.0
    for t in terms
        s = 0.0
        for idx in CartesianIndices(t.folded)
            w = t.folded[idx]
            w == 0.0 && continue
            for (i, sl) in enumerate(t.slots)
                l = sl.factor.l
                μ = idx[i] - l - 1
                a = t.atoms[sl.site]
                if sl.factor.channel === SLCE.SPIN
                    w *= Zlm(l, μ, SVector{3,Float64}(e[1, a], e[2, a], e[3, a]))
                else
                    uv = SVector{3,Float64}(u[1, a], u[2, a], u[3, a])
                    w *= dot(uv, uv)^sl.factor.k * SLCE.SolidHarmonics.Rlm(l, μ, uv)
                end
            end
            s += w
        end
        tot += t.coef * t.scale * s
    end
    return tot
end

# Energy from the bilinear / single-ion 3×3 matrices: Σ eₐ'·M·e_b + Σ eₐ'·A·eₐ.
function _energy_from_bilinear(bt, e)
    tot = 0.0
    for ((a, b, _), M) in bt.pairs
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        eb = SVector{3,Float64}(e[1, b], e[2, b], e[3, b])
        tot += dot(ea, M * eb)
    end
    for (a, A) in bt.onsites
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        tot += dot(ea, A * ea)
    end
    return tot
end

@testset "fitted-model introspection" begin
    rng = MersenneTwister(2024)
    b = _multichannel_basis()
    keys = b.salc_basis.keys
    K = n_salcs(b)

    @testset "n_atoms(model)" begin
        model = SLCEModel(b, 0.0, randn(rng, K), keys)
        @test n_atoms(model) == 2
    end

    @testset "spin_multipole_terms reproduces predict_energy − j0" begin
        j0 = 0.37
        model = SLCEModel(b, j0, 0.1 .* randn(rng, K), keys)
        terms = spin_multipole_terms(model)
        @test !isempty(terms)
        # The raw coefficient is the fitted jϕ (no (4π)^(N/2) baked in); the body order spans
        # both 1-body (single-ion) and 2-body channels.
        @test Set(t.body for t in terms) == Set([1, 2])
        @test intercept(model) ≈ j0
        for _ = 1:8
            e = _rand_config(rng, 2)
            @test _energy_from_terms(terms, e) ≈ predict_energy(model, e) - j0 atol = 1e-10
        end
    end

    @testset "SpinMultipoleTerm field contract (downstream consumers pin on this)" begin
        # SLCETools.jl's bridge reads exactly these fields; a rename/retype must fail here
        # (and then be synchronized downstream), not slip through the energy gates.
        @test fieldnames(SpinMultipoleTerm) == (:coef, :body, :atoms, :shifts, :ls, :folded)
        terms = spin_multipole_terms(SLCEModel(b, 0.0, randn(MersenneTwister(3), K), keys))
        @test terms isa Vector{SpinMultipoleTerm}
        t = terms[1]
        @test t.coef isa Float64 && t.body isa Int
        @test t.atoms isa Vector{Int} && t.ls isa Vector{Int}
        @test t.shifts isa Vector{SVector{3,Int}}
        @test t.folded isa Array{Float64}
        @test length(t.atoms) == length(t.shifts) == length(t.ls) == t.body == ndims(t.folded)
    end

    @testset "zero-coefficient SALCs are dropped" begin
        jphi = zeros(K)
        jphi[1] = 0.5
        terms = spin_multipole_terms(SLCEModel(b, 0.0, jphi, keys))
        @test all(t -> t.coef == 0.5, terms)        # only the one nonzero SALC's members
    end

    @testset "bilinear_terms reproduces a bilinear + single-ion model" begin
        # Keep only the Sunny-representable channels (ls = [1,1] pair, ls = [2] single-ion);
        # then the bilinear extraction captures the whole energy (the skipped channels exist
        # in the basis but carry a zero coefficient, so they contribute nothing).
        jphi = [SLCE.spin_ls(keys[k]) in ([1, 1], [2]) ? randn(rng) : 0.0 for k = 1:K]
        model = SLCEModel(b, 0.0, jphi, keys)
        bt = bilinear_terms(model)
        for _ = 1:8
            e = _rand_config(rng, 2)
            @test _energy_from_bilinear(bt, e) ≈ predict_energy(model, e) atol = 1e-10
        end
    end

    @testset "bilinear_terms reports the higher-order channels it drops" begin
        model = SLCEModel(b, 0.0, ones(K), keys)
        bt = bilinear_terms(model)
        @test !isempty(bt.skipped)                  # the [1,2] / [2,2] biquadratic channels
    end

    # --- the general (channel-decorated) surface, M4 slice 1 ---------------------
    @testset "decorated_terms and the (4π)^{n_spin_slots/2} scale rule" begin
        crj = Crystal(Lattice(Matrix(3.0 * I(3))),
                      [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        bj = SLCEBasis(crj, BasisSpec(crj; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        kj = bj.salc_basis.keys
        Kj = length(kj)
        j0 = 0.19
        mj = SLCEModel(bj, j0, 0.1 .* randn(rng, Kj), kj)
        terms = decorated_terms(mj)
        @test !isempty(terms) && terms isa Vector{DecoratedTerm}

        # The reconstruction gate: an INDEPENDENT slot-by-slot evaluation of the
        # published fields must reproduce the model's own joint energy.
        for _ = 1:8
            e = _rand_config(rng, 2)
            u = 0.07 .* randn(rng, 3, 2)
            @test _energy_from_decorated(terms, e, u) ≈
                  predict_energy(mj, e, u) - j0 atol = 1e-10
        end

        # Teeth for the scale rule: the fixture must contain terms where the general
        # rule and the pure-spin-era `(4π)^(body/2)` shortcut DISAGREE, and using the
        # shortcut must visibly break the reconstruction. Otherwise the gate above
        # would pass for a consumer that derives the scale from the cluster shape.
        @test any(t -> t.scale != (4π)^(t.body / 2), terms)
        shortcut = [DecoratedTerm(t.coef, (4π)^(t.body / 2), t.body, t.atoms,
                                  t.shifts, t.slots, t.folded) for t in terms]
        e = _rand_config(rng, 2)
        u = 0.07 .* randn(rng, 3, 2)
        @test !isapprox(_energy_from_decorated(shortcut, e, u),
                        predict_energy(mj, e, u) - j0; atol = 1e-6)

        # Slot bookkeeping: canonical axis order (all SPIN, then all DISP), one axis
        # per slot, and a site may carry two slots — that is the whole point of the
        # slot → site map replacing the v4 axis ≡ site identity.
        for t in terms
            @test length(t.slots) == ndims(t.folded)
            @test length(t.atoms) == length(t.shifts) == t.body
            @test all(1 <= s.site <= t.body for s in t.slots)
            chans = [s.factor.channel for s in t.slots]
            @test issorted(chans; by = c -> c === SLCE.SPIN ? 0 : 1)
            @test t.scale ≈ (4π)^(count(==(SLCE.SPIN), chans) / 2)
        end
        @test any(t -> length(t.slots) > t.body, terms)   # a site with two slots
        @test fieldnames(DecoratedTerm) ==
              (:coef, :scale, :body, :atoms, :shifts, :slots, :folded)
    end

    @testset "decorated_terms ≡ spin_multipole_terms on a pure-spin model" begin
        model = SLCEModel(b, 0.0, 0.1 .* randn(rng, K), keys)
        mt = spin_multipole_terms(model)
        dt = decorated_terms(model)
        @test length(dt) == length(mt)
        for (d, m) in zip(dt, mt)
            @test d.coef == m.coef && d.body == m.body
            @test d.atoms == m.atoms && d.shifts == m.shifts && d.folded === m.folded
            @test [s.factor.l for s in d.slots] == m.ls          # the v4 `ls` view
            @test all(s -> s.factor.channel === SLCE.SPIN, d.slots)
            @test d.scale == (4π)^(d.body / 2)                   # the rules agree here
        end
        @test decorated_terms(SLCEModel(b, 0.0, zeros(K), keys)) == DecoratedTerm[]
    end

    @testset "spin_multipole_terms refuses a displacement model and names both hatches" begin
        crj = Crystal(Lattice(Matrix(3.0 * I(3))),
                      [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        specj = BasisSpec(crj; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)])
        bj = SLCEBasis(crj, specj)
        mj = SLCEModel(bj, 0.5, randn(rng, n_salcs(bj)))
        err = try
            spin_multipole_terms(mj)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("decorated_terms", err.msg)
        @test occursin("restrict(model, :spin)", err.msg)
        # the trigger is the SPEC: a model whose displacement couplings all vanish is
        # still a p ≥ 1 model and must still be refused (fail loud, never open)
        zeroed = SLCEModel(bj, 0.5, [any(SLCE.has_disp, k.decors) ? 0.0 : 1.0
                                     for k in bj.salc_basis.keys])
        @test_throws ArgumentError spin_multipole_terms(zeroed)
    end

    @testset "restrict(model, :spin) is the exact clamped-ion sub-model" begin
        crj = Crystal(Lattice(Matrix(3.0 * I(3))),
                      [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        bj = SLCEBasis(crj, BasisSpec(crj; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        kj = bj.salc_basis.keys
        mj = SLCEModel(bj, 0.42, randn(rng, length(kj)))
        ms = restrict(mj, :spin)

        pure = findall(SLCE.is_pure_spin, kj)
        @test 0 < length(pure) < length(kj)            # the fixture has both kinds
        @test ms.keys == kj[pure]
        @test ms.jphi == mj.jphi[pure]                 # coefficients untouched
        @test ms.j0 === mj.j0
        # THE gate: the restricted model IS the joint model at u = 0, bitwise
        z = zeros(3, 2)
        for _ = 1:8
            e = _rand_config(rng, 2)
            @test predict_energy(ms, e) == predict_energy(mj, e, z)
            @test predict_torque(ms, e) == predict_torque(mj, e, z)
        end
        # and the restricted spec is honest, so the pure-spin surfaces accept it
        @test !SLCE._basis_has_disp(ms.basis)
        @test all(iszero, ms.basis.spec.pmax)
        @test !isempty(spin_multipole_terms(ms))
        @test length(decorated_terms(ms)) == length(spin_multipole_terms(ms))
        # a LATTICE-ONLY model has no spin content to keep: the clamped-ion sub-model
        # is the empty one, whose energy is j0 alone — which is still exactly the joint
        # model at u = 0, so the contract holds rather than degenerating
        blat = SLCEBasis(crj, BasisSpec(crj; lmax = 1, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        mlat = SLCEModel(blat, 0.7, randn(rng, n_salcs(blat)))
        rlat = restrict(mlat, :spin)
        @test n_salcs(rlat.basis) == 0 && isempty(rlat.basis.spec.sector_rules)
        @test predict_energy(rlat, _rand_config(rng, 2)) == 0.7 ==
              predict_energy(mlat, _rand_config(rng, 2), z)
        @test spin_multipole_terms(rlat) == SpinMultipoleTerm[]
        # idempotent; a pure-spin model is returned untouched; only :spin is a channel
        @test restrict(ms, :spin) === ms
        @test restrict(SLCEModel(b, 0.0, ones(K), keys), :spin) isa SLCEModel
        @test_throws ArgumentError restrict(mj, :disp)
        # the restricted basis survives a persistence round-trip (its spec must be a
        # legal, disp-free BasisSpec, not a doctored one)
        path = tempname() * ".toml"
        SLCE.save(path, ms)
        back = SLCE.load(SLCEModel, path)
        @test back.jphi == ms.jphi && back.keys == ms.keys
        @test predict_energy(back, _rand_config(rng, 2)) isa Float64
        rm(path)
    end
    @testset "keep_zero: the index map is a property of the basis, not of the values" begin
        # THE HAZARD. By default a SALC with an exactly-zero coefficient emits no term, so
        # the term list — and the index → SALC map a consumer addresses it by — depends on
        # the coefficient VALUES. Two models on the SAME basis (two points of a volume
        # grid, an active-learning refit) can then emit lists of EQUAL LENGTH whose maps
        # are shifted relative to each other, and a coefficient hot-swap writes each value
        # onto a neighbouring cluster with every length check passing.
        rng = MersenneTwister(0x4b30)
        b = _multichannel_basis()
        K = n_salcs(b)
        @test K >= 4
        base = randn(rng, K)
        ja, jb = copy(base), copy(base)
        ja[2] = 0.0                       # same NUMBER of exact zeros...
        jb[3] = 0.0                       # ...at different keys
        ma, mb = SLCEModel(b, 0.0, ja), SLCEModel(b, 0.0, jb)

        shape(t) = (t.body, t.atoms, t.shifts, [(s.site, s.factor) for s in t.slots],
                    t.scale, t.folded)
        da, db = decorated_terms(ma), decorated_terms(mb)
        # the hazard, demonstrated rather than asserted away: equal length, different map
        @test length(da) == length(db)
        @test shape.(da) != shape.(db)

        # ...and the fix: with `keep_zero` the two lists are structurally IDENTICAL and
        # differ only in the coefficients, which is what makes an index-addressed rewrite
        # sound. This is the precondition `SLCEMonteCarlo.set_coefficients!` needs across a
        # `StrainedModels` grid.
        ka, kb = decorated_terms(ma; keep_zero = true), decorated_terms(mb; keep_zero = true)
        @test shape.(ka) == shape.(kb)
        @test length(ka) > length(da)
        @test [t.coef for t in ka] != [t.coef for t in kb]
        # every term of the pruned view survives in the kept one, unchanged
        @test issubset(Set(shape.(da)), Set(shape.(ka)))

        # the default is byte-identical to the pre-keyword behaviour, on both surfaces
        @test shape.(decorated_terms(ma)) == shape.(da)
        sa = spin_multipole_terms(ma)
        @test length(spin_multipole_terms(ma; keep_zero = true)) > length(sa)
        @test [t.coef for t in sa] == [t.coef for t in
                                       spin_multipole_terms(ma; keep_zero = true)
                                       if t.coef != 0.0]
        # a model with no exact zeros is unaffected by the keyword
        mfull = SLCEModel(b, 0.0, base)
        @test shape.(decorated_terms(mfull)) ==
              shape.(decorated_terms(mfull; keep_zero = true))
    end
end
