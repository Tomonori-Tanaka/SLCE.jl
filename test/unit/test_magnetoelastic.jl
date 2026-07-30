# The ε-linear magnetoelastic tier (src/slce/magnetoelastic.jl) — design record §12
# gate (u): the B₁/B₂ OUTPUT convention.
#
# What this file exists to pin. Gate (e2) (test_sectorbasis.jl) fixes the SPAN of the
# magnetoelastic block on an octahedral O_h fixture and says in its own comment that it
# fixes no normalization or sign — "C2 is the fit gauge". Everything a published B₁/B₂
# depends on therefore lived nowhere in the package: tensor vs engineering shear (a factor
# 2 on B₂), `α²` vs `α² − 1/3`, `Σ_{i<j}` vs `Σ_{i≠j}` (another factor 2), the overall sign
# of `E_me`, and clamped vs relaxed ion.
#
# The gate is against a closed form written out by hand on the SAME fixture, evaluated
# through the production SALC evaluator at finite (α, ε) — never through the deliverable's
# own path. `magnetoelastic_constants` reaches its answer through
# `strain_derivatives` → `solid_harmonic_poly` monomial coefficients → a least-squares
# projection; the reference here reaches it through `evaluate_salc` → `_eval_term_mixed`
# → a hand-written `Σ_i ε_ii(α_i² − 1/3)` / `2 Σ_{i<j} ε_ij α_i α_j`. Two implementations,
# one number. That is the same fence the (4π) scale rule and the force constants use.

using Test
using SLCE
using SLCE: evaluate_salc, salcs, build_neighbor_list, build_clusters, build_salc_basis,
            _assemble_spacegroup, asr_residual, build_asr, _lift_gamma, has_disp,
            has_spin
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

# The gate-(e2) fixture, promoted to a periodic crystal with a model on it: an Fe(O)₆
# octahedron in a box, spin `l = 2` on the centre and a `degree = 1` displacement shell.
# `zscale ≠ 1` stretches the z bonds and the box together, which drops O_h to D_4h — a
# crystal whose ε-linear response is NOT of the two-constant cubic form, and therefore the
# fixture that shows `residual` doing its job.
#
# Two departures from gate (e2)'s subblock, both forced by the fact that this one carries
# a MODEL and not just a basis. (i) The centre is displacement-active and a pure-lattice
# `degree = 2` sector is present, because a basis whose every SALC is individually
# translation-invariant makes the ASR constraint matrix identically zero — which
# `build_asr` refuses as a broken expansion. (ii) The coefficients are lifted through the
# ASR null space, so the model satisfies the strain path's hard precondition exactly
# rather than by construction. The `degree = 2` content contributes to `O(ε²)` only and
# cannot touch B₁/B₂, which is itself worth having in the fixture.
#
# Whatever the SALC count, the α-dependent ε-linear energy of a CUBIC crystal has exactly
# two invariants — `ε_sym = A_1g ⊕ E_g ⊕ T_2g` against the traceless `α²` content
# `E_g ⊕ T_2g` pairs to `E_g·E_g ⊕ T_2g·T_2g` — which is why the hand fit below is exact
# and why B₁/B₂ is the complete answer here.
function _me_octahedron(; zscale = 1.0, dsh = 2.0, L = 10.0)
    Lz = L * zscale
    lat = Lattice(SMatrix{3,3,Float64}([L 0 0; 0 L 0; 0 0 Lz]))
    c = SVector{3,Float64}(0.5, 0.5, 0.5)
    dirs = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0),
            SVector(0.0, 1.0, 0.0), SVector(0.0, -1.0, 0.0),
            SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0)]
    rvec = [SVector{3,Float64}(dsh * d[1], dsh * d[2], dsh * zscale * d[3]) for d in dirs]
    frac = reduce(hcat, vcat([c], [c + SVector(r[1] / L, r[2] / L, r[3] / Lz)
                                   for r in rvec]))
    cr = Crystal(lat, frac, [1, 2, 2, 2, 2, 2, 2], ["Fe", "O"])
    # O_h for the cube; for the stretched cell keep only the operations that fix the z
    # axis (|R[3,3]| = 1), which is D_4h. `_assemble_spacegroup` validates closure, so a
    # filter that broke the group errors here rather than under-symmetrizing quietly.
    rots = [R for R in oh48_matrices() if zscale == 1 || abs(abs(R[3, 3]) - 1) < 1e-12]
    trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
    sg = _assemble_spacegroup(cr, rots, trans, zscale == 1 ? "Pm-3m" : "P4/mmm", 0;
                              tol = 1e-6)
    rcut = dsh * max(1.0, zscale) + 0.1
    nl = build_neighbor_list(cr, rcut)
    cs = build_clusters(cr, nl, sg; nbody = 2)
    spec = BasisSpec(cr; lmax = ["Fe" => 2, "O" => 0], pmax = ["Fe" => 2, "O" => 2],
                     sectors = [Sector(spin = [2], disp = (degree = 1,), sites = 2,
                                       cutoff = rcut),
                                Sector(disp = (degree = 2,), sites = 1:2, cutoff = rcut)])
    sb = build_salc_basis(cr, sg, cs, spec; neighbors = nl)
    return cr, SLCEBasis(cr, sg, sb, spec), rvec
end

# An ASR-feasible random coefficient vector: the strain path refuses anything else, and
# lifting through the null space is how a fitted model gets there too.
function _me_model(rng, basis; j0 = 0.37, scale = 0.6)
    rep = build_asr(basis)
    rep === nothing && error("fixture lost its displacement content")
    return SLCEModel(basis, j0, scale .* _lift_gamma(rep, randn(rng, size(rep.Z, 2))))
end

# The reference energy, by a route that shares nothing with the deliverable: the affine
# displacement `u_i = ε·(R_i − R_Fe)` substituted BY HAND into the production SALC
# evaluator. The strain origin is the octahedron's centre, so `u_Fe = 0`; the answer must
# not depend on that choice, and does not, because the model satisfies the ASR.
function _me_energy_raw(model, α, ε, rvec)
    nat = n_atoms(model.basis.crystal)
    e = repeat(reshape(collect(Float64, α), 3, 1), 1, nat)
    u = zeros(3, nat)
    for j in eachindex(rvec)
        u[:, j + 1] = ε * rvec[j]
    end
    ss = salcs(model.basis)
    return sum(model.jphi[k] * evaluate_salc(ss[k], e, u) for k in eachindex(model.jphi))
end

# The ε-LINEAR part of it, isolated exactly rather than by a small-ε limit: the odd part
# in ε kills every even power, and the fixture's displacement content stops at degree 2,
# so what is left is the linear term and nothing else. (With degree-3 content this would
# leave an `ε³` remainder — the identity is "odd powers", not "linear".)
_me_energy(model, α, ε, rvec) =
    (_me_energy_raw(model, α, ε, rvec) - _me_energy_raw(model, α, -ε, rvec)) / 2

# The pinned closed form, written out longhand — the `2` and the `i < j` range are typed
# here as literals so that a change to either side of the convention breaks the test.
_me_f1(α, ε) = sum(ε[i, i] * (α[i]^2 - 1 / 3) for i = 1:3)
_me_f2(α, ε) = 2 * sum(ε[i, j] * α[i] * α[j] for i = 1:3 for j = (i + 1):3)
_me_form(B1, B2, V, α, ε) = V * (B1 * _me_f1(α, ε) + B2 * _me_f2(α, ε))

_me_unit(rng) = normalize(randn(rng, 3))
_me_strain(rng, s = 0.1) = (S = s * randn(rng, 3, 3); (S + transpose(S)) / 2)

@testset "magnetoelastic constants (gate u)" begin

    @testset "residual sees content outside the two-constant cubic form" begin
        rng = MersenneTwister(0x5171)
        _, tbasis, _ = _me_octahedron(; zscale = 1.2)
        tmodel = _me_model(rng, tbasis)
        res = @test_logs (:warn, r"two-constant cubic form") match_mode = :any (
            magnetoelastic_constants(tmodel))
        # a tetragonal crystal has three independent ε-linear magnetoelastic constants,
        # not two, so the projection leaves an O(1) fraction behind — the point being
        # that it is REPORTED rather than absorbed into a plausible B₁/B₂
        @test res.residual > 1e-3
        @test isfinite(res.B1) && isfinite(res.B2)
        @test res.ion === :clamped
    end

    @testset "gate (u): B₁/B₂ against a hand-derived closed form" begin
        rng = MersenneTwister(0x0075)
        _, basis, rvec = _me_octahedron()
        # the magnetoelastic block is the (e2) one: two spin-carrying degree-1 invariants
        @test count(s -> any(has_disp, s.key.decors) && any(has_spin, s.key.decors),
                    salcs(basis)) == 2
        model = _me_model(rng, basis)
        # the strain path's hard precondition, met through the ASR null space
        @test asr_residual(model) < 1e-12

        V = abs(det(basis.crystal.lattice.vectors))
        # Independent extraction: sample the hand-substituted energy and solve the 2×2
        # problem against the hand-written closed forms. The tiny fit residual is itself a
        # statement — it says this model IS of the two-constant form, so the comparison
        # below is a convention check and not an approximation.
        M = 20
        A = zeros(M, 2)
        y = zeros(M)
        for m = 1:M
            α, ε = _me_unit(rng), _me_strain(rng)
            A[m, 1] = _me_f1(α, ε)
            A[m, 2] = _me_f2(α, ε)
            y[m] = _me_energy(model, α, ε, rvec)
        end
        x = A \ y
        @test norm(A * x - y) < 1e-10 * norm(y)
        B1_hand, B2_hand = x[1] / V, x[2] / V
        @test abs(B1_hand) > 1e-6 && abs(B2_hand) > 1e-6   # both constants are live

        got = magnetoelastic_constants(model)
        @test got.B1 ≈ B1_hand rtol = 1e-9
        @test got.B2 ≈ B2_hand rtol = 1e-9
        @test got.volume ≈ V
        @test got.residual < 1e-9
        # the clamped-ion qualifier rides in the return value, so `result.B2` cannot be
        # quoted without it (design record §7)
        @test got.ion === :clamped
        @test propertynames(got) == (:B1, :B2, :ion, :residual, :signal, :volume)
        @test got.signal > 1e-6                     # a live projection, not an empty one
    end

    # An EMPTY problem must not score as a perfect one. When the whole ε-linear tier is
    # frozen out by the boundary tie — the generic case on a standard cell of a cubic
    # magnet — `resid / ‖dev‖` is 0/0, and reporting `0.0` would be the best possible
    # value beside `B₁ = B₂ = 0`, indistinguishable from a determined answer. The oracle is
    # the frozen set itself (`unresolvable_columns`, an independent classifier) plus the
    # closed-form fact that a model with no ε-linear content has no response to project.
    @testset "an absent tier reports NaN, not a perfect residual" begin
        # B2 FeRh in its standard cubic cell: the Fe–Rh pair reaches all eight body
        # corners at once, so the ε-linear tier — odd under exchanging a bond's two ends —
        # is annihilated. The model below is a GENERIC ASR-feasible one, so the zero
        # response comes from the freeze and not from a zero coefficient vector.
        cr = Crystal(Lattice(Matrix(2.99 * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 2],
                     ["Fe", "Rh"])
        rots = oh48_matrices()
        sg = _assemble_spacegroup(cr, rots,
                                  [SVector{3,Float64}(0, 0, 0) for _ in rots],
                                  "Pm-3m", 0; tol = 1e-6)
        rcut = 2.7                       # the 1NN Fe–Rh distance is √3/2 · 2.99 = 2.589
        spec = BasisSpec(cr; lmax = 2, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = rcut),
            Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = rcut),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = rcut)])
        nl = build_neighbor_list(cr, rcut)
        cs = build_clusters(cr, nl, sg; nbody = 2)
        sb = build_salc_basis(cr, sg, cs, spec; neighbors = nl)
        basis = SLCEBasis(cr, sg, sb, spec)
        elin = [j for (j, s) in enumerate(salcs(basis))
                if any(has_disp, s.key.decors) && any(has_spin, s.key.decors)]
        @test !isempty(elin)                        # the fixture HAS an ε-linear tier ...
        @test elin ⊆ unresolvable_columns(basis)    # ... and this cell resolves none of it
        model = _me_model(MersenneTwister(0x0b2), basis)
        @test all(==(0.0), model.jphi[elin])        # frozen, exactly
        @test any(!=(0.0), model.jphi)              # but the model is not the zero vector
        got = @test_logs((:warn, r"NO magnetization-dependent"), match_mode = :any,
                         magnetoelastic_constants(model))
        @test isnan(got.residual)                   # 0/0 is undefined, never 0.0
        @test got.B1 == 0.0 && abs(got.B2) == 0.0
        @test got.signal < 1e-12
    end

    @testset "the convention itself: shear factor, range, trace, sign" begin
        rng = MersenneTwister(0x0175)
        _, basis, rvec = _me_octahedron()
        model = _me_model(rng, basis)
        V = abs(det(basis.crystal.lattice.vectors))
        c = magnetoelastic_constants(model)

        # The full statement of the pin. Differencing two magnetization directions at the
        # same ε removes the α-independent response (the reference stress), leaving
        # exactly `E_me`, and the right-hand side is the literal formula. This one
        # assertion pins the factor 2, the `Σ_{i<j}` range, the `−1/3`, and the sign.
        for _ = 1:12
            α1, α2 = _me_unit(rng), _me_unit(rng)
            ε = _me_strain(rng)
            lhs = _me_energy(model, α1, ε, rvec) - _me_energy(model, α2, ε, rvec)
            rhs = _me_form(c.B1, c.B2, V, α1, ε) - _me_form(c.B1, c.B2, V, α2, ε)
            @test lhs ≈ rhs rtol = 1e-9
        end

        # Shear alone isolates B₂, and it is the tensor shear: ε_xy = ε_yx = s (an
        # engineering γ = 2s). With the pinned form the response is `V·B₂·2·s·α_xα_y`, so
        # a convention off by the usual factor 2 fails here by exactly 2.
        s = 0.03
        εs = [0.0 s 0.0; s 0.0 0.0; 0.0 0.0 0.0]
        αd, αz = normalize([1.0, 1.0, 0.0]), SVector(0.0, 0.0, 1.0)
        Δ = _me_energy(model, αd, εs, rvec) - _me_energy(model, αz, εs, rvec)
        @test Δ ≈ V * c.B2 * 2 * s * 0.5 rtol = 1e-9

        # The `−1/3` is what makes B₁ orthogonal to the volume response: under a
        # hydrostatic strain `Σ_i(α_i² − 1/3) = 0`, so a cubic model's volume
        # magnetostriction sits entirely in the α-independent part and never in B₁.
        εh = 0.02 * Matrix(I(3))
        for _ = 1:6
            α1, α2 = _me_unit(rng), _me_unit(rng)
            @test _me_energy(model, α1, εh, rvec) ≈ _me_energy(model, α2, εh, rvec) atol =
                1e-12 * max(1.0, abs(_me_energy(model, α1, εh, rvec)))
        end
    end

    @testset "sublattice signs, and what they may be" begin
        rng = MersenneTwister(0x0275)
        _, basis, _ = _me_octahedron()
        model = _me_model(rng, basis)
        base = magnetoelastic_constants(model)
        # `l = 2` spin content is time-reversal even, so reversing the (only) magnetic
        # sublattice must leave both constants bit-identical — the flip is a real gate on
        # the sign convention of the spin factor, not a tautology
        flipped = magnetoelastic_constants(model; signs = fill(-1.0, 7))
        @test flipped.B1 == base.B1
        @test flipped.B2 == base.B2
        @test_throws DimensionMismatch magnetoelastic_constants(model; signs = [1.0, -1.0])
        @test_throws ArgumentError magnetoelastic_constants(model; signs = fill(0.5, 7))
    end

    @testset "error surface" begin
        rng = MersenneTwister(0x0375)
        _, basis, _ = _me_octahedron()
        model = _me_model(rng, basis)
        @test_throws ArgumentError magnetoelastic_constants(model; tol = -1.0)
        # a lattice-only model has no magnetic state to differentiate against, and
        # returning B₁ = B₂ = 0 for it would read as a physical statement
        cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1],
                     ["Fe"])
        lb = SLCEBasis(cr, BasisSpec(cr; lmax = 0, pmax = 2,
                                     sectors = [Sector(disp = (degree = 2,),
                                                       sites = 1:2, cutoff = 1.1)]))
        lm = SLCEModel(lb, 0.0, randn(rng, n_salcs(lb)))
        @test_throws ArgumentError magnetoelastic_constants(lm)
    end
end

# ---------------------------------------------------------------------------------------
# `exchange_strain_derivatives` — the per-bond half of the same ε-linear tier.
# ---------------------------------------------------------------------------------------
#
# The acceptance gate is a RECONSTRUCTION: contracting every bond's `∂M/∂ε` with the spin
# configuration must rebuild the cell's total ε-linear response, which `strain_derivatives`
# computes by a completely different route (monomial coefficients contracted with site
# positions, never touching the tesseral → Cartesian bilinear conversion). It is the same
# fence `_reconstruct_energy` puts under `bilinear_terms`, one derivative up.

# A dimer chain carrying bilinear (`ls = [1,1]`) exchange dressed with a `degree = 1`
# displacement — the magnetoelastic pair content — plus a pure-lattice `degree = 2` sector
# so the ASR constraint matrix is nonzero. The degree-2 content is `O(ε²)` and cannot
# appear in a first derivative, which is exactly what makes it a useful passenger.
function _ex_chain(rng; extra = Sector[], j0 = 0.0)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    spec = BasisSpec(cr; lmax = 2, pmax = 2,
                     sectors = vcat([Sector(spin = [1, 1], disp = (degree = 1,),
                                            sites = 1:2, cutoff = 1.1),
                                     Sector(disp = (degree = 2,), sites = 1:2,
                                            cutoff = 1.1)], extra))
    basis = SLCEBasis(cr, spec)
    return cr, basis, _me_model(rng, basis; j0 = j0)
end

# The total ε-linear response rebuilt from the per-bond derivatives — `_reconstruct_energy`
# written for a derivative instead of an energy, and by hand rather than through any
# package function.
function _ex_reconstruct(x, e)
    D = zeros(3, 3)
    for ((a, b, _), T) in x.pairs, γ = 1:3, δ = 1:3
        D[γ, δ] += dot(e[:, a], T[:, :, γ, δ] * e[:, b])
    end
    for (a, T) in x.onsites, γ = 1:3, δ = 1:3
        D[γ, δ] += dot(e[:, a], T[:, :, γ, δ] * e[:, a])
    end
    return D
end

@testset "exchange strain derivatives" begin

    @testset "the bonds rebuild the cell's ε-linear response" begin
        rng = MersenneTwister(0x6578)
        _, basis, model = _ex_chain(rng)
        x = exchange_strain_derivatives(model)
        @test !isempty(x.pairs)
        @test isempty(x.skipped)                     # nothing magnetoelastic is missing
        @test x.origin == SVector(0.0, 0.0, 0.0)
        @test all(k -> k[1] <= k[2], keys(x.pairs))  # canonical bond orientation
        for _ = 1:8
            e = reduce(hcat, [_me_unit(rng) for _ = 1:2])
            # UNSYMMETRIZED: the per-bond derivative is with respect to a general affine
            # map, so the comparison has to be against the same convention
            D = strain_derivatives(model; spins = e, order = 1, symmetrize = false)
            @test _ex_reconstruct(x, e) ≈ D rtol = 1e-11
        end
    end

    @testset "the per-bond split is origin-free here, and that is measured" begin
        rng = MersenneTwister(0x6579)
        _, _, model = _ex_chain(rng)
        base = exchange_strain_derivatives(model)                 # check_origin = true
        for o in ([0.7, -1.1, 0.4], [-3.0, 3.0, 12.0])
            other = exchange_strain_derivatives(model; origin = o)
            @test keys(other.pairs) == keys(base.pairs)
            scale = maximum(norm(T) for (_, T) in base.pairs)
            @test maximum(norm(T - other.pairs[k]) for (k, T) in base.pairs) < 1e-10 * scale
            @test other.origin == SVector{3,Float64}(o)
        end
        # a bond orbit with a site-swap operation admits only the relative combination
        # `u_b − u_a`, which is why this passes rather than being an accident of the fit
        @test exchange_strain_derivatives(model; check_origin = false).pairs |> length ==
              length(base.pairs)
    end

    @testset "what is reported missing, and what is not" begin
        rng = MersenneTwister(0x657a)
        # `ls = [2,2]` pair content is real magnetoelastic coupling that the bilinear view
        # cannot express — it must be NAMED, not silently dropped
        _, _, model = _ex_chain(rng; extra = [Sector(spin = [2, 2], disp = (degree = 1,),
                                                     sites = 1:2, cutoff = 1.1)])
        x = exchange_strain_derivatives(model)
        @test !isempty(x.skipped)
        @test any(m -> occursin("bilinear", m), x.skipped)
        # ...while pure-spin SALCs (the couplings themselves) and degree-2 displacement
        # factors (which are O(ε²)) are absent from a first derivative by definition and
        # are not reported as losses
        rng2 = MersenneTwister(0x657b)
        _, _, plain = _ex_chain(rng2; extra = [Sector(spin = (sites = 1:2,), cutoff = 1.1)])
        @test isempty(exchange_strain_derivatives(plain).skipped)
    end

    @testset "error surface" begin
        rng = MersenneTwister(0x657c)
        _, basis, model = _ex_chain(rng)
        @test_throws DimensionMismatch exchange_strain_derivatives(model;
                                                                    origin = [0.0, 0.0])
        @test_throws ArgumentError exchange_strain_derivatives(model; origin = [NaN, 0, 0])
        # the ASR is the same hard precondition as on the strain path itself
        bad = SLCEModel(basis, 0.0, randn(rng, n_salcs(basis)))
        @test asr_residual(bad) > 1e-12
        err = try
            exchange_strain_derivatives(bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("acoustic sum rule", err.msg)
        x = exchange_strain_derivatives(model)
        @test occursin("ExchangeStrainDerivatives", sprint(show, x))
    end
end
