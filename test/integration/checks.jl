# The columns of the coverage matrix: one function per column, each taking the row
# and the prepared context, each asserting a property whose oracle is independent
# of the code path under test.
#
# What the oracles are, per column, because that is the only thing that makes an
# end-to-end test worth running:
#
#   census        the International Tables (stated in `roster.jl`, external)
#   structure     structural invariants of a canonical member (sorted, anchored, unique)
#   resolvability rank–nullity of the ASR null space, and the two cells of one crystal
#   invariance    the crystal's own space group: E(g·config) = E(config), a physical identity
#   affine        documented conventions of the rotation diagnostics on real geometry
#   asr           the acoustic modes of D(0), against a deliberately violating truth
#   recovery      the forward map is inverted (property gate, not a value comparison)
#   phonons       Hermiticity, D(−q) = conj D(q), acoustic count
#   fd_hessian    central differences of `predict_energy` (a different code path)
#   effective     an exact identity between two expansions of the same surface
#   restrict      a bitwise identity with the joint model at u = 0
#   persist       a round trip through TOML must not move a prediction
#   terms         re-evaluating the energy from the published term fields only
#   magnetoelastic the pinned B₁/B₂ convention, plus the no-spin refusal
#   strain        central differences of `affine_energy` (a different code path)
#   selection     an exactly representable target must survive every fold
#   selfpair      a documented limitation, pinned so that fixing it shows up here
#
# `check_canonical_members` / `randcfg` / `rand_unit` come from the unit suite's
# shared helpers: the canonical-member gate is the definition of a well-formed
# member, and a second copy of it here would be a second definition.

using Test
using SLCE
using SLCE: salcs, analyze_symmetry, n_ops, build_asr, _basis_has_spin,
            _basis_has_disp
using LinearAlgebra
using Random
using StaticArrays
using Statistics

const Zlm = SLCE.Harmonics.Zlm
const Rlm = SLCE.SolidHarmonics.Rlm

# Columns that actually executed, as (row, column). The driver compares this with
# what the roster declared: a check that silently never ran is the failure mode a
# coverage matrix exists to prevent.
const RAN = Set{Tuple{Symbol,Symbol}}()
mark(row, col) = push!(RAN, (row.id, col))

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

# Everything a row's columns share: the basis, an ASR-feasible ground truth, the
# data it generates, the fit that recovers it, and the model read out of the fit.
#
# The truth is drawn as `Z·g` with `Z` the ASR null-space basis, so it is exactly
# translation invariant AND — because `Z` has exact zero rows at the unresolvable
# columns — supported entirely on the content this cell can determine. Sampling is
# centre-of-mass free (`Σ_a u_a = 0`), the drift-free protocol a displaced-cell DFT
# calculation actually produces.
function prepare(row::Row; seed::Integer = 20260730)
    basis = SLCEBasis(row.crystal, row.spec; backend = SpglibBackend())
    nat = n_atoms(row.crystal)
    m = n_salcs(basis)
    rep = build_asr(basis)
    q = size(rep.Z, 2)
    rng = MersenneTwister(seed)
    beta = rep.Z * randn(rng, q)
    model = SLCEModel(basis, 0.35, beta)
    spinful = _basis_has_spin(basis)
    prov = DatumProvenance(; torque_qualified = spinful, reference_id = "ref",
                           reference_fingerprint = crystal_fingerprint(row.crystal),
                           setup_id = "integration")
    ncfg = max(48, 4q)
    cfgs = Vector{Matrix{Float64}}(undef, ncfg)
    disps = Vector{Matrix{Float64}}(undef, ncfg)
    data = Vector{TrainingDatum}(undef, ncfg)
    for i = 1:ncfg
        e = randcfg(rng, nat)
        u = 0.04 * randn(rng, 3, nat)
        u .-= mean(u; dims = 2)                 # centre-of-mass free
        cfgs[i] = e
        disps[i] = u
        s = spinful ? e : nothing
        data[i] = TrainingDatum(; energy = predict_energy(model, s, u),
                                directions = e, magmoms = spinful ? ones(nat) : zeros(nat),
                                displacements = u,
                                forces = predict_force(model, s, u),
                                torques = spinful ? predict_torque(model, s, u) : nothing,
                                provenance = prov)
    end
    dataset = SLCEDataset(basis, data)
    wT = spinful ? 0.25 : 0.0
    f = fit(SLCEFit, dataset, OLS(); torque_weight = wT, force_weight = 0.25)
    fitted = SLCEModel(f)
    e0 = randcfg(rng, nat)
    return (; basis, nat, m, rep, q, beta, model, fitted, dataset, fit = f, rng,
            spinful, cfgs, disps, prov, e0, spins = spinful ? e0 : nothing,
            frozen = unresolvable_columns(basis), wT)
end

# ---------------------------------------------------------------------------
# census / structure
# ---------------------------------------------------------------------------

function check_census(row::Row, ctx)
    @testset "census" begin
        sg = analyze_symmetry(SpglibBackend(), row.crystal)
        # The Tables, not the package: symbol, number, and |G| = (point ops) ×
        # (lattice points in this cell). `_assemble_spacegroup` has already
        # validated closure and the site map, so a group that lost an element
        # throws before reaching here.
        @test sg.symbol == row.symbol
        @test sg.number == row.number
        @test n_ops(sg) == row.n_ops
        # every operation permutes the atoms and preserves the species
        for o = 1:n_ops(sg)
            p = sg.map_sym[:, o]
            @test sort(p) == collect(1:ctx.nat)
            @test all(row.crystal.species[p[a]] == row.crystal.species[a]
                      for a = 1:ctx.nat)
            @test abs(abs(det(Matrix(sg.ops[o].rotation_cart))) - 1) < 1e-10
        end
        mark(row, :census)
    end
end

function check_structure(row::Row, ctx)
    @testset "structure" begin
        ss = salcs(ctx.basis)
        @test !isempty(ss)
        @test length(ss) == ctx.m
        # one column per key: a duplicated key would make two columns the same
        # function and every readout ambiguous about which one it answered for
        @test allunique(ctx.basis.salc_basis.keys)
        A = Matrix(row.crystal.lattice.vectors)
        P = cartesian_positions(row.crystal)
        # `_superset_cutoff` is a per-species-pair matrix (BasisSpec carries per-body
        # × per-pair cutoffs); the widest entry bounds every member.
        cut = maximum(SLCE._superset_cutoff(row.spec))
        for s in ss
            check_canonical_members(s)
            for mem in s.members
                @test all(1 .<= mem.atoms .<= ctx.nat)
                # every pair of sites inside a member is within the spec's widest
                # cutoff — the cluster builder's own contract, re-derived here from
                # the crystal rather than read back from the neighbor list
                for i in eachindex(mem.atoms), j in eachindex(mem.atoms)
                    i < j || continue
                    d = P[:, mem.atoms[j]] + A * mem.shifts[j] -
                        (P[:, mem.atoms[i]] + A * mem.shifts[i])
                    @test norm(d) <= cut + 1e-8
                end
            end
        end
        mark(row, :structure)
    end
end

# ---------------------------------------------------------------------------
# resolvability
# ---------------------------------------------------------------------------

function check_resolvability(row::Row, ctx)
    @testset "resolvability" begin
        frozen = ctx.frozen
        @test issorted(frozen) && allunique(frozen)
        @test all(1 .<= frozen .<= ctx.m)
        # The freeze is exact, not approximate: the ASR null-space basis has
        # identically zero rows there, so no feasible model can put content in a
        # frozen column and `fit` cannot report one.
        if !isempty(frozen)
            @test maximum(abs, ctx.rep.Z[frozen, :]) == 0.0
            @test all(==(0.0), coef(ctx.fit)[frozen])
        end
        # The reparameterization's ledger: every column is either frozen, spent on
        # a constraint, or free. The freeze and the sum rule are COMPLEMENTARY —
        # `rank` counts constraints among the columns that survived the freeze —
        # so the three counts add up to the basis exactly.
        @test ctx.m == ctx.q + ctx.rep.rank + length(frozen)
        # a frozen column is an identically zero FUNCTION: unit content there
        # changes no energy the cell can express
        for c in Iterators.take(frozen, 4)
            unit = zeros(ctx.m)
            unit[c] = 1.0
            probe = SLCEModel(ctx.basis, 0.0, unit)
            for i = 1:6
                s = ctx.spinful ? ctx.cfgs[i] : nothing
                @test abs(predict_energy(probe, s, ctx.disps[i])) < 1e-12
            end
        end
        mark(row, :resolvability)
    end
end

# ---------------------------------------------------------------------------
# invariance under the crystal's own space group
# ---------------------------------------------------------------------------

# Act with operation `o` on a joint configuration: atoms permute by `map_sym`,
# displacements are POLAR (`u → R u`), spin directions are AXIAL
# (`e → det(R)·R·e`). The energy of a space-group image must be the energy itself
# — the defining property of the SALC projection, and independent of every
# numerical path that produced the coefficients.
function _act_config(sg, o, e, u)
    R = Matrix(sg.ops[o].rotation_cart)
    p = sg.map_sym[:, o]
    ep = e === nothing ? nothing : similar(e)
    up = similar(u)
    d = det(R)
    for a in axes(u, 2)
        up[:, p[a]] = R * u[:, a]
        e === nothing || (ep[:, p[a]] = d * R * e[:, a])
    end
    return ep, up
end

function check_invariance(row::Row, ctx)
    @testset "invariance" begin
        sg = analyze_symmetry(SpglibBackend(), row.crystal)
        model = ctx.model
        for i = 1:3
            s = ctx.spinful ? ctx.cfgs[i] : nothing
            u = ctx.disps[i]
            ref = predict_energy(model, s, u)
            scale = max(abs(ref), 1e-3)
            worst = 0.0
            for o = 1:n_ops(sg)
                sp, up = _act_config(sg, o, s, u)
                worst = max(worst, abs(predict_energy(model, sp, up) - ref))
            end
            @test worst / scale < 1e-11
        end
        # and the invariance is not vacuous: a random rotation that is NOT in the
        # group moves the energy (otherwise the test above would pass on a
        # constant model)
        s = ctx.spinful ? ctx.cfgs[1] : nothing
        u = ctx.disps[1]
        ref = predict_energy(model, s, u)
        R = rand_rotation(MersenneTwister(5))
        up = R * u
        sp = s === nothing ? nothing : R * s
        @test abs(predict_energy(model, sp, up) - ref) > 1e-8 * max(abs(ref), 1e-3)
        mark(row, :invariance)
    end
end

# ---------------------------------------------------------------------------
# affine diagnostics (documented conventions, on real geometry)
# ---------------------------------------------------------------------------

function check_affine(row::Row, ctx)
    @testset "affine" begin
        model = ctx.fitted
        s = ctx.spins
        origins = ([0.0, 0.0, 0.0], [7.0, -3.0, 2.5], [1.3, 0.9, -2.2])
        rr = [rotational_residual(model, s; omega = 0.05, origin = o) for o in origins]
        tr = [rotation_transfer_residual(model, s; omega = 0.05, origin = o)
              for o in origins]
        @test all(isfinite, rr) && all(>=(0.0), rr)
        @test all(isfinite, tr) && all(>=(0.0), tr)
        # Two regimes, and which one a row lands in is a property of the
        # TRUNCATION, not of the fit: a random ASR-feasible model of the joint
        # spin-dressed bases is far from rotationally invariant (measured 0.23 …
        # 0.95 across rows and seeds), while every ASR-feasible model of the
        # lattice-only wurtzite truncation is invariant to machine precision at
        # every origin (measured 1e-16 … 1e-14, both seeds). Assert whichever
        # regime the row is in, and in the invariant one assert it at ALL three
        # origins — a small number at the default origin alone is exactly the
        # false negative the docstring warns about.
        if maximum(rr) > 1e-6
            # the finite-angle contraction has to stay alive: linearizing the
            # rotation returns zero however badly invariance is violated, so a
            # residual that collapses when the angle is halved is the diagnostic
            # breaking, not the model improving
            r1 = rotational_residual(model, s; omega = 0.05)
            r2 = rotational_residual(model, s; omega = 0.025)
            @test r2 > 0.2 * r1
        else
            @test all(<(1e-10), rr)
        end
        # a model with no displacement content has no affine response at all
        if ctx.spinful
            spin_only = restrict(model, :spin)
            @test rotational_residual(spin_only, ctx.e0) == 0.0
        end
        mark(row, :affine)
    end
end

# ---------------------------------------------------------------------------
# ASR
# ---------------------------------------------------------------------------

function check_asr(row::Row, ctx)
    @testset "asr" begin
        @test asr_residual(ctx.fitted) < 1e-12
        @test ctx.dataset.asr isa SLCE.ASRReparam
        @test ctx.dataset.asr.rank == ctx.rep.rank
        # The physical reading of the constraint: D(0) has three zero eigenvalues.
        # "At least" three, not exactly — a truncation can leave further flat
        # directions (measured: the wurtzite lattice-only row has six), and that
        # is a statement about the basis, not about translation invariance.
        nzero(model) = begin
            fcs = force_constants(model; spins = ctx.spins, order = 2)
            ev = eigvals(Hermitian(dynamical_matrix(fcs, [0.0, 0.0, 0.0])))
            scale = maximum(abs, ev)
            scale == 0.0 ? length(ev) : count(x -> abs(x) < 1e-8 * scale, ev)
        end
        @test nzero(ctx.fitted) >= 3
        # THE CONTROL, and it has to be built deliberately. Refitting the same
        # data with `asr = false` proves nothing: the target was generated by a
        # feasible model and the design has no flat direction, so the
        # unconstrained solve returns the same coefficients (measured 3e-15 on the
        # tie-free rows). The constraint only bites when the data pull away from
        # it — so generate a target from a truth that VIOLATES translation
        # invariance and refit both ways.
        rng = MersenneTwister(8123)
        bad_beta = randn(rng, ctx.m)
        bad_beta[ctx.frozen] .= 0.0            # frozen columns move no energy
        if norm(ctx.rep.Z * (ctx.rep.Z' * bad_beta) .- bad_beta) > 1e-8
            bad = SLCEModel(ctx.basis, 0.1, bad_beta)
            mkdata(us) = map(eachindex(ctx.cfgs)) do i
                s = ctx.spinful ? ctx.cfgs[i] : nothing
                TrainingDatum(; energy = predict_energy(bad, s, us[i]),
                              directions = ctx.cfgs[i],
                              magmoms = ctx.spinful ? ones(ctx.nat) : zeros(ctx.nat),
                              displacements = us[i],
                              forces = predict_force(bad, s, us[i]),
                              torques = ctx.spinful ?
                                        predict_torque(bad, s, us[i]) : nothing,
                              provenance = ctx.prov)
            end
            both(ds) = (fit(SLCEFit, ds, OLS(); torque_weight = ctx.wT,
                            force_weight = 0.25, asr = false),
                        fit(SLCEFit, ds, OLS(); torque_weight = ctx.wT,
                            force_weight = 0.25))
            ds = SLCEDataset(ctx.basis, mkdata(ctx.disps))     # centre-of-mass free
            free, held = both(ds)
            # unconstrained: the violation is recovered, and D(0) loses its
            # acoustic modes — that is what the sum rule is buying
            @test asr_residual(SLCEModel(free)) > 1e-6
            @test nzero(SLCEModel(free)) < 3
            # constrained: honored exactly even though the target is outside the
            # feasible space
            @test asr_residual(SLCEModel(held)) < 1e-12
            @test nzero(SLCEModel(held)) >= 3
            # and the two ARE different models — the constraint moved the answer
            @test maximum(abs, coef(held) .- coef(free)) > 1e-6
            # How much that costs in fit is a property of the ROW, not a general
            # fact: with `Σ_a u_a = 0` the energy block barely sees the violating
            # content, but the force block does (`Σ_a f_a` is its signature), so
            # the cost ranges from machine precision on the heavily frozen cells
            # to O(1) on the tie-free ones. Measured on this roster: 1e-27 …
            # 0.69. What IS general is that a sample allowed to drift makes the
            # disagreement observable in every row.
            rngd = MersenneTwister(99)
            drift = [0.04 * randn(rngd, 3, ctx.nat) for _ in eachindex(ctx.cfgs)]
            dsd = SLCEDataset(ctx.basis, mkdata(drift))
            freed, heldd = both(dsd)
            @test rss_energy(heldd) > 10 * rss_energy(freed)
            @test asr_residual(SLCEModel(heldd)) < 1e-12
        end
        mark(row, :asr)
    end
end

# ---------------------------------------------------------------------------
# recovery
# ---------------------------------------------------------------------------

function check_recovery(row::Row, ctx)
    @testset "recovery" begin
        # in-span recovery is a theoretical guarantee, so the tolerance is
        # machine precision times the coefficient scale, not a fitted number
        scale = max(maximum(abs, ctx.beta), 1.0)
        @test maximum(abs, coef(ctx.fit) .- ctx.beta) < 1e-9 * scale
        @test abs(intercept(ctx.fit) - 0.35) < 1e-9
        @test r2_energy(ctx.fit) > 1 - 1e-12
        # the design determines what was fitted: no flat direction left
        id = identifiability(ctx.dataset; torque_weight = ctx.wT, force_weight = 0.25)
        @test id.nullity == 0
        @test id.ncols == ctx.q
        # held-out configurations, drawn after the fit: the recovered model is the
        # same FUNCTION, not merely the same numbers on the training slice
        rng = MersenneTwister(97)
        for _ = 1:6
            e = randcfg(rng, ctx.nat)
            u = 0.06 * randn(rng, 3, ctx.nat)
            u .-= mean(u; dims = 2)
            s = ctx.spinful ? e : nothing
            ref = predict_energy(ctx.model, s, u)
            @test abs(predict_energy(ctx.fitted, s, u) - ref) < 1e-9 * max(abs(ref), 1.0)
            @test maximum(abs, predict_force(ctx.fitted, s, u) .-
                               predict_force(ctx.model, s, u)) < 1e-9
            ctx.spinful && @test maximum(abs, predict_torque(ctx.fitted, s, u) .-
                                              predict_torque(ctx.model, s, u)) < 1e-9
        end
        mark(row, :recovery)
    end
end

# ---------------------------------------------------------------------------
# phonons
# ---------------------------------------------------------------------------

function check_phonons(row::Row, ctx)
    @testset "phonons" begin
        fcs = force_constants(ctx.fitted; spins = ctx.spins, order = 2)
        @test fcs.order == 2
        @test !isempty(fcs.constants)
        D0 = dynamical_matrix(fcs, [0.0, 0.0, 0.0])
        @test size(D0) == (3 * ctx.nat, 3 * ctx.nat)
        @test maximum(abs, D0 .- D0') < 1e-10 * max(maximum(abs, D0), 1.0)
        for q in ([0.25, 0.0, 0.0], [1 / 3, 1 / 3, 0.0], [0.5, 0.5, 0.5],
                  [0.13, -0.29, 0.41])
            D = dynamical_matrix(fcs, q)
            # Hermitian, and the real-field condition D(−q) = conj D(q)
            @test maximum(abs, D .- D') < 1e-10 * max(maximum(abs, D), 1.0)
            @test maximum(abs, dynamical_matrix(fcs, -q) .- conj.(D)) <
                  1e-10 * max(maximum(abs, D), 1.0)
            @test all(isreal, eigvals(Hermitian(D)))
        end
        # masses rescale the matrix by 1/√(m_a m_b) — the same D at unit masses
        mass = fill(2.0, ctx.nat)
        @test maximum(abs, dynamical_matrix(fcs, [0.25, 0.0, 0.0]; masses = mass) .-
                           dynamical_matrix(fcs, [0.25, 0.0, 0.0]) ./ 2.0) < 1e-12
        mark(row, :phonons)
    end
end

# Γ-restricted Hessian of the model's own energy by central differences: the
# uniform-per-cell displacement field the evaluator takes, differentiated
# numerically. An independent path to the same object `force_constants` builds
# analytically from the monomial coefficients.
function _fd_hessian(model, s, nat; h = 1e-5)
    H = zeros(3nat, 3nat)
    δ(i, x) = (z = zeros(3, nat); z[x, i] += h; z)
    for a = 1:nat, α = 1:3, b = 1:nat, β = 1:3
        d1, d2 = δ(a, α), δ(b, β)
        H[3(a - 1) + α, 3(b - 1) + β] =
            (predict_energy(model, s, d1 .+ d2) - predict_energy(model, s, d1 .- d2) -
             predict_energy(model, s, -d1 .+ d2) + predict_energy(model, s, -d1 .- d2)) /
            (4h^2)
    end
    return H
end

# Σ_R of the anchored constants, flattened atom-major / Cartesian-minor: every
# lattice shift folds onto the same pair of atoms at Γ.
function _gamma_sum(fcs, nat)
    S = zeros(ntuple(_ -> 3nat, fcs.order))
    for ((atoms, _), T) in fcs.constants
        for cidx in CartesianIndices(T)
            idx = ntuple(i -> 3 * (atoms[i] - 1) + cidx[i], fcs.order)
            S[idx...] += T[cidx]
        end
    end
    return S
end

function check_fd_hessian(row::Row, ctx)
    @testset "fd_hessian" begin
        fcs = force_constants(ctx.fitted; spins = ctx.spins, order = 2)
        S = _gamma_sum(fcs, ctx.nat)
        H = _fd_hessian(ctx.fitted, ctx.spins, ctx.nat)
        scale = max(maximum(abs, H), 1.0)
        @test maximum(abs, S .- H) < 1e-5 * scale          # h² central-difference floor
        # the Hessian is symmetric under the pair exchange, and Σ_a Φ_ab = 0 is the
        # sum rule in its Γ form
        @test maximum(abs, S .- S') < 1e-10 * scale
        blocks = [sum(S[3(a - 1) + α, 3(b - 1) + β] for a = 1:ctx.nat)
                  for α = 1:3, b = 1:ctx.nat, β = 1:3]
        @test maximum(abs, blocks) < 1e-9 * scale
        mark(row, :fd_hessian)
    end
end

# ---------------------------------------------------------------------------
# effective model / restrict
# ---------------------------------------------------------------------------

function check_effective(row::Row, ctx)
    @testset "effective" begin
        rng = MersenneTwister(303)
        u0 = 0.03 * randn(rng, 3, ctx.nat)
        u0 .-= mean(u0; dims = 2)
        em = effective_model(ctx.fitted; u0 = u0)
        for _ = 1:5
            du = 0.02 * randn(rng, 3, ctx.nat)
            ref = predict_energy(ctx.fitted, ctx.spins, u0 .+ du)
            got = predict_energy(em, ctx.spins, du)
            @test abs(got - ref) < 1e-10 * max(abs(ref), 1.0)
        end
        # δu = 0 is the reference energy itself
        @test abs(predict_energy(em, ctx.spins, zeros(3, ctx.nat)) -
                  predict_energy(ctx.fitted, ctx.spins, u0)) < 1e-10
        mark(row, :effective)
    end
end

function check_restrict(row::Row, ctx)
    @testset "restrict" begin
        sub = restrict(ctx.fitted, :spin)
        @test !_basis_has_disp(sub.basis)
        z = zeros(3, ctx.nat)
        if ctx.spinful
            for i = 1:5
                # bitwise, not approximately: the clamped-ion sub-model IS the
                # joint model at u = 0
                @test predict_energy(sub, ctx.cfgs[i]) ==
                      predict_energy(ctx.fitted, ctx.cfgs[i], z)
                @test predict_torque(sub, ctx.cfgs[i]) ==
                      predict_torque(ctx.fitted, ctx.cfgs[i], z)
            end
        else
            # a lattice-only model has no spin content to keep: the clamped-ion
            # sub-model is the empty one, whose energy is j0 alone — and that is
            # still exactly the joint model at u = 0
            @test n_salcs(sub.basis) == 0
            @test predict_energy(sub, ctx.cfgs[1]) ==
                  predict_energy(ctx.fitted, nothing, z)
        end
        mark(row, :restrict)
    end
end

# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------

function check_persist(row::Row, ctx)
    @testset "persist" begin
        path = tempname() * ".toml"
        try
            SLCE.save(path, ctx.fitted)
            back = SLCE.load(SLCEModel, path)
            @test back.jphi == ctx.fitted.jphi
            @test back.keys == ctx.fitted.keys
            @test back.j0 === ctx.fitted.j0
            # the reloaded model is the same predictor, bitwise, on all three
            # observables — the round trip has to survive the DECORATED keys, not
            # only the pure-spin ones
            for i = 1:4
                s = ctx.spinful ? ctx.cfgs[i] : nothing
                u = ctx.disps[i]
                @test predict_energy(back, s, u) == predict_energy(ctx.fitted, s, u)
                @test predict_force(back, s, u) == predict_force(ctx.fitted, s, u)
                ctx.spinful &&
                    @test predict_torque(back, s, u) == predict_torque(ctx.fitted, s, u)
            end
            # and the physics readouts agree with the ones taken from the original
            @test asr_residual(back) == asr_residual(ctx.fitted)
            fb = force_constants(back; spins = ctx.spins, order = 2)
            fo = force_constants(ctx.fitted; spins = ctx.spins, order = 2)
            @test keys(fb.constants) == keys(fo.constants)
            @test all(fb.constants[k] == fo.constants[k] for k in keys(fo.constants))
            # the basis survives its own round trip too
            bpath = tempname() * ".toml"
            SLCE.save(bpath, ctx.basis)
            bb = SLCE.load(SLCEBasis, bpath)
            @test n_salcs(bb) == ctx.m
            @test bb.salc_basis.keys == ctx.basis.salc_basis.keys
            rm(bpath)
        finally
            isfile(path) && rm(path)
        end
        mark(row, :persist)
    end
end

# ---------------------------------------------------------------------------
# published term contract
# ---------------------------------------------------------------------------

# Energy from the published `DecoratedTerm` fields ONLY (atoms, slots, folded,
# coef, scale) — the reconstruction a downstream consumer writes. `slots[i].site`
# indexes the term's own `atoms`, the displacement field is cell-periodic so an
# atom index is enough, and the `(4π)` power comes from the `scale` field, never
# from the cluster shape.
function _energy_from_terms(terms, e, u)
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
                    w *= dot(uv, uv)^sl.factor.k * Rlm(l, μ, uv)
                end
            end
            s += w
        end
        tot += t.coef * t.scale * s
    end
    return tot
end

function check_terms(row::Row, ctx)
    @testset "terms" begin
        terms = decorated_terms(ctx.fitted)
        @test !isempty(terms)
        # the contract downstream prunes on: a term carried at all has a nonzero
        # coefficient, and the zeros are EXACT (`coef != 0.0`, not a tolerance)
        @test all(t -> t.coef != 0.0, terms)
        for i = 1:4
            ee = ctx.spinful ? ctx.cfgs[i] : zeros(3, ctx.nat)
            u = ctx.disps[i]
            ref = predict_energy(ctx.fitted, ctx.spinful ? ee : nothing, u) -
                  ctx.fitted.j0
            got = _energy_from_terms(terms, ee, u)
            @test abs(got - ref) < 1e-9 * max(abs(ref), 1.0)
        end
        # every slot's site index addresses the term's own atom list, and every
        # term's atoms exist in the crystal
        for t in terms
            @test all(1 .<= t.atoms .<= ctx.nat)
            @test all(1 <= sl.site <= length(t.atoms) for sl in t.slots)
            @test ndims(t.folded) == length(t.slots)
        end
        mark(row, :terms)
    end
end

# ---------------------------------------------------------------------------
# magnetoelastic
# ---------------------------------------------------------------------------

# The ε-linear tier, in both of the states a real crystal puts it in.
#
# On B2 FeRh in its CONVENTIONAL cell the tier is absent — every degree-1
# spin-dressed column is frozen by the boundary tie — and the deliverable must
# say so rather than return the zeros it never measured. In a 2×2×2 tiling of the
# same crystal the same ten columns are alive and the constants are real numbers.
# One crystal, two cells, both faces of the same guard.
function check_magnetoelastic(row::Row, ctx)
    @testset "magnetoelastic" begin
        if !ctx.spinful
            # without spin content the strain response is the same for every
            # magnetic state, and the deliverable refuses rather than reporting
            # B₁ = B₂ = 0 as if it had measured them
            @test_throws ArgumentError magnetoelastic_constants(ctx.fitted)
            mark(row, :magnetoelastic)
            return
        end
        # face 1: the tier is not there. `residual` is NaN and `signal` is zero —
        # the two fields that keep "no ε-linear content in this truncation" from
        # reading as "a perfect fit to zero coupling".
        absent = magnetoelastic_constants(ctx.fitted)
        @test absent.ion === :clamped              # the qualifier rides in the value
        @test absent.volume ≈ abs(det(Matrix(row.crystal.lattice.vectors)))
        @test isnan(absent.residual)
        @test absent.signal == 0.0
        @test absent.B1 == 0.0 && absent.B2 == 0.0

        # face 2: the same crystal in a cell that resolves the tier
        big = tile(row.crystal, (2, 2, 2))
        bb = SLCEBasis(big, ferh_spec(big, 2.7); backend = SpglibBackend())
        rep = build_asr(bb)
        beta = rep.Z * randn(MersenneTwister(661), size(rep.Z, 2))
        model = SLCEModel(bb, 0.0, beta)
        me = magnetoelastic_constants(model)
        @test isfinite(me.B1) && isfinite(me.B2)
        @test me.signal > 0.0
        @test me.residual < 1e-10                  # the fit of the ε-linear form
        @test me.ion === :clamped
        # The pinned convention, against a different code path:
        #   E_me/V = B₁ Σ_i ε_ii (α_i² − 1/3) + 2 B₂ Σ_{i<j} ε_ij α_i α_j
        # so the DIFFERENCE of the ε-linear response between two magnetic states
        # is the magnetoelastic part, and `strain_derivatives` computes it by
        # contracting the monomial coefficients rather than by fitting the form.
        V = me.volume
        nb = n_atoms(big)
        α1 = [0.0, 0.0, 1.0]
        α2 = [1.0, 1.0, 0.0] / sqrt(2)
        s1 = strain_derivatives(model; spins = repeat(α1, 1, nb), order = 1)
        s2 = strain_derivatives(model; spins = repeat(α2, 1, nb), order = 1)
        pred(i, j) = i == j ? V * me.B1 * ((α1[i]^2 - 1 / 3) - (α2[i]^2 - 1 / 3)) :
                     2 * V * me.B2 * (α1[i] * α1[j] - α2[i] * α2[j]) / 2
        scale = max(maximum(abs, s1 .- s2), 1.0)
        for (i, j) in ((1, 1), (3, 3), (1, 2), (2, 3))
            @test abs((s1[i, j] - s2[i, j]) - pred(i, j)) < 1e-8 * scale
        end
        # the bond-resolved form of the same content
        xs = exchange_strain_derivatives(model)
        @test !isempty(xs.pairs) || !isempty(xs.onsites)
        mark(row, :magnetoelastic)
    end
end

# ---------------------------------------------------------------------------
# strain
# ---------------------------------------------------------------------------

function check_strain(row::Row, ctx)
    @testset "strain" begin
        model = ctx.fitted
        s = ctx.spins
        # ∂E/∂ε by central differences of `affine_energy`, which reaches the
        # affine field through the periodic evaluator — a different code path from
        # the analytic monomial contraction `strain_derivatives` performs.
        h = 1e-6
        an = strain_derivatives(model; spins = s, order = 1, symmetrize = false)
        fd = zeros(3, 3)
        for i = 1:3, j = 1:3
            E = zeros(3, 3)
            E[i, j] = h
            fd[i, j] = (affine_energy(model, s, E) - affine_energy(model, s, -E)) / (2h)
        end
        scale = max(maximum(abs, fd), 1.0)
        @test maximum(abs, an .- fd) < 1e-6 * scale
        # order 1 is the reference stress times the volume, so the symmetrized
        # form is the symmetric part of the same object
        sym = strain_derivatives(model; spins = s, order = 1, symmetrize = true)
        @test maximum(abs, sym .- (an .+ an') ./ 2) < 1e-10 * scale
        @test maximum(abs, sym .- sym') == 0.0
        mark(row, :strain)
    end
end

# ---------------------------------------------------------------------------
# model selection, on a real basis
# ---------------------------------------------------------------------------

# The selection subsystem — grouped cross-validation, the support path, the
# closed-form hat-matrix diagnostics — reached through a real crystal's fold
# structure rather than a hand-built design. The property gate: the target is
# exactly representable, so every fold's held-out error must be at machine
# precision. A fold that mis-slices the ragged torque or force block breaks that
# immediately, and nothing about it is visible in a single full-data fit.
function check_selection(row::Row, ctx)
    @testset "selection" begin
        cv = cross_validate(ctx.dataset, OLS(); torque_weight = ctx.wT,
                            force_weight = 0.25, nfolds = 4, seed = 3)
        @test cv.nfolds == 4
        @test all(isfinite, cv.score)
        @test cv.pooled_rmse_energy < 1e-8            # in-span in every fold
        @test cv.pooled_rmse_force < 1e-8
        ctx.spinful && @test cv.pooled_rmse_torque < 1e-8
        # the closed-form diagnostics agree with the free-parameter count: an OLS
        # fit spends one degree of freedom per free coefficient, plus the intercept
        @test effective_dof(ctx.fit) ≈ ctx.q + 1
        @test isfinite(gcv(ctx.fit)) && gcv(ctx.fit) >= 0
        # the support path: with an exactly representable target the sparsest
        # support within tolerance still reproduces it
        sp = select_support(ctx.fit; npoints = 6)
        @test 1 <= sp.selected <= length(sp.threshold)
        @test issorted(sp.threshold; rev = true)
        @test minimum(sp.rmse_energy) < 1e-8
        mark(row, :selection)
    end
end

# ---------------------------------------------------------------------------
# the same-atom pair a minimum-image convention cannot express
# ---------------------------------------------------------------------------

# DOCUMENTED LIMITATION, pinned as a change detector — not an endorsement.
# `MinimumImage` enumerates one image per (atom, atom) pair, so a pair joining an
# atom to a periodic image of ITSELF — the (a, 0)–(a, R) pair — is never built.
# In B2 FeRh the 2NN shell is exactly Fe–Fe and Rh–Rh, so widening the cutoff to
# admit it adds nothing; in rocksalt MnO the whole magnetic problem is Mn–Mn, so
# a superexchange spec has NO columns at all, which the dataset boundary must
# refuse rather than fit as an intercept.
#
# WHEN THIS IS FIXED both branches below change: the FeRh basis grows, and MnO
# stops being empty. Update the row, do not delete it.
function check_selfpair(row::Row, ctx)
    @testset "selfpair" begin
        cr = row.crystal
        if row.id === :B2_FeRh
            narrow = SLCEBasis(cr, ferh_spec(cr, 2.7); backend = SpglibBackend())
            wide = SLCEBasis(cr, ferh_spec(cr, 3.1); backend = SpglibBackend())
            # the 2NN shell at 2.99 Å is inside 3.1 and yet contributes nothing
            @test n_salcs(wide) == n_salcs(narrow)
            @test wide.salc_basis.keys == narrow.salc_basis.keys
            # no member anywhere joins an atom to an image of itself
            @test all(s -> all(m -> allunique(m.atoms), s.members), salcs(wide))
        else    # :rs_MnO — the PRIMITIVE cell of the row's crystal
            prim = CR_MNO_PRIM
            spec = BasisSpec(prim; lmax = ["*" => 2, "O" => 0], pmax = 0,
                             sectors = [Sector(spin = (sites = 1:2, lmax = 2),
                                               cutoff = 3.2)])
            empty = SLCEBasis(prim, spec; backend = SpglibBackend())
            @test n_salcs(empty) == 0        # the Mn–Mn superexchange problem
            # and the boundary refuses it: a zero-column basis can only ever fit
            # the mean energy, which is not a model of anything
            rng = MersenneTwister(4)
            data = [spin_datum(0.1 * randn(rng), 2.5 .* randcfg(rng, 2)) for _ = 1:8]
            @test_throws ArgumentError SLCEDataset(empty, data; use_torque = false)
            # the remedy, on the same crystal: in the conventional cell the two
            # cations are distinct atoms, so the identical spec has content
            conv = SLCEBasis(cr, BasisSpec(cr; lmax = ["*" => 2, "O" => 0], pmax = 0,
                                           sectors = [Sector(spin = (sites = 1:2,
                                                                     lmax = 2),
                                                             cutoff = 3.2)]);
                             backend = SpglibBackend())
            @test n_salcs(conv) > 0
        end
        mark(row, :selfpair)
    end
end

# ---------------------------------------------------------------------------
# dispatch
# ---------------------------------------------------------------------------

const CHECKS = Dict(
    :census => check_census, :structure => check_structure,
    :resolvability => check_resolvability, :invariance => check_invariance,
    :affine => check_affine, :asr => check_asr, :recovery => check_recovery,
    :phonons => check_phonons, :fd_hessian => check_fd_hessian,
    :effective => check_effective, :restrict => check_restrict,
    :persist => check_persist, :terms => check_terms,
    :magnetoelastic => check_magnetoelastic, :strain => check_strain,
    :selection => check_selection, :selfpair => check_selfpair)
