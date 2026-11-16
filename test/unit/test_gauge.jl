# The gauge inside a block (2026-11-16): the columns of a block are chosen along axes fixed
# to the crystal (ported from SphericalTensorFC `c7b9957`), and the subspace of a block
# that vanishes on the reference cell is made columns of its own and frozen.
#
# Every oracle here is outside the construction: rotated columns are compared through the
# EVALUATOR on rotated configurations; the vanishing subspace is the null space of a
# design SAMPLED through `predict_energy`, never the symbolic expansion the build uses;
# fitted models are compared through force constants mapped by Cartesian position; and
# the leak and the zero-cost drop are measured on the solution itself. The groups are
# found by brute force (no Spglib in the core suite).
#
# Mutations run by hand on 2026-11-16 (each restored afterwards):
#   - re-gauge skipped (`_regauge_vanishing` returns its input): the frozen-count gate,
#     the AdaptiveRidge / metric-Ridge leak gates, the zero-cost-drop gate and the
#     force-constant covariance gate fail on hcp (2,2,2) at degree 3 (and on the
#     rhombohedral cell); the numbers are in the CHANGELOG entry.
#   - crystal frame reverted to the Cartesian axes (`_build_frame_cache` fed I): the
#     column-rotation gate fails (it is also asserted below as an in-suite mutation).
#   - block weight of `penalty_metric` reverted to per column: the metric-Ridge leak gate
#     fails on the turned pure-spin hcp cell.

using SLCE: _assemble_spacegroup, build_asr, unresolvable_columns, regauged_columns,
            salcs, evaluate_salc, symmetrized_lattice
using Base.CoreLogging: with_logger, NullLogger
using LinearAlgebra
using Random
using StaticArrays

_gauge_rot(axis, θ) = (k = normalize(axis); K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0];
                       Matrix(I + sin(θ) * K + (1 - cos(θ)) * K * K))

# A supercell `M` (columns: its vectors in units of the primitive ones) of the primitive
# lattice `A0` with fractional basis `F`, turned by `R`.
function _gauge_supercell(A0, F, M, R)
    Minv = inv(Float64.(M))
    pos = Vector{Vector{Float64}}()
    for i = -4:4, j = -4:4, k = -4:4, s = 1:size(F, 2)
        f = Minv * (F[:, s] .+ [i, j, k])
        f = f .- floor.(f .+ 1e-9)
        any(p -> norm(p .- f) < 1e-6, pos) || push!(pos, f)
    end
    frac = reduce(hcat, pos)
    return Crystal(Lattice(R * A0 * M), frac, fill(1, size(frac, 2)), ["X"]), frac
end

# The space group by brute force: the integer matrices with entries in -1:1 that keep the
# PRIMITIVE metric, conjugated into the cell, each with every translation that maps the
# atom set onto itself (that covers the nonsymmorphic hcp operations).
struct _GaugeOps <: AbstractSymmetryBackend
    rots::Vector{SMatrix{3,3,Float64,9}}
    trans::Vector{SVector{3,Float64}}
end
SLCE.analyze_symmetry(b::_GaugeOps, cr::Crystal; tol::Real = 1e-5) =
    _assemble_spacegroup(cr, b.rots, b.trans, "brute force", 0; tol = tol)

function _gauge_ops(A0, M, frac)
    G0 = A0' * A0
    Mf = Float64.(M)
    on_atom(x) = any(p -> maximum(abs.(mod.(x .- p .+ 0.5, 1.0) .- 0.5)) < 1e-8,
                     eachcol(frac))
    rots, trans = SMatrix{3,3,Float64,9}[], SVector{3,Float64}[]
    for t in Iterators.product(ntuple(_ -> -1:1, 9)...)
        W0 = reshape(collect(Float64, t), 3, 3)
        abs(abs(det(W0)) - 1) < 1e-9 || continue
        maximum(abs, W0' * G0 * W0 - G0) < 1e-9 * maximum(abs, G0) || continue
        W = round.(Mf \ (W0 * Mf))
        for b in axes(frac, 2)
            tr = mod.(frac[:, b] .- W * frac[:, 1], 1.0)
            all(q -> on_atom(W * q .+ tr), eachcol(frac)) || continue
            push!(rots, SMatrix{3,3,Float64}(W))
            push!(trans, SVector{3,Float64}(tr))
        end
    end
    return _GaugeOps(rots, trans)
end

const _HCP_A0 = [2.5 -1.25 0; 0 2.5 * sqrt(3) / 2 0; 0 0 4.08]
const _HCP_F = [1/3 2/3; 2/3 1/3; 1/4 3/4]
function _rh_a0(α = deg2rad(80.0), a = 3.0)
    c = cos(α); s = sin(α); cy = (c - c^2) / s
    return a * [1 c c; 0 s cy; 0 0 sqrt(1 - c^2 - cy^2)]
end
const _FCC_A0 = 1.8 * [0 1 1; 1 0 1; 1 1 0]
const _ORIGIN = reshape([0.0, 0, 0], 3, 1)
const _M222 = [2 0 0; 0 2 0; 0 0 2]
const _M_TILTED = [2 0 2; 0 2 0; 0 0 2]          # lattice vectors (a1, a2, a1 + a3)

_quiet(f) = with_logger(f, NullLogger())

# (crystal, basis) of one description of a crystal.
function _gauge_case(A0, F, M, R, spec_of)
    cr, frac = _gauge_supercell(A0, F, M, R)
    b = _quiet(() -> SLCEBasis(cr, spec_of(cr); backend = _gauge_ops(A0, M, frac),
                               tol = 1e-6))
    return cr, b
end
_disp_spec(degrees, cut) = cr -> BasisSpec(cr; lmax = 0, pmax = maximum(degrees),
    sectors = [Sector(disp = (degree = degrees,), sites = 1:2, cutoff = cut)])

# The cells on which the gauge was measured before the change (2026-11-15): each
# froze a different set of columns along the axes, turned, and with other lattice
# vectors.
const _GAUGE_CASES = (
    ("hcp (2,2,2) degree 2", _HCP_A0, _HCP_F, _disp_spec(2:2, 4.2)),
    ("hcp (2,2,2) degree 3", _HCP_A0, _HCP_F, _disp_spec(3:3, 2.6)),
    ("hcp (2,2,2) degree 4", _HCP_A0, _HCP_F, _disp_spec(4:4, 2.6)),
    ("rhombohedral (2,2,2) degree 2", _rh_a0(), _ORIGIN, _disp_spec(2:2, 6.5)),
    ("fcc primitive (2,2,2) degree 2", _FCC_A0, _ORIGIN, _disp_spec(2:2, 6.0)))

# Cartesian site of atom `a` in cell `R` of `cr`, and the inverse map.
_gauge_cart(cr, a, R) = cr.lattice.vectors * (cr.frac_positions[:, a] .+ R)
function _gauge_locate(cr, x)
    A = cr.lattice.vectors
    f = A \ x
    for a in axes(cr.frac_positions, 2)
        d = f .- cr.frac_positions[:, a]
        R = round.(Int, d)
        norm(A * (d .- R)) < 1e-6 && return a, SVector{3,Int}(R)
    end
    error("no atom at $x")
end
# perm[i]: the atom of `cb` sitting where atom i of `ca` sits after turning by R.
_gauge_atom_map(ca, cb, R) = [_gauge_locate(cb, R * _gauge_cart(ca, a, [0, 0, 0]))[1]
                              for a in axes(ca.frac_positions, 2)]
_gauge_turn(T, R) = (out = T; for ax = 1:ndims(T)
                         out = mapslices(v -> R * v, out; dims = ax)
                     end; out)

# max |R·Φ_a − Φ_b| / max |Φ_a| over every index tuple, mapped by Cartesian position.
function _gauge_fc_distance(fa, ca, fb, cb, R)
    num = 0.0; den = 0.0
    seen = Set{Any}()
    for ((atoms, shifts), Φ) in fa.constants
        loc = [_gauge_locate(cb, R * _gauge_cart(ca, atoms[i], shifts[i]))
               for i in eachindex(atoms)]
        key = ([l[1] for l in loc], [l[2] - loc[1][2] for l in loc])
        push!(seen, key)
        num = max(num, maximum(abs, _gauge_turn(Φ, R) .- get(fb.constants, key, zero(Φ))))
        den = max(den, maximum(abs, Φ))
    end
    for (key, Φ) in fb.constants
        key in seen || (num = max(num, maximum(abs, Φ)))
    end
    return num / max(den, eps())
end

# The design of unit coefficients on random fields: the sampled evaluator.
function _gauge_design(b, us)
    p = n_salcs(b)
    X = zeros(length(us), p)
    for j = 1:p
        e = zeros(p); e[j] = 1.0
        m = SLCEModel(b, 0.0, e)
        for c in eachindex(us)
            X[c, j] = predict_energy(m, nothing, us[c])
        end
    end
    return X
end
function _gauge_null(X; rtol = 1e-8)
    F = svd(X; full = true)
    return F.V[:, (count(>(rtol * F.S[1]), F.S) + 1):end]
end
_gauge_blocks(b) = (g = Dict{Any,Vector{Int}}();
                    for (j, s) in enumerate(salcs(b))
                        push!(get!(g, SLCE._block_key(s), Int[]), j)
                    end; collect(values(g)))

@testset "the columns of a block turn with the crystal" begin
    # Turning the crystal by any orthogonal matrix — proper or not — must turn every
    # column: Φⱼ of the turned basis at the turned configuration equals Φⱼ of the
    # original at the original one. Measured through the evaluator, so the oracle shares
    # nothing with the gauge. A spin enters through even total rank, so turning it as a
    # polar vector is the same as turning it as an axial one.
    rng = MersenneTwister(301)
    joint = cr -> BasisSpec(cr; lmax = 2, pmax = 2,
        sectors = [Sector(spin = (sites = 1:2,), cutoff = 2.6),
                   Sector(spin = [1, 1], disp = (degree = 1:2,), sites = 2, cutoff = 2.6,
                          soc = true),
                   Sector(disp = (degree = 1:2,), sites = 1:2, cutoff = 2.6)])
    cases = (("fcc primitive joint soc", _FCC_A0, _ORIGIN, _M222, joint),
             ("hcp (2,2,2) degree 3", _HCP_A0, _HCP_F, _M222, _disp_spec(3:3, 2.6)),
             ("rhombohedral (2,2,2) degree 2", _rh_a0(), _ORIGIN, _M222,
              _disp_spec(2:2, 6.5)))
    turns = (_gauge_rot([1, 2, 3], 0.3), _gauge_rot([0.3, -0.7, 0.5], 2.1),
             -_gauge_rot([1, -1, 2], 0.8))                  # the last one is improper
    for (name, A0, F, M, spec_of) in cases
        cr0, b0 = _gauge_case(A0, F, M, I(3), spec_of)
        nat = n_atoms(cr0)
        es = [(E = randn(rng, 3, nat); E ./ sqrt.(sum(abs2, E; dims = 1))) for _ = 1:6]
        us = [0.05 .* randn(rng, 3, nat) for _ = 1:6]
        frozen0 = _quiet(() -> unresolvable_columns(b0))
        axes0 = symmetrized_lattice(b0).crystal_axes
        @test axes0' * axes0 ≈ I atol = 1e-14
        for R in turns
            cr, b = _gauge_case(A0, F, M, R, spec_of)
            # the axes turn with the crystal (an improper R gives improper axes)
            @test maximum(abs, symmetrized_lattice(b).crystal_axes - R * axes0) <= 1e-13
            @test b.salc_basis.keys == b0.salc_basis.keys
            @test _quiet(() -> unresolvable_columns(b)) == frozen0
            worst = 0.0
            scale = 0.0
            for (j, s) in enumerate(salcs(b)), c in eachindex(es)
                v0 = evaluate_salc(salcs(b0)[j], es[c], us[c])
                v = evaluate_salc(s, R * es[c], R * us[c])
                worst = max(worst, abs(v - v0))
                scale = max(scale, abs(v0))
            end
            @test worst <= 1e-10 * scale
        end
    end
    # In-suite mutation: the Cartesian axes in place of the crystal frame. Built through
    # the orbit engine with the frame table fed the identity, the turned crystal's
    # columns are NOT the turned columns — the gate above has something to catch.
    # A block of several columns is needed for the gauge to matter: the hcp pair block
    # of displacement degree 3.
    spec3 = _disp_spec(3:3, 2.6)
    cr0, b0 = _gauge_case(_HCP_A0, _HCP_F, _M222, I(3), spec3)
    R = turns[1]
    cr, b = _gauge_case(_HCP_A0, _HCP_F, _M222, R, spec3)
    wide = first(cols for cols in _gauge_blocks(b0) if length(cols) >= 2)
    key = salcs(b0)[first(wide)].key
    label = [key.decors]
    raw(c, s, frame) = (cs = SLCE.build_clusters(c, SLCE.build_neighbor_list(c, 2.6), s;
                                                 nbody = 2);
                        SLCE._orbit_salcs_decors(c, s, 2, key.orbit_id,
                                                 cs.by_body[2][key.orbit_id], label, true,
                                                 SLCE._build_wig_cache(s, 3), frame))
    aligned = raw(cr0, b0.spacegroup, SLCE._build_frame_cache(cr0, b0.spacegroup, 3))
    proper = raw(cr, b.spacegroup, SLCE._build_frame_cache(cr, b.spacegroup, 3))
    mutant = raw(cr, b.spacegroup, SLCE._build_frame_cache(Matrix(1.0I, 3, 3), 3))
    @test length(aligned) == length(proper) == length(mutant) >= 2
    nat = n_atoms(cr0)
    e = (E = randn(rng, 3, nat); E ./ sqrt.(sum(abs2, E; dims = 1)))
    u = 0.05 .* randn(rng, 3, nat)
    gap(set) = maximum(abs(evaluate_salc(set[j], R * e, R * u) -
                           evaluate_salc(aligned[j], e, u)) for j in eachindex(set))
    @test gap(proper) <= 1e-10
    @test gap(mutant) > 1e-3
end

@testset "spin slots of an L_S = 0 path do not move the gauge" begin
    # The frame turns spin slots with the same matrices as displacement slots. On a
    # `soc = false` path the spin slots couple to a scalar, so turning them alone must
    # leave every coupled tensor unchanged: the spin index handling cannot change a
    # `soc = false` column. Checked on the tensors themselves, for a generic frame.
    Q = _gauge_rot([0.2, 0.9, -0.4], 1.3)
    frame = SLCE._build_frame_cache(Q, 3)
    labels = ([SLCE.SiteDecor(spin = 1), SLCE.SiteDecor(spin = 1, disp = (0, 1))],
              [SLCE.SiteDecor(spin = 2, disp = (0, 2)), SLCE.SiteDecor(spin = 2)],
              [SLCE.SiteDecor(spin = 1), SLCE.SiteDecor(spin = 1),
               SLCE.SiteDecor(disp = (1, 1))])
    checked = 0
    for label in labels
        slots = SLCE._assignment_slots(label)
        for (L_S, Lf, T) in SLCE._decor_coupled_bases(slots, false)
            @test L_S == 0
            for Mf = 1:(2Lf + 1)
                v = SLCE._mf_slice(T, Mf)
                w = v
                for j in eachindex(slots)
                    slots[j].factor.channel == SLCE.SPIN || continue
                    w = SLCE.AngularMomentum.nmode_mul(w, SLCE._wigner_d(frame,
                                                       slots[j].factor.l, 1), j)
                end
                @test maximum(abs, w .- v) <= 1e-13
                checked += 1
            end
        end
    end
    @test checked > 0
    # And the pure-spin `soc = false` columns (Lf = 0) do not see the frame at all:
    # built along the Cartesian axes or along a generic frame, they are the same.
    cr, b = _gauge_case(_FCC_A0, _ORIGIN, _M222, _gauge_rot([1, 2, 3], 0.3),
                        cr -> BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [2], soc = false))
    sg = b.spacegroup
    cs = SLCE.build_clusters(cr, SLCE.build_neighbor_list(cr, 2.6), sg; nbody = 2)
    wc = SLCE._build_wig_cache(sg, 2)
    for O in cs.by_body[2]
        a = SLCE._orbit_salcs(cr, sg, 2, 1, O, [2], SLCE.LSUM_UNCAPPED, true, wc,
                              SLCE._build_frame_cache(Matrix(1.0I, 3, 3), 2))
        c = SLCE._orbit_salcs(cr, sg, 2, 1, O, [2], SLCE.LSUM_UNCAPPED, true, wc,
                              SLCE._build_frame_cache(Q, 2))
        @test length(a) == length(c)
        @test all(maximum(abs, ta.folded .- tc.folded) <= 1e-14
                  for (sa, sc) in zip(a, c) for (ma, mc) in zip(sa.members, sc.members)
                  for (ta, tc) in zip(ma.terms, mc.terms))
    end
end

@testset "the vanishing subspace of a block is frozen" begin
    # V is the null space of a design SAMPLED through `predict_energy` on random fields.
    # Within each block its dimension must equal the number of frozen columns of the
    # block; the whole null space is the frozen columns plus whatever `residual_flat`
    # reports across blocks.
    for (name, A0, F, spec_of) in _GAUGE_CASES
        cr, b = _gauge_case(A0, F, _M222, I(3), spec_of)
        rng = MersenneTwister(17)
        us = [0.05 .* randn(rng, 3, n_atoms(cr)) for _ = 1:(4 * n_salcs(b))]
        X = _gauge_design(b, us)
        ref = maximum(norm(view(X, :, j)) for j in axes(X, 2))
        split = _quiet(() -> SLCE._unresolvable_split(b))
        frozen = Set(split.columns)
        flagged = Set(findall(coeftable(SLCEModel(b, 0.0, zeros(n_salcs(b)))).regauged))
        for cols in _gauge_blocks(b)
            s = svdvals(X[:, cols])
            nullity = length(cols) - count(>(1e-8 * ref), s)
            @test nullity == count(in(frozen), cols)
            # the table marks exactly the blocks of which a proper part vanishes
            @test all(in(flagged), cols) == (0 < nullity < length(cols))
            @test any(in(flagged), cols) == (0 < nullity < length(cols))
        end
        @test size(_gauge_null(X), 2) == length(split.columns) + split.residual_flat
        # the frozen columns ARE the zero function on this cell
        @test all(j -> norm(view(X, :, j)) <= 1e-10 * ref, split.columns)
    end
end

@testset "the rank decision of a block sits far from its cut" begin
    # The re-gauge decides the rank of each block at `_CANCELLATION_RTOL` (1e-10) of the
    # block's gross mass and refuses a singular value between that and
    # `_REGAUGE_BAND_TOP` (1e-6). Measured 2026-11-16 on these cells, along the axes and
    # turned: cancellations at most 3.1e-16, content at least 1.9e-2. The gate keeps
    # three decades of headroom on each side of the band.
    worst_zero = 0.0
    least_content = Inf
    for (name, A0, F, spec_of) in _GAUGE_CASES, R in (I(3), _gauge_rot([1, 2, 3], 0.3))
        cr, b = _gauge_case(A0, F, _M222, R, spec_of)
        for cols in _gauge_blocks(b)
            S, gross = SLCE._signature_of(salcs(b)[cols])
            for r in SLCE._block_split(S, gross).ratios
                r <= SLCE._CANCELLATION_RTOL ? (worst_zero = max(worst_zero, r)) :
                                              (least_content = min(least_content, r))
            end
        end
    end
    @test worst_zero < 1e-13
    @test least_content > 1e-3
end

# Data of one physical crystal for every description: displacements and forces turned by
# R and relabelled by the atom map, energies unchanged.
function _gauge_data(truth, cr0, cr, R, us)
    perm = _gauge_atom_map(cr0, cr, R)
    data = TrainingDatum[]
    for u in us
        E = predict_energy(truth, nothing, u)
        f = predict_force(truth, nothing, u)
        v = similar(u); g = similar(f)
        v[:, perm] = R * u
        g[:, perm] = R * f
        push!(data, lattice_datum(E; displacements = v, forces = g, reference = cr))
    end
    return data
end

@testset "one crystal, three descriptions, one fitted model" begin
    # The same crystal along the axes, turned, and with lattice vectors (a1, a2, a1+a3)
    # must give the same force constants (turned back) under every estimator and after
    # a refit, with the same column, frozen and constraint counts. Before the change the
    # turned hcp cell froze 1 column where the aligned one froze 3, and AdaptiveRidge
    # moved Φ by 1.3e-2 between them (`test_gauge.jl` header, CHANGELOG).
    turn = _gauge_rot([1, 2, 3], 0.3)
    for (name, A0, F, spec_of) in _GAUGE_CASES
        descriptions = ((_M222, Matrix(1.0I, 3, 3)), (_M222, turn), (_M_TILTED,
                                                                     Matrix(1.0I, 3, 3)))
        cr0, b0 = _gauge_case(A0, F, _M222, I(3), spec_of)
        degree = sum(SLCE.disp_degree, first(salcs(b0)).decors)
        rep0 = _quiet(() -> build_asr(b0))
        rng = MersenneTwister(23)
        truth = SLCEModel(b0, 0.0, rep0.Z * randn(rng, size(rep0.Z, 2)))
        us = [0.05 .* randn(rng, 3, n_atoms(cr0)) for _ = 1:(3 * n_salcs(b0))]
        fcs = Dict{Symbol,Vector{Any}}()
        counts = []
        for (M, R) in descriptions
            cr, b = _gauge_case(A0, F, M, R, spec_of)
            data = _gauge_data(truth, cr0, cr, R, us)
            ds = _quiet(() -> SLCEDataset(b, data))
            rep = ds.asr
            # the vanishing subspace from the sampled evaluator
            V = _gauge_null(_gauge_design(b, [d.displacements for d in data]))
            ests = (:OLS => OLS(), :Ridge => Ridge(lambda = 1e-6),
                    :AdaptiveRidge => AdaptiveRidge(lambda = 1e-6, tol = 1e-13,
                                                    max_iter = 500))
            for (tag, est) in ests
                f = _quiet(() -> fit(SLCEFit, ds, est; force_weight = 0.5))
                @test norm(V' * f.jphi) <= 1e-12 * norm(f.jphi)
                push!(get!(fcs, tag, Any[]),
                      (cr, R, force_constants(SLCEModel(f); order = degree)))
                if tag === :AdaptiveRidge
                    g = _quiet(() -> refit(f))
                    @test norm(V' * g.jphi) <= 1e-12 * norm(g.jphi)
                    push!(get!(fcs, :refit, Any[]),
                          (cr, R, force_constants(SLCEModel(g); order = degree)))
                    push!(counts, (n_salcs(b), length(unresolvable_columns(b)), rep.rank,
                                   size(rep.Z, 2), count(!iszero, g.jphi)))
                end
            end
            # no column can be dropped at zero in-cell cost: every free column carries
            # content the cell sees
            X = _gauge_design(b, [d.displacements for d in data])
            y = [d.energy for d in data]          # the truth lies in this span exactly
            Z = rep.Z
            free_drop = 0
            for j in axes(Z, 1)
                norm(view(Z, j, :)) < 1e-12 && continue
                Zs = Z * nullspace(Z[j:j, :]; atol = 1e-12)
                βs = Zs * (pinv(X * Zs) * y)
                norm(X * βs - y) <= 1e-8 * norm(y) && (free_drop += 1)
            end
            @test free_drop == 0
        end
        @test allequal(counts)
        for (tag, list) in fcs
            (ca, Ra, fa) = list[1]
            for (cb, Rb, fb) in list[2:end]
                @test _gauge_fc_distance(fa, ca, fb, cb, Rb) <= 1e-10
            end
        end
    end
end

# The energy each cluster instance contributes: Σⱼ βⱼ·(member of column j) at `e`. This
# is what a supercell sees (a tile carries the members, not the orbit sums), it does
# not depend on the gauge, and it turns with the crystal. Keyed by the instance in the
# indexing of `home` (turned back by `R`), so descriptions can be compared.
function _gauge_member_energies(model, e, cr, home, R)
    out = Dict{Any,Float64}()
    for (j, s) in enumerate(salcs(model.basis))
        model.jphi[j] == 0.0 && continue
        for m in s.members
            one = SLCE.SALC(s.key, s.body, s.decors, s.L_S, s.Lf, [m])
            loc = sort([_gauge_locate(home, R' * _gauge_cart(cr, m.atoms[i], m.shifts[i]))
                        for i in eachindex(m.atoms)]; by = l -> (l[1], Tuple(l[2])))
            key = ([l[1] for l in loc], [l[2] - loc[1][2] for l in loc])
            out[key] = get(out, key, 0.0) + model.jphi[j] * evaluate_salc(one, e)
        end
    end
    return out
end

@testset "pure spin: the penalty metric weighs a block, not its columns" begin
    # hcp (2,2,2), spin l ≤ 2 with SOC, along the axes, turned, and with lattice vectors
    # (a1, a2, a1+a3). Before the change the cell froze 12 columns along the axes and 5
    # when turned (7 vanishing combinations left free), the metric Ridge was refused
    # along the axes ("zero reference norm") and leaked 8.8 % of β into the vanishing
    # subspace when turned. A weight per COLUMN is also what made a metric fit depend on
    # the lattice vectors once the gauge follows them; one weight per block does not.
    spec_of = cr -> BasisSpec(cr; lmax = 2, pmax = 0, soc = true,
                              sectors = [Sector(spin = (sites = 1:2,), cutoff = 4.2)])
    turn = _gauge_rot([0.3, -0.7, 0.5], 0.9)
    cr0, b0 = _gauge_case(_HCP_A0, _HCP_F, _M222, I(3), spec_of)
    rep0 = _quiet(() -> build_asr(b0))
    rng = MersenneTwister(41)
    nat = n_atoms(cr0)
    truth = SLCEModel(b0, 0.0, rep0.Z * randn(rng, size(rep0.Z, 2)))
    unit_cfg() = (E = randn(rng, 3, nat); E ./ sqrt.(sum(abs2, E; dims = 1)))
    configs = [unit_cfg() for _ = 1:(3 * n_salcs(b0))]
    energies = [predict_energy(truth, e) for e in configs]
    probe = unit_cfg()
    readouts = Dict{Symbol,Vector{Any}}()
    for (M, R) in ((_M222, Matrix(1.0I, 3, 3)), (_M222, turn), (_M_TILTED, Matrix(1.0I, 3, 3)))
        cr, b = _gauge_case(_HCP_A0, _HCP_F, M, R, spec_of)
        perm = _gauge_atom_map(cr0, cr, R)
        place(e) = (out = similar(e); out[:, perm] = R * e; out)
        frozen = _quiet(() -> unresolvable_columns(b))
        @test length(frozen) == length(_quiet(() -> unresolvable_columns(b0)))
        m = penalty_metric(b)
        @test all(iszero, m[frozen])
        @test all(>(0), m[setdiff(1:n_salcs(b), frozen)])
        for cols in _gauge_blocks(b)
            @test allequal(m[setdiff(cols, frozen)])
        end
        placed = [place(e) for e in configs]
        ds = _quiet(() -> SLCEDataset(b, placed, energies))
        X = zeros(length(placed), n_salcs(b))
        for j = 1:n_salcs(b)
            unit = SLCEModel(b, 0.0, Float64.(1:n_salcs(b) .== j))
            X[:, j] = [predict_energy(unit, e) for e in placed]
        end
        V = _gauge_null(X)
        @test size(V, 2) == length(frozen)
        for (tag, est) in (:OLS => OLS(), :metric_Ridge => Ridge(b; lambda = 1e-3),
                           :AdaptiveRidge => AdaptiveRidge(b; lambda = 1e-3, tol = 1e-13,
                                                           max_iter = 500))
            f = _quiet(() -> fit(SLCEFit, ds, est))
            @test norm(V' * f.jphi) <= 1e-12 * norm(f.jphi)
            @test isfinite(effective_dof(f)) && isfinite(gcv(f))
            @test dof(f) == n_salcs(b) - length(frozen) + 1
            push!(get!(readouts, tag, Any[]),
                  _gauge_member_energies(SLCEModel(f), place(probe), cr, cr0, R))
        end
        # the λ path solves in the free coordinates too: a frozen column's weight 0
        # must not reach the solver as an unpenalized column (it used to throw here)
        for criterion in (:gcv, :cv)
            path = _quiet(() -> select_fit(ds, GroupAdaptiveRidge(b; lambda = 1e-3);
                                           lambdas = [1e-4, 1e-3, 1e-2],
                                           criterion = criterion))
            @test all(iszero, path.fit.jphi[frozen])
            @test all(isfinite, path.score)
            @test norm(V' * path.fit.jphi) <= 1e-12 * norm(path.fit.jphi)
        end
    end
    for (tag, list) in readouts
        a = list[1]
        scale = maximum(abs, values(a))
        for t in list[2:end]
            @test keys(a) == keys(t)
            gap = maximum(abs(a[k] - t[k]) for k in keys(a))
            gap <= 1e-10 * scale || @info "member energies differ" tag gap scale
            @test gap <= 1e-10 * scale
        end
    end
    # `dof` leaves out exactly the directions of the reparameterization that vanish on
    # the cell: offered the whole column space, it finds the frozen columns.
    @test SLCE._flat_in_span(b0, Matrix(1.0I, n_salcs(b0), n_salcs(b0))) ==
          length(_quiet(() -> unresolvable_columns(b0)))
end

@testset "a block without a vanishing part keeps its columns" begin
    # The re-gauge touches a block only when part of it vanishes. On a cell without a
    # tie nothing is even expanded, and on a tied cell every other block is the
    # projection's own output, bit for bit.
    for (name, A0, F, spec_of) in _GAUGE_CASES
        cr, b = _gauge_case(A0, F, _M222, I(3), spec_of)
        sg = b.spacegroup
        spec = b.spec
        nl = SLCE.build_neighbor_list(cr, SLCE._superset_cutoff(spec), MinimumImage())
        cs = SLCE.build_clusters(cr, nl, sg; nbody = spec.nbody, cutoff = spec.cutoff)
        maxl = max(maximum(spec.lmax; init = 0),
                   maximum(r.disp_degree[2] for r in spec.sector_rules; init = 0))
        wc = SLCE._build_wig_cache(sg, maxl)
        fc = SLCE._build_frame_cache(cr, sg, maxl)
        raw = SLCE.SALC[]
        cart = SLCE.cartesian_positions(cr)
        dmin2 = SLCE._dmin2_matrix(nl, n_atoms(cr))
        # `_orbit_salcs_sectors` without its last step (the re-gauge)
        for N in sort(collect(keys(cs.by_body))), (oid, O) in enumerate(cs.by_body[N])
            gates = SLCE._orbit_edge_gates(cr, O, cart, dmin2, true)
            soc_by_label = Dict{Vector{SLCE.SiteDecor},Bool}()
            for rule in spec.sector_rules
                SLCE._sector_admits(rule, N, gates, nl.tol) || continue
                for label in SLCE._sector_orbit_labels(rule, N, O.species, spec.lmax,
                                                       spec.pmax, spec.lsum[N])
                    soc_by_label[label] = get(soc_by_label, label, false) | rule.soc
                end
            end
            for soc in (true, false)
                labels = sort!([l for (l, v) in soc_by_label if v == soc])
                isempty(labels) && continue
                append!(raw, SLCE._orbit_salcs_decors(cr, sg, N, oid, O, labels, soc,
                                                      wc, fc; lmax_by_species = spec.lmax,
                                                      pmax_by_species = spec.pmax))
            end
        end
        sort!(raw; by = s -> s.key)
        @test [s.key for s in raw] == b.salc_basis.keys
        touched = Set(regauged_columns(b))
        for (j, s) in enumerate(salcs(b))
            j in touched && continue
            same = length(s.members) == length(raw[j].members) &&
                   all(m.atoms == r.atoms && m.shifts == r.shifts &&
                       all(t.slots == q.slots && t.folded == q.folded
                           for (t, q) in zip(m.terms, r.terms))
                       for (m, r) in zip(s.members, raw[j].members))
            @test same
        end
        # a re-gauged block holds its whole vanishing subspace in its frozen columns
        frozen = Set(unresolvable_columns(b))
        @test !isempty(touched) || isempty(frozen) || all(
            cols -> count(in(frozen), cols) in (0, length(cols)), _gauge_blocks(b))
    end
    # an untied cell: the build never expands a block (the step is the identity)
    cr, b = _gauge_case(_FCC_A0, _ORIGIN, [3 0 0; 0 3 0; 0 0 3], I(3), _disp_spec(2:3, 2.6))
    @test isempty(regauged_columns(b))
    @test isempty(unresolvable_columns(b))
    for cols in _gauge_blocks(b)
        @test SLCE._regauge_vanishing(salcs(b)[cols], SLCE._energy_block_signature,
                                      SLCE._no_class) == salcs(b)[cols]
    end
end

@testset "pointed basis: the vanishing subspace of a block is frozen" begin
    # hcp (2,2,2), pair stars, mark and environment l ≤ 1, SOC on. Each block (one mark
    # class) must vanish on the sampled pointed design exactly as far as
    # `moment_resolvability` freezes it, and every other column must be seen by the
    # data. Without the re-gauge (mutation, 2026-11-16) 4 of the 5 vanishing directions
    # were columns and one block failed this gate.
    cr, frac = _gauge_supercell(_HCP_A0, _HCP_F, _M222, I(3))
    spec = MomentSpec(; lmax_env = [1], sampled = [true], lmax_mark = 1, nbody = 2,
                      cutoff_pair = 4.2, soc = true)
    mb = _quiet(() -> MomentBasis(cr, spec; backend = _gauge_ops(_HCP_A0, _M222, frac)))
    vanishing = Set(moment_resolvability(mb).vanishing)
    @test length(vanishing) == 5
    rng = MersenneTwister(53)
    nat = n_atoms(cr)
    es = [(E = randn(rng, 3, nat); E ./ sqrt.(sum(abs2, E; dims = 1))) for _ = 1:30]
    X = SLCE._design_moment(mb, es, [copy(e) for e in es])
    ref = maximum(norm(view(X, :, j)) for j in axes(X, 2))
    groups = Dict{Any,Vector{Int}}()
    for (j, s) in enumerate(salcs(mb))
        push!(get!(groups, (SLCE._block_key(s), SLCE._mark_class(s)), Int[]), j)
    end
    flagged = Set(regauged_columns(mb))
    for cols in values(groups)
        nullity = length(cols) - count(>(1e-8 * ref), svdvals(X[:, cols]))
        @test nullity == count(in(vanishing), cols)
        @test all(in(flagged), cols) == (0 < nullity < length(cols))
    end
    @test all(j -> norm(view(X, :, j)) <= 1e-12 * ref, vanishing)
    # the pointed metric weighs a block, frozen columns exempt
    m = penalty_metric(mb)
    @test all(iszero, m[collect(vanishing)])
end
