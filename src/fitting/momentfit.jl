# The moment channel's dataset + fit — the pointed basis's counterpart of
# `SLCEDataset` / `fit(SLCEFit, …)`. An INDEPENDENT vertical slice: its own rows
# (config, marked atom), its own targets y = ê·M_int, its own coefficient vector V
# (the l = 0 1-body [MARK] columns are the per-orbit intercepts μ₀) — it never
# touches the energy/torque/force row bookkeeping. Design record: _brain_storming/
# adiabatic-moment-sce (D8 addendum: marked-column substitution + the mode rule;
# D9′: periodic resolvability wired at the dataset door; M2 corrections:
# decomposability gate, coverage floor, both-coefficients disclosure).

# ── MomentDataset ──────────────────────────────────────────────────────────────────

"""
    MomentDataset(basis::MomentBasis, data::AbstractVector{TrainingDatum};
                  gate_eps, coverage_floor = 0.5) -> MomentDataset

Assemble the moment channel's regression problem. Rows are (configuration-major,
marked-atom-minor) pairs over `basis`'s marked atoms; the target of row `(c, a)` is
the decomposable projection `y = ê·M` of the bare internal moment `M =
moments_bare[:, a]` onto the row's axis `ê`.

The axis follows the **mode rule** (D8 addendum), per datum:

- `constraint_mode == 4` (direction-pinning): `ê = directions[:, a]` — the
  marked-column substitution is the identity;
- `constraint_mode == 1` (transverse-penalty): `ê = constraint_axes[:, a]`. A marked
  atom whose axis column is exactly zero (unconstrained in the source calculation)
  has no defined projection: its rows are excluded from every fit (`defined =
  false`) and recorded in the per-orbit report.

Modes may be mixed in one dataset. Every datum must carry `moments_bare` and
`constraint_mode` (loud otherwise), the whole vector must come from one
computational setup AND one reference identity (same checks as `SLCEDataset`), and
a datum carrying nonzero `displacements` is refused: the v1 moment channel expands
`m_i(e)` at the clamped-ion reference geometry only (a displaced configuration
would be silently fitted as if it sat at the reference — the joint expansion is a
recorded follow-up, not a silent approximation).

The **decomposability gate** keeps rows whose transverse remainder is small:
`g = ‖M⊥‖²/|M| = |M| sin²θ ≤ gate_eps` (μB, computed in the cancellation-free form
`M⊥ = M − y ê`). `|M| = 0` gives `g = 0` and passes — a quenched moment with a
zero target is data, not noise. There is deliberately no `m_min` gate (recorded as
an applicability limit, not enforced). `gate_eps` has no default: the tolerance is
a physical statement about the source calculation's constraint quality, so the
caller must state it. `gate_eps = 0` keeps only rows with `g` exactly zero — in
practice the `|M| = 0` rows, since a decomposable row's `g` is roundoff-scale but
rarely exact. Two recorded applicability limits: the gate reads the MARKED atom
only, so a collapsed/quenched ENVIRONMENT atom enters every neighbouring row
through its `directions` column with no gate of its own; and the gate is even in
`y`, so a mode-1 row whose converged moment is antiparallel to the recorded axis
passes like a parallel one — exactly covariant for odd mark ranks, but the
even-mark-rank columns (μ₀, `l_mark = 2`) assume the recorded axis is oriented on
the converged moment's side. The per-orbit report therefore counts antiparallel
rows (`n_anti`, mode-1 rows with `ê·e_MW < 0`).

The design matrix `X` is built for ALL rows; `defined` and `keep` masks select the
fit rows, so both the gated and the ungated coefficient sets can be disclosed from
one dataset ([`fit`](@ref)). Rows with `defined = false` carry `y = NaN` (loud on
accidental use) and a placeholder design row (evaluated with `ê = directions`,
never used by any fit).

Per marked-atom orbit (the space-group orbit of the marked atom, `map_sym`
min-atom representative), the constructor records row counts, survival `n_kept /
n_rows`, and the rms transverse remainder `‖M⊥‖` over the DEFINED rows —
gate-rejected rows included by design (the report discloses what the gate saw,
not what it kept); an orbit with no defined row reports `NaN`. Any orbit with
survival below `coverage_floor` is refused loudly — a fit that silently loses one
sublattice's rows would report a model it cannot support.

The constructor also runs [`moment_resolvability`](@ref) on the basis (before
paying for the design build): an unclassifiable basis propagates that gate's
refusal, identically-vanishing columns are recorded in `vanishing` (frozen to
exact zero by [`fit`](@ref)), and structurally dependent column combinations are
warned loudly and stored in `dependent` — their fitted coefficients are one
arbitrary min-norm representative, disclosed, never silent.
"""
struct MomentDataset
    basis::MomentBasis
    X::Matrix{Float64}
    y::Vector{Float64}
    defined::BitVector
    keep::BitVector
    gate::Vector{Float64}
    row_config::Vector{Int}
    row_atom::Vector{Int}
    orbit_rep::Vector{Int}
    orbit_report::Vector{@NamedTuple{orbit::Int, atoms::Vector{Int}, n_rows::Int,
                                     n_defined::Int, n_kept::Int, n_anti::Int,
                                     survival::Float64, mperp_rms::Float64}}
    vanishing::Vector{Int}
    dependent::Vector{Vector{Tuple{Int,Float64}}}
    gate_eps::Float64
    coverage_floor::Float64
    provenance::DatumProvenance
end

function MomentDataset(basis::MomentBasis, data::AbstractVector{TrainingDatum};
                       gate_eps::Real, coverage_floor::Real = 0.5)::MomentDataset
    isempty(data) && throw(ArgumentError("no training data"))
    gate_eps >= 0 || throw(ArgumentError("gate_eps = $gate_eps must be ≥ 0"))
    0 <= coverage_floor <= 1 ||
        throw(ArgumentError("coverage_floor = $coverage_floor must be in [0, 1]"))
    _check_setup_uniformity(data)
    # One reference identity per dataset (the provenance summary below claims datum
    # 1's reference for the whole batch, so the batch must actually share it).
    p1 = data[1].provenance
    for (i, d) in enumerate(data)
        p = d.provenance
        (p.reference_id == p1.reference_id &&
         p.reference_fingerprint == p1.reference_fingerprint) || throw(ArgumentError(
            "config $i has reference_id = $(repr(p.reference_id)) but config 1 has " *
            "$(repr(p1.reference_id)) — one MomentDataset expands m_i(e) at ONE " *
            "reference geometry; build one dataset per reference"))
    end
    nat = n_atoms(basis.crystal)
    atoms = basis.marked_atoms
    nmark = length(atoms)
    ncfg = length(data)
    nrow = ncfg * nmark

    # Resolve the per-datum axis matrix (mode rule) and assemble targets + masks
    # BEFORE the design build: the coverage refusal must fire without paying for X.
    configs = Vector{Matrix{Float64}}(undef, ncfg)
    axes = Vector{Matrix{Float64}}(undef, ncfg)
    y = Vector{Float64}(undef, nrow)
    defined = trues(nrow)
    keep = falses(nrow)
    anti = falses(nrow)
    gate = Vector{Float64}(undef, nrow)
    mperp = Vector{Float64}(undef, nrow)
    row_config = Vector{Int}(undef, nrow)
    row_atom = Vector{Int}(undef, nrow)
    for (ci, d) in enumerate(data)
        size(d.directions, 2) == nat || throw(ArgumentError(
            "config $ci has $(size(d.directions, 2)) atoms but the basis crystal " *
            "has $nat"))
        d.moments_bare === nothing && throw(ArgumentError(
            "config $ci carries no `moments_bare` — the moment channel fits the " *
            "bare internal moments; rebuild the datum from a source that provides " *
            "them (extxyz `mint`, EMBSET_mint)"))
        d.constraint_mode === nothing && throw(ArgumentError(
            "config $ci carries no `constraint_mode` — the axis of y = ê·M is " *
            "resolved by the mode rule (4 → directions, 1 → constraint_axes), so " *
            "a datum without a declared mode has no defined target"))
        if d.displacements !== nothing && !all(iszero, d.displacements)
            throw(ArgumentError(
                "config $ci carries nonzero displacements — the v1 moment channel " *
                "expands m_i(e) at the clamped-ion reference geometry only, and " *
                "fitting a displaced configuration as if it sat at the reference " *
                "silently mixes two structures (the joint m_i(e, u) expansion is a " *
                "recorded follow-up). Drop the displaced configurations"))
        end
        M = d.moments_bare::Matrix{Float64}
        mode = d.constraint_mode::Int
        # mode == 1 ⇒ constraint_axes present is a TrainingDatum ctor invariant.
        ax = mode == 4 ? d.directions : (d.constraint_axes::Matrix{Float64})
        configs[ci] = d.directions
        # Always a copy: the zero-axis placeholder below writes into `axes[ci]`,
        # and aliasing the caller's datum field would turn that into silent
        # mutation the day the unit-column invariant on `directions` is loosened.
        axes[ci] = copy(ax)
        for (ai, a) in enumerate(atoms)
            r = (ci - 1) * nmark + ai
            row_config[r] = ci
            row_atom[r] = a
            e1, e2, e3 = ax[1, a], ax[2, a], ax[3, a]
            if e1 == 0.0 && e2 == 0.0 && e3 == 0.0
                # mode-1 unconstrained marked atom: no axis, no target. The design
                # row still gets built (placeholder axis = the datum's direction),
                # but `defined = false` bars it from every fit and y = NaN is loud
                # on accidental raw use.
                defined[r] = false
                y[r] = NaN
                gate[r] = NaN
                mperp[r] = NaN
                axes[ci][1, a] = d.directions[1, a]
                axes[ci][2, a] = d.directions[2, a]
                axes[ci][3, a] = d.directions[3, a]
                continue
            end
            yv = e1 * M[1, a] + e2 * M[2, a] + e3 * M[3, a]
            mm = sqrt(M[1, a]^2 + M[2, a]^2 + M[3, a]^2)
            # Cancellation-free transverse remainder: M⊥ = M − y ê exactly, then
            # g = ‖M⊥‖²/|M| = |M| sin²θ (non-negative by construction — the naive
            # |M| − y²/|M| cancels to O(sin²θ) and can round negative).
            w1 = M[1, a] - yv * e1
            w2 = M[2, a] - yv * e2
            w3 = M[3, a] - yv * e3
            mp = sqrt(w1^2 + w2^2 + w3^2)
            y[r] = yv
            mperp[r] = mp
            g = mm == 0.0 ? 0.0 : mp^2 / mm
            gate[r] = g
            keep[r] = g <= gate_eps
            if mode == 1
                anti[r] = e1 * d.directions[1, a] + e2 * d.directions[2, a] +
                          e3 * d.directions[3, a] < 0.0
            end
        end
    end

    # Marked-atom orbits (space-group orbit, min-atom representative) + survival.
    rep = [minimum(@view basis.spacegroup.map_sym[a, :]) for a in atoms]
    orbit_rep = Vector{Int}(undef, nrow)
    for r = 1:nrow
        orbit_rep[r] = rep[(r - 1) % nmark + 1]
    end
    report = @NamedTuple{orbit::Int, atoms::Vector{Int}, n_rows::Int, n_defined::Int,
                         n_kept::Int, n_anti::Int, survival::Float64,
                         mperp_rms::Float64}[]
    for o in sort!(unique(rep))
        rows = findall(==(o), orbit_rep)
        nd = count(r -> defined[r], rows)
        nk = count(r -> keep[r], rows)
        na = count(r -> anti[r], rows)
        surv = nk / length(rows)
        drows = [r for r in rows if defined[r]]
        mrms = isempty(drows) ? NaN :
               sqrt(sum(mperp[r]^2 for r in drows) / length(drows))
        push!(report, (; orbit = o, atoms = [a for (a, rp) in zip(atoms, rep) if rp == o],
                       n_rows = length(rows), n_defined = nd, n_kept = nk,
                       n_anti = na, survival = surv, mperp_rms = mrms))
    end
    for t in report
        t.survival >= coverage_floor || throw(ArgumentError(
            "marked-atom orbit $(t.orbit) keeps $(t.n_kept)/$(t.n_rows) rows " *
            "(survival $(round(t.survival, digits = 3)) < coverage_floor " *
            "$coverage_floor; $(t.n_rows - t.n_defined) undefined-axis, " *
            "$(t.n_defined - t.n_kept) gate-rejected at gate_eps = $gate_eps μB) — " *
            "a fit that loses this sublattice's rows cannot claim its columns. " *
            "Raise gate_eps deliberately, or drop the source configurations"))
    end
    ndrop = nrow - count(keep)
    if ndrop > 0
        @info "MomentDataset: $ndrop of $nrow rows excluded " *
              "($(nrow - count(defined)) undefined-axis, " *
              "$(count(defined) - count(keep)) gate-rejected at gate_eps = " *
              "$gate_eps μB); per-orbit survival " *
              join(["$(t.orbit): $(t.n_kept)/$(t.n_rows)" for t in report], ", ")
    end

    # The D9′ periodic-resolvability gate, wired at the dataset door (before the
    # design build): what this cell cannot determine must not come back as a
    # silently-arbitrary coefficient. An unclassifiable basis (repeated-image
    # environments) propagates that gate's own loud refusal.
    mr = moment_resolvability(basis)
    if !isempty(mr.vanishing)
        msg = "MomentDataset: $(length(mr.vanishing)) pointed column(s) vanish " *
              "identically on this cell's periodic data and will be frozen to " *
              "exact zero by fit"
        @warn msg columns = mr.vanishing
    end
    if !isempty(mr.null_combinations)
        msg = "MomentDataset: the design carries $(length(mr.null_combinations)) " *
              "structurally dependent column combination(s) on this cell — the " *
              "fitted coefficients along these directions are one arbitrary " *
              "min-norm representative (predictions on this cell are unaffected)"
        @warn msg combinations = mr.null_combinations
    end

    X = _design_moment(basis, configs, axes)
    ident = DatumProvenance(; reference_id = p1.reference_id,
                            reference_fingerprint = p1.reference_fingerprint,
                            setup_id = p1.setup_id, soc = p1.soc)
    return MomentDataset(basis, X, y, defined, keep, gate, row_config, row_atom,
                         orbit_rep, report, collect(Int, mr.vanishing),
                         mr.null_combinations, Float64(gate_eps),
                         Float64(coverage_floor), ident)
end

function Base.show(io::IO, ds::MomentDataset)
    print(io, "MomentDataset(", size(ds.X, 1), " rows (", count(ds.keep), " kept, ",
          count(ds.defined) - count(ds.keep), " gated, ",
          length(ds.defined) - count(ds.defined), " undefined), ",
          size(ds.X, 2), " columns, gate_eps = ", ds.gate_eps, " μB)")
end

# ── fit ────────────────────────────────────────────────────────────────────────────

"""
    fit(MomentFit, ds::MomentDataset, estimator::AbstractEstimator = OLS())
        -> MomentFit

Solve the moment channel's regression on the gated rows (`ds.keep`), and — for
disclosure — a second solve on all defined rows (`ds.defined`); a large gap between
the two coefficient sets means the gate is doing physics, not cleanup, and must be
reported alongside any result.

Columns named by the dataset's resolvability record as identically vanishing on
this cell (`ds.vanishing`) are excluded from both solves and their coefficients
frozen to **exact zero** — the same discipline as the energy side's frozen
columns, so a downstream `coef != 0` test reads structure, not solver noise.
Structurally dependent combinations (`ds.dependent`) stay in the solve: their
min-norm representative is disclosed by the dataset's construction warning.

Both solves pass `row_groups = ds.row_config` for their rows: the marked-atom
rows of one configuration are one physical sample, and a resampling estimator
must never split them across folds.

There is **no centering and no global intercept**: the `l = 0` 1-body `[MARK]`
columns are the per-orbit intercepts μ₀ (one per marked-atom orbit — never shared
across species; the `l = 2` 1-body columns are on-site ê anisotropies, not
intercepts), so the design is passed to the estimator exactly as built. Note for
regularized estimators (`Ridge`, …): the μ₀ columns are penalized like every other
column — v1 is OLS-first; shrinking intercepts is a deliberate choice, not a
default to reach for.
"""
struct MomentFit
    dataset::MomentDataset
    estimator::AbstractEstimator
    coeffs::Vector{Float64}
    coeffs_ungated::Vector{Float64}
end

function fit(::Type{MomentFit}, ds::MomentDataset,
             estimator::AbstractEstimator = OLS())::MomentFit
    any(ds.keep) || throw(ArgumentError("no rows survive the gate"))
    p = size(ds.X, 2)
    active = trues(p)
    active[ds.vanishing] .= false
    any(active) || throw(ArgumentError("every pointed column vanishes on this cell"))
    function _solve(mask::BitVector)::Vector{Float64}
        c = solve_coefficients(estimator, ds.X[mask, active], ds.y[mask];
                               row_groups = ds.row_config[mask])
        full = zeros(p)              # frozen columns: exact zero, not solver noise
        full[active] = c
        return full
    end
    return MomentFit(ds, estimator, _solve(ds.keep), _solve(ds.defined))
end

coef(f::MomentFit)::Vector{Float64} = f.coeffs

"""
    residuals(f::MomentFit; gated = true) -> Vector{Float64}

Fit residuals `y − X·V` on the gated rows (`gated = false`: all defined rows, with
the ungated coefficients).
"""
function residuals(f::MomentFit; gated::Bool = true)::Vector{Float64}
    ds = f.dataset
    mask = gated ? ds.keep : ds.defined
    c = gated ? f.coeffs : f.coeffs_ungated
    return ds.y[mask] .- ds.X[mask, :] * c
end

"""
    rmse_moment(f::MomentFit; gated = true) -> Float64

Root-mean-square residual of `y = ê·M` [μB] over the fit's own rows.
"""
function rmse_moment(f::MomentFit; gated::Bool = true)::Float64
    r = residuals(f; gated)
    return sqrt(sum(abs2, r) / length(r))
end

function Base.show(io::IO, f::MomentFit)
    print(io, "MomentFit(", length(f.coeffs), " coefficients, ",
          count(f.dataset.keep), " rows, rmse = ",
          round(rmse_moment(f), sigdigits = 4), " μB)")
end

# ── MomentModel + prediction ───────────────────────────────────────────────────────

"""
    MomentModel(basis, coeffs, provenance)
    MomentModel(f::MomentFit)

The deployable adiabatic-moment model `m_i(e) = Σ_φ V_φ Φ_φ(i; e)` — the pointed
basis with its fitted (gated) coefficients. Carries provenance only: there is no
mode flag — the constraint mode is a property of how each *training datum's* axis
was resolved, not of the model (D8 addendum), and at run time the axis is the
consumer's spin direction (`predict_moment`'s default `axes = e`).
"""
struct MomentModel
    basis::MomentBasis
    coeffs::Vector{Float64}
    provenance::DatumProvenance

    function MomentModel(basis::MomentBasis, coeffs::Vector{Float64},
                         provenance::DatumProvenance)
        length(coeffs) == n_salcs(basis) || throw(DimensionMismatch(
            "$(length(coeffs)) coefficients for $(n_salcs(basis)) pointed SALCs"))
        return new(basis, coeffs, provenance)
    end
end

MomentModel(f::MomentFit)::MomentModel =
    MomentModel(f.dataset.basis, f.coeffs, f.dataset.provenance)

function Base.show(io::IO, m::MomentModel)
    print(io, "MomentModel(", length(m.coeffs), " coefficients, ",
          length(m.basis.marked_atoms), " marked atoms)")
end

"""
    predict_moment(model, e; axes = e) -> Vector{Float64}

Predict the projected site moments `y_a = ê_a·m_a(e)` [μB] for one spin
configuration `e` (3 × n_atoms unit columns), one entry per marked atom in
`model.basis.marked_atoms` order. `axes` is the per-atom projection-axis matrix;
the run-time default is the configuration itself (`ê_a = e_a` — the consumer asks
for the moment magnitude along each spin), which is exactly the mode-4 identity
substitution. Pass training-style axes to reproduce fit rows.

This is a spin-reading entry point, so it is a validating DOOR: `e` must be unit
columns throughout (it feeds every environment factor), and `axes` must be unit
on the MARKED columns — the only ones the substitution reads; unmarked `axes`
columns are free, so a caller extracting closed-form components at `ê = x̂, ŷ, ẑ`
need not fabricate axes for atoms it is not asking about.
"""
function predict_moment(model::MomentModel, e::AbstractMatrix{<:Real};
                        axes::AbstractMatrix{<:Real} = e)::Vector{Float64}
    nat = n_atoms(model.basis.crystal)
    _validate_config(e, nat)
    em = Matrix{Float64}(e)
    am = Matrix{Float64}(axes)
    size(am) == size(em) || throw(DimensionMismatch(
        "axes is $(size(am)), expected $(size(em))"))
    for a in model.basis.marked_atoms
        u = SVector{3,Float64}(am[1, a], am[2, a], am[3, a])
        _validate_direction(u, "`axes` column $a")
    end
    X = _design_moment(model.basis, [em], [am])
    return X * model.coeffs
end

predict_moment(f::MomentFit, e::AbstractMatrix{<:Real};
               axes::AbstractMatrix{<:Real} = e)::Vector{Float64} =
    predict_moment(MomentModel(f), e; axes)
