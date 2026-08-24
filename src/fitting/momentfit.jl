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
                  gate_eps, coverage_floor = 0.5, zero_moment_atol = 1e-10)
        -> MomentDataset

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
rarely exact. Every atom the pointed basis references (marked atoms and their
environment sites) must carry a nonzero magnetic moment `‖MW‖ > zero_moment_atol`
in every configuration: a quenched moment's `directions` column is the ẑ
placeholder the moments constructor fabricates, which would enter every
neighbouring row as a fake environment coordinate (and, in mode 4, become that
row's own axis while the `|M| = 0 → g = 0` convention waves the row through). The
constructor refuses such a datum by name — the same door as `SLCEDataset`'s
`zero_moment_atol`, with the same obligation: the placeholder was fabricated by
the READER at ITS `zero_moment_atol` (`read_extxyz`, `read_embset_pair`, the
moments constructor), so if the data were built with a custom value, **pass the
same value here**. Two recorded applicability limits: the gate reads the MARKED
atom only, so a SOFT (small but above `zero_moment_atol`) environment moment
enters every neighbouring row through its `directions` column with no gate of
its own — only the hard placeholder case is refused; and the gate is even in
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
    order::Vector{Float64}
    vanishing::Vector{Int}
    dependent::Vector{Vector{Tuple{Int,Float64}}}
    gate_eps::Float64
    coverage_floor::Float64
    provenance::DatumProvenance
end

# The mode rule (D8 addendum), stated ONCE: mode 4 evaluates on the datum's
# `directions` (identity substitution), mode 1 on its `constraint_axes` (present
# is a TrainingDatum ctor invariant). Every consumer needing a row axis — the
# dataset constructor and the local-field diagnostics — resolves through this
# function; a second inline copy is the coupled-site drift hazard.
_moment_axis_matrix(d::TrainingDatum)::Matrix{Float64} =
    (d.constraint_mode::Int) == 4 ? d.directions : (d.constraint_axes::Matrix{Float64})

function MomentDataset(basis::MomentBasis, data::AbstractVector{TrainingDatum};
                       gate_eps::Real, coverage_floor::Real = 0.5,
                       zero_moment_atol::Real = 1e-10)::MomentDataset
    isempty(data) && throw(ArgumentError("no training data"))
    gate_eps >= 0 || throw(ArgumentError("gate_eps = $gate_eps must be ≥ 0"))
    0 <= coverage_floor <= 1 ||
        throw(ArgumentError("coverage_floor = $coverage_floor must be in [0, 1]"))
    zero_moment_atol >= 0 ||
        throw(ArgumentError("zero_moment_atol = $zero_moment_atol must be ≥ 0"))
    referenced = _referenced_atoms(basis)
    labels = basis.crystal.species_labels
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
        length(d.magmoms) == nat || throw(ArgumentError(
            "config $ci has $(length(d.magmoms)) magmoms but the basis crystal " *
            "has $nat"))
        # The zero-moment placeholder door (the moment channel's analogue of
        # `_check_referenced_moments`): a referenced atom with ‖MW‖ ≤ atol carries
        # the fabricated ẑ direction. [Backported from SCEFitting.jl bb94992.]
        for a = 1:nat
            (referenced[a] && d.magmoms[a] <= zero_moment_atol) || continue
            throw(ArgumentError(
                "config $ci: atom $a ($(labels[basis.crystal.species[a]])) has a " *
                "zero magnetic moment (‖MW‖ = $(d.magmoms[a]) ≤ $zero_moment_atol) " *
                "but is referenced by the pointed basis (a marked atom or an " *
                "environment site) — its placeholder ẑ direction would enter the " *
                "design as a fabricated coordinate. Drop the configuration, or " *
                "unsample the species if it is non-magnetic"))
        end
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
        ax = _moment_axis_matrix(d)
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
            # moment_simple_floor's pairing door replays this expression
            # BITWISE — keep muladd/@fastmath out of both sites, or relax both
            # to isapprox together
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
        # a WIDE signature block can carry O(columns) dense combinations; the
        # log summarizes past a handful (full list stays on `ds.dependent`)
        if length(mr.null_combinations) <= 8
            @warn msg combinations = mr.null_combinations
        else
            cols = sort!(unique!([j for c in mr.null_combinations for (j, _) in c]))
            @warn msg n_combinations = length(mr.null_combinations) columns = cols
        end
    end

    # Per-config coverage coordinate: the marked-sublattice order parameter
    # |⟨e⟩| = ‖Σ_a e_a‖ / n_marked — the axis of the band-profile diagnostic
    # (design record M2-8/L2-2).
    order = Vector{Float64}(undef, ncfg)
    for (ci, e) in enumerate(configs)
        s1 = 0.0; s2 = 0.0; s3 = 0.0
        for a in atoms
            s1 += e[1, a]; s2 += e[2, a]; s3 += e[3, a]
        end
        order[ci] = sqrt(s1^2 + s2^2 + s3^2) / nmark
    end

    X = _design_moment(basis, configs, axes)
    ident = DatumProvenance(; reference_id = p1.reference_id,
                            reference_fingerprint = p1.reference_fingerprint,
                            setup_id = p1.setup_id, soc = p1.soc)
    return MomentDataset(basis, X, y, defined, keep, gate, row_config, row_atom,
                         orbit_rep, report, order, collect(Int, mr.vanishing),
                         # copied, like `vanishing`: the record is CACHED on the
                         # basis, so storing the reference would alias one object
                         # across the cache and every dataset built from it
                         [copy(c) for c in mr.null_combinations], Float64(gate_eps),
                         Float64(coverage_floor), ident)
end

# Every atom a pointed SALC member touches — marked atoms and their environment
# sites; the zero-moment door reads it (mirror of `_referenced_atoms(::SLCEBasis)`).
function _referenced_atoms(mb::MomentBasis)::BitVector
    ref = falses(n_atoms(mb.crystal))
    for s in salcs(mb), mem in s.members, a in mem.atoms
        ref[a] = true
    end
    return ref
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
    _check_metric_provenance(estimator, :moment, ds.basis.salc_basis.fingerprint, 0.0)
    p = size(ds.X, 2)
    active = trues(p)
    active[ds.vanishing] .= false
    any(active) || throw(ArgumentError("every pointed column vanishes on this cell"))
    est = _reduce_to_active(estimator, active)
    function _solve(mask::BitVector)::Vector{Float64}
        c = solve_coefficients(est, ds.X[mask, active], ds.y[mask];
                               row_groups = ds.row_config[mask])
        full = zeros(p)              # frozen columns: exact zero, not solver noise
        full[active] = c
        return full
    end
    # the UN-reduced estimator is stored (user-facing provenance); the reduced
    # one that actually produced the coefficients is a derived object — rebuild
    # via _reduce_to_active if a future diagnostic needs it
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

# ── band-profile diagnostic ────────────────────────────────────────────────────────

"""
    moment_band_profile(model::MomentModel, ds::MomentDataset; nbins = 4)
    moment_band_profile(f::MomentFit; nbins = 4)

The coverage-band residual profile (design record M2-8/L2-2): per-configuration
mean residuals of the moment channel, organized along the marked-sublattice order
parameter `|⟨e⟩|` (`ds.order`). Returns

- `bands` — `nbins` equal-count bins in order of increasing `|⟨e⟩|`, each
  `(; lo, hi, mean_residual, n)` (equal COUNT, not equal width: with tied
  `|⟨e⟩|` values adjacent bins can share an edge value, so `[lo, hi]` ranges
  need not partition the axis);
- `slope`, `intercept` — the least-squares line of per-config mean residual vs
  `|⟨e⟩|` (the bin-free statement of the same trend);
- `r` — the Pearson correlation of the two;
- `order`, `mean_residual` — the underlying per-config points (plot fodder).

Residuals are `y − X·V` over the KEPT rows only; a configuration with no kept
row is absent from the profile, and every present configuration counts once
regardless of how many of its rows survived the gate. The per-config mean runs
over ALL marked orbits together — on a multi-species marked basis a trend in a
small-moment orbit can be masked by a large-moment one (disaggregate by
`ds.orbit_rep` when that matters). NaN conventions: a degenerate order spread
(`sxx = 0`) gives `slope = r = NaN` with the intercept still the plain mean
residual; a perfect fit (`syy = 0`) gives `slope = 0` with `r = NaN`. A
systematic band trend on held-out data is the basis-insufficiency signature the
design record's L2-2 reinterpretation names — report it next to any σ.
"""
function moment_band_profile(model::MomentModel, ds::MomentDataset; nbins::Int = 4)
    nbins >= 1 || throw(ArgumentError("nbins = $nbins must be ≥ 1"))
    model.basis === ds.basis || throw(ArgumentError(
        "the model and the dataset carry different bases — a same-width mismatch " *
        "would profile nonsense silently, so identity is required"))
    pred = ds.X * model.coeffs
    ncfg = length(ds.order)
    sums = zeros(ncfg)
    counts = zeros(Int, ncfg)
    for r in eachindex(ds.y)
        ds.keep[r] || continue
        c = ds.row_config[r]
        sums[c] += ds.y[r] - pred[r]
        counts[c] += 1
    end
    cfgs = [c for c = 1:ncfg if counts[c] > 0]
    isempty(cfgs) && throw(ArgumentError("no configuration has a kept row"))
    mres = [sums[c] / counts[c] for c in cfgs]
    ord = ds.order[cfgs]
    q = sortperm(ord)
    n = length(cfgs)
    bands = @NamedTuple{lo::Float64, hi::Float64, mean_residual::Float64, n::Int}[]
    nb = min(nbins, n)          # effective bin count: every config lands in a bin
    for b = 1:nb
        sel = q[div((b - 1) * n, nb)+1:div(b * n, nb)]
        isempty(sel) && continue
        push!(bands, (; lo = minimum(ord[sel]), hi = maximum(ord[sel]),
                      mean_residual = sum(mres[sel]) / length(sel), n = length(sel)))
    end
    mo = sum(ord) / n
    mr_ = sum(mres) / n
    sxx = sum((o - mo)^2 for o in ord)
    sxy = sum((ord[i] - mo) * (mres[i] - mr_) for i = 1:n)
    syy = sum((v - mr_)^2 for v in mres)
    slope = sxx == 0.0 ? NaN : sxy / sxx
    r = (sxx == 0.0 || syy == 0.0) ? NaN : sxy / sqrt(sxx * syy)
    # Degenerate order spread (sxx = 0): the slope is undefined (NaN), but the
    # intercept is still the plain mean residual — never NaN-poisoned.
    intercept = sxx == 0.0 ? mr_ : mr_ - slope * mo
    return (; bands, slope, intercept, r, order = ord, mean_residual = mres)
end

moment_band_profile(f::MomentFit; nbins::Int = 4) =
    moment_band_profile(MomentModel(f), f.dataset; nbins)

# --- group labels for group-adaptive shrinkage over pointed columns ----------------

"""
    salc_groups(mb::MomentBasis) -> Vector{Int}

Per-design-matrix-column group labels for the pointed basis (contiguous `1:G`, one
label per SALC in `SALCKey` order): columns grouped by
`(key.body, key.orbit_id, key.decors, mark class)`, where the mark class is the
stabilizer orbit of mark placements — read off as the sorted set of representative-
member atoms that carry the DISP (mark) slot across the SALC's terms.

The energy-side key `(body, orbit_id, decors)` is NOT enough here: pointed SALC keys
sort the decoration into a canonical multiset, so two stabilizer-inequivalent mark
placements of one cluster — e.g. an Fe–Ge pair marked on Fe versus marked on Ge,
which predict different atoms' moments — share `(body, orbit_id, decors)` and differ
only in `block`. Folding them into one group would couple the adaptive shrinkage of
physically distinct channels. Gauge blocks (`L_S`/`Lf`/`block`) of ONE mark class
still share a label, mirroring the energy-side convention.

One SALC spans exactly one stabilizer orbit of assignments, so its terms' mark
placements enumerate exactly one mark class. The class is fingerprinted on the
FIRST member (deterministic: `_canonicalize_members` sorts members by
`(atoms, shifts)`) by BOTH the marked reference-cell atoms AND the marked site
indices: the atom set alone is not injective — a canonical member carrying two
periodic images of one atom projects two distinct mark placements onto the same
atom set (reproduced on a 2-atom P1 cell, `nbody = 3`: member atoms `[1, 1, 2]`,
mark on site 1 vs site 2, both "atom 1"), and folding those couples physically
distinct channels. The site set can in principle be finer than the true class
partition across gauge blocks (zero-folded terms are dropped per block), which
errs on the safe side: over-splitting adds a group, merging couples channels.
Feed the labels to [`GroupAdaptiveRidge`](@ref), or use the
`GroupAdaptiveRidge(mb; lambda, ...)` convenience (unit weights: the moment
channel has no Monte-Carlo contraction cost to weight by).
"""
function salc_groups(mb::MomentBasis)::Vector{Int}
    sal = salcs(mb)
    labels = Vector{Int}(undef, length(sal))
    seen = Dict{Tuple{Int,Int,Vector{SiteDecor},Vector{Int},Vector{Int}},Int}()
    for (j, s) in enumerate(sal)
        mem = s.members[1]                       # canonical representative member
        marks = Int[]
        sites = Int[]
        for t in mem.terms
            # the one-mark invariant (the same one `_mark_term_index` asserts):
            # a second DISP slot would make "the mark class" ill-defined
            nmk = count(sl -> sl.factor.channel == DISP, t.slots)
            nmk == 1 || error("pointed SALC $j carries $nmk mark slots in one " *
                              "term; the mark-class key assumes exactly one")
            ms = findfirst(sl -> sl.factor.channel == DISP, t.slots)
            push!(sites, t.slots[ms].site)
            push!(marks, mem.atoms[t.slots[ms].site])
        end
        key = (s.key.body, s.key.orbit_id, s.key.decors, sort!(unique!(marks)),
               sort!(unique!(sites)))
        labels[j] = get!(seen, key, length(seen) + 1)
    end
    return labels
end

# ── the penalty metric of the pointed channel ──────────────────────────────────────

"""
    _intercept_columns(mb::MomentBasis) -> Vector{Int}

The μ₀ columns of the pointed design: the 1-body pointed SALCs whose single site is
the mark with `spin_l = 0`, i.e. the constant `Φ = 1` per mark class. They set the
reference moment magnitude, so shrinking them toward zero has no physical meaning —
[`penalty_metric`](@ref)`(mb)` exempts them by setting their scale to exactly `0`.

The discriminant cannot misfire: a `SiteDecor` with `spin_l == 0` must carry a
displacement factor (its inner constructor refuses a bare `l = 0` spin decor), and in
a pointed label the displacement factor **is** the mark, so `body == 1` with a single
`spin_l == 0` decor is exactly the marked constant.
"""
function _intercept_columns(mb::MomentBasis)::Vector{Int}
    out = Int[]
    for (j, s) in enumerate(salcs(mb))
        s.key.body == 1 && length(s.key.decors) == 1 || continue
        d = s.key.decors[1]
        (d.spin_l == 0 && is_marked(d)) && push!(out, j)
    end
    return out
end

"""
    penalty_metric(mb::MomentBasis; free_intercepts = true, nconfig = 2048, seed = 1)
        -> Vector{Float64}

The per-column penalty scale of a pointed moment basis, in design-column order — the
moment channel's counterpart of [`penalty_metric`](@ref)`(::SLCEBasis)`, and the same
argument for it: `λ·Σⱼβⱼ²` is not invariant under rescaling a column, and pointed SALC
column norms are set by the star's member count and the ordering multiplicity the
member fold absorbs, so the plain penalty prefers large orbits and high body order for
reasons that are conventions rather than physics.

    mⱼ = E[Φⱼ²]

over `nconfig` independent uniform-random spin configurations, evaluated on the pointed
design rows (one per marked reference-cell atom per configuration) with the
identity-substituted evaluation axis. Not centered, unlike the energy channel: the
moment design has no column centering to match — its rows carry the intercept columns
explicitly.

Two families of column get an exact `0`, which the estimators read as **unpenalized**:

- the μ₀ intercepts (`_intercept_columns`), unless `free_intercepts = false`. Shrinking
  the reference moment magnitude toward zero is not a modelling choice anyone wants;
  `false` exists only to make the comparison measurable.
- the columns [`moment_resolvability`](@ref) reports as identically vanishing on this
  cell. They are frozen out of the solve anyway; a zero scale keeps the metric
  consistent with the design the solver actually sees.

Any OTHER zero is refused: zero means a structural exemption, so it is never inferred
from a sample.

Unlike the energy channel, the moment design reaches the estimator with **no row
whitening**, so the column norms the solver sees are `n_rows · mⱼ`. Relative weighting
and scale invariance are unaffected — it is a uniform factor — but λ on this channel is
therefore not comparable across datasets of different row count, where the energy
channel's `√(1/n_E)` makes it so.
"""
function penalty_metric(mb::MomentBasis; free_intercepts::Bool = true,
                        nconfig::Integer = 2048, seed::Integer = 1)::Vector{Float64}
    # The structural exemptions come FIRST: `moment_resolvability` is also the
    # `UnclassifiableBasis` door, so on a basis this cell cannot resolve the user should
    # not first pay a full reference-ensemble evaluation inside a constructor. The
    # columns it names are then skipped rather than measured and discarded.
    res = moment_resolvability(mb)
    structural = copy(res.vanishing)
    free_intercepts && append!(structural, _intercept_columns(mb))
    sort!(unique!(structural))
    p = n_salcs(mb)
    measured = setdiff(1:p, structural)

    nat = n_atoms(mb.crystal)
    cfgs = _reference_configs(nat, Int(nconfig), Int(seed))
    nmark = max(1, length(mb.marked_atoms))
    # Accumulate over configuration chunks rather than building the whole reference
    # design: the full `nconfig · n_marked × p` block is ~100 MB for a 3x3x3 supercell
    # basis, and this runs in a constructor. The chunk is sized by BYTES, so the peak
    # buffer stays put as the column count grows.
    chunk = clamp(cld(_METRIC_CHUNK_BYTES, 8 * nmark * max(1, p)), 1, length(cfgs))
    index = _mark_term_index(salcs(mb), mb.marked_atoms)   # once, not once per chunk
    acc = zeros(Float64, p)
    nrow = 0
    for lo = 1:chunk:length(cfgs)
        sub = cfgs[lo:min(lo + chunk - 1, length(cfgs))]
        X = _design_moment(mb, sub, sub; index = index)    # identity axes: mode-4
        nrow += size(X, 1)
        # A plain sequential row loop, not `sum`: `sum`'s pairwise tree would make the
        # last bits of every entry of `m` a function of the chunk size, i.e. of a
        # tuning constant. This way the accumulation order is the row order whatever
        # the chunking.
        for j in measured
            a = acc[j]
            @inbounds for i = 1:size(X, 1)
                a += abs2(X[i, j])
            end
            acc[j] = a
        end
    end
    m = acc ./ nrow
    _refuse_zero_metric(m, "penalty_metric(::MomentBasis)"; exempt = structural,
                        hint = " `moment_resolvability` should have named it, and " *
                               "that it did not is worth understanding before fitting.")
    return m
end

# Resolve the `metric` keyword of a pointed basis-aware constructor. A separate name
# from the energy side's `_basis_metric` on purpose: the two would otherwise carry
# different meanings in the same positional slot (`torque_weight` there,
# `free_intercepts` here), which is the readability trap the divergence ledger's first
# row exists for, reintroduced inside the package.
function _pointed_metric(mb::MomentBasis, metric, free_intercepts::Bool,
                         nconfig::Integer, seed::Integer)
    metric === :basis || return (_checked_metric_keyword(metric), nothing)
    m = penalty_metric(mb; free_intercepts = free_intercepts, nconfig = nconfig,
                       seed = seed)
    pv = MetricProvenance(:moment, 0.0, free_intercepts, nconfig, seed,
                          mb.salc_basis.fingerprint)
    return (m, pv)
end

"""
    GroupAdaptiveRidge(mb::MomentBasis; lambda, epsilon = 1e-8, max_iter = 50,
                       tol = 1e-6, metric = :basis, free_intercepts = true,
                       metric_nconfig = 2048, metric_seed = 1)

Group-adaptive estimator for a pointed basis: [`salc_groups`](@ref)`(mb)` labels
with UNIT weights (the moment channel has no MC contraction cost; the energy-side
`cost_weights` story does not apply), and by default
[`penalty_metric`](@ref)`(mb; free_intercepts, ...)`. See the primary
[`GroupAdaptiveRidge`](@ref) constructor for the estimator itself.

The metric is what keeps the μ₀ intercept columns **out** of the penalty, and it does
so identically for all three estimators — a group weight could only have done it for
the group form, leaving `fit(MomentFit, ds, Ridge(λ))` quietly shrinking the reference
moment. Pass `metric = nothing` for the unweighted penalty (which does shrink μ₀), or
a vector of your own.
"""
function GroupAdaptiveRidge(mb::MomentBasis; lambda::Real, epsilon::Real = 1e-8,
                            max_iter::Integer = 50, tol::Real = 1e-6,
                            metric = :basis, free_intercepts::Bool = true,
                            metric_nconfig::Integer = 2048, metric_seed::Integer = 1)
    cg = salc_groups(mb)
    m, pv = _pointed_metric(mb, metric, free_intercepts, metric_nconfig, metric_seed)
    return GroupAdaptiveRidge(cg, ones(maximum(cg)); lambda = lambda,
                              epsilon = epsilon, max_iter = max_iter, tol = tol,
                              metric = m, metric_provenance = pv)
end

"""
    Ridge(mb::MomentBasis; lambda, metric = :basis, free_intercepts = true,
          metric_nconfig = 2048, metric_seed = 1)

Ridge for a pointed basis, carrying [`penalty_metric`](@ref)`(mb; free_intercepts,
...)`. That metric is what keeps the μ₀ intercept columns out of the penalty; see
[`GroupAdaptiveRidge`](@ref)`(mb; ...)` for why it is the metric that does it rather
than a group weight.
"""
function Ridge(mb::MomentBasis; lambda::Real, metric = :basis,
               free_intercepts::Bool = true, metric_nconfig::Integer = 2048,
               metric_seed::Integer = 1)
    m, pv = _pointed_metric(mb, metric, free_intercepts, metric_nconfig, metric_seed)
    return Ridge(lambda, m, pv)
end

"""
    AdaptiveRidge(mb::MomentBasis; lambda, epsilon = 1e-8, max_iter = 50, tol = 1e-6,
                  metric = :basis, free_intercepts = true, metric_nconfig = 2048,
                  metric_seed = 1)

The per-coefficient adaptive ridge for a pointed basis. Keyword semantics as in
[`Ridge`](@ref)`(mb; ...)`.
"""
function AdaptiveRidge(mb::MomentBasis; lambda::Real, epsilon::Real = 1e-8,
                       max_iter::Integer = 50, tol::Real = 1e-6, metric = :basis,
                       free_intercepts::Bool = true, metric_nconfig::Integer = 2048,
                       metric_seed::Integer = 1)
    m, pv = _pointed_metric(mb, metric, free_intercepts, metric_nconfig, metric_seed)
    return AdaptiveRidge(; lambda = lambda, epsilon = epsilon, max_iter = max_iter,
                         tol = tol, metric = m, metric_provenance = pv)
end

# Column-structured estimators must follow fit's vanishing-column reduction: the
# frozen columns are removed from the solve, so per-column metadata has to shrink
# with them. For GroupAdaptiveRidge the reduction preserves every group NORM
# exactly (frozen coefficients are exact zeros), but the group SIZE p_g drops by
# the frozen count, so the weight w_g = v_g/(‖β_g‖² + p_g·ε) moves at O(ε) —
# material only for a group already at the ε floor, and defensible there: ε is
# documented as a per-coefficient floor and a frozen column carries no
# coefficient. Note the deliberate divergence from the energy side, where
# ASR-frozen columns STAY in column_groups and keep their p_g contribution.
# Groups emptied by the reduction are relabeled away (the estimator's
# every-label-present contract). Every other estimator passes through unchanged.
_reduce_to_active(estimator::AbstractEstimator, ::BitVector) = estimator
function _reduce_to_active(estimator::GroupAdaptiveRidge,
                           active::BitVector)::GroupAdaptiveRidge
    all(active) && return estimator
    length(estimator.column_groups) == length(active) || throw(DimensionMismatch(
        "GroupAdaptiveRidge column_groups length $(length(estimator.column_groups)) " *
        "does not match the pointed design column count $(length(active)); build " *
        "the labels on THIS basis (salc_groups(mb))"))
    sub = estimator.column_groups[findall(active)]
    remap = Dict{Int,Int}()
    labels = [get!(remap, g, length(remap) + 1) for g in sub]
    old = Vector{Int}(undef, length(remap))
    for (g, n) in remap
        old[n] = g
    end
    return GroupAdaptiveRidge(labels, estimator.group_weights[old];
                              lambda = estimator.lambda, epsilon = estimator.epsilon,
                              max_iter = estimator.max_iter, tol = estimator.tol,
                              metric = _reduce_metric(estimator.metric, active,
                                                      "GroupAdaptiveRidge"),
                              metric_provenance = estimator.metric_provenance)
end

# --- local-field diagnostics + the simple-feature nested floor ----------------------

# The pair-consistent neighbor sets of the marked atoms: for each marked atom, the
# reference-cell atoms j reachable through the spec's `cutoff_pair` MinimumImage
# enumeration, one entry PER TIED IMAGE (the pair basis's member multiplicity),
# same-atom images excluded (the pair enumeration's i == j drop), restricted to
# species the basis reads (`lmax_env > 0`). The tie band is read off the basis
# (`mb.tie_tol`, stored at construction), so the enumeration matches the basis's
# by construction — no manual matching.
function _pair_neighbors(mb::MomentBasis)
    spec = mb.spec
    sp = mb.crystal.species
    nbrs = Dict(a => Int[] for a in mb.marked_atoms)
    # a 1-body basis reads no environment at all: `cutoff_pair` is a required
    # spec field regardless of `nbody`, so the shell it names is not one the
    # model ever sees — report an empty field (h₁ = 0, alignment undefined)
    # rather than a coordinate with no bearing on the fit
    spec.nbody >= 2 || return nbrs
    nl = build_neighbor_list(mb.crystal, spec.cutoff_pair, MinimumImage();
                             tol = mb.tie_tol)
    for p in nl.pairs
        haskey(nbrs, p.i) || continue
        spec.lmax_env[sp[p.j]] > 0 || continue
        push!(nbrs[p.i], p.j)
    end
    return nbrs
end

@inline _legendre(l::Int, x::Float64)::Float64 =
    l == 1 ? x :
    l == 2 ? (3.0 * x^2 - 1.0) / 2 :
    l == 3 ? (5.0 * x^3 - 3.0 * x) / 2 :
    throw(ArgumentError("Legendre order $l not supported (1 ≤ l ≤ 3)"))

"""
    moment_local_field(mb::MomentBasis, configs; axes = configs)
    moment_local_field(mb::MomentBasis, data::AbstractVector{TrainingDatum})
        -> NamedTuple

Per-row local-field diagnostics of the moment channel, rows aligned with
[`MomentDataset`](@ref) (configuration-major, marked-atom-minor). For each row
`(c, i)` the pair-consistent local direction field is `h₁ = Σ ê_j` over the marked
atom's `cutoff_pair` MinimumImage neighbors — one term per tied image (the pair
basis's member multiplicity), same-atom images excluded, species with
`lmax_env = 0` excluded (the basis never reads them). Returns

- `h1` — `‖h₁‖` per row (`0.0` when the basis reads no environment species);
- `edoth` — `ê·h₁/‖h₁‖`, the alignment of the row's evaluation axis with the
  local field (`NaN` when `‖h₁‖ = 0` or the row axis is undefined — the mode-1
  zero-axis case). This is the collapse coordinate: on FeGe, anti-alignment
  (`ê·ĥ < 0`), not weak `|h₁|`, is what marks the longitudinal-collapse rows;
- `row_config`, `row_atom` — the row bookkeeping.

The `TrainingDatum` method resolves the row axis by the mode rule
(`_moment_axis_matrix`, the same function the dataset constructor reads); the raw
method's `axes = configs` default IS the mode-4 identity, exactly like
[`predict_moment`](@ref) — pass explicit axes for a mode-1-style readout. A
validating door: configuration columns unit everywhere; axes columns unit on the
MARKED atoms, with an exactly-zero column allowed (undefined row, `edoth = NaN`).
The neighbor tie band is the basis's own (`mb.tie_tol`).
"""
function moment_local_field(mb::MomentBasis,
                            configs::AbstractVector{<:AbstractMatrix{<:Real}};
                            axes::AbstractVector{<:AbstractMatrix{<:Real}} = configs)
    isempty(configs) && throw(ArgumentError("no configurations"))
    nat = n_atoms(mb.crystal)
    length(axes) == length(configs) ||
        throw(ArgumentError("$(length(axes)) axis matrices for " *
                            "$(length(configs)) configurations"))
    nbrs = _pair_neighbors(mb)
    marked = mb.marked_atoms
    nmark = length(marked)
    nrow = length(configs) * nmark
    h1 = Vector{Float64}(undef, nrow)
    edoth = Vector{Float64}(undef, nrow)
    row_config = Vector{Int}(undef, nrow)
    row_atom = Vector{Int}(undef, nrow)
    for (ci, e) in enumerate(configs)
        _validate_config(e, nat)
        ax = axes[ci]
        size(ax) == (3, nat) ||
            throw(ArgumentError("config $ci: axes are $(size(ax)), expected (3, $nat)"))
        for (ai, a) in enumerate(marked)
            r = (ci - 1) * nmark + ai
            row_config[r] = ci
            row_atom[r] = a
            hx = hy = hz = 0.0
            for j in nbrs[a]
                hx += e[1, j]; hy += e[2, j]; hz += e[3, j]
            end
            h = sqrt(hx^2 + hy^2 + hz^2)
            h1[r] = h
            a1, a2, a3 = Float64(ax[1, a]), Float64(ax[2, a]), Float64(ax[3, a])
            if a1 == 0.0 && a2 == 0.0 && a3 == 0.0
                edoth[r] = NaN                       # undefined row axis (mode 1)
            else
                _validate_direction(SVector{3,Float64}(a1, a2, a3),
                                    "axes column $a (config $ci)")
                edoth[r] = h == 0.0 ? NaN : (a1 * hx + a2 * hy + a3 * hz) / h
            end
        end
    end
    return (; h1, edoth, row_config, row_atom)
end

function moment_local_field(mb::MomentBasis, data::AbstractVector{TrainingDatum})
    isempty(data) && throw(ArgumentError("no data"))
    for (ci, d) in enumerate(data)
        d.constraint_mode === nothing && throw(ArgumentError(
            "config $ci carries no `constraint_mode` — the row axis is resolved " *
            "by the mode rule"))
    end
    return moment_local_field(mb, [d.directions for d in data];
                              axes = [_moment_axis_matrix(d) for d in data])
end

"""
    moment_coverage(train::NamedTuple, new::NamedTuple; q = 0.99) -> NamedTuple

Feature-space coverage monitor (design record M2-8): compare a runtime/validation
batch against the training distribution in the [`moment_local_field`](@ref)
coordinates. Returns

- `threshold` — the `q`-quantile of the TRAINING `h1`;
- `frac_beyond` — the fraction of NEW rows with `h1 > threshold` (≈ `1 − q` in
  distribution; substantially more flags extrapolation in the local-field
  strength);
- `frac_anti` — the fraction of NEW rows (among those with a defined `edoth`)
  with `ê·ĥ < 0`: the collapse-risk band the training set rarely visits — rows
  there carry the model's largest errors (measured on FeGe), so a consumer
  sampling them is outside the fitted regime;
- `n_rows`, `n_defined` — the new batch's row counts.

Both arguments are [`moment_local_field`](@ref) outputs (or any NamedTuple with
`h1` and `edoth`) — from the SAME basis, which the monitor cannot verify. The
`h1` flag is UPPER-tail only by design (the recorded M2-8 axis; on a degenerate
training distribution — every row at one `|h₁|` — it can never fire); the
weak-field side is monitored through `frac_anti`, since the measured collapse
coordinate is anti-alignment, not field strength.
"""
function moment_coverage(train::NamedTuple, new::NamedTuple; q::Real = 0.99)
    0 < q < 1 || throw(ArgumentError("q = $q must lie in (0, 1)"))
    isempty(train.h1) && throw(ArgumentError("empty training rows"))
    isempty(new.h1) && throw(ArgumentError("empty new rows"))
    thr = quantile(train.h1, Float64(q))
    defined = .!isnan.(new.edoth)
    nd = count(defined)
    return (; threshold = thr, q = Float64(q),
            frac_beyond = count(>(thr), new.h1) / length(new.h1),
            frac_anti = nd == 0 ? NaN : count(<(0.0), new.edoth[defined]) / nd,
            n_rows = length(new.h1), n_defined = nd)
end

"""
    moment_simple_floor(f::MomentFit, data::AbstractVector{TrainingDatum};
                        lmax = 2) -> NamedTuple

The simple-feature NESTED performance floor (design record M2-5): fit, on exactly
the SALC fit's kept rows, the trivial geometric model

    y ≈ μ_g + Σ_{l = 1}^{lmax} b_{g,l} Σ_j P_l(ê_i · ê_j)

— one intercept and one Legendre shell-sum slope per marked-atom orbit `g`, the
sum over the same `cutoff_pair` neighbors as [`moment_local_field`](@ref). Returns

- `sigma_floor` / `sigma_model` — `std` of the simple model's and the SALC fit's
  gated residuals. The nested bound `sigma_model ≤ sigma_floor` holds on the
  training rows by construction only under BOTH conditions the return value
  reports: the features lie in the SALC column span (a pair basis with
  `lmax_mark ≥ l` and `lmax_env ≥ l` contains the `P_l` shell sums — read
  `inclusion`) AND the fit is unregularized (`nested_bound = true` ⇔
  `f.estimator isa OLS`; a shrinkage estimator legitimately trades training
  residual for variance, so the bound does not apply and its violation
  diagnoses nothing). An OLS fit with `inclusion ≈ 0` reporting
  `sigma_model > sigma_floor` IS mis-assembled. The `std` comparison rides on
  both residual vectors having zero mean, which holds because both designs
  span the per-orbit constants (the floor's intercepts; the SALC design's μ₀
  columns). With `design_rank ≥ n_rows` (a saturated design) both sigmas are
  trivially ~0 and the statement is vacuous — the returned counts disclose it;
- `inclusion` — per feature column, the relative residual of projecting it onto
  the SALC design's kept rows (`≈ 0` ⇔ the feature is representable; reported,
  never assumed);
- `coef`, `n_features`, `feature_labels` — the simple model itself.

Both sigmas are Bessel-corrected `std` (`n − 1`), NOT [`rmse_moment`](@ref)'s
`√(Σr²/n)` — compare like with like; with a single kept row both are `NaN`.

`data` must be the very vector the fit's dataset was built from — checked loudly:
the row count, a bitwise target recomputation on every defined row, and a
replay of ONE configuration's design rows (the first whose marked axes are all
nonzero) through the production build. The replay covers the environment
columns the target check cannot see, but for that one configuration only — a
co-rotated `(e, M)` substitution confined to another configuration passes the
door. A dataset in which every configuration carries a zero-axis marked atom
falls back to the target-only door.
"""
function moment_simple_floor(f::MomentFit, data::AbstractVector{TrainingDatum};
                             lmax::Integer = 2)
    1 <= lmax <= 3 || throw(ArgumentError("lmax = $lmax must lie in 1:3"))
    ds = f.dataset
    mb = ds.basis
    marked = mb.marked_atoms
    nmark = length(marked)
    length(data) * nmark == length(ds.y) || throw(ArgumentError(
        "$(length(data)) configurations × $nmark marked atoms ≠ " *
        "$(length(ds.y)) dataset rows — pass the vector the dataset was built from"))
    # pairing door, two halves. (1) Targets: recompute through the SAME
    # expression the dataset constructor evaluates (bitwise — keep muladd /
    # @fastmath out of BOTH sites, or relax both to isapprox together).
    for r in eachindex(ds.y)
        ds.defined[r] || continue
        d = data[ds.row_config[r]]
        (d.moments_bare === nothing || d.constraint_mode === nothing) &&
            throw(ArgumentError("config $(ds.row_config[r]) lacks the moment " *
                                "channel fields — not the dataset's data"))
        a = ds.row_atom[r]
        ax = _moment_axis_matrix(d)
        M = d.moments_bare::Matrix{Float64}
        yv = ax[1, a] * M[1, a] + ax[2, a] * M[2, a] + ax[3, a] * M[3, a]
        yv == ds.y[r] || throw(ArgumentError(
            "recomputed target of row $r ($(yv)) ≠ dataset target ($(ds.y[r])) — " *
            "`data` does not pair with the fit's dataset"))
    end
    # (2) Environment: the target check reads only the marked columns, but the
    # floor's features read the whole `directions` matrix — replay ONE
    # configuration's design rows through the production build and compare
    # exactly (same code path ⇒ ==). The first config whose marked axis
    # columns are all nonzero replays verbatim; a dataset without one (every
    # config carries a zero-axis marked atom) falls back to the target-only
    # door, disclosed here rather than silently weakened.
    ci_full = findfirst(ci -> all(!iszero, eachcol(
                            _moment_axis_matrix(data[ci])[:, marked])),
                        eachindex(data))
    if ci_full !== nothing
        d = data[ci_full]
        Xrep = _design_moment(mb, [d.directions], [_moment_axis_matrix(d)])
        rows = ((ci_full - 1) * nmark + 1):(ci_full * nmark)
        Xrep == ds.X[rows, :] || throw(ArgumentError(
            "config $ci_full's replayed design rows differ from the dataset's — " *
            "`data` does not pair with the fit's dataset (environment columns " *
            "moved)"))
    end

    nbrs = _pair_neighbors(mb)
    kept = findall(ds.keep)
    # orbits with at least one KEPT row only: an orbit surviving the coverage
    # floor with zero kept rows (possible at a relaxed floor) would otherwise
    # contribute identically-zero feature columns and a silent rank deficiency
    orbits = sort(unique(ds.orbit_rep[kept]))
    gidx = Dict(g => k for (k, g) in enumerate(orbits))
    nfeat = length(orbits) * (1 + lmax)
    col(g, l) = (gidx[g] - 1) * (1 + lmax) + 1 + l        # l = 0 is the intercept
    F = zeros(length(kept), nfeat)
    # the feature x = ê_i·ê_j is a cosine only on unit columns, and P₂/P₃ are
    # nonlinear — validate every configuration once (a diagnostic entry point
    # is a door, deliberately stricter than the dataset constructor)
    for d in data
        _validate_config(d.directions, n_atoms(mb.crystal))
    end
    for (kr, r) in enumerate(kept)
        d = data[ds.row_config[r]]
        e = d.directions
        ax = _moment_axis_matrix(d)
        a = ds.row_atom[r]
        g = ds.orbit_rep[r]
        F[kr, col(g, 0)] = 1.0
        for j in nbrs[a]
            x = ax[1, a] * e[1, j] + ax[2, a] * e[2, j] + ax[3, a] * e[3, j]
            for l = 1:lmax
                F[kr, col(g, l)] += _legendre(l, x)
            end
        end
    end
    yk = ds.y[kept]
    c = F \ yk
    sigma_floor = std(yk - F * c)
    sigma_model = std(residuals(f))
    # one-sided inclusion: is each feature representable in the SALC design's
    # kept rows? (reported, never assumed — the nesting claim rides on it)
    # Projection onto span(X_kept) through an explicit orthonormal range basis
    # with a stated rank cut — never `Xk \ F`: the vanishing columns are
    # identically zero BY CONSTRUCTION (the design is rank-deficient whenever
    # the resolvability record is nonempty), and a square kept design would
    # even dispatch `\` to LU (SingularException / garbage). The active-column
    # reduction mirrors fit's.
    act = setdiff(1:size(ds.X, 2), ds.vanishing)
    Xk = ds.X[kept, act]
    S = svd(Xk)
    rank_cut = maximum(S.S; init = 0.0) * maximum(size(Xk)) * eps(Float64)
    nrank = count(>(rank_cut), S.S)
    U = @view S.U[:, 1:nrank]
    proj = U * (U' * F)
    inclusion = [begin
                     nf = norm(@view F[:, k])
                     nf == 0.0 ? 0.0 : norm(@view(F[:, k]) - @view(proj[:, k])) / nf
                 end for k = 1:nfeat]
    labels = [l == 0 ? "orbit $g: intercept" : "orbit $g: P$l shell sum"
              for g in orbits for l = 0:lmax]
    return (; sigma_floor, sigma_model, nested_bound = f.estimator isa OLS,
            coef = c, inclusion, n_features = nfeat, feature_labels = labels,
            n_rows = length(kept), design_rank = nrank)
end
