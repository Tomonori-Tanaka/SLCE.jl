# Fitting and prediction: assemble the regression problem, run the estimator
# (fit/refit), and evaluate the fitted surface (predict_energy/predict_torque).

# Assemble the regression problem the estimator actually sees: the column-centered
# energy-only design (`wT == wF == 0`), or the whitened, stacked energy(+torque)(+force)
# design, together with the centering constants `xbar`/`ybar` (for the analytic `j0`)
# and the per-row resampling-unit `groups`. Shared by `fit` and `refit` so the two build
# byte-identical designs. The force block is stored COMPACT (columns = the
# displacement-active SALCs in `dataset.force_cols`); it is scattered into the full
# design width here — the structurally zero columns exist only in the assembled
# stack, never in the dataset.
#
# With an ASR reparameterization `rep` (built once at dataset construction — this
# function only APPLIES it, design record §6 amendment 1) every block is
# right-multiplied by the orthonormal null-space basis `Z` BEFORE stacking (the
# compact force block via `Z[force_cols, :]`, never materializing the scattered
# zeros), so the estimator solves in γ with `β = Z·γ` exactly satisfying `A·β = 0`.
# `xbar` is then the γ-space column mean — `j0 = ȳ − x̄ᵀγ = ȳ − (x̄_β)ᵀβ` either
# way. `rep === nothing` (pure-spin, or `asr = false`) is the structural fast path,
# bitwise identical to the unconstrained assembly.
function _assemble_problem(dataset::SLCEDataset, wT::Float64, wF::Float64 = 0.0,
                           rep::Union{Nothing,ASRReparam} = nothing)
    # Affine reparameterization `β = beta_p + Z·γ` (a staged fit: `beta_p` carries
    # the frozen coefficients plus, when the frozen part violates the ASR, the
    # particular solution of `A_free·β_free = −A_frozen·β_frozen`). The offset is a
    # KNOWN contribution to every observable, so it moves to the target side —
    # `ỹ = y − X_β·beta_p` — before centering and whitening, leaving the estimator
    # the homogeneous problem in γ. `beta_p ≡ 0` (every non-staged fit) reduces to
    # the previous code path bitwise: `_offset` returns the target untouched.
    affine = rep !== nothing && any(!iszero, rep.beta_p)
    X_E = rep === nothing ? dataset.X_E : dataset.X_E * rep.Z
    y_E = affine ? dataset.y_E .- dataset.X_E * rep.beta_p : dataset.y_E
    xbar = vec(mean(X_E; dims = 1))
    ybar = mean(y_E)
    if wT > 0 || wF > 0
        n_E = length(y_E)
        # max guards the two-rounding complement: the caller gates wT + wF ≤ 1 on
        # the summed value, and 1 − wT − wF can round to −eps for sums at the
        # boundary — sqrt of that is a DomainError, not a fit.
        se = sqrt(max(0.0, 1 - wT - wF) / n_E)
        blocksX = [(X_E .- xbar') .* se]
        blocksy = [(y_E .- ybar) .* se]
        # Resampling-unit labels for grouped cross-validation: a configuration's energy
        # row and all its derivative-component rows share its index, so a CV-based
        # estimator never splits one configuration across folds. The derivative-row
        # labels are READ from the dataset's stored per-row config indices
        # (`torque_config` / `force_config`) — never re-derived from a uniform-block
        # assumption, which a mixed (ragged) design violates.
        groups = collect(1:n_E)
        if wT > 0
            n_T = length(dataset.y_T)
            # A mixed dataset can be sliced down to a torque-free part (e.g. a CV
            # training fold that happens to hold no torque-bearing config); √(w/0)
            # would silently produce a degenerate objective, so fail loudly here.
            n_T > 0 || throw(ArgumentError(
                "torque_weight = $wT > 0 but this dataset (or data split) has no " *
                "torque rows — stratify the split by torque presence or use " *
                "torque_weight = 0"))
            # On a mixed dataset the torque-bearing MINORITY carries the whole weight
            # wT (the objective is per-block means). That is the intended semantics,
            # but it is qualitatively different from the all-torque case, so say so
            # once instead of silently redefining what torque_weight weighs. The
            # per-config row count is read off the stored bookkeeping (3 × n_atoms
            # on the pure-spin path; 3 × n_spin_referenced on the joint path).
            tblock = length(searchsorted(dataset.torque_config,
                                         dataset.torque_config[1]))
            if n_T < tblock * n_E
                @warn "mixed dataset: torque rows cover only part of the configurations" *
                      " — torque_weight = $wT concentrates on the torque-bearing subset" coverage =
                    round(n_T / (tblock * n_E); digits = 3) maxlog = 1
            end
            sm = sqrt(wT / n_T)
            push!(blocksX, rep === nothing ? dataset.X_T .* sm :
                           (dataset.X_T * rep.Z) .* sm)
            y_T = affine ? dataset.y_T .- dataset.X_T * rep.beta_p : dataset.y_T
            push!(blocksy, y_T .* sm)
            append!(groups, dataset.torque_config)
        end
        if wF > 0
            n_F = length(dataset.y_F)
            n_F > 0 || throw(ArgumentError(
                "force_weight = $wF > 0 but this dataset (or data split) has no " *
                "force rows — stratify the split by force presence or use " *
                "force_weight = 0"))
            # Per-config force-row count read off the stored bookkeeping (every
            # force-bearing config has the same 3 × n_referenced_atoms rows) —
            # avoids re-walking the SALC basis on every assembly.
            fblock = length(searchsorted(dataset.force_config,
                                         dataset.force_config[1]))
            if n_F < fblock * n_E
                @warn "mixed dataset: force rows cover only part of the configurations" *
                      " — force_weight = $wF concentrates on the force-bearing subset" coverage =
                    round(n_F / (fblock * n_E); digits = 3) maxlog = 1
            end
            sf = sqrt(wF / n_F)
            if rep === nothing
                Xf = zeros(Float64, n_F, size(X_E, 2))
                @views Xf[:, dataset.force_cols] .= dataset.X_F .* sf
            else
                # compact-block path: only the force-active rows of Z are needed
                Xf = (dataset.X_F .* sf) * rep.Z[dataset.force_cols, :]
            end
            push!(blocksX, Xf)
            # the force block is COMPACT: its columns are `force_cols`, so the
            # offset reads the matching slice of `beta_p`
            y_F = affine ?
                  dataset.y_F .- dataset.X_F * view(rep.beta_p, dataset.force_cols) :
                  dataset.y_F
            push!(blocksy, y_F .* sf)
            append!(groups, dataset.force_config)
        end
        # ≥ 2 blocks always: wT > 0 || wF > 0 held, so a derivative block joined
        X = reduce(vcat, blocksX)
        y = reduce(vcat, blocksy)
    else
        X = X_E .- xbar'
        y = y_E .- ybar
        groups = nothing                            # each row is its own configuration
    end
    return X, y, xbar, ybar, groups
end

# Weight / channel validation shared by `fit` and the pre-fit diagnostics
# ([`identifiability`](@ref)), so a diagnostic assembled off a dataset raises the
# SAME errors the corresponding fit would instead of failing deeper in the assembly.
function _validate_fit_request(dataset::SLCEDataset, w::Float64, wF::Float64)
    isempty(dataset.y_E) && throw(ArgumentError("dataset has no observations"))
    (0.0 <= w <= 1.0) || throw(ArgumentError("torque_weight must be in [0, 1]; got $w"))
    (0.0 <= wF <= 1.0) ||
        throw(ArgumentError("force_weight must be in [0, 1]; got $wF"))
    w + wF <= 1.0 ||
        throw(ArgumentError("torque_weight + force_weight must be ≤ 1 (the energy " *
                            "block carries the remaining 1 − w_T − w_F); got " *
                            "$w + $wF = $(w + wF)"))
    if w > 0 && !has_torque(dataset)
        throw(ArgumentError("torque_weight = $w but the dataset has no torque data; " *
                            "build it with SLCEDataset(basis, configs, energies, torques)"))
    end
    if wF > 0 && !has_force(dataset)
        throw(ArgumentError("force_weight = $wF but the dataset has no force data; " *
                            "build it from TrainingDatum vectors with force channels " *
                            "against a displacement-decorated basis"))
    end
    return nothing
end

# A displacement-decorated basis without a reparameterization (AllImages
# self-images, or a hand-built dataset that skipped `build_asr`) must not silently
# take the pure-spin fast path under `asr = true` — unconstrained joint fits are an
# explicit opt-out.
function _resolve_asr_rep(dataset::SLCEDataset, asr::Bool)::Union{Nothing,ASRReparam}
    rep = asr ? dataset.asr : nothing
    if asr && rep === nothing && _basis_has_disp(dataset.basis)
        throw(ArgumentError("asr = true but this displacement-decorated dataset " *
                            "carries no ASR reparameterization (AllImages " *
                            "self-image basis, or a hand-built dataset without " *
                            "SLCE.build_asr) — pass asr = false to fit " *
                            "unconstrained deliberately"))
    end
    return rep
end

# Relative cut for a design column that carries no information. Matches the ASR
# builder's residue/dead-row convention (`fitting/asr.jl`) — every zero decision in
# this package is relative, and an exact `iszero` test has measured false negatives:
# a column that is a multiple of `Σ_a u_a` evaluates to ~1e-19 (not 0) on
# center-of-mass-free samples, and the energy block's centering leaves a
# structurally constant column at ~1e-17 rather than exactly 0.
const _DEAD_COL_RTOL = 1e-12

# Standing identifiability check on the assembled design, cheap enough (O(n·q), one
# pass) to run on every fit: a column whose norm is negligible against the largest
# carries no information — the objective is (numerically) flat along it and whatever
# value comes back is the estimator's null-space convention, not an estimate. The
# archetype is a derivative-only fit (`force_weight = 1`), where every pure-spin
# column is structurally zero. This catches only the per-COLUMN case; a flat
# direction is generally a combination of columns ([`identifiability`](@ref) computes
# the full rank deficiency at O(n·q²), too expensive to run unasked).
#
# Deliberately NOT `maxlog`-limited, unlike the mixed-dataset coverage warnings
# above: this one reports a property of THIS fit request, and a session that fits
# several weightings (energy-only, then force-only) must be warned about each one —
# `maxlog` silences a call site for the whole process. The cost is `k` identical
# lines when `cross_validate` refits per fold, which is the informative direction.
function _warn_unidentified(basis::SLCEBasis, X::AbstractMatrix,
                            rep::Union{Nothing,ASRReparam})
    isempty(X) && return nothing
    nrm = [norm(@view X[:, j]) for j in axes(X, 2)]
    nmax = maximum(nrm)
    nmax == 0.0 && return nothing                  # an all-zero design: nothing to rank
    dead = findall(<=(_DEAD_COL_RTOL * nmax), nrm)
    isempty(dead) && return nothing
    if rep !== nothing
        # Under the reparameterization the indices are γ directions, NOT `jphi`
        # positions: neither the `refit`-drops-them advice (a β-space support rule)
        # nor the per-SALC structural classification below transfers, because there
        # is no SALC at index j. Say only what is true in these coordinates.
        @warn "fit: $(length(dead)) design column(s) carry no information — this " *
              "fit's data say nothing about them, so their coefficients are an " *
              "estimator artifact rather than an estimate. `identifiability` " *
              "reports the full rank deficiency. They are directions of the ASR " *
              "null-space basis, not `jphi` positions." columns =
            first(dead, 10) coordinates = "reparameterized (γ)"
        return nothing
    end
    structural = _identically_zero_salcs(basis, dead)
    starved = setdiff(dead, structural)
    isempty(structural) ||
        @warn "fit: $(length(structural)) SALC(s) are identically zero on this " *
              "cell — their members cancel for EVERY configuration, so no amount " *
              "of data can determine them and `n_salcs` overstates the model's " *
              "real freedom. This is a property of the basis, not of this fit: " *
              "widen the cell or drop the channel. `refit` removes them from the " *
              "support." columns = first(structural, 10)
    isempty(starved) ||
        @warn "fit: $(length(starved)) design column(s) carry no information — this " *
              "fit's data say nothing about them (a force-only fit sees no " *
              "pure-spin column, for example), so their coefficients are an " *
              "estimator artifact rather than an estimate. `identifiability` " *
              "reports the full rank deficiency. `refit` drops them (their scaled " *
              "magnitude is zero)." columns = first(starved, 10) coordinates =
            "jphi (β)"
    return nothing
end

# Which of `cols` name SALCs that are zero for EVERY configuration, not merely for
# this fit's data. The two call for opposite responses — collect more data, versus
# nothing will ever help — and the old single message asserted the first for both,
# which is advice that cannot be taken when a two-atom cell's minimum-image ties
# annihilate a bond (measured: 1 of 3 columns on a bcc 2-atom lattice-only basis).
#
# Decided by evaluating the basis at a few random configurations: a handful of design
# ROWS, negligible beside the design already assembled, and reached only once a dead
# column exists. The seed is fixed — a diagnostic must not depend on ambient RNG
# state — and the threshold is the same relative cut the caller used, because an
# identically-zero SALC cancels to roundoff, not to exact zero.
function _identically_zero_salcs(basis::SLCEBasis, cols::AbstractVector{Int};
                                 nprobe::Int = 3)::Vector{Int}
    ss = salcs(basis)
    isempty(ss) && return Int[]
    nat = n_atoms(basis.crystal)
    rng = MersenneTwister(0x5A1CDEAD)
    peak = zeros(Float64, length(ss))
    scratch = SALCScratch()
    for _ = 1:nprobe
        e = randn(rng, 3, nat)
        for a = 1:nat
            c = norm(@view e[:, a])
            c > 0 && (@views e[:, a] ./= c)
        end
        u = 0.05 .* randn(rng, 3, nat)
        for j in eachindex(ss)
            peak[j] = max(peak[j], abs(evaluate_salc(ss[j], e, u, scratch)))
        end
    end
    ref = maximum(peak)
    ref == 0.0 && return [j for j in cols if j <= length(peak)]
    return [j for j in cols if j <= length(peak) && peak[j] <= _DEAD_COL_RTOL * ref]
end

"""
    fit(SLCEFit, dataset, estimator; torque_weight = 0.0, force_weight = 0.0) -> SLCEFit

Fit the SLCE coefficients. The energy design matrix is column-centered, so the
reference energy `j0` is recovered analytically as `mean(y_E − X_E·jϕ)`
(independent of the estimator), and the centered problem is handed to
`solve_coefficients`.

With `torque_weight = w_T` and/or `force_weight = w_F` (each in `[0, 1]`,
`w_T + w_F ≤ 1`) and a dataset carrying the corresponding channel (see
[`SLCEDataset`](@ref)), the fit minimizes the per-sample-normalized objective

    L = (1 − w_T − w_F)·MSE_energy + w_T·MSE_torque + w_F·MSE_force

by whitening: the centered energy block is row-scaled by `√((1−w_T−w_F)/n_E)`, the
torque block by `√(w_T/n_T)`, and the force block by `√(w_F/n_F)`, then stacked.
`j0` does not enter the derivative blocks, so it is still recovered from the energy
data alone. `w_T = w_F = 0` (the default) is a pure-energy fit; requesting a
positive weight without the matching data is an error.

A resampling estimator (a cross-validating [`ElasticNet`](@ref) / [`Lasso`](@ref))
receives per-row group labels so its folds are **grouped by configuration** — a
configuration's energy row and its torque-/force-component rows are never split
across the train/holdout boundary, which would otherwise leak within-configuration
structure into the CV estimate and bias λ selection.

With `asr = true` (the default) and a displacement-decorated basis, the fit is
solved under the exact acoustic-sum-rule constraint `A·β = 0` (energy invariance
under rigid translations `u_a → u_a + t`) by the null-space reparameterization
`β = Z·γ` stored on `dataset.asr` — the constraint holds by construction, not
approximately. Pure-spin bases have no constraints and take a structurally
identical (bitwise) path. `asr = false` fits unconstrained — for ablations and
the translation-violation demonstration only: an unconstrained joint model's
energy is not translation-invariant.

Design columns the assembled problem cannot see at all (a pure-spin column in a
force-only fit, say — the test is a relative norm cut) are **warned about**: the
data constrain nothing along them. That standing check is per-column only — call
[`identifiability`](@ref) for the full rank deficiency (flat directions are
generally combinations of columns, e.g. the translation-violating directions a
center-of-mass-free displacement sample cannot see).

## Staged (hierarchical) fits

`sector_mask` restricts the fit to a subset of the columns and `frozen` holds the
rest at the coefficients of a previously fitted [`SLCEModel`](@ref) — the standard
way to build a joint model in physical stages (fit the exchange first, then the
spin–lattice coupling against the frozen exchange, then the force constants):

```julia
f1 = fit(SLCEFit, ds, OLS(); sector_mask = :spin)
f2 = fit(SLCEFit, ds, OLS(); sector_mask = [:coupled, :lattice],
         frozen = SLCEModel(f1), torque_weight = 0.3, force_weight = 0.3)
```

`sector_mask` takes a selector symbol, a collection of them (union), or explicit
columns — see [`SLCE.sector_columns`](@ref) for the selector table. Frozen
coefficients are matched to this basis by `SALCKey`, never positionally; a frozen
value on a column the mask leaves free is ignored (that column is being re-fitted).
`j0` is never frozen — it is recovered analytically at every stage.

Under the ASR the stage's constraint becomes affine, `A_free·β_free =
−A_frozen·β_frozen`, and is solved exactly (particular solution + null space), so
**the staged model is translation-invariant as a whole**, not stage by stage. When
the frozen part was itself fitted under the ASR the right-hand side is zero and the
stage stays homogeneous — the reason a chain of stages costs nothing in exactness.
Freezing a model that violates the ASR is legal and lights up the affine path; if
its violation lies on constraint rows the free columns cannot balance, the fit
refuses and names those rows. Note that a penalty then shrinks γ toward the
particular solution rather than toward zero (an unavoidable consequence of an
affine feasible set — the fit warns).
"""
function fit(::Type{SLCEFit}, dataset::SLCEDataset, estimator::AbstractEstimator;
             torque_weight::Real = 0.0, force_weight::Real = 0.0,
             asr::Bool = true, frozen::Union{Nothing,SLCEModel} = nothing,
             sector_mask::Union{Symbol,AbstractVector} = :all)::SLCEFit
    w = Float64(torque_weight)
    wF = Float64(force_weight)
    _validate_fit_request(dataset, w, wF)
    # The converse of the `wF > 0 && !has_force` error above, and the one that costs
    # a user something: `force_weight` defaults to 0, so a dataset built from force
    # data fits its energies alone unless the weight is passed. Nothing downstream
    # can tell that apart from a deliberate energy-only fit, which is why it is said
    # here rather than left to the residuals.
    if wF == 0.0 && has_force(dataset)
        @warn "fit: the dataset carries forces but force_weight = 0 (the default) — " *
              "this fit is determined by the energies" *
              (w > 0 ? " and torques" : "") *
              " alone. Pass force_weight > 0 to use the force channel." maxlog = 1
    end
    rep = _resolve_asr_rep(dataset, asr)
    if frozen !== nothing || sector_mask !== :all
        rep = _fit_stage(dataset, rep, frozen, sector_mask, estimator)
    end
    X, y, xbar, ybar, groups = _assemble_problem(dataset, w, wF, rep)
    _warn_unidentified(dataset.basis, X, rep)
    gamma = solve_coefficients(estimator, X, y; row_groups = groups,
                               nullspace = rep === nothing ? nothing : rep.Z)
    j0 = ybar - dot(xbar, gamma)          # = ȳ − x̄_βᵀβ (γ-space x̄ under rep)
    jphi = rep === nothing ? gamma : rep.beta_p .+ rep.Z * gamma
    residuals = dataset.y_E .- (j0 .+ dataset.X_E * jphi)
    # `asr` records the CONSTRAINT, not the reparameterization: a mask-only stage
    # on a pure-spin basis carries a `Z` but no constraint rows. (Written as a
    # short-circuit on `rep === nothing` so the residual call sees the narrowed
    # type — a `Bool` flag would leave `Union{Nothing,ASRReparam}` for JET.)
    resid = rep === nothing || size(rep.A, 1) == 0 ? 0.0 : _asr_residual(rep, jphi)
    return SLCEFit(dataset, j0, jphi, estimator, residuals, w, wF,
                   rep !== nothing && size(rep.A, 1) > 0, resid, rep)
end

# Resolve the staging request into the reparameterization the solve runs under.
function _fit_stage(dataset::SLCEDataset, rep::Union{Nothing,ASRReparam},
                    frozen::Union{Nothing,SLCEModel}, sector_mask,
                    estimator::AbstractEstimator)::ASRReparam
    basis = dataset.basis
    free = sector_columns(basis, sector_mask)
    isempty(free) &&
        throw(ArgumentError("sector_mask = $(repr(sector_mask)) selects no column " *
                            "of this basis — the stage would fit nothing"))
    beta_f = frozen === nothing ? zeros(Float64, n_salcs(basis)) :
             _frozen_coefficients(basis, frozen)
    had_frozen = any(!iszero, beta_f)
    # A frozen value on a FREE column is ignored: that column is being re-fitted.
    beta_f[free] .= 0.0
    if had_frozen && !any(!iszero, beta_f)
        @warn "staged fit: `sector_mask` leaves every column of the frozen model " *
              "free, so nothing is actually frozen — narrow the mask to the " *
              "columns this stage should fit" n_free = length(free)
    end
    stage = _stage_reparam(basis, free, beta_f, rep === nothing ? nothing : rep.A)
    if any(!iszero, view(stage.beta_p, free)) && !(estimator isa OLS)
        @warn "staged fit: the frozen coefficients violate the ASR, so the " *
              "feasible set is affine and this estimator's penalty shrinks toward " *
              "the particular solution, not toward zero. Freeze an ASR-satisfying " *
              "model (any stage fitted under the ASR is one) to keep the usual " *
              "shrinkage semantics."
    end
    return stage
end

"""
    refit(f::SLCEFit, estimator = OLS(); threshold = 0.0) -> SLCEFit

Re-fit on the **support** of an existing fit `f` — the classic de-biasing step that
follows a sparse fit: select the basis functions a regularized estimator kept, then
re-solve on just those columns (by default with [`OLS`](@ref)) to remove the shrinkage
bias on the survivors. The dataset and `torque_weight` / `force_weight` of `f` are
reused, so the design is assembled exactly as in [`fit`](@ref).

A column `j` is in the support when its **scaled-magnitude contribution** exceeds
`threshold`:

    |coef(f)[j]| · ‖X[:, j]‖ > threshold

where `X` is the assembled (centered / whitened) design. `threshold = 0` (the default)
keeps the columns of nonzero scaled magnitude — the nonzero support of `f`, minus any
column the centering annihilates to a zero column. Coefficients off the support are set to
zero; `j0` is recovered analytically as in `fit`. If the support is empty, an all-zero `jϕ`
is returned (with a warning) and `j0` falls back to `mean(y_E)` — on a staged fit the
frozen part and the constraint's particular solution are kept instead (that is the
`γ = 0` model), and `j0` is re-derived from it.

A [`FixedCoefficients`](@ref) (or an [`AdaptiveLasso`](@ref) whose pilot is one) is
rejected: its fixed coefficient vector has the original column count, not the refit
support length, and is meaningless once a support has been chosen.

On an ASR-constrained fit the support restriction changes the feasible space: the
null space of `A[:, support]` is **re-derived** (design record §6 amendment 5 —
reusing the full-basis `Z` restricted to the support is not a null space, and an
unconstrained refit would silently re-break translation invariance in the de-bias
step). Setting the off-support coefficients to zero is always feasible (the
constraint is homogeneous), but a support that splits a constraint-coupled column
set can force some survivors to zero — legal, and surfaced with a warning.

On a **staged** fit (`frozen` / `sector_mask`) the de-biasing stays inside the
stage: the frozen coefficients keep their values (they were not fitted, so they
cannot be thresholded away), the support is intersected with the stage's free
columns, and the constraint is re-derived for that sub-stage — so the refitted
model is exactly as translation-invariant as the staged one.
"""
function refit(f::SLCEFit, estimator::AbstractEstimator = OLS();
               threshold::Real = 0.0)::SLCEFit
    threshold >= 0 || throw(ArgumentError("refit threshold must be ≥ 0; got $threshold"))
    _reject_fixed_coefficients(estimator)
    dataset = f.dataset
    w = f.torque_weight
    wF = f.force_weight
    # β-space assembly: the support rule and the analytic j0 live in β coordinates
    # regardless of the constraint (the alive rule measures physical contribution).
    X, y, xbar, ybar, groups = _assemble_problem(dataset, w, wF)
    jphi_in = f.jphi
    # Scaled-magnitude support on the assembled design: a column survives when its
    # contribution magnitude |jϕ_j|·‖X[:, j]‖ clears the threshold.
    support = findall(j -> abs(jphi_in[j]) * norm(@view X[:, j]) > threshold,
                      eachindex(jphi_in))
    rep = f.reparam
    # Columns the fit could actually move: everything for a plain fit, the stage's
    # free (and not structurally dead) columns for a staged one. A frozen column is
    # not a fitted coefficient, so it is never thresholded away.
    #
    # `jϕ` starts from the FROZEN part alone — `beta_p` with every free column
    # cleared. Clearing only the `movable` ones would keep the previous stage's
    # PARTICULAR SOLUTION (which lives on the free columns of an affine stage) while
    # dropping the γ part that balanced it, i.e. hand back a model that violates the
    # ASR while still reporting the constraint as applied. That distinction is why
    # the reparameterization records its own `free` set.
    jphi = zeros(Float64, length(jphi_in))
    movable = Int[]
    if rep !== nothing
        movable = [j for j in axes(rep.Z, 1) if norm(@view rep.Z[j, :]) >= 1e-12]
        support = intersect(support, movable)
        jphi = copy(rep.beta_p)
        jphi[rep.free] .= 0.0
    end
    if rep !== nothing && isempty(support)
        # Short-circuit before `solve_coefficients` (a GLMNet-backed estimator would
        # error on a zero-column design). γ = 0 is the well-defined degenerate fit,
        # and on a stage that is `beta_p` itself — feasible by construction, frozen
        # part and particular solution intact.
        @warn "refit: empty support — every fitted coefficient is at or below the " *
              "scaled-magnitude threshold. Returning the stage's particular " *
              "solution (γ = 0); j0 is re-derived." threshold
        jphi = copy(rep.beta_p)
    elseif isempty(support)
        @warn "refit: empty support — every coefficient is at or below the " *
              "scaled-magnitude threshold. Returning an all-zero jϕ; " *
              "j0 falls back to mean(y_E)." threshold
    elseif rep === nothing
        jphi[support] .= solve_coefficients(estimator, view(X, :, support), y;
                                            row_groups = groups)
    else
        # Constrained / staged refit: the surviving columns form a new stage over
        # the SAME frozen part, so the constraint is re-derived as
        # `A[:, S]·β_S = −A·β_frozen` (zero on the right whenever the frozen part is
        # itself ASR-feasible). Free columns the constraint structurally zeroes are
        # kept in the sub-stage's free set — they carry no γ but may carry a
        # particular-solution value, which must be re-derived rather than inherited.
        sub_free = sort(union(support, setdiff(rep.free, movable)))
        stage = _stage_reparam(dataset.basis, sub_free, jphi,
                               size(rep.A, 1) == 0 ? nothing : rep.A;
                               remedy = "lower the `refit` threshold so the support " *
                                        "keeps the balancing columns")
        if size(stage.Z, 2) == 0
            @warn "refit: the ASR constraint annihilates the selected support — " *
                  "every surviving column needs a dropped partner to stay " *
                  "translation-invariant. Returning the constraint's particular " *
                  "solution." threshold
            jphi = stage.beta_p
        else
            dead = [j for j in support
                    if norm(@view stage.Z[j, :]) < 1e-12 &&
                       norm(@view rep.Z[j, :]) >= 1e-12]
            isempty(dead) ||
                @warn "refit: the support splits an ASR-coupled column set — " *
                      "these surviving columns are structurally zeroed by the " *
                      "constraint" columns = dead
            Xs = X * stage.Z
            ys = any(!iszero, stage.beta_p) ? y .- X * stage.beta_p : y
            gamma = solve_coefficients(estimator, Xs, ys;
                                       row_groups = groups, nullspace = stage.Z)
            jphi = stage.beta_p .+ stage.Z * gamma
        end
    end
    j0 = ybar - dot(xbar, jphi)
    residuals = dataset.y_E .- (j0 .+ dataset.X_E * jphi)
    resid = rep === nothing || size(rep.A, 1) == 0 ? 0.0 : _asr_residual(rep, jphi)
    return SLCEFit(dataset, j0, jphi, estimator, residuals, w, wF,
                   rep !== nothing && size(rep.A, 1) > 0, resid, rep)
end

# A `FixedCoefficients` carries a fixed, full-design coefficient vector. `refit` and
# `cross_validate` must both reject it (with rationales of their own), so the predicate
# lives here as the single definition both rejection sites share.
_carries_fixed_coefficients(e::AbstractEstimator)::Bool =
    e isa FixedCoefficients || (e isa AdaptiveLasso && e.pilot isa FixedCoefficients)

# `refit` re-solves on a column sub-matrix `X[:, support]`, so a `FixedCoefficients` —
# whose fixed coefficient vector carries the original full column count — would throw a
# `DimensionMismatch` deep in `solve_coefficients`, and is meaningless once a support has
# been chosen. Reject it (and an `AdaptiveLasso` carrying one) upfront with a clear message.
function _reject_fixed_coefficients(e::AbstractEstimator)
    if e isa FixedCoefficients
        throw(ArgumentError("refit does not accept a FixedCoefficients: its fixed " *
            "coefficient vector has the original column count, not the refit support " *
            "length. Pass a fresh estimator such as OLS()."))
    elseif e isa AdaptiveLasso && e.pilot isa FixedCoefficients
        throw(ArgumentError("refit does not accept an AdaptiveLasso whose pilot is a " *
            "FixedCoefficients: the pilot's fixed coefficient vector has the original " *
            "column count, not the refit support length. Use a fresh pilot such as OLS()."))
    end
    return nothing
end

"""
    SLCEModel(f::SLCEFit) -> SLCEModel

Extract the lightweight predictor from a fit.
"""
SLCEModel(f::SLCEFit) = SLCEModel(f.dataset.basis, f.j0, f.jphi, f.dataset.basis.salc_basis.keys)

# A displacement-decorated model has no unambiguous spin-only prediction: silently
# assuming u = 0 would hide forgotten displacement data, so the two-argument predict
# forms refuse and name the joint form. (The exact clamped-ion sub-model is the
# future `restrict(model, :spin)`.)
function _require_pure_spin_predict(model::SLCEModel, what::AbstractString)
    _basis_has_disp(model.basis) &&
        throw(ArgumentError("the model's basis is displacement-decorated — pass " *
                            "the displacement field explicitly: $what(model, e, u) " *
                            "(u = zeros(3, n_atoms) for the clamped-ion reference; " *
                            "e = nothing for a lattice-only model)"))
    return nothing
end

"""
    predict_energy(model, data) -> Float64 or Vector{Float64}
    predict_energy(model, e, u) -> Float64

Predict the energy of a spin configuration (`3 × n_atoms` matrix) or a vector of
configurations. The vector form is evaluated in parallel over `Threads.nthreads()`
threads (set `julia -t` / `JULIA_NUM_THREADS`); the result is thread-count-independent.

For a displacement-decorated (joint) model the displacement field `u` (`3 × n_atoms`,
Å from the clamped-ion reference, same column convention) must be passed explicitly —
the two-argument form refuses rather than silently assume `u = 0`.
"""
function predict_energy(model::SLCEModel, config::AbstractMatrix{<:Real})::Float64
    _require_pure_spin_predict(model, "predict_energy")
    _validate_config(config, n_atoms(model.basis.crystal))
    salcs = model.basis.salc_basis.salcs
    e = model.j0
    scratch = SALCScratch()                 # reused workspace (dnPl + harmonic tables)
    @inbounds for k in eachindex(model.jphi)
        e += model.jphi[k] * evaluate_salc(salcs[k], config, scratch)
    end
    return e
end
function predict_energy(model::SLCEModel, configs::AbstractVector)::Vector{Float64}
    nat = n_atoms(model.basis.crystal)
    for (i, c) in enumerate(configs)                   # serial: clean errors, not wrapped
        _validate_config(c, nat; label = "config $i")
    end
    out = Vector{Float64}(undef, length(configs))
    Threads.@threads for i in eachindex(configs, out)  # independent slots
        out[i] = predict_energy(model, configs[i])
    end
    return out
end
predict_energy(f::SLCEFit, data) = predict_energy(SLCEModel(f), data)

"""
    predict_torque(model, config) -> Matrix{Float64}
    predict_torque(model, e, u) -> Matrix{Float64}

Predict the per-atom torque `τ_a = −e_a × ∂E/∂e_a` of a spin configuration
(`3 × n_atoms`), returned as a `3 × n_atoms` matrix. A vector of configurations
returns a vector of such matrices. This is the Landau–Lifshitz / physical torque
`m_a × B_eff,a` (the negative rotation-gradient of the energy), the analytic
derivative of the same surface [`predict_energy`](@ref) evaluates, so the two are
consistent by construction (`τ = −e × ∇E`). The vector form is evaluated in parallel
over `Threads.nthreads()` threads; the result is thread-count-independent.

For a displacement-decorated (joint) model the displacement field `u` must be passed
explicitly (see [`predict_energy`](@ref)); the torque is then the spin gradient of
the joint surface at `(e, u)`.
"""
function predict_torque(model::SLCEModel, config::AbstractMatrix{<:Real})::Matrix{Float64}
    _require_pure_spin_predict(model, "predict_torque")
    nat = n_atoms(model.basis.crystal)
    _validate_config(config, nat)
    salcs = model.basis.salc_basis.salcs
    G = zeros(Float64, 3, nat)
    scratch = SALCScratch()                 # reused workspace (dnPl + harmonic tables)
    @inbounds for k in eachindex(model.jphi)
        accumulate_grad!(G, salcs[k], config, model.jphi[k], scratch)
    end
    T = Matrix{Float64}(undef, 3, nat)
    @inbounds for a = 1:nat
        ea = SVector{3,Float64}(config[1, a], config[2, a], config[3, a])
        ga = SVector{3,Float64}(G[1, a], G[2, a], G[3, a])
        t = cross(ga, ea)                # τ = ∇E × e = −e × ∇E  (physical / LL torque)
        T[1, a] = t[1]
        T[2, a] = t[2]
        T[3, a] = t[3]
    end
    return T
end
function predict_torque(model::SLCEModel, configs::AbstractVector)::Vector{Matrix{Float64}}
    nat = n_atoms(model.basis.crystal)
    for (i, c) in enumerate(configs)                   # serial: clean errors, not wrapped
        _validate_config(c, nat; label = "config $i")
    end
    out = Vector{Matrix{Float64}}(undef, length(configs))
    Threads.@threads for i in eachindex(configs, out)  # independent slots
        out[i] = predict_torque(model, configs[i])
    end
    return out
end
predict_torque(f::SLCEFit, data) = predict_torque(SLCEModel(f), data)

# --- joint (spin + displacement) prediction ---------------------------------------

# Shared validation for the joint predict forms: unit spin columns and a finite
# 3 × n_atoms displacement field. Works on pure-spin models too (u is then inert —
# the joint kernels reduce to the spin-only values).
function _validate_config_pair(model::SLCEModel, e::AbstractMatrix{<:Real},
                               u::AbstractMatrix{<:Real})
    nat = n_atoms(model.basis.crystal)
    if _basis_has_spin(model.basis)
        _validate_config(e, nat)
    else
        # A lattice-only model reads no spin factor, so there is no unit-vector
        # postcondition to protect — only the shape has to line up. This is what lets
        # a caller pass `nothing` (below) or an all-zero placeholder without the
        # package pretending the placeholder is a magnetic state.
        size(e) == (3, nat) ||
            throw(DimensionMismatch("spin configuration is $(size(e, 1)) × " *
                                    "$(size(e, 2)), model expects 3 × $nat"))
    end
    size(u) == (3, nat) ||
        throw(DimensionMismatch("displacement field u is $(size(u, 1)) × " *
                                "$(size(u, 2)), model expects 3 × $nat"))
    all(isfinite, u) ||
        throw(ArgumentError("displacement field u contains non-finite entries"))
    return nat
end

# `nothing` in the spin slot: the lattice-only caller has no magnetic state, and the
# alternative — inventing one — is how a placeholder becomes a fabricated ferromagnet
# the moment the model does carry spin content. Legal exactly when there is no spin
# factor to evaluate; the predicate is `_basis_has_spin`, never `is_soc_free`.
function _no_spins(model::SLCEModel)::Matrix{Float64}
    _basis_has_spin(model.basis) && throw(ArgumentError(
        "the spin configuration is required: this model's basis carries spin " *
        "content, so the prediction depends on the magnetic state. `nothing` is " *
        "accepted only for a lattice-only model."))
    return zeros(Float64, 3, n_atoms(model.basis.crystal))
end

predict_energy(model::SLCEModel, ::Nothing, u::AbstractMatrix{<:Real})::Float64 =
    predict_energy(model, _no_spins(model), u)

function predict_energy(model::SLCEModel, e::AbstractMatrix{<:Real},
                        u::AbstractMatrix{<:Real})::Float64
    _validate_config_pair(model, e, u)
    salcs = model.basis.salc_basis.salcs
    val = model.j0
    scratch = SALCScratch()                 # reused workspace (dnPl + harmonic tables)
    @inbounds for k in eachindex(model.jphi)
        val += model.jphi[k] * evaluate_salc(salcs[k], e, u, scratch)
    end
    return val
end

function predict_torque(model::SLCEModel, e::AbstractMatrix{<:Real},
                        u::AbstractMatrix{<:Real})::Matrix{Float64}
    nat = _validate_config_pair(model, e, u)
    Ge, _ = _accumulate_joint_grads(model, e, u, nat)
    T = Matrix{Float64}(undef, 3, nat)
    @inbounds for a = 1:nat
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        ga = SVector{3,Float64}(Ge[1, a], Ge[2, a], Ge[3, a])
        t = cross(ga, ea)                # τ = ∇E × e = −e × ∇E  (physical / LL torque)
        T[1, a] = t[1]
        T[2, a] = t[2]
        T[3, a] = t[3]
    end
    return T
end

"""
    predict_force(model, e, u) -> Matrix{Float64}

Predict the per-atom force `f_a = −∂E/∂u_a` (eV/Å; the Euclidean displacement
gradient, sign pinned in the design record §6 and [`TrainingDatum`](@ref)) of the
joint configuration `(e, u)`, returned `3 × n_atoms`. The force is the analytic
derivative of the same surface [`predict_energy`](@ref)`(model, e, u)` evaluates, so
the two are consistent by construction. On a pure-spin model (or on atoms no SALC
displacement slot reads) the force is exactly zero — the energy does not depend on
those displacements.
"""
predict_force(model::SLCEModel, ::Nothing,
              u::AbstractMatrix{<:Real})::Matrix{Float64} =
    predict_force(model, _no_spins(model), u)

function predict_force(model::SLCEModel, e::AbstractMatrix{<:Real},
                       u::AbstractMatrix{<:Real})::Matrix{Float64}
    nat = _validate_config_pair(model, e, u)
    _, Gu = _accumulate_joint_grads(model, e, u, nat)
    Gu .= .-Gu                           # f = −∂E/∂u
    return Gu
end
predict_force(f::SLCEFit, e, u) = predict_force(SLCEModel(f), e, u)

predict_energy(f::SLCEFit, e, u) = predict_energy(SLCEModel(f), e, u)
predict_torque(f::SLCEFit, e, u) = predict_torque(SLCEModel(f), e, u)

# Coefficient-weighted joint gradient accumulation shared by the derivative
# predictors (one pass fills both buffers).
function _accumulate_joint_grads(model::SLCEModel, e::AbstractMatrix{<:Real},
                                 u::AbstractMatrix{<:Real}, nat::Int)
    salcs = model.basis.salc_basis.salcs
    Ge = zeros(Float64, 3, nat)
    Gu = zeros(Float64, 3, nat)
    scratch = SALCScratch()                 # reused workspace (dnPl + harmonic tables)
    @inbounds for k in eachindex(model.jphi)
        accumulate_grad!(Ge, Gu, salcs[k], e, u, model.jphi[k], scratch)
    end
    return Ge, Gu
end
