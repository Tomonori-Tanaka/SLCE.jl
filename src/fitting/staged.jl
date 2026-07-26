# Staged ("hierarchical") fitting: which columns a stage FITS (`sector_mask`) and
# which coefficients it holds FIXED (`frozen`), realized as one affine
# reparameterization `β = beta_p + Z·γ` that the ordinary assembly/solve path then
# consumes unchanged:
#
#   * `Z` has a zero row on every frozen column, so γ cannot move it, and the
#     stage's free columns carry the ASR null space restricted to them;
#   * `beta_p` carries the frozen values — plus, when the frozen part violates the
#     ASR on its own, a particular solution of `A_free·β_free = −A_frozen·β_frozen`.
#
# The staging axis (`sector_mask`/`frozen`) is deliberately NOT the truncation axis
# (`Sector(soc = …)`, which defines the model's support): a mask chooses what a
# STAGE fits, the spec chooses what the model can express. The two share the
# `soc`-free predicate (`is_soc_free`) so they cannot drift apart.

const _SECTOR_SELECTORS = (:all, :spin, :lattice, :coupled, :soc_free, :soc)

"""
    sector_columns(basis, selector) -> Vector{Int}

The design-matrix columns a `selector` picks out, ascending. The selector is a
`Symbol`, a collection of them (their **union**), or an explicit column list
(`AbstractVector{<:Integer}`, validated) / mask (`AbstractVector{Bool}`).

| selector | columns |
|:--|:--|
| `:all` | every column |
| `:spin` | pure-spin SALCs (no displacement factor anywhere) |
| `:lattice` | spin-free SALCs (displacement only — force constants) |
| `:coupled` | SALCs carrying both channels |
| `:soc_free` | `L_S = 0` — the channel a SOC-less calculation can produce |
| `:soc` | `L_S ≠ 0` — the spin-orbit-only channel |

`:spin` / `:lattice` / `:coupled` partition the basis; `:soc_free` / `:soc`
partition it a second, crosscutting way. `:soc_free` uses the same predicate as
the basis-level `Sector(soc = false)` truncation ([`SLCE.is_soc_free`](@ref)), so
the staging axis and the truncation axis cannot drift apart — but they stay
different axes: masking `:soc_free` on a `soc = true` basis freezes the `L_S ≠ 0`
columns of the model it can still express, it does not rebuild the model.

Used by [`fit`](@ref)'s `sector_mask`; public (unexported) for inspecting a
staging plan before running it.
"""
function sector_columns(basis::SLCEBasis, sel::Symbol)::Vector{Int}
    sel in _SECTOR_SELECTORS ||
        throw(ArgumentError("unknown sector selector :$sel; expected one of " *
                            "$(_SECTOR_SELECTORS) (or an explicit column list)"))
    ks = basis.salc_basis.keys
    sel === :all && return collect(eachindex(ks))
    pred = if sel === :spin
        k -> !any(has_disp, k.decors)
    elseif sel === :lattice
        k -> !any(has_spin, k.decors)
    elseif sel === :coupled
        k -> any(has_spin, k.decors) && any(has_disp, k.decors)
    elseif sel === :soc_free
        is_soc_free
    else                                            # :soc
        k -> !is_soc_free(k)
    end
    return findall(pred, ks)
end

function sector_columns(basis::SLCEBasis, sels)::Vector{Int}
    if eltype(sels) === Bool                        # a per-column mask
        length(sels) == n_salcs(basis) || throw(DimensionMismatch(
            "sector mask has $(length(sels)) entries, basis has $(n_salcs(basis)) " *
            "columns"))
        return findall(sels)
    elseif eltype(sels) <: Integer                  # an explicit column list
        cols = sort(unique(Int.(sels)))
        (isempty(cols) || (first(cols) >= 1 && last(cols) <= n_salcs(basis))) ||
            throw(ArgumentError("sector_mask column indices must lie in " *
                                "1:$(n_salcs(basis)); got extremes " *
                                "$(first(cols)) and $(last(cols))"))
        return cols
    end
    # a collection of selectors: the union
    cols = Int[]
    for s in sels
        append!(cols, sector_columns(basis, s))
    end
    return sort(unique(cols))
end

# Feasibility tolerance for the affine constraint `A_free·β_free = −A_frozen·β_frozen`
# (relative to ‖rhs‖ on the row-normalized A). Loose enough to accept a frozen
# vector that satisfies the ASR only to fit precision, tight enough that a genuine
# straddle — a constraint row the free columns cannot balance — is refused.
const _STAGE_FEAS_RTOL = 1e-8

# When the frozen part counts as ASR-satisfying: the SAME relative measure
# `asr_residual` reports (‖A·β‖ / ‖A‖‖β‖), not `A·β == 0`. A stage fitted under the
# ASR leaves a residual at roundoff (~1e-16), and taking that for a violation would
# send every chained stage down the affine path — where a roundoff-sized right-hand
# side is generically OUT of range(A_free) and would be refused as infeasible.
# What skipping the correction costs is bounded: the uncorrected violation is
# `‖A·β_frozen‖ ≤ rtol·‖A‖·‖β_frozen‖` and `‖β_final‖ ≥ ‖β_frozen‖` is not
# guaranteed, so state the bar this buys — the final model's `asr_residual` stays
# at or below this cut, which is looser than the 1e-12/1e-13 bar the invariance
# gates use. A frozen part at 1e-11 therefore passes here and is NOT corrected.
const _STAGE_HOMOGENEOUS_RTOL = 1e-10

# Build the stage reparameterization. `A === nothing` is the unconstrained (or
# pure-spin) case: `Z` is then the plain selection matrix of the free columns —
# orthonormal, so every estimator's γ-space contract still holds verbatim.
function _stage_reparam(basis::SLCEBasis, free::Vector{Int},
                        beta_frozen::Vector{Float64},
                        A::Union{Nothing,Matrix{Float64}};
                        remedy::AbstractString = "widen `sector_mask` to include " *
                                                 "the coupled columns")::ASRReparam
    p = n_salcs(basis)
    length(beta_frozen) == p ||
        throw(DimensionMismatch("frozen coefficient vector has " *
                                "$(length(beta_frozen)) entries, basis has $p"))
    if A === nothing
        Z = zeros(Float64, p, length(free))
        for (k, j) in enumerate(free)
            Z[j, k] = 1.0
        end
        return ASRReparam(zeros(Float64, 0, p), Z, beta_frozen, 0, free)
    end
    Z_S, rank_S = _asr_nullspace(A[:, free])
    beta_p = copy(beta_frozen)
    rhs = -(A * beta_frozen)
    scale = norm(A) * norm(beta_frozen)
    if norm(rhs) > _STAGE_HOMOGENEOUS_RTOL * scale
        # The frozen coefficients do not satisfy the ASR by themselves, so the
        # stage's constraint is AFFINE. (It is homogeneous — `rhs = 0`, the fast
        # path — whenever the frozen part was itself fitted under the ASR: that is
        # the staged-fit theorem, and the reason a chain of stages stays exact.)
        A_S = A[:, free]
        x = _stage_particular(A_S, rhs)
        bad = A_S * x - rhs
        resid = norm(bad)
        # Absolute floor next to the relative test: at κ(A_S) ≳ 1e8 — which the
        # null space's own forbidden band still admits — a purely numerical
        # residual can exceed `rtol·‖rhs‖`, and refusing then would be a false
        # alarm rather than a straddling constraint.
        tol = max(_STAGE_FEAS_RTOL * norm(rhs), 8 * eps(Float64) * norm(A_S) * norm(x))
        if resid > tol
            rows = sortperm(abs.(bad); rev = true)
            worst = [r for r in rows[1:min(5, length(rows))]
                     if abs(bad[r]) > tol / sqrt(length(bad))]
            throw(ArgumentError(
                "staged fit is infeasible: the frozen coefficients violate the " *
                "ASR in a way the free columns cannot balance (relative residual " *
                "$(round(resid / norm(rhs); sigdigits = 3)) on constraint rows " *
                "$(worst)). Those rows straddle the fitted set — they couple " *
                "frozen and free columns — so either $remedy, or freeze a model " *
                "that satisfies the ASR (`asr_residual(model)`; a stage fitted " *
                "under the ASR always does)"))
        end
        beta_p[free] .+= x
    end
    Z = zeros(Float64, p, size(Z_S, 2))
    Z[free, :] .= Z_S
    return ASRReparam(A, Z, beta_p, rank_S, free)
end

# Particular solution of `A_S·x = rhs` under the SAME rank policy the null space
# uses (`_ASR_RTOL_ZERO`). `A_S \\ rhs` would decide rank at LAPACK's own
# `min(m,n)·eps` instead: a direction the null space calls null but the solve calls
# invertible gets amplified by 1/σ (up to ~1e13 here), and the feasibility test
# then PASSES while `β = beta_p + Z·γ` loses those digits to cancellation. Dropping
# the below-cut directions instead leaves their contribution in the residual, where
# the feasibility test can see it.
function _stage_particular(A_S::Matrix{Float64}, rhs::Vector{Float64})::Vector{Float64}
    F = svd(A_S)
    isempty(F.S) && return zeros(Float64, size(A_S, 2))
    cut = _ASR_RTOL_ZERO * F.S[1]
    x = zeros(Float64, size(A_S, 2))
    ut = F.U' * rhs
    for k in eachindex(F.S)
        F.S[k] > cut || continue
        @views x .+= (ut[k] / F.S[k]) .* F.V[:, k]
    end
    return x
end

# The frozen model's coefficients in THIS basis's column order, matched by
# `SALCKey` (never positionally — the reload contract). A key the target basis does
# not have is an error when it carries a nonzero coefficient: silently dropping a
# frozen interaction would change the model being staged.
function _frozen_coefficients(basis::SLCEBasis, model::SLCEModel)::Vector{Float64}
    ks = basis.salc_basis.keys
    pos = Dict(k => i for (i, k) in enumerate(ks))
    out = zeros(Float64, length(ks))
    # Key equality is not basis equality: `orbit_id` is a per-build enumeration
    # index, so two structurally different crystals can produce colliding keys.
    # The crystal is the cheap invariant that separates them and still admits the
    # legitimate workflow (a narrower spec on the SAME crystal).
    basis.crystal == model.basis.crystal || throw(ArgumentError(
        "the frozen model was built on a different crystal than this dataset's " *
        "basis — SALCKeys are only comparable within one crystal (`orbit_id` is a " *
        "per-build index), so matching them would freeze physically unrelated " *
        "coefficients"))
    orphans = 0
    vcut = 1e-12 * (isempty(model.jphi) ? 0.0 : norm(model.jphi))
    for (k, v) in zip(model.keys, model.jphi)
        i = get(pos, k, 0)
        if i == 0
            abs(v) <= vcut || (orphans += 1)     # relative cut, package convention
            continue
        end
        out[i] = v
    end
    orphans == 0 || throw(ArgumentError(
        "the frozen model carries $orphans nonzero coefficient(s) whose SALCKeys " *
        "are absent from this dataset's basis — the two bases disagree on the " *
        "model's support, so freezing would silently drop those interactions " *
        "(fit the stages against one basis, or check the basis fingerprints)"))
    return out
end
