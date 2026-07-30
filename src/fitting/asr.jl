# ASR (acoustic sum rule): translation-invariance of the joint energy surface as
# exact linear equality constraints A·β = 0 on the SALC coefficients, realized as
# a null-space reparameterization β = Z·γ (design record §6 + the 2026-07-26
# amendments). This file builds the constraint machinery; `_assemble_problem`
# (fitting/fit.jl) applies it.
#
# The builder expands D·Φ_j (D = Σ_a ∇_{u_a}, the translation generator) of every
# displacement-decorated SALC into ONE common unsymmetrized monomial basis:
# spin factors are opaque symbols (atom, l, m) — D never touches them — and
# displacement factors |u|^{2k} R_{l,m}(u) are expanded to exact monomials by
# `SolidHarmonics.solid_harmonic_poly` (which reruns the evaluator's own
# recurrences over coefficient dictionaries; the random-point agreement gate in
# the test suite fences the two). A is expressed in DESIGN-COLUMN coordinates:
# each column carries the same `(4π)^{n_spin/2}` scale the evaluator applies, so
# `A·β` is the literal monomial expansion of Σ_a ∂E/∂u_a of the fitted model.
# `disp_scale ≠ 1` is still rejected at the BasisSpec boundary; the assertion
# below keeps this builder honest the day that guard lifts.

# Row identity of the common monomial basis: one Cartesian generator component
# `axis` ∈ 1:3, the spin symbol multiset, and the per-atom displacement monomial
# exponents. Atoms within one member are distinct (asserted), so each atom carries
# at most one spin symbol and one displacement monomial — the product basis of
# independent per-site function families, hence linearly independent keys.
struct _ASRRowKey
    axis::Int
    spins::Vector{NTuple{3,Int}}   # sorted (atom, l, m)
    disps::Vector{NTuple{4,Int}}   # sorted (atom, ex, ey, ez); zero exponents kept out
end
Base.hash(k::_ASRRowKey, h::UInt) = hash(k.disps, hash(k.spins, hash(k.axis, h)))
Base.:(==)(a::_ASRRowKey, b::_ASRRowKey) =
    a.axis == b.axis && a.spins == b.spins && a.disps == b.disps
Base.isless(a::_ASRRowKey, b::_ASRRowKey) =
    (a.axis, a.spins, a.disps) < (b.axis, b.spins, b.disps)

"""
    _asr_matrix(basis::SLCEBasis) -> Matrix{Float64}

The ASR constraint matrix `A` (rows = common monomials of `Σ_a ∂E/∂u_a`, columns
= SALC coefficients in design order): `A·β = 0` ⇔ the model energy is invariant
under every rigid translation `u_a → u_a + t`. Pure-spin columns are identically
zero. Rows are sorted by their monomial key for determinism and NOT yet
normalized — [`_asr_nullspace`](@ref) row-normalizes before the rank decision.
"""
function _asr_matrix(basis::SLCEBasis)::Matrix{Float64}
    basis.spec.disp_scale == 1.0 ||
        throw(ArgumentError("the ASR builder predates disp_scale ≠ 1 support — " *
                            "its monomial expansion must be rescaled per degree " *
                            "when that guard lifts (design record §6 amendment 3)"))
    ss = salcs(basis)
    p = length(ss)
    polycache = Dict{NTuple{3,Int},SolidHarmonics._Poly}()
    diffcache = Dict{NTuple{4,Int},SolidHarmonics._Poly}()
    rows = Dict{_ASRRowKey,Dict{Int,Float64}}()
    for (j, s) in enumerate(ss)
        any(has_disp, s.key.decors) || continue
        # The evaluator's column scale, carried so `A` lives in the same design
        # coordinates the fit solves in. It does not actually move the null space:
        # `_ASRRowKey` keys on the spin monomial itself, so every column a given row
        # touches carries the identical spin factor — hence the identical
        # `n_spin` — and a factor constant along a row is divided out again by the
        # relative row normalization below. Keep it anyway: it is what makes the
        # matrix's ENTRIES comparable to the design's (the symbolic-vs-numerical rank
        # gate and any future per-degree `disp_scale` rescaling both read them), and
        # a row that ever couples mixed `n_spin` would need it for real.
        scale = (4π)^(count(has_spin, s.decors) / 2)
        for mem in s.members
            allunique(mem.atoms) ||
                throw(ArgumentError("ASR builder: member with repeated atoms " *
                                    "(an AllImages self-image cluster) — same-site " *
                                    "factor products need a Gaunt expansion the " *
                                    "builder does not implement; ASR on AllImages " *
                                    "joint bases is not supported"))
            for t in mem.terms
                _asr_accumulate_term!(rows, j, scale, t, mem.atoms, polycache,
                                      diffcache)
            end
        end
    end
    keyorder = sort!(collect(keys(rows)))
    A = zeros(Float64, length(keyorder), p)
    for (r, key) in enumerate(keyorder)
        for (j, v) in rows[key]
            A[r, j] = v
        end
    end
    isempty(A) && return A
    # Cancellation residue must become EXACT zeros, not stay as ~1e-16 relative
    # entries: `refit` subselects columns, and a row whose real columns were all
    # dropped would otherwise get its residue entry blown up to a full-strength,
    # BLAS-rounding-determined constraint by the row normalization (review
    # blocker). Prune per row, then drop rows that are pure residue — both cuts
    # RELATIVE (per-row / global max), so a uniformly scaled A is not silently
    # emptied.
    gmax = maximum(abs, A)
    keep = Int[]
    for r in axes(A, 1)
        rmax = maximum(abs, @view A[r, :])
        for j in axes(A, 2)
            abs(A[r, j]) <= 1e-12 * rmax && (A[r, j] = 0.0)
        end
        rmax > 1e-12 * gmax && push!(keep, r)
    end
    return A[keep, :]
end

# One SALC term's contribution to the translation-generator monomial expansion.
function _asr_accumulate_term!(rows::Dict{_ASRRowKey,Dict{Int,Float64}}, j::Int,
                               scale::Float64, t::SALCTerm, atoms::Vector{Int},
                               polycache::Dict{NTuple{3,Int},SolidHarmonics._Poly},
                               diffcache::Dict{NTuple{4,Int},SolidHarmonics._Poly})
    slots = t.slots
    D = length(slots)
    spin_ids = Int[i for i = 1:D if slots[i].factor.channel == SPIN]
    disp_ids = Int[i for i = 1:D if slots[i].factor.channel != SPIN]
    isempty(disp_ids) && return nothing
    folded = t.folded
    for idx in CartesianIndices(folded)
        w = folded[idx]
        w == 0.0 && continue
        spins = NTuple{3,Int}[(atoms[slots[i].site], slots[i].factor.l,
                               idx[i] - slots[i].factor.l - 1) for i in spin_ids]
        sort!(spins)
        # per disp slot: its (atom, poly) at this m index
        dslots = Vector{Tuple{Int,SolidHarmonics._Poly}}(undef, length(disp_ids))
        for (n, i) in enumerate(disp_ids)
            f = slots[i].factor
            m = idx[i] - f.l - 1
            poly = get!(polycache, (f.k, f.l, m)) do
                SolidHarmonics.solid_harmonic_poly(f.k, f.l, m)
            end
            dslots[n] = (atoms[slots[i].site], poly)
        end
        # differentiate each disp slot in turn, product-expand the rest
        for (n, i) in enumerate(disp_ids)
            f = slots[i].factor
            m = idx[i] - f.l - 1
            for axis = 1:3
                dp = get!(diffcache, (f.k, f.l, m, axis)) do
                    SolidHarmonics._poly_diff(polycache[(f.k, f.l, m)], axis)
                end
                isempty(dp) && continue
                _asr_expand_product!(rows, j, w * scale, axis, spins, dslots, n,
                                     dp)
            end
        end
    end
    return nothing
end

# Accumulate w · dp(slot n) · Π_{o ≠ n} poly_o over all monomial combinations.
function _asr_expand_product!(rows::Dict{_ASRRowKey,Dict{Int,Float64}}, j::Int,
                              w::Float64, axis::Int,
                              spins::Vector{NTuple{3,Int}},
                              dslots::Vector{Tuple{Int,SolidHarmonics._Poly}},
                              n::Int, dp::SolidHarmonics._Poly)
    others = [dslots[o] for o in eachindex(dslots) if o != n]
    atom_n = dslots[n][1]
    # iterate the cartesian product of monomial choices
    function rec(o::Int, coeff::Float64, acc::Vector{NTuple{4,Int}})
        if o > length(others)
            key = _ASRRowKey(axis, spins, sort(acc; by = first))
            row = get!(() -> Dict{Int,Float64}(), rows, key)
            row[j] = get(row, j, 0.0) + coeff
            return nothing
        end
        (atom_o, poly_o) = others[o]
        for (mono, v) in poly_o
            if mono == (0, 0, 0)
                rec(o + 1, coeff * v, acc)
            else
                push!(acc, (atom_o, mono[1], mono[2], mono[3]))
                rec(o + 1, coeff * v, acc)
                pop!(acc)
            end
        end
        return nothing
    end
    for (mono, v) in dp
        if mono == (0, 0, 0)
            rec(1, w * v, NTuple{4,Int}[])
        else
            rec(1, w * v, NTuple{4,Int}[(atom_n, mono[1], mono[2], mono[3])])
        end
    end
    return nothing
end

# Rank / null-space policy (design record §6 amendment 3): row-normalize, split
# into exact connected components of the column–row bipartite graph, decide rank
# per component by SVD with cut σ ≤ RTOL_ZERO·σ_max, and REFUSE ambiguous spectra
# (any σ inside the forbidden band). Z is orthonormal per component, identity on
# untouched (pure-spin) columns.
const _ASR_RTOL_ZERO = 1e-10
const _ASR_BAND = (1e-12, 1e-8)
# A column is STRUCTURALLY ZEROED by a constraint when its row of the null-space basis
# vanishes: no feasible β can carry it. `Z` is orthonormal, so the row norm is on the
# scale of 1 and an absolute cut is the right form. One definition, four readers —
# `build_asr`'s basis-level warning, `refit`'s movable-column and split-coupled-set
# rules (fit.jl), and `group_costs`' structural discount (selection.jl).
const _ASR_DEAD_ROW = 1e-12

# β = beta_p + Z·γ, with the numerically dead rows of `Z` snapped back to the
# particular solution. `Z`'s row for a column no feasible model can carry is a
# NUMERICALLY zero row (~1e-16 out of the SVD), not a structurally zero one, so the
# plain product leaves ~1e-15 on a direction the constraint forbids. That junk is not
# cosmetic — two consumers test coefficients EXACTLY: `select_support`'s alive rule
# (`!= 0.0`) would report the group alive and charge its Monte-Carlo cost, and
# SLCEMonteCarlo's term prune (`hamiltonian.jl`, `t.coef != 0.0`) would buy it a full
# set of site programs. The correct value is `beta_p[j]` alone: exactly 0.0 under a
# homogeneous reparameterization, and the particular-solution value on an affine stage,
# where a free column the constraint zeroes may still legitimately carry one. Used by
# both lifts — `fit`'s basis-level one and `refit`'s per-support sub-stage.
function _lift_gamma(rep::ASRReparam, gamma::Vector{Float64})::Vector{Float64}
    beta = rep.beta_p .+ rep.Z * gamma
    @inbounds for j in axes(rep.Z, 1)
        norm(@view rep.Z[j, :]) < _ASR_DEAD_ROW && (beta[j] = rep.beta_p[j])
    end
    return beta
end

"""
    _asr_nullspace(A) -> (Z::Matrix{Float64}, rank::Int)

Orthonormal null-space basis of the (row-normalized) constraint matrix `A` and
its rank. Deterministic per platform; the basis GAUGE is factorization-dependent
— never persist `Z` or `γ`, only `β = Z·γ` (gauge-invariant). Throws on an
ambiguous singular spectrum (forbidden band) instead of guessing the rank.
"""
function _asr_nullspace(A::Matrix{Float64})
    p = size(A, 2)
    # (Near-)zero rows impose nothing — they cannot occur in a fresh
    # `_asr_matrix` (residue-pruned) but DO occur under column subselection
    # (`refit`'s `A[:, support]`: a constraint whose every column was dropped).
    # The drop is RELATIVE to the largest row so a caller-supplied unpruned
    # matrix cannot promote residue to a unit-norm constraint.
    rnorms = [norm(@view A[r, :]) for r in axes(A, 1)]
    rmax = isempty(rnorms) ? 0.0 : maximum(rnorms)
    nzrows = [r for r in axes(A, 1) if rnorms[r] > 1e-12 * rmax]
    m = length(nzrows)
    m == 0 && return Matrix{Float64}(I, p, p), 0
    A = A[nzrows, :]
    An = A ./ rnorms[nzrows]
    # connected components over columns (union-find via shared rows)
    parent = collect(1:p)
    find(x) = (while parent[x] != x; parent[x] = parent[parent[x]]; x = parent[x]; end; x)
    for r = 1:m
        first_col = 0
        for c = 1:p
            An[r, c] == 0.0 && continue
            if first_col == 0
                first_col = c
            else
                parent[find(c)] = find(first_col)
            end
        end
    end
    comps = Dict{Int,Vector{Int}}()
    for c = 1:p
        push!(get!(() -> Int[], comps, find(c)), c)
    end
    # deterministic γ order: walk columns ascending; free columns → identity;
    # first column of a constrained component → that component's null block
    zcols = Vector{Vector{Float64}}()
    total_rank = 0
    seen = Set{Int}()
    for c = 1:p
        root = find(c)
        root in seen && continue
        push!(seen, root)
        cols = sort(comps[root])
        rws = [r for r = 1:m if any(!=(0.0), @view An[r, cols])]
        if isempty(rws)
            for cc in cols                 # untouched columns: identity block
                e = zeros(p)
                e[cc] = 1.0
                push!(zcols, e)
            end
            continue
        end
        sub = An[rws, cols]
        F = svd(sub; full = true)
        smax = F.S[1]
        for σ in F.S
            if _ASR_BAND[1] * smax <= σ <= _ASR_BAND[2] * smax
                throw(ArgumentError("ASR null space: ambiguous singular value " *
                                    "σ/σ_max = $(σ / smax) inside the forbidden " *
                                    "band $(_ASR_BAND) — refusing to guess the " *
                                    "constraint rank (component of " *
                                    "$(length(cols)) columns)"))
            end
        end
        r = count(>(_ASR_RTOL_ZERO * smax), F.S)
        total_rank += r
        for k = (r + 1):length(cols)
            e = zeros(p)
            e[cols] .= F.V[:, k]
            push!(zcols, e)
        end
    end
    Z = isempty(zcols) ? zeros(Float64, p, 0) : reduce(hcat, zcols)
    return Z, total_rank
end

"""
    build_asr(basis::SLCEBasis; translation = true, warn = true) -> Union{Nothing,ASRReparam}

Build the reparameterization every fit on `basis` solves in. It encodes two exact
facts about the basis, kept in separate fields so neither is mistaken for the other:

  * **translation invariance** — the row-normalized ASR constraint matrix `A` and its
    orthonormal null-space basis `Z`, so `β = Z·γ` satisfies `A·β = 0` by
    construction. Skipped entirely with `translation = false` (an ablation path).
  * **the unresolvable-column freeze** — the columns
    [`unresolvable_columns`](@ref) names are excluded from `free`, hence held at
    exactly `0`. They are identically zero on every configuration this reference cell
    can express, so any value a fit returns for them is an estimator artifact that a
    Monte-Carlo supercell or a `q ≠ 0` dynamical matrix would amplify into physics the
    data never constrained. This part is applied whatever `translation` says.

Returns `nothing` only when there is nothing to encode: a basis with no displacement
columns and no unresolvable ones (the structural fast path, bitwise identical to an
unconstrained fit). Called once at `SLCEDataset` construction and stored on
`dataset.asr` (the `force_cols` discipline).

`rank` counts the constraints on the *free* columns, so a basis with no unresolvable
columns reparameterizes exactly as it did before the freeze existed.

`warn = false` silences the two diagnostics about the *basis* (no
translation-invariant displacement content at all; individual structurally zeroed
columns). Pass it from any caller that is **re-deriving** the constraint rather than
constructing it — [`asr_residual`](@ref) and every derived-quantity path that gates on
it do, because the truncation has not changed since the construction that already said
so, and one `asr_residual` per output turns one true statement into a screenful. The
refusals (a broken symbolic expansion, a forbidden band) are NOT silenced by it: they
say the answer would be wrong, not that the truncation is narrow.

Both diagnostics also carry `maxlog = 1`, so a script that builds a *grid* of
structurally identical bases (a [`StrainedModels`](@ref) volume scan) states the fact
once rather than once per point. The cost is deliberate and bounded: a session that later
builds a genuinely different narrow basis will not repeat the advice, and
[`identifiability`](@ref) / [`asr_residual`](@ref) remain the on-demand answers.
"""
function build_asr(basis::SLCEBasis; translation::Bool = true,
                   warn::Bool = true)::Union{Nothing,ASRReparam}
    p = n_salcs(basis)
    hasdisp = any(s -> any(has_disp, s.key.decors), salcs(basis))
    frozen = unresolvable_columns(basis)
    (isempty(frozen) && !(translation && hasdisp)) && return nothing
    free = isempty(frozen) ? collect(1:p) : setdiff(1:p, frozen)
    isempty(frozen) || !warn ||
        @warn "this reference cell cannot resolve $(length(frozen)) of its $p " *
              "columns: they are identically zero on every configuration it can " *
              "express, so the fit holds them at exactly zero. They are " *
              "UNIDENTIFIABLE, not physically zero — the same basis functions are " *
              "nonzero under a uniform strain and in a Monte-Carlo supercell, where " *
              "a fitted value would become physics the data never constrained. The " *
              "cause is a Wigner-Seitz boundary tie; to determine these " *
              "coefficients, describe the crystal with a cell in which the offending " *
              "pair's minimum image is unique (`unresolvable_columns` for the " *
              "columns, the Periodic resolvability chapter for the remedy)" columns =
            first(frozen, 10) maxlog = 1
    # Freeze-only: no constraint rows at all (a pure-spin basis, or `translation =
    # false`). `_stage_reparam`'s `A === nothing` path gives the plain selection
    # matrix of the free columns — orthonormal, so every estimator's γ-space contract
    # still holds verbatim.
    (translation && hasdisp) || return _stage_reparam(basis, free, zeros(p), nothing)
    A = _asr_matrix(basis)
    m = size(A, 1)
    # store the row-normalized form (the residual gate's conditioning);
    # `_asr_nullspace` normalizes internally (idempotent on this input)
    An = m == 0 ? A : A ./ [norm(@view A[r, :]) for r = 1:m]
    rep = _stage_reparam(basis, free, zeros(p), An)
    Z, rank = rep.Z, rep.rank
    ndisp = length(intersect(_disp_active_cols(basis), free))
    # rank 0 on a displacement-active basis is physically impossible (any single
    # displacement factor has a nonzero translation derivative) — it means the
    # expansion silently produced nothing, and every downstream diagnostic would
    # report success while the ASR is a no-op. Refuse.
    rank == 0 && ndisp > 0 &&
        throw(ArgumentError("ASR builder produced no constraints on a " *
                            "displacement-active basis — the symbolic expansion " *
                            "is broken (or A was scaled below the residue cut)"))
    # A truncation may admit NO translation-invariant displacement content at all
    # (e.g. pair (1,1) splits without their on-site degree-2 partners under
    # pmax = 1): then ASR constrains every displacement coefficient to zero.
    # Correct — the invariant subspace of that span is empty — but it means the
    # sector choice cannot express any lattice coupling, so say it loudly.
    if rank == ndisp && ndisp > 0
        warn && @warn "ASR: the basis admits no translation-invariant displacement " *
              "content — every displacement-active coefficient is constrained " *
              "to zero (the truncation lacks the on-site/pair partners needed " *
              "to form difference invariants; widen pmax or the displacement " *
              "sectors)" n_disp_columns = ndisp maxlog = 1
    else
        # BASIS-level structurally zero columns: coefficients no translation-
        # invariant model can carry, for every support. Reported here, once —
        # `refit` warns only about columns its support ADDITIONALLY kills.
        # Only the FREE columns: a frozen (unresolvable) column also has an all-zero
        # `Z` row, and blaming the sum rule for it would name the wrong cause and the
        # wrong remedy — that one is reported by the freeze diagnostic above.
        dead = [j for j in free if norm(@view Z[j, :]) < _ASR_DEAD_ROW]
        (isempty(dead) || !warn) ||
            @warn "ASR: some columns cannot appear in any translation-invariant " *
                  "model (structurally zeroed by the constraint for every support). " *
                  "The sum rule holds separately in each spin sector, so a column " *
                  "survives only if the truncation carries a partner with the SAME " *
                  "spin invariant: for a spin-free displacement channel that means " *
                  "further pair orbits (widen the cutoff), for a spin-dressed one a " *
                  "term dressing the same spin factor differently — e.g. a displaced " *
                  "ligand. A PAIR channel whose exchange parity (the parity of its spin " *
                  "factor times (-1)^Lf) comes out EVEN can only couple to u_a + u_b: no " *
                  "extra pair orbit and no wider cutoff changes that parity, so its " *
                  "partner has to come from another sector — a displaced third atom. " *
                  "Either way the coefficient is excluded by the sum rule, not by crystal " *
                  "symmetry: the basis function itself is generally nonzero" columns = dead maxlog = 1
    end
    return rep
end

"""
    asr_residual(model::SLCEModel) -> Float64

Relative ASR residual `‖A·β‖ / (‖A‖·‖β‖)` (Frobenius norm on `A`) of a model's
coefficients — `0.0` for a pure-spin basis (no constraints) or an exactly
translation-invariant coefficient vector. This is the public verifier: nothing
about ASR is persisted (the fingerprint precedent — recompute, never trust), so
physical consumers (force-constant deliverables, Monte-Carlo ingest of joint
models) gate on this value instead of a stored flag. A hand-built violating
model is legal; it simply reports a large residual.
"""
function asr_residual(model::SLCEModel)::Float64
    rep = build_asr(model.basis; warn = false)   # re-derivation: construction already spoke
    rep === nothing && return 0.0
    return _asr_residual(rep, model.jphi)
end

function _asr_residual(rep::ASRReparam, beta::Vector{Float64})::Float64
    isempty(rep.A) && return 0.0
    denom = norm(rep.A) * norm(beta)
    denom == 0.0 && return 0.0
    return norm(rep.A * beta) / denom
end
