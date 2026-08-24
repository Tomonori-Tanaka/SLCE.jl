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

Cancellation residue is pruned to exact zeros by [`_prune_residue!`](@ref); a caller
that must distinguish "the expansion deposited nothing" from "everything it deposited
cancelled" reads [`_asr_expansion`](@ref) directly.
"""
_asr_matrix(basis::SLCEBasis)::Matrix{Float64} = _prune_residue!(_asr_expansion(basis)...)

"""
    _asr_expansion(basis::SLCEBasis) -> (A, G)

The ASR expansion before the residue cut: `A[r, j]` as accumulated, and the GROSS
mass `G[r, j] = Σ |contributions|` that landed in it. `G[r, j] > 0` says the
expansion visited that entry at all; `|A[r, j]| / G[r, j]` says whether what it
deposited survived. Both halves are needed — the pruner judges each entry against its
own gross mass, and [`build_asr`](@ref)'s broken-expansion refusal must not fire on a
basis whose entries were all visited and all cancelled.
"""
function _asr_expansion(basis::SLCEBasis)
    basis.spec.disp_scale == 1.0 ||
        throw(ArgumentError("the ASR builder predates disp_scale ≠ 1 support — " *
                            "its monomial expansion must be rescaled per degree " *
                            "when that guard lifts (design record §6 amendment 3)"))
    ss = salcs(basis)
    p = length(ss)
    polycache = Dict{NTuple{3,Int},SolidHarmonics._Poly}()
    diffcache = Dict{NTuple{4,Int},SolidHarmonics._Poly}()
    # per entry: (net accumulated value, gross accumulated |value|)
    rows = Dict{_ASRRowKey,Dict{Int,Tuple{Float64,Float64}}}()
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
                throw(UnclassifiableBasis("ASR builder: member with repeated atoms " *
                                          "(an AllImages self-image cluster) — " *
                                          "same-site factor products need a Gaunt " *
                                          "expansion the builder does not implement; " *
                                          "ASR on AllImages joint bases is not " *
                                          "supported"))
            for t in mem.terms
                _asr_accumulate_term!(rows, j, scale, t, mem.atoms, polycache,
                                      diffcache)
            end
        end
    end
    keyorder = sort!(collect(keys(rows)))
    A = zeros(Float64, length(keyorder), p)
    G = zeros(Float64, length(keyorder), p)
    for (r, key) in enumerate(keyorder)
        for (j, (v, g)) in rows[key]
            A[r, j] = v
            G[r, j] = g
        end
    end
    return A, G
end

"""
    _prune_residue!(A, G) -> Matrix{Float64}

Snap cancellation residue in `A` to EXACT zeros and drop the rows left empty, judging
each entry against its own gross accumulation `G[r, j]` (see
[`_asr_expansion`](@ref)).

Exact zeros are the contract, not a cosmetic: `refit` subselects columns, and a row
whose real columns were all dropped would otherwise have its residue entry blown up to
a full-strength, BLAS-rounding-determined constraint by [`_asr_nullspace`](@ref)'s row
normalization.

The scale must be the entry's own gross mass — "did this entry cancel?", never "is this
entry small?". A cut taken against the matrix's own maximum (per row or global) is blind
to the case where EVERY entry cancels, which is exactly what a Wigner–Seitz-tied cell
produces: measured on a bcc spin × displacement basis whose `max|A|` is 3.6e-15, the old
global cut kept all of it and [`asr_residual`](@ref) reported **0.299** for a
hand-built model, i.e. a physical consumer's gate refusing a legal model over rounding
noise. Same constant, same reading as `unresolvable_columns`
(`_CANCELLATION_RTOL`, `basis/resolvability.jl`).
"""
function _prune_residue!(A::Matrix{Float64}, G::Matrix{Float64})::Matrix{Float64}
    size(A) == size(G) ||
        throw(DimensionMismatch("ASR expansion: value and gross matrices differ in " *
                                "size ($(size(A)) vs $(size(G)))"))
    @inbounds for j in axes(A, 2), r in axes(A, 1)
        abs(A[r, j]) <= _CANCELLATION_RTOL * G[r, j] && (A[r, j] = 0.0)
    end
    keep = [r for r in axes(A, 1) if any(!=(0.0), @view A[r, :])]
    return A[keep, :]
end

# One SALC term's contribution to the translation-generator monomial expansion.
function _asr_accumulate_term!(rows::Dict{_ASRRowKey,Dict{Int,Tuple{Float64,Float64}}},
                               j::Int,
                               scale::Float64, t::SALCTerm, atoms::Vector{Int},
                               polycache::Dict{NTuple{3,Int},SolidHarmonics._Poly},
                               diffcache::Dict{NTuple{4,Int},SolidHarmonics._Poly})
    slots = t.slots
    D = length(slots)
    spin_ids = Int[i for i = 1:D if slots[i].factor.channel == SPIN]
    disp_ids = Int[i for i = 1:D if slots[i].factor.channel != SPIN]
    isempty(disp_ids) && return nothing
    folded = t.folded
    for index in CartesianIndices(folded)
        w = folded[index]
        w == 0.0 && continue
        spins = NTuple{3,Int}[(atoms[slots[i].site], slots[i].factor.l,
                               index[i] - slots[i].factor.l - 1) for i in spin_ids]
        sort!(spins)
        # per disp slot: its (atom, poly) at this m index
        dslots = Vector{Tuple{Int,SolidHarmonics._Poly}}(undef, length(disp_ids))
        for (n, i) in enumerate(disp_ids)
            f = slots[i].factor
            m = index[i] - f.l - 1
            poly = get!(polycache, (f.k, f.l, m)) do
                SolidHarmonics.solid_harmonic_poly(f.k, f.l, m)
            end
            dslots[n] = (atoms[slots[i].site], poly)
        end
        # differentiate each disp slot in turn, product-expand the rest
        for (n, i) in enumerate(disp_ids)
            f = slots[i].factor
            m = index[i] - f.l - 1
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
function _asr_expand_product!(rows::Dict{_ASRRowKey,Dict{Int,Tuple{Float64,Float64}}},
                              j::Int, w::Float64, axis::Int,
                              spins::Vector{NTuple{3,Int}},
                              dslots::Vector{Tuple{Int,SolidHarmonics._Poly}},
                              n::Int, dp::SolidHarmonics._Poly)
    others = [dslots[o] for o in eachindex(dslots) if o != n]
    atom_n = dslots[n][1]
    # iterate the cartesian product of monomial choices
    function rec(o::Int, coeff::Float64, acc::Vector{NTuple{4,Int}})
        if o > length(others)
            key = _ASRRowKey(axis, spins, sort(acc; by = first))
            row = get!(() -> Dict{Int,Tuple{Float64,Float64}}(), rows, key)
            (net, gross) = get(row, j, (0.0, 0.0))
            row[j] = (net + coeff, gross + abs(coeff))
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
    can express, so any value a fit returns for them is an estimator artifact — and one
    a Monte-Carlo supercell (where the tie is resolved, so the function is nonzero) or a
    `q ≠ 0` dynamical matrix would amplify into physics the data never constrained. This
    part is applied whatever `translation` says, and on pure-spin bases too.

Returns `nothing` only when there is nothing to encode: a basis with no displacement
columns and no unresolvable ones (the structural fast path, bitwise identical to an
unconstrained fit). Called once at `SLCEDataset` construction and stored on
`dataset.asr` (the `force_cols` discipline).

`rank` counts the constraints on the *free* columns, so a basis with no unresolvable
columns reparameterizes exactly as it did before the freeze existed. When *every*
displacement column is frozen there is nothing left for the translation constraint to
act on: `free` comes back empty, `Z` has no columns, and neither the
no-invariant-content warning nor the broken-expansion refusal fires (both compare
`rank` against a free displacement count that is then zero). The fit degenerates
cleanly to `jphi ≡ 0` with the analytic intercept.

`rank == 0` beside free displacement columns is likewise legal, and only when the
expansion *deposited* monomials that then cancelled: the free columns carry difference
content only, so the sum rule constrains nothing and `Z` is the identity on them. What
is refused is an expansion that deposited nothing at all — that means the symbolic
machinery is broken, not that the basis is invariant.

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
    # An AllImages self-image basis cannot be classified at all. Proceeding with
    # NOTHING frozen is the honest reading of that (unknown, not none) and it keeps
    # `build_asr` usable as an ANALYSIS entry point on a hand-built or tiling-template
    # model. It is no longer a fitting route: `SLCEDataset` refuses such a basis at the
    # door, so nothing that reaches a solver comes through here. Said once, because it
    # withdraws a guarantee.
    split = try
        _unresolvable_split(basis)
    catch err
        err isa UnclassifiableBasis || rethrow()
        warn && @warn "cannot tell which columns this reference cell resolves — " *
                      "$(err.reason). Nothing is frozen, so a fit may return values " *
                      "for columns no data can determine" maxlog = 1
        (; columns = Int[], vanishing = Int[], undetermined = Int[],
         multisets = Vector{Int}[], residual_flat = 0)
    end
    frozen = split.columns
    (isempty(frozen) && !(translation && hasdisp)) && return nothing
    free = isempty(frozen) ? collect(1:p) : setdiff(1:p, frozen)
    # TWO reasons, TWO messages. Face (a) is "this column is the zero function"; face (b)
    # is "these couplings are real but only their sum is determined, and no split is
    # justified". Merging them would give the reader one count and the wrong remedy for
    # half of it — and face (b) additionally changes what a good fit looks like, since
    # dropping the determined sum leaves a residual on purpose.
    isempty(split.vanishing) || !warn ||
        @warn "this reference cell cannot resolve $(length(split.vanishing)) of its $p " *
              "columns: they are identically zero on every configuration it can " *
              "express, so the fit holds them at exactly zero. They are " *
              "UNIDENTIFIABLE, not physically zero: tiling this cell maps the tied " *
              "images onto DISTINCT atoms, so nothing cancels in a Monte-Carlo " *
              "supercell and a fitted value would become physics the data never " *
              "constrained there. (A uniform strain reveals some of them as well, but " *
              "never a pure-spin one — it has no displacement slot for a strain to " *
              "act on.) The " *
              "cause is a Wigner-Seitz boundary tie; to determine these " *
              "coefficients, describe the crystal with a cell in which the offending " *
              "pair's minimum image is unique (`unresolvable_columns` for the " *
              "columns, the Periodic resolvability chapter for the remedy)" columns =
            first(split.vanishing, 10) maxlog = 1
    isempty(split.undetermined) || !warn ||
        @warn "this reference cell reaches $(length(split.multisets)) atom group(s) " *
              "through MORE THAN ONE cluster orbit (a Wigner-Seitz boundary tie that " *
              "symmetry does not fuse), so those orbits span the SAME space of " *
              "functions of any configuration it can express and only the total of " *
              "their couplings is determined. Splitting it has no physical basis — the images " *
              "carry the same phase only at q = 0 — so the whole interaction is " *
              "DROPPED: $(length(split.undetermined)) column(s) are held at exactly " *
              "zero. The fit residual will not reach zero on data that contains them, " *
              "which is the intended signal. To fit these couplings, describe the " *
              "crystal with a cell in which that group's minimum image is unique " *
              "(never a wider cutoff — see the Periodic resolvability chapter)" atoms =
            first(split.multisets, 5) columns = first(split.undetermined, 10) maxlog = 1
    # Orbit granularity can in principle leave a flat combination behind (two orbits that
    # share nothing but are still dependent). Measured never to happen, but a silent
    # residue here is exactly the failure the freeze exists to end, so it is reported.
    split.residual_flat == 0 || !warn ||
        @warn "after dropping the tied orbits, $(split.residual_flat) flat " *
              "direction(s) remain in the structural expansion: some coefficient " *
              "combination is still undetermined on this cell. Run " *
              "`identifiability` on the fit and treat q ≠ 0 readouts as " *
              "unconstrained" maxlog = 1
    # Freeze-only: no constraint rows at all (a pure-spin basis, or `translation =
    # false`). `_stage_reparam`'s `A === nothing` path gives the plain selection
    # matrix of the free columns — orthonormal, so every estimator's γ-space contract
    # still holds verbatim.
    (translation && hasdisp) || return _stage_reparam(basis, free, zeros(p), nothing)
    A_gross, G = _asr_expansion(basis)
    # Whether the expansion deposited anything AT ALL, read before the residue cut:
    # it is the only thing that separates a broken expansion from one whose every
    # deposit cancelled (a basis carrying only difference invariants — nothing left
    # for the sum rule to constrain, which is legal).
    visited = any(>(0.0), G)
    A = _prune_residue!(A_gross, G)
    m = size(A, 1)
    # store the row-normalized form (the residual gate's conditioning);
    # `_asr_nullspace` normalizes internally (idempotent on this input)
    An = m == 0 ? A : A ./ [norm(@view A[r, :]) for r = 1:m]
    rep = _stage_reparam(basis, free, zeros(p), An)
    Z, rank = rep.Z, rep.rank
    ndisp = length(intersect(_disp_active_cols(basis), free))
    # An expansion that deposited NOTHING on a displacement-active basis is broken:
    # a displacement factor has a nonzero translation derivative term by term, so the
    # monomial walk must visit something, and every downstream diagnostic would report
    # success while the ASR is a silent no-op. Refuse. Depositing and then cancelling is
    # a different statement — the free columns are already translation-invariant (only
    # difference content survives the projection) — and it must NOT be refused: rank 0 is
    # then the right answer and `Z = I` on those columns is the right reparameterization.
    rank == 0 && ndisp > 0 && !visited &&
        throw(ArgumentError("ASR builder deposited no monomial at all on a " *
                            "displacement-active basis — the symbolic expansion " *
                            "is broken"))
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
                  "ligand. WHEN a symmetry operation exchanges the bond's two ends and " *
                  "both ends carry the same spin rank, the parity is decidable in " *
                  "advance: translation invariance needs the coupling odd under that " *
                  "exchange (it can only reach u_a - u_b), the exchange parity is the " *
                  "spin factor's times (-1)^Lf, and an EVEN product therefore has no " *
                  "partner among pair orbits at all — it needs a displaced third atom. " *
                  "On a bond whose ends NO operation exchanges (two different species, " *
                  "say) that argument does not apply and the partner can be the same " *
                  "channel with the displacement on the other end. Either way the " *
                  "coefficient is excluded by the sum rule, not by crystal symmetry: the " *
                  "basis function itself is generally nonzero" columns = dead maxlog = 1
    end
    return rep
end

"""
    asr_residual(model::SLCEModel) -> Float64
    asr_residual(f::SLCEFit) -> Float64

Relative ASR residual `‖A·β‖ / (‖A‖·‖β‖)` (Frobenius norm on `A`) of a model's
coefficients — `0.0` for a pure-spin basis (no constraints) or an exactly
translation-invariant coefficient vector. This is the public verifier: nothing
about ASR is persisted (the fingerprint precedent — recompute, never trust), so
physical consumers (force-constant deliverables, Monte-Carlo ingest of joint
models) gate on this value instead of a stored flag. A hand-built violating
model is legal; it simply reports a large residual.

On a fit the value is the one recorded at fit time (`f.asr_residual`), so the method is
there for uniformity with the other diagnostics rather than to recompute anything.

An [`AllImages`](@ref) basis with self-image clusters **raises** an
[`UnclassifiableBasis`](@ref) instead of returning a number: the constraint matrix needs
a Gaunt expansion of the same-site factor product, which the builder does not implement.
That is deliberate — a `NaN` would pass every `residual > tol` gate downstream. Such a
basis is a tiling template rather than a fitted one; [`SLCEDataset`](@ref) refuses it at
the fitting door, so this diagnostic only ever meets one on a hand-built model.
"""
function asr_residual(model::SLCEModel)::Float64
    basis = model.basis
    # The ASR matrix alone, not a whole reparameterization: this measures translation
    # invariance, which the unresolvable-column freeze has nothing to do with. Building
    # the reparameterization here would also make a public diagnostic pay for the
    # classification (measured 40 ms / 70 MiB on a pure-spin basis of 334 columns, where
    # it used to be `nothing`) and inherit its refusals.
    any(s -> any(has_disp, s.key.decors), salcs(basis)) || return 0.0
    A = _asr_matrix(basis)
    isempty(A) && return 0.0
    return _asr_residual(A ./ [norm(@view A[r, :]) for r in axes(A, 1)], model.jphi)
end

# The fit RECORDS its residual at fit time, so this reads the field rather than rebuilding
# `A` — which would also disagree with the recorded value for a staged fit, whose constraint
# is its stage's and not the basis-level one.
asr_residual(f::SLCEFit)::Float64 = f.asr_residual

_asr_residual(rep::ASRReparam, beta::Vector{Float64}) = _asr_residual(rep.A, beta)

# One definition of the residual, two callers (a model's basis, and a fit's stored
# reparameterization).
function _asr_residual(A::AbstractMatrix{Float64}, beta::Vector{Float64})::Float64
    isempty(A) && return 0.0
    denom = norm(A) * norm(beta)
    denom == 0.0 && return 0.0
    return norm(A * beta) / denom
end
