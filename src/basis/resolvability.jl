# Which SALCs a reference cell can resolve at all.
#
# A SALC's value on training data is the sum over its orbit members. On a FINITE
# cell two members can join the same atoms of the reference cell — the
# Wigner–Seitz boundary ties `theory/resolvability.md` describes, where one atom
# pair carries several equidistant minimum images. A `TrainingDatum` stores one
# spin and one displacement per reference-cell atom, so every tied member sees
# identical arguments, and whatever content is odd under the operations permuting
# the ties can cancel: the SALC is then identically zero as a function of
# cell-periodic data. A tie is NECESSARY (with a unique minimum image every member
# has its own atom content) but not sufficient, and which content dies is a
# property of the whole group rather than of the tie alone — the projection is
# taken over the stabilizer of ONE member and whether the transported invariants
# cancel depends on the rest. Measured: for a tie of two, images ±d, it is the
# odd-`Lf` part that goes (the DMI-like `Lf = 1` included) and the even part that
# stays, which is what bond reversal's `(-1)^Lf` predicts; the eight-fold bcc
# corner tie removes `Lf = 2` as well. Hence this file measures rather than
# reasons.
#
# Such a column is UNIDENTIFIABLE, not absent, and the SUPERCELL is the reason:
# tiling this cell maps the tied images onto distinct atoms, so nothing cancels
# there and the same coefficient multiplies a nonzero function throughout a
# Monte-Carlo run. That holds for every frozen column, which is why they are kept
# (the basis describes the infinite crystal) and merely pinned to exactly zero for
# any fit on this cell — `build_asr`'s reparameterization does the pinning, and the
# remedy for someone who needs the coefficient is a reference cell in which the
# pair's minimum image is unique.
#
# A uniform strain reveals SOME of them, and it matters which not: measured,
# `affine_energy` is nonzero for 6 of the 8 frozen columns of a bcc degree-3
# channel and 4 of 10 on a spin × degree-1 one, exactly zero for the bcc harmonic
# one, and STRUCTURALLY zero for every pure-spin column — no displacement slot for
# a strain to act on, so `affine_energy` reduces to `predict_energy`, which is the
# annihilated orbit sum. Never quote the strain response as the reason a column is
# kept; the supercell argument is the one that holds unconditionally.
#
# The test is STRUCTURAL, not statistical: expand every SALC into the one common
# monomial/symbol basis (spin factors are opaque symbols `(atom, l, m)`,
# displacement factors are exact monomials from
# `SolidHarmonics.solid_harmonic_poly`) and compare each column's assembled norm
# against the GROSS norm it accumulated. Those row functions — products of
# distinct spherical harmonics in the site spins times monomials in the site
# displacements — are linearly independent, so a column vanishes in this basis iff
# the function is identically zero on cell-periodic configurations. Comparing
# against the column's own gross accumulation makes the cut scale-free per column:
# it asks "did this column cancel?", not "is this column small?", which a global
# threshold cannot distinguish for a basis mixing spin and high-degree
# displacement channels.

"""
    UnclassifiableBasis <: Exception

Raised when a basis cannot be expanded into the common monomial/symbol basis at all,
so no statement about which of its columns a cell can resolve is available: a member
whose cluster reuses one atom (an [`AllImages`](@ref) self-image) would need a Gaunt
expansion of the same-site factor product, which this expansion does not implement.

Callers that must keep working on such a basis catch it and proceed with *nothing
frozen* — the honest "unknown", not "none". An `AllImages` joint basis is already an
explicit opt-out (`SLCEDataset` says fits must pass `asr = false`), and it would be a
regression for the classification to remove a route that used to work.
"""
struct UnclassifiableBasis <: Exception
    reason::String
end
Base.showerror(io::IO, e::UnclassifiableBasis) =
    print(io, "UnclassifiableBasis: ", e.reason)

# Row identity: the spin symbol multiset and the per-atom displacement monomial
# exponents. This is `_ASRRowKey` (fitting/asr.jl) without the generator axis —
# that builder expands `Σ_a ∂E/∂u_a`, this one expands `E` itself.
struct _FuncRowKey
    spins::Vector{NTuple{3,Int}}   # sorted (atom, l, m)
    disps::Vector{NTuple{4,Int}}   # sorted (atom, ex, ey, ez); zero exponents kept out
end
Base.hash(k::_FuncRowKey, h::UInt) = hash(k.disps, hash(k.spins, h))
Base.:(==)(a::_FuncRowKey, b::_FuncRowKey) = a.spins == b.spins && a.disps == b.disps
Base.isless(a::_FuncRowKey, b::_FuncRowKey) = (a.spins, a.disps) < (b.spins, b.disps)

# Relative cut on (assembled value) / (gross accumulated mass). Cancellation to
# roundoff lands at ~N·eps of the gross mass; a column that merely has few terms
# still has a ratio of order 1. The margin is deliberate: it must cover the term
# count of a large orbit sum (N ~ 1e4 already reaches 1e-12) and stay far below any
# genuine ratio.
#
# ONE definition, two readers, because it answers one question — "did this cancel?",
# never "is this small?" — at two granularities: `unresolvable_columns` below judges
# a whole COLUMN of the undifferentiated expansion, `_prune_residue!`
# (`fitting/asr.jl`) judges each ENTRY of the differentiated one. A cut taken
# against the matrix's own maximum instead cannot see the case where EVERYTHING
# cancels, which is exactly the case a Wigner–Seitz-tied cell produces.
const _CANCELLATION_RTOL = 1e-10

"""
    _signature_matrix(basis::SLCEBasis) -> (S, gross)

Expansion of every SALC into the common monomial/symbol basis: `S[r, j]` is the
coefficient of row function `r` in SALC `j`, and `gross[j]` is the sum of the
absolute values that were accumulated into column `j`. `norm(S[:, j])` is zero (to
roundoff) exactly when SALC `j` is identically zero on cell-periodic
configurations; `gross[j]` is the scale that zero must be judged against.
"""
function _signature_matrix(basis::SLCEBasis)
    # The ratio test is per-column scale-invariant (a SALC key fixes every slot's
    # `2k + l`, so the evaluator's scale is constant along a column), but the entries are
    # meant to be comparable with `_asr_matrix`'s, and that comparison would silently
    # shift under a per-degree rescaling. Same guard, same reason (design record §6
    # amendment 3).
    basis.spec.disp_scale == 1.0 ||
        throw(ArgumentError("the resolvability expansion predates disp_scale ≠ 1 " *
                            "support — its monomial expansion must be rescaled per " *
                            "degree when that guard lifts"))
    ss = salcs(basis)
    p = length(ss)
    polycache = Dict{NTuple{3,Int},SolidHarmonics._Poly}()
    rows = Dict{_FuncRowKey,Dict{Int,Float64}}()
    gross = zeros(Float64, p)
    for (j, s) in enumerate(ss)
        # The evaluator's column scale, so `S` lives in design coordinates like
        # `_asr_matrix` does. It is constant along a column, hence irrelevant to
        # the ratio test, but it keeps the two expansions comparable entry by entry.
        scale = (4π)^(count(has_spin, s.decors) / 2)
        for mem in s.members
            allunique(mem.atoms) ||
                throw(UnclassifiableBasis("member with repeated atoms (an AllImages " *
                                          "self-image cluster): same-site factor " *
                                          "products need a Gaunt expansion this " *
                                          "monomial expansion does not implement"))
            for t in mem.terms
                _sig_accumulate_term!(rows, gross, j, scale, t, mem.atoms, polycache)
            end
        end
    end
    keyorder = sort!(collect(keys(rows)))
    S = zeros(Float64, length(keyorder), p)
    for (r, key) in enumerate(keyorder)
        for (j, v) in rows[key]
            S[r, j] = v
        end
    end
    return S, gross
end

# One SALC term's contribution to the undifferentiated monomial expansion.
function _sig_accumulate_term!(rows::Dict{_FuncRowKey,Dict{Int,Float64}},
                               gross::Vector{Float64}, j::Int, scale::Float64,
                               t::SALCTerm, atoms::Vector{Int},
                               polycache::Dict{NTuple{3,Int},SolidHarmonics._Poly})
    slots = t.slots
    D = length(slots)
    spin_ids = Int[i for i = 1:D if slots[i].factor.channel == SPIN]
    disp_ids = Int[i for i = 1:D if slots[i].factor.channel != SPIN]
    folded = t.folded
    for idx in CartesianIndices(folded)
        w = folded[idx]
        w == 0.0 && continue
        spins = NTuple{3,Int}[(atoms[slots[i].site], slots[i].factor.l,
                               idx[i] - slots[i].factor.l - 1) for i in spin_ids]
        sort!(spins)
        dslots = Vector{Tuple{Int,SolidHarmonics._Poly}}(undef, length(disp_ids))
        for (n, i) in enumerate(disp_ids)
            f = slots[i].factor
            m = idx[i] - f.l - 1
            dslots[n] = (atoms[slots[i].site],
                         get!(polycache, (f.k, f.l, m)) do
                             SolidHarmonics.solid_harmonic_poly(f.k, f.l, m)
                         end)
        end
        _sig_expand_product!(rows, gross, j, w * scale, spins, dslots)
    end
    return nothing
end

# Accumulate `w · Π_o poly_o` over every monomial combination of the displacement
# slots. With no displacement slot this deposits `w` on the pure-spin row, which is
# how a pure-spin SALC gets classified by the same machinery.
function _sig_expand_product!(rows::Dict{_FuncRowKey,Dict{Int,Float64}},
                              gross::Vector{Float64}, j::Int, w::Float64,
                              spins::Vector{NTuple{3,Int}},
                              dslots::Vector{Tuple{Int,SolidHarmonics._Poly}})
    function rec(o::Int, coeff::Float64, acc::Vector{NTuple{4,Int}})
        if o > length(dslots)
            key = _FuncRowKey(spins, sort(acc; by = first))
            row = get!(() -> Dict{Int,Float64}(), rows, key)
            row[j] = get(row, j, 0.0) + coeff
            gross[j] += abs(coeff)
            return nothing
        end
        (atom_o, poly_o) = dslots[o]
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
    rec(1, w, NTuple{4,Int}[])
    return nothing
end

"""
    unresolvable_columns(basis::SLCEBasis) -> Vector{Int}

The design columns that are identically zero on every cell-periodic configuration
this reference cell can express, in ascending order. Their coefficients are
**unidentifiable from any amount of training data on this cell** — not physically
zero: tiling this cell maps the tied images onto *distinct* atoms, so the same basis
functions are nonzero throughout a Monte-Carlo supercell run. (A uniform strain
reveals some of them as well, but not all, and never a pure-spin one — that has no
displacement slot for a strain to act on.)

The cause is a Wigner–Seitz boundary tie (see the [Periodic
resolvability](@ref "Periodic resolvability") chapter): the pair's minimum image is
not unique, so its tied members join the same two reference-cell atoms and the
content odd under permuting them cancels in the orbit sum. To determine such a
coefficient, describe the crystal with a cell in which that pair's minimum image
*is* unique — break the tie in every direction whose separation component equals
half the cell length.

[`fit`](@ref) pins these coefficients at exactly zero and says so; the columns
themselves are kept, because the basis describes the infinite crystal and a
Monte-Carlo supercell consumes them.

The classification is exact and deterministic — a symbolic expansion into a common
monomial basis, not a numerical probe of the evaluator. A cheap structural pre-check
(does any orbit put two members on the same reference-cell atoms?) rules the whole
phenomenon out before that expansion runs, so a cell with unique minimum images pays
nothing for asking.
"""
function unresolvable_columns(basis::SLCEBasis)::Vector{Int}
    # The refusal first, so it does not depend on whether the fast path fires.
    for s in salcs(basis), mem in s.members
        allunique(mem.atoms) ||
            throw(UnclassifiableBasis("member with repeated atoms (an AllImages " *
                                      "self-image cluster): same-site factor " *
                                      "products need a Gaunt expansion this " *
                                      "monomial expansion does not implement"))
    end
    _has_boundary_tie(basis) || return Int[]
    return _unresolvable_expanded(basis)
end

# A TIE IS NECESSARY, and cheap to rule out. Two members can only cancel each other if
# they deposit on the same row, and a row is keyed by its per-ATOM content — so members
# whose atom multisets differ never meet. What is left is a member cancelling on its own,
# which cannot happen: a member's contribution is the invariant projected on the
# stabilizer of that one instance, and it is nonzero by construction (the same
# single-member values gate (A) in `test/unit/test_resolvability.jl` uses as its scale).
#
# So an untied basis — every cell with unique minimum images, which is most of them —
# skips the expansion entirely. That matters because the diagnostics at the physical
# readouts (`_warn_unresolvable`, `slce/forceconstants.jl`) call this on every deliverable,
# and the expansion is not free (measured 40 ms / 70 MiB on 334 columns). The equivalence
# is gated: gate (A) compares BOTH this function and the unconditional
# `_unresolvable_expanded` against the evaluator's verdict, on tied and untied fixtures.
function _has_boundary_tie(basis::SLCEBasis)::Bool
    seen = Set{Vector{Int}}()
    for s in salcs(basis)
        length(s.members) > 1 || continue
        empty!(seen)
        for mem in s.members
            key = sort(mem.atoms)
            key in seen && return true
            push!(seen, key)
        end
    end
    return false
end

# The classification proper, without the fast path: the expansion's own verdict.
function _unresolvable_expanded(basis::SLCEBasis)::Vector{Int}
    S, gross = _signature_matrix(basis)
    p = n_salcs(basis)
    isempty(S) && return [j for j = 1:p if gross[j] > 0.0]
    return [j for j = 1:p
            if gross[j] > 0.0 && norm(@view S[:, j]) <= _CANCELLATION_RTOL * gross[j]]
end
