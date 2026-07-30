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
# Such a column is UNIDENTIFIABLE, not absent. The same basis function is in
# general nonzero under `affine_energy` — a uniform strain displaces the tied
# images by different amounts, so the cancellation is incomplete and the relative,
# bond-stretch content survives — and nonzero in a Monte-Carlo supercell. So the
# column is kept (the basis describes the infinite crystal) and pinned to exactly
# zero for any fit on this cell, which is what `build_asr`'s reparameterization
# does with it; the remedy for someone who needs the coefficient is a reference
# cell in which the pair's minimum image is unique.
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

# Relative cut on (assembled norm) / (gross accumulated norm). Cancellation to
# roundoff lands at ~1e-16 of the gross norm; a column that merely has few terms
# still has a ratio of order 1. The three orders of margin are deliberate.
const _RESOLVE_RTOL = 1e-10

"""
    _signature_matrix(basis::SLCEBasis) -> (S, gross)

Expansion of every SALC into the common monomial/symbol basis: `S[r, j]` is the
coefficient of row function `r` in SALC `j`, and `gross[j]` is the sum of the
absolute values that were accumulated into column `j`. `norm(S[:, j])` is zero (to
roundoff) exactly when SALC `j` is identically zero on cell-periodic
configurations; `gross[j]` is the scale that zero must be judged against.
"""
function _signature_matrix(basis::SLCEBasis)
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
                throw(ArgumentError("resolvability: member with repeated atoms (an " *
                                    "AllImages self-image cluster) — same-site factor " *
                                    "products need a Gaunt expansion this expansion " *
                                    "does not implement"))
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
zero: the same basis functions are in general nonzero under [`affine_energy`](@ref)
and in a Monte-Carlo supercell.

The cause is a Wigner–Seitz boundary tie (see the [Periodic
resolvability](@ref "Periodic resolvability") chapter): the pair's minimum image is
not unique, so its tied members join the same two reference-cell atoms and the
content odd under permuting them cancels in the orbit sum. To determine such a
coefficient, describe the crystal with a cell in which that pair's minimum image
*is* unique — break the tie in every direction whose separation component equals
half the cell length.

[`fit`](@ref) pins these coefficients at exactly zero and says so; the columns
themselves are kept, because the basis describes the infinite crystal and both
[`strain_derivatives`](@ref) and a Monte-Carlo supercell consume them.

The classification is exact and deterministic — a symbolic expansion into a common
monomial basis, not a numerical probe of the evaluator.
"""
function unresolvable_columns(basis::SLCEBasis)::Vector{Int}
    S, gross = _signature_matrix(basis)
    p = n_salcs(basis)
    isempty(S) && return [j for j = 1:p if gross[j] > 0.0]
    return [j for j = 1:p
            if gross[j] > 0.0 && norm(@view S[:, j]) <= _RESOLVE_RTOL * gross[j]]
end
