# The vanishing subspace of a block gets columns of its own.
#
# A BLOCK is the set of columns that share every label of their key but `block`: one
# orbit, one decoration multiset, one `(L_S, Lf)` (and, in the pointed basis, one mark
# class). The symmetry fixes the space the block spans, not the columns inside it; the
# columns are a gauge, chosen along the crystal axes (`_project_and_fold`).
#
# On a finite reference cell a Wigner–Seitz boundary tie can make a SUBSPACE of a block
# vanish identically on every cell-periodic configuration (face (a) of
# `basis/resolvability.jl`). Whether that subspace happens to be spanned by single
# columns depends on the gauge, and the per-column freeze (`unresolvable_columns`,
# route A) sees single columns only. Measured on 0add649: hcp (2,2,2) at displacement
# degree 3 has a vanishing subspace of dimension 5 in one block, of which ONE direction
# was a column, along the crystal axes or not; the four others were left to the
# estimator, and AdaptiveRidge / `refit` / metric Ridge put undetermined content there
# that the cell cannot see and a supercell or a `q ≠ 0` readout does.
#
# So the build re-chooses the gauge of every block that has such a subspace: the block
# is expanded symbolically (`_signature_of` for the energy basis, `_moment_signature_of`
# for the pointed one — the expansions `unresolvable_columns` and `moment_resolvability`
# judge with), its right null space `N0` is taken against the block's gross mass, and
# the columns become `[canon(N1) canon(N0)]` in the coordinates of the old columns, the
# non-vanishing ones first. The old columns are already the crystal-frame gauge, so
# these coordinates do not depend on how the crystal is turned, and `_canonical_basis`
# picks the new columns with the same cuts. The per-column test then freezes `N0`
# exactly: route A, unchanged, now reaching the whole vanishing subspace.
#
# A combination of columns of DIFFERENT blocks can vanish too; no gauge choice inside a
# block reaches it. `_unresolvable_split` reports it per orbit and `build_asr` warns.

# The relative band in which a singular value of a block's expansion would be neither a
# cancellation nor content. A cancellation lands at round-off of the gross mass (below
# `_CANCELLATION_RTOL`); genuine content sits at O(1) of it, whatever the orientation,
# because turning the crystal changes the expansion by an invertible map of bounded
# condition. A value between the two means the rank cannot be decided, and the build
# refuses instead of guessing. The test suite holds the measured margins on both sides.
const _REGAUGE_BAND_TOP = 1e-6

"""
    _block_split(S, gross) -> (; null, kept, ratios)

The rank decision of one block: `S` is the block's expansion (rows × block columns) and
`gross` the gross mass of each column. Returns the right singular vectors spanning the
vanishing subspace (`null`, one per column) and its complement (`kept`), and the
singular values divided by the block's gross mass (`ratios`, descending, padded with
zeros to the block size). A singular value is a cancellation when it is at most
`_CANCELLATION_RTOL` times the largest gross mass of the block: the reading of the
per-column test (`_unresolvable_split`) taken over the block.
"""
function _block_split(S::AbstractMatrix{Float64}, gross::AbstractVector{Float64})
    k = size(S, 2)
    mass = maximum(gross; init = 0.0)
    # QR first: the singular values and right vectors of `S` are those of its `R`
    # factor, and `R` is at most `k × k` whatever the number of rows.
    R = size(S, 1) == 0 ? zeros(Float64, 0, k) : Matrix(qr(S).R)
    F = svd(R; full = true)
    sv = vcat(F.S, zeros(Float64, k - length(F.S)))
    ratios = mass > 0.0 ? sv ./ mass : zeros(Float64, k)
    vanish = [r <= _CANCELLATION_RTOL for r in ratios]
    return (; null = F.V[:, vanish], kept = F.V[:, .!vanish], ratios)
end

# The new gauge of a block in the coordinates of its old columns, or `nothing` when the
# block keeps its columns: no vanishing direction, or nothing else. Entries that are
# round-off against a unit vector are set to zero, so a block whose vanishing subspace
# was already spanned by single columns comes back as an exact permutation and its
# tensors are copied, not recombined.
function _vanishing_gauge(S::AbstractMatrix{Float64}, gross::AbstractVector{Float64},
                          what)::Union{Nothing,Matrix{Float64}}
    split = _block_split(S, gross)
    k = size(S, 2)
    n0 = size(split.null, 2)
    undecided = findall(r -> _CANCELLATION_RTOL < r <= _REGAUGE_BAND_TOP, split.ratios)
    isempty(undecided) || error(
        "cannot decide which combinations of the columns of block $what vanish on " *
        "this reference cell: relative singular values $(split.ratios[undecided]) lie " *
        "between the cancellation cut $(_CANCELLATION_RTOL) and $(_REGAUGE_BAND_TOP). " *
        "This is a bug or a cell far outside what the gauge was measured on; please " *
        "report the crystal and the spec")
    (n0 == 0 || n0 == k) && return nothing
    W = hcat(_canonical_basis(split.kept), _canonical_basis(split.null))
    for i in eachindex(W)
        abs(W[i]) < _ROUNDOFF_RTOL && (W[i] = 0.0)
    end
    W == Matrix{Float64}(I, k, k) && return nothing
    return W
end

# The SALC `Σᵢ w[i]·columns[i]` under `key`. The columns of one block share their labels,
# so the combination is the term-by-term sum of their member tensors; the canonical
# member fold does the summing. Round-off of the sums is set to zero the way the
# projection does it: an entry against its own tensor, a whole tensor against the
# column (whose coefficient vector is a unit vector).
function _combine_salcs(columns::Vector{SALC}, w::AbstractVector{Float64},
                        key::SALCKey)::SALC
    raw = SALCMember[]
    for (i, s) in enumerate(columns)
        w[i] == 0.0 && continue
        for m in s.members
            push!(raw, SALCMember(m.atoms, m.shifts,
                                  [SALCTerm(t.slots, w[i] .* t.folded) for t in m.terms]))
        end
    end
    folded = _canonicalize_members(raw)
    big = maximum((maximum(abs, t.folded) for m in folded for t in m.terms); init = 0.0)
    members = SALCMember[]
    for m in folded
        terms = SALCTerm[]
        for t in m.terms
            maximum(abs, t.folded) > _ROUNDOFF_RTOL * big || continue
            push!(terms, SALCTerm(t.slots, _zero_roundoff!(copy(t.folded))))
        end
        isempty(terms) || push!(members, SALCMember(m.atoms, m.shifts, terms))
    end
    return SALC(key, key.body, key.decors, key.L_S, key.Lf, members)
end

# A tie is necessary for a vanishing direction: rows are keyed by per-atom content, so
# a block whose members all reach different atom multisets cannot cancel (the argument
# of `_has_boundary_tie`, here inside one block).
function _block_has_tie(columns::Vector{SALC})::Bool
    seen = Set{Vector{Int}}()
    for s in columns
        empty!(seen)
        for m in s.members
            atoms = sort(m.atoms)
            atoms in seen && return true
            push!(seen, atoms)
        end
    end
    return false
end

# The two expansions, as the build calls them. Each returns `nothing` for a block it
# cannot classify — the build leaves such a block alone, and the doors
# (`unresolvable_columns`, `moment_resolvability`) still refuse it by name.
function _energy_block_signature(columns::Vector{SALC})
    all(s -> all(m -> allunique(m.atoms), s.members), columns) || return nothing
    return _signature_of(columns)
end

# The same refusal as `_moment_resolvability`'s: two spin factors on one reference-cell
# atom (two images of one neighbour, or an environment on an image of the mark).
function _moment_block_signature(columns::Vector{SALC})
    for s in columns, m in s.members, t in m.terms
        mark = findfirst(sl -> sl.factor.channel == DISP, t.slots)
        mark === nothing && return nothing
        mark_site = t.slots[mark].site
        atoms = Int[m.atoms[mark_site]]
        for sl in t.slots
            (sl.factor.channel == SPIN && sl.site != mark_site) || continue
            push!(atoms, m.atoms[sl.site])
        end
        allunique(atoms) || return nothing
    end
    return _moment_signature_of(columns)
end

# No refinement of the block: the energy basis.
_no_class(::SALC) = nothing

# The block of a column: its key without `block`.
_block_key(s::SALC) = (s.key.body, s.key.orbit_id, s.key.decors, s.key.L_S, s.key.Lf)

# The mark class of a pointed column (the `salc_groups(::MomentBasis)` fingerprint): the
# atoms and the sites that carry the mark in the representative member. Columns of one
# energy-side block with different mark classes predict different atoms, so they are
# never mixed.
function _mark_class(s::SALC)
    mem = s.members[1]
    marks = Int[]
    sites = Int[]
    for t in mem.terms
        mark = findfirst(sl -> sl.factor.channel == DISP, t.slots)
        mark === nothing && continue
        push!(sites, t.slots[mark].site)
        push!(marks, mem.atoms[t.slots[mark].site])
    end
    return (sort!(unique!(marks)), sort!(unique!(sites)))
end

"""
    _regauge_vanishing(columns, signature, class) -> Vector{SALC}

The columns of one orbit with every block whose vanishing subspace is a proper,
nonzero part of it re-gauged as `[canon(N1) canon(N0)]` (see the head of this file).
`signature` expands a list of columns (`_energy_block_signature` or
`_moment_block_signature`), `class` refines the block (`_mark_class` for the pointed
basis, `_ -> nothing` otherwise). The re-gauged columns of a block take the block's
`block` indices in ascending order, so the keys of the orbit do not change; only the
functions behind the keys of a re-gauged block do. Blocks without a tie are not
expanded at all, so a cell with unique minimum images builds exactly the basis it built
without this step.
"""
function _regauge_vanishing(columns::Vector{SALC}, signature::Function,
                            class::Function)::Vector{SALC}
    length(columns) < 2 && return columns
    groups = Dict{Any,Vector{Int}}()
    order = Any[]
    for (j, s) in enumerate(columns)
        g = (_block_key(s), class(s))
        haskey(groups, g) || push!(order, g)
        push!(get!(groups, g, Int[]), j)
    end
    out = copy(columns)
    for g in order
        index = sort(groups[g]; by = j -> columns[j].key.block)
        length(index) < 2 && continue
        block = columns[index]
        _block_has_tie(block) || continue
        expansion = signature(block)
        expansion === nothing && continue
        S, gross = expansion
        W = _vanishing_gauge(S, gross, block[1].key)
        W === nothing && continue
        for (b, j) in enumerate(index)
            k = columns[j].key
            out[j] = _combine_salcs(block, view(W, :, b),
                                    SALCKey(k.body, k.orbit_id, k.decors, k.L_S, k.Lf,
                                            k.block))
        end
    end
    return out
end

"""
    regauged_columns(basis::SLCEBasis) -> Vector{Int}
    regauged_columns(mb::MomentBasis) -> Vector{Int}

The columns, in ascending order, of every block whose gauge this reference cell chose: a
block (the columns sharing every label of their [`SALCKey`](@ref) but `block`; in a
pointed basis also one mark class) of which a proper, nonzero part vanishes identically
on every configuration the cell can express. The build rotated such a block so that the
vanishing part is columns of its own, which the fit then freezes
([`unresolvable_columns`](@ref), [`moment_resolvability`](@ref)); the remaining columns
of the block are the complement, also chosen along the crystal axes.

These columns depend on the cell — the same crystal on a (3,3,3) supercell, or with
other lattice vectors, has other columns in that block — so their coefficients are not
comparable column by column across cells. Compare models through readouts that do not
depend on the gauge (`force_constants`, `bilinear_terms`, predicted energies).
[`coeftable`](@ref) carries the same flag as its `regauged` column. A basis the
classification cannot read (an `AllImages` template) reports none.
"""
function regauged_columns(basis::SLCEBasis)::Vector{Int}
    vanishing = try
        _unresolvable_split(basis).vanishing
    catch err
        err isa UnclassifiableBasis || rethrow()
        return Int[]
    end
    return _regauged(salcs(basis), vanishing, _no_class)
end

function regauged_columns(mb::MomentBasis)::Vector{Int}
    vanishing = try
        moment_resolvability(mb).vanishing
    catch err
        err isa UnclassifiableBasis || rethrow()
        return Int[]
    end
    return _regauged(salcs(mb), vanishing, _mark_class)
end

function _regauged(sal::AbstractVector{SALC}, vanishing::AbstractVector{Int},
                   class::Function)::Vector{Int}
    isempty(vanishing) && return Int[]
    groups = Dict{Any,Vector{Int}}()
    for (j, s) in enumerate(sal)
        push!(get!(groups, (_block_key(s), class(s)), Int[]), j)
    end
    vset = Set(vanishing)
    out = Int[]
    for cols in values(groups)
        n0 = count(in(vset), cols)
        0 < n0 < length(cols) && append!(out, cols)
    end
    return sort!(out)
end
