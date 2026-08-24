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
# A tie has TWO algebraic faces, and only the first one is a null column.
#
#  (a) the tied images sit in ONE orbit — the point group permutes them, so the orbit
#      sum weights them equally and the odd content cancels: the column is identically
#      zero and is frozen at exactly zero. Equal weighting here is SYMMETRY, not a
#      gauge, so the surviving even content is a determined coupling and stays.
#  (b) the tied images sit in DIFFERENT orbits — no operation relates them, so each
#      carries its own coupling. Every column is individually nonzero, and the columns
#      of the two orbits are NOT equal either (a member's tensors carry its own bond
#      geometry). What collapses is the SPAN: every member of either orbit reads its
#      sites' displacements off the same reference-cell atoms, so the orbits span the
#      SAME function space and the data determine only how much of that space is used
#      in total, never how it divides. Nothing is zero and nothing cancels; a whole
#      COMBINATION of columns is flat.
#
# Face (b) needs low symmetry — measured with real space groups, `nullity(S)` is
# exactly the count of null columns on bcc Fe, B2 FeRh, hcp Co, wurtzite GaN and
# rocksalt MnO, and face (b) appears on P1 and on monoclinic C2/c (CuO). It is the
# more dangerous of the two: the fit is perfect (`rmse_E` ~ 1e-16), `asr_residual` is
# clean, `D(0)` is exact — and `D(q ≠ 0)` moves by 50 % between two models the data
# cannot tell apart.
#
# There is NO justified split of the sum. Equal division is what phonopy/ALM-style
# codes do with aliased images, but for two bonds no symmetry relates it is an
# interpolation ansatz, not a measurement: the cell determines the pair only at the `q`
# where `exp(2πi q·(R₁ − R₂)) = 1`, and `R₁ − R₂` is a lattice vector of THIS cell, so
# that is `q = 0` alone. Rather than turn an undetermined split into a
# publishable-looking dispersion, the whole interaction is dropped: every column of
# every orbit sharing an atom multiset is frozen at exactly zero and named. That
# deliberately discards the determined sum too — the fit residual stops being zero,
# which is the loud failure — and the remedy is the same as for face (a): a reference
# cell in which the minimum image is unique.
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

It is also what the fitting doors raise: [`SLCEDataset`](@ref) refuses a basis with
self-image members outright (`_refuse_self_image_basis`), because on the reference cell
those columns are redundant by construction — every image of the atom carries the same
spin — so no fit can determine them and an unconstrained one returns an arbitrary
representative. Building such a basis stays legal: it is the tiling template a
downstream consumer expands onto a supercell, where the images become distinct sites.

Callers that only *analyze* a basis (`build_asr` on a hand-built model) catch this and
proceed with *nothing frozen* — the honest "unknown", not "none".
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
    for index in CartesianIndices(folded)
        w = folded[index]
        w == 0.0 && continue
        spins = NTuple{3,Int}[(atoms[slots[i].site], slots[i].factor.l,
                               index[i] - slots[i].factor.l - 1) for i in spin_ids]
        sort!(spins)
        dslots = Vector{Tuple{Int,SolidHarmonics._Poly}}(undef, length(disp_ids))
        for (n, i) in enumerate(disp_ids)
            f = slots[i].factor
            m = index[i] - f.l - 1
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

The design columns whose coefficients are **unidentifiable from any amount of training
data on this cell**, in ascending order — not physically zero: tiling this cell maps the
tied images onto *distinct* atoms, so the same basis functions are nonzero throughout a
Monte-Carlo supercell run.

The cause is always a Wigner–Seitz boundary tie (see the [Periodic
resolvability](@ref "Periodic resolvability") chapter) — the pair's minimum image is not
unique, so tied members join the same two reference-cell atoms — and it has two faces:

1. **the column vanishes.** The point group permutes the tied images, so they sit in one
   orbit whose sum weights them equally, and the content odd under the permutation
   cancels: the column is identically zero as a function of cell-periodic data. Equal
   weighting is symmetry here, so the *even* content of the same orbit is a determined
   coupling and is kept.
2. **the whole interaction is dropped.** In low symmetry (P1, monoclinic) no operation
   relates the tied images, so they sit in *different* orbits with independent
   couplings. Each column is nonzero, and the columns of the two orbits are not equal to
   each other either — a member's tensors carry its own bond geometry. What collapses is
   the SPAN: every member of either orbit reads its sites' displacements off the same
   reference-cell atoms, so the orbits span the same function space and the data fix only
   how much of that space is used in total. Splitting that total has no physical basis
   (the cell determines the pair only at `q = 0`, where both images carry the same
   phase), so every column of every orbit sharing an atom multiset is frozen instead of
   split. This discards the determined part as well: the fit residual stops being zero,
   deliberately, because a silent split would publish a dispersion the data never
   constrained.

   The undetermined fraction is NOT "half" in general. It is `length(frozen) −
   rank(S[:, frozen])`: for a `k`-fold tie whose images the point group fully separates
   into `k` orbits it is `1 − 1/k` of the block, and under PARTIAL fusion faces 1 and 2
   mix within one basis (measured on a four-fold tie under `{E, m_y}`: 8 vanishing plus
   10 undetermined, true flat dimension 13).

Either way the remedy is the same, and it is never a wider cutoff: describe the crystal
with a cell in which that pair's minimum image *is* unique — break the tie in every
direction whose separation component equals half the cell length.

Face 2 is the dangerous one and was missed until it was measured: a fit on a tied P1 cell
reached `rmse_energy = 4e-16` with a clean `asr_residual` and an exact `D(0)`, while
`D(q ≠ 0)` was 52 % wrong and every coefficient of the tied shell was arbitrary.

[`fit`](@ref) pins these coefficients at exactly zero and says so; the columns
themselves are kept, because the basis describes the infinite crystal and a
Monte-Carlo supercell consumes them.

The classification is exact and deterministic — a symbolic expansion into a common
monomial basis, not a numerical probe of the evaluator. A cheap structural pre-check
(is any atom multiset reached twice — by two members of one orbit, or by two orbits?)
rules the whole phenomenon out before that expansion runs, so a cell with unique minimum
images pays nothing for asking. That pre-check is exact because a tie is NECESSARY, and
the necessity is a proof rather than a measurement: under `MinimumImage` every edge sits
at its atom pair's minimum image, so fixing site 1 at the origin fixes every other site's
image uniquely as long as each of those minimum images is unique — one cluster per atom
multiset, up to translation. Two orbits over one multiset therefore force some edge to
have a second equidistant image.

**At `N ≥ 3` the freeze protects CONGRUENT siblings only.** A tie produces several
clusters over one atom multiset, and `candidate_clusters` admits a cluster only when all
`C(N,2)` edges sit at their minimum image simultaneously (the compact-cluster criterion).
Congruent siblings pass or fail that test together, so they are enumerated together and
the freeze sees them. A NON-congruent sibling — same atom multiset, but reached through
an image that puts one of its other edges on a longer shell — is rejected there, and the
tie then leaves no trace at all: `_has_boundary_tie` returns `false` and nothing is
frozen. That is not a hidden indeterminacy. The rejected cluster's coupling is absorbed
by the admitted one exactly as a farther shell of a pair is, because a design column is a
function of the cell-periodic field: measured on a P1 cell whose `(1,2)` edge is tied, a
3-body degree-`(1,1,1)` sector spans the FULL trilinear space (rank 27 = 3³, identical
span to an independently built basis of all 27 monomials), so nothing is unidentifiable
and `identifiability` reports `nullity = 0`. What the aliasing costs is the `R` label:
[`force_constants`](@ref) attributes the whole coupling to the admitted geometry, which
matters when the cubic constants are exported and read as `Φ(R₁, R₂)`.
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
    return _unresolvable_split(basis).columns
end

# The orbit a SALC belongs to. `SALCKey`'s first two fields address the cluster orbit;
# the rest address a channel within it, so every column of one orbit shares this.
_orbit_of(s::SALC)::Tuple{Int,Int} = (s.key.body, s.key.orbit_id)

"""
    _shared_multisets(basis) -> (orbits, multisets)

The atom multisets reached by MORE THAN ONE cluster orbit, and the set of orbits that
reach them. Under [`MinimumImage`](@ref) this is exactly tie face (b): a second orbit
over the same atoms requires some edge to have a second equidistant minimum image
(measured — widening the cutoff to admit a farther shell of the same pair does *not*
produce one, because only the minimum image is enumerated).

Under [`AllImages`](@ref) it also catches genuinely distinct shells of one pair, which
ARE redundant for the cell-periodic evaluator this package has, and would stop being
redundant only on the future generalized-Bloch path where `exp(iq·R)` separates the
images. That path must revisit this function; today freezing them is the honest reading.
"""
function _shared_multisets(basis::SLCEBasis)
    owners = Dict{Vector{Int},Set{Tuple{Int,Int}}}()
    for s in salcs(basis)
        o = _orbit_of(s)
        for mem in s.members
            push!(get!(() -> Set{Tuple{Int,Int}}(), owners, sort(mem.atoms)), o)
        end
    end
    multisets = sort!([k for (k, v) in owners if length(v) > 1])
    orbits = Set{Tuple{Int,Int}}()
    for k in multisets
        union!(orbits, owners[k])
    end
    return orbits, multisets
end

"""
    _unresolvable_split(basis) -> (; columns, vanishing, undetermined, multisets,
                                     residual_flat)

The freeze, with the two reasons kept apart because their messages differ:

- `vanishing` — tie face (a): identically zero on every configuration this cell can
  express (the null-column test);
- `undetermined` — tie face (b): a column of an orbit that shares an atom multiset with
  another orbit, so only the sum of their couplings is determined and no split is
  justified. The whole interaction is dropped;
- `columns` — what `fit` freezes, the union in ascending order;
- `multisets` — the offending atom multisets, for the message;
- `residual_flat` — flat directions LEFT after the freeze, computed whenever the
  expansion ran. A nonzero value is reported rather than swallowed.

It used to be computed only when face (b) fired, on the argument that whole-orbit
granularity is the only thing that can under- or over-shoot. That argument is sound and
covers only ONE of the two routes to a leftover flat direction. The other needs no second
orbit at all: two channels of a SINGLE orbit, each individually nonzero, can become
dependent once the tie collapses their arguments. A per-column test cannot see that by
construction, and under the old condition nothing else looked either — so the one
diagnostic that could have caught it was switched off in exactly the case it was needed.
No basis has yet produced a nonzero value (about 70 were tried: real crystals and
hand-grouped fixtures, `N = 1..4`, degree ≤ 6, tie multiplicities 2/4/6/8, `frozen` equal
to the sampled nullity every time), which is a reason to keep the check cheap, not a
reason to skip it. `S` is already built by the time we get here and the rank costs about
2 % of building it.
"""
function _unresolvable_split(basis::SLCEBasis)
    orbits, multisets = _shared_multisets(basis)
    empty_result = (; columns = Int[], vanishing = Int[], undetermined = Int[],
                    multisets = Vector{Int}[], residual_flat = 0)
    # One pre-check for both faces: `_has_boundary_tie` is true when a single SALC
    # repeats an atom multiset (face a) or when `orbits` is nonempty (face b).
    _has_boundary_tie(basis) || return empty_result
    S, gross = _signature_matrix(basis)
    p = n_salcs(basis)
    vanishing = isempty(S) ? [j for j = 1:p if gross[j] > 0.0] :
                [j for j = 1:p
                 if gross[j] > 0.0 &&
                    norm(@view S[:, j]) <= _CANCELLATION_RTOL * gross[j]]
    vset = Set(vanishing)
    undetermined = isempty(orbits) ? Int[] :
                   [j for (j, s) in enumerate(salcs(basis))
                    if _orbit_of(s) in orbits && !(j in vset)]
    columns = sort!(vcat(vanishing, undetermined))
    # NOT gated on `undetermined`: a leftover flat direction can also come from two
    # channels of ONE orbit going dependent, which face (a) alone reaches — see the
    # docstring. `S` is already built, so this is the cheap half of the step.
    residual_flat = 0
    if !isempty(S)
        kept = setdiff(1:p, columns)
        # `identifiability`'s cut, on the structural expansion instead of the design:
        # relative to σ_max at `min(size)·eps`, so the two reports are read the same way.
        residual_flat = isempty(kept) ? 0 : length(kept) - _struct_rank(S[:, kept])
    end
    return (; columns, vanishing, undetermined, multisets, residual_flat)
end

# A TIE IS NECESSARY, and cheap to rule out. Columns interact only through shared rows,
# and a row is keyed by its per-ATOM content — so content over different atom multisets
# never meets. Two members of ONE SALC over the same multiset is face (a); the same
# multiset reached by TWO orbits is face (b). Both are one pass over the member lists.
#
# **This check used to look inside one SALC only, and that was a real hole**: on a P1 cell
# whose pair sits on the Wigner–Seitz face the two tied images land in different orbits,
# so `_has_boundary_tie` returned `false`, the expansion never ran, and nine flat
# directions reached `dynamical_matrix(q ≠ 0)` with a 52 % error while every gate was
# green. The necessity argument was sound for a null COLUMN and silent about a null
# COMBINATION. Never narrow this back to the within-SALC scan.
#
# An untied basis — every cell with unique minimum images, which is most of them — still
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
    orbits, _ = _shared_multisets(basis)
    return !isempty(orbits)
end

# Numerical rank of the structural expansion, at `identifiability`'s convention
# (`min(size)·eps` relative to σ_max) so a `residual_flat` count and an
# `identifiability` report are read against the same cut.
function _struct_rank(M::AbstractMatrix)::Int
    isempty(M) && return 0
    s = svdvals(M)
    return count(>(minimum(size(M)) * eps(Float64) * s[1]), s)
end

# The classification proper, without the fast path: the expansion's own verdict.
#
# Two guards read oddly and mean the same thing. `gross[j] > 0.0` exempts a column that
# deposited NOTHING: such a column is zero as well, but not by cancellation — it would be
# a builder bug (a SALC with no terms), and calling it unresolvable would name a tie that
# is not there and send the reader to the wrong chapter with the wrong remedy. The
# `isempty(S)` branch is the same condition for the whole matrix, hence vacuous — no rows
# means no deposits means every `gross` is zero and the comprehension returns nothing
# anyway — and is kept only so the shape is explicit rather than implied.
function _unresolvable_expanded(basis::SLCEBasis)::Vector{Int}
    S, gross = _signature_matrix(basis)
    p = n_salcs(basis)
    isempty(S) && return [j for j = 1:p if gross[j] > 0.0]
    return [j for j = 1:p
            if gross[j] > 0.0 && norm(@view S[:, j]) <= _CANCELLATION_RTOL * gross[j]]
end
