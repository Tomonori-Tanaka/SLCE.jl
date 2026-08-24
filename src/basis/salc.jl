"""
    SALCKey

Canonical structural address of a SALC (design-matrix column): a tuple of value
labels that identifies the column independent of construction order. The
`(body, orbit_id, decors, L_S, Lf)` part is gauge-invariant; `block` indexes
within a degenerate invariant subspace (made reproducible by the canonical
gauge).

`decors` is the sorted [`SiteDecor`](@ref) multiset label — the joint
(spin + displacement) generalization of the v4 sorted `ls` (a pure-spin key's
label is `spin_decors(ls)`, and [`spin_ls`](@ref) reads the `ls` back). `L_S`
is the total **spin** rank of the spin-first coupling — a good quantum number
of the grey-group projection; for a pure-spin key `L_S == Lf` (the v4 → v5
map).
"""
struct SALCKey
    body::Int
    orbit_id::Int
    decors::Vector{SiteDecor}
    L_S::Int
    Lf::Int
    block::Int
end

_keytuple(k::SALCKey) = (k.body, k.orbit_id, k.decors, k.L_S, k.Lf, k.block)
Base.isless(a::SALCKey, b::SALCKey) = _keytuple(a) < _keytuple(b)
Base.:(==)(a::SALCKey, b::SALCKey) = _keytuple(a) == _keytuple(b)
Base.hash(k::SALCKey, h::UInt) = hash(_keytuple(k), h)

"""
    spin_ls(k::SALCKey) -> Vector{Int}

The spin ranks of the key's spin-carrying decors, in decor order. For a
pure-spin key this is exactly the v4 sorted `ls` label.
"""
spin_ls(k::SALCKey)::Vector{Int} =
    Int[d.spin_l for d in k.decors if has_spin(d)]

"""
    is_pure_spin(k::SALCKey) -> Bool

Whether every decor of the key is a bare spin factor (the v4 decoration shape).
"""
is_pure_spin(k::SALCKey)::Bool = all(is_pure_spin, k.decors)

"""
    is_soc_free(k::SALCKey) -> Bool
    is_soc_free(L_S::Integer) -> Bool

Whether the key belongs to the SOC-less channel: `L_S == 0`, the total-spin-scalar
block. This is the ONE definition two different axes share — the basis-level
truncation `Sector(soc = false)` (which drops every other block at build time,
`basis/salcbasis.jl`) and the fit-level staging selector `:soc_free`
([`SLCE.sector_columns`](@ref), which freezes them instead) — so the two cannot
drift apart. Within the time-reversal-even basis this package builds, `L_S = 0`
plus the even-`Σl` parity rule is exactly the SOC-less content (`L_S = 0` alone
would admit the T-odd scalar chirality).
"""
is_soc_free(L_S::Integer)::Bool = L_S == 0
is_soc_free(k::SALCKey)::Bool = is_soc_free(k.L_S)

"""
    Slot

One tensor axis ("slot") of a SALC term: the member-site index it contracts
against (an index into the member's `atoms`, not an atom number) plus its
decoration factor. Mixed-channel SALCs may carry several slots on one site (a
spin factor and a displacement factor); the slot → site map is what generalizes
the v4 axis-`i` ↔ site-`i` identity, and it is what a
[`DecoratedTerm`](@ref) publishes.
"""
struct Slot
    site::Int
    factor::SiteFactor
end

_slotkey(s::Slot) = (s.factor.channel, s.site, s.factor.k, s.factor.l)

"""
    spin_slots(ls) -> Vector{Slot}

The identity pure-spin slot list of a per-site `l` assignment: axis `i` is
`SiteFactor(SPIN, 0, ls[i])` on site `i` (the v4 term shape).
"""
spin_slots(ls::AbstractVector{<:Integer})::Vector{Slot} =
    Slot[Slot(i, SiteFactor(SPIN, 0, ls[i])) for i in eachindex(ls)]

"""
    SALCTerm

One factor assignment contributing to a (member of a) SALC: the per-axis
[`Slot`](@ref)s (canonical order: `SPIN` axes before `DISP` axes, each by
member-site order) and the real coefficient tensor `folded` (one axis per slot,
`Mf` axis already contracted against the SALC coefficient). A SALC built from a
decoration multiset with unequal factors on symmetry-equivalent sites carries
one term per site→factor assignment; the common case (1/2-body, or equal
factors) is a single term. Pure-spin terms have the identity slot list
`spin_slots(ls)`.
"""
struct SALCTerm
    slots::Vector{Slot}
    folded::Array{Float64}
end

# The SPIN-axis ranks of a term, in slot order (the v4 `ls` view for a
# pure-spin identity term; DISP axes are excluded by construction).
_term_spin_ls(t::SALCTerm)::Vector{Int} =
    Int[s.factor.l for s in t.slots if s.factor.channel == SPIN]

"""
    SALCMember

One concrete cluster instance contributing to a SALC's orbit sum, in the
**canonical form** produced by `_canonicalize_members`: sites sorted by
`(atom, shift)`, shifts re-anchored so `shifts[1] = 0`, and one
[`SALCTerm`](@ref) per distinct site→`l` assignment (each physical instance
appears exactly once per SALC).
"""
struct SALCMember
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    terms::Vector{SALCTerm}
end

"""
    _canonicalize_members(members) -> Vector{SALCMember}

Fold a SALC's transported members into the canonical, duplicate-free form.

The projection/transport construction works with **ordered, anchored** cluster
images (the space in which a stabilizer operation acts by simply permuting site
axes — see `basis/salcbasis.jl`), so the same physical cluster instance arrives
here once per site ordering (`N!` times at `N` distinct sites), each copy
carrying an axis-permuted tensor. The contraction `Σ_μ folded[μ] ∏ᵢ Z_{lsᵢ,μᵢ}`
is invariant under jointly permuting the site slots and the tensor axes, so the
copies fold *exactly* into one member per physical instance:

- sites sorted by `(atom, shift)`; shifts re-anchored to the first sorted site
  (restoring the `shifts[1] = 0` home-cell convention — under periodic
  evaluation and supercell tiling a re-anchored member is the same instance);
- each term's slot sites remapped to the sorted order and its `folded` axes
  brought to the canonical slot order (`permutedims`), then summed per
  resulting site→factor assignment.

Terms whose merged tensor is exactly zero are dropped (they contribute
nothing), and members left with no terms are dropped. Accumulation follows the
stored member/term order and the output is sorted, so the result is
deterministic and idempotent. `Φ(e)` and its gradient are unchanged up to
floating-point regrouping (the merge pre-sums tensors that were previously
summed after contraction).
"""
function _canonicalize_members(members::Vector{SALCMember})::Vector{SALCMember}
    Key = Tuple{Vector{Int},Vector{NTuple{3,Int}}}
    acc = Dict{Key,Vector{Pair{Vector{Slot},Array{Float64}}}}()
    for m in members
        N = length(m.atoms)
        perm = sortperm(1:N; by = i -> (m.atoms[i], Tuple(m.shifts[i])))
        pos = invperm(perm)                  # old site index -> sorted position
        catoms = m.atoms[perm]
        s0 = m.shifts[perm[1]]
        cshifts = NTuple{3,Int}[Tuple(m.shifts[perm[i]] - s0) for i = 1:N]
        terms = get!(() -> Pair{Vector{Slot},Array{Float64}}[], acc,
                     (catoms, cshifts))
        for t in m.terms
            # Remap slot sites to the sorted order, then bring the axes to the
            # canonical slot order (SPIN before DISP, each by site — for a
            # pure-spin identity term this is exactly the old axes-follow-sites
            # `permutedims(folded, perm)`, so the fold is bit-identical to v4).
            remapped = Slot[Slot(pos[s.site], s.factor) for s in t.slots]
            q = sortperm(remapped; by = _slotkey)
            cslots = remapped[q]
            n = length(q)
            pf = q == 1:n ? copy(t.folded) : permutedims(t.folded, q)
            slot = findfirst(p -> first(p) == cslots, terms)
            if slot === nothing
                push!(terms, cslots => pf)
            else
                terms[slot].second .+= pf
            end
        end
    end
    out = SALCMember[]
    for k in sort!(collect(keys(acc)))
        tl = SALCTerm[]
        for (slots, F) in sort(acc[k]; by = p -> map(_slotkey, first(p)))
            any(!=(0.0), F) || continue
            push!(tl, SALCTerm(slots, F))
        end
        isempty(tl) && continue
        shifts = SVector{3,Int}[SVector{3,Int}(s) for s in k[2]]
        push!(out, SALCMember(k[1], shifts, tl))
    end
    return out
end

"""
    SALC

One symmetry-adapted, time-reversal-even scalar basis function over a cluster
orbit. The function is the orbit sum `Φ(e) = (4π)^(N/2) Σ_members Σ_terms Σ_μ
folded[μ] ∏ᵢ Z_{lsᵢ,μᵢ}(e_{site})`, where each member contributes one term per
site→`l` assignment. `decors` / `L_S` / `Lf` mirror the key's gauge-invariant
labels (`decors` is the sorted decoration multiset; the per-term `ls` are its
spin assignments). See [`build_salc_basis`](@ref).
"""
struct SALC
    key::SALCKey
    body::Int
    decors::Vector{SiteDecor}
    L_S::Int
    Lf::Int
    members::Vector{SALCMember}
end

"""
    SALCBasis(salcs, keys)

An ordered list of [`SALC`](@ref)s (sorted by key), the matching `keys` vector
(used to address design-matrix columns), and a structural `fingerprint`
(`hash` of the sorted keys; gauge-independent).

The constructor enforces the column-addressing contract every consumer relies on
— `keys[i] == salcs[i].key`, strictly increasing (hence sorted and injective) —
and *derives* `fingerprint`, so it cannot drift from the keys it summarizes.
There is no field-wise constructor: an unchecked basis would misaddress
coefficients rather than fail.
"""
struct SALCBasis
    salcs::Vector{SALC}
    keys::Vector{SALCKey}
    fingerprint::UInt64

    function SALCBasis(salcs::Vector{SALC}, keys::Vector{SALCKey})
        length(salcs) == length(keys) || throw(DimensionMismatch(
            "SALCBasis: $(length(salcs)) SALCs but $(length(keys)) keys"))
        for i in eachindex(keys)
            salcs[i].key == keys[i] || throw(ArgumentError(
                "SALCBasis: keys[$i] does not mirror salcs[$i].key — design-matrix " *
                "columns are addressed by key, never by construction order"))
            i == 1 && continue
            keys[i - 1] == keys[i] && throw(ArgumentError(
                "SALCBasis: duplicate key at column $i (duplicate design-matrix column)"))
            keys[i - 1] < keys[i] || throw(ArgumentError(
                "SALCBasis: keys are not sorted at column $i — column order is key order"))
        end
        return new(salcs, keys, hash(keys))
    end
end

Base.length(b::SALCBasis) = length(b.salcs)

# Reusable workspace for the SALC evaluation kernels: the dnPl recursion buffer of
# the cache-threaded `Harmonics.Zlm_unsafe` plus per-site harmonic (`Z`) and gradient
# (`∇Z`) tables, all grown on demand. Threading one through `evaluate_salc` /
# `accumulate_grad!` removes every per-term table allocation in the design-matrix hot
# loop; values are bit-identical with or without it (same calls, same order — the
# tables merely change owner). Contents are scratch: never read across calls.
struct SALCScratch
    dnpl::Vector{Float64}                  # Zlm_unsafe recursion workspace
    z::Vector{Vector{Float64}}             # z[i][μ+lᵢ+1] = per-axis factor table
    g::Vector{Vector{SVector{3,Float64}}}  # g[i][μ+lᵢ+1] = per-axis factor gradient
    rl::Vector{Float64}                    # SolidHarmonics batch buffer (disp axes)
    rlg::Vector{Float64}                   # SolidHarmonics gradient pool (3 × n, flat)
end
SALCScratch() = SALCScratch(Vector{Float64}(undef, 4), Vector{Float64}[],
                            Vector{SVector{3,Float64}}[], Vector{Float64}(undef, 4),
                            Vector{Float64}(undef, 12))
# wrap a caller-supplied dnPl vector (the pre-scratch `cache` compatibility surface)
_wrap_scratch(cache::Vector{Float64}) =
    SALCScratch(cache, Vector{Float64}[], Vector{SVector{3,Float64}}[],
                Vector{Float64}(undef, 4), Vector{Float64}(undef, 12))

"""
    evaluate_salc(salc, e[, cache]) -> Float64

Evaluate `Φ(e)` for a single (super)cell spin configuration `e` (`3 × n_atoms`,
unit columns). Periodic with the cell, so lattice shifts map back to the same
column (`site(a, R) → a`). Hot loops may pass `cache` — either a reusable
`Vector{Float64}` (the dnPl recursion workspace of the cache-threaded
`Harmonics.Zlm_unsafe`) or a full `SLCE.SALCScratch` (dnPl workspace plus
the per-site harmonic tables, removing every per-term allocation); values are
identical in all three forms.
"""
evaluate_salc(salc::SALC, e::AbstractMatrix{<:Real})::Float64 =
    evaluate_salc(salc, e, SALCScratch())
evaluate_salc(salc::SALC, e::AbstractMatrix{<:Real}, cache::Vector{Float64})::Float64 =
    evaluate_salc(salc, e, _wrap_scratch(cache))
function evaluate_salc(salc::SALC, e::AbstractMatrix{<:Real},
                       scratch::SALCScratch)::Float64
    # Refuse a displacement-decorated SALC: this form would silently read a
    # DISP rank as a spin harmonic and use the wrong (4π) scale (the same
    # refusing-beats-mis-scaling rule as `spin_multipole_terms`). O(body) per call.
    all(is_pure_spin, salc.decors) || throw(ArgumentError(
        "displacement-decorated SALC: use evaluate_salc(salc, e, u)"))
    scale = (4π)^(salc.body / 2)
    total = 0.0
    @inbounds for m in salc.members
        atoms = m.atoms
        for t in m.terms
            total += _eval_term(t.folded, t.slots, atoms, e, scratch)
        end
    end
    return scale * total
end

# Per-term kernel behind a function barrier: `SALCTerm.folded` is stored as the
# abstract `Array{Float64}` (rank not in the type), so the multi-index loop is
# type-unstable when written inline. Dispatching here specializes on the concrete
# rank `D` (== body order) — same values and the same multiply/accumulate order, so
# the result is bit-identical to the inlined loop.
@inline function _eval_term(folded::Array{Float64,D}, slots::Vector{Slot}, atoms,
                            e::AbstractMatrix{<:Real}, s::SALCScratch) where {D}
    # Per-site harmonic tables. The site direction `u_i` is fixed for this term, so
    # `Z_{lsᵢ,μ}` depends only on `(i, μ)`; tabulate it once over `μ ∈ -lsᵢ:lsᵢ` (the
    # tensor axis index is `index[i] = μ+lsᵢ+1`) instead of recomputing inside the
    # nonzero multi-index loop, where each call hit `dnPl` and dominated the design-
    # matrix cost. Same values, same multiply order ⇒ bit-identical. `u_i` is a unit
    # column by the config contract, so the unchecked harmonic is safe.
    _fill_ztables!(s, Val(D), slots, atoms, e)
    z = s.z
    acc = 0.0
    @inbounds for index in CartesianIndices(folded)
        w = folded[index]
        w == 0.0 && continue
        for i = 1:D
            w *= z[i][index[i]]
        end
        acc += w
    end
    return acc
end

# Fill `s.z[1:D]` with the per-site harmonic tables `s.z[i][μ+lsᵢ+1] = Z_{lsᵢ,μ}(u_i)`
# for one term, growing the pooled table vectors on demand (a pooled vector may stay
# longer than 2lsᵢ+1 from an earlier term; only 1:2lsᵢ+1 is written/read). Same
# evaluation order as the former per-call comprehensions ⇒ identical values, zero
# allocation once the pools are grown.
@inline function _fill_ztables!(s::SALCScratch, ::Val{D}, slots::Vector{Slot}, atoms,
                                e::AbstractMatrix{<:Real}) where {D}
    while length(s.z) < D
        push!(s.z, Float64[])
    end
    @inbounds for i = 1:D
        a = atoms[slots[i].site]
        u = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        li = slots[i].factor.l
        length(s.dnpl) < li + 1 && resize!(s.dnpl, li + 1)
        t = s.z[i]
        length(t) < 2 * li + 1 && resize!(t, 2 * li + 1)
        for μ = -li:li
            t[μ + li + 1] = Harmonics.Zlm_unsafe(li, μ, u, s.dnpl)
        end
    end
    return nothing
end

"""
    accumulate_grad!(G, salc, e, weight[, cache]) -> G

Accumulate `weight · ∇Φ(e)` into `G` (a `3 × n_atoms` buffer), where `∇Φ` is the
per-site direction gradient of the SALC orbit sum: column `a` of `G` receives
`weight · ∂Φ/∂e_a`. Summing this over a model's SALCs (with `weight = jϕ`) yields
`Σ_ϕ jϕ ∇Φ_ϕ`, whose cross product with the spins gives the torque
`τ_a = −e_a × ∂E/∂e_a = ∂E/∂e_a × e_a` (the physical / Landau–Lifshitz sign).

The gradient distributes over the product rule: for each cluster member, each
folded-tensor multi-index `μ`, and each site `i` of the member,
`∂/∂e_{aᵢ} ∏ₖ Zₗₖμₖ = (∏_{k≠i} Zₗₖμₖ) ∇Zₗᵢμᵢ`, landing on column `aᵢ`. A site that
appears more than once in a member contributes once per occurrence (the chain
rule), so repeated sites are handled correctly. `∇Zₗₘ` is the tangent-projected
gradient; the radial part it drops would cancel in `e × ∇Φ` anyway.
"""
accumulate_grad!(G::AbstractMatrix{Float64}, salc::SALC, e::AbstractMatrix{<:Real},
                 weight::Real) = accumulate_grad!(G, salc, e, weight, SALCScratch())
accumulate_grad!(G::AbstractMatrix{Float64}, salc::SALC, e::AbstractMatrix{<:Real},
                 weight::Real, cache::Vector{Float64}) =
    accumulate_grad!(G, salc, e, weight, _wrap_scratch(cache))
function accumulate_grad!(G::AbstractMatrix{Float64}, salc::SALC,
                          e::AbstractMatrix{<:Real}, weight::Real,
                          scratch::SALCScratch)
    all(is_pure_spin, salc.decors) || throw(ArgumentError(
        "displacement-decorated SALC: use the joint two-buffer form " *
        "accumulate_grad!(Ge, Gu, salc, e, u, weight)"))
    weight == 0.0 && return G
    scale = weight * (4π)^(salc.body / 2)
    @inbounds for m in salc.members
        atoms = m.atoms
        for t in m.terms
            _accum_grad_term!(G, t.folded, t.slots, atoms, e, scale, scratch)
        end
    end
    return G
end

# Gradient counterpart of `_eval_term`: a function barrier specializing on the
# concrete rank `D`. The site harmonics `Z` and gradients `∇Z` are tabulated per
# site (same rationale as `_eval_term`: `u_i` is fixed for the term, so each entry
# depends only on `(i, μ)`), then the product-rule expansion reads them back. Same
# expansion, same evaluation order ⇒ the accumulated gradient is bit-identical.
@inline function _accum_grad_term!(G::AbstractMatrix{Float64}, folded::Array{Float64,D},
                                   slots::Vector{Slot}, atoms,
                                   e::AbstractMatrix{<:Real}, scale::Float64,
                                   s::SALCScratch) where {D}
    _fill_ztables!(s, Val(D), slots, atoms, e)
    _fill_gtables!(s, Val(D), slots, atoms, e)
    ztab = s.z
    gtab = s.g
    @inbounds for index in CartesianIndices(folded)
        w = scale * folded[index]
        w == 0.0 && continue
        for i = 1:D
            # leave-one-out product of the other sites' harmonics
            p = 1.0
            for k = 1:D
                k == i && continue
                p *= ztab[k][index[k]]
            end
            p == 0.0 && continue
            gi = gtab[i][index[i]]
            c = w * p
            a = atoms[slots[i].site]
            G[1, a] += c * gi[1]
            G[2, a] += c * gi[2]
            G[3, a] += c * gi[3]
        end
    end
    return G
end

# Gradient sibling of `_fill_ztables!`: `s.g[i][μ+lsᵢ+1] = ∇Z_{lsᵢ,μ}(u_i)`.
@inline function _fill_gtables!(s::SALCScratch, ::Val{D}, slots::Vector{Slot}, atoms,
                                e::AbstractMatrix{<:Real}) where {D}
    while length(s.g) < D
        push!(s.g, SVector{3,Float64}[])
    end
    @inbounds for i = 1:D
        a = atoms[slots[i].site]
        u = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        li = slots[i].factor.l
        length(s.dnpl) < li + 1 && resize!(s.dnpl, li + 1)
        t = s.g[i]
        length(t) < 2 * li + 1 && resize!(t, 2 * li + 1)
        for μ = -li:li
            t[μ + li + 1] = Harmonics.grad_Zlm_unsafe(li, μ, u, s.dnpl)
        end
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Mixed-channel (joint spin–lattice) evaluation — M2b-2
# ---------------------------------------------------------------------------

"""
    evaluate_salc(salc, e, u[, scratch]) -> Float64

Joint evaluation of a (possibly displacement-decorated) SALC: `e` is the
`3 × n_atoms` unit spin configuration, `u` the `3 × n_atoms` Cartesian
displacement field (same column convention; arbitrary norm — the displacement
factors are polynomials, exact at `u = 0`). Spin axes contribute
`Z_{lm}(ê_a)`, displacement axes `|u_a|^{2k} R_{lm}(u_a)` (the 4π-free
`SolidHarmonics` kernel), and the scale is `(4π)^(n_spin/2)` over the SALC's
spin slots — a pure-spin SALC evaluates identically to the two-argument form,
and a displacement-decorated SALC evaluates to exactly `0` at `u = 0` (every
disp factor is homogeneous of degree ≥ 1).
"""
evaluate_salc(salc::SALC, e::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real}) =
    evaluate_salc(salc, e, u, SALCScratch())
function evaluate_salc(salc::SALC, e::AbstractMatrix{<:Real},
                       u::AbstractMatrix{<:Real}, scratch::SALCScratch)::Float64
    size(u) == size(e) || throw(ArgumentError(
        "displacement field u has size $(size(u)); expected $(size(e)) " *
        "(same 3 × n_atoms column convention as the spin configuration)"))
    n_spin = count(has_spin, salc.decors)
    scale = (4π)^(n_spin / 2)
    total = 0.0
    @inbounds for m in salc.members
        atoms = m.atoms
        for t in m.terms
            total += _eval_term_mixed(t.folded, t.slots, atoms, e, u, scratch)
        end
    end
    return scale * total
end

# Mixed sibling of `_eval_term`: per-axis factor tables channel-dispatched.
#
# `D` is the SLOT count, not the body order: a decorated term carries one axis per
# FACTOR, and a site may hold two (a spin rank and a displacement). For a pointed
# `N`-body term that is `N` when the mark's spin rank is 0 — its `|u|²R₀₀` factor is
# the only one it contributes — and `N + 1` when the mark also carries a spin rank.
# (`_eval_term`'s `D == body order` holds because it is the pure-spin kernel, where
# every site contributes exactly one factor.)
@inline function _eval_term_mixed(folded::Array{Float64,D}, slots::Vector{Slot},
                                  atoms, e::AbstractMatrix{<:Real},
                                  u::AbstractMatrix{<:Real},
                                  s::SALCScratch) where {D}
    _fill_ztables_mixed!(s, Val(D), slots, atoms, e, u)
    z = s.z
    acc = 0.0
    @inbounds for index in CartesianIndices(folded)
        w = folded[index]
        w == 0.0 && continue
        for i = 1:D
            w *= z[i][index[i]]
        end
        acc += w
    end
    return acc
end

"""
    accumulate_grad!(Ge, Gu, salc, e, u, weight[, scratch]) -> (Ge, Gu)

Joint gradient of a (possibly displacement-decorated) SALC: accumulate
`weight · ∂Φ/∂e_a` into column `a` of `Ge` and `weight · ∂Φ/∂u_a` into column
`a` of `Gu` (both `3 × n_atoms` buffers, same column convention as `e`/`u`).

Channel conventions (design record §6):

- **spin axes** contribute the tangent-projected direction gradient `∇Z_{lm}`
  (the radial part they drop would cancel in the torque cross product
  `τ_a = ∂E/∂e_a × e_a` — identical to the single-buffer spin-only
  [`accumulate_grad!`](@ref), and bit-identical to it on a pure-spin SALC);
- **displacement axes** contribute the plain Euclidean gradient
  `∂(|u|^{2k} R_{lm}(u))/∂u` with NO projection — the force convention is
  `f_a = −∂E/∂u_a`, so a model's forces are `−Σ_ϕ jϕ · Gu_ϕ`. Displacement
  factors are polynomials, hence smooth at `u = 0` (a degree-1 factor has a
  constant gradient there; higher degrees vanish).

The scale is `(4π)^(n_spin/2)` over the SALC's spin slots, exactly matching
the joint `evaluate_salc(salc, e, u)`; the gradient kernel shares that
evaluator's value tables (`_fill_ztables_mixed!`), and the scale expression —
written in both kernels — is fenced by the gate (j) finite-difference suite.
"""
accumulate_grad!(Ge::AbstractMatrix{Float64}, Gu::AbstractMatrix{Float64},
                 salc::SALC, e::AbstractMatrix{<:Real}, u::AbstractMatrix{<:Real},
                 weight::Real) =
    accumulate_grad!(Ge, Gu, salc, e, u, weight, SALCScratch())
function accumulate_grad!(Ge::AbstractMatrix{Float64}, Gu::AbstractMatrix{Float64},
                          salc::SALC, e::AbstractMatrix{<:Real},
                          u::AbstractMatrix{<:Real}, weight::Real,
                          scratch::SALCScratch)
    size(u) == size(e) || throw(ArgumentError(
        "displacement field u has size $(size(u)); expected $(size(e)) " *
        "(same 3 × n_atoms column convention as the spin configuration)"))
    size(Ge) == size(e) || throw(ArgumentError(
        "spin-gradient buffer Ge has size $(size(Ge)); expected $(size(e))"))
    size(Gu) == size(e) || throw(ArgumentError(
        "displacement-gradient buffer Gu has size $(size(Gu)); expected $(size(e))"))
    Ge === Gu && throw(ArgumentError(
        "Ge and Gu are the same array: the two buffers carry different " *
        "conventions (tangent-projected direction gradient vs Euclidean " *
        "force gradient) and must not be summed"))
    weight == 0.0 && return (Ge, Gu)
    n_spin = count(has_spin, salc.decors)
    scale = weight * (4π)^(n_spin / 2)
    @inbounds for m in salc.members
        atoms = m.atoms
        for t in m.terms
            _accum_grad_term_mixed!(Ge, Gu, t.folded, t.slots, atoms, e, u, scale,
                                    scratch)
        end
    end
    return (Ge, Gu)
end

# Joint sibling of `_accum_grad_term!`: identical product-rule expansion and
# loop order (bit-identity with the spin-only path on pure-spin content), with
# the per-axis value/gradient tables channel-dispatched at fill time.
@inline function _accum_grad_term_mixed!(Ge::AbstractMatrix{Float64},
                                         Gu::AbstractMatrix{Float64},
                                         folded::Array{Float64,D},
                                         slots::Vector{Slot}, atoms,
                                         e::AbstractMatrix{<:Real},
                                         u::AbstractMatrix{<:Real}, scale::Float64,
                                         s::SALCScratch) where {D}
    _fill_ztables_mixed!(s, Val(D), slots, atoms, e, u)
    _fill_gtables_mixed!(s, Val(D), slots, atoms, e, u)
    ztab = s.z
    gtab = s.g
    @inbounds for index in CartesianIndices(folded)
        w = scale * folded[index]
        w == 0.0 && continue
        for i = 1:D
            # leave-one-out product of the other axes' factors
            p = 1.0
            for k = 1:D
                k == i && continue
                p *= ztab[k][index[k]]
            end
            p == 0.0 && continue
            gi = gtab[i][index[i]]
            c = w * p
            a = atoms[slots[i].site]
            if slots[i].factor.channel == SPIN
                Ge[1, a] += c * gi[1]
                Ge[2, a] += c * gi[2]
                Ge[3, a] += c * gi[3]
            else
                Gu[1, a] += c * gi[1]
                Gu[2, a] += c * gi[2]
                Gu[3, a] += c * gi[3]
            end
        end
    end
    return nothing
end

# Joint sibling of `_fill_gtables!`: per-axis factor gradients, channel-
# dispatched. Spin axes reuse the tangent-projected `∇Z_{lm}` kernel; a
# displacement axis `|u|^{2k} R_{lm}(u)` gets the product rule
# `|u|^{2k} ∇R_{lm} + 2k |u|^{2(k−1)} R_{lm} · u` (Euclidean, polynomial-exact;
# the radial term is skipped at k = 0 and vanishes with `u` otherwise). The
# call overwrites `s.rl` with the same values `_fill_ztables_mixed!` produced
# (one shared implementation). Fill the z-tables first as a rule — not a live
# dependency today (both fills recompute `s.rl` per axis before reading it),
# but the rule keeps a future partial-reuse refactor from aliasing.
@inline function _fill_gtables_mixed!(s::SALCScratch, ::Val{D},
                                      slots::Vector{Slot}, atoms,
                                      e::AbstractMatrix{<:Real},
                                      u::AbstractMatrix{<:Real}) where {D}
    while length(s.g) < D
        push!(s.g, SVector{3,Float64}[])
    end
    @inbounds for i = 1:D
        sl = slots[i]
        a = atoms[sl.site]
        li = sl.factor.l
        t = s.g[i]
        length(t) < 2 * li + 1 && resize!(t, 2 * li + 1)
        if sl.factor.channel == SPIN
            ev = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
            length(s.dnpl) < li + 1 && resize!(s.dnpl, li + 1)
            for μ = -li:li
                t[μ + li + 1] = Harmonics.grad_Zlm_unsafe(li, μ, ev, s.dnpl)
            end
        else
            uv = SVector{3,Float64}(u[1, a], u[2, a], u[3, a])
            nsh = SolidHarmonics.num_solid_harmonics(li)
            length(s.rl) < nsh && resize!(s.rl, nsh)
            length(s.rlg) < 3 * nsh && resize!(s.rlg, 3 * nsh)
            Gm = reshape(view(s.rlg, 1:(3 * nsh)), 3, nsh)
            SolidHarmonics.solid_harmonics_grad!(s.rl, Gm, li, uv)
            k = sl.factor.k
            r2 = uv[1] * uv[1] + uv[2] * uv[2] + uv[3] * uv[3]
            r2k = r2^k
            for μ = -li:li
                j = SolidHarmonics.solid_harmonic_index(li, μ)
                g = SVector{3,Float64}(r2k * Gm[1, j], r2k * Gm[2, j],
                                       r2k * Gm[3, j])
                if k >= 1
                    g += (2 * k * r2^(k - 1) * s.rl[j]) .* uv
                end
                t[μ + li + 1] = g
            end
        end
    end
    return nothing
end

@inline function _fill_ztables_mixed!(s::SALCScratch, ::Val{D},
                                      slots::Vector{Slot}, atoms,
                                      e::AbstractMatrix{<:Real},
                                      u::AbstractMatrix{<:Real}) where {D}
    while length(s.z) < D
        push!(s.z, Float64[])
    end
    @inbounds for i = 1:D
        sl = slots[i]
        a = atoms[sl.site]
        li = sl.factor.l
        t = s.z[i]
        length(t) < 2 * li + 1 && resize!(t, 2 * li + 1)
        if sl.factor.channel == SPIN
            ev = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
            length(s.dnpl) < li + 1 && resize!(s.dnpl, li + 1)
            for μ = -li:li
                t[μ + li + 1] = Harmonics.Zlm_unsafe(li, μ, ev, s.dnpl)
            end
        else
            uv = SVector{3,Float64}(u[1, a], u[2, a], u[3, a])
            nsh = SolidHarmonics.num_solid_harmonics(li)
            length(s.rl) < nsh && resize!(s.rl, nsh)
            SolidHarmonics.solid_harmonics!(s.rl, li, uv)
            r2k = (uv[1] * uv[1] + uv[2] * uv[2] + uv[3] * uv[3])^sl.factor.k
            for μ = -li:li
                t[μ + li + 1] = r2k * s.rl[SolidHarmonics.solid_harmonic_index(li, μ)]
            end
        end
    end
    return nothing
end
