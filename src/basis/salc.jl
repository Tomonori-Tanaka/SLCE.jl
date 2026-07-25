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
    SlotRef

One tensor axis ("slot") of a [`SALCTerm`](@ref): the member-site index it
contracts against plus its decoration factor. Mixed-channel SALCs may carry
several slots on one site (a spin factor and a displacement factor); the slot →
site map is what generalizes the v4 axis-`i` ↔ site-`i` identity.
"""
struct SlotRef
    site::Int
    factor::SiteFactor
end

_slotkey(s::SlotRef) = (s.factor.channel, s.site, s.factor.k, s.factor.l)

"""
    spin_slots(ls) -> Vector{SlotRef}

The identity pure-spin slot list of a per-site `l` assignment: axis `i` is
`SiteFactor(SPIN, 0, ls[i])` on site `i` (the v4 term shape).
"""
spin_slots(ls::AbstractVector{<:Integer})::Vector{SlotRef} =
    SlotRef[SlotRef(i, SiteFactor(SPIN, 0, ls[i])) for i in eachindex(ls)]

"""
    SALCTerm

One factor assignment contributing to a (member of a) SALC: the per-axis
[`SlotRef`](@ref)s (canonical order: `SPIN` axes before `DISP` axes, each by
member-site order) and the real coefficient tensor `folded` (one axis per slot,
`Mf` axis already contracted against the SALC coefficient). A SALC built from a
decoration multiset with unequal factors on symmetry-equivalent sites carries
one term per site→factor assignment; the common case (1/2-body, or equal
factors) is a single term. Pure-spin terms have the identity slot list
`spin_slots(ls)`.
"""
struct SALCTerm
    slots::Vector{SlotRef}
    folded::Array{Float64}
end

# The per-axis spin ranks of a pure-spin term (the v4 `ls` view).
_term_spin_ls(t::SALCTerm)::Vector{Int} = Int[s.factor.l for s in t.slots]

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
    acc = Dict{Key,Vector{Pair{Vector{SlotRef},Array{Float64}}}}()
    for m in members
        N = length(m.atoms)
        perm = sortperm(1:N; by = i -> (m.atoms[i], Tuple(m.shifts[i])))
        pos = invperm(perm)                  # old site index -> sorted position
        catoms = m.atoms[perm]
        s0 = m.shifts[perm[1]]
        cshifts = NTuple{3,Int}[Tuple(m.shifts[perm[i]] - s0) for i = 1:N]
        terms = get!(() -> Pair{Vector{SlotRef},Array{Float64}}[], acc,
                     (catoms, cshifts))
        for t in m.terms
            # Remap slot sites to the sorted order, then bring the axes to the
            # canonical slot order (SPIN before DISP, each by site — for a
            # pure-spin identity term this is exactly the old axes-follow-sites
            # `permutedims(folded, perm)`, so the fold is bit-identical to v4).
            remapped = SlotRef[SlotRef(pos[s.site], s.factor) for s in t.slots]
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
    SALCBasis

An ordered list of [`SALC`](@ref)s (sorted by key), the matching `keys` vector
(used to address design-matrix columns), and a structural `fingerprint`
(`hash` of the sorted keys; gauge-independent).
"""
struct SALCBasis
    salcs::Vector{SALC}
    keys::Vector{SALCKey}
    fingerprint::UInt64
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
    z::Vector{Vector{Float64}}             # z[i][μ+lᵢ+1] = Z_{lᵢμ}(u_i), per term site
    g::Vector{Vector{SVector{3,Float64}}}  # g[i][μ+lᵢ+1] = ∇Z_{lᵢμ}(u_i)
end
SALCScratch() = SALCScratch(Vector{Float64}(undef, 4), Vector{Float64}[],
                            Vector{SVector{3,Float64}}[])
# wrap a caller-supplied dnPl vector (the pre-scratch `cache` compatibility surface)
_wrap_scratch(cache::Vector{Float64}) =
    SALCScratch(cache, Vector{Float64}[], Vector{SVector{3,Float64}}[])

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
@inline function _eval_term(folded::Array{Float64,D}, slots::Vector{SlotRef}, atoms,
                            e::AbstractMatrix{<:Real}, s::SALCScratch) where {D}
    # Per-site harmonic tables. The site direction `u_i` is fixed for this term, so
    # `Z_{lsᵢ,μ}` depends only on `(i, μ)`; tabulate it once over `μ ∈ -lsᵢ:lsᵢ` (the
    # tensor axis index is `idx[i] = μ+lsᵢ+1`) instead of recomputing inside the
    # nonzero multi-index loop, where each call hit `dnPl` and dominated the design-
    # matrix cost. Same values, same multiply order ⇒ bit-identical. `u_i` is a unit
    # column by the config contract, so the unchecked harmonic is safe.
    _fill_ztables!(s, Val(D), slots, atoms, e)
    z = s.z
    acc = 0.0
    @inbounds for idx in CartesianIndices(folded)
        w = folded[idx]
        w == 0.0 && continue
        for i = 1:D
            w *= z[i][idx[i]]
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
@inline function _fill_ztables!(s::SALCScratch, ::Val{D}, slots::Vector{SlotRef}, atoms,
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
                                   slots::Vector{SlotRef}, atoms,
                                   e::AbstractMatrix{<:Real}, scale::Float64,
                                   s::SALCScratch) where {D}
    _fill_ztables!(s, Val(D), slots, atoms, e)
    _fill_gtables!(s, Val(D), slots, atoms, e)
    ztab = s.z
    gtab = s.g
    @inbounds for idx in CartesianIndices(folded)
        w = scale * folded[idx]
        w == 0.0 && continue
        for i = 1:D
            # leave-one-out product of the other sites' harmonics
            p = 1.0
            for k = 1:D
                k == i && continue
                p *= ztab[k][idx[k]]
            end
            p == 0.0 && continue
            gi = gtab[i][idx[i]]
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
@inline function _fill_gtables!(s::SALCScratch, ::Val{D}, slots::Vector{SlotRef}, atoms,
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
