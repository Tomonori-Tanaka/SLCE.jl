# Effective models at a displaced reference — the re-expansion of design record §9d.
#
# `effective_model(model; u0)` rewrites the energy of a fitted model as an expansion
# around the displaced structure `R + u0` instead of the reference `R`. This is exact,
# not a Taylor truncation, for the same reason the force constants are exact: every
# displacement site factor `|u|^{2k} R_{lm}(u)` is a HOMOGENEOUS POLYNOMIAL, so the
# shift `u → u0 + δu` acts on the coefficients as a finite linear map. The map is
# **lower triangular in total degree** — a degree-`d` factor lands in degrees `0…d`,
# never higher — which is why the re-expansion generates degree-0 and degree-1 content
# that the reference expansion does not have.
#
# Two consequences, both intended (§9d):
#
#   1. The output CANNOT be an `SLCEModel`. Its symmetry is the stabilizer of `u0`
#      inside the reference group, generally a proper subgroup, so the reference SALCs
#      do not span it. What comes back is the unsymmetrized decorated-monomial form the
#      design record permits: one term per (spin factors) × (displacement monomial).
#   2. `δu = 0` of the effective model is NOT the clamped-ion spin model — it is the
#      `u0`-frozen one. That difference is the entire point: it is what reaches a
#      symmetry-broken distorted phase (a condensed soft mode, a relaxed structure)
#      with no common-subgroup grid, and it is the template for the SCP-style thermal
#      renormalization of the finite-temperature milestone.
#
# The homogeneous-strain pattern `u_i = ε·R_i` must NOT be routed through here: it is
# not cell-periodic (§9d), and the strain response is extracted analytically on
# relative coordinates by `exchange_strain_derivatives` instead. The `strain` keyword
# exists only to say so out loud rather than fail as a `MethodError`.

"""
    EffectiveTerm

One term of an [`EffectiveModel`](@ref): the product

```
coef · ∏ Z_{l,m}(ê_a)  ·  ∏ δu[α, a]^{p_α}
```

`spins` lists `(atom, l, m)` — one tesseral spin factor each, in the same convention as
the rest of the package. `disps` lists `(atom, (p_x, p_y, p_z))` — one Cartesian
displacement monomial per atom that appears, with the zero exponent triple never
stored. Both are sorted, so a term is identified by its content.

A term with an empty `disps` is displacement-independent: those are the couplings of
the `u0`-frozen spin model. A term with an empty `spins` is spin-independent: pure
lattice content, including the reference forces that a displaced structure carries.

!!! warning "`scaled_coef` already carries the `(4π)` scale — do not apply it again"
    This is the OPPOSITE convention from the package's other two public term views,
    and the difference is deliberate rather than an oversight — which is why the
    field is **not** called `coef`. [`SpinMultipoleTerm`](@ref)`.coef` and
    [`DecoratedTerm`](@ref)`.coef` are the raw fitted `jϕ`, with the consumer scale
    left to the caller (`(4π)^(body/2)`) or shipped beside it
    (`DecoratedTerm.scale`). `EffectiveTerm.scaled_coef` has
    `(4π)^{n_spin_slots/2}` — and the SALC's `folded` weight, and the shifted
    polynomial's coefficient — **already folded in**. A consumer migrating from
    either of the other views hits a `type EffectiveTerm has no field coef` error
    instead of a silent `(4π)^{n_spin/2}` over-count.

    It has to be: one `EffectiveTerm` merges contributions from many SALCs, so there
    is no "raw `jϕ`" to hand back. Re-applying `SLCE._slot_scale` to a term from here
    double-counts `4π` per spin slot.
"""
struct EffectiveTerm
    scaled_coef::Float64
    spins::Vector{NTuple{3,Int}}
    disps::Vector{Tuple{Int,NTuple{3,Int}}}
end

"""
    EffectiveModel

A [`SLCEModel`](@ref) re-expanded around a displaced reference structure, as produced
by [`effective_model`](@ref). Evaluate it with
[`predict_energy(::EffectiveModel, e, du)`](@ref).

`u0` is the displacement field the expansion is centred on (`3 × n_atoms`, Cartesian,
in the model's length unit), so the energy is a function of the INCREMENTAL
displacement `δu` measured from `R + u0`:

```
predict_energy(em, e, δu) == predict_energy(model, e, u0 .+ δu)
```

exactly (to roundoff). `j0` absorbs the constant the re-expansion generates, and
`terms` carries everything else.

!!! note "This is not a `SLCEModel`, and cannot be"
    The symmetry of the displaced structure is the stabilizer of `u0` in the reference
    group — generally a proper subgroup — so the reference SALCs cannot span the
    result. `EffectiveModel` is the unsymmetrized decorated-monomial form instead: it
    evaluates and differentiates, but it carries no `SALCKey`s, is not fittable, and
    is not persisted.
"""
struct EffectiveModel
    crystal::Crystal
    u0::Matrix{Float64}
    j0::Float64
    terms::Vector{EffectiveTerm}
end

Base.length(em::EffectiveModel) = length(em.terms)

function Base.show(io::IO, em::EffectiveModel)
    d = isempty(em.terms) ? 0 :
        maximum(sum(sum(p) for (_, p) in t.disps; init = 0) for t in em.terms)
    print(io, "EffectiveModel(", length(em.terms), " terms, max δu degree ", d,
          ", ", n_atoms(em.crystal), " atoms)")
end

const _EMPTY_DISPS = Tuple{Int,NTuple{3,Int}}[]

# `P(u0 + δ)` as a polynomial in `δ`, given `P`'s monomial dictionary. Each monomial
# `∏_α u_α^{p_α}` expands binomially, `∏_α Σ_{j_α ≤ p_α} C(p_α, j_α) u0_α^{p_α−j_α}
# δ_α^{j_α}`, and the δ-monomials collect across the whole expansion. The result has
# degrees `0 … deg P` — the lower-triangular structure, realized.
function _shift_poly(poly::SolidHarmonics._Poly,
                     u0::SVector{3,Float64})::SolidHarmonics._Poly
    out = SolidHarmonics._Poly()
    for (p, c) in poly
        for j1 = 0:p[1], j2 = 0:p[2], j3 = 0:p[3]
            w = c * binomial(p[1], j1) * binomial(p[2], j2) * binomial(p[3], j3) *
                u0[1]^(p[1] - j1) * u0[2]^(p[2] - j2) * u0[3]^(p[3] - j3)
            w == 0.0 && continue
            key = (j1, j2, j3)
            out[key] = get(out, key, 0.0) + w
        end
    end
    return out
end

const _EffKey = Tuple{Vector{NTuple{3,Int}},Vector{Tuple{Int,NTuple{3,Int}}}}

"""
    effective_model(model::SLCEModel; u0, atol = 0.0) -> EffectiveModel

Re-expand `model` around the displaced reference `R + u0`, exactly.

`u0` is a `3 × n_atoms` Cartesian displacement field in the model's length unit. The
result satisfies

```
predict_energy(effective_model(model; u0), e, δu) == predict_energy(model, e, u0 .+ δu)
```

for every spin configuration `e` and every increment `δu`, to roundoff — it is an
exact change of expansion point, not a Taylor truncation, because every displacement
site factor is a homogeneous polynomial and the shift is therefore a finite linear map
on the coefficients (design record §9d).

Use it to reach a symmetry-broken distorted structure — a relaxed cell, a condensed
soft mode — without building a common-subgroup grid, and as the starting point for
renormalizing coefficients onto a thermally displaced reference.

`atol` prunes terms whose accumulated coefficient falls at or below it in absolute
value; the default `0.0` drops only the exact cancellations. Raising it makes the
model smaller and inexact, so the exactness identity above no longer holds — raise it
only when you have decided what error you can afford.

!!! warning "The `u0` = 0 point moves, on purpose"
    The re-expansion generates degree-0 and degree-1 displacement content, so
    `predict_energy(em, e, zeros(3, n))` is the `u0`-FROZEN spin energy, not the
    clamped-ion one, and the effective model carries reference forces that the
    original does not. That is what makes it a model of the displaced structure.

Throws on a pure-spin model (nothing to re-expand — `u0` could not change it) and on a
size mismatch. A homogeneous strain must not be passed as `u0 = ε·R`: the affine field
is not cell-periodic, and the strain response is extracted analytically on relative
coordinates instead.

See also [`EffectiveModel`](@ref), [`force_constants`](@ref), [`restrict`](@ref).
"""
function effective_model(model::SLCEModel; u0::AbstractMatrix{<:Real},
                         atol::Real = 0.0, strain = nothing)::EffectiveModel
    strain === nothing || throw(ArgumentError(
        "effective_model re-expands around a DISPLACEMENT field, not a strain: the " *
        "affine pattern u_i = ε·R_i is not cell-periodic and must never be routed " *
        "through a periodic evaluator (design record §9d). The homogeneous-strain " *
        "response is extracted analytically on the relative coordinates " *
        "u_i − u_j = ε·d_ij instead"))
    _basis_has_disp(model.basis) || throw(ArgumentError(
        "effective_model needs a displacement-decorated model: this one is pure " *
        "spin, so re-expanding it around u0 would return it unchanged. A pure-spin " *
        "model already describes one fixed geometry"))
    nat = n_atoms(model.basis.crystal)
    size(u0) == (3, nat) || throw(DimensionMismatch(
        "u0 is $(size(u0)); expected (3, $nat) — one Cartesian column per atom"))
    all(isfinite, u0) || throw(ArgumentError("u0 must be finite"))
    (isfinite(atol) && atol >= 0) ||
        throw(ArgumentError("atol must be finite and ≥ 0; got $atol"))
    U0 = Matrix{Float64}(u0)
    acc = Dict{_EffKey,Float64}()
    # Shifted polynomials are keyed by (k, l, m, atom): the same site factor recurs
    # across members and terms, and shifting it is the only nontrivial work here.
    cache = Dict{NTuple{4,Int},SolidHarmonics._Poly}()
    salcs = model.basis.salc_basis.salcs
    for i in eachindex(model.jphi)
        jphi = model.jphi[i]
        jphi == 0.0 && continue
        salc = salcs[i]
        weight = jphi * (4π)^(count(has_spin, salc.decors) / 2)
        for mem in salc.members
            for t in mem.terms
                _accumulate_effective!(acc, weight, t, mem, U0, cache)
            end
        end
    end
    # Fold the pure constant into j0 and drop it from the term list; a spin-only or
    # displacement-only term is content, but a term with neither is a number.
    j0 = model.j0
    terms = EffectiveTerm[]
    for (key, c) in acc
        abs(c) > atol || continue
        if isempty(key[1]) && isempty(key[2])
            j0 += c
        else
            push!(terms, EffectiveTerm(c, key[1], key[2]))
        end
    end
    # Sorted so the artifact is reproducible: `Dict` iteration order is an
    # implementation detail, and this object is meant to be compared and diffed.
    sort!(terms; by = t -> (t.disps, t.spins))
    return EffectiveModel(model.basis.crystal, U0, j0, terms)
end

# One SALC term's contribution. The term is a sum over `folded`'s index grid; each
# index fixes an `m` per axis, after which the product factorizes site by site — spin
# axes stay symbolic (they are untouched by a displacement shift), displacement axes
# become the shifted polynomial of their own site factor.
function _accumulate_effective!(acc::Dict{_EffKey,Float64}, weight::Float64,
                                t::SALCTerm, mem::SALCMember, U0::Matrix{Float64},
                                cache::Dict{NTuple{4,Int},SolidHarmonics._Poly})
    D = length(t.slots)
    sslots = [i for i = 1:D if t.slots[i].factor.channel == SPIN]
    dslots = [i for i = 1:D if t.slots[i].factor.channel == DISP]
    polys = Vector{SolidHarmonics._Poly}(undef, length(dslots))
    datoms = Vector{Int}(undef, length(dslots))
    for idx in CartesianIndices(t.folded)
        w = t.folded[idx]
        w == 0.0 && continue
        spins = Vector{NTuple{3,Int}}(undef, length(sslots))
        for (n, i) in enumerate(sslots)
            sl = t.slots[i]
            l = sl.factor.l
            spins[n] = (mem.atoms[sl.site], l, idx[i] - l - 1)
        end
        sort!(spins)
        if isempty(dslots)
            _push_effective!(acc, spins, _EMPTY_DISPS, weight * w)
            continue
        end
        for (n, i) in enumerate(dslots)
            sl = t.slots[i]
            a = mem.atoms[sl.site]
            l = sl.factor.l
            key = (sl.factor.k, l, idx[i] - l - 1, a)
            polys[n] = get!(cache, key) do
                _shift_poly(SolidHarmonics.solid_harmonic_poly(key[1], key[2], key[3]),
                            SVector{3,Float64}(U0[1, a], U0[2, a], U0[3, a]))
            end
            datoms[n] = a
        end
        _expand_disp_product!(acc, weight * w, spins, polys, datoms)
    end
    return nothing
end

# Multiply the slots' shifted polynomials together and push each resulting monomial.
#
# The variables are per ATOM, not per slot: two displacement slots on member sites that
# resolve to the same atom (the same atom in different cells — displacements are
# cell-periodic) are the SAME variable, so their exponents add. Reachable through an
# `AllImages` self-bond, and gated there.
#
# To be precise about what this buys, since the temptation is to overstate it: keying by
# slot instead would give the same ENERGY, because evaluation multiplies the factors
# either way. What it would break is the canonical form — one entry per (spins,
# monomial) — so like terms would not merge, coefficients would arrive split across
# duplicates, exact cancellations would not happen, and `atol` would prune halves of a
# term that the merged coefficient would have kept.
function _expand_disp_product!(acc::Dict{_EffKey,Float64}, base::Float64,
                               spins::Vector{NTuple{3,Int}},
                               polys::Vector{SolidHarmonics._Poly},
                               datoms::Vector{Int})
    n = length(polys)
    items = [collect(p) for p in polys]
    lens = [length(v) for v in items]
    any(iszero, lens) && return nothing
    at = ones(Int, n)
    ex = Dict{Int,MVector{3,Int}}()
    while true
        c = base
        for j = 1:n
            c *= items[j][at[j]].second
            c == 0.0 && break
        end
        if c != 0.0
            empty!(ex)
            for j = 1:n
                v = get!(ex, datoms[j]) do
                    MVector(0, 0, 0)
                end
                mono = items[j][at[j]].first
                v[1] += mono[1]
                v[2] += mono[2]
                v[3] += mono[3]
            end
            disps = Tuple{Int,NTuple{3,Int}}[]
            for a in sort!(collect(keys(ex)))
                v = ex[a]
                (v[1] == 0 && v[2] == 0 && v[3] == 0) && continue
                push!(disps, (a, (v[1], v[2], v[3])))
            end
            _push_effective!(acc, spins, disps, c)
        end
        j = n
        while j >= 1
            at[j] += 1
            at[j] <= lens[j] && break
            at[j] = 1
            j -= 1
        end
        j == 0 && break
    end
    return nothing
end

function _push_effective!(acc::Dict{_EffKey,Float64}, spins::Vector{NTuple{3,Int}},
                          disps::Vector{Tuple{Int,NTuple{3,Int}}}, c::Float64)
    key = (spins, disps)
    acc[key] = get(acc, key, 0.0) + c
    return nothing
end

"""
    predict_energy(em::EffectiveModel, e, du) -> Float64

Energy of the effective model at spin configuration `e` (`3 × n_atoms`, unit columns)
and INCREMENTAL displacement `du` (`3 × n_atoms`, Cartesian), measured from the
displaced reference `R + em.u0`.

By construction this equals `predict_energy(model, e, em.u0 .+ du)` for the model the
effective one came from. Note that `du = 0` is the `u0`-frozen point, not the
clamped-ion one.
"""
function predict_energy(em::EffectiveModel, e::AbstractMatrix{<:Real},
                        du::AbstractMatrix{<:Real})::Float64
    nat = n_atoms(em.crystal)
    size(e) == (3, nat) || throw(DimensionMismatch(
        "spin configuration is $(size(e)); expected (3, $nat)"))
    size(du) == size(e) || throw(ArgumentError(
        "displacement increment du has size $(size(du)); expected $(size(e)) " *
        "(same 3 × n_atoms column convention as the spin configuration)"))
    total = em.j0
    @inbounds for t in em.terms
        w = t.scaled_coef
        for (a, l, m) in t.spins
            w *= Harmonics.Zlm(l, m, SVector{3,Float64}(e[1, a], e[2, a], e[3, a]))
            w == 0.0 && break
        end
        w == 0.0 && continue
        for (a, p) in t.disps
            p[1] == 0 || (w *= du[1, a]^p[1])
            p[2] == 0 || (w *= du[2, a]^p[2])
            p[3] == 0 || (w *= du[3, a]^p[3])
        end
        total += w
    end
    return total
end
