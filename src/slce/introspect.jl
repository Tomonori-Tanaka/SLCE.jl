# Fitted-model introspection — a stable, code-neutral view of the terms of a fitted
# SCE. Downstream packages (e.g. the mean-field samplers in `SLCETools.jl`) consume the
# fitted Hamiltonian through this surface instead of reaching into the SALC-basis
# internals (`model.basis.salc_basis.salcs`, `SALCMember` / `SALCTerm` fields), so the
# basis representation can evolve without breaking them.
#
# `decorated_terms` is the general surface and the one new consumers should read: one
# term per (member, slot layout), carrying the per-slot (channel, k, l) labels and the
# scale ALREADY COMPUTED. `multipole_terms` is its frozen pure-spin predecessor — the
# p = 0 view — kept bit-identical for the consumers written against it, and refusing
# any displacement-decorated model rather than mis-scaling it. `bilinear_terms` is the
# `3×3` Cartesian extraction (reusing the validated Sunny conversion in
# `interop/sunny.jl`), and `restrict(model, :spin)` is the bridge that turns a joint
# model into something the pure-spin surfaces accept.
#
# THE SCALE RULE (design record §7). A term's energy contribution carries
# `(4π)^{n_spin_slots/2}` — one `√(4π)` per SPIN slot, because the 4π is an artifact of
# the spin-sphere measure and the displacement kernel is normalized 4π-free (§3). It is
# NOT `(4π)^{body/2}`: those agree only when every site carries exactly one spin factor
# and nothing else, which is precisely the pure-spin case `multipole_terms` covers. Both
# existing consumers derive it from `length(atoms)`, which is why a joint model must not
# reach them. `DecoratedTerm` therefore ships the scale as a field: derive it from slot
# channels, never from the cluster shape.

"""
    MultipoleTerm

One multipole term of a fitted [`SLCEModel`](@ref): the **raw** fitted coefficient
`coef = jϕ_k` (the per-N scale `(4π)^(body/2)` is *not* applied — apply it once, at the
consumer, from `body`), the cluster `body` order, the member `atoms` and their periodic
`shifts`, the per-site angular momenta `ls`, and the rank-`body` real coefficient tensor
`folded`. The term's energy contribution on a configuration `e` (`3 × n_atoms`, unit
columns) is

```
coef · (4π)^(body/2) · Σ_μ folded[μ] ∏ᵢ Z_{ls[i], μ_i}(e_{atoms[i]}),   μ_i = idx_i − ls[i] − 1
```

(the same kernel as [`evaluate_salc`](@ref) for a single SALC term). See [`multipole_terms`](@ref).
"""
struct MultipoleTerm
    coef::Float64
    body::Int
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    ls::Vector{Int}
    folded::Array{Float64}
end

# The number of atoms in the (training-cell) crystal the model was fitted on (a method of
# the exported `n_atoms`, so downstream consumers need not reach into `model.basis.crystal`).
n_atoms(model::SLCEModel)::Int = n_atoms(model.basis.crystal)

"""
    multipole_terms(model::SLCEModel) -> Vector{MultipoleTerm}

A code-neutral, flat view of a fitted SCE: one [`MultipoleTerm`](@ref) per cluster member
and `l`-ordering of every SALC with a nonzero coefficient. This is the stable public
contract a downstream consumer (a mean-field sampler, an energy evaluator, …) reads
*instead of* the SALC-basis internals; the per-N scale `(4π)^(body/2)` is left for the
consumer to apply once, so the returned `coef` is exactly the fitted `jϕ`.

`multipole_terms` is the frozen **pure-spin** surface: a model whose basis carries a
displacement sector throws, and names the two hatches —
[`decorated_terms`](@ref) (the general view) and
`multipole_terms(restrict(model, :spin))` (the clamped-ion sub-model). A
`MultipoleTerm` has no displacement factors, and its consumers derive the scale from
the term shape as `(4π)^(body/2)`, which is only the same thing as the general
`(4π)^{n_spin_slots/2}` rule when every site carries exactly one spin factor — so a
mixed model reaching them would be silently mis-scaled, which is worse than refusing.

The trigger is the **spec**, not the surviving coefficients: a basis whose displacement
sector happened to produce no SALC (or whose displacement couplings all fitted to zero)
was still built and fitted in a `p ≥ 1` setting, and reporting it as pure spin would
fail open on exactly the invariant this refusal exists to enforce.
"""
function multipole_terms(model::SLCEModel)::Vector{MultipoleTerm}
    salcs = model.basis.salc_basis.salcs
    _basis_has_disp(model.basis) &&
        throw(ArgumentError(
            "the model's basis carries a displacement sector; multipole_terms is " *
            "the pure-spin introspection surface (its consumers derive the scale " *
            "as (4π)^(body/2), while the general rule is (4π)^(n_spin_slots/2)). " *
            "Use `decorated_terms(model)` for the general per-slot view, or " *
            "`multipole_terms(restrict(model, :spin))` for the clamped-ion " *
            "(u = 0) sub-model"))
    out = MultipoleTerm[]
    @inbounds for k in eachindex(model.jphi)
        j = model.jphi[k]
        j == 0.0 && continue
        salc = salcs[k]
        for mem in salc.members
            for t in mem.terms
                push!(out, MultipoleTerm(j, salc.body, mem.atoms, mem.shifts,
                                         _term_spin_ls(t), t.folded))
            end
        end
    end
    return out
end

"""
    DecoratedTerm

One term of a fitted [`SLCEModel`](@ref) in the general (channel-decorated) form —
the successor of [`MultipoleTerm`](@ref), and the surface new consumers should read.

| field | meaning |
|:--|:--|
| `coef` | the **raw** fitted coefficient `jϕ_k` (unscaled, as in `MultipoleTerm`) |
| `scale` | `(4π)^(n_spin_slots / 2)` — the factor the energy expression needs |
| `body` | the cluster's site count |
| `atoms`, `shifts` | the member's sites and their periodic images (`shifts[1] = 0`) |
| `slots` | one [`SLCE.SlotRef`](@ref) per tensor axis: which site, and the site factor `(channel, k, l)` on it |
| `folded` | the rank-`length(slots)` real coefficient tensor |

The term's contribution on a configuration `(e, u)` is

```
coef · scale · Σ_μ folded[μ] ∏ᵢ fᵢ(μ_i),   μ_i = idx_i − slots[i].factor.l − 1
```

with `fᵢ` the slot's own factor — `Z_{l,m}(e_a)` on a `SPIN` slot, `|u_a|^{2k}·R_{l,m}(u_a)`
on a `DISP` one (the same kernel [`evaluate_salc`](@ref) runs). Slots are in canonical
axis order (all `SPIN` slots, then all `DISP`, each in site order), and several slots
may share a site: a site is not an axis.

!!! warning "Take `scale` from the field, never from the cluster shape"
    The scale is `(4π)` to the half-power of the **`SPIN` slot count** — the 4π is an
    artifact of the spin-sphere measure and the displacement kernel carries none. It
    coincides with `(4π)^(body/2)` only when every site holds exactly one spin factor
    and nothing else. Deriving it from `body` or `length(atoms)` (as the pure-spin-era
    consumers do) silently mis-scales every mixed term.

See [`decorated_terms`](@ref).
"""
struct DecoratedTerm
    coef::Float64
    scale::Float64
    body::Int
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    slots::Vector{SlotRef}
    folded::Array{Float64}
end

# The one definition of the consumer scale rule (design record §7). Everything that
# needs it — `DecoratedTerm`, the reconstruction gate — calls this, so the rule cannot
# drift into a second, cluster-shaped form.
_slot_scale(slots::AbstractVector{SlotRef})::Float64 =
    (4π)^(count(s -> s.factor.channel == SPIN, slots) / 2)

"""
    decorated_terms(model::SLCEModel) -> Vector{DecoratedTerm}

A code-neutral, flat view of a fitted SCE of any channel content: one
[`DecoratedTerm`](@ref) per cluster member and slot layout of every SALC with a
nonzero coefficient. This is the general successor of [`multipole_terms`](@ref) —
it accepts joint (spin + displacement) models, labels every tensor axis with its
own `(channel, k, l)`, and carries the `(4π)^{n_spin_slots/2}` scale as a field
rather than leaving consumers to re-derive it.

On a pure-spin model the returned terms are the same terms `multipole_terms` reports,
with `scale == (4π)^(body/2)` — the two rules agree exactly there.
"""
function decorated_terms(model::SLCEModel)::Vector{DecoratedTerm}
    salcs = model.basis.salc_basis.salcs
    out = DecoratedTerm[]
    @inbounds for k in eachindex(model.jphi)
        j = model.jphi[k]
        j == 0.0 && continue
        salc = salcs[k]
        for mem in salc.members
            for t in mem.terms
                push!(out, DecoratedTerm(j, _slot_scale(t.slots), salc.body,
                                         mem.atoms, mem.shifts, t.slots, t.folded))
            end
        end
    end
    return out
end

"""
    restrict(model::SLCEModel, channel::Symbol) -> SLCEModel

The exact sub-model of one channel. `channel` must be `:spin`: the result is the
**clamped-ion** model — the energy surface of `model` at `u = 0` — carrying the
pure-spin SALCs of the original basis with their fitted coefficients unchanged, so

```julia
predict_energy(restrict(model, :spin), e) == predict_energy(model, e, zeros(3, n))
```

holds exactly (bitwise), and the pure-spin surfaces ([`multipole_terms`](@ref),
[`bilinear_terms`](@ref), [`to_sunny`](@ref)) accept the result. A model that is
already pure spin is returned unchanged.

!!! warning "`restrict` is not a refit"
    The coefficients you get back are the **reference-geometry clamped-ion**
    couplings: the value each spin interaction takes with every atom at its ideal
    site. They are *not* what fitting a spin-only model to the same data would
    give, and not the couplings a real (relaxed, or finite-temperature) structure
    exhibits.

    A spin-only fit has no displacement coordinate to attribute anything to, so
    whatever the lattice was doing in the training structures gets absorbed into
    the spin couplings — the fitted `J` silently carries a `⟨u⟩`- and
    `⟨uu⟩`-renormalized contribution. `restrict` deliberately drops exactly that:
    it evaluates the joint model at `u = 0`. The difference is the physics the
    joint model was built to separate, not a discrepancy to be reconciled; if you
    want the renormalized couplings at some displacement state, that is a
    thermodynamic average over the joint model, not a restriction of it.
"""
function restrict(model::SLCEModel, channel::Symbol)::SLCEModel
    channel === :spin || throw(ArgumentError(
        "restrict supports channel = :spin (the clamped-ion sub-model); got " *
        ":$channel"))
    basis = model.basis
    _basis_has_disp(basis) || return model         # already the p = 0 view
    keep = findall(is_pure_spin, basis.salc_basis.keys)
    ks = basis.salc_basis.keys[keep]
    sb = SALCBasis(basis.salc_basis.salcs[keep], ks, hash(ks))
    b2 = SLCEBasis(basis.crystal, basis.spacegroup, sb, _spin_spec(basis.spec))
    return SLCEModel(b2, model.j0, model.jphi[keep], ks)
end

# The spec of the restricted basis: `pmax` zeroed and every sector reduced to its
# displacement-degree-0 row. A sector whose degrees start above 0 contributed no
# pure-spin SALC at all and is dropped; so is a purely lattice sector, which has no
# spin content to keep. The spec has to be honest — `_basis_has_disp` reads `pmax`,
# persistence writes the sector table, and a restricted model that still advertised a
# displacement sector would be refused by the very surfaces `restrict` exists to reach.
function _spin_spec(spec::BasisSpec)::BasisSpec
    rules = SectorRule[]
    for r in spec.sectors
        (r.disp_degree[1] == 0 && r.spin_mode !== :none) || continue
        push!(rules, SectorRule(r.spin_mode, r.spin_ls, r.spin_nsites, r.spin_lmax,
                                r.spin_lsum, (0, 0), r.nbody, r.soc, r.cutoff))
    end
    return BasisSpec(spec.nbody, spec.lmax, zeros(Int, length(spec.pmax)), spec.lsum,
                     spec.cutoff, spec.soc, rules, spec.disp_scale,
                     spec.species_labels)
end

"""
    bilinear_terms(model::SLCEModel)

The bilinear pair (`ls=[1,1]`: Heisenberg + Dzyaloshinskii–Moriya + symmetric-anisotropic)
and single-ion (`ls=[2]`) channels of a fitted SCE, extracted as Cartesian `3×3` matrices
and folded onto the training supercell. Returns a named tuple `(; pairs, onsites, skipped)`:

- `pairs::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3}}` — one matrix `M` per bond
  `(a, b, R)` with `a ≤ b` (canonical members carry each physical bond once, both
  directed contributions pre-summed), energy `Σ eₐ'·M·e_b`;
- `onsites::Dict{Int, SMatrix{3,3}}` — one single-ion matrix `A` per atom, energy `eₐ'·A·eₐ`;
- `skipped::Vector{String}` — the higher-order / higher-`l` SALCs not representable as a
  bilinear pair or single-ion term.

Reuses the same validated tesseral→Cartesian conversion as the Sunny export
([`to_sunny`](@ref)); the higher-order channels it lists are kept by the full
[`multipole_terms`](@ref) view.
"""
function bilinear_terms(model::SLCEModel)
    t = _bilinear_terms(model)
    return (; pairs = t.pairs, onsites = t.onsites, skipped = t.skipped)
end
