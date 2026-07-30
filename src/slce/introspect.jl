# Fitted-model introspection — a stable, code-neutral view of the terms of a fitted
# SLCE. Downstream packages (e.g. the mean-field samplers in `SLCETools.jl`) consume the
# fitted Hamiltonian through this surface instead of reaching into the SALC-basis
# internals (`model.basis.salc_basis.salcs`, `SALCMember` / `SALCTerm` fields), so the
# basis representation can evolve without breaking them.
#
# `decorated_terms` is the general surface and the one new consumers should read: one
# term per (member, slot layout), carrying the per-slot (channel, k, l) labels and the
# scale ALREADY COMPUTED. `spin_multipole_terms` is its frozen pure-spin predecessor — the
# p = 0 view — kept bit-identical for the consumers written against it, and refusing
# any displacement-decorated model rather than mis-scaling it. `bilinear_terms` is the
# `3×3` Cartesian extraction (reusing the validated Sunny conversion in
# `interop/sunny.jl`), and `restrict(model, :spin)` is the bridge that turns a joint
# model into something the pure-spin surfaces accept.
#
# THE SCALE RULE (design record §7). A term's energy contribution carries
# `(4π)^{n_spin_slots/2}` — one `√(4π)` per SPIN slot, because the 4π is an artifact of
# the spin-sphere measure and the displacement kernel is normalized 4π-free (§3). It is
# NOT `(4π)^{body/2}`: those agree exactly when the SPIN-slot count equals the body order
# — every site carrying a spin factor, with or without a displacement one alongside, which
# covers the pure-spin case `spin_multipole_terms` is restricted to. Both
# existing consumers derive it from `length(atoms)`, which is why a joint model must not
# reach them. `DecoratedTerm` therefore ships the scale as a field: derive it from slot
# channels, never from the cluster shape.

"""
    SpinMultipoleTerm

One multipole term of a fitted [`SLCEModel`](@ref): the **raw** fitted coefficient
`coef = jϕ_k` (the per-N scale `(4π)^(body/2)` is *not* applied — apply it once, at the
consumer, from `body`), the cluster `body` order, the member `atoms` and their periodic
`shifts`, the per-site angular momenta `ls`, and the rank-`body` real coefficient tensor
`folded`. The term's energy contribution on a configuration `e` (`3 × n_atoms`, unit
columns) is

```
coef · (4π)^(body/2) · Σ_μ folded[μ] ∏ᵢ Z_{ls[i], μ_i}(e_{atoms[i]}),   μ_i = idx_i − ls[i] − 1
```

(the same kernel as [`evaluate_salc`](@ref) for a single SALC term). See [`spin_multipole_terms`](@ref).
"""
struct SpinMultipoleTerm
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
    spin_multipole_terms(model::SLCEModel; keep_zero = false) -> Vector{SpinMultipoleTerm}

A code-neutral, flat view of a fitted SLCE: one [`SpinMultipoleTerm`](@ref) per cluster member
and `l`-ordering of every SALC with a nonzero coefficient. This is the stable public
contract a downstream consumer (a mean-field sampler, an energy evaluator, …) reads
*instead of* the SALC-basis internals; the per-N scale `(4π)^(body/2)` is left for the
consumer to apply once, so the returned `coef` is exactly the fitted `jϕ`.

`spin_multipole_terms` is the frozen **pure-spin** surface: a model whose basis carries a
displacement sector throws, and names the two hatches —
[`decorated_terms`](@ref) (the general view) and
`spin_multipole_terms(restrict(model, :spin))` (the clamped-ion sub-model). A
`SpinMultipoleTerm` has no displacement factors, and its consumers derive the scale from
the term shape as `(4π)^(body/2)`, which is only the same thing as the general
`(4π)^{n_spin_slots/2}` rule when every site carries exactly one spin factor — so a
mixed model reaching them would be silently mis-scaled, which is worse than refusing.

The trigger is the **spec**, not the surviving coefficients: a basis whose displacement
sector happened to produce no SALC (or whose displacement couplings all fitted to zero)
was still built and fitted in a `p ≥ 1` setting, and reporting it as pure spin would
fail open on exactly the invariant this refusal exists to enforce.

`keep_zero = true` emits a term for every SALC member regardless of its coefficient — see
[`decorated_terms`](@ref) for why that matters to any consumer that rewrites coefficients
in place.
"""
function spin_multipole_terms(model::SLCEModel;
                              keep_zero::Bool = false)::Vector{SpinMultipoleTerm}
    salcs = model.basis.salc_basis.salcs
    _basis_has_disp(model.basis) &&
        throw(ArgumentError(
            "the model's basis carries a displacement sector; spin_multipole_terms is " *
            "the pure-spin introspection surface (its consumers derive the scale " *
            "as (4π)^(body/2), while the general rule is (4π)^(n_spin_slots/2)). " *
            "Use `decorated_terms(model)` for the general per-slot view, or " *
            "`spin_multipole_terms(restrict(model, :spin))` for the clamped-ion " *
            "(u = 0) sub-model"))
    out = SpinMultipoleTerm[]
    @inbounds for k in eachindex(model.jphi)
        j = model.jphi[k]
        (keep_zero || j != 0.0) || continue
        salc = salcs[k]
        for mem in salc.members
            for t in mem.terms
                push!(out, SpinMultipoleTerm(j, salc.body, mem.atoms, mem.shifts,
                                         _term_spin_ls(t), t.folded))
            end
        end
    end
    return out
end

"""
    DecoratedTerm

One term of a fitted [`SLCEModel`](@ref) in the general (channel-decorated) form —
the successor of [`SpinMultipoleTerm`](@ref), and the surface new consumers should read.

| field | meaning |
|:--|:--|
| `coef` | the **raw** fitted coefficient `jϕ_k` (unscaled, as in `SpinMultipoleTerm`) |
| `scale` | `(4π)^(n_spin_slots / 2)` — the factor the energy expression needs |
| `body` | the cluster's site count |
| `atoms`, `shifts` | the member's sites and their periodic images (`shifts[1] = 0`) |
| `slots` | one [`SLCE.Slot`](@ref) per tensor axis: which site, and the site factor `(channel, k, l)` on it |
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
    coincides with `(4π)^(body/2)` exactly when the spin-slot count equals the body
    order — every site holding a spin factor, alone or beside a displacement one — and
    parts ways as soon as one site is pure displacement. Deriving it from `body` or
    `length(atoms)` (as the pure-spin-era consumers do) silently mis-scales those terms.

See [`decorated_terms`](@ref).
"""
struct DecoratedTerm
    coef::Float64
    scale::Float64
    body::Int
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    slots::Vector{Slot}
    folded::Array{Float64}
end

# The one definition of the consumer scale rule (design record §7). Everything that
# needs it — `DecoratedTerm`, the reconstruction gate — calls this, so the rule cannot
# drift into a second, cluster-shaped form.
_slot_scale(slots::AbstractVector{Slot})::Float64 =
    (4π)^(count(s -> s.factor.channel == SPIN, slots) / 2)

"""
    decorated_terms(model::SLCEModel; keep_zero = false) -> Vector{DecoratedTerm}

A code-neutral, flat view of a fitted SLCE of any channel content: one
[`DecoratedTerm`](@ref) per cluster member and slot layout of every SALC with a
nonzero coefficient. This is the general successor of [`spin_multipole_terms`](@ref) —
it accepts joint (spin + displacement) models, labels every tensor axis with its
own `(channel, k, l)`, and carries the `(4π)^{n_spin_slots/2}` scale as a field
rather than leaving consumers to re-derive it.

On a pure-spin model the returned terms are the same terms `spin_multipole_terms` reports,
with `scale == (4π)^(body/2)` — the two rules agree exactly there.

!!! warning "`keep_zero = true` when the coefficients will be rewritten in place"
    By default a SALC whose coefficient is exactly zero emits no term, so **the term list
    — and therefore the index → SALC map a consumer addresses it by — is a function of the
    coefficient VALUES, not of the basis**. That is harmless for a reader and wrong for a
    rewriter: sparse estimators and `refit` produce exact zeros routinely, so two models
    on the SAME basis (two points of a [`StrainedModels`](@ref) grid, an active-learning
    refit) can emit lists that differ in length — or, worse, lists of EQUAL length whose
    maps are shifted relative to each other, at which point a coefficient hot-swap writes
    each value onto a neighbouring cluster and every length check still passes.

    `keep_zero = true` emits one term per SALC member unconditionally, making the index
    map a property of `salc_basis` alone — which is exactly the object a volume grid
    asserts identical across its points. Any consumer that will later rewrite coefficients
    in place (`SLCEMonteCarlo.set_coefficients!`) must build from this view.

    The default stays `false` so every existing consumer, byte-comparison and benchmark is
    unaffected.
"""
function decorated_terms(model::SLCEModel;
                         keep_zero::Bool = false)::Vector{DecoratedTerm}
    # The Monte-Carlo hand-off is where a frozen channel stops being a bookkeeping
    # nicety: a supercell resolves the tie, so the coefficient this cell could not
    # determine multiplies a nonzero function throughout the run.
    _warn_unresolvable(model, "decorated_terms")
    salcs = model.basis.salc_basis.salcs
    out = DecoratedTerm[]
    @inbounds for k in eachindex(model.jphi)
        j = model.jphi[k]
        (keep_zero || j != 0.0) || continue
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
    restrict(model::SLCEModel, sector::Symbol) -> SLCEModel

The exact sub-model of one sector, `:spin` or `:lattice` — the same two selector
names [`SLCE.sector_columns`](@ref) uses, and deliberately *not* the [`SLCE.Channel`](@ref)
enum: `:lattice` is "carries no spin factor", which is a property of a whole SALC,
while `DISP` labels one site factor.

`:lattice` keeps the SALCs carrying **no spin factor at all** — the part of the
energy surface no magnetic state can change. Its predictions are independent of `e`
by construction, so it takes `nothing` in the spin slot and needs no `spins` argument
anywhere:

```julia
lat = restrict(model, :lattice)
predict_energy(lat, nothing, u)          # ≡ predict_energy(lat, any_e, u)
force_constants(lat; order = 2)          # the spin-independent force constants
```

The complement is the physics the joint expansion exists to separate:
`predict_energy(model, e, u) − predict_energy(lat, nothing, u)` is everything that
depends on the magnetic state. Note that this is *not* a spin average — a term with a
rank-0 spin factor is constant in `e` yet still carries a spin slot, so it lives in
the remainder, not in `:lattice`.

`:spin` is the **clamped-ion** model — the energy surface of `model` at `u = 0` —
carrying the pure-spin SALCs of the original basis with their fitted coefficients
unchanged, so

```julia
predict_energy(restrict(model, :spin), e) == predict_energy(model, e, zeros(3, n))
```

holds exactly (bitwise), and the pure-spin surfaces ([`spin_multipole_terms`](@ref),
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
function restrict(model::SLCEModel, sector::Symbol)::SLCEModel
    sector in (:spin, :lattice) || throw(ArgumentError(
        "restrict supports sector = :spin (the clamped-ion sub-model) or " *
        ":lattice (the spin-independent sub-model); got :$sector"))
    basis = model.basis
    if sector === :spin
        _basis_has_disp(basis) || return model     # already the p = 0 view
        keep = findall(is_pure_spin, basis.salc_basis.keys)
        spec = _spin_spec(basis.spec)
    else
        _basis_has_spin(basis) || return model     # already spin-free
        keep = findall(k -> !any(has_spin, k.decors), basis.salc_basis.keys)
        spec = _lattice_spec(basis.spec)
    end
    ks = basis.salc_basis.keys[keep]
    sb = SALCBasis(basis.salc_basis.salcs[keep], ks)
    b2 = SLCEBasis(basis.crystal, basis.spacegroup, sb, spec)
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
    for r in spec.sector_rules
        (r.disp_degree[1] == 0 && r.spin_mode !== :none) || continue
        push!(rules, SectorRule(r.spin_mode, r.spin_ls, r.spin_nsites, r.spin_lmax,
                                r.spin_lsum, (0, 0), r.sites, r.soc, r.cutoff))
    end
    return BasisSpec(spec.nbody, spec.lmax, zeros(Int, length(spec.pmax)), spec.lsum,
                     spec.cutoff, spec.soc, rules, spec.disp_scale,
                     spec.species_labels)
end

# The mirror for `:lattice`: `lmax` zeroed and every sector reduced to its spin-free
# rows. Same honesty requirement as `_spin_spec`, pointing the other way — the spec is
# what `_basis_has_spin` reads, so a restricted model that still advertised a spin
# sector would keep demanding the very `spins` argument `restrict` exists to remove.
function _lattice_spec(spec::BasisSpec)::BasisSpec
    rules = SectorRule[r for r in spec.sector_rules
                       if r.spin_mode === :none && r.disp_degree[2] > 0]
    # No spin-FREE displacement sector means the lattice sub-model has no displacement
    # content either (spin-free SALCs come only from `spin_mode === :none` rows), so `pmax`
    # must be zeroed along with the rules. Keeping the cap while dropping every rule made
    # the `BasisSpec` constructor refuse its own dense form — "pmax > 0 needs a sector
    # table with displacement content" — which named a keyword the caller had passed and
    # made `restrict(model, :lattice)` unusable on exactly the minimal magnetoelastic spec
    # (`Sector(spin = [1,1], disp = (degree = 1,))` alone). The empty lattice sub-model is
    # the right answer there; `restrict(model, :spin)` already returns its empty
    # counterpart without complaint.
    pmax = isempty(rules) ? zeros(Int, length(spec.pmax)) : spec.pmax
    return BasisSpec(spec.nbody, zeros(Int, length(spec.lmax)), pmax, spec.lsum,
                     spec.cutoff, spec.soc, rules, spec.disp_scale,
                     spec.species_labels)
end

"""
    bilinear_terms(model::SLCEModel)

The bilinear pair (`ls=[1,1]`: Heisenberg + Dzyaloshinskii–Moriya + symmetric-anisotropic)
and single-ion (`ls=[2]`) channels of a fitted SLCE, extracted as Cartesian `3×3` matrices
and folded onto the training supercell. Returns a named tuple `(; pairs, onsites, skipped)`:

- `pairs::Dict{Tuple{Int,Int,SVector{3,Int}}, SMatrix{3,3}}` — one matrix `M` per bond
  `(a, b, R)` with `a ≤ b` (canonical members carry each physical bond once, both
  directed contributions pre-summed), energy `Σ eₐ'·M·e_b`;
- `onsites::Dict{Int, SMatrix{3,3}}` — one single-ion matrix `A` per atom, energy `eₐ'·A·eₐ`;
- `skipped::Vector{String}` — the higher-order / higher-`l` SALCs not representable as a
  bilinear pair or single-ion term.

Reuses the same validated tesseral→Cartesian conversion as the Sunny export
([`to_sunny`](@ref)); the higher-order channels it lists are kept by the full
[`spin_multipole_terms`](@ref) view.
"""
function bilinear_terms(model::SLCEModel)
    t = _bilinear_terms(model)
    return (; pairs = t.pairs, onsites = t.onsites, skipped = t.skipped)
end
