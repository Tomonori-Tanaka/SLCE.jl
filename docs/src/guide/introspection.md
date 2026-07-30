# Reading a fitted model

A fitted [`SLCEModel`](@ref) predicts energies, torques and forces, but sooner or
later you want the *terms* — to hand the Hamiltonian to a sampler, to compare a
coupling against a published number, to see which interactions survived the fit.
That is what this page is about. Downstream packages read a model through these
functions and never through `model.basis.salc_basis.salcs`, so the basis
representation can change without breaking them.

| function | what you get |
|:--|:--|
| [`decorated_terms`](@ref) | every term, any channel — the general surface |
| [`spin_multipole_terms`](@ref) | the frozen pure-spin view (refuses displacement models) |
| [`bilinear_terms`](@ref) | `ls = [1,1]` pairs and `ls = [2]` single-ion terms as Cartesian `3×3` matrices |
| [`restrict`](@ref) | a sub-model of a joint model: `:spin` is the clamped-ion (`u = 0`) one, `:lattice` the spin-independent one a phonon consumer wants |
| [`to_sunny`](@ref) | a `Sunny.System` (see [Sunny export](sunny.md)) |

## The scale rule

Every term carries a scale factor that the *consumer* applies:

```
E_term = coef · scale · Σ_μ folded[μ] ∏ᵢ fᵢ(μ_i)
```

with one factor `fᵢ` per tensor axis — `Z_{l,m}(ê_a)` on a spin axis,
`|u_a|^{2k} R_{l,m}(u_a)` on a displacement one.

**The scale is `(4π)^(n_spin_slots / 2)`**: one `√(4π)` per *spin* factor. The 4π
is an artifact of the spin-sphere measure, and the displacement kernel is
normalized 4π-free, so displacement axes contribute none.

The pure-spin-era shortcut `(4π)^(body/2)` — one factor per *site* — agrees with
this exactly when the spin-slot count equals the body order, i.e. when every site
carries a spin factor (with or without a displacement one alongside). As soon as
one site is pure displacement it is wrong, and wrong in a way that still produces
plausible-looking numbers. [`DecoratedTerm`](@ref) therefore ships `scale` as a
field: read it, never re-derive it from `body` or `length(atoms)`.

```@example introspect
using SLCE, LinearAlgebra, Random

cr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
    Sector(spin = (sites = 1:2,), cutoff = 1.1),                      # pure spin
    Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),  # coupled
    Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)])         # force constants
basis = SLCEBasis(cr, spec)
model = SLCEModel(basis, 0.0, randn(MersenneTwister(1), n_salcs(basis)))

terms = decorated_terms(model)
sig(t) = (t.body, [s.factor.channel for s in t.slots])   # one line per slot signature

println(rpad("body", 6), rpad("slots", 34), rpad("scale", 10), "(4π)^(body/2)")
for s in unique(sig.(terms))
    t = terms[findfirst(x -> sig(x) == s, terms)]
    chans = [Symbol(s.factor.channel) for s in t.slots]
    println(rpad(t.body, 6), rpad(string(chans), 34),
            rpad(round(t.scale; sigdigits = 5), 10),
            round((4π)^(t.body / 2); sigdigits = 5),
            t.scale ≈ (4π)^(t.body / 2) ? "" : "   ← differs")
end
```

The rows where the two rules part ways are the ones with a site that carries **no**
spin factor — a force-constant term, whose sites are pure displacement. There the
shortcut invents a `√(4π)` per site out of nothing. The two agree exactly when the
spin-slot count equals the body order, which includes coupled terms whose every site
holds a displacement factor *as well as* a spin one — that coincidence is why the
shortcut survived so long.

A displacement-decorated model is refused by [`spin_multipole_terms`](@ref) rather than
mis-scaled, and the error names the two ways forward: `decorated_terms(model)`, or
`spin_multipole_terms(restrict(model, :spin))`.

### If you are going to *rewrite* coefficients, pass `keep_zero = true`

By default both surfaces skip a SALC whose coefficient is exactly zero. That makes the term
list — and the index a consumer addresses it by — a function of the coefficient *values*
rather than of the basis. For a reader that is harmless. For anything that later rewrites
coefficients in place against those indices (a Monte Carlo hot-swap across a volume grid, an
active-learning refit) it is not: sparse estimators and [`refit`](@ref) produce exact zeros
routinely, so two models built on the *same* basis can emit lists of **equal length whose
maps are shifted**, and every length check still passes while each coefficient lands on a
neighbouring cluster.

```@example introspect
zeroed = SLCEModel(basis, 0.0, [i == 2 ? 0.0 : c for (i, c) in enumerate(model.jphi)])
other  = SLCEModel(basis, 0.0, [i == 3 ? 0.0 : c for (i, c) in enumerate(model.jphi)])
# The identity of the interaction a term stands for is its member shape AND its folded
# tensor: two SALCs of one orbit can share every site and slot and differ only in the
# invariant they project onto, which is exactly the pair that makes this hazard invisible
# to a shape-only check.
ident(t) = (t.atoms, t.shifts, [(s.site, s.factor) for s in t.slots], t.folded)
println("pruned:    equal length = ",
        length(decorated_terms(zeroed)) == length(decorated_terms(other)),
        ",  same map = ", ident.(decorated_terms(zeroed)) == ident.(decorated_terms(other)))
println("keep_zero: same map = ",
        ident.(decorated_terms(zeroed; keep_zero = true)) ==
        ident.(decorated_terms(other; keep_zero = true)))
```

Equal length, different map — every length check a consumer could write passes while each
coefficient lands on a neighbouring interaction. With `keep_zero = true` the map is a
property of the SALC basis alone, which is exactly the object a [`StrainedModels`](@ref)
grid asserts identical across its points. The default stays `false` so existing consumers
are untouched.

## `restrict` is not a refit

[`restrict(model, :spin)`](@ref restrict) returns the joint model evaluated at
`u = 0` — the clamped-ion sub-model. It is exact: `predict_energy(restrict(m,
:spin), e)` equals `predict_energy(m, e, zeros(3, n))` bitwise, and the result is
an ordinary pure-spin model that every spin-only surface accepts.

!!! warning "Clamped-ion couplings are not spin-only-fit couplings"
    The couplings `restrict` hands back are the values each spin interaction takes
    with every atom at its **ideal reference site**. They are *not* the couplings
    you get by fitting a spin-only model to the same data, and not the ones a
    relaxed or finite-temperature structure exhibits.

    A spin-only model has no displacement coordinate to attribute anything to, so
    whatever the lattice was doing in the training structures is absorbed into its
    spin couplings: the fitted `J` silently carries a `⟨u⟩`- and
    `⟨uu⟩`-renormalized contribution. `restrict` drops exactly that. The gap
    between the two is the physics the joint model exists to separate — not a
    discrepancy to reconcile. If you want renormalized couplings at some
    displacement state, that is a thermodynamic average over the joint model, not
    a restriction of it.

Here is the difference, measured. We generate data from a known joint model on
displaced structures, then fit two models to the *same* energies — one joint, one
spin-only:

```@example introspect
nat = 2
fp = crystal_fingerprint(cr)
prov = DatumProvenance(; reference_id = "ref", reference_fingerprint = fp)
rng = MersenneTwister(7)
randcfg() = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])

data = map(1:200) do _
    e = randcfg()
    u = 0.08 .* randn(rng, 3, nat)          # the structures are displaced
    TrainingDatum(; energy = predict_energy(model, e, u), directions = e,
                  magmoms = ones(nat), displacements = u, provenance = prov)
end

jds = SLCEDataset(basis, data; use_torque = false, use_force = false)
joint = SLCEModel(fit(SLCEFit, jds, OLS(); asr = false))
clamped = restrict(joint, :spin)             # exact: the joint model at u = 0

# the same energies, fitted with no displacement coordinate at all
sbasis = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
    Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
sonly = SLCEModel(fit(SLCEFit, SLCEDataset(sbasis, [d.directions for d in data],
                                           [d.energy for d in data]), OLS()))

println("clamped-ion (restrict):  ", round.(coef(clamped); sigdigits = 3))
println("spin-only refit:         ", round.(coef(sonly); sigdigits = 3))
println("max |difference|:        ",
        round(maximum(abs, coef(clamped) .- coef(sonly)); sigdigits = 3))
```

The spin-only fit reproduces the *energies* it was given about as well as the joint
one does, and lands on different couplings — which is the whole point: those two
numbers answer different questions, and only the joint model can tell you which
part of the coupling came from the lattice.


## Where the derived quantities live

The term views above are the *model*. Its derivatives each have their own page:

| you want | page |
|:--|:--|
| force constants, `D(q)`, phonopy / ALAMODE export | [Force constants and phonons](lattice_dynamics.md) |
| strain response, magnetoelastic constants, magnon–phonon vertices, volume grids | [Strain, magnetoelasticity and volume grids](strain.md) |
| a `Sunny.System` for spin-wave theory | [Sunny export](sunny.md) |
| how the pieces fit together on a joint model | [A joint spin–lattice model, end to end](joint.md) |
