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
| [`multipole_terms`](@ref) | the frozen pure-spin view (refuses displacement models) |
| [`bilinear_terms`](@ref) | `ls = [1,1]` pairs and `ls = [2]` single-ion terms as Cartesian `3×3` matrices |
| [`restrict`](@ref) | the clamped-ion (`u = 0`) sub-model of a joint model |
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
this only when every site carries exactly one spin factor and nothing else. On a
joint model it is simply wrong, and wrong in a way that still produces
plausible-looking numbers. [`DecoratedTerm`](@ref) therefore ships `scale` as a
field: read it, never re-derive it from `body` or `length(atoms)`.

```@example introspect
using SLCE, LinearAlgebra, Random

cr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
    Sector(spin = (nbody = 1:2,), cutoff = 1.1),                      # pure spin
    Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1),  # coupled
    Sector(disp = (degree = 2,), nbody = 1:2, cutoff = 1.1)])         # force constants
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
shortcut invents a `√(4π)` per site out of nothing. (A coupled term whose every site
happens to hold a spin factor as well as a displacement one agrees by coincidence,
which is exactly why the shortcut survived so long.)

A displacement-decorated model is refused by [`multipole_terms`](@ref) rather than
mis-scaled, and the error names the two ways forward: `decorated_terms(model)`, or
`multipole_terms(restrict(model, :spin))`.

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
    Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
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

## Handing the lattice channel to phonopy

[`write_phonopy`](@ref) writes a `ForceConstantSet` as a phonopy calculation — a
`POSCAR` and a `FORCE_CONSTANTS` — so band structures, DOS, thermal properties,
group velocities and thermal conductivity come from phonopy rather than from
reimplementations here:

```julia
fcs = force_constants(model; spins = afm, order = 2)
out = write_phonopy("phonons/afm", fcs)     # out.dim is the --dim to pass
```

The magnetic state is already baked into `fcs`, and phonopy has no notion of one:
export two states to two directories and the difference between the band structures
*is* the magnetoelastic content of the model. Note the two files must be used
together — the `POSCAR` is what fixes the atom order the `FORCE_CONSTANTS` is written
in, and phonopy's supercell ordering is pinned by a test that runs phonopy itself
(`test/phonopy/`), because a permuted export still diagonalizes and still has three
acoustic modes at Γ.

## Force constants and phonons

[`force_constants`](@ref) differentiates the energy with respect to displacements at
`u = 0`, **with the spins held fixed** — so the lattice dynamics you get out is the
lattice dynamics *of that magnetic state*. The derivatives are exact, not finite
differences: every displacement factor is a homogeneous polynomial, so the constants
are read off its monomial coefficients.

```@example introspect
spins = randcfg()
fcs = force_constants(joint; spins = spins, order = 2)

for ((atoms, shifts), Φ) in sort(collect(fcs.constants); by = k -> (k[1][1], k[1][2]))
    println("Φ[(", atoms[1], ",0), (", atoms[2], ",", Tuple(shifts[2]), ")]  ‖Φ‖ = ",
            round(norm(Φ); sigdigits = 4))
end
```

Indices follow the lattice-dynamics convention `Φ[(a,0),(b,R)] = ∂²E/∂uₐ(0)∂u_b(R)`,
anchored so the first index is in the home cell. The reverse ordering is a separate
key, equal to the transpose.

[`dynamical_matrix`](@ref) folds them into reciprocal space at a wavevector given in
**fractional reciprocal coordinates**. Pass `masses` to get an `ω²` spectrum; omit it
for the bare force-constant matrix:

```@example introspect
D0 = dynamical_matrix(fcs, [0.0, 0.0, 0.0])
println("eigenvalues of D(0): ", round.(sort(real(eigvals(Hermitian(D0)))); sigdigits = 3))
```

None of those are zero — and they should be. Three eigenvalues of `D(0)` must vanish:
those are the acoustic branches, the statement that translating the whole crystal
costs no energy. They do not here because `joint` above was deliberately fitted with
`asr = false`. Fit the same data under the acoustic sum rule — the default — and they
appear:

```@example introspect
constrained = SLCEModel(fit(SLCEFit, jds, OLS(); asr = true))
Dc = dynamical_matrix(force_constants(constrained; spins = spins), [0.0, 0.0, 0.0])
println("asr = true:   ", round.(sort(real(eigvals(Hermitian(Dc)))); sigdigits = 3))
println("asr_residual: ", round(asr_residual(constrained); sigdigits = 3),
        "   vs unconstrained ", round(asr_residual(joint); sigdigits = 3))
```

That is what `fit(...; asr = true)` buys, and [`asr_residual`](@ref) is the direct
check on any model before you trust its phonons. (Zero acoustic frequencies are not
the same as *stable* phonons: a negative eigenvalue is a real instability of the
fitted model, not a bug in the fold.)

The magnetic contribution to the dynamics is the difference between two spin states:

```@example introspect
D_other = dynamical_matrix(force_constants(joint; spins = randcfg()), [0.0, 0.0, 0.0])
println("max |ΔD(0)| between two magnetic states: ",
        round(maximum(abs, D_other .- D0); sigdigits = 4))
```
