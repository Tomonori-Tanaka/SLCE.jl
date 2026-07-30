# Force constants and phonons

```@meta
CurrentModule = SLCE
```

A joint model's displacement channel is a lattice-dynamics model, and this page is how you
read it out: [`force_constants`](@ref) for the real-space constants,
[`dynamical_matrix`](@ref) for `D(q)`, and [`write_phonopy`](@ref) /
[`write_alamode`](@ref) to hand the result to the established phonon codes. Everything
here is a derivative of the *same* fitted energy surface the spin channel comes from —
see [Reading a fitted model](introspection.md) for the term-level view and
[Strain, magnetoelasticity and volume grids](strain.md) for the cell-deformation half.

## The model on this page

One two-atom cell, a spin sector, a spin-dressed displacement sector and a pure
displacement sector, trained on synthetic energies from a known model at displaced
structures. Two fits of the *same* data, one unconstrained and one under the acoustic sum
rule, because the contrast between them is the physics of this page:

```@example fc
using SLCE, LinearAlgebra, Random

nat = 2
cr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
    Sector(spin = (sites = 1:2,), cutoff = 1.1),                           # pure spin
    Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),  # coupled
    Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)])              # force constants
basis = SLCEBasis(cr, spec)
truth = SLCEModel(basis, 0.0, randn(MersenneTwister(1), n_salcs(basis)))

prov = DatumProvenance(; reference_id = "ref", reference_fingerprint = crystal_fingerprint(cr))
rng = MersenneTwister(7)
randcfg() = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])

data = map(1:200) do _
    e = randcfg()
    u = 0.08 .* randn(rng, 3, nat)
    TrainingDatum(; energy = predict_energy(truth, e, u), directions = e,
                  magmoms = ones(nat), displacements = u, provenance = prov)
end
jds = SLCEDataset(basis, data; use_torque = false, use_force = false)

joint = SLCEModel(fit(SLCEFit, jds, OLS(); asr = false))   # deliberately unconstrained
nothing # hide
```

## Real-space constants

[`force_constants`](@ref) differentiates the energy with respect to displacements at
`u = 0`, **with the spins held fixed** — so the lattice dynamics you get out is the
lattice dynamics *of that magnetic state*. The derivatives are exact, not finite
differences: every displacement factor is a homogeneous polynomial, so the constants
are read off its monomial coefficients.

```@example fc
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

!!! warning "Which pairs a reference cell can carry: `a ≠ b`, always"
    The cluster enumeration admits a pair only between **two distinct atoms of the
    reference cell** (at their minimum-image separation — which reaches the
    Wigner–Seitz corner, so a cutoff cap is not what limits it). An atom paired with
    its own periodic image is never admitted, so **`Φ[(a,0),(a,R≠0)]` is absent from
    this list by construction**, and the ``q``-dependence it would carry is absent from
    `D(q)`. For the fit that costs nothing — a datum's displacement field is
    cell-periodic, so both ends of such a pair move together and the columns are
    dependent anyway — but for the read-out it means a same-sublattice force constant
    exists only if the two atoms are separate atoms *of the cell you built the basis on*.
    A one-atom cell therefore carries no pair content at all; `build_asr` warns that
    there is no translation-invariant displacement content, and `D(q) ≡ 0`. Describe the
    crystal with a large enough cell (the same reason a force-constant fit wants ≥ 3
    periods along each fitted direction) and the same bonds appear as ordinary
    distinct-atom pairs. Full statement in
    [Periodic resolvability](../theory/resolvability.md#Which-interactions-the-enumeration-can-represent);
    `write_phonopy` / `write_alamode` export whatever this list holds, without a warning
    of their own.

!!! note "The result carries the magnetic space group, without one being declared"
    This is the joint expansion's headline claim on the lattice side. The SALCs are
    projected with the paramagnetic grey group ``G \\times \\{1, T\\}``; fixing `spins`
    reduces that to the **magnetic** stabilizer of the state, antiunitary elements
    included — which is the correct group for a time-reversal-even quantity like ``\\Phi``.
    Both neighbouring choices are wrong and silent: a lattice-only basis imposes the
    paramagnetic group (too large), and splitting atom types by their moment — what
    passing `MAGMOM` to a phonon code does — imposes the unitary subgroup alone (too
    small). The operation counts on a stripe-AFM fixture are gated in the test suite; the
    sector-level version of the argument is in [Building the basis](basis.md).

[`dynamical_matrix`](@ref) folds them into reciprocal space at a wavevector given in
**fractional reciprocal coordinates**. Pass `masses` (in **amu**) to get an `ω²`
spectrum; omit it for the bare force-constant matrix:

```@example fc
D0 = dynamical_matrix(fcs, [0.0, 0.0, 0.0])
println("eigenvalues of D(0): ", round.(sort(real(eigvals(Hermitian(D0)))); sigdigits = 3))
```

That call passed no `masses`, so those are eigenvalues of the **bare** force-constant
matrix, in eV/Å² — not frequencies of anything. Pass `masses` in amu and they become
`ω²` **in eV/Å²/amu**, which is still not a frequency until you convert it:

```@example fc
λ = sort(real(eigvals(Hermitian(dynamical_matrix(fcs, [0.0, 0.0, 0.0];
                                                 masses = [55.845, 55.845])))))
signed_sqrt(x) = x < 0 ? -sqrt(-x) : sqrt(x)          # imaginary modes stay negative
println("ω² [eV/Å²/amu]: ", round.(λ; sigdigits = 3))
println("ν  [THz]      : ", round.(signed_sqrt.(λ) .* 15.633302; sigdigits = 3))
```

The `15.633302` is phonopy's `VaspToTHz` (the `2π` is already in it, so this is the
ordinary frequency `ν`, not the angular `ω`); a further `× 33.35641` gives cm⁻¹. The
`test/alamode/` suite pins both against `anphon`'s own cm⁻¹ output.

None of those are zero — and they should be. Three eigenvalues of `D(0)` must vanish:
those are the acoustic branches, the statement that translating the whole crystal
costs no energy. They do not here because `joint` above was deliberately fitted with
`asr = false`. Fit the same data under the acoustic sum rule — the default — and they
appear:

```@example fc
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

```@example fc
D_other = dynamical_matrix(force_constants(joint; spins = randcfg()), [0.0, 0.0, 0.0])
println("max |ΔD(0)| between two magnetic states: ",
        round(maximum(abs, D_other .- D0); sigdigits = 4))
```


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

### Anharmonic constants: ALAMODE

phonopy takes the harmonic channel and stops. [`write_alamode`](@ref) writes an
ALAMODE `FCSXML` carrying the harmonic set *and* any anharmonic ones, which is what
`anphon` needs for relaxation-time thermal conductivity, self-consistent phonons and
Grüneisen parameters:

```julia
write_alamode("si.xml", force_constants(model; spins = e, order = 2),
                        force_constants(model; spins = e, order = 3))
```

All sets must come from one crystal and one spin state — they are the expansion of a
single energy surface, and `anphon` has no notion of a magnetic state.


Next: the cell-deformation half of the lattice channel —
[Strain, magnetoelasticity and volume grids](strain.md).
