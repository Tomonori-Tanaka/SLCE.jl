# A joint spin–lattice model, end to end

```@meta
CurrentModule = SLCE
```

The pure-spin path is one page ([Getting started](../getting_started.md)) because there is
one decision in it: which clusters. A joint spin–lattice fit has five, spread over four
pages, and this page is the map. Nothing here is new machinery — every step links to the
page that does it properly.

The whole point of the joint expansion is separation: a spin-only fit absorbs whatever the
lattice was doing in the training structures into its exchange couplings, and cannot tell
you which part of a coupling was magnetic and which was structural. A joint fit can, and
the price is that you have to be explicit about five things.

## 1. Declare the sectors — the model's support

A joint basis is specified as a **sector table**: each [`Sector`](@ref) row is one family of
decorated clusters, with its own spin content, displacement degree, SOC switch and cutoff.
The rows are the physics you intend to resolve, and a row you leave out is content the
model *cannot* express, no matter how much data you give it.

```julia
spec = BasisSpec(crystal; lmax = 2, pmax = ["*" => 0, "Fe" => 3], sectors = [
    Sector(spin = (sites = 2:3, lmax = 2), cutoff = 8.0),                  # pure spin
    Sector(spin = [1, 1], disp = (degree = 1,), soc = false, cutoff = 5.0),# dJ/dr
    Sector(disp = (degree = 2:3,), cutoff = 6.0)])                         # force constants
```

Which row feeds which deliverable is a rule worth internalizing before fitting, because a
missing row shows up as a *silently* state-independent answer rather than an error: a
`degree = 1` spin-dressed row feeds forces and the magnetoelastic tier, `degree = 2` feeds
magnetic-state-dependent force constants, and a basis with spin content but no
spin-*carrying* displacement term yields phonons identical for every magnetic state.
→ [Building the basis](basis.md), and its "which sector feeds which deliverable" section.

## 2. Bring data that carries the extra channels

Spin configurations alone cannot train a displacement-decorated basis, and the
spin-configuration dataset path refuses one rather than fitting a lattice channel from
nothing. Build the dataset from [`TrainingDatum`](@ref) vectors, which carry displacements
and forces alongside energies and torques:

```julia
data = [joint_datum(; energy, directions, magmoms, displacements, forces, provenance) …]
dataset = SLCEDataset(basis, data)
```

Two invariants do real work here. Every datum must come from **one** computational setup and
**one** clamped-ion reference — mixing them reintroduces a family-correlated energy offset
that the fit reads as physics. And a channel that was not computed must be `nothing`, never
a fabricated zero: a present all-zero field means "computed, and it vanishes", which is a
different statement. → [Persistence and I/O](io.md).

## 3. Choose the three block weights

Energies, torques and forces are three observables of the same surface, and the fit
minimizes `(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F`. Both weights default to zero, so
a joint fit that should be using its forces will happily ignore them — pass them
explicitly. The blocks are ragged by construction (a torque row exists only for a
torque-qualified configuration; a force row only for a displacement-referenced atom), which
is bookkeeping the dataset stores rather than re-derives.
→ [Data and fitting](fitting.md).

## 4. Keep the acoustic sum rule (it is the default)

Translation invariance is *not* a space-group symmetry, so no basis projection can remove
it: it is a linear constraint on coefficients, imposed exactly by fitting in the
constraint's null space (`asr = true`). This is on by default on a joint basis, and it is
what makes the lattice channel physical — its visible consequence is that `D(0)` has
exactly three zero eigenvalues. Turn it off only to demonstrate what breaks.

Then ask whether the data determine what you fitted. [`identifiability`](@ref) reports the
design's numerical rank, and the answer is frequently *no* in a way no residual reveals:
torques are blind to every spin-independent term, so a torque-only fit cannot determine
force constants at all, and a basis with a lattice-only sector keeps a deficiency the sum
rule cannot cure. → [Data and fitting](fitting.md), the ASR and identifiability sections.

## 5. Read the model, not the coefficients

A fitted joint model is read through the introspection surface, and each derivative has its
own contract:

| you want | where |
|:--|:--|
| the terms, for a sampler or a comparison | [Reading a fitted model](introspection.md) |
| the clamped-ion spin model (`u = 0`, exact) | [`restrict`](@ref) — and read the "not a refit" warning |
| force constants, `D(q)`, phonopy / ALAMODE | [Force constants and phonons](lattice_dynamics.md) |
| elastic / magnetoelastic response, `dJ/dr`, magnon–phonon vertices | [Strain, magnetoelasticity and volume grids](strain.md) |
| coefficients as functions of volume | the volume-grid section of the same page |
| spin-wave theory | [Sunny export](sunny.md) |

## Staging it, if one shot does not converge

A joint fit can also be built in physical stages instead of one shot — fit the pure-spin
sector, freeze it, then fit the coupled and lattice sectors under their own constraint. The
staging axis (`frozen` / `sector_mask`: what a stage *moves*) is deliberately separate from
the truncation axis (`Sector(soc = …)`: what the model *contains*).
→ [Data and fitting](fitting.md), the staged-fit section.
