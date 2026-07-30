# Integration test environment

Whole-pipeline passes on **named real crystals**: build → data → ASR → fit →
recovery → every downstream deliverable → persistence, six times over, on cells
chosen so that each one forces a different code path.

## Run

```bash
julia --project=test/integration -e 'using Pkg; Pkg.instantiate()'   # once
julia --project=test/integration test/integration/runtests.jl
```

`Manifest.toml` here is gitignored. Threads are not required — nothing in this tier
compares a threaded result against a serial one (the core suite owns those gates).

## Why it is separate from `Pkg.test()`

It needs **Spglib**, and the core suite deliberately does not: `Pkg.test()` must
never depend on a symmetry backend, so the unit fixtures assemble their space
groups by hand. Here that would defeat the purpose. The tier is about the path a
user actually takes, and the real space group of a real crystal is part of it —
Spglib is also an *independent* implementation of the one thing a hand-assembled
fixture would otherwise assert about itself.

## Why it exists at all, given ~48k unit assertions

Every unit file gates one stage against one hand-built fixture. Nothing gated the
*combination* on a crystal with a name — and the last three real defects in this
package were found by walking one real system through the whole chain, not by
adding assertions to a fixture. This tier is that walk, written down.

It found one on its first run: a basis with **zero** SALC columns was accepted all
the way through `SLCEDataset` and `fit`, which then reported `r2_energy == 0.0`, an
empty `coef`, and a `predict_energy` that returns the same number for every
configuration — a silent constant dressed as a model. It is reachable from an
ordinary spec (rocksalt cation–cation superexchange; see `:selfpair` below), so the
dataset boundary now refuses it.

Two more facts it measured, both of which had been asserted the other way in a
first draft of the tier and are now recorded where they belong:

* **The conventional cell of a real crystal is usually the tied one.** bcc Fe
  freezes 14 of its 22 joint columns, B2 FeRh 35 of 46, hcp Co 40 of 57, wurtzite
  GaN 15 of 35. The freeze is the norm at minimal cell size, not an edge case —
  a primitive cell has one atom per orbit, so any coordination shell with
  multiplicity greater than one reaches the same atom through several images.
* **What the sum rule costs is a property of the row.** Refitting the same
  centre-of-mass-free data with `asr = false` proves nothing when the target came
  from a feasible model — the unconstrained solve returns the same coefficients
  (3e-15). Against a deliberately violating target the cost ranges from machine
  precision on the heavily frozen cells to 0.69 on the tie-free ones, because the
  energy block barely sees the violating content while the force block does. Only
  a sample allowed to drift makes it observable in every row.

## The roster

| row | space group | the feature it forces |
|:--|:--|:--|
| `bcc-Fe` | Im-3m (229), 96 ops | a Wigner–Seitz boundary tie: the 1NN shell reaches **eight images of one atom**, all at the same distance, so 14 of 22 columns are unresolvable |
| `bcc-Fe-2x2x2` | Im-3m (229), 768 ops | the same crystal in a cell that separates those eight images — same 22 columns, **none** frozen, 10 free parameters instead of 4 |
| `B2-FeRh` | Pm-3m (221), 48 ops | two species and two magnetic sublattices; the degree-1 spin-dressed sector that survives the sum rule, hence B₁/B₂ |
| `hcp-Co` | P6₃/mmc (194), 24 ops | a hexagonal (non-orthogonal) lattice, two like sites at inequivalent positions |
| `wz-GaN` | P6₃mc (186), 12 ops | **no inversion centre**, four atoms, the lattice-only entry path — and the truncation whose every ASR-feasible model is rotationally invariant to machine precision |
| `rs-MnO` | Fm-3m (225), 192 ops | the cation–cation shell, next to the primitive cell of the same crystal where that shell joins an atom to an image of *itself* |

## The coverage matrix

`roster.jl` declares, per row, which of the 17 columns run and — **with a reason, in
the file** — which do not. The driver refuses a row that leaves a column
unaccounted for, and refuses to report success if a declared column never
executed. A tier that quietly covers less than it claims is the failure this
structure exists to prevent; it is the same reason `test/runtests.jl` refuses an
unrecognized `TEST_MODE` rather than running zero tests.

Each column's oracle is independent of the code path it checks — the point of the
tier is not that the pipeline runs, but that what comes out is checkable against
something else:

| column | oracle |
|:--|:--|
| `census` | the International Tables (symbol, number, \|G\|), stated in `roster.jl` |
| `structure` | structural invariants of a canonical member, re-derived from the crystal |
| `resolvability` | rank–nullity of the ASR null space; unit content in a frozen column moves no energy |
| `invariance` | the crystal's own space group: `E(g·config) = E(config)` |
| `affine` | documented conventions of the rotation diagnostics, on real geometry |
| `asr` | three zero eigenvalues of `D(0)`, against an `asr = false` control with none |
| `recovery` | the forward map is inverted, on configurations drawn after the fit |
| `phonons` | Hermiticity, `D(−q) = conj D(q)`, the mass scaling |
| `fd_hessian` | central differences of `predict_energy` |
| `effective` | an exact identity between two expansions of one surface |
| `restrict` | a **bitwise** identity with the joint model at `u = 0` |
| `persist` | a TOML round trip must not move a prediction, bitwise |
| `terms` | the energy re-evaluated from the published term fields only |
| `magnetoelastic` | the pinned B₁/B₂ convention vs `strain_derivatives` at two magnetic states |
| `strain` | central differences of `affine_energy` |
| `selection` | an exactly representable target must survive every cross-validation fold |
| `selfpair` | a documented limitation, pinned so that fixing it shows up here |

## `:selfpair` is a pin on a limitation, not an endorsement

`MinimumImage` enumerates one image per (atom, atom) pair, so a pair joining an
atom to a periodic image of *itself* is never built. In B2 FeRh the 2NN shell is
exactly Fe–Fe and Rh–Rh, so widening the cutoff to admit it adds nothing; in
rocksalt MnO the whole magnetic problem is Mn–Mn, so a superexchange spec has no
columns at all. Both branches are asserted as they stand today. **When this is
fixed, both change** — the FeRh basis grows and MnO stops being empty. Update the
row; do not delete it.
