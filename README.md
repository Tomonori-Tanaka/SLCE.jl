# SLCE.jl

A clean, extensible, Julia-native rebuild of **Magesty.jl**, extended into a
**spin–lattice cluster expansion** — one symmetry-adapted basis over spin
*and* displacement degrees of freedom, fitted to noncollinear DFT data.

> **Status: work in progress.** The numerical core (tesseral spherical harmonics,
> solid harmonics, Clebsch–Gordan coupling, symmetry-adapted basis, design
> matrices, regression) is reimplemented from scratch and validated against
> Magesty.jl as a pinned numerical oracle. This is an architectural exploration;
> for production pure-spin work see Magesty.jl. Three DFT observables — **energy**,
> per-atom **torque**, and per-atom **force** — are fitted; see
> [Status](#status) for what is and isn't implemented.

## What it does

Given a crystal structure and a set of noncollinear spin configurations with their
energies, fit

```
E({e_a}) = j0 + Σ_φ J_φ Φ_φ({e_a})
```

where the spins `e_a` are unit vectors and the `Φ_φ` are symmetry-adapted,
time-reversal-even scalar invariants built from real tesseral spherical harmonics
over clusters of spins. Fitting recovers the cluster coefficients `J_φ`.

The same coefficients also fix the per-atom **torque** `τ_a = −e_a × ∂E/∂e_a` (the
physical / Landau–Lifshitz torque `m_a × B_eff,a`), the SCE's other DFT observable. Passing per-configuration torques to `SLCEDataset` and a
`torque_weight ∈ (0, 1]` to `fit` runs an energy+torque co-fit that minimizes
`(1 − w)·MSE_energy + w·MSE_torque`.

### The joint spin–lattice expansion

Sites may also carry **displacement** factors — real solid harmonics `Rₗₘ(u_a)` of
the Cartesian displacement from the reference position — so one basis spans pure
spin, pure lattice (force constants), and genuinely coupled terms:

```
E({e_a}, {u_a}) = j0 + Σ_φ J_φ Φ_φ({e_a}, {u_a})
```

Which decorations exist is declared by a **sector table** rather than by one global
truncation, so each physical channel gets its own body order, ranks, radii, and its
own answer to *does this term need spin–orbit coupling*:

```julia
spec = BasisSpec(crystal;
    lmax = 2, pmax = 2,
    sectors = [Sector(spin = (sites = 2:3, lmax = 2), cutoff = 6.0),       # pure spin
               Sector(disp = (degree = 2,), cutoff = 6.0),                 # force constants
               Sector(spin = [2, 2], disp = (degree = 1,), cutoff = 4.5)]) # magnetoelastic
```

The third observable is the per-atom **force** `F_a = −∂E/∂u_a`; a
`force_weight` runs a three-block (energy / torque / force) co-fit. Translation
invariance is imposed exactly, as a linear **acoustic sum rule** on the
coefficients (`asr = true` by default) rather than corrected afterwards, and
`asr_residual` / `identifiability` report how well determined the result is.

From a fitted joint model:

| | |
|---|---|
| `restrict(model, :spin)` | the exact clamped-ion spin model (evaluate at `u = 0`) |
| `force_constants(model; spins, order)` | harmonic (or higher) force constants at a fixed spin state |
| `dynamical_matrix(fcs, q)` | `D(q)` for phonons — its `q = 0` acoustic zeros are the ASR, measured |
| `effective_model(model; u0)` | an exact re-expansion around a displaced structure |
| `decorated_terms(model)` | the raw term contract for downstream samplers |

`force_constants` is where the joint expansion earns its keep. Its output is
invariant under the **magnetic space group** of the spin state you evaluate at —
antiunitary elements included — without that group ever being declared: the SALCs
are projected with the paramagnetic grey group, and fixing `spins` reduces it to the
magnetic stabilizer automatically. Fitting a *lattice-only* basis to magnetically
ordered data instead imposes the paramagnetic group, which is too large, and zeroes
the components the order breaks. Note which sector feeds which deliverable, though:
the `degree = 1` magnetoelastic row above contributes to the **forces**; harmonic
constants that depend on the magnetic state need `degree = 2` under a spin-carrying
sector.

## Usage

```julia
using SLCE
import Spglib                     # load it to activate the SpglibBackend extension
                                 # (`import`, not `using`, to avoid a `Lattice` name clash)
using LinearAlgebra

# A 4-atom chain of identical spins along z
lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
frac = [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75]
chain = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])

# nearest-neighbor 2-body interaction, isotropic (Heisenberg) channel only
interaction = BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], soc = false)
basis = SLCEBasis(chain, interaction; backend = SpglibBackend())

# synthetic Heisenberg data E = J Σ_⟨ij⟩ e_i·e_j, then fit
configs = [mapreduce(_ -> (v = randn(3); v / norm(v)), hcat, 1:4) for _ in 1:30]
heis = SLCE.salcs(basis)[1]   # public-but-unexported: call it qualified
J = 0.0137
E = [J * sum(c[:, m.atoms[1]]' * c[:, m.atoms[2]] for m in heis.members) for c in configs]

f = fit(SLCEFit, SLCEDataset(basis, configs, E), OLS())
r2_energy(f)                      # ≈ 1.0
2 * sqrt(3) * coef(f)[1]          # ≈ J  (recovered coupling)
```

Standalone runnable versions are in [`examples/heisenberg_chain.jl`](examples/heisenberg_chain.jl)
and [`examples/kagome_threebody.jl`](examples/kagome_threebody.jl) (3-body / multi-term SALCs).

### Persistence and input files

`fit` returns an `SLCEFit` — the heavyweight result that keeps the data and answers
diagnostics (`r2_energy`, `residuals_energy`, …). For prediction and storage, wrap it
in the lightweight, persistable `SLCEModel` with `SLCEModel(f)`.

Save a fitted model (or just a basis) to a self-contained, human-readable **TOML**
document and reload it later. Coefficients re-pair to the basis by `SALCKey`, so a
reloaded model predicts identically:

```julia
SLCE.save("model.toml", SLCEModel(f))     # or save("basis.toml", basis)
model = SLCE.load(SLCEModel, "model.toml")
predict_energy(model, configs)
```

A basis can also be built from a human-authored `input.toml` (inline crystal +
interaction + optional symmetry) instead of constructing `Crystal` / `BasisSpec`
in Julia. Training data and the estimator stay in Julia:

```toml
# input.toml
[structure]
lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]]  # each entry = one lattice vector
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75]]
species = [1, 1, 1, 1]
species_labels = ["Fe"]

[interaction]
nbody = 2
cutoff = 2.6
lmax = [1]
soc      = false

[symmetry]
backend = "spglib"   # or "none"
tol = 1.0e-5
```

```julia
basis = SLCEBasis("input.toml")     # reads [symmetry] backend/tol from the file
```

See [`examples/persist_and_input.jl`](examples/persist_and_input.jl) for the full loop.

### Inspecting fitted coefficients

`coeftable(f)` returns a Tables.jl source (one row per SALC: `body`, `orbit_id`, `ls`,
`Lf`, `block`, `J`), so the coefficients drop into any table / IO package:

```julia
using DataFrames
df = DataFrame(coeftable(f))       # or CSV.write("J.csv", coeftable(f))
intercept(f)                       # the reference energy j0 (not a row)
```

Standard diagnostics are split by observable: `r2_energy` / `rmse_energy` / `rss_energy` /
`residuals_energy` (and the `_torque` equivalents for a co-fit), plus `nobs` and `dof`.
The generic names — `predict` / `residuals` / `r2` (defaulting to the energy block) and
`coef` / `fit` / `nobs` / `dof` / `coeftable` / `islinear` — extend
[StatsAPI](https://github.com/JuliaStats/StatsAPI.jl), so they compose with the
StatsBase / GLM ecosystem instead of clashing on `using`.

### Refitting on a selected support

After a sparse fit, [`refit`](src/fitting/fit.jl) keeps the surviving support and re-solves on
just those columns (by default with `OLS`) — the de-biasing step that removes the penalty's
shrinkage from the selected coefficients:

```julia
fsparse = fit(SLCEFit, dataset, Lasso())   # selects a support (some J exactly zero)
fdebias = refit(fsparse)                   # unshrunk OLS on that support
```

### Regularized fits (sparse / L0-like selection)

`AdaptiveRidge` is an in-tree, closed-form estimator (no extra dependency): an iterative
reweighted ridge that approximates an L0 penalty, driving small coefficients toward zero.

```julia
fit(SLCEFit, dataset, AdaptiveRidge(lambda = 1e-3))     # L0-like selection, analytic
```

For L1 selection, load GLMNet to activate the `Lasso` / `ElasticNet` / `AdaptiveLasso`
estimators. With `lambda = nothing` (the default) the penalty is chosen by
cross-validation; pass a number to fit at a fixed penalty. The column-centering /
analytic-`j0` contract is identical to `OLS`/`Ridge`, so the rest of the pipeline is
unchanged:

```julia
using GLMNet                         # activates the estimator extension

fit(SLCEFit, dataset, Lasso())                          # CV-selected λ, sparse model
fit(SLCEFit, dataset, Lasso(select = :lambda_1se))      # the parsimonious 1-SE model
fit(SLCEFit, dataset, ElasticNet(alpha = 0.5))          # L1/L2 mix
fit(SLCEFit, dataset, Lasso(lambda = 1e-3))             # a fixed penalty (no CV)
fit(SLCEFit, dataset, AdaptiveLasso())                  # pilot-reweighted Lasso (Zou 2006)
```

`AdaptiveLasso` runs a pilot estimator (default `OLS`, any estimator allowed — including a
`PrecomputedPilot` reusing a prior fit) and then a weighted Lasso that spares the columns
the pilot found large. For an energy+torque co-fit the cross-validation folds are grouped
by configuration.

### Reading DFT data

DFT-code I/O is isolated at the training-data boundary: the core owns only the
code-agnostic `TrainingDatum` / `SLCEDataset` seam (`read_configs(src::AbstractDFTSource)`), so
once you have the data the originating code is irrelevant. The **concrete VASP adapter lives
in the companion [SLCETools.jl](https://github.com/Tomonori-Tanaka/SLCETools.jl) package**:

```julia
using SLCE, SLCETools
using SLCETools.VASP: read_poscar, Oszicar

crystal = read_poscar("POSCAR")                          # → Crystal
basis   = SLCEBasis(crystal, interaction)

# constrained-noncollinear OSZICARs → energy + spin directions + torque target (τ = m×B)
src     = Oszicar(["run1/OSZICAR", "run2/OSZICAR"])      # an AbstractDFTSource
dataset = SLCEDataset(basis, src)                         # read_configs(src) under the hood
fit(SLCEFit, dataset, OLS(); torque_weight = 0.5)
```

Adding another DFT code is one sibling adapter in SLCETools — the core and its exports do not
change. (SLCETools also writes the inverse direction: sampled configurations →
constrained-noncollinear VASP inputs.)

## Design highlights

- **Pluggable seams** via multiple dispatch + Julia package extensions: symmetry
  backends (`AbstractSymmetryBackend`; `NoSymmetry` in-tree, `SpglibBackend` in an
  extension) and estimators (`AbstractEstimator`; `OLS`/`Ridge`/`AdaptiveRidge` in-tree,
  `Lasso` / `ElasticNet` / `AdaptiveLasso` with cross-validated regularization paths in a
  GLMNet extension). The lightweight core loads with no heavy dependencies.
- **Generalized cutoff neighbor list** — no fixed image grid; correct for
  triclinic cells and cutoffs spanning many lattice translations.
- **Minimum-image periodic resolvability** — the default `MinimumImage` selection
  enumerates exactly the Wigner–Seitz-cell pairs a finite supercell can resolve (the
  corner `(L/2,L/2,L/2)` at `√3·L/2`, not a sphere of radius `L/2`), so no aliased,
  collinear interactions enter the fit; `cutoff = Inf` takes the whole cell.
- **Real Wigner-D from the package's own `Zₗₘ`** by an exact least-squares fit —
  convention-consistent by construction, handles improper rotations natively.
- **Orbit–stabilizer SALC projection** with a deterministic gauge and canonical
  **`SALCKey`** column addressing — a design-matrix column always means a fixed
  interaction, independent of construction order.
- **Spin-spiral-ready hooks** — neighbor pairs and cluster orbits retain the
  inter-site lattice translation `R`, leaving a clean seam for generalized-Bloch
  `E(q)` training data.
- **One projector for both channels** — spin factors transform as axial and
  displacement factors as polar quantities, and the parity screen makes a single
  polar Wigner-D cache exact for both; the mixed-channel counts are gated against
  an independent counting oracle rather than against the engine itself.
- **Translation invariance as a constraint, not a correction** — the acoustic sum
  rule is built symbolically from the same polynomials the force constants read and
  applied as an affine reparameterization of the coefficient vector, so every fit
  is exactly translation-invariant by construction. `identifiability` reports what
  the data cannot determine (torques alone, for instance, are blind to everything
  spin-independent — force constants are not recoverable from them at any weight).
- **Sunny.jl export** — `to_sunny(model; spin_length)` builds a real `Sunny.System`
  (bilinear exchange + single-ion) for linear spin-wave theory, on the exact training
  supercell or unfolded onto the chemical primitive cell; the conversion math is a
  dependency-free core layer gated by energy reconstruction.

See [`docs/design-notes.md`](docs/design-notes.md) for the rationale behind these
refinements over Magesty.jl, and [`SPEC.md`](SPEC.md) for the realized architecture.

## Documentation

A full Documenter.jl site (home, getting started, a guide, narrated tutorials, theory, and
the API reference) lives under [`docs/`](docs/). Build and read it locally:

```bash
make -C docs serve      # build, then serve at http://localhost:8000 with live reload
# or:  make -C docs build && make -C docs open
```

The first build resolves `docs/Project.toml` (Documenter + Spglib + the package). The site
is not yet deployed — add a remote and `deploydocs` when one exists.

## Status

**Pure spin (v0, feature-complete):** geometry → symmetry (pluggable backend) →
**arbitrary-body-order** cluster orbits → SALC basis (isotropic and anisotropic
channels, including the `N ≥ 3` coupling-path / `l`-ordering mixing) → **energy and
torque** design matrices → `OLS` / `Ridge` / `AdaptiveRidge` / `Lasso` / `ElasticNet` /
`AdaptiveLasso` fit (energy-only or energy+torque co-fit) → `predict_energy` /
`predict_torque`. Cross-validated against Magesty.jl through 3-body (invariant-subspace
dimensions agree exactly).

**Joint spin–lattice:** the displacement channel and its solid-harmonic kernel, the
`Sector` truncation table, the mixed-channel projector (with an independent counting
oracle), the joint data layer (`TrainingDatum` with per-datum spins / displacements /
forces / torques), the force design block and three-block co-fit, the exact acoustic
sum rule with `identifiability` diagnostics, staged (`frozen` / `sector_mask`)
fitting, and the downstream deliverables — `restrict`, `decorated_terms`,
`force_constants`, `dynamical_matrix`, `effective_model` — are implemented. Strain
and finite-temperature utilities are in progress; the strain measure is pinned
(Biot / Seth–Hill `m = 1`) in [`docs/specs/`](docs/specs/).

**Fit-side tooling:** cost-weighted group selection (`GroupAdaptiveRidge`,
`select_fit`, `select_support`), `gcv` / `effective_dof`, and grouped
`cross_validate`.

Basis/model **persistence**, a human-authored **`input.toml`**, **tabular coefficient
output** (`coeftable`), a **code-agnostic DFT-source seam** (`TrainingDatum` / `SLCEDataset`; the
concrete VASP adapter lives in the companion `SLCETools.jl`), **GLMNet** Lasso / elastic-net /
adaptive-Lasso estimators, and **Sunny.jl export** are implemented as extensions.

## References

- R. Drautz & M. Fähnle, *Phys. Rev. B* **69**, 104404 (2004) — spin-cluster
  expansion.
- R. Drautz, *Phys. Rev. B* **102**, 024104 (2020) — tesseral-harmonic /
  cluster-expansion formalism.
- T. Tanaka & Y. Gohda, *Phys. Rev. Research* **8**, 023300 (2026).

## License

MIT — see [LICENSE](LICENSE).
