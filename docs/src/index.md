# SLCE.jl

```@meta
CurrentModule = SLCE
```

A clean, extensible, Julia-native rebuild of [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl) —
fitting **spin–lattice cluster expansion (SLCE)** models to noncollinear DFT data.

Given a crystal and DFT data on it, the package fits

```math
E\bigl(\{\hat{\boldsymbol e}_a\},\{\boldsymbol u_a\}\bigr)
  = j_0 + \sum_{\varphi} J_\varphi\,\Phi_\varphi\bigl(\{\hat{\boldsymbol e}_a\},\{\boldsymbol u_a\}\bigr),
```

a linear model in symmetry-adapted invariants ``\Phi_\varphi`` over clusters of atoms. Each
cluster site is *decorated*: a spin factor built from real tesseral spherical harmonics of
the unit direction ``\hat{\boldsymbol e}_a``, a displacement factor built from solid
harmonics of ``\boldsymbol u_a``, or both. A pure-spin model — the Drautz–Fähnle
spin-cluster expansion — is the sub-case where every site carries a spin factor and nothing
else, and it is still the most common one.

Three DFT observables train it, and all three are derivatives of that one surface:

| observable | what it is |
|---|---|
| energy ``E`` | the fitted quantity |
| torque ``\boldsymbol\tau_a = -\hat{\boldsymbol e}_a \times \partial E/\partial\hat{\boldsymbol e}_a`` | the Landau–Lifshitz torque ``\boldsymbol m_a \times \boldsymbol B_{\mathrm{eff},a}`` |
| force ``\boldsymbol F_a = -\partial E/\partial \boldsymbol u_a`` | the lattice channel's target |

Translation invariance is not a space-group symmetry, so it enters as an exact linear
**acoustic sum rule** on the coefficients (on by default) rather than as a correction
afterwards.

!!! note "Status — an architectural exploration (v0)"
    This is a from-scratch re-architecture of Magesty.jl's SLCE fitting, written as a
    clean-room exercise. The numerical core (tesseral harmonics, Clebsch–Gordan
    coupling, the symmetry-adapted basis, design matrices, regression) is reimplemented
    independently and validated against Magesty.jl as a *pinned numerical oracle*.
    Bit-for-bit agreement with Magesty is **not** a goal — refining methods so results
    differ slightly is allowed and is the point. For production **pure-spin** work, see
    Magesty.jl; the joint spin–lattice channel exists only here.

## What it does

The pipeline is one straight line, each stage a small, typed object:

```
Crystal + BasisSpec ──▶ SLCEBasis ──▶ SLCEDataset ──▶ fit ──▶ SLCEModel
  (geometry, range,     (symmetry,     (DFT data:      (OLS / Ridge /  (predict_energy/torque/force,
   sector table)         SALC basis)    E, τ, F;        Lasso / …,      restrict, decorated_terms,
                                        the ASR)        staged)         force_constants, dynamical_matrix,
                                                                        strain_derivatives, to_sunny, save)
```

- **`SLCEBasis`** analyzes symmetry, enumerates cluster orbits, and builds the
  symmetry-adapted SALC basis for a `Crystal` and a `BasisSpec` — body order, cutoffs,
  per-species `l`/`p` caps, per-sector SOC selection, or a full `Sector` table for a
  joint model.
- **`SLCEDataset`** pairs the basis with training data, materializes the energy, torque
  and force design matrices, and builds the acoustic sum rule's null space.
- **`fit`** solves for the coefficients with a pluggable estimator, co-fitting the three
  blocks under chosen weights, in one shot or in physical stages.
- **`SLCEModel`** predicts all three observables, tabulates coefficients, persists to a
  human-readable TOML, and is the input to the derivative readouts — force constants and
  dynamical matrices, strain and magnetoelastic response, magnon–phonon vertices, volume
  grids — plus export to [Sunny.jl](https://sunnysuite.github.io/Sunny/), phonopy and
  ALAMODE.

## The package family

This package builds and fits models. Three siblings consume them, each through the public
introspection surface only — and `KB_EV`, `resolve_kt` and the generics `n_atoms` /
`has_disp` are defined here once and extended there, never copied.

| Package | Role |
|---|---|
| **`SLCE.jl`** | the basis and the fit |
| [`SLCETools.jl`](https://github.com/Tomonori-Tanaka/SLCETools.jl) | the VASP adapter, mean-field sampling, single-cell configuration MC |
| [`SLCEMonteCarlo.jl`](https://github.com/Tomonori-Tanaka/SLCEMonteCarlo.jl) | full supercell Monte Carlo — spins, displacements, NPT strain moves, parallel tempering |
| [`SLCEDynamics.jl`](https://github.com/Tomonori-Tanaka/SLCEDynamics.jl) | spin dynamics: LLG / sLLG, quantum thermostat, ``S(q,\omega)`` |

## Documentation

| Page | What's there |
|------|--------------|
| [Getting started](getting_started.md) | Install, then fit and recover a Heisenberg coupling in a dozen lines |
| [Guide: building the basis](guide/basis.md) | `Crystal`, `BasisSpec`, the `Sector` table, periodic resolvability, symmetry, body order |
| [Guide: persistence and I/O](guide/io.md) | Training data (`TrainingDatum`), save/reload, human-authored `input.toml`, the DFT-source seam |
| [Guide: data and fitting](guide/fitting.md) | Datasets, the three-block co-fit, the acoustic sum rule, identifiability, estimators, staged fits, selection, diagnostics |
| [Guide: a joint model end to end](guide/joint.md) | The five decisions a spin–lattice fit needs, and which page settles each |
| [Guide: reading a fitted model](guide/introspection.md) | `decorated_terms`, the scale rule, `restrict` — the surface downstream packages read |
| [Guide: force constants and phonons](guide/lattice_dynamics.md) | `force_constants`, `dynamical_matrix`, the magnetic space group, phonopy / ALAMODE export |
| [Guide: strain and volume grids](guide/strain.md) | `strain_derivatives`, magnetoelastic constants, magnon–phonon vertices, `StrainedModels` |
| [Guide: Sunny export](guide/sunny.md) | Turn a fitted model into a `Sunny.System` for linear spin-wave theory |
| [Tutorials](tutorials/index.md) | Narrated end-to-end runs (Heisenberg chain, kagome three-body, bcc Fe from real DFT data) |
| [Theory](theory/index.md) | The spin-channel formalism, minimum-image/Wigner–Seitz resolvability, the rebuild's architecture |
| [Verification](verification/angular_momentum.md) | Human-readable numerical checks, recomputed at every docs build (Clebsch–Gordan, Wigner-D) |
| [API reference](api.md) | The exported and public (qualified) types and functions |

## Key features

- **Pluggable seams** via multiple dispatch + package extensions: symmetry backends
  (`NoSymmetry` in-tree, `SpglibBackend` in an extension) and estimators (`OLS`/`Ridge`
  in-tree, `Lasso`/`ElasticNet` in a GLMNet extension). The core loads with no heavy
  dependencies.
- **Arbitrary body order** — pairwise-within-cutoff cliques, with the SALC projection
  generalized to the combined (ordering × coupling-path × `Lf`) space so that
  permutation-equivalent sites at `N ≥ 3` are handled.
- **Minimum-image / Wigner–Seitz resolvability** — the default enumerates exactly the
  pairs (and N-body clusters) a finite supercell can resolve, so no aliased, collinear
  interactions enter the fit. See [Periodic resolvability](theory/resolvability.md).
- **Energy + torque + force co-fit** — all three are analytic derivatives of the same
  energy surface, so they cannot drift apart, and translation invariance is imposed
  exactly rather than corrected.
- **Joint spin–lattice channel** — a `Sector` table declares which decorated cluster
  families exist; the mixed projector carries the paramagnetic grey group, so evaluating a
  derivative at a fixed magnetic state reduces it to that state's **magnetic** space group
  without the group ever being declared.
- **Lattice-dynamics and strain readouts** — force constants and dynamical matrices,
  elastic and magnetoelastic response with the strain measure pinned, magnon–phonon
  vertices, re-expanded effective models, and volume grids for magnetovolume coupling;
  phonopy and ALAMODE export are pinned against those codes themselves.
- **It says when it cannot answer** — `identifiability` reports what the data determine,
  `asr_residual` and the origin-independence check refuse a description-dependent elastic
  constant, and unrepresentable channels are reported as skipped rather than dropped.
- **Sunny.jl export** — bilinear exchange + single-ion, on the training supercell or
  unfolded onto the chemical primitive cell.

## Citation

The method this package re-implements is described in:

> T. Tanaka and Y. Gohda, "General spin models from noncollinear spin density functional
> theory and spin-cluster expansion", *Phys. Rev. Research* **8**, 023300 (2026).

## References

1. R. Drautz and M. Fähnle, "Spin-cluster expansion: Parametrization of the general
   adiabatic magnetic energy surface with ab initio accuracy", *Phys. Rev. B* **69**,
   104404 (2004).
2. R. Drautz, "Spin-cluster expansion: tesseral-harmonic / cluster-expansion formalism",
   *Phys. Rev. B* **102**, 024104 (2020).
