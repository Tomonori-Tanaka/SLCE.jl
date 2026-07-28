# SLCE.jl

```@meta
CurrentModule = SLCE
```

A clean, extensible, Julia-native rebuild of [Magesty.jl](https://github.com/Tomonori-Tanaka/Magesty.jl) —
fitting **spin–lattice cluster expansion (SLCE)** models to noncollinear DFT data.

Given a crystal and a set of noncollinear spin configurations with their DFT energies
(and, optionally, torques), the package fits

```math
E\bigl(\{\hat{\boldsymbol e}_a\}\bigr) = j_0 + \sum_{\varphi} J_\varphi\,\Phi_\varphi\bigl(\{\hat{\boldsymbol e}_a\}\bigr),
```

where the spins ``\hat{\boldsymbol e}_a`` are unit vectors and the ``\Phi_\varphi`` are
symmetry-adapted, time-reversal-even scalar invariants built from real tesseral
spherical harmonics over clusters of spins. Fitting recovers the cluster coefficients
``J_\varphi``; the same coefficients fix the per-atom **torque**
``\boldsymbol\tau_a = -\hat{\boldsymbol e}_a \times \partial E/\partial\hat{\boldsymbol e}_a``
(the Landau–Lifshitz / physical torque ``\boldsymbol m_a \times \boldsymbol B_{\mathrm{eff},a}``),
the SLCE's other DFT observable.

!!! note "Status — an architectural exploration (v0)"
    This is a from-scratch re-architecture of Magesty.jl's SLCE fitting, written as a
    clean-room exercise. The numerical core (tesseral harmonics, Clebsch–Gordan
    coupling, the symmetry-adapted basis, design matrices, regression) is reimplemented
    independently and validated against Magesty.jl as a *pinned numerical oracle*.
    Bit-for-bit agreement with Magesty is **not** a goal — refining methods so results
    differ slightly is allowed and is the point. For production use, see Magesty.jl.

## What it does

The pipeline is one straight line, each stage a small, typed object:

```
Crystal + BasisSpec ──▶ SLCEBasis ──▶ SLCEDataset ──▶ fit ──▶ SLCEModel
   (geometry, range)      (symmetry,    (DFT data:     (OLS / Ridge /   (predict_energy,
                           SALC basis)   E, τ)          Lasso / …)        predict_torque,
                                                                          to_sunny, save)
```

- **`SLCEBasis`** analyzes symmetry, enumerates cluster orbits, and builds the
  symmetry-adapted SALC basis for a `Crystal` and a `BasisSpec` (body order, cutoff,
  per-species `l`, SOC selection).
- **`SLCEDataset`** pairs the basis with training data and materializes the energy
  (and torque) design matrices.
- **`fit`** solves for the coefficients with a pluggable estimator, optionally co-fitting
  energy and torque.
- **`SLCEModel`** predicts energies and torques, tabulates coefficients, persists to a
  human-readable TOML, and exports to [Sunny.jl](https://sunnysuite.github.io/Sunny/) for
  spin-wave theory.

!!! tip "Sampling and active learning live in SLCETools.jl"
    This package is focused on **building and fitting** SLCE models. To *generate* spin
    configurations from a fitted model — finite-temperature mean-field (MFA) sampling, and
    (planned) active-learning model construction — use the companion package
    [SLCETools.jl](https://github.com/Tomonori-Tanaka/SLCETools.jl). It depends on
    SLCE and reads a fitted model only through the public
    [introspection surface](api.md#Fitted-model-introspection)
    (`spin_multipole_terms`, `bilinear_terms`, `SLCE.Harmonics`).

## Documentation

| Page | What's there |
|------|--------------|
| [Getting started](getting_started.md) | Install, then fit and recover a Heisenberg coupling in a dozen lines |
| [Guide: building the basis](guide/basis.md) | `Crystal`, `BasisSpec`, periodic resolvability, symmetry, body order |
| [Guide: data and fitting](guide/fitting.md) | Datasets, the energy + torque co-fit, estimators (OLS / Ridge / Lasso / elastic-net), diagnostics |
| [Guide: persistence and I/O](guide/io.md) | Save/reload models, human-authored `input.toml`, the DFT-source (VASP) seam |
| [Guide: Sunny export](guide/sunny.md) | Turn a fitted model into a `Sunny.System` for linear spin-wave theory |
| [Tutorials](tutorials/index.md) | Narrated end-to-end runs (Heisenberg chain, kagome three-body) |
| [Theory](theory/index.md) | The SLCE formalism, minimum-image/Wigner–Seitz resolvability, the rebuild's architecture |
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
- **Energy + torque co-fit** — the torque is the analytic derivative of the same energy
  surface, so the two are consistent by construction.
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
