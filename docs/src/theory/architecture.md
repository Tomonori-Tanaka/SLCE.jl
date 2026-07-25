# Architecture

```@meta
CurrentModule = SLCE
```

This chapter collects the design choices that distinguish the rebuild from Magesty.jl. The
goal of the rebuild is a clean, extensible re-architecture of the same SCE method; these are
the seams that make it so.

## Pluggable seams via package extensions

Two axes of behavior are behind abstract types so that heavy or optional dependencies stay
out of the core: symmetry backends ([`AbstractSymmetryBackend`](@ref)) and estimators
([`AbstractEstimator`](@ref)). The *types* live in the core — so they can be named,
validated, and dispatched on without the dependency — while the heavy *method* lives in an
extension that lights up when you load its backend:

| Type (core) | Method (extension) | Activated by |
|-------------|--------------------|--------------|
| [`SpglibBackend`](@ref) | `analyze_symmetry(::SpglibBackend, …)` | `import Spglib` |
| [`ElasticNet`](@ref) / [`Lasso`](@ref) | `solve_coefficients(::ElasticNet, …)` | `using GLMNet` |
| (a fitted model) | [`to_sunny`](@ref) | `using Sunny` |

Calling an unloaded backend raises a friendly "load the backend" error on the abstract
supertype rather than a bare `MethodError`. The core thus loads with no heavy dependencies,
and adding `NoSymmetry` / `OLS` / `Ridge` in-tree keeps a working baseline with zero
optional packages.

## The core/extension split

The export to Sunny.jl follows a sharper version of the same idea, because the conversion is
numerically load-bearing. The **conversion math** — the harmonic-to-Cartesian exchange and
single-ion matrices, the directed-member folding, the primitive-cell unfold — lives in a
**dependency-free core layer** and is gated *without Sunny* by reconstructing the energy:
the core checks that ``\sum \hat{\boldsymbol e}^\top M \hat{\boldsymbol e} \approx
\text{predict\_energy} - j_0``. The extension is then a thin layer that only assembles a
`Sunny.System` from the core-computed matrices, and a separate test environment checks the
real `Sunny.System` energy. The numerically critical part is therefore tested in the main
suite with no heavy dependency, and the same pattern (critical math in core, object
assembly in the extension) is the template for future exporters.

## Canonical `SALCKey` addressing

A design-matrix column must mean a fixed interaction regardless of the order in which the
basis happened to be built. Every SALC is therefore addressed by a canonical
[`SALCKey`](@ref) — `(body, orbit_id, sorted l-multiset, Lf, block)` — that is injective
across the basis. The columns of the design matrix, the coefficients written by
[`coeftable`](@ref), and the per-key coefficients in a persisted document all key off it, so
a reloaded model re-pairs its coefficients to a freshly built basis **by key**, not by
position. A fingerprint (`hash` of the sorted keys) guards against pairing coefficients to
the wrong basis.

## Combined-space projection

At ``N \ge 3`` a space-group operation can *permute* the sites of a cluster. When the
permuted sites carry unequal angular momenta, this mixes both the intermediate coupling
paths and the ``l``-orderings, and a single coupling-path / single-``l`` projection can no
longer express the invariant. The rebuild runs the SALC projection over the combined
**(ordering × coupling-path × ``L_f``)** space: it builds the action of each stabilizer
operation by contracting against the package's own orthonormal coupled tensors (no
``6j``/``9j`` symbols, the same spirit as deriving the Wigner-``D`` from ``Z_{l,m}``), forms
the projector as the average over the stabilizer, and reads its eigenvalue-1 subspace. A
SALC is consequently **multi-term**: it can carry several ``l``-orderings, one *term* each,
and [`evaluate_salc`](@ref) and the torque-gradient kernel both loop over them. This is what the
[kagome three-body tutorial](../tutorials/kagome_threebody.md) exercises. Cross-validated
against Magesty through three-body, the per-`(body, ls, Lf)` invariant-subspace dimensions
agree exactly.

## Validation against a numerical oracle

Bit-for-bit agreement with Magesty.jl is explicitly **not** a goal — refining methods so
results differ slightly is allowed and is the point. Validation is therefore intrinsic and
physical rather than reference-matching:

- **Symmetry invariance** — every SALC satisfies ``\Phi(g\cdot\boldsymbol e) =
  \Phi(\boldsymbol e)`` and ``\Phi(-\boldsymbol e) = \Phi(\boldsymbol e)`` (the projector's
  exactly-0/1 eigenvalues are the in-band gate).
- **Finite differences** — `predict_torque` matches ``-\hat{\boldsymbol e}\times\nabla
  E_{\text{FD}}`` on the sphere, so the torque is provably the (negative rotation-)
  derivative of the energy surface.
- **Known-coupling recovery** — a Heisenberg chain recovers ``J = 2\sqrt3\,j_\varphi``; a
  synthetic ``N``-body model is recovered from energies and torques.
- **A pinned oracle** — a separate environment `dev`s a pinned Magesty.jl and cross-checks
  *convention-fixed kernels and gauge-invariant aggregates* (invariant-subspace dimensions,
  bit-identical harmonics), never raw gauge-dependent SALC coefficients.

The heavy or convention-sensitive validations each live in their own environment — the
Magesty oracle, the Sunny `System` energy check, the GLMNet estimator suite — so the core
test suite stays dependency-free.
