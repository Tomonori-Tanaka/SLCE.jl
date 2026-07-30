# The spin-cluster expansion

```@meta
CurrentModule = SLCE
```

This page derives the **spin channel** of the expansion — the classical spin-cluster
expansion of Drautz and Fähnle, where every cluster factor is a function of the spin
directions alone. That is the sector the package started from and still the one most
models use. The full spin–lattice cluster expansion (SLCE) the package now implements
decorates the same clusters with displacement factors as well; its construction is the
decision record `docs/specs/spin-lattice-ce-design.md`, and its user-facing surface is
[Reading a fitted model](../guide/introspection.md).

## The problem

A noncollinear spin-DFT calculation returns, for one fixed set of spin directions
``\{\hat{\boldsymbol e}_a\}``, a total energy and the torque on every magnetic atom. We want
a closed-form *effective spin model* — an energy function

```math
E\bigl(\{\hat{\boldsymbol e}_a\}\bigr)
```

that reproduces the DFT energies (and torques) across the whole space of noncollinear
configurations, not just the handful that were computed. Each ``\hat{\boldsymbol e}_a`` is a
unit vector along the classical spin on atom ``a``; the moment magnitude is a separate
quantity.

## The expansion

The spin-cluster expansion writes the energy as a sum over *clusters* of atoms — single
sites, pairs, triplets, … — each contributing a basis function of the spins on its atoms:

```math
E\bigl(\{\hat{\boldsymbol e}_a\}\bigr)
  = j_0 + \sum_{\varphi} J_\varphi\,\Phi_\varphi\bigl(\{\hat{\boldsymbol e}_a\}\bigr).
```

``j_0`` is a constant reference energy and each ``J_\varphi`` is a scalar coefficient to be
fitted. This is the angular-momentum analogue of the cluster expansion for substitutional
alloys: the degrees of freedom are continuous directions on the unit sphere, so the per-site
factors are spherical harmonics rather than occupation numbers.

## Building the invariants

Each basis function is built from per-site **real (tesseral) spherical harmonics**
``Z_{l,m}(\hat{\boldsymbol e})``. For a cluster of ``n`` atoms the elementary block is a
product of one harmonic per atom,

```math
\prod_{i=1}^{n} Z_{l_i, m_i}(\hat{\boldsymbol e}_{a_i}),
```

and the design matrix carries a factor ``(4\pi)^{n/2}`` so an ``n``-body term is ``O(1)``.

!!! note "Both rules below are rules about the SPIN channel"
    On this page every cluster site carries exactly one spin harmonic, so the ``4\pi``
    count and the parity screen can be written per *site*. In the full expansion they are
    per **spin slot**: the scale is ``(4\pi)^{n_{\mathrm{spin\,slots}}/2}`` (the
    displacement kernel is normalized ``4\pi``-free, so displacement axes contribute
    none), and the time-reversal screen applies to ``\sum_i l_i`` over the spin slots
    only, leaving displacement ranks unconstrained. The two readings coincide exactly
    when the spin-slot count equals the body order. A consumer must read the scale off
    [`DecoratedTerm`](@ref)'s `scale` field rather than re-derive it from the cluster
    shape — see [Reading a fitted model](../guide/introspection.md).

Two requirements turn these products into physical basis functions:

1. **Time-reversal symmetry.** Reversing all spins, ``\hat{\boldsymbol e}_a \to
   -\hat{\boldsymbol e}_a``, must leave the energy unchanged. Since ``Z_{l,m}(-\hat{\boldsymbol e})
   = (-1)^l Z_{l,m}(\hat{\boldsymbol e})``, only products with even total spin degree
   ``\sum_i l_i`` survive — so only those ``l``-assignments are enumerated.

2. **Space-group invariance.** The energy must be invariant under the crystal's symmetry.
   The harmonics of one cluster are coupled to a total angular momentum ``L_f`` (via
   Clebsch–Gordan coefficients), and only the combinations transforming as the identity are
   kept. Summed over the cluster's symmetry orbit, these are the **symmetry-adapted linear
   combinations (SALCs)** ``\Phi_\varphi``. The package computes the real Wigner-``D``
   matrices for this projection directly from its own ``Z_{l,m}`` (by an exact
   least-squares fit), so the convention is consistent by construction and improper
   rotations are handled natively.

The isotropic channel ``L_f = 0`` gives the familiar rotation invariants — for a pair with
``l = (1,1)`` this is the Heisenberg ``\hat{\boldsymbol e}_i \cdot \hat{\boldsymbol e}_j``;
the anisotropic ``L_f > 0`` channels give Dzyaloshinskii–Moriya and symmetric-Γ terms (for
``l = (1,1)``) and single-ion / biquadratic terms (for higher ``l``).

## Evaluation

A SALC stores, per symmetry-orbit *member*, the participating atoms and a *folded*
coefficient tensor; its value on a configuration is

```math
\Phi_\varphi(\{\hat{\boldsymbol e}_a\})
  = (4\pi)^{n/2}\sum_{\text{members}}\sum_{\text{terms}}\sum_{\mu}
    \text{folded}[\mu]\prod_{i} Z_{l_i,\mu_i}(\hat{\boldsymbol e}_{a_i}).
```

(The inner *terms* sum is over `l`-orderings and is a single term except for the
unequal-`l`, permutation-equivalent case at ``N \ge 3``; see
[Architecture](architecture.md#Combined-space-projection).) [`evaluate_salc`](@ref) implements
this kernel.

## The design matrix and the fit

Stacking ``\Phi_\varphi`` over configurations gives the **energy design matrix**
``X_E[\text{config}, \varphi] = \Phi_\varphi(\text{config})``, materialized by
[`SLCEDataset`](@ref). The fit is then a linear regression of the DFT energies on ``X_E``.
[`fit`](@ref) column-centers ``X_E``, so the intercept ``j_0`` is recovered analytically as
the mean residual — *independent of the estimator* — and the centered problem is solved by
the chosen [`AbstractEstimator`](@ref) (OLS, ridge, or a GLMNet Lasso / elastic net).

## The torque

Its second observable is the per-atom torque — the Landau–Lifshitz / physical
torque ``\boldsymbol m_a \times \boldsymbol B_{\mathrm{eff},a}``, i.e. the negative of the
energy's rotation gradient:

```math
\boldsymbol\tau_a = -\,\hat{\boldsymbol e}_a \times \frac{\partial E}{\partial \hat{\boldsymbol e}_a}.
```

This is the *analytic* gradient of the same energy surface, so [`predict_torque`](@ref) and
[`predict_energy`](@ref) can never drift apart: the gradient kernel shares the energy
kernel's ``\mu``-mapping and ``(4\pi)^{n/2}`` scale (the gate is a finite-difference
self-consistency check, ``\boldsymbol\tau \approx -\hat{\boldsymbol e}\times\nabla E_{\text{FD}}``).
This matches the convention of the *General spin models* paper (Ref. 3); it is the opposite
sign of the energy-rotation-gradient ``+\hat{\boldsymbol e}\times\nabla E``.
Torques carry their own design matrix ``X_T``; an energy + torque co-fit minimizes
``(1-w)\,\mathrm{MSE}_E + w\,\mathrm{MSE}_T`` by whitening and stacking the two blocks (see
[Data and fitting](../guide/fitting.md)).

Both blocks above are the spin channel's. A joint fit adds a **third** block, the forces
``X_F`` (``\boldsymbol f = -\partial E/\partial \boldsymbol u``), and — because translation
invariance is a constraint on coefficients rather than a symmetry the basis can project out
— solves in the acoustic sum rule's null space by default (`fit(...; asr = true)`). Both are
in [Data and fitting](../guide/fitting.md).

## References

1. R. Drautz and M. Fähnle, *Phys. Rev. B* **69**, 104404 (2004).
2. R. Drautz, *Phys. Rev. B* **102**, 024104 (2020).
3. T. Tanaka and Y. Gohda, *Phys. Rev. Research* **8**, 023300 (2026).
