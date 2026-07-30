# Theory

This section explains *what* the package computes and *why the rebuild is shaped the way
it is*. It has three chapters, readable independently:

```@contents
Pages = ["slce.md", "resolvability.md", "architecture.md"]
Depth = 2
```

- **[The spin-cluster expansion](slce.md)** — the formalism: the energy as a linear model in
  symmetry-adapted invariants of real spherical harmonics, the design matrix, the fit, and
  the torque as the analytic derivative of the same surface.

!!! note "The joint spin–lattice construction is not (yet) a chapter here"
    These three chapters derive the **spin channel**. The displacement channel — solid
    harmonics, per-site decorations, the coupled spin rank ``L_S``, the mixed projector, and
    why the acoustic sum rule has to be a constraint on coefficients rather than a basis
    projection — is derived in the repository's decision record
    `docs/specs/spin-lattice-ce-design.md`, which is **not part of this published site**.
    What *is* published is its user-facing half: [a joint model end to
    end](../guide/joint.md), the sector table in [Building the basis](../guide/basis.md),
    and the sum rule as an operational matter in [Data and
    fitting](../guide/fitting.md#The-acoustic-sum-rule-(translation-invariance)).
- **[Periodic resolvability](resolvability.md)** — why a finite supercell can only resolve
  the minimum-image (Wigner–Seitz-cell) interactions, the face/edge/corner boundary ties,
  and the compact-cluster criterion that extends this to three- and four-body clusters.
- **[Architecture](architecture.md)** — the design choices that distinguish this rebuild
  from Magesty.jl: pluggable seams, the core/extension split, canonical `SALCKey`
  addressing, the combined-space projection for `N ≥ 3`, and the numerical-oracle
  validation methodology.

For the prose-heavy rationale behind individual refinements, the repository's
`docs/design-notes.md` is the long-form companion to this section.
