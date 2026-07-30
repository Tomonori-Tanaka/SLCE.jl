# Tutorials

Narrated, end-to-end runs of the full workflow, all executed when this site is built. The
first two also exist as standalone scripts under
[`examples/`](https://github.com/Tomonori-Tanaka/SLCE.jl) in the repository, each with its
own `@assert` gate on the coupling it recovers; the bcc Fe case study runs only here.

All three are **pure-spin**. For the joint spin–lattice path, follow [a joint model end to
end](../guide/joint.md).

```@contents
Pages = ["heisenberg_chain.md", "kagome_threebody.md", "case1_bcc_fe.md"]
Depth = 1
```

- **[Heisenberg chain](heisenberg_chain.md)** — the canonical first run: build a 2-body
  isotropic basis, fit synthetic Heisenberg energies, recover the coupling `J`, then add a
  torque co-fit. Establishes the `basis → dataset → fit → predict` rhythm.
- **[Kagome three-body](kagome_threebody.md)** — beyond pairs: a 3-body interaction on the
  kagome lattice, where symmetry-equivalent sites force several `l`-orderings into one
  multi-term SALC. Recovers a synthetic 3-body model from energies and torques.
- **[Case 1: bcc Fe](case1_bcc_fe.md)** — a real worked example: fit a 128-atom bcc Fe
  supercell against constrained-noncollinear DFT energies and torques, read the result
  back out as Heisenberg exchange constants ``J_{ij}``, and compute the magnon dispersion
  with Sunny.jl against neutron data.

For shorter, feature-by-feature snippets see the [Guide](../guide/basis.md); for the
fastest possible first fit see [Getting started](../getting_started.md).
