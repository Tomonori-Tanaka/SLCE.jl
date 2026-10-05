# Weißenhofer 2023 — rotationally invariant spin–lattice coupling

**Citation.** M. Weißenhofer, H. Lange, A. Kamra, S. Mankovsky, S. Polesya,
H. Ebert, U. Nowak, "Rotationally invariant formulation of spin-lattice coupling
in multi-scale modeling," *Phys. Rev. B* **108**, L060404 (2023).
arXiv:2211.02382 (v1 read 2026-09-08; the published Letter may differ in detail).

## What it does

Starts from the lab-frame expansion of the relativistic spin Hamiltonian in
displacements (Hellsvik 2019 / Mankovsky 2022), `Σ J_ij^{αβ} S_i^α S_j^β +
Σ J_ij,k^{αβ,μ} S_i^α S_j^β (u_k^μ − u_i^μ) + …`, notes that it is not
rotationally invariant (spins in the lab frame, lattice rotated), and repairs it
*by construction*: every Cartesian spin component is replaced by the projection
onto a **local frame built from neighbour bond vectors**,
`e_i^{α(±)} = (r_i^{α±} − r_i)/|…|`, and every displacement by a bond-length
change `(r_k − r_i)·e_k^μ − R_ki`. The result depends only on dot products of
spins and position differences, so it conserves total momentum, angular
momentum and energy, and lets spin angular momentum flow into the lattice
(Einstein–de Haas). They admit the frame choice is **neither unique nor
trivial** (which neighbours define the axis changes the equations of motion).
The rest of the Letter is a spin–lattice-dynamics (Suzuki–Trotter) study of a
free 4³ nanocube: FMR mode, easy-axis precession at `ω = 6μ_s/(m l² γ)`, and an
Einstein–de Haas rotation.

The Supplemental Material relates the microscopic tensors to continuum
magnetoelasticity by the affine substitution `u_k ≈ u_i + R_ik·∇u` and the
uniform-magnetisation limit, giving closed sums for `B₁, B₂, A₁, A₂` and a
displacement-induced DMI tensor. The energy density is written to first order in
**both** the strain `ε_μν` and the rotation tensor `ω_μν` (symmetric and
antisymmetric parts of `∇u`), citing Melcher, PRL 25, 1201 (1970) and PRL 28,
165 (1972) for the role of the rotation tensor.

## Why it matters here

- **Rotation rule of the SLCE paper (Sec. constraints).** The physics statement
  "with SOC only `𝓡_U + 𝓡_S` is a symmetry, so antisymmetric displacement
  gradients (local rotations) couple legitimately to anisotropy" is exactly
  their `B^as ω_μν` term and Melcher's point. Cite **Melcher 1970/1972** for the
  classical statement and **this paper** for the microscopic one; that resolves
  the `todoRotationalME` stub. What is *not* in their paper: the rule as a
  linear constraint on a symmetry-adapted coefficient vector, its
  block-diagonality in `L_S` (`𝓡_U E = 0` on `L_S = 0`, `𝓡_U E = −𝓡_S E`
  otherwise), the order-`n`→`n+1` grading, and the periodic-supercell argument
  for treating it as a diagnostic. Those claims stand.
- **Two different repairs.** Theirs: build invariance into the functional form
  via a heuristic local frame (a modelling choice for dynamics, no fit). Ours:
  keep a fixed reference frame, expand in a complete symmetry-adapted basis,
  and let the rotational rule be a linear condition on coefficients. The
  strain-channel design (material frame `u := Rᵀx − U R⁰` after polar
  decomposition, design record §9b/§9e) makes rigid rotation exact by
  convention without any neighbour-frame heuristic.
- **`B₁/B₂` formulas = our affine path.** Their SM Eqs. (11)–(12) are the
  first-order, uniform-`S` limit of `magnetoelastic_constants` /
  `affine_energy`. Their SM Eq. (1) writes the shear term as
  `B₂ Σ_α Σ_{β≠α} S^α S^β ε_αβ`, i.e. the Kittel constant — the same convention
  as `magnetoelastic.jl` and the corrected paper Eq. (me).
- **Possible gap in their SM (our reading, unverified against the published
  version).** They state that only on-site `J_ii,k` terms feed `B₁/B₂`; in the
  uniform-`S` limit the symmetric anisotropic part of two-site `J_ij,k`
  (`i ≠ j`, the "two-site anisotropy" their own Table 1 calls dominant in FePt)
  also contributes. The SLCE readout sums over all pairs, so no change is needed
  here; worth a sentence if we compare numbers.
- **Not a strain-as-DoF paper.** Strain enters only as the continuum limit of
  displacements; there is no global strain variable, no stress observable, no
  fit. The strain-channel novelty claims (N1–N5 in
  `STUDY-2026-09-07-strain-channel-panel.md` §6) are untouched.

(PDF at `papers/weissenhofer-2023-rotational-slc.pdf` — local only.)
