# Spin–lattice coupled cluster expansion — feasibility and implementation map

**Status**: feasibility settled (2026-07-25); implementation not started.

Assessment of the unified spin+displacement cluster expansion (the "modified
cluster" formalism: spin sites carrying tesseral ranks × displacement sites
carrying solid-harmonic ranks, CG-coupled and projected to the identity of the
grey magnetic point group) against the current SCEFitting / SCEMonteCarlo
architecture. Source theory: the private working notes under
`~/Documents/TokyoTech/grant/kakenhi/2026/` (theory note §1–§7 and the
displacement-basis addendum). Decision: **implement inside SCEFitting.jl** as a
generalization of the existing basis layer, not as a fork of another package.

## Verdict

Feasible, and cheaper than the theory note assumed. The note lists four new
modules — (a) modified cluster type, (b) product representation (axial ×
polar), (c) real CG tables, (d) site-permutation recoupling — of which **(c)
and (d) are already implemented here**, and (b) is 90% done with one
correctness hazard. (a) is the real work, plus items the note does not cover
(evaluation kernels, data layer, MC plumbing, the strain layer).

## Mapping the four "new" modules to existing code

### (c) Real CG tables — DONE

`AngularMomentum` already has the full stack: `clebsch_gordan` (Racah form,
overflow-guarded), `coupling_paths` (all left-coupling trees), 
`coeff_tensor_complex`, `c2r_matrix` + `complex_to_real_tensor` (tesseral
realization with the global phase), `nmode_mul`
(`src/basis/AngularMomentum.jl:51-301`). All of it is a pure function of an
integer rank list — it does not know whether a rank is a spin `l` or a
displacement rank. Genuine CG, not Gaunt (the note's warning (i) is moot).

### (d) Site-permutation recoupling — DONE

`_project_and_fold` (`src/basis/salcbasis.jl:173-244`) projects in the
combined **(ordering × coupling path × Mf)** space precisely because a
stabilizer permutation mixes coupling paths and `l`-orderings. The "6j-type
recoupling" the note's review flagged as a fourth module is already handled —
exercised in production by unequal-`l` clusters with permuting stabilizers
(the l044 model). No new machinery needed.

### (b) Product representation (axial × polar) — one hazard

`wignerD_real(l, R)` (`src/basis/AngularMomentum.jl:116-136`, Fibonacci-fit
construction) is generic in `l` and correct for improper `R`; it is exactly
the matrix a **polar** rank-`l` tensor needs, so displacement factors reuse it
verbatim.

**Hazard**: the code never applies the axial factor `det(R)^l` for spins —
correctness currently rests on the implicit cancellation
`det(R)^{Σl} = +1` from the Σl-even rule (see `docs/design-notes.md`, the
found-and-fixed centrosymmetric bug). In a joint basis the cancellation needs
`Σl_spin` even **separately**. That is exactly the physical time-reversal rule
(spins are the only T-odd DOF), so the fix is confined to the parity filter —
but it must become explicit and tested, on a centrosymmetric cell, with a
mixed spin+displacement cluster. `SymOp.is_proper` is currently dead code;
either use it in the test or document why the polar Wigner remains correct
for both channels.

### (a) Modified cluster type — the real work

Cluster enumeration/orbits (`src/clusters/`) are DOF-agnostic (atoms +
integer shifts); including non-magnetic ligand sites is structurally natural.
What changes is the per-site label: from `l::Int` to `(channel, degree k,
rank l)`. Cascade: `SALCTerm.ls` / `SALCKey.ls` (`src/basis/salc.jl:9-34`),
persist schema (v4 → v5, back-read v4 as spin-channel-only), `coeftable`,
`salc_groups`/`group_costs` (`src/fitting/selection.jl`), `MultipoleTerm`
(`src/sce/introspect.jl:25` — superseded by D0: frozen p = 0 view that throws
on joint models; successor `DecoratedTerm`, see `spin-lattice-ce-design.md`
§7).

## Displacement basis convention (settled in the addendum)

No ACE-style radial basis. Per-site degree-`p` monomials decompose as
`Sym^p(R³) ≅ ⊕_k |u|^{2k} · {rank-(p−2k) solid harmonics}`, so the site
label set at degree `p` is `(k, l)` with `2k + l = p`, i.e. `l ≤ p`,
`l ≡ p (mod 2)`, `k = (p−l)/2`, each once. Consequences:

- The CG/projection engine sees only `l`; `k` is a passive degree label.
- Enumerating only harmonic-decomposition labels builds the `Sym^k`
  restriction (`u × u = 0`) in by construction — antisymmetric channels never
  reach the projector, and the character-formula count (cycle-type +
  plethysm) must match the projector dimension exactly (unit test).
- `l = 0` is legitimate for displacement sites (`k ≥ 2` even: the `|u|^k`
  trace factors) — the current `l ≥ 1` rule (`src/basis/salcbasis.jl:38`)
  becomes channel-dependent. Do not drop the trace channels: they *are* the
  radial dependence.
- Same-site repeated displacement is a site **degree**, not repeated member
  slots — preserving the distinct-member-sites invariant that both the
  aliasing argument here and SCEMonteCarlo's exact single-site ΔE rely on.
- **Evaluation must be polynomial**: `Z_lm(û)` is direction-singular at
  `u → 0`, which is the most-sampled point. Evaluate real solid harmonics
  `R_lm(u)` directly as homogeneous polynomials (smooth at 0), with the
  full **Euclidean** gradient for forces. The spin kernels
  (`Harmonics.Zlm`, tangent-projected `grad_Zlm`) cannot be reused — the
  displacement evaluator is a separate small module.
- By-product: all displacement factors have `k ≥ 1`, so `E` at `u = 0`
  degenerates **exactly** to the clamped-ion spin SCE — the double-counting
  protocol (theory note §7, item 1) is guaranteed structurally.

## Isotropy generalization (SOC-less sector)

Couple the tree spin-sites-first: the total spin rank `L_S` is then an
intermediate label in `Lseq`, and the SOC-less constraint (`L_S = 0`, lattice
factor projected by the plain space group) is a one-line filter of the same
shape as the current `isotropy && Lf != 0` (`src/basis/AngularMomentum.jl:296`).
The "switch the group action" claim of the theory note holds at code level.

## Fitting layer

- Data: `SpinDatum` (`src/io/dftsource.jl:40`) has no positions /
  displacements / forces. New joint datum type + a force design matrix
  (structural sibling of `_design_torque`, `src/fitting/design.jl:24-53`,
  but with the unprojected gradient).
- Estimators (`GroupAdaptiveRidge`, GCV/`effective_dof`, `select_fit`,
  grouped CV) are generic linear regression — reusable verbatim. The
  hierarchical fit (SOC-less `p ≤ 2` sectors from forces+torques → freeze →
  SOC sectors from energy differences) needs only a frozen-subset offset in
  `fit`.
- ASR: per-spin-sector linear equality constraints on `K` (new: equality
  constraints in the estimator layer, or a relative-coordinate basis).
- Conditioning: displacement monomials are radially non-orthogonal. The
  existing centering/whitening in `_assemble_problem` is the first line of
  defense; fix a `u/d_NN` scaling convention in the basis (precedent: the
  `(4π)^(body/2)` convention is likewise documented and left to consumers).

## SCEMonteCarlo side (surveyed 2026-07-25)

Lighter than expected. The hot kernels touch state only through
`zrows[row, site]` — a generic per-site basis row. With joint rows
(spin `lm` × displacement `(k,l,m)` entries):

- `_ContractionPrograms` / `_total_energy` / `site_coeffs!` / `delta_energy`
  work verbatim (the requirement is multilinearity across sites, not
  spin-ness); only row-index arithmetic and `ScaledTerm` (per-slot channel
  labels) generalize. Folded-tensor axis dims stay `2l+1`.
- A per-site displacement Metropolis move drops into the existing
  color-parallel sweeps: same touched-instance sets → same conflict graph →
  same coloring. Plumbing: `disp` in `ChainState`, second adaptive step
  scalar, `_swap_payload!` (PT), checkpoint schema bump.
- `site_active` must split into spin-active / displacement-active (ligands
  are spin-inactive, displacement-active).
- `Observable` signature (`f(config, energy, H)`) widens to carry
  displacements — ripples to `standard_observables`, PT, and
  SCESpinDynamics' trajectory observable.
- Conventions to document: `Evaluable` normalizer (`n_active` = magnetic
  sites is the wrong intensive denominator with lattice DOF); MC specific
  heat is configurational only (no momenta).

**Uniform strain is the one genuinely new layer** (the note's own BLOCKER):
a global move outside the sweep/coloring machinery — full-energy Metropolis
step in the outer temperature loop (cost ≈ one sweep, fine at 1/sweep) with
`K(ε)` on a strain grid + elastic term. Boundedness diagnostics (even-order
truncation, min-eigenvalue monitoring of `Φ^(2)({ê})`, displacement-radius
check) are new runtime guards.

SCESpinDynamics has no structural blocker for later spin–lattice dynamics
(parallel Verlet/Langevin layer + `−∂E/∂u` force channel); the prerequisite
is entirely upstream, and any new DOF must join the keyed-Philox
determinism discipline.

## Unit tests with known answers (theory note §7 series)

1. Spin-less limit → spherical-tensor force-constant basis. The external
   cross-check against the JPSJ 95, 053601 code was **dropped by user decision
   (2026-07-27)**; the limit is gated internally instead — counting oracle for the
   basis dimension, finite-difference forces and the Γ-restricted `Σ_R Φ(R)` vs an
   FD Hessian for the derivatives, and the three zero eigenvalues of `D(0)` under
   the ASR. See design record §12 gate (m) for what the drop does and does not cover.
2. Displacement-less limit → current SCE basis (automatic: it *is* this
   package; pin bit-identity of the spin-only path across the change).
3. Cubic single-site `Z_2 ⊗ [u⊗u]` → exactly 2 invariants (`B_1`, `B_2`).
4. SOC-less pair + ligand, `p = 1` → `dJ/dr` term + ligand term; no
   DMI-type invariant.
5. Character-formula count (cycle-type + plethysm) ≡ projector dimension for
   every cluster — **must include a pair cluster with site-permuting
   stabilizer** (the naive product formula coincides on permutation-free
   cases) and a centrosymmetric cell (the `det(R)^{Σl_spin}` hazard).
6. Dimension asserts: single site `p = 2` → 6 = 5+1, `p = 3` → 10 = 7+3.

## Risk ordering

1. Explicit `det(R)^{Σl_spin}` parity handling (correctness; killed by test 5).
2. `(channel, k, l)` label cascade (mechanical but wide: persist / coeftable /
   selection / introspect / `ScaledTerm`).
3. Solid-harmonic evaluator + Euclidean gradient (new kernel; polynomial,
   exact gates easy).
4. Strain + NPT layer (genuinely new architecture; design sketched above).
5. Energy-scale hierarchy + ASR equality constraints (fit quality; estimators
   reusable).

## Non-goals (this stage)

Longitudinal-moment (amplitude) extension — the only place a radial basis
`R_n(|m|)` ever becomes necessary; velocity-coupling / Berry-curvature
phonon effects (outside a static energy surface by construction).
