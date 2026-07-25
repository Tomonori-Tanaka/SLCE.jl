# Spin–lattice cluster expansion (SLCE) — design decision record

**Status: SETTLED (D0 closed 2026-07-25). This is the normative design.**

Companions: `spin-lattice-ce.md` (feasibility study, code-verified mapping onto the
existing packages) and `spin-lattice-ce-d0.md` (the full D0 discussion record —
three designer proposals, the adversarial review, and every argument behind the
decisions below; consult it before reopening anything here). Theory source: the
kakenhi-2026 theory note (unified spin–lattice CE), whose conventions this record
implements.

Every decision below is final for the implementation window. Reopening one
requires a written reason appended to this file, not a silent divergence.

---

## 1. Scope

One symmetry-adapted cluster expansion over decorated clusters: each cluster site
carries factors from typed **channels** — spin (axial tesseral harmonics, rank
l ≥ 1, T-parity (−1)^l) and displacement (polar solid harmonics, label (k, l),
degree 2k + l ≥ 1, T-even) — CG-coupled into cluster invariants and projected in
the grey magnetic point group of a single high-symmetry reference structure.

Principles with the force of law:

- **The joint expansion is the primary abstraction.** Spin-only SCE is the p = 0
  value of the general types, never a sibling code path.
- **u = 0 exact degeneracy**: every displacement factor has degree ≥ 1, so the
  joint model at u = 0 degenerates bitwise to the clamped-ion spin SCE. This is
  both a design principle and a permanent gate.
- **No radial basis for displacements**: solid harmonics are evaluated as
  homogeneous polynomials (recurrence, no libm), which is what makes u = 0
  regular, gradients Euclidean, and the evaluator GPU-portable.
- **Selection rules are structural**: T-parity (Σl_spin even) enforced at
  enumeration; axial-vs-polar handled without a det branch in the projector
  (§4); illegal labels unconstructable (§3).

Non-goals for this window: occupation channel (§10 reserves it), quantum
phonons, NPT/full-cell dynamics, long-range dipole blocks (§10), spin-spiral /
generalized-Bloch paths.

## 2. Naming and rename plan (D-1/D-2/D-3)

The method is the **spin–lattice cluster expansion (SLCE)**; method name =
package name.

| current | new repo | module |
|---|---|---|
| SCEFitting.jl | **SLCE.jl** | SLCE |
| SCEMonteCarlo.jl | SLCEMonteCarlo.jl | SLCEMonteCarlo |
| SCESpinDynamics.jl | SLCEDynamics.jl | SLCEDynamics |
| SCETools.jl | SLCETools.jl | SLCETools |

- The rename is the **FIRST step**: an isolated, purely mechanical commit series
  before any semantic work (never interleaved — when a gate fails mid-migration,
  `git diff` must show physics, not renames).
- **UUIDs kept** (unregistered path-dev packages; Manifests stay resolvable; the
  migration oracle uses serialized prediction fixtures, never co-loading).
- Same change updates: path-dev entries, kugui rsync target dirs, the shared env
  `@sce` → `@slce` (+ offline test-extras redo, kugui footgun #3), qualified
  cross-package references (GPU `Zlm_unsafe` port comments, Dynamics'
  `_gradient_lane_ref!` gates), docs/specs/memory cross-references.
- Gate: full local suites + kugui on-cluster suites green, before any semantic
  commit lands.
- Registry: `SLCE` is collision-free; 4-char names need a manual merge
  (precedents CUDA/JACC/STAC/Gabs/PoGO); registration deferred to publication.
  All sibling names AutoMerge cleanly.
- **One-rename principle**: no further renames after this window.

## 3. Channel abstraction

Two layers with distinct jobs (theory proposal, ratified):

**Layer 1 — isbits labels** (keys, persistence, hot loops):

- `Channel::UInt8` enum with the total order **SPIN < DISP < OCC**; OCC is
  RESERVED now (never enumerated in this window) so a future occupation channel
  is additive (§10).
- `SiteFactor(channel, k, l)` — constructor-validated: spin ⇒ k = 0, l ≥ 1;
  disp ⇒ k ≥ 0, l ≥ 0, 2k + l ≥ 1. Degree-0 constants and spin radial degrees
  are unconstructable. `l = 0, k ≥ 1` disp factors (the |u|^{2k} trace channels)
  are legal — they ARE the radial dependence.
- `SiteDecor` — the per-site combined decor (a site may carry a spin factor AND
  a displacement factor); `SALCKey` carries a sorted `Vector{SiteDecor}`.
- **Invariant: at most one factor per (site, channel) slot pair.** Slots ≠
  sites. This generalizes the distinct-member-sites rule and is exactly what
  preserves the MC's exact single-DOF ΔE.

**Layer 2 — the rep-provider trait interface** (projection/enumeration; cold):
a channel supplies (rep dimension, rep matrices D(g), T-parity, evaluation
kernel, gradient kind). Spin and disp are implemented AGAINST this interface —
never hardwiring 2l+1/Wigner — so a permutation-matrix channel (OCC) or a
site-less global channel (strain, §9) plugs in without touching the projector.
The axial/polar distinction lives in one visible place
(`rep_scale(trait, detR, l)`), though production never hits the det ≠ +1 case
for spin (§4).

Rejected: `Vector{AbstractSlot}` / Union eltypes in SALC storage; type-parameter
slots; packed-integer cleverness; per-species row layouts.

## 4. Group action and selection rules

- **ONE projection: the grey group, always.** There is no second group action
  and no `SectorConstraint` type.
- **Σl_spin-even is enforced at enumeration** (per-channel T-parity product
  = +1; disp factors T-even). Consequence: det(R)^{Σl_spin} = +1 identically,
  so the existing polar Wigner cache (`wignerD_real`, serial-precomputed,
  read-only) is exactly correct for BOTH channels — no det branch in the
  projector loop. The old implicit det(R)^{Σl} cancellation becomes an explicit
  theorem with gates (§11 o).
- **Spin-first canonical coupling order** (a corollary of the SPIN < DISP label
  order): all spin axes precede disp axes; left-coupling makes **L_S (total
  spin rank) a good quantum number** of the projection; it goes into `SALCKey`.
- Production keeps the coupled-tensor `_project_and_fold` engine (combined
  ordering × path × Mf space) + `_canonicalize_members`. The DisplacementBases
  prototype (polynomial-composition slot matrices, cycle-wise character
  counting + plethysm) is **demoted to the independent test oracle**, after
  fixing its review findings (§12). Its solid-harmonic evaluator
  (`_solid_harmonics_impl!`) ports to production as the displacement kernel.

## 5. Sectors and truncation (D-4)

- The truncation spec is a **sector table** (union of sectors, not a product
  grid), mapping 1:1 onto the theory note §3/§7. Public noun:
  `Sector(; spin, disp, soc = true, cutoff)`. Sugar resolved once to a dense
  canonical form (BasisSpec discipline); dense storage channel-keyed.
- **`soc::Bool` is per sector**, and it is a *truncation rule at enumeration*:
  `soc = false` ⇔ "enumerate only L_S = 0". Exactness is theorem-backed (the
  SOC-less basis is the L_S = 0 column subset of the one grey-group basis).
  The docstring states the theorem; `isotropy = true` gets a deprecation error
  pointing at `soc = false`.
- `soc` (truncation axis, defines model support) ≠ `sector_mask`/`frozen`
  (staging axis, defines fit stages). Never merged. The L_S = 0 mask correctly
  freezes the L_S = 0 columns of `soc = true` sectors too.
- Per-species knobs: `lmax = 0` (spin-inactive species), `pmax = 0` (clamped
  species) — ligands are `lmax = 0, pmax > 0`.
- `disp_scale` (u/d_NN scaling) is fixed in the basis and persisted (precedent:
  the (4π) rule). Constructor warns (not errors) on odd max total displacement
  degree; runtime boundedness guards live in MC (§8).

## 6. Data and fitting layer (D-7)

- **`TrainingDatum`** (channels optional; `SpinDatum(...)` survives as a
  convenience constructor). Fields are per-configuration observables only:
  energy, directions/magmoms, displacements, forces (Euclidean gradient),
  field/torques. **No strain field.** Future stress/occupations = versioned
  additive fields.
- **`DatumProvenance`** (constrained, torque_qualified, reference_id) is
  load-bearing; the `SLCEDataset` constructor pins ONE reference Crystal and
  asserts every datum's `reference_id` (the double-counting protocol as an
  invariant, not a docs warning). `torque_qualified` gates torque rows.
- Three-block co-fit (w_E, w_T, w_F); X_F is block-sparse by construction
  (p = 0 SALCs have zero force rows) — `_assemble_problem` scatters
  column-subset blocks, never materializing zeros.
- **ASR = exact linear equality constraints, null-space reparameterization
  inside `_assemble_problem`** (QR of Aᵀ once; solve in γ; β = Z·γ; group
  penalties on β unchanged). Never a penalty (a violated ASR is
  unbounded-below energy); never relative coordinates; never basis-level
  recombination.
- Hierarchical fit = `fit(...; frozen = ...)` offset + L_S-keyed
  `sector_mask`. The estimator layer (GroupAdaptiveRidge / GCV / select_fit /
  cross_validate) survives verbatim; `group_costs` gains a channel split.

## 7. Downstream contract (D-5)

- Successor term type: **`DecoratedTerm` / `decorated_terms(model)`** — per-slot
  (channel, k, l) + slot→site map + folded tensor.
- **Consumer scale rule: `(4π)^{n_spin_slots/2}`** (4π is a spin-sphere-measure
  artifact; disp rows carry none). Every consumer derives the scale from slot
  channels, NEVER from `length(atoms)` — the existing consumers do exactly the
  latter, which is why:
- **`multipole_terms(model)` THROWS on any model containing a p ≥ 1 sector**
  (trigger = sector presence in the spec, not coefficient values), with the
  two-hatch message naming `decorated_terms(model)` and
  `multipole_terms(restrict(model, :spin))`. On pure-spin models (any schema
  vintage) it returns today's result bit-identically — the frozen p = 0 view.
- `restrict(model, :spin)` = the exact clamped-ion sub-model (E at u = 0).
  **restrict ≠ refit**: its coefficients are not what a spin-only fit on
  relaxed structures would give — one boxed docs paragraph + comparison
  example is mandatory (§13 risk 5).
- `row_layout(model)` (per-channel row-offset table) is public contract.
- Deliverables tier (each with an exact-gate doctest): `bilinear_terms(model;
  displacements)`, `force_constants(model; spins, order)` → `ForceConstantSet`,
  `dynamical_matrix`, `exchange_strain_derivatives`, `magnetoelastic_constants`
  (cubic ⇒ `(; B1, B2)`), `magnon_phonon_vertices` (docstring says
  "adiabatic"), `to_sunny(model; clamp = true)`. Phonon-band tooling
  (phonopy writer) lives in SLCETools.

## 8. Monte Carlo implications

- Per-site basis rows in one flat matrix, stacked in channel-enum order
  (spin lm block ‖ disp (k,l,m) block ‖ future occ), offsets baked into
  programs at construction — gather loops textually unchanged.
- Per-(site, channel) activity replaces `site_active`; leave-one-out ΔE per
  (site, channel); contraction programs, dnPl cache, and fast paths generalize
  by slot count.
- Displacement move: per-site Gaussian with its own adaptive step, inside the
  existing color-parallel sweeps. **v0 keeps site-level coloring** (D-6);
  channel-split conflict graphs are a later refinement gated on an l044+p≤2
  measurement.
- **Pass scheduling is the bit-compat crux**: no disp pass on spin-only models
  ⇒ RNG consumption unchanged ⇒ converted-model bit-identity (§11 b).
- `set_coefficients!(H, coefs)` (fused coef-stream rewrite over coef-factored
  programs, ~1 sweep cost) — serves the strain outer move and doubles as the
  active-learning hot-swap hook.
- Strain: **outer-loop full-energy Metropolis move** over a K(ε) grid +
  elastic term (V/2)εᵀCε — never inside the sweep layer (§9).
- Observable signature breaks ONCE to a struct view: `MCView` (spins, disps,
  strain, energy) + per-channel `counts`; accessors so the next field is not
  another break; configurational-only specific heat documented.
- Runtime boundedness guards: min-eigenvalue monitor of Φ^(2)({ê}) +
  displacement-radius check, via observables.
- Checkpoint schema bump (disp, step_u, strain, per-channel activity,
  fingerprint over slot vectors), coordinated with SLCEDynamics; GPU: taller
  device feature matrix, polynomial evaluator ports bitwise more easily than
  Zlm did, Philox slot map gains disp draws (QTB precedent) — re-measure table
  memory before promising lattice sizes.

## 9. Strain and the K(ε) reference problem

A strained reference generically has a lower point group ⇒ naive per-grid-point
bases have different keys. The common-subgroup route is NOT viable for general
strain (the common subgroup over arbitrary ε is {E, (i)}; the basis explodes).
Settled resolution, split by physics:

- **(a) v0: K(ε) grids restricted to isotropic volume strain ε = ηI only** —
  full point group preserved, keys identical across grid points,
  `StrainedModels` + `model_at(sm, ε)` + MC `set_coefficients!` work as
  specced. Covers FeRh's ~1% volume jump.
- **(b) Symmetry-breaking strain: an explicit global strain channel** (rank-2
  symmetric, polar, translation-invariant, projected in the UNSTRAINED group)
  — keys ε-invariant by construction, ε-dependence polynomial in the basis
  functions. Needs the "global DoF" seat (a slot without a site) in the
  rep-provider interface (§3). Perturbative; the sibling of B₁/B₂.
- **(c) Specific low-dimensional strain paths** (c/a, tetragonal): a
  common-subgroup grid is fine (subgroup stays D₄h-class) but must be labeled
  path-specific.
- **(d) Coefficient renormalization** (Masuki–Nomoto–Arita–Tadano IFC
  renormalization / SCP structural optimization, PRB 106, 224104 (2022) +
  PRB 107, 134119 (2023)): because the displacement basis is polynomial, the
  shift u → u₀ + δu is an exact finite re-expansion (regular-solid-harmonic
  translation addition theorem — a linear map mixing (k, l) within one total
  degree). A planned `effective_model(model; u0)` utility reaches
  symmetry-broken distorted phases (soft-mode condensation) with NO
  common-subgroup grid; homogeneous strain is the affine pattern
  u_i = ε·R_i + δu_i, so a first strain response is already contained in the
  fitted polynomial up to its truncation. SCP-style thermal contraction
  (⟨uu⟩ → J_eff(T); ⟨Φ_spin⟩ → magnetic-state-dependent force constants) is
  the finite-T deliverable template. Caveat inherited: a distorted minimum
  must lie inside the truncated polynomial's validity region.

The K(ε) pipeline, explicitly: per grid point ε = ηI, build the strained
reference Crystal → basis (identical keys) → dataset (own reference_id) → fit
→ entry in `StrainedModels`. **The strain convention (metric, Voigt order,
shear factor) is pinned once and persisted before the first strained artifact
exists** (§13 risk 2).

## 10. Extension contract (D-9: occupation channel, ratified)

Spin–lattice ships first. The occupation (chemical-disorder) channel stays a
natural later phase iff these five things are shaped NOW (no OCC code today):

1. The rep-provider channel interface (§3) — OCC = orthonormal point functions,
   permutation-matrix symrep, T-even, SO(3)-trivial (never enters the CG tree).
2. Channel-tagged (channel, a, b) label triples in SALCKey, persist v5, and the
   term contract, with OCC enum RESERVED and SPIN < DISP < OCC fixed — adding
   OCC later changes no byte of existing models (v6 additive).
3. The distinct-(site, channel) slot invariant — the receptacle for the future
   indicator-gated ι_A(σ)·Z_l(ê) conditional-spin form.
4. Channel-enum row layout with a shipped offset table.
5. Species-keyed spec knobs (lmax/pmax per species; allowed-species lists join
   the same vocabulary).

Deferred with no back-propagation risk: the conditional-DOF measure convention,
semi-grand-canonical MC, the datum `occupations` field, eCE embedding, MIQP
ground-state constraints. Positioning vs CASM (nearest scope competitor) and
the technology radar (dipole long-range block for polar insulators — REQUIRED
before any oxide-magnet application; stress as a training target; Bayesian UQ;
one-hot OCC encoding): see `spin-lattice-ce-d0.md` §2b/§2c. Two verification
debts before any paper quotes the CASM comparison: torque/constrained-DFT
fitting path in CASM? noncollinear-magspin production applications?

## 11. Persistence

Schema **v5**: decor table + L_S in keys; factors + slot→site maps; spec
sectors + soc + disp_scale (+ strain convention when it lands). **Transparent
back-read of v4 in the loader, NO migrate-once tool** — the v4→v5 map is total
and value-preserving (`ls → SiteDecor(l,0,0)`, `L_S := Lf`,
`isotropy → soc = false`). Gate: v4-loaded models reproduce predictions AND MC
program arrays bit/byte-identically. Fingerprint recomputed on load, never
trusted. Drop v2/v3 branches only after confirming no files circulate.

## 12. Verification battery

Migration gates: (a) relabel bit-identity v4→v5 (evaluate/grad bit-identical,
MC program arrays byte-equal); (b) spin-only MC bit-identity (l044 8³ fixed
seeds, incremental-energy + config hash, multitask + GPU); (c) l044
predictions oracle via serialized full-digit fixtures (no co-loading);
(d) span equivalence old vs p = 0-restricted new basis; (i) u = 0 bitwise
degeneracy.

New-physics exact gates: (e) cubic single-site l=2×p: exactly 2 invariants
(B₁/B₂); (f) SOC-less pair+ligand p=1: dJ/dr + ligand term, NO DMI — **on a
mixed spec** (soc-false coupled sectors coexisting with a soc-true sector);
(g) count ≡ projector rank (prototype oracle) incl. permuting-pair AND
centrosymmetric mixed cluster (spin l=1 × disp l=1 ⇒ 0 invariants — the
kill-shot for the T-parity/det rule); (h) Sym² = 5+1, Sym³ = 7+3;
(n) sector-mask ≡ soc-false-basis equivalence, also on a mixed spec;
(o) inversion representation pin per channel (spin: +I ∀l; disp: (−1)^l I) +
mutation test (reinstating a global det^{Σl_all} rule must fail (f)/(g)).

Derivative/constraint gates: (j) forces ≡ −∂E/∂u finite differences (plus the
existing torque FD); (k) ASR residual ≤ 1e-13 + uniform-translation invariance;
(m) spin-less limit vs the JPSJ 95, 053601 force-constant code.

MC gates: (l) ΔE ≡ total-energy diff for spin/disp/strain moves; serial ≡
parallel bitwise; checkpoint resume bitwise; `set_coefficients!` round-trip
byte-identical; the (4π)^{n_spin_slots/2} pin includes a HAND-BUILT
mixed-channel term and a reduced-cell joint model (`reduce_cell` is a
forgotten scale consumer — §13 risk 3).

Prototype-oracle preconditions (from the DisplacementBases review): fix the
`_pair_swapped` exchange-sign blocker (per-slot swap flags consumed by both
`_slot_matrix` and `_cycle_character`; reject site_a == site_b; regression =
4-site/2-pair ⟨C4z⟩ case), add slot quantum-number validation, reject repeated
same-site slots, add the group-closure/integrality guard.

## 13. Residual risks and mitigations

1. K(ε) symmetry breaking — RESOLVED by §9 (volume-only grid / global strain
   channel / path-specific fallback / renormalization).
2. Strain convention unpinned — pin + persist before the first strained
   artifact (CASM alone ships three metrics; do not import ambiguity).
3. `reduce_cell` scale migration — covered by the extended (4π) gate (§12).
4. soc-vs-sector_mask drift — gates (f)/(n) MUST run on mixed specs.
5. restrict ≠ refit category error — boxed docs paragraph + comparison example.

## 14. Milestone plan

- **M0 — rename** (§2). Mechanical only; gate = all suites green locally + on
  kugui.
- **M1 — displacement evaluator + counting oracle.** Port
  `_solid_harmonics_impl!` (+ gradient) into the basis layer with unit gates
  (Zlm cross-check l ≤ 16, u = 0 exactness); port the prototype counting
  utility into test-support with the §12 blocker fixes; wire gate (g)
  infrastructure.
- **M2 — channel-first basis core.** SiteFactor/SiteDecor + rep-provider
  interface; SALCKey with L_S; sector-table spec + per-sector soc;
  `_project_and_fold` generalized to mixed channels; persist v5 + v4
  back-read. Gates (a)/(d)/(e)/(f)/(g)/(h)/(i)/(n)/(o).
- **M3 — data + fit.** TrainingDatum/DatumProvenance/dataset pinning; force
  design block + three-block co-fit; ASR null-space; hierarchical
  frozen/sector_mask. Gates (c)/(j)/(k)/(m).
- **M4 — MC.** DecoratedTerm contract + scale rule + multipole_terms throw +
  restrict; SLCEMonteCarlo joint programs, disp move, MCView,
  `set_coefficients!`; checkpoint bump; GPU port + re-measure. Gates (b)/(l)
  + the (4π) pin.
- **M5 — strain.** Pin the strain convention; `StrainedModels` + volume grid +
  elastic term + outer MC move; `exchange_strain_derivatives` /
  `magnetoelastic_constants`; `effective_model(model; u0)` (renormalization
  utility, §9d). Global strain channel (§9b) only after a felt need.
- Docs land with each milestone (migration.md carries the dropped-noun table
  from d0 Appendix D); the flagship tutorials (B₁/B₂ rederivation, dJ/dr
  no-DMI, force-constant Si vs JPSJ) are part of M2–M5 exit criteria, not an
  afterthought.
