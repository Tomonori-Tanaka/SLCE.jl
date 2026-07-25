# Spin–lattice CE — D0 design-discussion record

**Status**: CLOSED 2026-07-25 — all conflicts settled. The normative design is
**`spin-lattice-ce-design.md`** (the decision record); this file remains as the
discussion record (proposals, adversarial report, and rationale). Companion
feasibility note: `spin-lattice-ce.md`.

**SETTLED 2026-07-25 (user decision): D-1/D-2/D-3 — naming.**
Rename the whole family to the SLCE stem, core package = **`SLCE.jl`**
(method = package name; General-registry 4-char manual merge accepted, with
precedents CUDA/JACC/STAC and registration deferred to publication time);
`SCEMonteCarlo.jl → SLCEMonteCarlo.jl`, `SCETools.jl → SLCETools.jl`,
`SCESpinDynamics.jl → SLCEDynamics.jl` (drop "Spin"). Timing: the rename is
the FIRST step — an isolated mechanical commit series before any semantic
work, gated by full local suites + kugui on-cluster suites green; UUIDs kept,
kugui rsync paths + `@sce` → `@slce` env updated in the same change. The
one-rename principle applies: no further renames after this window.

This file preserves the raw material of the D0 (design-phase) discussion so it
can resume in a later session without re-deriving anything.

---

## 1. Where the three designers AGREE (candidate settled points)

All three proposals converged on the following — treat these as near-settled
unless the adversarial review overturns them:

1. **Joint expansion is the primary abstraction; spin-only = the p = 0 value**
   of the general types, not a sibling code path. The u = 0 exact degeneracy to
   clamped-ion spin SCE is both a design principle and a bitwise gate.
2. **Per-site/per-slot label design**: an isbits label carrying
   (channel, radial degree k, rank l), totally ordered with SPIN < DISP;
   constructor-validated (spin: k = 0, l ≥ 1; disp: degree 2k+l ≥ 1, l = 0
   legal). Slots decouple from sites (a site may carry a spin factor AND a
   displacement factor; single-site Z₂⊗[u⊗u] has 2 slots on 1 site). The
   distinct-member-sites invariant generalizes to **distinct (site, channel)
   slot pairs** — this is what preserves the MC's exact single-DOF ΔE.
3. **Σl_spin-even enforced at enumeration** (per-channel time-reversal parity;
   displacement factors T-even). With that, the polar Wigner cache
   (`wignerD_real`) is exactly correct for BOTH channels — no det branch in the
   projector. The centrosymmetric mixed-cluster test (spin l=1 × disp l=1 must
   yield 0 invariants under inversion) is the kill-shot gate.
4. **Spin-first canonical coupling order → L_S (total spin rank) is a good
   quantum number of the grey-group projection and goes into SALCKey.** The
   SOC-less mode (`L_S == 0`) becomes a column mask of the full basis — the
   hierarchical fit needs no second basis build. Legacy `isotropy = true` is
   exactly the spin-only special case (L_S = Lf).
5. **The DisplacementBases prototype is demoted to the independent test
   oracle** (polynomial-composition slot matrices, cycle-wise character
   counting + plethysm ≡ projector rank). Production keeps the coupled-tensor
   `_project_and_fold` engine + one Wigner cache. The solid-harmonic
   *evaluator* (`_solid_harmonics_impl!`) ports to production as the new
   displacement kernel (polynomial recurrences, no libm — GPU-friendly).
6. **Truncation spec = sector table** (union of sectors, not a product grid),
   mapping 1:1 onto theory-note §3/§7; sugar resolved once to a dense
   canonical form (existing BasisSpec discipline). Per-species `pmax = 0` is
   the clamped-species knob mirroring `lmax = 0` (ligands: spin-inactive,
   disp-active).
7. **Data layer**: one joint datum type (spins + displacements + energy +
   forces + torques; SpinDatum embeds/remains as convenience); reference-
   structure pinning enforced at the dataset constructor (double-counting
   protocol as an invariant, not a docs warning); torque-qualification flag
   gates torque rows; forces use the Euclidean (unprojected) gradient;
   `u/d_NN` scaling convention fixed in the basis and persisted (precedent:
   the (4π)^{N/2} rule).
8. **ASR = exact linear equality constraints, eliminated by null-space
   reparameterization inside `_assemble_problem`** — never a penalty
   (boundedness bug), never a relative-coordinate basis (destroys the flat
   per-site row layout and ΔE locality), never basis-level recombination
   (destroys key injectivity/groups/persist).
9. **Hierarchical fit = frozen-subset offset** (`fit(...; frozen = ...)`
   subtracts X[:, frozen]·β from targets), with `sector_mask` keyed on L_S.
   Estimator layer (GroupAdaptiveRidge/GCV/select_fit/CV) survives verbatim.
10. **Downstream contract**: the MultipoleTerm successor carries per-slot
    (channel, k, l) + slot→site map; **the consumer scale rule changes to
    (4π)^{n_spin_slots/2}** (4π is a spin-sphere-measure artifact; disp rows
    carry none). At p = 0 the value is unchanged. Joint per-site basis rows:
    spin lm block ‖ disp (k,l,m) block in one flat matrix (zrows
    generalization); contraction programs/fast paths survive structurally.
11. **Persistence: schema v5, transparent back-read of v4 in the loader, NO
    migrate-once tool.** The v4→v5 relabeling is total and value-preserving ⇒
    gate: v4-loaded models reproduce predictions (and MC program arrays)
    bit/byte-identically.
12. **MC**: per-(site,channel) activity replaces `site_active`; per-site
    displacement Metropolis move drops into the existing color-parallel
    sweeps; spin-only models must consume RNG identically to today (bit-compat
    gate on converted models); strain = outer-loop full-energy Metropolis move
    over a K(ε) grid + elastic term (never inside the sweep layer);
    boundedness guards (even-degree warn at spec, min-eigenvalue monitor of
    Φ^(2)({ê}) + displacement-radius check at runtime).
13. **Observable signature breaks ONCE to a struct view** (spins +
    displacements + strain + energy), with per-channel counts replacing the
    single `n` normalizer; document configurational-only specific heat.
14. **Verification battery** (union of the three proposals, deduplicated):
    (a) relabel bit-identity v4→v5; (b) spin-only MC bit-identity incl.
    multitask + GPU; (c) l044 predictions oracle via serialized fixtures (no
    co-loading); (d) span equivalence old vs p=0-restricted new basis;
    (e) B₁/B₂ = exactly 2; (f) SOC-less pair+ligand p=1: dJ/dr + ligand, no
    DMI; (g) count ≡ projector rank incl. permuting-pair + centrosymmetric
    mixed cluster; (h) Sym² = 5+1, Sym³ = 7+3; (i) u = 0 bitwise degeneracy;
    (j) forces ≡ −∂E/∂u FD; (k) ASR residual ≤ 1e-13 + uniform-translation
    invariance; (l) ΔE ≡ total-energy diff for spin/disp/strain moves;
    (m) spin-less limit vs the JPSJ 95, 053601 force-constant code;
    (n) sector-mask ≡ SpinIsotropic-basis equivalence; (o) inversion
    representation pin per channel + mutation test (wrong global det rule must
    fail).

## 2. Where they DISAGREE — ALL NOW SETTLED (table kept as the historical positions)

Resolution map: D-1/D-2/D-3 by user decision (header above); D-6 by
sequencing (row below); D-4/D-5/D-7/D-8 by the adversarial review — see
§2d for verdicts and Appendix D for the full report. D-9's extension
contract is ratified in §2b.

| # | Question | Positions |
|---|---|---|
| D-1 | **Package naming** | API designer: rename family to SLCE*, core = `SLCE.jl` (method = package name; "Fitting" undersells). Theory designer: rename family to SLCE*, core = `SLCEFitting.jl`; both agree SCESpinDynamics → `SLCEDynamics`. Systems designer: **keep all SCE names** ("never stack a mechanical rename on a semantic migration; when a gate fails mid-migration the diff must show physics, not renames"); rename—if ever—only after all J-gates are green, as an isolated mechanical commit series. |
| D-2 | **Rename timing** (if renaming) | API/theory: at the START of D0, before v5 lands, so every new artifact is born under the correct name. Systems: at the END (see D-1). A possible synthesis: decide the name now, apply it as either the first or the last isolated commit series — never interleaved. |
| D-3 | **Core package name length** | `SLCE.jl` (4 chars) vs `SLCEFitting.jl`. Registry facts (checked 2026-07-25, local General snapshot): no SLCE/Slice-stem collisions (closest: SliceMap, SliceSampling); 4-char names cannot AutoMerge (5-char minimum) but manual merges of 4-char acronyms have precedent (CUDA, Zarr, UNet, JACC, STAC, Gabs, PoGO) — 3-day wait + maintainer approval + justification ("acronym of the implemented method, introduced in <paper>"). Only the core is affected; all sibling names ≥ 5 chars AutoMerge fine. |
| D-4 | **Per-sector SOC granularity** | Theory designer: ONE global `constraint ∈ {GreyGroup, SpinIsotropic}` in v0 ("two physical settings; a per-sector table is configuration surface without a theory"). API + systems designers: `soc::Bool`/`spin_mode` PER SECTOR (the theory note's §7 truncation and the hierarchical-fit protocol mix SOC-less coupled sectors WITH a SOC single-site sector in one model — this seems to require per-sector granularity; the adversarial review should settle whether the theory designer's global constraint can express the Y1 truncation at all). |
| D-5 | **MultipoleTerm successor naming** | Theory designer: NEW name (`DecoratedTerm`), `multipole_terms` frozen as exact p=0 view that throws on joint models ("a contract whose invariants changed under an unchanged name is how downstream silently mis-scales"). Systems designer: widen under a new name too (`JointTerm`) but keep `multipole_terms` working. API designer: widen `multipole_terms` compatibly. Real decision: does the p=0 view keep working on joint models (restrict-then-view) or throw? |
| D-6 | **Adjacency/coloring granularity in MC** — SETTLED 2026-07-25: v0 uses site-level coloring (conservative, correct, today's code verbatim); per-(site,channel) conflict graphs + channel-split CSR are a later refinement gated on an l044+p≤2 measurement showing a real payoff. Both designers' positions are compatible under this sequencing. | — |
| D-7 | **Strain in the datum** | Systems designer: `strain::SMatrix` field on the datum now (zero default). Theory designer: NO strain field in v0 ("not a per-datum observable of the clamped-cell protocol; K(ε) grid = separate datasets at strained references, each with its own reference_id; dead fields breed convention bugs"). |
| D-8 | **Naming of user-facing nouns** | API designer's table (SCEPredictor → `SLCEModel`, TrainingDatum, Sector API with presets, deliverables tier: `force_constants` / `dynamical_matrix` / `exchange_strain_derivatives` / `magnetoelastic_constants` / `magnon_phonon_vertices` / `restrict(model, :spin)`) — no direct conflict, but only one designer worked this out; needs a pass against the others' type sketches. |
| D-9 | **Occupation channel (chemical disorder)** — added 2026-07-25, see §2b | Design constraint proposed as near-settled: make the channel interface **rep-provider generic** now (channel supplies dim, D(g), T-parity, eval kernel — not hardwired 2l+1/Wigner); implement the OCC channel itself in a later phase. Open sub-question: the conditional-DOF convention (indicator-gated spin factors ι_A(σ)·Z_l(ê) for species-dependent spin existence). |

## 2b. D-9: occupation channel (chemical disorder) — analysis, 2026-07-25

Motivation (user): extend the abstraction so alloy/occupation cluster expansion
is also expressible, and pin down the differentiation vs CASM.

**Fit with the settled channel abstraction — clean, one interface lift:**

- Occupation σ_i ∈ {0..M−1}: site basis = orthonormal point functions φ_n
  (Sanchez–Ducastelle–Gratias); n = 0 constant excluded (same shape as the
  spin l ≥ 1 rule). T-even. SO(3)-trivial (rank 0) ⇒ **occupation factors
  never enter the CG tree** — they multiply as scalars; L_S sector structure
  is untouched.
- Group action: ops don't rotate σ, but an op mapping a site whose allowed
  species list is reordered acts as a **permutation matrix on the
  point-function index** (finite-group symrep, not a Wigner matrix). This is
  the ONE design lift: the channel interface must be generalized from
  "rank l ⇒ 2l+1 dims + Δ^l(R) (± det factor)" to "channel provides
  (rep dimension, rep matrices D(g), T-parity, evaluation kernel, gradient
  kind)". `_project_and_fold` already consumes per-axis matrices — feeding a
  permutation matrix is structurally trivial. Cost of deciding this NOW: near
  zero (it is the same interface the theory designer sketched, with the
  provider abstracted). Cost of retrofitting later: another key/persist
  migration.
- MC: occupation flip = discrete single-site move; the exact leave-one-out ΔE
  machinery works verbatim (zrows gains M−1 occ rows per site). Semi-grand
  canonical (chemical-potential term) is the only new sampler concept.
- Fitting data: enumerated/SQS supercells; joint spin×occupation sampling is
  combinatorially expensive — honest caveat, mitigated by the active-learning
  route (ActiveSCE).
- **The genuine formal open point — conditional DOF existence**: spin exists
  only when a magnetic species occupies the site. Natural formulation:
  indicator-gated spin factors ι_A(σ_i)·Z_l(ê_i) (the vacancy trick of alloy
  CE); needs a convention decision (basis orthogonality on the joint measure,
  and what the dummy-spin measure is for non-magnetic occupants). This is the
  D-9 sub-question to settle before any OCC implementation.
- Strategic alignment: the Y3–4 grant target (Mn,Fe)₂(P,Si) is chemically
  disordered — OCC channel is the natural later phase, not scope creep.

**Extension contract (ratified 2026-07-25): implement spin–lattice first;
OCC stays natural iff these five things are shaped now (no OCC code today):**

1. Channel rep-provider interface: a channel supplies (rep dimension, D(g),
   T-parity, evaluation kernel, gradient kind); spin/disp implemented against
   it, never hardwiring Wigner/2l+1.
2. Channel-tagged label triples (channel, a, b) from day one in SALCKey,
   persist v5, and the MC term contract, with the OCC enum value RESERVED and
   the total order SPIN < DISP < OCC fixed — so adding OCC later changes no
   byte of existing spin-lattice models/fingerprints (v6 additive; old
   checkpoints stay valid).
3. The distinct-(site, channel) slot invariant (already agreed) — it is the
   receptacle for a same-site spin + occupation factor, i.e. the future
   indicator-gated ι_A(σ)·Z_l(ê) form.
4. Row-layout convention: per-site basis rows stacked in channel-enum order
   (spin ‖ disp ‖ future occ) with a shipped offset table.
5. Species-keyed spec knobs (lmax/pmax per species — already the design), so
   per-site allowed-species lists join the same vocabulary.

**Deferred with no back-propagation risk:** the conditional-DOF measure/
orthogonality convention (spin–lattice models are the σ-frozen slice);
semi-grand-canonical MC (chemical-potential term); the datum `occupations`
field (versioned addition); eCE embedding / MIQP ground-state constraints
(radar items, all sit on top of this contract).

**CASM facts (verified from official docs/repos, 2026-07-25):**

- CASM v2's `casm-bset` states its scope as "generating coupled cluster
  expansion Hamiltonians of occupation, strain, displacement, and magnetic
  spin DoF". Magnetic spin comes in 6 flavors (Cmagspin/NCmagspin/SOmagspin
  + unit-constrained variants); displacement `disp` [d_x,d_y,d_z]; strain in
  3 metrics (GL/Hencky/EA, 6-component). Crystal basis functions =
  polynomials of site basis functions; occupation site bases =
  Chebychev/occupation-indicator; C++ Clexulator code generation; mature
  occupation MC (semi-grand, kinetic).
- ⇒ **At the scope-claim level CASM is the nearest competitor of the unified
  expansion** — any "no existing symmetry-adapted coupled-DOF CE" claim is
  falsified. The grant note §11 has been updated (CASM + MCE rows added,
  2026-07-25).
- Differentiation (same axis as vs PASP, now for the coupled case):
  (1) CASM symmetrizes generic component polynomials numerically under the
  crystal factor group — no per-site angular-momentum ranks, no CG tree, no
  L_S sectors, no Σl selection rules as structure ⇒ no closed-form recovery
  of known models, no character-formula pre-counting, no physics-labeled
  coefficients (B₁/B₂/DMI/Φ). (2) Component polynomials on |e| = 1 are
  overcomplete (r² redundancy) and non-orthogonal — tesseral harmonics are
  the canonical reduction (Drautz–Fähnle completeness is our spin-side
  foundation, production-validated). (3) No constrained-noncollinear-DFT
  torque/force co-fit protocol found (TO VERIFY before quoting in a paper).
  (4) No production noncollinear-magspin application found (TO VERIFY).
  (5) Acknowledge CASM strengths: occupation thermodynamics maturity,
  Clexulator codegen, community.
- Also added to §11: MCE (Lavrentiev–Dudarev Fe-Cr magnetic cluster
  expansion, PRB 81, 184202 (2010)) — occupation × vector spin pioneer,
  bilinear-Heisenberg-limited, no symmetry-adapted high-rank machinery.

## 2d. Adversarial-review verdicts — D-4/D-5/D-7/D-8 SETTLED (2026-07-25)

Full report: Appendix D. Compact verdicts:

- **D-4**: per-sector `soc::Bool`, reframed — **ONE projection (grey group,
  always); `soc = false` = "enumerate only L_S = 0" as a truncation rule at
  enumeration** (justified by the theory designer's own reduction theorem).
  Global `SectorConstraint`/`GreyGroup`/`SpinIsotropic` types dropped. The
  global variant fails its own expressibility test (Y1 mixes :nosoc coupled
  sectors WITH the :soc B₁/B₂ sector; gate (f) is unreachable either way).
  `soc` (truncation axis) ≠ `sector_mask`/`frozen` (staging axis) — never
  merged. Gates (f)/(n) MUST run on a mixed spec.
- **D-5**: `multipole_terms(model)` **throws** on any model with a p ≥ 1
  sector (trigger = sector presence in the spec, not coefficient values),
  with a two-hatch error message naming `decorated_terms(model)` and
  `multipole_terms(restrict(model, :spin))`; bit-identical on pure-spin
  models. Successor named **`DecoratedTerm`/`decorated_terms`** (not
  JointTerm — the D-8 table kills the "Joint*" prefix everywhere). Code
  reality verified: both existing consumers derive the (4π) scale from
  `length(atoms)` — widening in place would silently mis-scale.
- **D-7**: **no strain field on `TrainingDatum`**. Strain lives at the
  reference level (each K(ε) grid point = its own strained reference Crystal
  + reference_id, pinned by the dataset ctor) and the model level
  (`StrainedModels`). Datum fields = per-configuration observables (energy,
  forces, torques, future stress/occupations as versioned additions);
  protocol parameters live on the reference. `DatumProvenance`
  (constrained, torque_qualified, reference_id) is load-bearing.
- **D-8**: reconciled noun table in Appendix D (5 tiers). Key deltas vs the
  API designer's version: `multipole_terms` freezes (D-5); `DatumProvenance`
  added; `TrainingDatum` (not JointDatum) with no strain field; `Sector`'s
  resolved dense form channel-keyed (OCC lands as a new key); `row_layout`
  promoted to the public contract; `MCView` adopted for the MC observable
  break. Dropped nouns (→ migration.md): JointDatum, JointBasisSpec,
  SectorSpec, SectorRule, SectorConstraint/GreyGroup/SpinIsotropic,
  SiteLabel, JointTerm, spin_mode, isotropy.

**Residual risks carried into the decision record (Appendix D §Residual):**
(1) K(ε) grid symmetry breaking — strained references have lower point
groups ⇒ different SALC keys per grid point; a common-basis convention must
be decided BEFORE `StrainedModels` is implemented (real hole, no designer
addressed it).
**Resolution direction (settled in discussion with the user, 2026-07-25):**
the common-subgroup route is NOT viable for general strain — the common
subgroup over arbitrary ε is {E, (i)} (only ops commuting with every
symmetric tensor), so the basis explodes to near-unsymmetrized size. Split
by physics instead: **(a) v0: K(ε) grids restricted to isotropic volume
strain ε = ηI only** — preserves the full point group, keys identical
across grid points, `StrainedModels`/`set_coefficients!` work as specced
(covers FeRh's ~1% volume jump; matches theory note §8.1 "初手は等方体積
のみ外部化"). **(b) Symmetry-breaking strain components: an explicit
global strain channel** (rank-2 symmetric, polar, translation-invariant
slot, projected in the UNSTRAINED group) — keys ε-invariant by
construction, ε-dependence explicit/polynomial in the basis functions,
coefficients constant; the natural sibling of the B₁/B₂ magnetoelastic
sector, and the "homogeneous-strain slot" gap from the prototype review.
Design impact: the channel abstraction needs a "global DoF" seat (a slot
without a site) — within the rep-provider generalization of the extension
contract. **(c) Fallback for specific low-dimensional strain paths**
(hexagonal c/a, tetragonal path): common-subgroup grid is fine there
(subgroup stays D_4h-class) but must be labeled path-specific. Hybrid
"volume = grid (non-perturbative), symmetry-breaking = explicit channel
(perturbative)" takes both. **(d) Coefficient renormalization (added
2026-07-25, user-supplied reference):** the Masuki–Nomoto–Arita–Tadano IFC
renormalization / SCP structural-optimization scheme (arXiv:2205.08789 =
PRB 106, 224104 (2022), BaTiO₃; arXiv:2302.04537 = PRB 107, 134119 (2023),
GaN/ZnO) demonstrates the same architecture we chose — ONE high-symmetry
reference with a high-order polynomial (Taylor) model, and *any* displaced
or strained structure reached by exact re-expansion u → u₀ + δu
(contracting higher-order coefficients with the frozen u₀), never by
re-projecting in the broken-symmetry subgroup. Our displacement basis is
polynomial (homogeneous solid harmonics), so the same shift is exact and
finite here: the regular-solid-harmonic translation addition theorem gives
R_{lm}(u₀+δu) as a finite binomial–CG sum, i.e. the shift is a computable
linear map on the basis (mixing (k,l) labels within one total polynomial
degree). Consequences: (i) symmetry-broken *distorted phases* (soft-mode
condensation à la tetragonal BaTiO₃) need NO common-subgroup grid — an
`effective_model(model; u0)` utility renormalizes to a lower-symmetry
effective model with the reference-group basis; (ii) homogeneous strain is
itself an affine displacement pattern u_i = ε·R_i + δu_i inside each
cluster, so the displacement polynomial already contains a strain response
up to its truncation order/cutoff — channel (b) stays for
beyond-truncation accuracy, but a free first estimate of strain couplings
falls out of the fitted model; (iii) the truncation-radius caveat
transfers too (BaTiO₃'s double well needed high Taylor order — a distorted
minimum must lie inside the fitted polynomial's validity region).
(2) Strain convention (metric, Voigt order, shear factor)
must be pinned once and persisted before the first strained artifact.
(3) `reduce_cell` is a forgotten (4π)-scale consumer — the scale gate must
include a hand-built mixed-channel term and a reduced-cell joint model.
(4) mixed-spec gate coverage (see D-4). (5) `restrict(model, :spin)` ≠
"a spin-only refit" — boxed docs paragraph + comparison example required.

## 2c. Technology radar — CE-framework trends (surveyed 2026-07-25)

Adoptability assessment of recent cluster-expansion-adjacent techniques for the
SLCE roadmap (user-initiated survey; seed = the TCE paper below).

**Design-relevant NOW (feed into D0):**
- **Long-range dipole block for the displacement channel in ionic crystals** —
  a genuine gap found by this survey: harmonic force constants of polar
  insulators need the nonanalytic dipole–dipole separation (Born effective
  charges + Ewald, the alamode/phonopy practice); the short-range CE fits the
  remainder. Irrelevant for metallic targets (FeRh, Mn₃Sn, CoMnSi) but
  physically mandatory before any oxide-magnet application. Add as a scope
  note / later-phase block in the design.
- **Stress as a training target** — one DFT cell stress (6 components/calc)
  calibrates the strain sector / K(ε) + elastic tensor; natural future
  `JointDatum` field alongside forces (precedent: force-constant codes).
- **IFC renormalization / SCP finite-T structural optimization**
  (Masuki–Nomoto–Arita–Tadano, arXiv:2205.08789 + arXiv:2302.04537;
  user-supplied 2026-07-25) — see residual-risk-1 option (d) for the
  displaced/strained-structure re-expansion. Beyond that, the SCP loop is
  the finite-temperature *deliverable* template: contracting the coupled
  spin-lattice model with thermal lattice fluctuations ⟨uu⟩ yields
  renormalized spin coefficients J_eff(T) (phonon-renormalized exchange),
  and contracting with spin correlations ⟨Φ_spin⟩ yields
  magnetic-state-dependent effective force constants — both are exact
  polynomial contractions on our basis, no new DFT. These two papers are
  the reference method (and validation target: BaTiO₃-class soft-mode
  physics) for the grant's magnon–phonon / finite-T free-energy story.
- **TCE one-hot occupation encoding** (Jeffries, Sun, Martinez, Comput.
  Mater. Sci. 262 (2026) 114338; tce-lib): correlation functions as sparse
  contractions of topology tensors × one-hot configuration tensors, O(N^0.05)
  swap ΔE, GPU/TPU-friendly. Performance claims are already matched by
  SCEMonteCarlo's sparse contraction programs (spin channel); the adoptable
  piece is the one-hot encoding as the OCC-channel zrows form (confirms D-9's
  point-function design — they are linear maps of one-hot). Their
  no-enumeration philosophy (all neighbor tuples, no symmetry projection) is
  the opposite of our interpretability core — not adopted.
- **MLFF-assisted data generation** (e.g. JCTC 2024 MLFF-aided CE): cheap
  (magnetic) MLIP for configuration generation / prior construction —
  validates the existing ActiveSCE plan (grant §4.1); concrete borrow:
  MLFF-relaxed displacement snapshots as joint-channel training data.

**Mid-term (Y2–3):**
- **Bayesian UQ with thermodynamically-informed priors** (arXiv:2309.12255 +
  arXiv:2509.07326, Van der Ven school): posterior ensembles constrained to
  preserve ground states / phase-diagram topology, propagated through MC to
  phase-boundary error bars — the concrete machinery for the FeRh T*
  sensitivity band (theory note §8.7).
- **Ground-state-preserving constrained fits** (MIQP tradition in alloy CE):
  the *magnetic* analogue — inequality constraints keeping the fitted model's
  known magnetic ground state — would be a novelty transfer into our
  estimator layer.
- **eCE chemical embedding** (npj Comput. Mater. 2025, arXiv:2409.06071):
  learned low-dim embedding of occupation point functions for many-component
  systems; a linear basis rotation, framework-compatible; relevant once the
  OCC channel meets HEA-class component counts.

**Far-term / watch-only:** QUBO/quantum-annealer ground-state search over the
multilinear binary form (OCC); nonlinear CE readouts (PRR 2025,
arXiv:2506.18695 — conflicts with the linear interpretability core);
equivariant-GNN relaxed-property surrogates (arXiv:2505.08121 — competitor
watch, or data generator); transfer-learning CE. ("Tensor-network loop
cluster expansions" in quantum many-body are an unrelated namesake.)

**Already have (≥ parity):** sparse contraction programs + GPU (= TCE's
performance claim, for spin); cost-weighted group selection (same goal as the
coherency/redundancy sparse-CE line, arXiv:2109.06905, but tied directly to
MC cost).

## 3. Registry check facts (D-3 input, verified locally 2026-07-25)

- General snapshot grep: no `SLCE`, no bare `Slice`/`SLICE`; nearest names
  SliceMap / SliceSampling / SlicedDistributions / SlicedWasserstein — all far
  by edit distance. `SCE*` stem: only ScenTrees/ScenarioTheory/SceneGraphs.
- AutoMerge requires ≥ 5 chars; existing 4-char packages went through manual
  merge. All family names except a hypothetical `SLCE.jl` AutoMerge cleanly.
- Registration requires a public repo at registration time — the decision is
  needed now only for naming, not for registering.

## 4. Prototype (Magesty.jl-coupledce DisplacementBases) review findings

Numerical-correctness review of the validated prototype (655/655 tests green),
2026-07-25. These findings matter for the PORT (the counting utility moves to
the SCE* test-oracle tier): 1 blocker / 3 major / 5 minor.

**Blocker** — `_pair_swapped` decides the CG exchange sign assuming a pair
slot maps ONTO ITSELF. With ≥ 2 SpinPairSlots, an op mapping pair A onto pair
B in reversed site order passes `_slot_matches` but drops the
`(−1)^{l_a+l_b−L_S}` factor. The wrong D(g) is still a homomorphism (1-dim
character twist) so the D1·D2 = D12 and eigenvalue-0/1 guards never fire, and
`count_invariants` errors out ("cycle length > 1") exactly on those inputs —
the two paths can never visibly disagree. Repro: 4 sites, pairs (1,2)/(3,4),
⟨C4z⟩, L_S = 1: projector rank 2, true count 3. Degenerate branch:
`SpinPairSlot(1,1,l,l,L)` gives D(e) = −I, count = −3. Fix: return per-slot
swap flags from `_slot_permutation`, consume in both `_slot_matrix` and
`_cycle_character` (cycle trace picks up the product of the cycle's signs);
reject site_a == site_b; regression-test the 4-site/2-pair case.
NOTE: in the SCE* port the SpinPairSlot layer is replaced by the existing
`_project_and_fold` recoupling, so the blocker matters only if the counting
utility is ported as-is.

**Major**: (1) no slot quantum-number validation (triangle rule, negative
l/k) — invalid specs return plausible numbers; (2) repeated same-site slots
silently over-count (`[SpinSlot(1,1), SpinSlot(1,1)]` → 2, truth: Sym² only);
no Sym^p slot type exists on the spin side; (3) `product_ops` docstring's
"commuting groups" claim is false as stated (lattice × spin-only ops don't
commute; closure requires the spin subgroup normalized by the lattice proper
parts) and `count_invariants` has no group-validity/integrality guard
(non-group input returned 0.5 silently).

**Minor**: integrality assert missing; τ well-definedness currently rests on
Major-2's absence; `B \ C` conditioning grows ~×5 per p+2 (fine to p ≈ 14);
`_double_factorial_odd` overflows ~n ≳ 150 (academic).

**Confirmed clean** (independent re-derivation): solid-harmonic recurrence +
normalization (matches Zₗₘ to 1e-12, l ≤ 16; CS phases cancel; u = 0 exact);
axial det(R)·R matches SALCBases convention; cycle-wise trace formula incl.
polar improper rewrite det^L χ(det·M) and plethysm h_p (valid for improper
orthogonal M since tr M^{-i} = tr M^i); CG exchange sign verified against
Magesty's real coupled tensors; representation bookkeeping is a homomorphism;
O_h B₁/B₂ and C4v counts verified analytically; octahedral spin average
suffices for SO(3) enforcement at l ≤ 3 (first nontrivial O-invariant at
l = 4); Δl no-transpose convention confirmed.

**Scope gaps** (not defects): time reversal absent (all current tests happen
to have Σl_spin even — `[SpinSlot(1,1), DispSlot(1,0,1)]` returns the T-odd
(ê×u)_z invariant); translation folding absent; SpinPairSlot never
materializes a CG tensor (Lseq/rephase untested); no homogeneous-strain slot;
no spin-side Sym^p.

## 5. Next steps when the discussion resumes

1. Settle D-1/D-2/D-3 (naming + timing) with the user — the registry facts
   above are the missing input that triggered the pause.
2. Resolve D-4, D-5, D-7 (likely via a short adversarial-review pass over the
   three proposals; D-4 looks decidable on the Y1-truncation expressibility
   argument alone). Ratify the D-9 design constraint (rep-provider-generic
   channel interface) and park the conditional-DOF convention as a later-phase
   decision.
2b. Verification debts from the CASM check (before any paper/grant text quotes
   them): whether CASM has any torque/constrained-DFT fitting path, and
   whether any noncollinear-magspin production application exists.
3. Write the settled design as `spin-lattice-ce-design.md` (the decision
   record), then the milestone plan (M1 = evaluator + counting oracle port,
   including the prototype blocker fix).

---

---

# Appendix A — Proposal 1: theory-first / type-system designer (verbatim)

Premise I hold throughout: the master object of the theory (note §2) is a **decorated cluster** — a set of sites, each carrying an irrep label of O(3) × {1,T} plus a radial degree — CG-coupled and Reynolds-projected onto the identity of the grey magnetic point group. Every type below is a direct transcription of that object. Spin-only SCE must be the p = 0 *value* of the general types, not a sibling code path. Where the mathematics gives a selection rule (T-parity, L_S = 0, axial vs polar), the type system should make the violating state unconstructable or at least unenumerable, so correctness never again rests on an implicit cancellation like the current `det(R)^{Σl}` one.

## A. Package naming

**Proposal: rename the whole ecosystem prefix `SCE` → `SLCE` (spin–lattice cluster expansion), all four packages, in one atomic operation.**

- `SCEFitting.jl` → **`SLCEFitting.jl`**
- `SCEMonteCarlo.jl` → **`SLCEMonteCarlo.jl`**
- `SCETools.jl` → **`SLCETools.jl`**
- `SCESpinDynamics.jl` → **`SLCEDynamics.jl`** (drop "Spin": the dynamics package is the one whose scope *most* visibly becomes joint — LLG plus a lattice Verlet/Langevin channel is already on its roadmap)

Rationale. The name is the outermost type annotation. A package named "spin cluster expansion" whose primary abstraction is a joint spin+displacement expansion misstates its own theory, and every future reader (including the grant reviewers this theory note feeds) will parse "SCE" as spin-only. The stated ground rules — zero external users, breaking changes allowed, prediction-level oracle — mean the rename cost is one afternoon: `git mv` the four directories, sed the module names, keep the existing UUIDs (unregistered path-dev packages; UUID continuity keeps Manifests resolvable), update the path-dev entries, the kugui rsync target paths and the `@sce` shared env, and the memory notes. The v4→v5 persistence break (H) is happening anyway, so file-format continuity is not an argument for keeping the name.

Consistency argument for renaming the siblings *together*: the siblings' public seams (`DecoratedTerm`, checkpoint schemas, Observable signature) all break in this redesign regardless (G, I). Renaming them later would mean paying the mechanical cost twice and living with a mixed-prefix ecosystem in between.

**What I would NOT do:**
- *Keep `SCEFitting` and reinterpret the acronym* ("symmetrized cluster expansion"): a backronym is a lie told slowly; every docstring would still say "spin".
- *Rename only the fitting package*: a split-prefix ecosystem (`SLCEFitting` + `SCEMonteCarlo`) is worse than either uniform choice.
- *`CCE` (coupled CE)* — collides head-on with coupled-cluster and with cluster-correlation-expansion in the decoherence literature. *`MCE` (modified CE)* — collides with magnetocaloric effect, an actual application domain of this code. `SLCE` has no collision I can find and reads literally.

Fallback if the rename is vetoed: keep repository names, rename only the user-facing story (README + docs say "spin–lattice CE; the S in SCE is historical"). I consider this strictly worse but it is the cheap option.

## B. Channel abstraction

The mathematical content of a per-site factor is: *(i)* which O(3) representation it carries (axial rank-l vs polar rank-l), *(ii)* its T-parity, *(iii)* its radial degree, *(iv)* how it is evaluated and differentiated. My design splits this into **two layers with distinct jobs**:

**Layer 1 — the storable, orderable, isbits label** (keys, persistence, hot loops):

```julia
@enum Channel::UInt8 SPIN = 0x01 DISP = 0x02

struct SiteFactor
    channel::Channel
    k::Int8            # radial degree; identically 0 for SPIN
    l::Int8            # harmonic rank
    function SiteFactor(c::Channel, k::Integer, l::Integer)
        if c == SPIN
            k == 0 || throw(ArgumentError("spin factor carries no radial degree"))
            l >= 1 || throw(ArgumentError("spin rank must be ≥ 1 (l = 0 is the constant)"))
        else # DISP
            (k >= 0 && l >= 0) || throw(ArgumentError("need k ≥ 0, l ≥ 0"))
            2k + l >= 1 || throw(ArgumentError("disp factor must have degree ≥ 1"))
        end
        return new(c, Int8(k), Int8(l))
    end
end
spinf(l::Integer)             = SiteFactor(SPIN, 0, l)
dispf(k::Integer, l::Integer) = SiteFactor(DISP, k, l)

_ftuple(f::SiteFactor) = (UInt8(f.channel), f.k, f.l)
Base.isless(a::SiteFactor, b::SiteFactor) = _ftuple(a) < _ftuple(b)
Base.hash(f::SiteFactor, h::UInt) = hash(_ftuple(f), h)

site_dim(f::SiteFactor) = 2Int(f.l) + 1
degree(f::SiteFactor)   = f.channel == DISP ? 2Int(f.k) + Int(f.l) : 0
time_parity(f::SiteFactor) = (f.channel == SPIN && isodd(f.l)) ? -1 : +1
```

Illegal states are unrepresentable at this layer: a spin factor with a radial degree, a rank-0 spin factor, and the degree-0 displacement constant cannot be constructed. `l = 0, k ≥ 1` displacement factors (the `|u|^{2k}` trace channels — the radial dependence) *are* constructable, killing the current `l ≥ 1` rule structurally rather than by a channel-dependent `if`. The total order puts SPIN before DISP, which is what makes the spin-first coupling canonical (C) — the *ordering of the label type* is the canonicalization, not a convention comment.

Because a cluster site may carry a spin factor *and* a displacement factor simultaneously (a magnetic atom that also moves), the per-**site** label is a second isbits type:

```julia
struct SiteDecor
    sl::Int8   # 0 = no spin factor
    dk::Int8
    dl::Int8   # (0,0) = no displacement factor
end
```

`SiteDecor` is what `SALCKey` carries (a sorted `Vector{SiteDecor}` replaces `ls::Vector{Int}`); `SiteFactor` is what tensor axes carry. The invariant **"at most one factor per (site, channel)"** is the joint generalization of the current distinct-member-sites rule, and it is exactly what preserves SCEMonteCarlo's exact single-DOF ΔE: the leave-one-out coefficient vector for the spin DOF at site *s* may legally depend on `u_s`, just never on `e_s`.

**Layer 2 — the trait layer** (projection, enumeration; cold code):

```julia
struct SpinChannel end
struct DispChannel end
channeltrait(f::SiteFactor) = f.channel == SPIN ? SpinChannel() : DispChannel()

is_axial(::SpinChannel) = true
is_axial(::DispChannel) = false

# The single place the axial/polar distinction touches a representation matrix:
#   D_spin(g) = det(R_g)^l · Δ^l(R_g)     (axial)
#   D_disp(g) = Δ^l(R_g)                  (polar)
rep_scale(::SpinChannel, detR::Float64, l::Int) = detR^l
rep_scale(::DispChannel, detR::Float64, l::Int) = 1.0
```

and the evaluation/gradient kernels dispatch on the trait (SpinChannel → Zlm + tangent-projected grad; DispChannel → |u|^{2k}·R_lm + Euclidean grad).

**Hot-loop type stability.** The stored SALC term keeps the existing rank-`D` function-barrier design; per axis the kernel does one integer branch on `f.channel`. `SALCTerm` becomes:

```julia
struct SALCTerm
    factors::Vector{SiteFactor}  # one per tensor axis, canonical order (spin axes first)
    sites::Vector{Int}           # axis → member-site slot
    folded::Array{Float64}
end
```

At p = 0 this degenerates *by value* to today's term.

**What I would NOT do:** store `Vector{AbstractSlot}` in SALCs (boxing/dispatch in design loops; prototype slot types are for the projection layer and the oracle); `Union{SpinSlot,DispSlot}` eltypes (GPU wants integer streams); clever integer packing (negative l encodes DISP — silent-sign-convention disease); type-parameter slots `SpinSlot{l}` (compile-time explosion for zero benefit).

## C. Coupling tree & sector constraints

**Canonical coupling order is a theorem of the label ordering**: axes sorted by (canonical site position, SiteFactor), SPIN < DISP ⇒ all spin axes precede displacement axes. Left-coupling ⇒ `L_S = Lseq[n_spin − 1]` (n_spin ≥ 2), `= l₁` (n=1), `= 0` (n=0).

```julia
abstract type SectorConstraint end
struct GreyGroup     <: SectorConstraint end   # SOC
struct SpinIsotropic <: SectorConstraint end   # SOC-less: L_S == 0

admits(::GreyGroup,     L_S::Int) = true
admits(::SpinIsotropic, L_S::Int) = L_S == 0
```

`SpinIsotropic` under the same grey-group projector reduces exactly to "spin factor is a spin-space scalar; project the lattice factor with the plain space group". At p = 0, L_S = Lf, so old `isotropy = true` is literally the special case.

**L_S is a good quantum number and belongs in the key** (projector block-diagonal over L_S; no op maps a spin axis onto a displacement axis):

```julia
struct SALCKey
    body::Int
    orbit_id::Int
    decor::Vector{SiteDecor}
    L_S::Int                   # NEW, gauge-invariant
    Lf::Int
    block::Int
end
```

Payoffs: SOC-less basis = column mask; enumeration filter and mask must agree (free structural gate); selection can weight by energy-scale sector via a persisted label.

**Time reversal per-channel and structural**: `prod(time_parity, factors) == +1` at enumeration — Σl over spin factors even, displacement factors unconstrained.

I deliberately do NOT allow per-sector constraints in v0 — physically there are exactly two settings. [Contested: see D-4.]

**What I would NOT do:** user-configurable coupling tree (internal gauge); `PinnedSpinRank(L)` general L (no physical use); keeping `isotropy::Bool` as sugar (deprecation error pointing at the constraint).

## D. Group action declaration

**Single source of truth: the channel trait, consuming ONE Wigner cache.** `wignerD_real(l, R)` is the polar O(3) polynomial rep — correct for improper R, exactly what a displacement block needs. Axial = scalar multiple det(R)^l. Projection applies `Di = _wig(wcache, l, g)` then `sc = rep_scale(trait, dets[g], l)` — the det factor is cheap and *visible*; `SymOp.is_proper` stops being dead code.

**Keep the coupled-tensor contraction engine; do NOT switch to the prototype's polynomial-composition route in production** (it lacks ordering×path×Mf recoupling, translation folding, deterministic gauge, thread discipline; B \ C per slot per op is slow and BLAS-dependent). The prototype's value: an **independent derivation of the same numbers** — promote `count_invariants` + `invariant_projector` to a test-support module and gate the production count against it. Plethysm disappears from production by construction (enumerating (k,l) labels IS the Sym^p restriction).

**The det(R)^{Σl_spin} test — three teeth:** (1) inversion representation pin per channel (spin: +I for every l; disp: (−1)^l I) — no cancellation possible; (2) mixed-cluster invariance on a centrosymmetric cell (spin (1,1) ⊗ disp l=1: Σl_all odd) asserting Φ(g·(e,u)) = Φ(e,u) for improper g; (3) mutation test — reinstating the wrong global det^{Σl_all} rule must fail gates 1–2.

**What I would NOT do:** a second Wigner cache keyed by channel (hides the det factor in cache construction); det(R)·R baked into a per-op "spin rotation" for production (fine for the oracle).

## E. Truncation spec (BasisSpec v2)

Dense caps ∩ explicit sector whitelist; sugar resolved once:

```julia
struct SectorRule
    spin_ls::Vector{Int}    # sorted spin-rank multiset; [] = spin-less; [-1] = :any
    psum::Int               # max Σ degree(2k+l) over disp factors
end

struct BasisSpec
    nbody::Int
    lmax::Vector{Int}                # per-species spin cap (0 = spin-inactive)
    lsum::Vector{Int}                # per-body Σ l_spin cap
    pmax::Vector{Int}                # per-species per-site disp-degree cap (0 = clamped)
    psum::Vector{Int}                # per-body Σ disp-degree cap
    sectors::Vector{SectorRule}
    constraint::SectorConstraint
    cutoff::Vector{Matrix{Float64}}
    species_labels::Vector{String}
end
```

Y1 truncation expressible verbatim (5 sector rules). `pmax = 0` = clamped species; ligands `lmax = 0, pmax > 0` = per-channel activity for the MC. Validator warns on odd max total displacement degree; `disp_scale::Float64` documented and persisted.

**What I would NOT do:** predicate-DSL closures in the spec (don't persist/hash/print); per-sector cutoffs in v0; auto-injecting even stabilizer terms (spec-is-canonical).

## F. Data layer

```julia
struct JointDatum <: AbstractTrainingDatum
    energy::Float64
    directions::Matrix{Float64}
    magmoms::Vector{Float64}
    displacements::Matrix{Float64}   # Å, from the reference structure
    forces::Matrix{Float64}          # eV/Å (empty = none)
    field::Matrix{Float64}
    torques::Matrix{Float64}
    provenance::DatumProvenance      # constrained, torque_qualified, reference_id
end
```

- **No strain field in v0** [contested: D-7]. K(ε) = separate datasets at strained references.
- Clamped-ion discipline = constructor assertion on reference_id.
- `torque_qualified` gates torque rows (penalty-constraint O(1/λ) systematic).
- Three-block co-fit: X_F sibling of _design_torque with Euclidean gradient; weights (w_E, w_T, w_F).
- Hierarchical fit = frozen-offset primitive + `sector_mask(basis; L_S = 0)`.
- **ASR via null-space reparameterization**: `asr_constraints(basis) -> C` (closed-form: translation generator on explicit homogeneous polynomials), `fit(...; constraints = C)` solves in `jphi = N θ`. Estimator-agnostic, exact, keeps centered-X contract.

**What I would NOT do:** relative-coordinate ASR (destroys per-site factor structure/MC contract); per-estimator Lagrange terms; a dead `strain` field.

## G. Downstream contract

```julia
struct DecoratedTerm
    coef::Float64
    body::Int
    atoms::Vector{Int}
    shifts::Vector{SVector{3,Int}}
    factors::Vector{SiteFactor}
    sites::Vector{Int}
    folded::Array{Float64}
end
decorated_terms(model)::Vector{DecoratedTerm}
```

Contract changes stated exactly: (1) scale rule (4π)^{nspin/2} — flagged in both packages' linked-sites docs simultaneously; (2) joint row space with shipped `row_layout(model)` offset table; (3) distinct (site, channel) invariant; (4) per-channel activity; (5) `multipole_terms` remains as the exact p = 0 view (throws on joint models) [contested: D-5].

**What I would NOT do:** widen `MultipoleTerm` in place under an unchanged name (silent downstream mis-scaling).

## H. Persistence

Schema v5 (decor n×3 table + L_S; factors n×3 + sites; spec fields + constraint + sectors + disp_scale). Fingerprint recomputed on load, never trusted. **Transparent back-read of v4, no converter**: the map is total/exact/canonical (`ls → SiteDecor(l,0,0)`, `L_S := Lf`, `isotropy → constraint`); reloaded v4 predictors reproduce bit-for-bit (gate). Ripples: coeftable columns; SCEMonteCarlo `model_fingerprint` change = checkpoint-schema territory downstream.

## I. MC-side implications (brief)

ChainState gains disp; PT swap widens; checkpoint bump. Spin moves unchanged; per-site Gaussian displacement move with own adaptive step in the existing color-parallel sweeps; **site-level coloring in v0** [contested: D-6]. Programs survive; leave-one-out per (site, channel); fast paths generalize by factor count. Observable signature widens ONCE to a struct (`f(state::MCState, energy, H)`); `n_active` splits per channel. Runtime guards: even-degree assert at construction; min-eigenvalue + displacement-radius diagnostics. Strain/NPT stays out of v0.

## J. Verification / oracle strategy

(1) spin-only regression = the migration oracle (bijective key map, bit-identical folded/design matrices, identical holdout predictions, v4 reload identity); (2) count-vs-projector dual-route (prototype = independent oracle; permuting-pair + centrosymmetric + mixed + same-site batteries); (3) det triple gate; (4) B₁/B₂; (5) dJ/dr + ligand, no DMI; (6) sector-mask ≡ SpinIsotropic equivalence; (7) analytic limits (Sym dims, u=0 bitwise, JPSJ spin-less); (8) derivative self-consistency (force FD + torque FD); (9) downstream (MC 1×1×1 ≡ predict_energy − intercept; per-DOF ΔE; (4π)^{nspin/2} pin).

## Load-bearing decisions

1. SiteDecor/SiteFactor isbits split, SPIN < DISP total order, at-most-one-factor-per-(site,channel) — everything else is a corollary.
2. L_S in SALCKey; SpinIsotropic provably a column mask (gate-enforced theorem).
3. One Wigner cache + visible det scalar at the trait; prototype retained solely as the counting oracle.
4. T-parity per spin block at enumeration — T-odd unrepresentable.
5. DecoratedTerm + (4π)^{nspin/2} + per-(site,channel) activity = new contract under a new name; `multipole_terms` frozen as the p = 0 view.

## Name recommendation

Rename the prefix atomically at the START of D0: SLCEFitting.jl / SLCEMonteCarlo.jl / SLCETools.jl / SLCEDynamics.jl. UUIDs kept; kugui paths + @sce env updated in the same change.

---

# Appendix B — Proposal 2: systems/performance designer (verbatim)

Perspective: everything judged against measured production realities — l044 (4672 SALCs, 70.7k canonical contraction terms, 2.2 ms/sweep 4³ CPU, 38× A100, 16³ GPU tables at the memory edge), the bitwise program-kernel discipline, and the fact that the sweep hot path touches state **only** through `zrows[row, site]` and integer CSR streams.

## A. Package naming — keep all four names

Keep `SCEFitting` / `SCEMonteCarlo` / `SCETools` / `SCESpinDynamics`. Re-expand "SCE" in docs as *symmetry-adapted cluster expansion*. If a publication-driven rename is ever wanted, do it **after** the joint expansion is validated ("rename-after-stabilize").

Measured costs: renames touch qualified bitwise-gated cross-package references (SCEMonteCarlo GPU reimplements `SCEFitting.Harmonics.Zlm_unsafe` operation-order-faithfully; SCESpinDynamics references `SCEMonteCarlo._gradient_lane_ref!` by name in gates); kugui rsync paths + @sce env Manifest + offline-extras redo (footgun #3); docs/memory/specs cross-references. **Never stack a mechanical rename on top of a semantic migration: when a gate fails mid-migration you want `git diff` to show physics, not renames.** UUID note: keeping name+UUID means old/new cannot co-load — fine, because the migration oracle should be **serialized prediction fixtures**, not co-loading.

Rejected: SLCEFitting/SpinLatticeCE (churn now, no functional gain); a fresh package beside SCEFitting (doubles linked-site maintenance for the whole window).

## B. Channel abstraction — one flat feature matrix, slots ≠ sites

**One flat per-site row matrix** (`frows :: Matrix{Float64}`, nrow_total × n_sites): spin rows 1:num_lm (unchanged lm_index order), then disp rows at `nlm_spin + disp_index(k,l,m)` — `disp_index` in fixed degree-major (p, k) order, m within label, frozen in persist. p ≤ 4 ⇒ ~34 extra rows; a column ≈ 400 B, cache-resident. Hot kernels never compute row indices (baked into programs at construction).

Why one matrix: `sfac_row` needs **no channel tag and no branch** — gather loops textually unchanged; spin move rewrites rows 1:nlm_spin of one column, disp move only disp rows; GPU layout carries over (taller matrix).

**Slots decouple from sites** (single-site Z₂·[u⊗u] = 2 slots, 1 site):

```julia
struct SiteLabel
    channel::UInt8; k::UInt8; l::UInt8
end
_packed(s) = (UInt32(s.channel) << 16) | (UInt32(s.k) << 8) | s.l

struct SALCTerm
    labels::Vector{SiteLabel}
    sites::Vector{Int8}        # slot → member atom index (NEW)
    folded::Array{Float64}
end
```

Constructor invariant: distinct `(site, channel)` slot pairs per instance ⇒ exact single-move ΔE generalizes (spin move never changes u_s; same-site disp slot is a constant neighbor factor). Repeated same-site displacement is a degree label, never a repeated slot.

**SALCKey** → `(body = slot count, orbit_id, labels (sorted by _packed), LS, Lf, block)`. Persist canonicalization: member sites sorted as v4; slots sorted by (site, channel, k, l) with folded axes permutedims-aligned — the new half of `_canonicalize_members` + round-trip gate.

Rejected: per-species row layouts (breaks flat gather + GPU tables); duplicating atoms per slot (breaks distinct-site sorting, ctor checks, reduce_cell census); Union slot vectors in hot data.

## C. Coupling tree and sector constraints

Spin-slots-first left coupling ⇒ `Lseq[n_spin] = L_S` well-defined; ops preserve channel ⇒ projector block-diagonal over L_S ⇒ `SALCKey.LS`. Spec knob `spin_mode::Symbol ∈ (:soc, :nosoc)` [per sector — see E]; `:nosoc` = one-line L_S == 0 filter; legacy `isotropy = true` ≡ `:nosoc` exactly. Store key + labels + members; NOT the Lseq path (stabilizer-mixed internal gauge). LS pays for: hierarchical freeze, sector masking, per-sector group selection.

## D. Group action — Wigner in production, polynomial composition as oracle

Production keeps the serial-precomputed read-only (l, g) Wigner cache for **both** channels; the axial/polar difference handled at the **enumeration filter**: **Σl_spin even** ⇒ det(R)^{Σl_spin} = +1 identically ⇒ polar Wigner exactly correct for the spin sector, no det branch in the projector loop. [Theory designer prefers a visible per-slot det scalar — same numbers; resolve in synthesis.]

The hazard made explicit: a mixed cluster with Σl_spin odd, Σl_total even (spin l=1 × disp l=1) passes the OLD filter and is wrongly admitted — axial l=1 is inversion-even, polar l=1 inversion-odd. Fix = split parity filter (physics rule). Gates: construction assert; centrosymmetric mixed-cluster count = 0; `is_proper` used by that test.

Polynomial composition (`_slot_matrix`): correct-by-construction but slow/BLAS-dependent — keep as the **independent test oracle** (`_slot_matrix ≈ D_channel(l, op)` over an op grid; plethysm count ≡ projector rank per cluster).

## E. Truncation spec — sector table

```julia
struct SectorSpec
    spin_body::Int; disp_body::Int
    lsum_spin::Int; pmax::Int
    spin_mode::Symbol          # :nosoc | :soc  (per sector)
    cutoff::Matrix{Float64}
end
struct JointBasisSpec
    lmax::Vector{Int}; dmax::Vector{Int}
    sectors::Vector{SectorSpec}
    species_labels::Vector{String}
end
```

§7 Y1 = five SectorSpec rows verbatim. v4 BasisSpec maps exactly to `[SectorSpec(N, 0, lsum[N], 0, mode, cutoff[N-1]) for N = 1:nbody]` — an embedding, not a shim. Boundedness: constructor *warns* on odd max total degree (error would block legitimate static-analysis fits). Group selection: `salc_groups` by (body, orbit_id, labels); `group_costs(basis; channel = :all|:spin|:disp)` — MC cost splits between spin and disp passes; Pareto axis takes a weighted sum matching the move mix. GAR/GCV/select_fit/CV survive verbatim.

## F. Data layer

```julia
struct JointDatum
    spins::Matrix{Float64}; disps::Matrix{Float64}
    strain::SMatrix{3,3,Float64,9}    # zero default [contested: D-7]
    energy::Float64
    forces::Matrix{Float64}; torques::Matrix{Float64}   # 0×0 sentinel
end
```

**Force-design memory**: X_F is 3N rows/config — naive dense ~12 GB at l044-class joint scale. But block-sparse by construction: p=0 SALCs have identically zero force rows; spin-less SALCs zero torque rows. `JointDesign` stores X_E dense + X_F/X_T with column-subset index vectors; `_assemble_problem` scatters into the whitened stacked system without materializing zero blocks; SALCScratch pattern extends (solid_harmonics_grad is a two-array fill). Hierarchical stage 1 touches only its sectors' columns.

**ASR — exact equality constraints in the estimator layer**, null-space reparameterization inside `_assemble_problem` (QR of Aᵀ once; solve in γ with X·Z; β = Z·γ; group penalties on β via `_gar_weights!` unchanged). Non-negotiables: (1) ASR exact, never penalized — a violated ASR is unbounded-below energy, a boundedness bug; (2) rejected relative coordinates — destroys frows layout, ΔE locality, adjacency ~doubles, GPU tables blow up; (3) rejected basis-level nullspace recombination — destroys key injectivity, group granularity, persist canonicalization.

Hierarchical fit: `fit(dataset; frozen::Dict{SALCKey,Float64})`; `sector_mask(basis; ...)` by key predicates.

## G. Downstream contract (SCEMonteCarlo)

```julia
struct JointTerm
    coef::Float64
    atoms::Vector{Int}; shifts::Vector{SVector{3,Int}}
    slot_site::Vector{Int8}; slot_channel::Vector{UInt8}
    slot_k::Vector{UInt8}; slot_l::Vector{UInt8}
    folded::Array{Float64}
end
```

Scale rule: `(4π)^{n_spin_slots/2}`, applied once in the TiledHamiltonian ctor. Instances = CSR over *slots* (`inst_slot_site`); gather textually unchanged. Fast paths generalize by **slot count, not body** (2-slot terms get site_col hoisting; site_col may equal the moved site itself — other-channel rows constant under the move). **Per-(site,channel) adjacency + coloring** (spin/disp CSR split so lattice-only instances don't inflate the spin conflict graph); fallback union-site coloring; measure on l044+p≤2 first [contested: D-6]. Per-DOF activity (`site_active_spin`/`_disp`, `n_active_*`). Displacement move: per-site Gaussian, second adaptive scalar; **pass scheduling is the bit-compat crux** — no disp pass on spin-only models ⇒ RNG consumption unchanged ⇒ converted-model bit-identity.

**Strain layer**: exploit coef-factored programs — add `sent_base` (raw folded) + `sent_term` and `set_coefficients!(H, coefs)` (fused stream rewrite, O(ms), once per strain move at 1/sweep). Strain move: propose ε′, interpolate K(ε′) from the 3–5-point grid, set_coefficients!, one full-energy pass, Metropolis with elastic term (V/2)εᵀCε. Outer-loop, outside sweeps/coloring. `set_coefficients!` doubles as the active-learning hot-swap hook. Runtime guards here: min-eigenvalue monitor of Φ^(2)({ê}), displacement-radius check, via observables.

Checkpoint v3 (MC): disp, step_u, strain + counters, per-channel activity; fingerprint extended over slot vectors — coordinated bump with SCESpinDynamics.

GPU: device feature matrix ~2–3× rows; table memory (16³ limiter) grows with entry count — **re-measure before promising sizes**. `_dlm_row_device`: the polynomial evaluator has no sqrt/acos/libm ⇒ easier bitwise device port than Zlm was. Philox slot map gains disp draws (schema/G-record bump, per the QTB precedent).

## H. Persistence — v5, back-read v4 in the loader

One format; spin-only models embed as all-channel-0 labels (writer never special-cases). Back-read v4 in `_basis_from_doc`: total mechanical value-preserving map ⇒ v4-loaded models must produce **bit-identical** Φ/∇Φ and byte-identical MC programs (the cheapest, strongest gate — a migrate-once tool would forfeit it and leave stale-file landmines on kugui). Drop v2/v3 branches only after checking no files remain in circulation.

## I. Observable/State API

```julia
struct MCView
    spins::SpinConfig
    disps::Vector{SVector{3,Float64}}   # empty for spin-only
    strain::SVector{6,Float64}
    energy::Float64
end
# Observable: f(view::MCView, H)
```

Break once; accessors so the next field is not another break. `Evaluable` n::Int → counts::NamedTuple (spin/disp/cells). Configurational-specific-heat convention documented. SCETools/SCESpinDynamics mechanical follow-ups; `disps` in the view anticipates the Verlet layer (one break, not two).

## J. Verification / oracle strategy

(1) relabel bit-identity (v4 file → v5 loader → bit-identical evaluate/grad; **byte-equal MC program arrays**); (2) spin-only MC bit-identity (l044 8³ fixed seeds, config hash + incremental energy, multitask + GPU); (3) predictions oracle via serialized full-digit fixtures (no co-loading despite shared UUID); (4) span equivalence (rank([X_old X_new]) == rank(X_old) == rank(X_new)); (5) new-physics exact gates (B₁/B₂; dJ/dr + ligand no-DMI; plethysm ≡ rank incl. permuting pair + centrosymmetric mixed; Sym dims; _slot_matrix ≈ Wigner); (6) u = 0 bitwise degeneracy; (7) FD forces + ASR residual ≤ 1e-13 + uniform-translation invariance at machine precision; (8) MC joint (ΔE ≡ total diff for spin/disp/strain; serial ≡ parallel bitwise; checkpoint resume bitwise; set_coefficients! no-op round-trip byte-identical); (9) spin-less limit vs JPSJ code.

## Load-bearing decisions

1. One flat feature matrix, slots ≠ sites, distinct (site, channel) ΔE contract ⇒ programs/fast paths/GPU survive structurally.
2. Σl_spin-even at enumeration + polar Wigner for both channels; prototype demoted to oracle; centrosymmetric mixed cluster is the kill-shot.
3. SALCKey with labels + LS; v5 with v4 back-read gated by relabel bit-identity.
4. ASR = exact equality constraints in `_assemble_problem`; never penalty/relative-coordinates/basis-recombination.
5. coef-factored programs + `set_coefficients!` enable the strain outer move at ~1 sweep cost with zero hot-sweep overhead; spin-only sweep path stays RNG- and byte-frozen.

## Name recommendation

Keep all four names/UUIDs/remotes/kugui paths; re-expand "SCE" as symmetry-adapted CE; revisit only after J-gates are green as an isolated mechanical commit series.

---

# Appendix C — Proposal 3: API / ecosystem / UX designer (verbatim)

Design principle: **the joint expansion is the primary object; spin-only must read as "one sector," never as "the real package plus a bolt-on."**

## A. Package naming (primary agenda)

**Rename the whole family to the stem `SLCE`, and drop "Fitting" from the core: `SLCE.jl`, `SLCEMonteCarlo.jl`, `SLCEDynamics.jl`, `SLCETools.jl`.**

1. **The method name and the package name must be the same word.** The papers need a name for the method; the natural coinage is *spin–lattice cluster expansion (SLCE)* — slots into the acronym ecology (ACE, SCE, SLD), unclaimed. "We implement the SLCE in the open-source package SLCE.jl."
2. **"SCE" is now conceptually false for three of the four packages** (force-constant-only fits are first-class; MC samples displacements + NPT strain; Dynamics goes coupled).
3. **"Cluster expansion" unqualified is a hazard** (alloy/occupation CE: CASM, icet, smol) — kills generic names.
4. **Dropping "Fitting" is honest**: the core defines the expansion — basis, symmetry, fitting, introspection (the scientific selling point). Precedent: DFTK.jl, Sunny.jl name the method/domain.
5. **The rename precedent cuts in favor** (MagestyRebuild → SCEFitting); zero external users + pre-publication = the last cheap moment; one-rename principle — everything in one breaking window, never again.

Rejected: keep-and-reinterpret (backronym of a published acronym — worst move for an audit-branded project); rename core only (incoherent family); non-acronym name (method term needed anyway); `SLCESpinDynamics` (wrong the day lattice dynamics lands — `SLCEDynamics` is forward-correct).

## B. User-facing API shape

| today | D0 |
|---|---|
| `BasisSpec` | `BasisSpec` (content = sector list) |
| `SCEBasis` | `SLCEBasis` |
| `SCEDataset` | `SLCEDataset` (pins reference::Crystal) |
| `SCEPredictor` | **`SLCEModel`** ("Predictor" reads ML-ish; this is an explicit auditable Hamiltonian) |
| `SCEFit` | `SLCEFit` |
| `SpinDatum` | **`TrainingDatum`** (channels optional) |

Pipeline preserved: Crystal + BasisSpec → SLCEBasis → SLCEDataset → fit → SLCEModel. StatsAPI generics unchanged.

Joint 5-liner:

```julia
spec = BasisSpec(crystal; sectors = [
    Sector(disp = (degree = 2:4,), cutoff = 6.0),
    Sector(spin = (nbody = 2:4, lmax = 3), cutoff = 8.0),
    Sector(spin = (nbody = 2, lmax = 1), disp = (degree = 1:2,), soc = false, cutoff = 5.0)])
basis  = SLCEBasis(crystal, spec)
result = fit(basis, SLCEDataset(basis, read_configs(VASP.ForceSet("scf/*")));
             estimator = GroupAdaptiveRidge())
```

Spin-only 5-liner = same shape with one sector — "the p = 0 slice of the one expansion" (docs must say exactly that). Evaluation: `predict_energy(model, spins; displacements = u)` etc., displacements defaulting to zero. Strain-grid family = core container `StrainedModels` + `model_at(sm, ε)`; sampling lives in MonteCarlo.

## C. Concept vocabulary

**Decorated cluster** (修飾クラスター); **channel** (spin/displacement; "slot" = internal term, public-unexported); **degree and rank** for the displacement label (trace channels = "the radial dependence" — one glossary line kills the radial-basis reviewer question); **sector** (the load-bearing user-facing concept; theory-note §3 table = the docs' conceptual map; each row literally a `Sector(...)` call); **cluster invariant** Θ_ωγ in papers/guides, **SALC** survives as the internal machinery term (persistence, selection, MC adjacency). Rule: papers say invariant; internals say SALC.

## D. Sector constraints API

```julia
Sector(; spin = nothing, disp = nothing, soc = true, cutoff = ...)
```

`soc = false` = "this sector is exchange-only" (L_S = 0 + plain space group) — replaces and inverts `isotropy` (a physicist thinks "does this term need SOC"). Presets: `soc_free(spec)`, `force_constant_spec(crystal; ...)`. Hierarchical fit first-class:

```julia
f1 = fit(basis, force_data; sectors = :soc_free)
f2 = fit(basis, rot_data;   frozen  = f1)
```

Sector selectors: `:all, :soc_free, :soc, :spin, :lattice, :coupled`. Rejected: global `soc::Bool` (hierarchy needs granularity); product-grid spec (explodes basis, hides requested physics).

## E. Introspection / deliverables API — the selling point

```julia
multipole_terms(model)                       # widened; stays the stable contract [contested: D-5]
bilinear_terms(model; displacements = u)     # J_ab(u)
force_constants(model; spins = ê, order = 2) # Φ^(2)({ê}) → ForceConstantSet
dynamical_matrix(fc, q; masses)
exchange_strain_derivatives(model)           # dJ_ab/dε closed-form lattice sums
magnetoelastic_constants(model)              # labeled list; cubic ⇒ (; B1, B2)
magnon_phonon_vertices(model, ground_state)  # "adiabatic" in the docstring (honesty)
restrict(model, :spin)                       # exact clamped-ion sub-model
to_sunny(model; clamp = true)
```

"One coefficient set, four outputs" (theory §4) = THE docs page. Each function's docstring carries its exact-gate example as a doctest. Rejected: phonon bands in core (ForceConstantSet + phonopy writer in Tools); monolithic `deliverables(model)`.

## F. Documentation architecture

index (sector table as conceptual map) / getting_started (joint 15-liner, then spin-only labeled "the p = 0 sector") / theory: unified_expansion, spin_sector, displacement_sector ("no radial basis" boxed note), sectors_and_groups (why DMI is exactly SOC) / guide: basis, data, fitting (hierarchical + ASR + conditioning), deliverables, migration / tutorials: heisenberg_chain + kagome (unchanged — proof), force_constants_si (JPSJ oracle), cubic_magnetoelastic (B₁/B₂), pair_ligand_dJ (DMI absence) / verification: angular_momentum, invariant_counting, exact_limits.

## G. Package boundaries — four-package shape survives; no fifth package

Strain/NPT in SLCEMonteCarlo (sampler concern; a new package would fragment checkpoint/determinism/GPU discipline). SLCEDynamics does spin-lattice dynamics in-package (the rename makes it the stated destiny). SLCETools grows (forces parsing, phonopy export). Core owns `StrainedModels` only. Rejected: SLCEPhonons.jl; samplers in core.

## H. Data layer (brief)

`TrainingDatum` with optional channels (spins / displacements / forces / field+torques, or nothing); `SLCEDataset(basis, configs)` pins the reference structure as a constructor invariant; `SpinDatum(...)` remains as convenience constructor; readers: `VASP.ForceSet` extending `read_configs(::AbstractDFTSource)` in Tools.

## I. Migration UX (brief)

Old model files load (v4 back-read); old numbers reproduce bit-identically (pinned gate, user-visible in verification docs); `guide/migration.md` carries the full old→new name table; no compat shims beyond `SpinDatum`.

## J. Verification (API angle)

User-visible documented gates: B₁/B₂ rederivation as the flagship tutorial ("it really is mechanical"); spinless limit vs JPSJ oracle env; spin-only bit-identity + restrict ≡ clamped evaluation; invariant counting documented on the permuting-pair and centrosymmetric cases; dJ/dr tutorial asserting DMI absence — selection rules as runnable docs.

## Load-bearing decisions

1. One stem, method = package: SLCE. Everything renamed in one breaking window, never again.
2. Sectors are the specification (§3 table 1:1; `soc` per sector; spin-only is one sector, literal via u=0 exactness).
3. `SLCEModel` + the deliverables tier as the paper-citable surface (plural physics nouns, symmetry-labeled returns, one exact gate each).
4. multipole_terms/SALC layering holds (papers: invariant; internals: SALC; MultipoleTerm widens compatibly).
5. Boundaries don't move; SLCEDataset pins the reference structure.

## Name recommendation

| current | new repo | module |
|---|---|---|
| SCEFitting.jl | SLCE.jl | SLCE |
| SCEMonteCarlo.jl | SLCEMonteCarlo.jl | SLCEMonteCarlo |
| SCESpinDynamics.jl | SLCEDynamics.jl | SLCEDynamics |
| SCETools.jl | SLCETools.jl | SLCETools |

Mechanics: GitHub renames (redirects keep the PRR citation resolving); Project.toml names changed, **UUIDs kept**; kugui rsync dirs + @sce → @slce; local checkouts renamed; migration.md records the mapping. Order: core first, then dependents, then the on-cluster suite as the gate. Persist schema carries a version, not a package name. (Flourish: SLCE pronounceable "slice" — one expansion, sliced into sectors.)

---

# Appendix D — Adversarial-review report on D-4/D-5/D-7/D-8 (verbatim, 2026-07-25)

## D-4 — SOC-constraint granularity

**VERDICT: Per-sector granularity wins, but reframed — there is ONE global projection (the grey group, always), and the per-sector knob `soc::Bool` is an L_S-selection *truncation rule* applied at enumeration, not a per-sector "constraint" or group switch. The theory designer's global `constraint ∈ {GreyGroup, SpinIsotropic}` is rejected.**

**The deciding argument.** The theory designer's position fails its own expressibility test, twice, on his own documents:

1. *Neither global value expresses the Y1 truncation.* Theory note §7 puts in ONE model: (i) the spin-only SCE sector with its PRR truncation — which contains SOC channels (single-ion anisotropy, Lf ≠ 0) and therefore needs the grey-group (SOC) columns; (ii) the pair l=(1,1)×p≤2 and 4-spin×p=1 coupled sectors *explicitly SOC-less* (the §7 hierarchy fits them from forces/torques precisely because their SOC channels are 0.01–1 meV noise to be excluded, not fitted); (iii) the SOC single-site l=2×p≤2 sector (B₁/B₂), which is L_S = 2 — killed outright by global `SpinIsotropic`. Global `GreyGroup` over-admits: it enumerates the DMI-type L_S ≠ 0 invariants of the coupled pair sector (they exist for any non-centrosymmetric bond — e.g. the pair+ligand geometry), contaminating the basis Y1 deliberately excludes. Global `SpinIsotropic` under-admits: it deletes the flagship B₁/B₂ sector. Appendix A's claim "Y1 truncation expressible verbatim (5 sector rules)" is false as written, because his `SectorRule` carries no SOC field and his single `constraint` cannot vary across the five rows.

2. *His own gate (f) presupposes per-sector granularity.* Verification item (f) — "SOC-less pair+ligand p=1: dJ/dr + ligand, no DMI" — is a statement about one sector of a model whose other sector (B₁/B₂) requires the grey group. Under global `GreyGroup` the DMI invariant *is* enumerated and the gate is unreachable; under global `SpinIsotropic` the model containing gate (e) (B₁/B₂ = exactly 2) cannot exist. The proposal's own verification battery is unsatisfiable without per-sector SOC.

**Why the "column mask suffices" escape fails.** The fit-time `sector_mask(L_S)` controls which columns are *fit in a stage*, not which columns *exist in the model*. Rescuing global-GreyGroup with a permanent mask means the excluded DMI-class columns still get enumerated, projected, persisted, group-costed, and compiled into MC programs, then carried dead behind a mask that must be re-applied in every stage and in every consumer — i.e. a second, shadow truncation spec. That is exactly the "configuration surface without a theory" and the dead-state convention-bug the theory designer warns against elsewhere (his D-7 argument, his own "spec-is-canonical" rule in Appendix A §E). The spec is the canonical statement of model support; support exclusion belongs in the spec.

**Why the reframing is nonetheless owed to the theory designer.** His underlying discomfort is legitimate and is fixed by the middle ground: there are *not* two group actions. Agreement §1.4 plus his own reduction theorem (Appendix A §C: "`SpinIsotropic` under the same grey-group projector reduces exactly to spin-scalar × plain-space-group projection") establish that the SOC-less basis is a theorem-backed *subset of columns* of the one grey-group basis — L_S = 0 exactly. So the settled design is: one projector (grey group, no per-sector group switch, no `SectorConstraint` type), and per sector `soc::Bool` where `soc = false` ⇔ "enumerate only L_S = 0" — the systems designer's mechanism (`:nosoc` = one-line L_S filter) justified by the theory designer's own theorem. A per-sector L_S selection rule has exactly as much theory as the global one — and §7's sector table arrives with SOC labels *per row*, so granularity is the theory note's, not an API invention.

**Hierarchical fit:** needs *both*, doing different jobs. The spec-level `soc` flag defines model support (truncation axis). The `L_S`-keyed `sector_mask` + `frozen` offset defines fit staging (staging axis) — and correctly freezes the L_S = 0 columns of `:soc` sectors too (they are SOC-inert physics by the same theorem). Do not merge the two mechanisms.

**Conditions.**
- Gates (f) and (n) MUST be exercised on a *mixed* spec (`:nosoc` coupled sectors coexisting with `:soc` sectors in one model), not on a globally-`:nosoc` spec — otherwise D-4's whole point goes untested.
- `GreyGroup`/`SpinIsotropic`/`SectorConstraint` are dropped as public types; legacy `isotropy = true` gets the deprecation error pointing at `soc = false` (theory designer's own suggestion, retargeted).
- Docs must state the reduction theorem as the *reason* `soc = false` is exact, and that `soc` (truncation) ≠ `sector_mask` (staging).

## D-5 — MultipoleTerm successor semantics

**VERDICT: Option (a)+(c) combined — `multipole_terms(model)` on any model containing a p ≥ 1 sector THROWS an `ArgumentError` whose message names both escape hatches (`decorated_terms(model)` for the new contract; `multipole_terms(restrict(model, :spin))` for the exact clamped-ion view); on pure-spin models (including every v4-loaded model) it returns today's `Vector{MultipoleTerm}` bit-identically. The successor is named `DecoratedTerm` / `decorated_terms`. Option (b) — silent p = 0 restriction — is rejected, and the API designer's "widen `multipole_terms` compatibly" is rejected on code-reality grounds.**

**The deciding argument.** The code reality kills widening-in-place outright: both consumers compute the scale *from term shape*. `TiledHamiltonian`'s ctor (SCEMonteCarlo `src/hamiltonian.jl:241` ff.) applies `(4π)^(body/2)` with `body = length(mt.atoms)`, and SCETools' `_scaled_multipole_terms` (`src/mfa/bridge.jl:63`) does the same. Under the settled scale rule `(4π)^{n_spin_slots/2}` (agreement §1.10), a widened term with any disp slot has `body ≠ n_spin`, so every unmigrated consumer mis-scales silently by powers of √(4π) — the theory designer's exact scenario, verified in source. The API designer's "stays the stable contract" is not a conservative option; it is the dangerous one.

Between throw and silent restriction, the decisive observation is that the **compatibility benefit of (b) is vacuous**: on spin-only models all three options behave identically (that is where SCETools' consumers actually live until they are taught displacements), so (b)'s only distinctive behavior is on *joint* models — where handing an old MFA sampler the clamped-ion restriction of a joint Hamiltonian is silent physics truncation, on models for which the old consumer has no meaningful semantics anyway. A package whose selling point is auditability cannot have its flagship introspection call silently discard the lattice channel. Meanwhile the migration burden of throwing is ~zero, because no existing code path can receive a joint model before its package is migrated. The API designer's discoverability concern is fully absorbed by the error message: the throw *is* the discoverable pointer to (c), and `restrict(model, :spin)` is already in his own deliverables tier, with its exactness guaranteed structurally (all disp factors have k ≥ 1, so E(u = 0) degenerates exactly to the clamped-ion spin SCE — agreement §1.1).

**Exact behavior, pinned:**
- `multipole_terms(model)`: pure-spin model (all sectors p = 0, any schema vintage) → today's return, bit-identical (this is gate 14(a)/(c)'s frozen contract). Any p ≥ 1 sector present → `throw(ArgumentError("model has displacement sectors: multipole_terms is the frozen spin-only (p = 0) contract. Use decorated_terms(model) (scale rule (4π)^{n_spin_slots/2}), or multipole_terms(restrict(model, :spin)) for the exact clamped-ion sub-model."))`. The trigger is *sector presence in the spec*, not fitted-coefficient values — a joint model whose disp coefficients happen to be zero still throws (predictability beats cleverness).
- `restrict(model, :spin)` returns a genuine pure-spin model on which `multipole_terms` then works — (c) is the sanctioned two-step, not a third semantics.

**Name.** `DecoratedTerm`, not `JointTerm`. The settled vocabulary (Appendix C §C, ratified in §1) makes "decorated cluster" the public concept and "channel" the public axis word; the D-8 reconciliation eliminates every other "Joint*" noun (`JointDatum` → `TrainingDatum`, `JointBasisSpec` → `BasisSpec`), so `JointTerm` would be the lone orphan of a dead prefix, while `decorated_terms(model)` reads as "the terms of the decorated-cluster expansion" — the paper sentence. (The systems designer's flat-vector field *layout* for the struct is untouched by this naming choice.)

**Caveat.** The `(4π)^{n_spin_slots/2}` rule is invariant across the options but still demands that *every* consumer derive the scale from slot channels, never from `length(atoms)`; see Residual risk 3.

## D-7 — strain field on the training datum

**VERDICT: The theory designer wins — no `strain` field on the training datum in v0. Strain metadata lives (i) at the reference level: each K(ε) grid point is its own strained reference `Crystal` with its own `reference_id`, pinned by the `SLCEDataset` constructor; and (ii) at the model level in the core `StrainedModels` container (`model_at(sm, ε)`), which owns the ε grid keys and the elastic term. Optionally, a documented helper deriving ε from a (reference, base-reference) lattice pair — derived, never stored.**

**The deciding argument.** Walk the actual K(ε) workflow through the settled types: `SLCEDataset` pins ONE reference (agreement §1.7; constructor invariant). Under the clamped-cell protocol, a datum's cell *is* its dataset's reference plus explicit displacements — so within any legal dataset, a per-datum `strain` field is *definitionally constant*, and equal to information the reference crystal's lattice vectors already carry. A stored constant duplicate of derivable reference data is two sources of truth with a contradiction channel (`datum.strain` vs. the strain implied by `dataset.reference` — which wins when they disagree?) and **no v0 reader**: no design matrix consumes it (there is no strain column; K(ε) is fit *per grid point*, not as a Taylor series in ε), no estimator sees it, no gate touches it. That is the textbook dead field. The systems designer's own one-source-of-truth discipline (spec-is-canonical, conventions live in one place) condemns his own field. Note the datum does not carry lattice vectors either — strain belongs wherever the lattice lives, and that is the reference.

The systems designer's forward-compatibility argument ("strained-reference datasets are coming") actually argues the theory side: they come *as separate datasets at strained references* — which is exactly the mechanism that makes the field redundant. And his best future card — stress as a training target (§2c) — sharpens the distinction rather than rescuing the field: stress is a genuine **per-configuration observable** (a DFT output that varies datum-to-datum), while strain is a **protocol parameter** fixed by the reference. Datum fields = per-configuration observables (energy, forces, torques, someday stress); dataset/reference level = protocol parameters (reference structure, strain). The `occupations` precedent in §2b already establishes the answer to "but another struct break later": versioned additive fields, which the v5 schema discipline supports.

No contradiction with agreement §1.13: `MCView.strain` is MC *state* (where ε is a sampled variable), a different object from a training observable — both sides already agree on that.

**Conditions.**
- The decision record must write the K(ε) pipeline explicitly (per grid point: strained `Crystal` → basis → dataset → fit → entry in `StrainedModels`), so "no field" does not decay into "no story".
- `DatumProvenance.reference_id` (theory F) is the load-bearing anchor and must survive into `TrainingDatum`; the `SLCEDataset` ctor asserts every datum's `reference_id` matches the basis reference.
- State the datum extension policy in the design doc: future per-configuration observables (stress, occupations) are versioned additive fields — this is the honest answer to the systems designer's real concern.
- See Residual risks 1–2: the K(ε) mechanism has an unaddressed symmetry problem and an unpinned strain convention that this verdict does *not* settle.

## D-8 — reconciled noun table

**Tier 1 — pipeline (user-facing, core `SLCE`)**

| noun | was | notes |
|---|---|---|
| `BasisSpec` | `BasisSpec` / systems' `JointBasisSpec` | keeps its name (agreement §1.6); content = resolved sector list; dense storage **channel-keyed** so the reserved OCC channel is a new key, not a struct break (§2b items 1/2/5) |
| `Sector(; spin, disp, soc = true, cutoff)` | theory `SectorRule` / systems `SectorSpec` | the ONE public sector noun; sugar resolved once (BasisSpec discipline); `soc::Bool` per D-4 = L_S = 0 truncation rule under the single grey-group projection; `spin_mode::Symbol` dropped |
| `SLCEBasis` | `SCEBasis` | — |
| `SLCEDataset` | `SCEDataset` | pins one `reference::Crystal`; asserts per-datum `reference_id` (D-7) |
| `SLCEFit` | `SCEFit` | — |
| `SLCEModel` | `SCEPredictor` | — |
| `StrainedModels` + `model_at(sm, ε)` | new | core container; the home of the ε grid + elastic term (D-7) |
| sector selectors `:all/:soc_free/:soc/:spin/:lattice/:coupled`; presets `soc_free`, `force_constant_spec` | new | unchanged from Appendix C §D |

**Tier 2 — data**

| noun | was | notes |
|---|---|---|
| `TrainingDatum` | `SpinDatum` / both designers' `JointDatum` | **`JointDatum` is dropped** — a spin-only user must not hold a "joint" noun; channels optional; **no `strain` field** (D-7); stress/occupations = future versioned additions |
| `SpinDatum(...)` | — | survives as convenience constructor returning `TrainingDatum` only |
| `DatumProvenance` | theory F | **added — missing from the API table**; carries `constrained`, `torque_qualified`, `reference_id` (agreement §1.7 requires all three) |

**Tier 3 — introspection / downstream contract**

| noun | was | notes |
|---|---|---|
| `DecoratedTerm` / `decorated_terms(model)` | `MultipoleTerm` / systems' `JointTerm` | the stable contract; per-slot (channel, k, l) + slot→site map; scale `(4π)^{n_spin_slots/2}` (D-5) |
| `MultipoleTerm` / `multipole_terms(model)` | itself | **frozen** p = 0 legacy view; bit-identical on pure-spin models; **throws** on joint models with the two-hatch message (D-5) |
| `restrict(model, :spin)` | new | exact clamped-ion sub-model; the sanctioned bridge to the frozen view |
| `row_layout(model)` | agreement §1.10 / §2b item 4 | **added — missing from the API table**; the shipped per-channel row-offset table is part of the public contract, not an internal |
| deliverables: `bilinear_terms(model; displacements)`, `force_constants` → `ForceConstantSet`, `dynamical_matrix`, `exchange_strain_derivatives`, `magnetoelastic_constants`, `magnon_phonon_vertices`, `to_sunny(model; clamp)` | Appendix C §E | kept unchanged; no collisions found |

**Tier 4 — internal (unexported; "slot"/"SALC" vocabulary per the settled glossary rule)**

| noun | was | notes |
|---|---|---|
| `SiteFactor` (isbits: channel, k, l) | systems' `SiteLabel` | **`SiteLabel` dropped as a misnomer by the systems designer's own axiom** (slots ≠ sites; the label decorates a *factor/slot*, a site may carry two); satisfies §2b's channel-tagged-triple + `SPIN < DISP < OCC` reserved order |
| `SiteDecor` | theory B | per-site combined decor in `SALCKey`; matches the "decorated cluster" vocabulary |
| `SALCKey` / `SALCTerm` / `SALC`, `sector_mask`, `salc_groups` | — | internal, per the "papers say invariant, internals say SALC" rule |

**Tier 5 — downstream (SLCEMonteCarlo)**

| noun | was | notes |
|---|---|---|
| `MCView` (spins, disps, strain, energy) + per-channel `counts` | systems I / agreement §1.13 | adopted — the API table didn't cover the MC observable break; accessors required so the next field is not another break |
| `set_coefficients!(H, coefs)` | systems G | kept (strain outer move + active-learning hot-swap) |

**Dropped nouns (must appear in `guide/migration.md`):** `JointDatum`, `JointBasisSpec`, `SectorSpec`, `SectorRule`, `SectorConstraint`/`GreyGroup`/`SpinIsotropic`, `SiteLabel`, `JointTerm`, `spin_mode`, `isotropy` (deprecation error → `soc = false`).

**Changes to the API designer's version:** (1) `multipole_terms` freezes + throws (D-5), his load-bearing decision 4 falls; (2) `DatumProvenance` added; (3) `TrainingDatum` carries no strain field + versioned-addition policy; (4) `Sector`'s dense resolved form channel-keyed (OCC = new key + `occ = ...` kwarg later, byte-compatible); (5) `soc = false` docstring = "restrict this sector to L_S = 0 under the one grey-group projection" (theorem-backed), truncation vs staging separated; (6) `row_layout(model)` added to the public tier; (7) `MCView` adopted; (8) dropped-nouns table goes into migration.md.

## Residual risks

1. **K(ε) grid vs. symmetry breaking — a real hole no designer addressed.** A strained reference generically has a *lower point group* than the unstrained cell, so per-grid-point bases have different SALC counts and different keys; `model_at(sm, ε)` interpolation and MC `set_coefficients!` both silently assume ONE program structure across the grid. Interpolating K(ε) requires a common basis (e.g. project all grid points in the common-subgroup symmetry of the strain family, or evaluate the unstrained-symmetry basis at strained geometry) — this must be decided before `StrainedModels` is implemented, or the strain-move design in §I is unimplementable as specced.
2. **No strain convention is pinned anywhere.** `MCView` carries `SVector{6}`, the rejected datum field was `SMatrix{3,3}`, and CASM ships three strain metrics (GL/Hencky/EA). Voigt ordering, symmetric-vs-engineering shear, and the metric must be fixed once (persisted, like `disp_scale` and the (4π) rule) before the first strained artifact exists.
3. **The `(4π)^{n_spin_slots/2}` migration has a consumer that everyone forgot: `reduce_cell`.** SCEMonteCarlo's cell-reduction emits raw `MultipoleTerm` lists and its `q·|det M|` census reasons over canonical members — both must migrate to `DecoratedTerm` with mixed channels, and any consumer deriving the scale from `length(atoms)` (the current ctor does exactly this) mis-scales silently on hand-built joint lists. The "(4π)^{nspin/2} pin" gate must include a hand-built mixed-channel term and a reduced-cell joint model, not just `decorated_terms` output.
4. **`soc` (truncation) vs. `sector_mask` (staging) can drift into folklore.** The L_S = 0 mask correctly freezes the L_S = 0 columns of `:soc` sectors too; gate (n) as currently phrased tests only the globally-SOC-less corner. Unless (f) and (n) run on a *mixed* spec (D-4 conditions), the design's central new expressibility ships untested.
5. **`restrict(model, :spin)` invites a category error the docs must pre-empt.** The restriction is exact as an evaluation statement (E at u = 0), but its coefficients are *not* what a spin-only fit on relaxed structures would give (that difference is the double-counting protocol's entire point). A user exporting `restrict(model, :spin)` to Sunny or MFA and comparing against a legacy relaxed-structure SCE fit will see discrepancies and file them as bugs — one boxed docs paragraph ("restrict ≠ refit") plus a comparison example closes it.
