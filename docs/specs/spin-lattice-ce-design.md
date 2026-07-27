# Spin–lattice cluster expansion (SLCE) — design decision record

**Status: SETTLED (D0 closed 2026-07-25). This is the normative design.**
**Revision 1 (2026-07-25): amended after a three-agent review** (numerical /
d0-consistency / codebase reality). Amendments are integrated in place; the
review reports are archived in the session record. One post-D0 pin was made at
review time: the displacement-kernel normalization (§3, 4π-free), forced by the
scale rule already ratified in D0.
**Revision 2 (2026-07-27): the M5 strain convention pinned** (§9), after a
second three-lens review (numerical correctness / implementation cost /
adversarial). This closes §13 risk 2 and settles the last decision M5 could not
start without. Four acceptance conditions the review surfaced are recorded with
it — the B₁/B₂ output convention (§7), the grid reference-geometry protocol
(§9a), the target ensemble and the move's action on `u` (§8), and the ASR
precondition of the strain deliverables (§9d) — plus one correction: the
existing rigid-rotation gate is first order only and does NOT close §9b hole
(i) (§9b, §12 q).

Companions: `spin-lattice-ce.md` (feasibility study, code-verified mapping onto
the existing packages) and `spin-lattice-ce-d0.md` (the full D0 discussion
record — three designer proposals, the adversarial review, and every argument
behind the decisions below; consult it before reopening anything here). Theory
source: the kakenhi-2026 theory note (unified spin–lattice CE), whose
conventions this record implements. The DisplacementBases prototype referenced
throughout lives in `~/Packages/Magesty.jl-coupledce/src/DisplacementBases.jl`.

Every decision below is final for the implementation window. Reopening one
requires a written reason appended to this file, not a silent divergence.

---

## 1. Scope

One symmetry-adapted cluster expansion over decorated clusters: each cluster
site carries factors from typed **channels** — spin (axial tesseral harmonics,
rank l ≥ 1, T-parity (−1)^l) and displacement (polar solid harmonics, label
(k, l), degree 2k + l ≥ 1, T-even) — CG-coupled into cluster invariants and
projected onto the identity of the grey magnetic group of a single
high-symmetry reference crystal (space-group ops act on sites/shifts; the
tensor-axis action is their point-group part).

Principles with the force of law:

- **The joint expansion is the primary abstraction.** Spin-only SCE is the
  p = 0 value of the general types, never a sibling code path.
- **u = 0 exact degeneracy**: every displacement factor has degree ≥ 1, so the
  joint model evaluated at u = 0 degenerates bitwise to the clamped-ion spin
  SCE. This is both a design principle and a permanent gate. (Evaluation
  statement only: the u = 0 *gradient* of degree-1 factors is nonzero —
  reference forces are real physics that `restrict` discards.)
- **No radial basis for displacements**: solid harmonics are evaluated as
  homogeneous polynomials (recurrence, no libm), which is what makes u = 0
  regular, gradients Euclidean, and the evaluator GPU-portable.
- **Selection rules are structural**: T-parity (Σl_spin even) enforced at
  enumeration; axial-vs-polar handled without a det branch in the projector
  (§4); illegal labels unconstructable (§3).
- **Vocabulary rule (ratified)**: papers and user docs say *cluster invariant*,
  *decorated cluster*, *channel*, *sector*; "SALC" and "slot" are internal
  vocabulary, never exported.

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

**Type renames (D-8 Tier 1), applied in the same M0 mechanical window:**
`SCEBasis → SLCEBasis`, `SCEDataset → SLCEDataset`, `SCEFit → SLCEFit`,
`SCEPredictor → SLCEModel` ("Predictor" reads ML-ish; this is an explicit
auditable Hamiltonian). `BasisSpec` keeps its name. Persisted artifacts store
schema structure, not Julia type names — verify this holds (no type-name
strings in v4 docs) as part of the M0 gate.

- The rename is the **FIRST step**: an isolated, purely mechanical commit
  series before any semantic work (never interleaved — when a gate fails
  mid-migration, `git diff` must show physics, not renames).
- **UUIDs kept** (unregistered path-dev packages; Manifests stay resolvable;
  the migration oracle uses serialized prediction fixtures, never co-loading).
- **Rename checklist** (grep-verified 2026-07-25; each item is a gate surface):
  - Root `Project.toml` name + `[deps]`/`[sources]` entries in all four
    packages.
  - Every sub-environment `Project.toml`/`Manifest.toml`:
    SCEFitting `{bench, docs, examples, test/oracle, test/sunny, test/glmnet}`;
    SCEMonteCarlo `{bench, bench/gpu, docs}`; SCESpinDynamics `{docs}`;
    SCETools `{docs, examples, test/oracle}` (the latter also path-devs
    `../../../Magesty.jl` — relative depth must survive).
  - GitHub Actions: `SCETools.jl/.github/workflows/CI.yml` hardcodes
    `repository: Tomonori-Tanaka/SCEFitting.jl` in a cross-repo checkout +
    `Pkg.develop(path = "SCEFitting.jl")`.
  - Documenter `docs/make.jl` in all four packages: `sitename`, `repolink`,
    `using`/`DocMeta.setdocmeta!` module names; `docs/Makefile` targets;
    regenerate (don't hand-edit) stale `docs/build/` artifacts.
  - Qualified cross-package references (SCEMonteCarlo GPU ports
    `SCEFitting.Harmonics.Zlm_unsafe` operation-order-faithfully;
    SCESpinDynamics gates call `MC._gradient_lane_ref!` by qualified name).
  - kugui rsync target dirs; shared env `@sce` → `@slce` (+ offline
    test-extras redo, kugui footgun #3); docs/specs/memory cross-references.
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
The axial/polar det factor has ONE seat, `rep_scale(trait, detR, l)` — but the
**production projector never applies it**: correctness for the spin channel is
the §4 parity theorem, and `rep_scale` is consumed only by the
validation/oracle layer (gate o) and by future channels that need it. (This
placement is a post-D0 synthesis of the two designer positions, ratified at
the 2026-07-25 review.)

**Displacement-kernel normalization (pinned at review, forced by §7):** the
displacement solid harmonics are **4π-free** (Racah-type: no per-factor
(4π)^{−1/2}). The prototype evaluator locks its normalization to the spin-side
`_plm_norm` (which carries (4π)^{−1/2}) — the M1 port MUST renormalize, and
the M1 "Zlm cross-check" gate is restated with the explicit √(4π/(2l+1))-class
factor. Without this, §7's scale rule mis-scales every disp slot by √(4π).

Rejected: `Vector{AbstractSlot}` / Union eltypes in SALC storage; type-parameter
slots; packed-integer cleverness; per-species row layouts.

## 4. Group action and selection rules

- **ONE projection: the grey group, always.** There is no second group action
  and no `SectorConstraint` type.
- **Σl_spin-even is enforced at enumeration** (per-channel T-parity product
  = +1; disp factors T-even). Consequence: det(R)^{Σl_spin} = +1 identically,
  so the existing polar Wigner cache (`wignerD_real`, serial-precomputed,
  read-only) is exactly correct for BOTH channels — no det branch in the
  projector loop. Why the enumeration filter is *exact* (not an approximation
  of a projection): T is central and acts as the scalar (−1)^{Σl_spin} on each
  label multiset, so the grey-group projector factorizes as (spatial average)
  × (1 + (−1)^{Σl_spin})/2. **The load-bearing edit**: today's `_enumerate_ls`
  enforces Σl-even over ALL sites; the joint version must count over SPIN
  slots only (an odd-l disp factor must not enter the parity count). Gate (o)'s
  mutation tests are the fence around this edit.
- **Spin-first canonical coupling order** (a corollary of the SPIN < DISP label
  order): all spin axes precede disp axes; left-coupling makes **L_S (total
  spin rank) a good quantum number** of the projection (stabilizer ops permute
  spin slots among spin slots; the common rotation preserves the coupled
  labels); it goes into `SALCKey`. Edge cases pinned: n_spin = 0 ⇒ L_S := 0;
  n_spin = 1 ⇒ L_S := l_spin; p = 0 ⇒ L_S = Lf (= the v4→v5 map, §11).
- Production keeps the coupled-tensor `_project_and_fold` engine (combined
  ordering × path × Mf space) + `_canonicalize_members`; the existing
  gauge-fixing (`Q = V₁V₁ᵀ` + Gram–Schmidt) inherits the L_S block structure,
  so canonical SALCs come out with definite L_S. The DisplacementBases
  prototype (polynomial-composition slot matrices, cycle-wise character
  counting + plethysm) is **demoted to the independent test oracle**, after
  fixing its review findings (§12). Its solid-harmonic evaluator
  (`_solid_harmonics_impl!`) ports to production as the displacement kernel
  (renormalized per §3).

## 5. Sectors and truncation (D-4)

- The truncation spec is a **sector table** (union of sectors, not a product
  grid), mapping 1:1 onto the theory note §3/§7. Public noun:
  `Sector(; spin, disp, soc = true, cutoff)`. Sugar resolved once to a dense
  canonical form (BasisSpec discipline); dense storage channel-keyed. **The
  union is a set-union on `SALCKey`**: overlapping sectors (e.g. a soc=false
  sector inside a soc=true sector's L_S = 0 block) must not produce duplicate —
  hence exactly collinear — design columns; key uniqueness is an asserted
  invariant of the resolved spec.
- **`soc::Bool` is per sector**, and it is a *truncation rule at enumeration*:
  `soc = false` ⇔ "enumerate only L_S = 0 (the Σl_spin-even rule applying as
  always)". Exactness is theorem-backed (the SOC-less basis is the L_S = 0
  column subset of the one grey-group basis). The docstring must state BOTH
  conditions — L_S = 0 alone would admit the T-odd scalar chirality
  ê_a·(ê_b×ê_c) (L_S = 0, Σl = 3); it is the parity rule that kills it.
- Migration note: today's `isotropy = true` filters `Lf == 0`
  (`AngularMomentum.jl` / `coupled_bases`). That is exact for p = 0 only
  (L_S ≡ Lf); the joint filter must move to a path-level **L_S** test, or
  dJ/dr-type sectors (L_S = 0, Lf ≠ 0) would be wrongly killed. `isotropy`
  gets a deprecation error pointing at `soc = false`.
- `soc` (truncation axis, defines model support) ≠ `sector_mask`/`frozen`
  (staging axis, defines fit stages). Never merged. The L_S = 0 mask correctly
  freezes the L_S = 0 columns of `soc = true` sectors too.
- Sector selectors `:all / :soc_free / :soc / :spin / :lattice / :coupled` and
  presets `soc_free(spec)`, `force_constant_spec(crystal; ...)` are public API.
- Per-species knobs: `lmax = 0` (spin-inactive species), `pmax = 0` (clamped
  species) — ligands are `lmax = 0, pmax > 0`.
- `disp_scale` (u/d_NN scaling) is fixed in the basis and persisted (precedent:
  the (4π) rule). The constructor warns (not errors) on odd max total
  displacement degree — a *necessary* boundedness condition only; even degree
  does not imply the leading form is positive (for every spin configuration).
  True boundedness is a runtime concern: the displacement-radius guard (§8) is
  the honest defense; the Φ^(2)({ê}) min-eigenvalue monitor diagnoses
  reference-point dynamical stability, not polynomial boundedness.

## 6. Data and fitting layer (D-7)

- **`TrainingDatum`** (channels optional; `SpinDatum(...)` survives as a
  convenience constructor). Fields are per-configuration observables only:
  energy, directions/magmoms, displacements, forces, field/torques. **No
  strain field.** Future stress/occupations = versioned additive fields.
- **Force sign pinned once**: `f_i = −∂E/∂u_i` (Euclidean gradient, no
  projection), stated identically in the `TrainingDatum` field docstring, the
  force-design-matrix docstring, and the DFT-adapter contract — the torque-sign
  precedent (`−e×∇E` convention block) applies verbatim.
- **`DatumProvenance`** (constrained, torque_qualified, reference_id) is
  load-bearing; the `SLCEDataset` constructor pins ONE reference Crystal and
  asserts every datum's `reference_id` (the double-counting protocol as an
  invariant, not a docs warning). `torque_qualified` gates torque rows.
- Three-block co-fit (w_E, w_T, w_F); X_F is block-sparse by construction
  (p = 0 SALCs have zero force rows) — `_assemble_problem` scatters
  column-subset blocks, never materializing zeros.
- **ASR = exact linear equality constraints, null-space reparameterization
  inside `_assemble_problem`** (β = Z·γ; group penalties on β unchanged).
  Never a penalty (a violated ASR is unbounded-below energy); never relative
  coordinates; never basis-level recombination. Machinery pins (review):
  the constraint rows come from the translation generator D = Σ_i ∇_{u_i}
  applied to the explicit homogeneous polynomials, collected in ONE common
  unsymmetrized monomial basis across orbits (real M3 machinery, not a
  one-liner); A is symmetric-redundant ⇒ the null space comes from a
  **rank-revealing (column-pivoted) QR on row-normalized A with a pinned
  tolerance**, and rank(A) is gated against an independent count. Residual
  gates are *relative* (`‖Aβ‖/(‖A‖‖β‖)`), plus the well-conditioned physical
  form Σ_i f_i = 0 per configuration (§12 k).
- **ASR amendments (2026-07-26, three-lens design review — numerics /
  statistics / architecture; reasons recorded per the reopening rule):**
  1. *Placement split (amends "inside `_assemble_problem`"):* Z is a pure
     function of the basis; rebuilding it per assembly re-runs the symbolic
     expansion + factorization ~26× in a `select_support` sweep and ~n_folds×
     in `cross_validate`, and buries the rank gate in every fit. **Build once
     at `SLCEDataset` construction and store on `dataset.asr` (the
     `force_cols` discipline — carried by slicing/`vcat`); `_assemble_problem`
     only APPLIES it** and returns the reparameterization in its (extended)
     result so `refit`/GCV/λ-path consumers stay consistent. Never persisted:
     Z is factorization-gauge-dependent; β is gauge-invariant and remains the
     only stored coefficient object (fingerprint precedent — recompute, never
     trust).
  2. *Z gauge:* per-block **orthonormal** Z (identity on pure-spin columns —
     structurally skipped via a `nothing` sentinel, giving bitwise identity
     for pure-spin fits); difference-stencil gauges are the banned
     relative-coordinate parameterization by stealth. Orthonormality is
     load-bearing: it makes Ridge-on-β equal Ridge-on-γ verbatim and keeps
     the IRLS compressed systems (Z'·D(β)·Z, dense only on displacement
     blocks) well-conditioned.
  3. *Rank decision (amends the CPQR pin):* per-block **SVD** on
     row-normalized A with cut σ ≤ 1e-10·σ_max and a **forbidden band**
     (any σ ∈ [1e-12, 1e-8]·σ_max ⇒ error — ambiguous rank must refuse,
     never guess); CPQR abandoned entirely — the SVD spectrum IS the
     ambiguity diagnostic, and the cross-check role is filled by the
     independent numerically built A (amendment 4), not by a second
     factorization of the same matrix. Blocks realized as exact
     connected components of A (gated to respect the (spin content × total
     disp degree) grading, which D preserves/lowers by one). A is expressed
     in design-column coordinates (same `(4π)^{n_spin/2}` and future
     `disp_scale` scaling as the evaluator — a stated invariant with a
     finite-t consistency gate; the builder must read `spec.disp_scale`
     the day its ≠1 guard lifts).
  4. *Gate meaning (amends §12 k reading):* `‖Aβ‖/(‖A‖‖β‖) ≤ 1e-13` is
     architecturally guaranteed by the reparameterization (‖AZ‖ ~ eps) and
     passes even if A is WRONG — keep it as a smoke test only. The real
     acceptance tests are (i) rank(A) equality against an independent
     numerically built A (random-point evaluation through the production
     `accumulate_grad!` Gu column sum, SVD rank + subspace angles), with
     analytic counts on closed fixtures, and (ii) finite-t uniform-translation
     invariance + per-config Σᵢfᵢ = 0 through the production evaluator —
     run **after `fit` AND after `refit`/`select_support`**.
  5. *Support interaction (new mandatory item):* `refit` on a support S must
     re-derive the null space of `A[:, S]` — reusing the full-basis Z
     restricted to S is not a null space, and an unconstrained refit silently
     re-breaks translation invariance in the de-bias step (contaminating
     every `select_support` point). Homogeneity makes any support feasible
     (β_{S^c} = 0 ⇒ A[:,S]β_S = 0), but a support splitting a
     constraint-coupled column set can structurally zero survivors — legal,
     and surfaced loudly (warn + alive/cost recomputation).
  6. *Estimator scope:* the "group penalties on β unchanged" pin applies to
     quadratic/IRLS estimators only (weights evaluated at β = Zγ; solves in
     γ). L1/GLMNet under Z is a generalized lasso GLMNet cannot solve, and
     L1-on-γ is gauge-dependent and selects nothing physical — **explicit
     error when constraints are nontrivial**, message pointing at
     GroupAdaptiveRidge. `effective_dof`/`gcv` generalize by Cholesky
     congruence of Z'DZ (the diagonal-whitening shortcut is silently wrong
     under Z); parametric `dof` becomes q + 1 = p − rank(A) + 1; GCV's n
     excludes zero-weight rows.
  7. *Model provenance:* `SLCEFit` records the `asr` flag + achieved residual
     (refit/GCV re-assembly contract, like torque/force weights); `SLCEModel`
     and persistence record NOTHING — the public verifier
     `asr_residual(model)` recomputes ‖Aβ‖/(‖A‖‖β‖) from the basis, and the
     physical consumers (M4 force constants / dynamical matrix, MC joint
     ingest) gate on it. Hand-built violating models stay legal (the §12 k
     violation demonstration requires them). `fit(...; asr = true)` default
     ON (landing now is the only free moment — no joint users exist);
     `asr = false` is for the violation demo and ablations.
  8. *Staged-fit bridge:* the L_S-keyed `sector_mask` DOES straddle
     constraint rows (e.g. L_S = 0 and L_S = 2 columns of an l=1×l=1×p=1
     shell share monomial rows), but **if each stage is fitted under its own
     ASR, the next stage's constraint stays homogeneous**
     (A_frozen·β_frozen = 0 as a vector). The reparameterization type carries
     an affine slot (`beta_p`, ≡ 0 in this slice) so the frozen/sector_mask
     slice adds only the solvability check (−A_frozen β_frozen ∈
     range(A_free), else throw naming the straddling rows) and the
     `y ← y − X·beta_p` line.
  9. *Identifiability payoff (recorded for plan B):* with center-of-mass-free
     displacement sampling, on-site/self coefficients fall in null(X_F)
     without ASR (the Φ_ii = −Σ Φ_ij mechanism) and become exactly determined
     with it; recovery test plan B gains a rank-accounting gate (rank
     deficiency without ASR = rank(A) on the harmonic block; full rank q with
     it), and the fitter gains a standing `rank(X̃) < q` residual-flat-direction
     warning. Rotational (Born–Huang) invariances remain OUT of scope (docs
     must not claim ASR closes all exact invariances).

     *Amended 2026-07-26 by the measured plan-B ledger* (D4h fixture: p = 198,
     rank(A) = 135, q = 63, 9 pure-spin columns;
     `test/unit/test_identifiability.jl`). Under center-of-mass-free sampling
     the deficiency is **channel-dependent**, so "= rank(A)" was too strong as
     a blanket claim:
     - *torque only*: nullity = 135 = rank(A) **exactly**, rank = q — but this
       equality is a property of THIS basis, not a law. The torque channel
       measures ∂E/∂e, so it is blind to **all spin-independent content**; in
       this fixture every displacement-decorated SALC also carries spin
       factors, which makes "spin-free" and "translation-violating" coincide.
       The general rule (gated on a second fixture that adds a lattice-only
       `Sector(disp = …)`: p = 219, rank(A) = 150, q = 69, 21 spin-free SALCs
       of which 6 span ASR-feasible directions) is
       `nullity_torque = rank(A) + dim{spin-free feasible}` = 156, and the ASR
       does **not** restore full rank there — the constrained torque-only
       design keeps nullity 6. Only the *pair* of derivative channels does
       (nullity 0 under the ASR in both fixtures), which is why plan B fits
       both. Force constants from torque data alone are not identifiable, with
       or without the constraint.
     - *force only*: nullity = 63 < rank(A). `Σ_a f_a` **is** `−D E` evaluated
       at the sample (`f = −∂E/∂u`), so the force channel does see the
       violating content whose `D E` does not vanish on the slice (it sees 81
       of the 135 directions); what it cannot see are the directions whose
       gradient vanishes on the slice — and since every displacement-decorated
       SALC is homogeneous in `u` (no constant part), that is exactly the
       directions vanishing to second order there (`E ∝ |Σ_a u_a|²` and
       relatives). It additionally leaves the 9 pure-spin columns
       **identically zero** (no spin-only SALC has a displacement derivative).
     - *torque + force (the plan-B fit)*: nullity = 54 > 0 unconstrained,
       **0** under the ASR — and `rank([X; A]) = p`, i.e. the constraint rows
       supply exactly the information the data lack. Recovery is exact
       (‖β̂ − β‖∞ ~ 1e-15) with the constraint and lands 0.97 away from the
       truth without it, at machine-precision derivative residuals either way.
     - the deficiency is a property of the **sampling**: generic (drifting)
       displacements give full rank with no constraint at all. The ASR's
       payoff is therefore on realistic COM-free data, and above all in
       predictions **off** the sampled slice — the violating model reproduces
       every training derivative and even `Σ_a f_a = 0` *on* the slice, and
       breaks both the moment `u` drifts.

     The "standing `rank(X̃) < q` warning" is split, because an exact rank test
     costs `O(n·q²)` — the same order as the fit itself at l044 scale, which
     is not acceptable unasked: the standing part is an `O(n·q)` **dead-column**
     check in `fit` (per column, at the package's usual *relative* cut — an
     exact `iszero` test has a measured false negative, the `Σ_a u_a` columns
     sitting at ~1e-19 on COM-free samples), and the exact statement is the
     opt-in `identifiability(fit_or_dataset)` diagnostic (returns
     `(; ncols, rank, nullity, sigma_min, sigma_max, tol, gap)`; the dataset
     method answers before fitting; `gap` = smallest kept / largest dropped
     singular value makes an ambiguous rank decision visible, since unlike
     `_asr_nullspace` a diagnostic must report rather than refuse; the cut is
     `min(size)·eps` — `max(size)·eps` would grow with the row count while the
     `1/√n` whitening keeps the spectrum sample-size-independent).
- Hierarchical fit = `fit(...; frozen = ...)` offset + L_S-keyed
  `sector_mask`. The estimator layer (GroupAdaptiveRidge / GCV / select_fit /
  cross_validate) survives verbatim; `group_costs` gains a channel split.

  *Realized 2026-07-26 (M3 slice 6, `fitting/staged.jl`).* One affine
  `ASRReparam` carries the whole staging request — `Z` zero-rowed on frozen
  columns (a plain orthonormal selection matrix when there is no ASR),
  `beta_p` = frozen values + particular solution — so the assembly and solve
  paths are the ASR paths verbatim, and `SLCEFit.reparam` records which one ran.
  Amendments the implementation forced:
  1. *Homogeneity is a RELATIVE test.* "The frozen part satisfies the ASR" must be
     judged by the same `‖Aβ‖/(‖A‖‖β‖)` measure `asr_residual` reports (cut
     `1e-10`), not by `A·β == 0`: a stage fitted under the ASR leaves ~1e-16, and
     taking that for a violation sends every chained stage down the affine path —
     where a roundoff-sized right-hand side is generically outside range(A_free)
     and would be refused as infeasible. (Caught by the three-stage chain gate.)
  2. *Which masks actually straddle* (measured, D4h fixture, 180 constraint rows):
     the CHANNEL masks (`:spin`/`:lattice`/`:coupled`) straddle **zero** rows — A's
     rows are graded by (spin content, total displacement degree), so they couple
     only same-channel columns — while the `L_S` masks straddle **54 of 180**, as
     §12 (p) anticipated. Amendment 8's straddling concern is therefore real but
     specific: a channel-staged chain is homogeneous by construction, and the
     affine machinery earns its keep on `L_S` staging and on externally frozen
     models.
  3. *Penalties under an affine feasible set* shrink γ toward the particular
     solution, not toward zero (`‖γ‖ = ‖β − beta_p‖`). Unavoidable; the fit warns
     when a non-OLS estimator meets a nonzero free-column `beta_p`.
  4. *`dof` generalizes to the reparameterization's column count* (`p` → `q` →
     stage free parameters), and `refit` de-biases INSIDE the stage: the support is
     intersected with the movable columns (frozen coefficients were never fitted,
     so they cannot be thresholded away) and the constraint re-derived as a
     sub-stage. `select_support` rejects staged fits (its group-cost front assumes
     every column is selectable).

## 7. Downstream contract (D-5)

- Successor term type: **`DecoratedTerm` / `decorated_terms(model)`** — per-slot
  (channel, k, l) + slot→site map + folded tensor.
- **Consumer scale rule: `(4π)^{n_spin_slots/2}`** (4π is a spin-sphere-measure
  artifact; disp factors carry none — guaranteed by the §3 kernel
  normalization pin). Every consumer derives the scale from slot channels,
  NEVER from `length(atoms)`/`body` — both existing consumers
  (`TiledHamiltonian` ctor, SCETools MFA bridge) do exactly the latter today,
  which is why:
- **`multipole_terms(model)` throws an `ArgumentError` on any model containing
  a p ≥ 1 sector** (trigger = sector presence in the spec, not coefficient
  values); the message states the scale rule and names both hatches:
  `decorated_terms(model)` and `multipole_terms(restrict(model, :spin))`. On
  pure-spin models (any schema vintage) it returns today's result
  bit-identically — the frozen p = 0 view.
- `restrict(model, :spin)` = the exact clamped-ion sub-model (E at u = 0).
  **restrict ≠ refit**: its coefficients are the *reference-geometry*
  clamped-ion couplings — they contain no ⟨u⟩ renormalization and are not what
  a spin-only fit on relaxed structures would give. One boxed docs paragraph +
  comparison example is mandatory (§13 risk 5).
- `row_layout(model)` (per-channel row-offset table) is public contract.
- Deliverables tier (each with an exact-gate doctest; milestone homes in
  §14): `bilinear_terms(model; displacements)`, `force_constants(model;
  spins, order)` → `ForceConstantSet`, `dynamical_matrix`,
  `exchange_strain_derivatives`, `magnetoelastic_constants` (cubic ⇒
  `(; B1, B2)`; ε-linear part comes from the p=1 relative-displacement
  sectors via the two-site affine substitution — single-site p=2 contributes
  O(ε²) only, see gate (e) correction), `magnon_phonon_vertices` (docstring says "adiabatic"),
  `to_sunny(model; clamp = true)`. **Clamped-ion caveat (review):** the
  strain deliverables computed by direct ε-substitution are clamped-ion
  values; measured B₁/B₂ and elastic constants are relaxed-ion (internal-strain
  tensor Λ via the force-constant inverse). v0 ships clamped-ion WITH the
  docstring caveat — and the caveat rides in the RETURN VALUE
  (`(; B1, B2, ion = :clamped)`), not only in prose, so the numbers cannot be
  quoted without it: the clamped-vs-relaxed difference is routinely a factor
  ~2 and can flip signs, which makes a bare clamped-ion B₂ compared against
  experiment a wrong published number. The relaxed-ion correction is a later
  utility (§9d for how to invert Φ safely). Phonon-band tooling (phonopy
  writer) lives in SLCETools.
- **The B₁/B₂ OUTPUT convention is pinned, and it is not the strain measure**
  (review 2026-07-27). Measure-independence across the Seth–Hill family (§9e)
  covers only the ε-LINEAR content, and the actual B₁/B₂ ambiguities live
  entirely outside that family: tensor vs engineering shear (factor 2 on B₂),
  `α²` vs `α² − 1/3` (shifts B₁ against the volume-magnetostriction constant),
  `Σ_{i<j}` vs `Σ_{i≠j}` (another factor 2), the overall sign of `E_me`, and
  clamped vs relaxed ion. **None of these is pinned anywhere in the repo
  today** — gate (e2) tests span and independence and says so explicitly
  ("C2 is the fit gauge — no B₁/B₂ normalization or sign convention is pinned
  here"). So the measure pin must not be read as making B₁/B₂ safe. Pin, in
  the `magnetoelastic_constants` docstring AND as a numeric gate (§12 u):
  `E_me = B₁ Σ_i ε_ii(α_i² − 1/3) + 2B₂ Σ_{i<j} ε_ij α_i α_j`, TENSOR shear
  `ε_ij` (engineering γ = 2ε is I/O only), the `Σ_{i<j}` range, the overall
  sign, and the clamped-ion qualifier — gated against an INDEPENDENTLY
  hand-derived closed form on the (e2) fixture, never against the evaluator's
  own output (the (4π) precedent, §12 l).
- **Later-phase deliverable — multipole readout** (d0 §2e, added 2026-07-25):
  a recoupling post-processing (6j-class, never touching the projector) that
  decomposes each invariant into conjugate-irrep pairs
  (Ξ^spin_Γ, Ξ^disp_Γ̄) with coefficients c^γ_Γn, plus a mechanical
  (L, P, T) → Q/M/G/T SAMB labeling function — the "magnetic order → phonon
  response" selection-rule dictionary. Docs gain the free scope line: the
  static CE captures Q/G-type lattice multipoles; M/T-type (phonon angular
  momentum) is kinetic physics outside it.

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
  ⇒ RNG consumption unchanged ⇒ converted-model bit-identity (§12 b).
- `set_coefficients!(H, coefs)` (new machinery: fused coef-stream rewrite over
  coef-factored programs, ~1 sweep cost) — serves the strain outer move and
  doubles as the active-learning hot-swap hook.
- Strain: **outer-loop full-energy Metropolis move** over a K(ε) grid +
  elastic term — never inside the sweep layer (§9). **The target ensemble,
  written down (2026-07-27, discharging the "write it down first" pin).**
  Isothermal–isobaric (NPT), configurational (no kinetic term), fixed N, T and
  HYDROSTATIC P, cell parameterized by the canonical strain (§9e), atoms at
  fixed SCALED coordinates:

  > `w(ε, {s}, {ê}) ∝ V(ε)^{N_mobile} · exp[ −β( E(ε, u, ê) + P·V(ε) ) ]`,
  > `V(ε) = V₀ det(I + ε)`; the observable is `G(T, P)`.

  Three preconditions the `N ln(V′/V)` Jacobian silently assumes, each a
  classic bug if left implicit: (i) **the move must rescale the displacements
  affinely**, `u_i → (F′F⁻¹)·u_i`, together with the reference — at fixed
  Cartesian `u` the configurational measure is unchanged and the Jacobian is
  **1**, not `(V′/V)^N`; (ii) **N is the number of atoms with active
  displacement DoF**, not `n_atoms` — clamped species (pmax = 0 ligands) have
  no configurational integral, and a centre-of-mass-free sampler uses N − 1;
  (iii) the exponent is `N ln(V′/V)` for a proposal uniform in V and
  `(N+1) ln(V′/V)` for one uniform in ln V. Two docstring obligations:
  constant-strain (fixed cell) is a DIFFERENT ensemble giving `F(T, ε)` with
  neither Jacobian nor PV term — magnetostriction needs the former, and the
  constant-strain and constant-stress specific heats differ; and v0 is
  hydrostatic-only on purpose, because `P·V(ε)` is a state function with no
  measure ambiguity while general applied stress is work-conjugate to the
  strain measure (2nd Piola–Kirchhoff ↔ Green–Lagrange, Biot stress ↔ Biot),
  so a general `σ_ext` would re-open §9e part (1). Also: `j0` is per training
  cell and is normally dropped from the sweep energy as a constant — the strain
  move is precisely where it stops being constant, so `set_coefficients!` must
  carry `j0` and the strain `ΔE` must include `n_cells · Δj0`.
  **No double counting**: each grid point's fit has its own j0(ε) which already
  contains lattice elastic energy — the explicit elastic term (V/2)εᵀCε is used
  ONLY if j0(ε) is referenced out; pick one source and gate it (§12 l). The
  sharper half of the same hazard — whether j0(ε) also already contains the
  internal relaxation — is pinned by the reference-geometry protocol in §9a.
- Observable signature breaks ONCE to a struct view: `MCView` (spins, disps,
  strain, energy) + per-channel `counts`; accessors so the next field is not
  another break; configurational-only specific heat documented.
- Runtime guards: displacement-radius check (the boundedness defense, §5) +
  min-eigenvalue monitor of Φ^(2)({ê}) (reference stability), via observables.
- Checkpoint schema bump from v2 (disp, step_u, strain, per-channel activity,
  fingerprint over slot vectors), coordinated with SLCEDynamics; GPU: taller
  device feature matrix; the polynomial evaluator ports bitwise more easily
  than Zlm did, BUT the mixed-magnitude accumulation is new — spin rows are
  O(1) while a degree-4 disp row is ~|disp_scale|⁴ ≈ 10⁻⁸, so the M4 GPU
  re-measure MUST include a precision/conditioning check (float32 paths
  especially), and re-measure table memory before promising lattice sizes.
  Philox slot map gains disp draws (QTB precedent).

## 9. Strain and the K(ε) reference problem

A strained reference generically has a lower point group ⇒ naive per-grid-point
bases have different keys. The common-subgroup route is NOT viable for general
strain (the common subgroup over arbitrary ε is {E, (i)}; the basis explodes).
Settled resolution, split by physics:

- **(a) v0: K(ε) grids restricted to isotropic volume strain ε = ηI only** —
  full point group preserved. **Key-stability pins (review):** cutoffs must be
  expressed in units of the (strained) d_NN — with an absolute cutoff a
  neighbor shell crosses it as η varies and the key sets diverge; key-set
  equality across the grid is an asserted construction-time invariant of
  `StrainedModels`. `disp_scale` is frozen at its η = 0 value so coefficients
  are interpolated in ONE normalization. Then `model_at(sm, ε)` + MC
  `set_coefficients!` work as specced. Covers FeRh's ~1% volume jump.
  **Reference-geometry protocol (review 2026-07-27) — the sharper double
  count.** The geometry that defines `u = 0` at a grid point must be exactly
  the geometry `j0(ε)` represents. v0 pins **affinely scaled, internally
  UNRELAXED** references: `u = 0` is the affine image of the η = 0 reference,
  `j0(ε)` is the clamped-ion cold curve, the internal relaxation is supplied
  exactly once by the MC's own displacement DoF, and grid points compose. If
  the DFT grid points were computed with internally relaxed coordinates
  instead, `j0(ε)` would already contain that relaxation and the MC would
  relax a second time — elastic response too soft, magnetostriction wrong,
  **and every gate green**. Nonzero reference forces at η ≠ 0 are expected
  and are exactly the p = 1 internal-strain content (§1: reference forces are
  real physics). Related: a fixed plane-wave cutoff across a volume grid puts
  a systematically V-dependent basis-set error into `j0`, and `−∂j0/∂V` **is**
  the pressure driving the MC strain move (Pulay stress) — require consistent
  converged cutoffs across the grid and gate `−dj0/dV` against the DFT stress
  reported at each point (§12 s).
- **(b) Symmetry-breaking strain: an explicit global strain channel** (rank-2
  symmetric = l = 0 ⊕ l = 2, polar, T-even, translation-invariant, projected
  in the UNSTRAINED group) — keys ε-invariant by construction, ε-dependence
  polynomial in the basis functions. Needs the "global DoF" seat (a slot
  without a site) in the rep-provider interface (§3); the sibling of B₁/B₂.
  **Two stated holes (review), to be resolved before (b) is implemented:**
  (i) *rotational invariance* — point-group projection does not imply the
  Huang/rotational sum rules relating the strain block to the low-order force
  constants; v0 ships a rigid-rotation-residual diagnostic, not the
  constraint. **Correction (review 2026-07-27): the diagnostic that exists
  today does not close this, and must not be read as closing it.** Gate (e2)'s
  rotation kill (`test/unit/test_sectorbasis.jl:409-413`) feeds `u = W·d` with
  antisymmetric `W` — the LINEAR part of a rotation only. A rigid rotation is
  `u = (exp W − I)d`, and the second-order piece `½W²d = ω(ω·d) − |ω|²d` is a
  strain-like shell contraction that survives; the linear kill passes by the
  point-group argument `T_1g ⊗ (E_g ⊕ T_2g) ⊅ A_1g` that the test's own comment
  states, so it is vacuous as a rotational-invariance check. The diagnostic must
  be evaluated at FINITE ω, `u_i = (exp W − I)d_i`, and reported relative to a
  strain-energy scale (§12 q). What a rotation-blind measure would buy, stated
  precisely so the hole is not overclaimed either way: no rotation-carrying
  columns in the strain channel, an exact `l = 0 ⊕ l = 2` split with no spurious
  axial piece, a single-valued lattice-pair → strain map, and mixed `u`–ε
  rotational conditions that become *statable*. What it does NOT buy: the pure-Φ
  Born–Huang conditions, the mixed conditions themselves (statable ≠ enforced),
  or the Huang conditions at a stressed reference. Internal displacements rotate
  too, so `δE = 0` still demands `Σ_i (∂E/∂u_i)·W(d_i + u_i) = 0` for every
  antisymmetric `W`, and no strain measure touches that; (ii) *identifiability*
  — a homogeneous strain is the affine
  displacement field, so the strain channel overlaps the disp channel's span
  and `TrainingDatum` carries no strain observable; the channel is
  identifiable only from cross-reference energy differences (i.e. K(ε)-style
  data), and the fit protocol must say how the overlap is broken.
- **(c) Specific low-dimensional strain paths** (c/a, tetragonal): a
  common-subgroup grid is fine (subgroup stays D₄h-class) but must be labeled
  path-specific.
- **(d) Coefficient renormalization** (Masuki–Nomoto–Arita–Tadano IFC
  renormalization / SCP structural optimization, PRB 106, 224104 (2022) +
  PRB 107, 134119 (2023)): because the displacement basis is polynomial, the
  shift u → u₀ + δu is an exact finite linear map — **lower-triangular in
  total degree** (degree d maps into degrees 0…d; it is NOT
  degree-preserving), realized most robustly by exact algebra on the Cartesian
  homogeneous-polynomial form (the addition theorem alone does not close in
  the (k, l) labels without a Gaunt-type relinearization step). Consequences,
  stated precisely: the planned `effective_model(model; u0)` utility reaches
  symmetry-broken distorted phases with NO common-subgroup grid, but (i) its
  output lives in the **u₀-stabilizer subgroup's basis** (or an unsymmetrized
  decorated-monomial model) — reference-group SALCs cannot span it; (ii) the
  re-expansion generates degree-0/1 terms, so the effective model's u = 0
  point is NOT the clamped-ion spin model (it is the u₀-frozen one — that is
  the point); (iii) gate: `E_shifted(δu) ≡ E(u₀ + δu)` to 1e-13 relative on
  random (u₀, δu). Homogeneous-strain response extraction: the affine pattern
  u_i = ε·R_i is NOT cell-periodic and must never be fed through the periodic
  evaluator — it is extracted analytically, cluster by cluster, via the
  relative substitution u_i − u_j = ε·d_ij (this is what
  `exchange_strain_derivatives` does), and it is origin-well-defined ONLY
  under exact ASR (§6) — a testable coupling. **The mechanism, and what it
  obliges (review 2026-07-27):** the affine field is u_i = ε·(R_i + t), so an
  origin shift `t` adds the uniform translation ε·t to every displacement and
  `∂E/∂t = ε : (Σ_i ∇_i E)`. The extracted strain response is therefore
  origin-independent **iff** `Σ_i ∇_i E ≡ 0`, i.e. exactly `A·β = 0`. Two
  consequences: (α) the strain deliverables must **hard-error**, not warn, on
  `asr_residual(model)` above tolerance — for a violating model the answer is
  not inaccurate but *undefined*, the origin being unbounded — and the
  tolerance must be TIGHTER than the 1e-10 used elsewhere, because the strain
  path samples `Σ_i ∇_i E` weighted by |R_i| and so amplifies a residual by
  ~R_cut/d_NN; (β) the origin-shift invariance itself has no gate today and
  needs one (§12 t) — gate (e2)'s translation kill exercises only the shell
  with the centre clamped (pmax Fe = 0), so it never touches the ASR coupling
  this claim is about. The later relaxed-ion correction `Λ = −Φ⁻¹Ξ` inherits
  the same discipline: under exact ASR `Φ` is singular with a null space of
  exactly the three known translations, so project onto the acoustic complement
  using those analytically known vectors — never a numerical rank cut, and
  never a near-singular inverse of a not-quite-ASR `Φ`, where three near-zero
  eigenvalues get inverted. SCP-style thermal contraction
  (⟨uu⟩ → J_eff(T); ⟨Φ_spin⟩ → magnetic-state-dependent force constants) is
  the finite-T deliverable template. Caveat inherited: a distorted minimum
  must lie inside the truncated polynomial's validity region.

The K(ε) pipeline, explicitly: per grid point ε = ηI, build the strained
reference Crystal → basis (identical keys, d_NN-scaled cutoffs) → dataset (own
reference_id) → fit → entry in `StrainedModels`.

### 9e. Strain convention — PINNED 2026-07-27 (closes §13 risk 2)

Three parts. Only the first is a physics choice; the other two are the
discipline that makes it cheap to revisit.

**(1) The measure is Biot, `E^(1) = U − I`.** The canonical strain is a
symmetric `ε` with the deformation DEFINED as `F := I + ε`, hence `U = I + ε`,
`R = I`, and the affine substitution `u_i − u_j = ε·d_ij` is exact *by
definition* rather than a linearization. Say **Seth–Hill m = 1** in the docs,
never "infinitesimal strain": the family is `E^(m) = (1/m)(U^m − I)` with
`E^(0) = ln U` (Green–Lagrange m = 2, Biot m = 1, Hencky m = 0), and naming the
member settles every higher-order question at once instead of leaving
"infinitesimal, redefined" hanging. Biot beats Hencky here because the
substitution is then exactly LINEAR in the label: `exchange_strain_derivatives`
/ `magnetoelastic_constants` come off the monomial coefficients in closed form
with no `d exp` chain rule, and gate (e2)'s exactness assertion
(`test/unit/test_sectorbasis.jl:391`) stays literally true.

**Switch conditions to Hencky, written down now so this is not re-litigated
from scratch.** Move to `h = ln U`, `u = (exp h − I)·d`, when EITHER holds:
(i) *the grid stops being isotropic* (§9c c/a paths, or the §9b global strain
channel) — under Biot the labels stop composing additively, `tr(ε)/3 ≠
(1/3) ln det U` at O(ε²), the `l = 0 ⊕ l = 2` split stops being exactly
volumetric/isochoric, and the incremental-vs-total chain rule in
`exchange_strain_derivatives` becomes a live factor-injection site; Hencky
makes all four exact. (ii) *any O(ε²) deliverable is planned* — third-order
elastic constants, second-order magnetoelastic constants, or an MC sampling
strains large enough that the SHAPE of `J(ε)` matters and not just its slope at
0. The polynomial machinery is indifferent either way: SALC projection,
`solid_harmonic_poly`, the ASR builder and `force_constants` all see a plain
Cartesian `u` and never the strain label, so the switch costs one symmetric 3×3
`exp` per strain change plus the `d exp` chain rule in the derivative
deliverables.

**(2) ONE object is stored — the symmetric right stretch `U`. Every label is a
derived function.** `ε := U − I`, `V := V₀ det U`, `η_log := (1/3) ln det U`,
and the Voigt vector are functions, never independently persisted fields.
Three independent reasons, each sufficient: `η_log` carries only the volumetric
part and so cannot label a §9b or §9c grid point (it does not survive v0);
`tr(ε)/3 ≠ η_log` at O(ε²) but agrees EXACTLY on the v0 isotropic grid, so a
two-label design hides its own inconsistency until §9c lands; and a mistaken
conversion in the MC acceptance is not rounding-error-sized (a 16³ cell at
η = 3% misplaces the Jacobian by ~5 kT, i.e. a factor ~250 in acceptance).
Deriving everything from `U` is also what makes the Biot→Hencky switch in (1) a
one-function change.

`η_log` survives only as the **interpolation abscissa** of the volume grid — an
internal, documented modelling choice, neither a label nor persisted. It is the
better variable on physics grounds (`J ~ V^{−n}` is near-linear in `ln V`), and
it is a genuine CHOICE rather than a coordinate change: a spline in η is a
different function from a spline in `ln(1+η)` pulled back. Gate it on its own
terms — leave-one-out on the grid against a directly fitted extra point — and
not by appeal to the label.

**(3) Voigt is an I/O view only**, order `1..6 = xx, yy, zz, yz, zx, xy`,
engineering shear factor 2 (`γ₄ = 2ε₂₃`), applied to strain and never to
stress; compliances then carry 2 and 4. Never canonical storage. For the day
stress becomes a training target (§10 radar): **VASP prints stress as
`XX YY ZZ XY YZ ZX`**, a different order — the adapter reorder is a bug waiting
in SLCETools, not a hypothetical.

**Persistence** (§11). Required fields, no `get(d, …, default)` fallback, an
unrecognized value an `ArgumentError` naming the known tags:
`strain_convention = "biot-v1: eps symmetric, F = I + eps"`, `voigt_order`,
`voigt_shear_factor`. No strained artifact exists yet, so a mandatory tag costs
nothing and needs no back-read branch — the only moment that will ever be true.
The second tooth matters more than the tag: store each grid point's actual
strained lattice vectors (`_crystal_doc` already writes them) and **recompute**
the label from the geometry on load, comparing at a tight tolerance. That is
the package's own recompute-never-trust discipline (fingerprints, `asr_residual`)
and it turns a mistagged *or untagged* file from silent absorption into a loud
failure, which a tag alone cannot do. It also bounds the blast radius of a wrong
pin: the coefficients were fitted to geometries, never to labels, so a convention
error is a re-derivation and never a re-fit.

**The derive-from-geometry helper is where rotation can enter, and the only
place.** `strain_between(ref, deformed)` must polar-decompose `F = A′A⁻¹ = R·U`,
keep `U`, discard `R` explicitly, and REPORT `‖R − I‖` instead of silently
symmetrizing. An externally standardized cell is the live case: spglib's setting
differs from ours by an arbitrary rotation, and `crystal_fingerprint`
deliberately treats a symmetry-transformed description as a different reference.
Writing `ε := sym(F) − I` naively injects a spurious isotropic compression
`θ²/2` — at θ = 5° that is 3.8e-3, the same order as FeRh's volume jump. Guard
`eigmin(I + ε) > tol` on every construction and every MC proposal; assert
symmetry with a RELATIVE `isapprox` and **throw** rather than symmetrize, since
silent symmetrization is exactly how a caller's rotation disappears.

**What symmetrizing does and does not discard.** `∂J/∂ε` is symmetrized because
strain *is* the symmetric part; the antisymmetric response is a different
quantity (rotational magnetoelasticity) and it is not lost information — under
SOC the rotation law `𝓡_U E = −𝓡_S E` (§6, and the theory paper) fixes it to
minus the axial response of the spin channel. So it is a **testable identity,
not a truncation**, and gate (r) below turns it into one. Note the O_h fixture
of gate (e2) cannot see it: cubic symmetry forbids l = 2 single-ion anisotropy,
so `𝓡_S E = 0` there and the residual vanishes for a reason that does not
generalize. The identity has teeth only on a fixture with l = 2 anisotropy
(D₄h-class).

## 10. Extension contract (D-9: occupation channel, ratified)

Spin–lattice ships first. The occupation (chemical-disorder) channel stays a
natural later phase iff these five things are shaped NOW (no OCC code today):

1. The rep-provider channel interface (§3) — OCC = orthonormal point functions,
   permutation-matrix symrep, T-even, SO(3)-trivial (never enters the CG tree).
2. Channel-tagged (channel, a, b) label triples in SALCKey, persist v5, and the
   term contract, with OCC enum RESERVED and SPIN < DISP < OCC fixed — adding
   OCC later changes **no byte of existing models or fingerprints** (v6
   additive; **old MC checkpoints stay valid**).
3. The distinct-(site, channel) slot invariant — the receptacle for the future
   indicator-gated ι_A(σ)·Z_l(ê) conditional-spin form.
4. Channel-enum row layout with a shipped offset table.
5. Species-keyed spec knobs (lmax/pmax per species; allowed-species lists join
   the same vocabulary).

Deferred with no back-propagation risk: the conditional-DOF measure convention,
semi-grand-canonical MC, the datum `occupations` field, eCE embedding, MIQP
ground-state constraints. Positioning vs CASM (nearest scope competitor) and
the technology radar (dipole long-range block for polar insulators — REQUIRED
before any oxide-magnet application; stress as a training target;
MLFF-generated displacement snapshots as joint-channel training data; Bayesian
UQ; one-hot OCC encoding): see `spin-lattice-ce-d0.md` §2b/§2c. Two
verification debts before any paper quotes the CASM comparison:
torque/constrained-DFT fitting path in CASM? noncollinear-magspin production
applications?

## 11. Persistence

Schema **v5**: decor table + L_S in keys; factors + slot→site maps; spec
sectors + soc + disp_scale. The strain fields land with M5 as specified in §9e
(`strain_convention` / `voigt_order` / `voigt_shear_factor`, required with no
default fallback, plus the per-grid-point strained lattice vectors from which
the label is RECOMPUTED and compared on load). No strained artifact exists yet,
so they are mandatory from birth and need no back-read branch. **Transparent
back-read of v4 in the loader, NO migrate-once tool** — the v4→v5 map is total
and value-preserving (per-site `l` → `SiteFactor(SPIN, k = 0, l)` decor,
`L_S := Lf`, `isotropy → soc = false`). Gate: v4-loaded models reproduce
predictions AND MC program arrays bit/byte-identically. Fingerprint recomputed
on load, never trusted. Drop v2/v3 branches only after confirming no files
circulate.

## 12. Verification battery

Migration gates: (a) relabel bit-identity v4→v5 (evaluate/grad bit-identical;
MC program arrays byte-equal — exercised at M2 through the *unmigrated* MC's
pure-spin `multipole_terms` path, and re-run after M4); (b) spin-only MC
bit-identity (l044 8³ fixed seeds, incremental-energy + config hash,
multitask + GPU); (c) l044 predictions oracle via serialized full-digit
fixtures (no co-loading); (d) span equivalence old vs p = 0-restricted new
basis; (i) u = 0 bitwise degeneracy.

New-physics exact gates: (e) cubic single-site l=2×p=2: exactly 2 invariants
(the O_h count E_g⊗E_g ⊕ T_2g⊗T_2g. **Correction 2026-07-25, paper
cold-read:** these are NOT the ε-linear B₁/B₂ — the affine relation is a
two-site difference, so a single-site p=2 monomial enters the strain
expansion only at O(ε²). The ε-linear B₁/B₂ live in the l=2 spin ×
neighbor-shell p=1 relative-displacement sectors; their affine substitution
u_i − u_a = ε·d_ia yields B₁/B₂ as shell sums, same closed form and
origin-independence as dJ/dε. The count 2 transfers because it is an
irrep-content statement); (e2) cubic l=2 × neighbor-shell p=1: the ε-linear
contraction reproduces exactly the two-constant cubic magnetoelastic form —
this, not (e), is the B₁/B₂ gate; (f) SOC-less pair+ligand p=1: dJ/dr + ligand term, NO DMI — **on a
mixed spec** (soc-false coupled sectors coexisting with a soc-true sector);
(g) count ≡ projector rank (prototype oracle) incl. permuting-pair AND
centrosymmetric mixed cluster (spin l=1 × disp l=1 ⇒ 0 invariants — the
kill-shot for the axial-vs-polar rep). **Gate-(g) conditions (review):** the
0-count holds only when inversion is in the *cluster stabilizer* — place the
mixed cluster on an inversion-symmetric site/bond (i mapping the cluster to a
different orbit member leaves a nonzero antisymmetric invariant killed only by
T-parity); and the case is reachable only by invoking the oracle with
*explicit slots bypassing the parity filter* (production never enumerates it).
(g2) chirality–twist invariant (ê_a×ê_b)·(u_a×u_b) on a two-site bond:
detected as a grey-group invariant with soc = true (Σl_spin = 2, L_S = 1,
parity-even even on an inversion-symmetric bond) and ABSENT under soc = false
— the positive two-site mixed-channel complement to (g)'s kill-shot; optional
cross-oracle: MultiPie (CMT-MU) SAMB generation for the p = 2 sector count
(d0 §2e). (h) Sym² = 5+1, Sym³ = 7+3; (n) sector-mask ≡ soc-false-basis
equivalence,
also on a mixed spec; (o) per-channel inversion pin — spin: +I ∀l; disp:
(−1)^l I — as a test of the trait function `rep_scale` (the production
projector never applies det factors; it is correct via §4's parity theorem —
state this in the gate or it reads as contradicting §4), plus TWO mutation
teeth, stated as effective reps on the polar Wigner product (precision
2026-07-26: det^{Σl_spin}·det^{Σl_disp} ≡ det^{Σl_all}, so "reinstate a
global det^{Σl_all} factor" and "keep spin axial but treat disp slots as
axial" are the SAME wrong rule — one tooth fences both prose mistakes):
det^{Σl_all}·⊗D_polar (every slot axial) must fail (f) AND the (g)
kill-shot, and det^{Σl_disp}·⊗D_polar (spin polar × disp axial) must fail
(f), both (g) fixtures being blind to it; (p) **L_S
block-diagonality**: `‖P[block_i, block_j]‖ < tol` for L_S_i ≠ L_S_j — cheap,
and the silent-failure fence under every L_S claim (soc, sector_mask,
hierarchy).

Derivative/constraint gates: (j) forces ≡ −∂E/∂u finite differences (plus the
existing torque FD); (k) ASR: relative residual `‖Aβ‖/(‖A‖‖β‖)` ≤ 1e-13,
rank(A) vs independent count, uniform-translation invariance gated *relative
to the largest intermediate term* at a physically sized t, and Σ_i f_i = 0 per
configuration; (m) spin-less limit vs the JPSJ 95, 053601 force-constant code
— **DROPPED by user decision, 2026-07-27.** It was the one gate resting on an
external oracle with no acquisition path in the repo family, and acquiring it was
never going to be worth the cost. Recorded here rather than deleted, because what
the drop costs has to stay visible: the displacement basis keeps every INTERNAL
gate — its dimension against the independent symbolic route ((g) count ≡ projector
rank, `CountingOracle`), its derivatives against the production evaluator ((j) FD
forces, (k) ASR residual + rank + finite-`t` translation invariance + `Σ_i f_i = 0`),
the Γ-restricted sum `Σ_R Φ(R)` against a finite-difference Hessian, and the
physical ASR signature that `D(0)` has exactly three zero eigenvalues when fitted
under the constraint and none without. What it no longer has is a **second
implementation** to agree with: nothing checks our spherical-tensor force-constant
convention against the community's, so a shared-convention error (a normalization,
an ordering, a factor absorbed consistently on both sides of our own FD gate) would
survive. Treat any future disagreement with published force constants as evidence
about the convention, not about the fit.

MC gates: (l) ΔE ≡ total-energy diff for spin/disp/strain moves (the strain
case also pins the j0(ε)-vs-elastic-term single-source rule, §8); serial ≡
parallel bitwise; checkpoint resume bitwise; `set_coefficients!` round-trip
byte-identical; the (4π)^{n_spin_slots/2} pin asserts an **independently
hand-derived value** (never the evaluator's own output) on a HAND-BUILT
mixed-channel term and a reduced-cell joint model — `reduce_cell` itself (raw
MultipoleTerm emission + q·|det M| census) migrates to `DecoratedTerm` with
mixed channels (§13 risk 3).

Prototype-oracle preconditions (from the DisplacementBases review): fix the
`_pair_swapped` exchange-sign blocker (per-slot swap flags consumed by both
`_slot_matrix` and `_cycle_character`; reject site_a == site_b; regression =
4-site/2-pair ⟨C4z⟩ case); add slot quantum-number validation; reject repeated
same-site slots; add the group-closure/integrality guard; fix the
`product_ops` "commuting groups" docstring (closure needs the spin subgroup
normalized by the lattice proper parts). **Scope restriction:** the oracle has
NO time reversal — on Σl_spin-odd inputs it happily returns T-odd invariants
(e.g. (ê×u)_z) — so oracle-vs-production count comparisons are valid ONLY on
Σl_spin-even inputs (or after teaching the oracle T-parity); it also lacks
translation folding, a strain slot, and spin-side Sym^p.

Strain gates (M5, added 2026-07-27 with the §9e pin):

- **(q) rigid rotation at FINITE ω** — `u_i = (exp W − I)d_i`, residual reported
  relative to a strain-energy scale. Replaces the linear-`W` kill as the §9b
  hole-(i) diagnostic, which it is not (§9b): the linear test passes by a point-
  group argument and never sees the `½W²d` contraction. Run it on a fixture
  where the linear kill is NOT symmetry-forced.
- **(r) the SOC rotation law as a two-sided identity** — `𝓡_U E ≡ −𝓡_S E` on a
  fitted model, on a D₄h-class fixture that HAS l = 2 single-ion anisotropy. Two
  reasons this is the right shape: it is the first implementation-level check of
  the theory paper's "the rotation rules are transferred, not absent", and it
  turns the antisymmetric part of the strain-derivative tensor from discarded
  content into a measured quantity. The O_h fixture of gate (e2) is blind to it
  — cubic forbids l = 2 anisotropy, so both sides vanish there.
- **(s) `−dj0/dV` vs the DFT stress** at each grid point — validates the
  interpolation, the single-source rule (§8) and the grid's basis-set
  consistency (§9a) in one test.
- **(t) origin-shift invariance of the strain deliverables** — recompute with
  every `R_i` shifted by a random `t` and require agreement; currently ungated,
  and the precondition that makes it hold is exact ASR (§9d). Pair it with the
  hard-error on `asr_residual` at the tighter strain-path tolerance.
- **(u) the B₁/B₂ output convention** against an independently HAND-DERIVED
  closed form (§7) — shear factor, summation range, trace subtraction, sign.
  Never against the evaluator's own output.
- **(v) strain-label round trip** — `U` → persisted geometry → recomputed label,
  plus the refusal surface: unknown convention tag, missing tag, and a
  geometry/label mismatch each error loudly (§9e).
- **(w) interpolation abscissa** — leave-one-out on the volume grid against a
  directly fitted extra point. The abscissa is a modelling choice, not a
  coordinate change, so it needs a gate of its own (§9e part 2).

## 13. Residual risks and mitigations

1. K(ε) symmetry breaking — RESOLVED by §9 (volume-only grid with key-stability
   pins / global strain channel with its two stated holes / path-specific
   fallback / renormalization with the corrected closure statement).
2. Strain convention unpinned — **RESOLVED 2026-07-27 by §9e**: Biot
   (Seth–Hill m = 1), one stored object `U` with every label derived, Voigt as
   an I/O view, a mandatory persisted tag AND a recompute-from-geometry check
   on load. Residual exposure after the pin is NOT the artifacts (coefficients
   were fitted to geometries, so a wrong pin is a re-derivation, never a
   re-fit) but (i) published constants and (ii) deliverable docstring
   semantics — which is why §7's B₁/B₂ output pin and gate (u) carry more of
   the risk than the measure choice does.
   **Measure-independence is narrower than it looks, and the elastic-constant
   case never closes.** ε-linear content (B₁/B₂, dJ/dε, the internal-strain
   response, the reference stress) is Seth–Hill-independent unconditionally.
   Second-order elastic constants coincide across measures only at a
   STRESS-FREE reference, differing otherwise by a term linear in the reference
   stress. In a spin–lattice model that reference stress is a function of the
   spin configuration, so a cell can be stress-free for at most one magnetic
   state — and the flagship deliverable is the magnetic-state DIFFERENCE of
   force constants. Therefore: record the measure AND the spin state with any
   elastic-constant output, and never argue "the reference is stress-free so
   it does not matter". Everything strain-quadratic and beyond (third-order
   elastic constants, second-order magnetoelastic constants, the interpolated
   J across the grid at O(Δη²)) is measure-dependent by construction.
3. `reduce_cell` scale migration — the emission and census migrate to
   `DecoratedTerm`; covered by the extended (4π) gate (§12 l).
4. soc-vs-sector_mask drift — gates (f)/(n) MUST run on mixed specs.
5. restrict ≠ refit category error — boxed docs paragraph + comparison example.

## 14. Milestone plan

- **M0 — rename** (§2, incl. the Tier-1 type renames and the full checklist).
  Mechanical only; gate = all suites green locally + on kugui.
- **M1 — displacement evaluator + counting oracle.** Port
  `_solid_harmonics_impl!` (+ gradient) into the basis layer **renormalized to
  the 4π-free convention** (§3), with unit gates (Zlm cross-check restated
  with the explicit factor, l ≤ 16; u = 0 exactness); port the prototype
  counting utility into test-support with the §12 blocker fixes + scope
  restriction; wire gate (g) infrastructure.
- **M2 — channel-first basis core.** SiteFactor/SiteDecor + rep-provider
  interface; SALCKey with L_S; sector-table spec + per-sector soc + key-union
  invariant; `_enumerate_ls` → Σl over spin slots only; `_project_and_fold`
  generalized to mixed channels; persist v5 + v4 back-read. Gates
  (a)/(d)/(e)/(f)/(g)/(h)/(i)/(n)/(o)/(p). **Status 2026-07-26: complete —
  every M2 gate (incl. (e2)/(g2) and the M2d batch (d)/(f)/(n)/(o)/(p) +
  spec-level (a)/(i)) runs in the unit suite; see SPEC.md "M2d verification
  gates" for the per-gate homes.**
- **M3 — data + fit.** TrainingDatum/DatumProvenance/dataset pinning (force
  sign pinned per §6); force design block + three-block co-fit; ASR null-space
  (common-monomial-basis collection + rank-revealing QR); hierarchical
  frozen/sector_mask. Gates (c)/(j)/(k).
- **M4 — downstream contract + MC.** `DecoratedTerm` + scale rule +
  `multipole_terms` throw + `restrict`; deliverables that need the term
  contract: `bilinear_terms`, `force_constants`/`ForceConstantSet`,
  `dynamical_matrix`, `to_sunny`;
  SLCEMonteCarlo joint programs, disp move, MCView, `set_coefficients!`;
  checkpoint bump; GPU port + precision/memory re-measure. Gates (b)/(l)
  + the (4π) pin + re-run (a).
- **M5 — strain + finite-T utilities.** Sliced, with the convention decision
  discharged up front (M5-0 done 2026-07-27):
  - **M5-0 — decision record.** The strain convention (§9e), the target
    ensemble (§8), the reference-geometry protocol (§9a), the B₁/B₂ output
    pin (§7), the ASR precondition (§9d), and gates (q)–(w). *Complete.*
  - **M5-1 — `effective_model(model; u0)`.** Convention-INDEPENDENT, so it
    could have started before M5-0 and can start now: subgroup-basis output,
    gate `E_shifted(δu) ≡ E(u₀ + δu)` to 1e-13 (§9d). One coupling only — it
    must REFUSE a strain argument, since the affine pattern u_i = ε·R_i is not
    cell-periodic and must never go through the periodic evaluator.
  - **M5-2 — clamped-ion strain deliverables.** `exchange_strain_derivatives`
    / `magnetoelastic_constants` / `magnon_phonon_vertices`. The true consumer
    of the convention. Name BOTH meanings of the strain derivative at a
    nonzero grid point — intra-model incremental vs grid finite-difference —
    and make their agreement an acceptance gate; it is the best end-to-end
    check the K(ε) design admits, catching a wrong shear factor, a mislabelled
    η and basis truncation in one test.
  - **M5-3 — `StrainedModels` + volume grid.** d_NN-scaled cutoffs (note BOTH
    cutoff surfaces scale: `BasisSpec.cutoff` and every `SectorRule.cutoff`),
    key-equality assertion, `disp_scale` frozen — assert the freeze now even
    though `disp_scale ≠ 1` is still refused, so it stays true the day that
    guard lifts. The invariant is a SIMILARITY statement (A_i = s_i·A_0,
    fractional positions unchanged ⇒ identical space group, clusters, orbits
    and `SALCKey` set), which is why it is stated in the linear stretch and
    not in any label.
  - **M5-4 — MC.** `set_coefficients!` (an M4 leftover, convention-free, and
    the longest independent pole — pull it forward and land it alongside M5-1)
    + the outer strain move + gate (l)'s strain half + the checkpoint strain
    field. **Blocker found 2026-07-27:** `SLCEMonteCarlo/src/hamiltonian.jl`
    prunes programs on coefficient VALUES (`filter(t -> t.coef != 0.0, …)` at
    :756/:781, and the `w == 0.0` skips in `_push_term_programs!`), so a
    coefficient that is exactly zero at one grid point and nonzero at another
    has no entry to rewrite — reachable, since sparse estimators and `refit`
    produce exact zeros routinely. The fix (gate the skip on `folded` rather
    than on `coef·folded`, keeping `sent_w` materialized) touches §12 gate
    (a)'s byte-equality contract; decide explicitly between a
    `keep_zero_terms` arm and a loud refusal in `set_coefficients!`.
  - **M5-5 — finite-T.** SCP-style ⟨uu⟩ → J_eff(T), ⟨Φ_spin⟩ →
    magnetic-state-dependent force constants (§9d template).
  Global strain channel (§9b) only after its two holes are resolved AND a felt
  need exists — and note §9b hole (i) is wider than the record said before
  Revision 2.
- Docs land with each milestone. `guide/migration.md` carries the dropped-noun
  table: **JointDatum, JointBasisSpec, SectorSpec, SectorRule,
  SectorConstraint/GreyGroup/SpinIsotropic, SiteLabel, JointTerm, spin_mode,
  isotropy** (each → its replacement). The flagship tutorials (B₁/B₂
  rederivation, dJ/dr no-DMI, force constants for Si) are part of M2–M5
  exit criteria, not an afterthought. The Si tutorial keeps its place; what
  it lost with gate (m) is the external comparison column — it gates on the
  internal battery above (FD Hessian, acoustic zeros of `D(0)`) instead.
