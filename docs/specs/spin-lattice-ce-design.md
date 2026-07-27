# Spin–lattice cluster expansion (SLCE) — design decision record

**Status: SETTLED (D0 closed 2026-07-25). This is the normative design.**
**Revision 1 (2026-07-25): amended after a three-agent review** (numerical /
d0-consistency / codebase reality). Amendments are integrated in place; the
review reports are archived in the session record. One post-D0 pin was made at
review time: the displacement-kernel normalization (§3, 4π-free), forced by the
scale rule already ratified in D0.

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
  docstring caveat; the relaxed-ion correction is a later utility. Phonon-band
  tooling (phonopy writer) lives in SLCETools.
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
  elastic term — never inside the sweep layer (§9). **Ensemble pins
  (review):** (i) the acceptance rule must carry the correct configurational
  measure for a cell change (the N ln(V′/V) scaled-coordinate Jacobian) and a
  +PV term if the target is constant pressure — write the target ensemble
  down before implementing; (ii) **no double counting**: each grid point's fit
  has its own j0(ε) which already contains lattice elastic energy — the
  explicit elastic term (V/2)εᵀCε is used ONLY if j0(ε) is referenced out;
  pick one source and gate it (§12 l).
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
- **(b) Symmetry-breaking strain: an explicit global strain channel** (rank-2
  symmetric = l = 0 ⊕ l = 2, polar, T-even, translation-invariant, projected
  in the UNSTRAINED group) — keys ε-invariant by construction, ε-dependence
  polynomial in the basis functions. Needs the "global DoF" seat (a slot
  without a site) in the rep-provider interface (§3); the sibling of B₁/B₂.
  **Two stated holes (review), to be resolved before (b) is implemented:**
  (i) *rotational invariance* — point-group projection does not imply the
  Huang/rotational sum rules relating the strain block to the low-order force
  constants; v0 ships a rigid-rotation-residual diagnostic, not the
  constraint; (ii) *identifiability* — a homogeneous strain is the affine
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
  under exact ASR (§6) — a testable coupling. SCP-style thermal contraction
  (⟨uu⟩ → J_eff(T); ⟨Φ_spin⟩ → magnetic-state-dependent force constants) is
  the finite-T deliverable template. Caveat inherited: a distorted minimum
  must lie inside the truncated polynomial's validity region.

The K(ε) pipeline, explicitly: per grid point ε = ηI, build the strained
reference Crystal → basis (identical keys, d_NN-scaled cutoffs) → dataset (own
reference_id) → fit → entry in `StrainedModels`. **The strain convention
(metric, Voigt order, shear factor) is pinned once and persisted before the
first strained artifact exists** (§13 risk 2).

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
sectors + soc + disp_scale (+ strain convention when it lands). **Transparent
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

## 13. Residual risks and mitigations

1. K(ε) symmetry breaking — RESOLVED by §9 (volume-only grid with key-stability
   pins / global strain channel with its two stated holes / path-specific
   fallback / renormalization with the corrected closure statement).
2. Strain convention unpinned — pin + persist before the first strained
   artifact (CASM alone ships three metrics; do not import ambiguity).
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
- **M5 — strain + finite-T utilities.** Pin the strain convention;
  `StrainedModels` + volume grid (d_NN-scaled cutoffs, key-equality
  assertion) + the single-source elastic-energy rule + outer MC move (target
  ensemble written down first); `exchange_strain_derivatives` /
  `magnetoelastic_constants` (clamped-ion, documented) /
  `magnon_phonon_vertices`; `effective_model(model; u0)` (subgroup-basis
  output, §9d gate). Global strain channel (§9b) only after its two holes
  are resolved AND a felt need exists.
- Docs land with each milestone. `guide/migration.md` carries the dropped-noun
  table: **JointDatum, JointBasisSpec, SectorSpec, SectorRule,
  SectorConstraint/GreyGroup/SpinIsotropic, SiteLabel, JointTerm, spin_mode,
  isotropy** (each → its replacement). The flagship tutorials (B₁/B₂
  rederivation, dJ/dr no-DMI, force constants for Si) are part of M2–M5
  exit criteria, not an afterthought. The Si tutorial keeps its place; what
  it lost with gate (m) is the external comparison column — it gates on the
  internal battery above (FD Hessian, acoustic zeros of `D(0)`) instead.
