# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this package predates a tagged
release, so everything lives under *Unreleased*.

## [Unreleased]

### Added — M2d verification gates (test-only; closes the §12 M2 battery)

- **Gate (o)** — representation pins + mutation teeth for `rep_scale`:
  op-by-op identity `D_spin(l, R) = rep_scale(SPIN, det R, l) · D_polar(l, R)`
  against the CountingOracle's independently derived (polynomial-composition)
  slot matrices over all of O_h, the inversion specialization (spin `+I` ∀l,
  disp `(−1)^l I`) on both the oracle matrices and the production Wigner
  kernel, and TWO mutation teeth built from `rep_scale` itself, stated as
  effective reps on the polar product: `det^{Σl_all}·⊗D_polar` (every slot
  axial — identically the "global `det^{Σl_all}`" AND the "disp-as-axial"
  prose rules, since `det^{Σl_spin}·det^{Σl_disp} ≡ det^{Σl_all}`) and
  `det^{Σl_disp}·⊗D_polar` (spin polar × disp axial). Both collapse the bent
  pair+ligand count 7 → 5; the all-axial rule also resurrects the kill-shot
  (0 → 1) while the other is blind there; the 9-count bond (`Σl_spin` and
  `Σl_disp` both even) is blind to both. The production-relevant parity is
  the even-`Σl_spin` screen (`det^{Σl_spin} ≡ +1` — the theorem the no-det
  polar cache rests on), which is also why the two teeth coincide on every
  production-reachable label (`test_countingoracle.jl`, `test_mixedsalc.jl`).
- **Gate (e2)** — THE B₁/B₂ magnetoelastic gate: on an octahedral Fe(O)₆ unit
  under the full 48-op O_h group, the `l = 2` spin × neighbor-shell `p = 1`
  sector emits exactly 2 SALCs (`L_S = 2`, `Lf ∈ {1, 3}`); substituting the
  affine shell displacement `u_j = ε·d_j` lands them exactly in the span of
  the two-constant cubic magnetoelastic forms (`Σ_i ε_ii (n_i² − 1/3)` and
  `Σ_{i≠j} ε_ij n_i n_j`, both invariant directions realized independently —
  the B₁/B₂ normalization itself is a fit gauge, not pinned), while
  uniform shell translations and rigid rotations vanish identically — the
  ε-linear B₁/B₂ statement of the 2026-07-25 design-record correction
  (`test_sectorbasis.jl`).
- **Gate (p)** — `L_S` block-diagonality: the FULL grey projector (every
  `(L_S, Lf)` coupled path, no block filter) and every per-op representation
  matrix are numerically block-diagonal in `L_S` (< 1e-9 cross-block), and the
  full-space rank equals the engine's emitted SALC count — the fence under
  every `L_S` claim (per-sector `soc`, masks, hierarchy)
  (`test_mixedsalc.jl`).
- **Gate (f)** — SOC-less pair + ligand `p = 1` on a MIXED spec (soc-false
  ligand and bond-stretch sectors coexisting with a soc-true doubly-decorated
  pair sector): the ligand sector emits exactly the 1 superexchange-path
  invariant `(ê₁·ê₂)(u_O)_y` (oracle-pinned L_S-resolved counts 1/2/4), the
  dJ/dr bond-stretch sector exactly the oracle-pinned 2 SALCs spanning
  `(ê₁·ê₂)(u₁−u₂)_x` / `(ê₁·ê₂)(u₁+u₂)_y`, every `L_S = 0` SALC is invariant
  under global spin rotations at fixed lattice while every `L_S ≥ 1` SALC
  violates it (the operational no-DMI statement), and flipping the ligand
  sector to `soc = true` adds exactly the DMI-like `L_S ≥ 1` blocks leaving
  `L_S = 0` bitwise unchanged (`test_sectorbasis.jl`).
- **Gate (n)** — sector-mask ≡ soc-false-basis on a mixed multi-sector spec:
  masking the all-`soc = true` build to `L_S = 0` equals the all-`soc = false`
  rebuild bitwise (`test_sectorbasis.jl`).
- **Gates (d)/(a)** — dense ≡ `p = 0`-restricted sector build down to the fit
  and MC surfaces: bitwise basis identity plus `==` on the energy AND torque
  design matrices (span carriers), and field-for-field identical
  `multipole_terms` under identical coefficients (the unmigrated
  SLCEMonteCarlo consumption path re-exercised at spec level)
  (`test_sectorbasis.jl`).
- **Gate (i) full** — `u = 0` bitwise degeneracy at spec level: the pure-spin
  subset of a mixed build IS the dense spin basis (keys + members), joint
  evaluation at `u = 0` returns `===`-identical floats to the dense
  evaluation, and every displacement-decorated SALC is exactly zero
  (`test_sectorbasis.jl`).

### Changed — sector-table BasisSpec + `soc` (joint spin–lattice M2b-3a, BREAKING)

- **`BasisSpec` gains the sector-table (joint spin–lattice) form**: the new
  exported [`Sector`](src/slce/truncation.jl) sugar
  (`Sector(; spin, disp, soc = true, cutoff, nbody)`) declares one row of the
  truncation table — spin content as `nothing` / an explicit even-`Σl` rank
  multiset / a `(nbody, lmax, lsum)` truncation, displacement content as a
  total-degree budget (the `(k, l)` harmonic labels are derived), per-sector
  `soc` (`false` = the exact `L_S = 0` SOC-free column set), and a per-sector
  cluster cutoff. `BasisSpec(labels; sectors = [...], lmax, pmax, ...)`
  resolves the sugar once into the dense canonical `SLCE.SectorRule` form
  (public unexported); the admitted labels are the **union of the rows ∩ the
  global caps**. In sector mode `nbody` and the per-body `cutoff` envelope are
  derived, and `pmax` (per-species per-site displacement-degree cap; `0` =
  clamped, a ligand is `lmax = 0, pmax > 0`) is required as soon as a sector
  carries displacement content. New canonical fields: `pmax`, `sectors`,
  `disp_scale` (the fixed, persisted displacement scale in Å).
- **BREAKING — `isotropy` is replaced by `soc` (inverted!)**: the dense
  pure-spin form now takes `soc::Bool = true`; `soc = false` keeps only the
  total-spin scalar channel (`L_S = 0 ≡ Lf = 0` at `p = 0`) — exactly the old
  `isotropy = true`. The `isotropy` keyword (and the `[interaction].isotropy`
  TOML key) now throw an `ArgumentError` naming the replacement and the
  inversion. Rationale: the physical content of the filter is the SOC-free
  selection rule, which generalizes per sector in the joint expansion (design
  record §5); `Lf = 0` does not (`dJ/dr`-type sectors are `L_S = 0, Lf ≠ 0`).
- **Persistence**: the spec document now stores `soc`, `pmax`, `disp_scale`,
  and the resolved `sectors` table; legacy `isotropy`-keyed spec documents
  (v2–v4, and early-v5 local files) still read via the `soc = !isotropy` map —
  reloads are value-preserving as before.

### Added — sector-driven basis construction (joint spin–lattice M2b-3b)

- **A sector spec now builds an `SLCEBasis`**: `build_salc_basis(crystal, sg,
  clusters, spec; neighbors, selection)` (`src/basis/sectorbasis.jl`) expands
  the resolved sector table per cluster orbit — per-sector cutoff
  re-admission with `candidate_clusters`'s exact banded semantics, spin/
  displacement label generation (the `(k, l)` harmonic labels per total
  degree, married onto the sites with shared spin+disp sites), per-label
  effective `soc` = OR over admitting sectors — and projects the labels
  through the mixed-channel decor engine with per-species `lmax`/`pmax`
  permutation-orbit filtering. Key uniqueness of the sector union is asserted
  (the key-union invariant). `SLCEBasis` routes automatically: a sector-less
  spec keeps using the pure-spin engine bit-identically; a sector-expressed
  pure-spin spec reproduces the legacy dense build bitwise (gated, incl. the
  fingerprint).
- The spin-configuration `SLCEDataset` constructors refuse a
  displacement-decorated basis with a clear boundary error (the joint data
  layer — displacements, forces — lands in M3).
- `SLCE.rep_scale(channel, detR, l)` (public unexported): the declared
  per-channel O(3) representation scale relative to the polar Wigner matrix
  (SPIN axial `det(R)^l`, DISP polar `1`, OCC reserved). Production relies on
  the even-`Σl_spin` screen instead of applying it; the seat anchors the
  gate (o) representation pins (M2d) and the verification oracle.

### Fixed — M2b-3 review (numerical-reviewer + code-reviewer)

- The canonical-members `J/2` ground-truth bug was fixed in only one of four
  coupled sites: the same `J*0.5*Σ_members` expression (and matching torque
  gradient) also lived in `examples/persist_and_input.jl`,
  `docs/src/getting_started.md`, and `README.md` — all fixed, and
  `persist_and_input.jl` now self-gates its `J` recovery with an `@assert`.
- `bench/` and `README.md` were not migrated to `soc` and were entirely dead
  (the fixture forwarded the rejected `isotropy` keyword; `bench_nd2fe14b`
  read the removed field and its asset TOML used the removed key). Migrated
  with the inversion applied (`isotropy = true` → `soc = false`; the internal
  `build_salc_basis` kwarg reads `isotropy = !spec.soc`).
- `disp_scale ≠ 1.0` is now refused: the field is persisted and compared but
  no displacement kernel applies it yet, so accepting it stored a unit
  convention the tensors did not honour. The joint data layer (M3) wires it
  in and lifts the guard.
- An `:explicit` spin multiset whose rank exceeds every species' `lmax` now
  errors ("the sector would contribute nothing") instead of silently building
  an empty sector.
- `cartesian_positions` hoisted out of the per-orbit threaded loop in the
  sector builder; `BasisSpec`'s canonical constructor now defensively copies
  its container fields (matching `SectorRule`); the `Sector` docstring's
  derived-body-order example corrected; the design-notes input-schema
  narrative updated to `soc`. The oracle suite gains a sector-vs-dense
  bit-identity case under a real space group with per-species caps (the unit
  suite only exercises P1 — Spglib is absent there).

### Changed — decoration labels + SALCKey v5 (joint spin–lattice M2a, BREAKING)

- **`SALCKey` layout**: the sorted `ls::Vector{Int}` label is replaced by the
  joint decoration label `decors::Vector{SiteDecor}` plus the total spin rank
  `L_S` (`(body, orbit_id, decors, L_S, Lf, block)`). New `isbits` label types
  in `src/basis/decor.jl`: `Channel` (`SPIN < DISP < OCC`, `OCC` reserved),
  `SiteFactor` (constructor-validated per channel), `SiteDecor` (at most one
  factor per `(site, channel)` slot), with accessors `has_spin` / `has_disp` /
  `spin_rank` / `disp_degree` / `factors` / `is_pure_spin` and the relabel
  helpers `spin_decors` / `spin_ls` (all public unexported). The pure-spin
  construction emits `decors = spin_decors(ls)`, `L_S = Lf` — the total,
  value-preserving v4 → v5 map.
- **Persistence schema v5**: keys store `decors` + `L_S` instead of `ls`.
  v2–v4 documents load transparently (pure-spin relabel on read; predictions,
  fingerprint, and `multipole_terms` are unchanged — gated bit-identically in
  `test/unit/test_persist.jl`). No migration tool; `save` always writes v5.
- **`coeftable` columns**: `ls` → `decors` (pure-spin rows render as the old
  comma string, displacement factors as `u(k,l)`), new `L_S` column.
- `salc_groups` now groups on `(body, orbit_id, decors)` (same granularity;
  pure-spin bases produce identical groups).
- **Slot-based term layout (M2b-1)**: `SALCTerm` stores per-axis `SlotRef`s
  (member-site index + `SiteFactor`) instead of the per-site `ls` — the slot →
  site map that lets mixed-channel terms carry several axes on one site.
  Pure-spin terms are the identity slot list `spin_slots(ls)`; construction,
  canonical fold, evaluation/gradient kernels, persistence (v5 terms store
  "slots"; v2–v4 "ls" terms map on read), and `group_costs` all moved and are
  **bit-identical** on pure-spin bases (cross-checked against a basis built and
  serialized by the pre-refactor commit). `multipole_terms` now throws on a
  displacement-decorated basis (it is the pure-spin introspection surface).

### Added — mixed-channel SALC engine (joint spin–lattice M2b-2)

- **Decor projection engine** (`src/basis/salcbasis.jl`): `_orbit_salcs_decors`
  builds SALCs for explicit decoration labels (sorted `SiteDecor` multisets) on
  a cluster orbit — multiset arrangements grouped into site-permutation orbits,
  spin-first slot coupling with `L_S` read off each path (`_path_LS`),
  projection per `(L_S, Lf)` block (`_project_and_fold_decors`), transport in
  canonical slot order, Σl_spin-even validation at the label, and per-sector
  `soc::Bool` (`false` ⇒ `L_S = 0` only). Both channels rotate through the one
  polar Wigner cache (exact under the parity screen, design record §4);
  `l = 0` trace axes skip rotation (`D⁰ = 1`). The pure-spin production path
  is untouched; the "engines agree on pure spin" testset is the anti-drift
  gate coupling the two.
- **Joint evaluation** `evaluate_salc(salc, e, u[, scratch])`: spin axes
  `Z_{lm}(ê)`, displacement axes `|u|^{2k} R_{lm}(u)` (`SolidHarmonics`),
  scale `(4π)^(n_spin/2)`; displacement-decorated SALCs vanish exactly at
  `u = 0` (`SALCScratch` gains the solid-harmonics buffer).
- Engine-level gates (`test/unit/test_mixedsalc.jl`, hand-assembled O_h /
  bond-d4h fixtures): gate (e) cubic single-site `l=2×(0,2)` = 2 and
  `l=2×(1,0)` = 0, with `soc = false` emptying the sector; gate (g) count ≡
  the CountingOracle's 9 on the doubly-decorated bond; gate (g2) the
  chirality-twist `L_S = 1` sector present with `soc = true`, absent under
  `soc = false`; gate (i) core `u = 0` degeneracy + pure-spin consistency
  (`===`); mixed space-group invariance (axial spins, polar displacements)
  and grey-group time reversal.

### Added — displacement kernel + counting oracle (joint spin–lattice M1)

- **`SolidHarmonics` submodule** (`src/basis/SolidHarmonics.jl`, public
  unexported): real solid harmonics `Rₗₘ(u)` as homogeneous Cartesian
  polynomials (regular/exact at `u = 0`) with Euclidean gradients, in the
  **4π-free Racah-type normalization** pinned by the joint design record
  (`Rₗₘ(û) = √(4π/(2l+1))·Zₗₘ(û)`; rank-1 factors are literally `y, z, x`).
  Batch + allocation-free in-place + single-`(l,m)` APIs. Unit gates: the
  explicit-factor cross-check vs `Harmonics.Zlm` to `l = 16`, homogeneity,
  exact `u = 0` values and gradients, central-difference gradients.
- **CountingOracle test-support module** (`test/support/countingoracle.jl`):
  the independent invariant-counting / Reynolds-projection oracle for
  decorated clusters (mixed axial spin and polar displacement slots), ported
  from the prototype with its review blockers fixed — per-slot pair-orientation
  swap flags consumed by both the representation matrices and the cycle-wise
  characters (regression: 4-site/2-pair cluster under ⟨C4z⟩), slot
  quantum-number validation, repeated same-(site, channel) slot rejection,
  and group-closure/integrality guards. Gate-(g) infrastructure wired in
  `test/unit/test_countingoracle.jl`: count ≡ projector rank on permuting and
  mixed-channel clusters, the axial-vs-polar inversion kill-shot
  (spin l=1 × disp l=1 ⇒ 0 invariants), the bent/collinear ligand case, and
  the chirality-twist decoration.

### Changed — BREAKING: package renamed SCEFitting.jl → SLCE.jl (SLCE family, M0)

- The whole family is renamed to the **spin–lattice cluster expansion (SLCE)**
  stem per `docs/specs/spin-lattice-ce-design.md` §2 (SLCE.jl /
  SLCEMonteCarlo.jl / SLCEDynamics.jl / SLCETools.jl). Package + module name
  changed; **UUID kept** (path-dev Manifests stay resolvable). Old model /
  checkpoint artifacts are unaffected (persistence schemas carry versions,
  not package names). Public Tier-1 types renamed: `SCEBasis → SLCEBasis`, `SCEDataset → SLCEDataset`, `SCEFit → SLCEFit`, `SCEPredictor → SLCEModel`; extensions renamed to `SLCEGLMNetExt`/`SLCESpglibExt`/`SLCESunnyExt`; internal `src/sce/` → `src/slce/`.

### Added — oracle fit-parity testset vs Magesty

- `test/oracle/`: an end-to-end **fit parity** testset — one shared EMBSET
  through both packages, bases with identical channel content (Magesty
  `body1 lmax = 2` + `body2 lsum = 4` ⟺ SCEFitting `lmax = [3]`,
  `lsum = [1 => 2, 2 => 4]`; SALC counts asserted equal), OLS at the
  single-block endpoints `torque_weight = 0` and `1`. Held-out energy
  predictions and the intercept must agree; held-out torques agree up to the
  **known global sign convention** (SCEFitting reports the physical/LL
  `τ = m × B`, Magesty reports `τ = −m (e × B)`; both flip target and
  predictor together, so the fitted function is identical). Closes the gap
  between the kernel-level oracle checks and "do the two packages return the
  same model".

### Changed — `SALCScratch`: allocation-free harmonic tables in the design hot loop

- The per-term `Z`/`∇Z` site tables of `evaluate_salc`/`accumulate_grad!` —
  previously fresh `Vector`s per (member, term) call — now live in a reusable
  internal `SALCScratch` workspace (dnPl buffer + pooled per-site tables,
  grown on demand). The design-matrix drivers and the predict paths thread
  one scratch per task/call; the `cache::Vector{Float64}` forms remain as a
  compatibility surface (wrapped into a scratch). Same calls in the same
  order ⇒ **bit-identical** design matrices (checked against a serialized
  pre-change reference, `nbody = 3` included). Bench (bcc-Fe 4³, 100
  configs, lmax 2): energy 543→464 ms and 767→230 MiB; torque 1176→906 ms
  and 1877→186 MiB.

### Fixed — review-pass hardening (whole-package review, 2026-07-18)

- `clebsch_gordan` now throws an `ArgumentError` for momenta beyond the `Float64`
  factorial range (`j1 + j2 + J + 1 > 170`) and on any internal overflow, instead
  of silently returning `Inf`/`NaN`. Far outside the validated small-`l` regime;
  in-range values are unchanged.
- `select_fit`: the selected row's `score`/`edof` (under `criterion = :gcv`) are
  now re-derived from the cold re-solve, like `n_alive`/`cost` already were, so
  `path.score[path.selected] == gcv(path.fit)` exactly. The selection decision
  itself still uses the warm path scores and is unchanged.
- `_bilinear_terms`: removed the dead `a > b` reverse-member transpose branch
  (canonical v4 members sort sites by `(atom, shift)`, so it was unreachable);
  a non-canonical member is now an internal error. Docstrings updated to the
  one-canonical-member-per-bond reality. No behavior change for real models.

### Added — cost-weighted group selection (`GroupAdaptiveRidge` + GCV + Pareto λ path)

- **`GroupAdaptiveRidge(column_groups, group_weights; lambda, epsilon, max_iter,
  tol)`** — in-core group extension of `AdaptiveRidge`: iterative reweighted
  ridge with one shared weight `wⱼ = v_g/(‖β_g‖² + p_g·ε)` per column group,
  approximating the weighted group-L0 penalty `λ·Σ_g v_g·1{β_g ≠ 0}` (at
  convergence a surviving group pays exactly `λ·v_g`). Degenerates exactly to
  `AdaptiveRidge` for singleton groups with unit weights; `lambda = 0` ⇒ OLS;
  `islinear` ⇒ `true`. Motivation: Monte-Carlo sweep cost is paid per
  contraction entry, and entries vanish only when a whole `(body, orbit, ls)`
  SALC group is zero — group-level elimination is the only sparsity that
  reduces MC cost.
- **Basis helpers** (public, unexported): `SCEFitting.salc_groups(basis)`
  (column → group labels by `(body, orbit_id, ls)`),
  `SCEFitting.group_costs(basis[, labels])` (per-group distinct-contraction-
  entry union over the canonical members; additive across groups), and
  `SCEFitting.cost_weights(basis; theta)` with `v_g = √p_g·(c_g/c̄)^θ` —
  `θ ∈ [0, 1]` tilts the penalty from cost-blind (`0`) to cost-proportional
  (`1`). Convenience constructor `GroupAdaptiveRidge(basis; lambda, theta, …)`.
- **`gcv(f)` / `effective_dof(f)`** (exported) — closed-form hat-matrix
  diagnostics for the linear estimators (`OLS` / `Ridge` / `AdaptiveRidge` /
  `GroupAdaptiveRidge`; adaptive members in the standard converged-weight
  sense): `effective_dof` = `tr(X(X'X+λD)⁻¹X') + 1`, `gcv` = `n·RSS/(n−df)²`
  on the assembled problem, `Inf` in the near-interpolating regime. The trace
  is an eigenproblem on the smaller Gram side — usable at `n < p`. `dof(f)`
  (the raw parametric count) is unchanged. On torque co-fits GCV is optimistic
  (correlated within-configuration rows); grouped CV is the documented ground
  truth there.
- **`select_fit(dataset, est; lambdas, criterion = :gcv|:cv, delta, …)` /
  `SelectionPath`** (exported) — warm-started descending λ path on a
  once-assembled Gram, scoring each fit by GCV or configuration-grouped K-fold
  CV (in core, deterministic seeded folds, per-fold Gram downdates), tracking
  alive groups (the `refit` scaled-magnitude support rule) and the predicted MC
  cost `Σ_{g alive} c_g`, then selecting the **cheapest λ within `(1 + delta)`
  of the minimum score** — the cost-aware generalization of `:lambda_1se`.
  Because the reweighted ridge crushes dead groups to tiny nonzero values, the
  default alive rule is a per-λ *relative* floor (1e-6 of the largest scaled
  magnitude); the effective absolute threshold at the selected λ is returned as
  `path.threshold`. The selected fit is re-solved cold (reproducible by a plain
  `fit`, its path row re-derived from that solve); follow with
  `refit(path.fit; threshold = path.threshold)` to de-bias exactly the reported
  support. `SelectionPath` is a Tables.jl source (one row per λ). Intended
  workflow: `GroupAdaptiveRidge(basis; theta)` → `select_fit` → `refit`,
  sweeping `theta` to trace the (MC cost, error) Pareto front.
- **`select_support(f; thresholds, delta, evalset, …)` / `SupportPath`**
  (exported) — the second knob: sweep the alive threshold at a fixed fit and
  trace the (cost, error) front of de-biased `refit`s (one OLS per point),
  scored on an evaluation dataset (pass a held-out slice) with the same
  Pareto rule. On real data the group-magnitude spectrum is continuous — no
  alive/dead gap for the λ path to expose — so this is where most of the
  cost–error trade is realized (validated on the production l044 model:
  38 % MC cost at a held-out torque RMSE better than the full model; 3 % at
  +19 %).
- **`cross_validate(dataset, estimator; torque_weight, nfolds, seed)` /
  `CVResult`** (exported) — generic configuration-grouped K-fold CV of any
  `fit` call: each fold refits from scratch (fold-local centering/whitening —
  nothing leaks across the split) and scores the held-out configurations in
  prediction space. Per-fold **and** pooled out-of-fold energy and torque
  RMSEs are reported independently of `torque_weight` (an energy-only fit
  still gets its torque error measured), alongside the fit's own
  `(1−w)·MSE_E + w·MSE_T` score. Deterministic seeded folds; `CVResult` is a
  Tables.jl source. Complements `select_fit(criterion = :cv)`, which whitens
  globally and only ranks a λ path.

### Changed — canonical SALC members (up to `N!`× smaller basis, persist v4)

- The SALC construction now folds its output into a **canonical, duplicate-free
  member form** (`_canonicalize_members`): the projection/transport still runs
  in the ordered-image space (where a stabilizer operation is a plain axis
  permutation — unchanged numerics), but the emitted members are normalized to
  one per physical cluster instance — sites sorted by `(atom, shift)`, shifts
  re-anchored to `shifts[1] = 0`, tensors axis-permuted and summed per site→`l`
  assignment. Previously every instance appeared once per site ordering (`N!`
  copies at `N` distinct sites), a construction-internal redundancy that leaked
  into evaluation. `Φ(e)`, gradients, fits, and SALC keys/fingerprints are
  unchanged up to floating-point regrouping (the merge pre-sums tensors that
  were previously summed after contraction); nothing is approximated.
  Measured on the production Nd₂Fe₁₄B `l044` model (nbody 3): 405,312 → 70,680
  multipole terms (5.73×), 4,392,744 → 744,636 tensor entries (5.90×) — the
  same factors apply to design-matrix/torque evaluation, model files, and
  downstream Monte-Carlo sweep cost.
- **Persist schema v4**: models/bases are saved in the canonical form (~6×
  smaller files for 3-body bases). v2/v3 documents remain readable — members
  are folded on load, so existing model files get the same speedups without a
  refit. Note the regrouping means energies of a reloaded pre-v4 model can
  differ from the previous build at the last-ulp level (bit-identical
  checkpoint resume across this version boundary is not preserved).

### Added — EMBSET reader (legacy Magesty training sets)

- `read_embset(path; n_atoms = nothing, zero_moment_atol = 1e-10)` and the
  `EmbsetFile` `AbstractDFTSource`, so a legacy Magesty training set drops
  straight into the pipeline: `SCEDataset(basis, EmbsetFile("EMBSET"))`. The
  format is code-agnostic (energy + per-atom moment and constraining-field
  vectors — exactly what `SpinDatum` stores), hence in-core rather than an
  SCETools adapter. Stricter than Magesty's reader on malformed input: the
  atom-index column must match the position in its block (Magesty silently
  ignores it), every numeric field must be finite (`NaN`/`Inf` — a failed SCF —
  is an error, not a silent training row), and every failure names the
  config/atom. More lenient on shape: block detection is token-based, so files
  without `#` separators parse (Magesty requires them). Cross-checked against
  `Magesty.read_embset` in the oracle suite.

### Added — dataset slicing / `vcat`, zero-moment guard

- `SCEDataset` now supports `length` (configuration count), configuration
  slicing `dataset[idx]` (integer vector/range, `Bool` mask, or `:`; duplicate
  indices allowed for bootstrap-style resampling), and `vcat` of datasets built
  on the same basis (checked by SALC fingerprint, so parts built from a
  persisted-and-reloaded basis concatenate; torque-bearing and energy-only
  parts do not mix). Design-matrix rows are sliced, never recomputed — the
  cheap path for train/test splits, filtering, and incremental data addition.
- `SCEDataset(basis, data::Vector{SpinDatum})` (and the `AbstractDFTSource`
  path through it) now **errors** when an atom referenced by the SALC basis
  carries a (near-)zero magnetic moment in some configuration
  (`zero_moment_atol = 1e-10` μ_B): such an atom previously entered the design
  matrix through `SpinDatum`'s placeholder `ẑ` direction and silently biased
  the fit. Unreferenced atoms (species removed with `lmax = 0`, sites outside
  every admitted cluster) are exempt. Matches Magesty's guard.

### Changed — BasisSpec truncation: per-body `lsum`, per-body × per-pair `cutoff`

**Breaking**: `BasisSpec`'s `pair_cutoff` keyword (and field) is replaced by
`cutoff`; passing `pair_cutoff` now throws with a migration hint (a scalar
`cutoff` is the exact equivalent). The `[interaction]` TOML key moves the same
way. Persisted documents bump to schema v3; **v2 files still load** (the legacy
scalar is expanded on read).

- `lsum` — a per-body-order budget on `Σl` over a cluster's sites (Magesty's
  `lsum`), as `lsum = [1 => 0, 2 => 4, 3 => 4]`, a scalar, or omitted (no cap).
  Enforced in the SALC `l`-tuple enumeration (`_enumerate_ls`), with the
  per-site ranges tightened to `lsum − (N − 1)` so oversized `lmax` values are
  never enumerated. This makes Magesty's l044/l064/l066-class models exactly
  expressible (`lmax` alone cannot cut `Σl`).
- `cutoff` — per body order **and** species pair: scalar, one pair table for
  all orders, or body-keyed (`[2 => Inf, 3 => ["Fe-*" => 6.0, "*-*" => 8.0]]`).
  Pair keys are unordered and resolve by specificity (concrete > `"A-*"` >
  `"*-*"`; equal-specificity conflicts error). The neighbor list is built at
  the element-wise max over orders and admits each pair against its own
  species-pair radius; `candidate_clusters` then checks **every** edge of an
  `N`-body cluster against that order's own radius (both image selections),
  with the `_SAME_DIST_RTOL` band applied per pair. `Inf` = no cutoff, `0`
  excludes a pair.
- `lmax` accepts label-keyed forms (`["*" => 3, "B" => 0]`) when the labels are
  supplied (`BasisSpec(labels; ...)` / `BasisSpec(crystal; ...)`); specs are
  stored resolved and dense (`lmax::Vector{Int}`, `lsum::Vector{Int}` with
  `LSUM_UNCAPPED`, `cutoff::Vector{Matrix{Float64}}`, `species_labels`).
  Unknown labels, uncovered species/pairs, and body orders outside `nbody` are
  errors (no silently-ignored sections); `display(spec)` prints the resolved
  truncation table.
- Gates: `test/unit/test_truncation.jl` — sugar/specificity resolution, the
  `lsum ≡ lmax` equivalences, per-pair neighbor-list and cluster admission vs
  an independent brute force, v3 + legacy-v2 persistence round-trips, and the
  new TOML forms. Validated on the production l044 model (Nd₂Fe₁₄B, nbody = 3,
  body2/3 `lsum = 4`) against Magesty's own fit.

### Added — allocation-free harmonics evaluation (cache-threaded variants)

- `Harmonics.Zlm_unsafe(l, m, u, cache)` and `Harmonics.grad_Zlm_unsafe(l, m, u,
  cache)`: passing a reusable `Vector{Float64}` workspace (length ≥ `l + 1`,
  contents irrelevant) hands it to LegendrePolynomials' `dnPl`, whose default
  argument otherwise allocates a fresh work vector on **every** call — previously
  the only allocation on these paths, and the dominant per-call cost of hot-loop
  consumers (SCEMonteCarlo's sweep kernels). Returned values are **bit-identical**
  to the cache-less methods, which are unchanged (gate: the NaN-poisoned-cache
  equivalence + zero-allocation testset in `test/unit/test_harmonics.jl`).
- The package's own hot consumers are wired through: `evaluate_salc` /
  `accumulate_grad!` accept an optional trailing `cache` (grown on demand inside
  the site-table builders), the design-matrix loops (`_design_energy` /
  `_design_torque`) hold one task-local workspace per threaded column, and the
  `predict_energy` / `predict_torque` SALC loops reuse one per call. Design-matrix
  cost roughly halves (`.claude/bench_log.md`): bcc Fe stress case `X_E` 1.96 →
  1.10 s and `X_T` 4.60 → 2.35 s; Nd₂Fe₁₄B (nbody 3, m 103) `X_E` 1.31 → 0.76 s
  and `X_T` 3.25 → 1.77 s. Values are bit-identical throughout (same summation
  order), so fitted coefficients do not change.

### Changed — bench suite: stress-scale defaults + Nd₂Fe₁₄B fixture

- The `bench/` scripts (which had broken on the `Interaction` → `BasisSpec` rename)
  are repaired and their defaults promoted from smoke sizes (16-atom bcc Fe, first
  shell) to the recorded seconds-scale stress baselines: 128-atom bcc Fe with
  multi-shell `cutoff = 6.0` Å (`bench_salcbasis` at `lmax = 3`, `bench_clusters` at
  `nbody = 3`, design/end-to-end at `m = 100` configs), with a `cutoff` positional
  argument added throughout. `bench_end_to_end` now also times `SCEDataset` assembly
  and the energy+torque co-fit.
- New realistic fixture: **Nd₂Fe₁₄B** (`bench/assets/nd2fe14b.toml`, `read_setup`
  schema — structure only, 68 atoms, 9 sublattice species, P4₂/mnm) with
  `bench/bench_nd2fe14b.jl` benching the full pipeline — basis build, design
  matrices, and OLS/Ridge energy+torque co-fits sized like the real training set
  (103 configs, 21 012 torque rows). Complements the high-symmetry bcc fixture with
  the few-ops / many-orbit, multi-species, non-magnetic-species regime.
- Baselines recorded in `.claude/bench_log.md` ("Stress baseline — 2026-07-14").

### Added — verification docs page (angular momentum)

- New **Verification** docs section (`docs/src/verification/angular_momentum.md`): a
  human-readable rendition of the `test/unit/test_angmom.jl` checks, recomputed at every
  docs build — a Clebsch–Gordan known-value table against closed forms (Varshalovich),
  orthonormality *and* completeness sweeps to `j ≤ 3`, the real Wigner-D functional
  identity on fresh directions (proper + improper), and the complex→real unitary closed
  form. Every section ends in an assertion, so a numerical regression fails the strict
  docs build; the unit tests remain the primary gate.
- The page closes with a worked example deriving the **tesseral CG coefficients for
  `l₁ = l₂ = 1`** from the complex ones via the package's own
  `coeff_tensor_complex` → `complex_to_real_tensor` pipeline: exact-form matrix
  displays (entries matched to `±p/√q`, unmatched throws), the `δ/√3` (Heisenberg),
  `ε/√2` (cross product, axial) and quadrupole identifications, and a proper-rotation
  equivariance gate against `wignerD_real`.

### Changed (breaking) — pre-registration API polish

- **The deprecated `Interaction` alias is removed** (it was a silent `const`, not a warning
  deprecation, and the package is unreleased — the window to drop it cheaply is now). Use
  `BasisSpec`.
- **`SCEBasis.interaction` → `SCEBasis.spec`** (`::BasisSpec`): the last remnant of the old
  name — reading `basis.interaction` and getting a `BasisSpec` back was the rename's ghost.
  Follows through `read_setup` (its named tuple now carries `spec`; the human-authored TOML
  section keeps its descriptive `[interaction]` name) and the persist schema (the basis
  document key `"interaction"` → `"spec"`, `schema_version` bumped to **2**; v1 files are
  rejected with a clean schema error).
- `BasisSpec` additionally validates `lmax` (nonempty, entries ≥ 0); `Ridge` validates
  `lambda` (finite, ≥ 0) in an inner constructor.

### Added — StatsAPI completion, `public` tiering, synthetic-predictor constructor

- `coeftable` and `islinear` now extend the **StatsAPI** generics instead of shadowing them
  (the same collision `coef`/`fit` already avoided — `using GLM`/`StatsBase` no longer
  clashes), and thin energy-block defaults `predict` (→ `predict_energy`), `residuals`
  (→ `residuals_energy`), and `r2` (→ `r2_energy`) are exported; the torque block keeps its
  explicit `*_torque` accessors.
- The public-but-unexported tier is now declared with the Julia **`public` keyword**
  (machine-checkable via `Base.ispublic` / Aqua) instead of a comment-only promise; `save`,
  `load`, `salcs`, `islinear`, `Harmonics`, and `AngularMomentum` join the declared list.
- **`SCEPredictor(basis, j0, jphi)`**: a public constructor filling `keys` from the basis,
  so synthetic models (hand-set couplings in tests, demos, downstream packages) no longer
  need the 4-argument form with `basis.salc_basis.keys`.
- `LICENSE` (MIT) and a CI workflow (`.github/workflows/CI.yml`: `TEST_MODE=all` on
  Ubuntu/macOS + a strict Documenter build) — the remaining registration blockers.

### Changed — layering and shared constants

- The bilinear / single-ion extraction (`BilinearTerms`, `_bilinear_terms`,
  `_l1_pair_matrix`, `_l2_onsite_matrix`, `_reconstruct_energy`) moves from
  `interop/sunny.jl` to **`sce/bilinear.jl`**: the public `bilinear_terms` introspection no
  longer depends on the interop layer (the layer inversion is gone); Sunny keeps only the
  primitive-cell unfold and `to_sunny`. Same code, same numbers.
- The tesseral constants `N1 = √(3/4π)`, `A2 = √(15/16π)`, `B2 = √(5/16π)` are defined once
  in `Harmonics` and consumed by the extraction (and by SCETools.jl's inverse mapping), so
  the forward/inverse conversions cannot drift. Bit-identical (same expressions moved).

### Fixed — torque-sign docstrings and test hygiene

- Three docstrings still carried the pre-Landau–Lifshitz torque sign
  (`SpinDatum.torques` in `io/dftsource.jl` — the file CLAUDE.md designates as the
  convention source —, `accumulate_grad!`, `SCEDataset`): all now state the actual
  convention, target `τ_a = m_a × B_a`, model `τ_a = −e_a × ∂E/∂e_a`. **Code was always
  correct**; the risk was a future edit "fixing" the code to match the wrong docs.
- New `test/unit/test_dftsource.jl`: the previously untested exported DFT data boundary —
  a closed-form `m·x̂ × B·ŷ = mB·ẑ` sign gate on the `SpinDatum` torque convention, the
  zero-moment placeholder branch, all validation throws, and a mock-source round trip.
- Shared test helpers (`rand_unit` / `rand_rotation` / `randcfg`) hoisted into
  `test/unit/testutils.jl` — the per-file copies overwrote each other in `Main` (warnings
  on every run; an edited copy would silently win). Bodies kept byte-identical, so all
  seeded fixtures draw the same stream.
- The diagnostics tests now check `rss`/`rmse`/`r2` against a from-scratch
  `y − predict_energy` reference instead of re-executing the accessors' own definitions;
  the oracle's Wigner-D comparison resolves the Magesty transpose convention once instead
  of accepting either per sample; dead `_connect` (superseded by `_connect_all`) deleted.

### Changed — source-tree reorganization (no behavior change)

- The 600-line `sce/model.jl` is split by responsibility: `sce/model.jl` keeps the
  pipeline **types** + constructors + config validation; `fitting/design.jl` the
  design-matrix assembly; `fitting/fit.jl` the `fit` / `refit` / `predict` logic; and
  `fitting/diagnostics.jl` the `coef` / `intercept` / residuals / R² / RMSE block.
- I/O is consolidated under `io/`: `persist.jl` and `input.jl` move there (joining
  `dftsource.jl`). The Sunny export moves to `interop/sunny.jl`, making the
  core/extension seam visible in the tree.
- The general Cartesian bilinear / single-ion extraction is renamed off the Sunny brand:
  `SunnyTerms` → `BilinearTerms`, `_sunny_supercell_terms` → `_bilinear_terms` (both
  internal/unexported), so the public `bilinear_terms` introspection no longer reads as
  Sunny-specific. `to_sunny` / `SunnyPrimitive` stay Sunny-named.

### Changed — diagnostics extend StatsAPI

- `coef`, `fit`, `nobs`, and `dof` are now **methods of the StatsAPI generics** rather than
  package-private functions, so they compose with the StatsBase / GLM ecosystem instead of
  clashing under `using` (`SCEFitting.coef === StatsAPI.coef`). Signatures are unchanged, so
  user code is unaffected. `intercept`, `refit`, `residuals_energy` / `residuals_torque`,
  and the `r2_* / rmse_* / rss_*` pairs stay package-specific (the two-observable energy +
  torque split has no single StatsAPI verb). Adds a lightweight `StatsAPI` dependency.

### Changed (breaking) — public API tiering and the `Interaction` rename

- **`Interaction` → `BasisSpec`.** The old name wrongly suggested a fitted coupling term;
  it is a basis/cluster *specification*. (An interim `const Interaction = BasisSpec` alias
  existed only within this unreleased cycle and is removed — see the entry above.)
  `BasisSpec` now validates in an **inner** constructor.
- **Export surface tiered.** The flat ~60-name export is split into the fitting workflow
  (still exported) and the *construction internals*, which are now **public but
  unexported** — reachable as `SCEFitting.build_clusters` / `SCEFitting.build_salc_basis`
  / `SCEFitting.evaluate_salc` / etc., but no longer dumped into `using SCEFitting`. The
  `SCEBasis` constructor already drives them, so typical user code is unaffected; advanced
  callers and tests qualify. Unexported: `build_neighbor_list`, `NeighborPair`,
  `NeighborList`, `interplanar_spacing`, `analyze_symmetry`, `n_ops`, `SymOp`,
  `SpaceGroup`, `build_clusters`, `ClusterMember`, `ClusterOrbit`, `ClusterSet`,
  `build_salc_basis`, `evaluate_salc`, `salcs`, `SALC`, `SALCKey`, `SALCBasis`,
  `solve_coefficients`, `AbstractTrainingDatum`. (Downstream `SCETools.jl` only uses the
  exported user API, so it is unaffected.)

### Performance — design-matrix / prediction hot path (bit-identical)

- `evaluate_salc` and `accumulate_grad!` now tabulate the per-site tesseral harmonics
  `Z_{lᵢμ}` (and `∇Z`) once per term and read them back in the multi-index loop, instead
  of recomputing them — and the expensive `dnPl` Legendre call inside — for every nonzero
  tensor entry. The per-term kernels sit behind a function barrier that specializes on the
  concrete tensor rank (`SALCTerm.folded` is stored as the rank-erased `Array{Float64}`),
  so the loop is type-stable. Same values and the same multiply/accumulate order ⇒ the
  output is **bit-for-bit identical** (verified against the Magesty oracle). Measured on a
  16-atom bcc 3-body, `lmax=2` basis × 200 configs (single thread): energy design matrix
  ~3.4× faster / 3.4× less allocation, torque design matrix ~5.1× faster / 4.8× less
  allocation. The same kernels back `predict_energy` / `predict_torque`.
- `build_salc_basis`: the coupled bases for each ordering are now built once and reused
  across every final `Lf` (the inner projection rebuilt the full chained-CG construction
  per `Lf`), and `_mfslice` returns a view instead of copying the multiplet slice. Both are
  bit-identical; modest build-time allocation reduction.

### Fixed

- `Lattice` now validates in an **inner** constructor, so the derived `reciprocal`
  (`inv(vectors)`, load-bearing for `interplanar_spacing` and the neighbor-list image
  range) can no longer be supplied independently via the auto-generated field constructor,
  and the singular-cell guard rejects **numerically degenerate** cells by a relative-volume
  threshold (`|det| > eps·‖A‖³`) rather than only exactly-zero volume.
- `build_salc_basis`'s docstring had detached and bound to the internal `_orbit_salcs`
  helper (it was inserted between the docstring and the function), leaving the exported
  `build_salc_basis` undocumented and breaking the `checkdocs = :exports` docs build. The
  docstring is back on `build_salc_basis`.

### Added — training-data boundary validation

- Spin configurations are now validated where they enter the pipeline (`SCEDataset`
  constructors, `predict_energy` / `predict_torque`): each must be `3 × n_atoms` with
  **finite, unit-norm columns** — the contract the harmonic kernels assume (they call
  `Zlm_unsafe`, which skips per-call checks). A non-normalized / NaN / wrong-shape config
  previously produced a silently biased design matrix or prediction; it now throws an
  actionable `ArgumentError` / `DimensionMismatch` naming the offending config and column.
  The `SCEDataset` constructors also check `length(configs) == length(energies)`. The unit-
  norm tolerance is the `atol` keyword (default `1e-6`). New `test/unit/test_validation.jl`.

### Added — thread-parallel SALC basis construction

- `build_salc_basis` now builds the cluster orbits in parallel (`Threads.@threads` over a
  flat orbit work list), each orbit producing its SALCs independently into a disjoint slot;
  the results are concatenated and sorted by `SALCKey` as before. The output is **byte-for-byte
  identical at any thread count** (verified: fingerprint *and* full folded-tensor content hash
  match across 1/4/8 threads) — orbits are independent and the final key-sort fixes the order.
- To make this race-free, the Wigner-D cache (`(l, g) → wignerD_real`) is now **precomputed
  serially** over the full, bounded `(l ≤ lmax, g ≤ n_ops)` grid and is **read-only** during
  the threaded loop (the previous lazy `get!` on a shared `Dict` would have raced). Speedup is
  allocation/GC-bound (the build is allocation-heavy): ~2.1× on 4 threads, ~2.4× on 8 for a
  128-atom FeRh 3-body basis (100 SALCs). Serial (1 thread) is unchanged.
- `test/unit/test_salc.jl` gains a determinism/thread-safety regression test (a rebuild must
  reproduce keys and folded tensors exactly), and `test/unit/test_nbody.jl` extends it to a
  **3-body, two-species, multi-term** orbit (`lmax_by_species = [2, 1]`, degenerate-multiset
  split) — the path where the Wigner cache bound and per-orbit `blockcount` matter most. The
  determinism / threading tests now `@warn` when run on a single thread (the threaded path is
  only exercised under `julia -t N>1`).

### Changed

- Internal cleanup following the threading refactor: dropped the now-dead `crystal` / `sg`
  parameters from `_project_and_fold` / `_transport_term` (unreferenced since the Wigner-D
  cache became a read-only `(l, g)` lookup). No behavior change.

### Added — thread-parallel design-matrix assembly and batch prediction

- The two design-matrix builders (`_design_energy`, `_design_torque`) and the vector
  `predict_energy` / `predict_torque` forms now parallelize over independent columns /
  configurations with `Threads.@threads`. Each task owns whole columns (or output slots),
  so writes are disjoint and the result is **identical at any thread count** (the per-atom
  gradient buffer `G` in the torque builder is now task-local). Speedup scales with
  `JULIA_NUM_THREADS` / `julia -t`; serial (1 thread) is unchanged. This is the cost that
  grows with dataset size — large active-learning batches and bigger supercells.
- New `test/unit/test_threading.jl` pins the threaded output to a race-free serial
  reference and to the independent scalar predict path (a data race would fail it under
  `julia -t N>1`).

### Added — `to_sunny` spin scaling routes (`:moment` / `:coupling`)

- `to_sunny` gains a `scaling` keyword so it can export a **non-half-integer** effective spin
  (e.g. the itinerant ``S_{\text{eff}} = m/(g\mu_B) \approx 1.1`` of bcc Fe), which Sunny's
  `Moment` (half-integer only) cannot carry directly:
  - `:moment` — put `S_eff` into the `Moment` and rescale exchange by `1/(SₐS_b)`; static
    energy *and* dispersion exact, but `S_eff` must be a half-integer (the previous, only
    behavior).
  - `:coupling` — keep `Moment` at a placeholder `s₀ = 1` and fold `S_eff` into the couplings
    (`J = M/(s₀√(SᵢSⱼ))`, single-ion `1/(s₀ Sᵢ)`); works for **any** positive `S_eff`. Only
    the magnon *dispersion* is physical (invariant under the overall spin scale `sᵢ→c sᵢ,
    J→J/c`); the static energy is rescaled.
  - `:auto` (default) picks `:moment` for half-integer spins, else `:coupling`. `mode` also
    gains `:auto` (`:dipole` for half-integer, `:dipole_uncorrected` otherwise). A `:coupling`
    placeholder cannot carry the quantum quadrupole, so single-ion + `:dipole` + `:coupling`
    is rejected. Modeled on the Magesty.jl Sunny export.
  - Validated: the `:moment` and `:coupling` dispersions agree to ``<10^{-7}`` for a
    half-integer spin where both apply, the `:coupling` static energy equals
    `(predict_energy − j0)/S`, and a non-half-integer `S_eff` now builds and disperses.

### Added — bcc Fe worked-example tutorial

- New tutorial [`docs/src/tutorials/case1_bcc_fe.md`](docs/src/tutorials/case1_bcc_fe.md):
  a real (non-synthetic) end-to-end fit for body-centered cubic iron — a 128-atom
  ``4\times4\times4`` supercell, isotropic two-body basis (``Im\bar3m``, 13 SALCs), fit from
  noncollinear spin-DFT energies and torques, with in-sample validation (parity plots),
  the isotropic ``J_{ij}`` read back out of [`bilinear_terms`](@ref) against pair distance,
  and a **live** [`to_sunny`](@ref) magnon dispersion (``\Gamma\text{–}H\text{–}N\text{–}\Gamma\text{–}P\text{–}H``
  and a ``\Gamma\text{–}N`` comparison to neutron data) using the new `:coupling` route at
  ``S_{\text{eff}} = 1.1``. The fit uses `torque_weight = 1.0` (a torque fit, constraining the
  energy gradient that sets the dispersion), reproducing the reference Magesty.jl tutorial's
  ``J_{ij}`` and magnon spectrum. The whole pipeline runs **live** in the docs (the ~10× `SCEBasis`
  speedup below makes the 128-atom build a few seconds). Ships the reference data — `POSCAR`,
  a 50-configuration `EMBSET`, and the experimental `febcc_spinwave.csv` — under
  `docs/src/tutorials/case1_inputs/`; the upstream VASP I/O and mean-field sampling that
  produced it live in the companion `SCETools.jl`. Adds `Sunny` to the docs environment.

### Performance — `SCEBasis` build (clusters + SALC)

- The symmetry-orbit and SALC construction were the dominant cost of building a basis on
  a large supercell (minutes for the 128-atom bcc Fe tutorial cell). Four numerics-
  preserving changes cut it to ~1.5 s (≈10×), and removed the GiB-scale transient
  allocations:
  - `build_clusters` reduces orbits by **growing each orbit from one representative**
    (`O(n_orbits · n_ops)`) instead of computing a canonical key per candidate
    (`O(n_candidates · n_ops)`); the canonical-key inner loop is now allocation-free
    (`Val(N)` static site tuples).
  - `build_salc_basis` connects all orbit members to the representative in **one
    `O(n_ops)` sweep** (was an `O(n_ops)` scan per member), and **memoizes the real
    Wigner-D matrices** `D^l(R_g)` by `(l, g)` across the build.
- Output is **byte-identical** (same orbits, same SALC keys/coefficients): verified by the
  full unit suite and the gauge-invariant Magesty oracle. Benchmarks and before/after
  numbers live in [`bench/`](bench/) and `.claude/bench_log.md`.

### Changed — public API naming consistency (BREAKING)

- A naming-and-usability pass renamed several exported symbols for consistency. All are
  **clean renames** (no deprecation aliases); update call sites accordingly:
  - **Count accessors** unified to an `n_*` prefix: `num_atoms` → `n_atoms`,
    `nsalc` → `n_salcs`. (`nobs` / `dof` are StatsAPI names and are unchanged.)
  - **SALC access** de-stuttered: the `SCEBasis` field `.salcs` (a `SALCBasis`) is renamed
    to `.salc_basis`, and a new exported accessor `salcs(basis)::Vector{SALC}` returns the
    basis functions directly — write `salcs(basis)[k]` instead of the old
    `basis.salcs.salcs[k]`.
  - **Predictor type** `SCEModel` → **`SCEPredictor`** (the lightweight, persistable
    predictor): the type, its constructor `SCEPredictor(fit)`, and `load(SCEPredictor, …)`.
  - **Setup reader** `read_input` → `read_setup` (returns the parsed crystal + interaction +
    symmetry setup; pairs with `read_configs` for training data).
  - **SALC kernel** `evaluate` → `evaluate_salc` (a less generic, collision-resistant name
    for the exported invariant evaluator).
- The persisted-model/basis **TOML schema is unaffected** (the on-disk tags `sce-basis` /
  `sce-model` and all doc keys are unchanged), so model and basis files written by an
  earlier build still load.
- Docs additions: the `SCEFit` (heavyweight, data-bearing) vs `SCEPredictor` (lightweight,
  persistable) roles are now contrasted in their docstrings, the README, and Getting
  Started; the `coeftable` columns (`body` / `orbit_id` / `ls` / `Lf` / `block` / `J`) gained
  a legend in the I/O guide.

### Added — lattice figures in the tutorials

- The Heisenberg-chain and kagome three-body tutorials now open with a generated lattice
  figure (CairoMakie) so the system is shown, not just described. Each figure draws the
  **unit (calculation) cell** and the periodic connectivity from the **actual** computed
  geometry — sites from `cartesian_positions`, bonds from `build_neighbor_list`, and the
  highlighted kagome triangle read back from `build_clusters` (its sites and periodic
  `shifts`). The chain figure shows the four sites closing into a ring (the dashed
  ``1\text{–}4`` bond across the cell boundary); the kagome figure shows the cell's three
  sites and a 3-body cluster that borrows two of its sites as periodic images from the
  neighbouring cell. The pictures cannot drift from what the basis is built on. `CairoMakie`
  is a `docs/` dependency only; the package itself gains no plotting dep.

### Changed — package renamed `MagestyRebuild` → `SCEFitting`

- The package, its module, and the repository directory were renamed from
  `MagestyRebuild` (`Magesty_rebuild.jl`) to **`SCEFitting`** (`SCEFitting.jl`), unifying the
  naming with the companion `SCETools.jl` under a shared `SCE*` family. The UUID is unchanged,
  so the package identity is preserved. The package extensions are now
  `SCEFittingGLMNetExt`, `SCEFittingSpglibExt`, and `SCEFittingSunnyExt`. The persisted-model
  TOML schema tags changed from `magesty-rebuild/sce-{basis,model}` to
  `scefitting/sce-{basis,model}` (`schema_version` is still `1`); any model TOML written by an
  earlier build must be re-saved. Downstream code updates `using MagestyRebuild` to
  `using SCEFitting`. The legacy `Magesty.jl` package (the design-reference original) is
  unaffected and keeps its name.

### Changed — VASP I/O moved to `SCETools.jl`

- The concrete VASP adapter (`SCEFitting.VASP`: `read_poscar`, `write_poscar`, `Oszicar`)
  has been **moved to the `SCETools.jl` package** (`SCETools.VASP`), joining the INCAR writer
  so all VASP I/O lives in one place. The core now keeps only the **code-agnostic DFT-data
  seam** — `AbstractDFTSource`, `AbstractTrainingDatum`, `SpinDatum`, `read_configs`, and
  `SCEDataset(basis, src)` — so the SCE pipeline stays code-agnostic. To read VASP training
  data, `using SCETools` and `SCETools.VASP.read_poscar` / `Oszicar` (the `SCEDataset(basis,
  src)` seam is unchanged). The `test/unit/test_vaspio.jl` unit tests, the
  `examples/vasp_dft_source.jl` example, and the VASP-vs-Magesty oracle cross-check moved to
  SCETools with it.

### Changed — sampling extracted into `SCETools.jl`; fitted-model introspection added

- The mean-field spin-configuration **sampler** (the P0–P4 work documented below) has been
  **moved out of this package** into the new auxiliary package `SCETools.jl`, which depends
  on `SCEFitting`. This package is now focused on building and fitting SCE models;
  generating spin configurations (and, later, active learning) lives in `SCETools.jl`. The
  removed exports are `AbstractSampler`, `MFASampler`, `MFASample`, `ExchangeModel`,
  `MultipoleField`, `sample`, `mfa_temperature_scale`, `mfa_sublattice_m`,
  `thermal_averaged_m`, and `tau_from_magnetization` (now exported by `SCETools`).
- **Added a code-neutral fitted-model introspection surface** so a downstream consumer reads
  the fitted Hamiltonian without reaching into the SALC-basis internals:
  - `multipole_terms(model)` returns a flat `Vector{MultipoleTerm}` — one record per cluster
    member / `l`-ordering of every SALC with a nonzero coefficient, carrying the raw `jϕ`
    coefficient (the per-N scale `(4π)^(body/2)` left for the consumer), the `body`, member
    `atoms` / `shifts`, per-site `ls`, and the `folded` tensor;
  - `bilinear_terms(model)` returns `(; pairs, onsites, skipped)` — the bilinear (`ls=[1,1]`)
    and single-ion (`ls=[2]`) channels as Cartesian `3×3` matrices, reusing the validated
    Sunny export conversion;
  - `num_atoms(model::SCEModel)` is a new method of the exported `num_atoms`.
  The gate is energy reconstruction (`test/unit/test_introspect.jl`): summing the per-term
  tesseral contraction reproduces `predict_energy − j0`.
- The tesseral spherical-harmonic submodule `SCEFitting.Harmonics` (`Zlm`, `lm_index`) is
  documented as a stable surface for downstream packages.

### Added — mean-field spin-configuration sampling: P4 (full multipole / many-body)

> The P0–P4 sampler entries below are retained as history; the code now lives in
> `SCETools.jl` (see the *Changed* entry above).

- **`MultipoleField` and `MFASampler(model::SCEModel; reference)`**: the full multipole
  mean-field sampler over **all** SCE clusters and harmonic orders (higher-order /
  many-body). The mean-field decoupling of any cluster term factorizes
  (`⟨∏ Z⟩ → ∏⟨Z⟩`), giving the generalized molecular field
  `h_a^{lm} = Σ_φ jφ·(4π)^(N/2)·folded · ∏_{b≠a} ⟨Z_{l_b}^{m_b}(e_b)⟩` — built by contracting
  each folded coefficient tensor against the *other* sites' multipole averages (the
  `accumulate_grad!` leave-one-out structure with the site-`a` harmonic left symbolic). The
  order parameters generalize from the magnetization to the **full per-atom multipole
  averages `⟨Z_lm⟩_a`** (`l ≤ lmax`), iterated to self-consistency by sphere quadrature; the
  `l=1` (bilinear) Perron eigenvalue sets `T_MF = ρ/3`, and the single-site Bingham /
  higher-multipole distribution is drawn with the Metropolis engine.
- `MFASampler(model::SCEModel; reference)` keeps every channel (bilinear, single-ion, and
  higher-order); `MFASampler(ExchangeModel(model); reference)` remains the bilinear-only
  truncation. Validated by the exact reduction to the single-global Langevin curve for a
  pure-bilinear model, by coupling-scale invariance, and by matching the single-site
  potential to the conditional mean SCE energy `⟨E | e_a⟩` of a biquadratic model (the
  many-body factorization check).

### Added — mean-field spin-configuration sampling: P3 (tensorial + single-ion)

- **Tensorial `ExchangeModel`**: `ExchangeModel` now carries the full bilinear tensor
  `bilinear[a,b] = S_ab` (Heisenberg + DM + anisotropic exchange) and single-ion anisotropy
  `onsite[a] = A_a` (the `ls=[2]` channel), with an `isotropic` flag for the P2 fast path.
  `ExchangeModel(model::SCEModel)` now extracts **all** of these (only the higher-order /
  higher-`l` SALCs are dropped); `ExchangeModel(Jiso; onsite)` and
  `ExchangeModel(bilinear; onsite)` are the raw constructors.
- **`MFASampler(exch; reference)` — tensorial path**: with DMI / anisotropic exchange or
  single-ion anisotropy the single-site potential `V_a(e) = β(e·g_a + e' A_a e)` (molecular
  field `g_a = Σ_b S_ab m_b ê_b`, `β = 3/(ρτ)`) gains an `l=2` Bingham factor that the
  closed-form vMF cannot represent, so the magnetizations are solved as `m_a = ⟨e·ê_a⟩` by
  sphere quadrature and the configurations are drawn with the **Metropolis** engine
  (per-atom chains, proposal scaled to the peak sharpness). The longitudinal molecular-field
  matrix `A[a,b] = −ê_a' S_ab ê_b` (so `T_MF = ρ/3`) generalizes the P2 isotropic case,
  which still takes the closed-form vMF path. **Noncollinear references** are supported
  (rigid cone axis, decision D2; a warning flags a non-stationary reference, e.g. DMI
  canting a collinear state). Because the single-ion enters with the same `β` as the
  exchange, the anisotropy persists above `T_MF` (an anisotropy-weighted paramagnet).
- An easy-axis single-ion sharpens the cone (and keeps order above the exchange `T_MF`); an
  easy-plane single-ion gives a girdle; the sampled `⟨Z_2m⟩` match the quadrature
  self-consistency. The shared sphere quadrature now caps its auto-sized node count (the
  large molecular fields of the `τ → 0` limit would otherwise blow it up).

### Added — mean-field spin-configuration sampling: P2 (multi-sublattice, isotropic)

- **`ExchangeModel`** (`docs/specs/mfa-sampling.md`): the neutral carrier of the isotropic
  bilinear exchange the mean-field sampler needs. `ExchangeModel(model::SCEModel)` extracts
  the symmetric `Jiso[a,b] = Σ_R J_iso(a,b,R)` from a fitted SCE by reusing the Sunny
  bilinear extraction and keeping the Heisenberg part `tr(M)/3` of each bond; the DMI /
  anisotropic, single-ion, and higher-order channels are dropped (P3/P4) and reported via
  `@warn`. `ExchangeModel(Jiso)` takes a raw symmetric matrix (an external-`Jij` reader is
  a P5 target).
- **`MFASampler(exch::ExchangeModel; reference)`**: the multi-sublattice sampler. The
  per-atom magnetizations `m_a(τ)` are solved from the coupled mean-field self-consistency
  `m_a = L(3(Ā m)_a/τ)`, where the molecular-field matrix `A[a,b] = −Jiso[a,b](ê_a·ê_b)`
  folds the reference directions in (so ferro / antiferro / ferri order all become
  ferromagnetic in the magnitude variables) and `Ā = A/ρ` is normalized by the Perron
  eigenvalue `ρ`, with `T_MF = ρ/3`. Distinct sublattices disorder at distinct rates under
  a single `T_MF`; because `Ā` is scale-free, only the coupling *ratios* matter — `m_a(τ)`
  is invariant under an overall coupling rescaling (decision D4). Each spin is then drawn
  from `vMF(ê_a, κ_a)` with the self-consistent per-atom concentration `κ_a = 3(Ā m)_a/τ`.
- **`mfa_sublattice_m(sampler, τ)`** returns the per-atom `m_a(τ)`. The coupled solve uses
  depth-1 Anderson acceleration (plain iteration suffers critical slowing as `τ → 1⁻`), and
  the construction verifies the reference is a stationary, sign-definite ordered state
  (warns on a non-stationary noncollinear or frustrated reference — the rigid-axis MFA is
  exact only for collinear references, decision D2). `MFASample.m` is now a per-atom vector
  per config.

### Added — mean-field spin-configuration sampling: P1 (single global, isotropic)

- **`MFASampler(reference)`** and the **`sample`** verb (`docs/specs/mfa-sampling.md`):
  the single global, isotropic mean-field sampler, building on the P0 single-site engine.
  Each spin is drawn from a von Mises–Fisher distribution `vMF(ê_a, κ)` about its reference
  direction, with one global concentration `κ = 3m/τ` fixed by the classical-Heisenberg
  mean-field self-consistency `m = L(3m/τ)` (`L` = Langevin function) in the reduced
  temperature `τ = T/T_MF`. Numerically equivalent to Magesty's `MfaSampling`, but with an
  explicit seeded `rng::AbstractRNG` and no Roots.jl dependency (a self-written bisection
  solves the monotone self-consistency).
- **`sample(sampler, n; tau | m, …)`** draws `n` configs at one control value;
  **`sample(sampler; tau | m, nsamples, …)`** sweeps a collection. Both return an
  **`MFASample`** (decision D1): `.configs::Vector{Matrix{Float64}}` plus parallel labels
  `.tau` and `.m`, iterable/indexable as its configs. Keywords `fixed` / `uniform` /
  `randomize` carry over from Magesty's `mfa_sweep`.
- **`AbstractSampler`** is the dispatch seam for the later model-backed samplers (P2+);
  **`thermal_averaged_m`** / **`tau_from_magnetization`** expose the self-consistency and
  its inverse; **`mfa_temperature_scale`** returns `T_MF` (decision D4 — `1.0`, reduced
  units, for the coupling-free global sampler).

### Added — `refit` and the regression-diagnostic accessors

- **`refit(f, estimator = OLS(); threshold = 0.0)`**: re-solve on the **support** of an
  existing fit — the de-biasing step that follows a sparse fit. A column survives when its
  scaled-magnitude contribution `|coef(f)[j]|·‖X[:, j]‖` exceeds `threshold` (`0` keeps the
  nonzero support exactly); coefficients off the support are zeroed and `j0` is recovered
  analytically. An empty support returns an all-zero `jϕ` (with a warning) and `j0 =
  mean(y_E)`. A `PrecomputedPilot` (or an `AdaptiveLasso` carrying one) is rejected — its
  fixed coefficient vector has the original column count, not the refit support length.
- **Diagnostic accessors**: `dof` (`length(coef(f)) + 1`), `rss_energy` / `rss_torque`
  (residual sums of squares), and `residuals_energy` / `residuals_torque` (the raw residual
  vectors). The energy and torque blocks are reported separately throughout, matching the
  existing `r2_*` / `rmse_*` split; `r2_energy` / `rmse_energy` / `r2_torque` / `rmse_torque`
  are refactored to build on the new `rss_*` so each metric has a single source of truth.
- Internal: the `(X, y)` centering / whitening / `groups` assembly is factored out of `fit`
  into `_assemble_problem`, shared by `fit` and `refit` so the two build identical designs
  (the oracle confirms `fit` is numerically unchanged).

### Added — adaptive / L0-approximating estimators

- **`AdaptiveRidge`** (in-tree, no extension): iterative reweighted ridge
  (Frommlet & Nuel 2016) that approximates an L0 penalty. It refits the analytic weighted
  ridge `(X'X + λ·Diagonal(w)) \ X'y` with `wⱼ = 1/(βⱼ² + ε)` until the relative ∞-norm
  change drops below `tol` (or `max_iter` steps), so large coefficients keep a light
  penalty and small ones are driven toward zero. `lambda = 0` reduces to `OLS`;
  `islinear ⇒ true` (a linear smoother in the converged-weight sense).
- **`AdaptiveLasso`** (type in core, GLMNet solve in `ext/SCEFittingGLMNetExt`): the
  one-shot Adaptive Lasso (Zou 2006). A `pilot` estimator (default `OLS`, any estimator
  allowed) supplies `β̂`, then a weighted Lasso is solved with per-column penalty factor
  `wⱼ = 1/max(|β̂ⱼ|, ε)^γ`. `gamma = 0` reduces exactly to a plain `Lasso`. It shares
  `ElasticNet`'s `lambda` / `standardize` / grouped-CV behavior (`lambda = nothing` selects
  λ by configuration-grouped CV with the adaptive weights held fixed).
- **`PrecomputedPilot`** (in-tree adapter): returns a fixed coefficient vector from
  `solve_coefficients` (length-checked against `size(X, 2)`), so an `AdaptiveLasso` can
  reuse a prior fit's `coef(f)` as its pilot instead of re-running a pilot regression.
- All three honor the column-centered `(X, y)` / analytic-`j0` contract. The GLMNet
  plumbing is shared between `ElasticNet` and `AdaptiveLasso` (an optional `penalty_factor`).
  Validated in `test/glmnet/` (support recovery, `γ = 0` ≡ plain Lasso, pilot
  pluggability, grouped-CV co-fit) and `test/unit/test_fit.jl` (core `AdaptiveRidge` /
  `PrecomputedPilot` solves, construction / validation, deferred-backend error).

### Changed — torque sign convention → Landau–Lifshitz / physical torque

- The per-atom torque is now `τ_a = −e_a × ∂E/∂e_a` (the physical / Landau–Lifshitz
  torque `m_a × B_eff,a`), matching the published *General spin models* paper
  (Phys. Rev. Research **8**, 023300 (2026)). Previously it was the energy-rotation-gradient
  `+e_a × ∂E/∂e_a` (the methods-paper Eq. 15 convention), the opposite sign.
- The change flips **both** sides together, so the fit is unchanged: `predict_torque` /
  `_design_torque` now compute `∂Φ/∂e × e = −e × ∂Φ/∂e`, and the DFT training target from a
  constrained OSZICAR is `τ_a = m_a × B_a` (was `−m_a × B_a`). Because the torque design
  matrix `X_T` and the target `y_T` both negate, the co-fit objective `‖X_T J − y_T‖²` is
  invariant — **`j0` and every coefficient `J` are identical**; only the *reported* torque
  sign changes. The energy fit, energies, and `predict_energy` are untouched.
- `predict_torque(model, config)` and the torque returned by `read_configs` / `SpinDatum`
  flip sign for downstream consumers. The finite-difference self-consistency gate
  (`test_torque.jl`, `test_nbody.jl`) and the oracle's analytic Heisenberg torque now check
  the `−e × ∇E` convention. Docs, examples, and the design-matrix convention notes updated.

### Added — Documenter.jl documentation site

- A full browser-viewable documentation site under `docs/` (Documenter.jl): home, a
  getting-started page, a four-part guide (building the basis, data and fitting,
  persistence and I/O, Sunny export), two narrated tutorials (Heisenberg chain, kagome
  three-body) with executed `@example` blocks, a three-part theory section (the SCE
  formalism, periodic resolvability, the architecture), and a complete API reference.
  Build with `make -C docs serve` (or `build` / `open`). Local-only for now
  (`remotes = nothing`, no `deploydocs` until a remote exists).
- **`docs` fixed several exported public APIs that silently had no runtime docstring** —
  a comment or a sibling definition sat between the docstring and the documented binding,
  so Julia never attached it. `coeftable`, `load`, `intercept`, `nobs`, `nsalc`,
  `rmse_energy`, `rmse_torque`, and the `AbstractDFTSource` / `AbstractTrainingDatum`
  abstract types now carry their own docstrings; every exported binding is documented
  (the API reference builds with `checkdocs = :exports`). No behavior change.

### Tested — N-body Wigner–Seitz cluster counting (`N ≥ 3`)

- Added `test/unit/test_ws_nbody.jl`, pinning the count of 3- and 4-body clusters on
  the Wigner–Seitz boundary against an **independent brute-force enumeration** of all
  compact clusters. The candidate set (for `N = 2, 3, 4`, including `pair_cutoff = Inf`)
  and the symmetry-orbit partition are checked to match exactly on cells deliberately
  seeded with face / edge / corner ties (cubic face-atoms, fcc, skewed hexagonal), and
  every emitted member is re-verified to have all `C(N,2)` edges at the minimum image.
- Documents and guards the **compact-cluster / third-edge** criterion: a cluster is
  admitted only when *all* its pairwise edges sit at their atom-pair minimum image
  simultaneously — individually minimum-image-resolvable pairs are not enough (the images
  minimizing `i–j` and `i–k` may push `j–k` onto a longer image). The regression includes
  an equal-spaced 1-D ring where every pair is minimum-image yet no compact triangle
  exists, and an over-/under-merge guard on the orbit reduction under a cubic point group.
  No behavior change — the enumeration was already correct; this makes it guaranteed.

### Added — GLMNet estimators (Lasso / elastic-net)

- **Estimator types in core** (`fitting/estimators.jl`): `ElasticNet(; alpha, lambda,
  standardize, nfolds, select, seed, nlambda)` and the `Lasso(; …)` convenience
  (`alpha = 1`). Like the Spglib/Sunny seam, the types live in the core package — named,
  validated, and dispatched on without the heavy dependency — while the actual solve
  lights up only under `using GLMNet`.
- **`solve_coefficients(::ElasticNet, X, y; groups)`** in the new
  `SCEFittingGLMNetExt` extension. GLMNet minimizes
  `(1/2n)·‖y − Xβ‖² + λ·[(1−α)/2·‖β‖₂² + α·‖β‖₁]` on the column-centered `(X, y)` the
  fit hands it, with `intercept = false` (so `j0` is still recovered analytically) and
  column `standardize` (the penalty acts per-column at `λ·std`, returning β on the
  original scale). `lambda = nothing` selects the penalty by K-fold cross-validation over
  GLMNet's automatic λ path — `select = :lambda_min` (lowest CV error) or `:lambda_1se`
  (sparsest within one SE); a numeric `lambda` fits at exactly that penalty.
- **Configuration-grouped, reproducible CV.** `fit` now passes per-row `groups` labels
  to `solve_coefficients`; for an energy+torque co-fit a configuration's energy row and
  all its torque-component rows share a label, so CV folds never split one configuration
  (which would leak within-configuration structure and bias λ selection). Folds are
  assigned deterministically by a seeded `hash` ranking — reproducible without taking on
  a `Random` dependency in the extension.
- The `AbstractEstimator` contract gains an optional `groups` keyword on
  `solve_coefficients`; `OLS`/`Ridge` accept and ignore it.
- Validated in a separate `test/glmnet/` environment (heavy Fortran-backed dependency,
  mirroring `test/sunny/` and `test/oracle/`): tiny-λ ≈ OLS, the analytic-`j0` centering
  invariant, CV support recovery and sparsity, `:lambda_1se` shrinkage, seeded
  reproducibility, ElasticNet ≠ Lasso, and an energy+torque grouped-CV co-fit. Core-only
  construction/validation and the deferred-backend error are covered in the main suite
  (`test/unit/test_fit.jl`).

### Added — Sunny.jl export (supercell + primitive-cell routes)

- **Conversion core** (`sce/sunny.jl`): turns a fitted `SCEModel` into the
  Sunny-representable channels — `ls=[1,1]` 2-body → a 3×3 exchange matrix
  (`M = (3/4π)·folded`, carrying Heisenberg / Dzyaloshinskii–Moriya / symmetric-Γ
  in one matrix) and `ls=[2]` 1-body → a traceless-symmetric single-ion tensor.
  `ls=[0…]` channels fold into `j0`; every other SALC (3-body+, higher `l`) is
  **skipped and reported**, since Sunny cannot represent it. The directed cluster
  members fold into one matrix per undirected supercell bond
  (`_sunny_supercell_terms`), and the whole conversion is gated **without Sunny** by
  reconstructing the energy (`_reconstruct_energy ≈ predict_energy − j0`).
- **`to_sunny(model; spins, g, mode, placement)`** in the `SCEFittingSunnyExt`
  extension (loaded by `using Sunny`): builds a real `Sunny.System` on the training
  supercell (P1, inhomogeneous), placing `set_exchange_at!` per bond (rescaled
  `J = M/(SₐS_b)`) and `set_onsite_coupling_at!` per atom. The system's classical
  energy reproduces `predict_energy − j0` exactly; exchange is independent of the
  spin length and Sunny mode, the single-ion term carries the
  classical/`:dipole`-quantum rescaling. Skipped channels are surfaced via `@warn`.
- **Primitive-cell unfold** (`placement = :primitive`/`:auto`, `_sunny_primitive`):
  recovers the chemical primitive cell from the space group's pure translations, groups
  supercell atoms into sublattices, and folds the supercell bonds onto one Sunny bond
  per **primitive** bond `(i, j, n)` (no multiplicity — Sunny's periodic replication
  restores it), for *unfolded* spin-wave dispersion. A `clean` flag detects when the
  model does not live on the primitive cell (interaction range reaching the supercell
  boundary), and the export falls back to the exact supercell route.
- Conversion math is a **core dependency-free** layer; only the `Sunny.System`
  assembly lives in the extension. Validated by `test/unit/test_sunny.jl` (Sunny-free:
  the `Z₁`/`Z₂` contractions, classification, energy reconstruction, the primitive fold
  and its clean detection, skip reporting) and the separate `test/sunny/` environment
  (the real `Sunny.System` energy vs the SCE energy across modes/spins for both routes
  — the primitive system reshaped back to the supercell reproduces the SCE energy,
  confirming the unfolded bonds and offsets — plus the skip warning and per-species
  spins).

### Changed — minimum-image periodic resolvability (Wigner–Seitz cell)

- **Periodic-image selection** (`geometry/neighborlist.jl`): `AbstractImageSelection`
  with `MinimumImage` (new default) and `AllImages`. Previously the neighbor list kept
  every periodic image within a *spherical* cutoff, which over-counts beyond `L/2`: a
  farther image of an atom carries the **same spin** as its minimum image, so the two
  interactions are not independently resolvable from a finite supercell (their design
  columns are collinear). `MinimumImage` keeps only the minimum-image, Wigner–Seitz-cell
  pairs — the physically resolvable set, which reaches the body-diagonal corner
  `(L/2,L/2,L/2)` at `√3·L/2`, **not** a sphere of radius `L/2` — with WS-boundary ties
  (faces 2-fold, edges 4-fold, corners 8-fold) kept as distinct members and `i==j`
  self-pairs dropped (same spin ⇒ a constant or a 1-body alias, never an independent
  pair). The image-box search is adaptive (provably sufficient on skewed / non-reduced
  cells). `AllImages` retains the old every-image behavior as the **generalized-Bloch /
  spin-spiral seam**, where `e^{iq·R}` resolves what one supercell cannot.
- **`pair_cutoff = Inf`** (`Interaction`) now means "every resolvable pair" — the whole
  WS cell — under `MinimumImage` (cf. Magesty's `-1` sentinel); `≤ 0` / `NaN` are
  rejected. `SCEBasis(...; images = MinimumImage())` and the `input.toml`
  `[interaction].images` / `pair_cutoff = inf` keys thread the choice through.
- **N-body min-image consistency** (`clusters/enumerate.jl`): a `MinimumImage` clique
  requires every edge at its atom-pair minimum-image distance and all atoms distinct, so
  a cluster is never built from an aliased bond or a reused atom image. The `AllImages`
  edge cutoff now uses the same relative tolerance as the tie band (was an absolute
  `+1e-9`).
- For a cutoff below half the smallest perpendicular cell width, `MinimumImage` and
  `AllImages` coincide (each in-cutoff image is already the minimum), so the existing
  physics is unchanged: the Heisenberg `J = 2√3·jϕ` recovery, the kagome 3-body co-fit,
  and the oracle all use cutoffs in this regime. Bit-for-bit Magesty agreement is not a
  goal; a single-atom cell's self-pair "bond" (1 orbit under Magesty/`AllImages`) is now
  0 orbits under the default `MinimumImage` — a deliberate refinement.
- Validated (`test/unit/test_imageselection.jl`): cutoff `< L/2` equivalence, the cubic
  8-fold corner and `L/2` 2-fold face ties, alias rejection above `L/2`, `AllImages`
  rejects `Inf`, hexagonal / skewed-cell search-box sufficiency, no-self-pair and
  distinct-atom clusters, a full-WS design matrix with full column rank under symmetry,
  `input.toml` `inf` + `images`, and a `pair_cutoff = Inf` persistence round-trip.

### Added — VASP I/O and a code-agnostic DFT-source seam

- **DFT-source boundary** (`io/dftsource.jl`): `AbstractDFTSource` +
  `read_configs(src) -> Vector{SpinDatum}`, the `SpinDatum` training datum (energy,
  `3×n` unit spin directions, moment magnitudes, constraining field, and the derived
  torque target `τ_a = −m_a × B_a`, eV), and `SCEDataset(basis, src | data; use_torque)`.
  This is the *only* thing the SCE pipeline consumes — the originating DFT code is
  irrelevant once you hold a `SpinDatum`/`SCEDataset`.
- **VASP adapter** (`io/vasp.jl`, `module SCEFitting.VASP`): `read_poscar` /
  `write_poscar` (POSCAR/CONTCAR ↔ `Crystal`; scaling incl. negative-volume,
  Direct/Cartesian, Selective dynamics, VASP4/5) and `Oszicar`, an `AbstractDFTSource`
  over constrained-noncollinear OSZICARs (energy `F=`/`E0`, `MW_int`/`M_int` moments,
  `lambda*MW_perp` field, SAXIS `Rz(α)·Ry(β)` rotation). Code-specific I/O is a
  **namespaced submodule** kept out of the core: the core and its export list do not grow
  as DFT codes are added; only the code-agnostic boundary is exported.
- A torque-carrying `SCEDataset` rejects all-zero torque targets (no constraining field
  found) so unconstrained data cannot be silently fit as "torque = 0".
- Validated (`test/unit/test_vaspio.jl`): POSCAR (Direct/Cartesian/VASP4/negative-volume/
  selective-dynamics/round-trip), OSZICAR (energy kinds, `mint`, SAXIS, multi-step,
  missing field, errors), the torque formula, and source → `SCEDataset`. The from-scratch
  POSCAR/OSZICAR readers are cross-checked **bit-for-bit against Magesty.jl**'s parsers in
  the oracle (synthetic files, since real VASP outputs are not vendored).

### Added — tabular coefficient output (Tables.jl)

- **`coeftable(f)` / `coeftable(model)` → `SCECoefficients`** (`sce/coeftable.jl`): a
  Tables.jl source — one row per SALC (`body`, `orbit_id`, `ls` as a comma string,
  `Lf`, `block`, `J`) — so the fitted coefficients drop into `DataFrame` /
  `CSV.write` / `Arrow.write`. The library owns the internal-storage → labeled-row
  mapping (the `J` column pairs with the basis keys positionally); the caller brings
  the table/IO package. `j0` is the intercept (`intercept(c)`), not a row. Tables.jl
  is a lightweight core dependency, the same seam that would later open tabular
  training-data ingestion.

### Added — persistence + TOML input files

- **Persistence** (`sce/persist.jl`): `SCEFitting.save(path, x)` and
  `SCEFitting.load(SCEBasis | SCEModel, path)` serialize a self-contained,
  human-readable **TOML** document — the crystal, the space-group ops, the
  interaction, and the *full* SALC basis (every member / term / folded tensor); a
  model adds `j0` and per-`SALCKey` coefficients. Reload rebuilds the basis verbatim
  (no re-projection) and re-pairs coefficients to the basis **by key**, not by
  position (a `SCEModel` saved by one build reloads correctly into a structurally
  identical basis). The `struct ⇄ Dict` schema (`_to_doc` / `_from_doc`) is
  format-agnostic and unit-tested without any serializer; TOML is the stdlib (no
  dependency) and round-trips `Float64` exactly.
- **TOML input** (`sce/input.jl`): `read_input(path)` and the new
  `SCEBasis(path::AbstractString; backend, tol)` constructor build a basis from a
  human-authored `input.toml` (`[structure]` inline crystal, `[interaction]`,
  optional `[symmetry]`); keyword arguments override the file's backend/tol.
  Training data and the estimator are kept out of the file (loaded/chosen in Julia),
  mirroring the basis/data separation. `read_input` is exported.
- Considered JSON vs TOML deliberately: stdlib TOML round-trips `Float64` exactly and
  expresses the full nested SALC document, so the persistence artifact and the input
  file share one zero-dependency format. The schema layer stays format-agnostic, so a
  JSON (or other) backend remains a thin future addition.
- Validated (`test/unit/test_persist.jl`, `test/unit/test_input.jl`): basis / model /
  fit round-trips (predictions bit-identical, coefficients re-paired by key under a
  scrambled on-disk order, multi-op space-group ops, empty basis), input parsing +
  defaults + keyword overrides + error paths.

### Added — arbitrary body order (N-body clusters)

- **Cluster enumeration** (`clusters/enumerate.jl`): `candidate_clusters` generalized
  to `N`-body pairwise-within-cutoff cliques (`N = 2` stays the directed neighbor pairs).
- **SALC projection** (`basis/salcbasis.jl`): general site stabilizer with induced
  permutations; projection over the combined (ordering × coupling-path × `Mf`) space
  with the action matrix built by contracting against the package's own orthonormal
  coupled tensors (no 6j/9j). Handles, at `N ≥ 3`, coupling-path mixing and — for
  unequal `l` on symmetry-equivalent sites (e.g. `l=(1,1,2)` on a triangle) —
  `l`-ordering mixing. Improper-op parity is automatic (no proper-part special case).
- **Multi-term SALCs** (`basis/salc.jl`): a SALC carries one `SALCTerm` (own `ls` +
  `folded`) per `l`-ordering; `evaluate` and `accumulate_grad!` loop over terms.
- **`SALCKey`** stays injective when a proper-subgroup stabilizer splits a degenerate
  multiset into several ordering orbits (`block` runs across them).
- Validated by ground-truth invariance / time-reversal / linear-independence tests at
  `N = 3` (incl. the multi-term channel) and an energy+torque 3-body recovery;
  cross-checked against Magesty.jl — per-`(body, ls, Lf)` invariant-subspace dimensions
  agree exactly through 3-body, and Magesty's own SALCs independently pass invariance.

### Added — torque observable (energy + torque co-fit)

- **Gradient kernel** (`basis/salc.jl`): `accumulate_grad!` sums `jϕ·∂Φ/∂e_a` per
  site (product rule over cluster members), sharing the energy kernel's `μ`-mapping
  and `(4π)^(N/2)` scale.
- **Torque design matrix** `X_T` and per-config torque targets via the four-argument
  `SCEDataset(basis, configs, energies, torques)`; entry `(4π)^(N/2)·(e_a × ∂Φ/∂e_a)`,
  rows flattened config/atom/`xyz`.
- **Energy+torque co-fit**: `fit(SCEFit, dataset, est; torque_weight = w)` minimizes
  `(1−w)·MSE_E + w·MSE_T` by per-block whitening; `j0` stays analytic from the
  energy block.
- **Prediction & metrics**: `predict_torque`, `r2_torque`, `rmse_torque`,
  `has_torque`. `predict_torque = e × ∇(predict_energy)` by construction (validated
  by on-sphere finite differences and the Heisenberg closed form).

### Added — v0 vertical slice (energy fitting end-to-end)

- **Geometry**: `Lattice`, `Crystal`, and a generalized cutoff `NeighborList`
  (`build_neighbor_list`) that replaces a fixed 27-cell image grid; `NeighborPair`
  retains the inter-site lattice translation `R`.
- **Harmonics** (`basis/Harmonics.jl`): real tesseral `Zₗₘ` + tangent-projected
  gradient (Drautz convention).
- **Angular momentum** (`basis/AngularMomentum.jl`): Clebsch–Gordan (Racah),
  `wignerD_real` (least-squares from the package's own `Zₗₘ`), complex→real coupled
  tensors; `CoupledBasis{R}`.
- **Symmetry**: `AbstractSymmetryBackend` with in-tree `NoSymmetry` and a
  `SpglibBackend` whose method lives in `ext/SCEFittingSpglibExt`.
- **Clusters**: orbit reduction with `R`-carrying members (`build_clusters`).
- **SALC basis**: orbit–stabilizer projector (isotropic and anisotropic channels),
  deterministic gauge, canonical `SALCKey` column addressing (`build_salc_basis`).
- **Fitting / API**: `Interaction`, `SCEBasis`, `SCEDataset` (energy design
  matrix), `SCEModel`/`SCEFit`, `fit`, `predict_energy`, `OLS`/`Ridge`,
  `coef`/`intercept`/`nobs`/`r2_energy`/`rmse_energy`.
- A runnable Heisenberg-chain example (`examples/heisenberg_chain.jl`) that
  recovers `J`; an oracle test environment (`test/oracle/`) validating against a
  pinned Magesty.jl.

### Notes

- Bit-for-bit agreement with Magesty.jl is explicitly not a goal — see
  `docs/design-notes.md`.
