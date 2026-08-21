# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/); this package predates a tagged
release, so everything lives under *Unreleased*.

## [Unreleased]

### Fixed — extxyz / EMBSET-pair hardening backported from SCEFitting.jl (2026-08-22)

Each of these let a file that says two things be silently arbitrated
(SCEFitting.jl `f966417`, found by its saboteur review; the reader code is
shared line for line):

- `_xyz_info` refuses a repeated info key; `_xyz_properties` refuses a repeated
  property name and any string column after `species` (both used to let the
  last value win — the second block overwrote the first, a second `S` column
  was read INTO `species`).
- `write_extxyz` quotes free-text values containing whitespace (`source`,
  `field_sign`, `comment`, `setup_id`, `reference_id`) and refuses a double
  quote or a line break inside one: the lexer has no escape, and an unquoted
  `source=C:\Program Files\x` produced a file its own reader rejected —
  against the docstring's promise.
- `read_embset_pair` refuses a pair whose moment blocks are all bitwise equal
  (the same file twice, a byte copy): it passed every sibling check and made
  `moments_bare ≡ MW`.

Gated in `test_extxyz.jl` (duplicate key / property / extra string column
refused; whitespace values round-trip quoted, a quote or newline refused at the
writer, provenance strings through the same door; identical files and a byte
copy refused by the pair reader).

### Fixed — moment-channel backports from SCEFitting.jl's M4/M5 review panels (2026-08-22)

- **Zero-moment placeholder door at `MomentDataset`** (`zero_moment_atol =
  1e-10`): a referenced atom (marked, or an environment site of any pointed
  member — `_referenced_atoms(::MomentBasis)`) with `‖MW‖ ≤ atol` is refused by
  name. Its `directions` column is the ẑ placeholder the moments constructor
  fabricates, which entered the design as a fake environment coordinate (and, in
  mode 4, the row's own axis) while the `|M| = 0 → g = 0` convention waved the
  row through. The docstring states the alignment obligation: the placeholder
  was fabricated by the reader at ITS `zero_moment_atol`, so pass the same value.
  `M_int = 0` on a marked atom is not this case and still passes.
- `_pair_neighbors` returns empty neighbour sets on a 1-body basis (the shell
  `cutoff_pair` names is one the model never reads), so `moment_local_field` /
  `moment_coverage` no longer report a coordinate with no bearing on the fit.
- `salc_groups(::MomentBasis)` asserts exactly one DISP slot per term — the same
  one-mark invariant `_mark_term_index` pins.
- `_design_moment`'s shape checks run once before the threaded loop (a throw
  inside it surfaced as a `TaskFailedException`), and the `axes` shape is
  checked too.
- Docstrings: band `[lo, hi]` ranges can overlap on tied `|⟨e⟩|` values;
  `moment_simple_floor`'s sigmas are Bessel `std`, not `rmse_moment`; its
  design-row replay covers one configuration only.

[SCEFitting.jl `9b95523` / `bb94992`; the `directions` door SCEFitting added is
not needed here — `TrainingDatum` validates `directions` at construction.]

### Fixed — backports from SCEFitting.jl's review panel (2026-08-21)

The pure-spin carve-out ported this package's decor engine, persistence and
coefficient table and had them reviewed; four findings apply to the shared
code verbatim and come back here as the same patches.

- **The persist reader validates a term against its slots.** `_term_from`
  accepted any slot list alongside any tensor: more slots than tensor axes, a
  slot addressing a site the member does not have, or an axis whose extent is
  not `2l + 1` was built as is (`SALCTerm` has no inner constructor) and failed
  later, inside a kernel, as a `BoundsError` or a silently truncated
  contraction. All three are refused with a named `ArgumentError`, on both the
  v5 `slots` and the v2–v4 `ls` spellings; `_member_from` hands the reader the
  member's site count. A second review then found the one **silent** member of
  the family: nothing checked a term against its **key**, so a DISP slot under
  a pure-spin key passed every per-term check and slipped past the spin-only
  kernels' refusal (which reads the key's `decors`) — the DISP axis evaluated
  as a spin harmonic under the wrong `(4π)` scale. `_salc_from` now requires
  `body == length(decors)`, `body` atoms per member, and every term's slots to
  reconstruct the key's label (`_term_decors`). Nine error-path tests.
- **`coeftable`'s `decors` column is comma-splittable again.** A displacement
  factor rendered as `u(k,l)` — a comma inside a comma-joined column — so a
  mixed row such as `"2+u(0,1),u(1,0)"` read back from CSV / Arrow as four
  sites. The token is now `u(k:l)`; pure-spin rows still read `"1,1,2"`.
- **`hash(::SiteFactor)` / `hash(::SiteDecor)` are content-based.** The default
  struct hash mixes in the type's identity, which differs between precompiled
  images of the same package; a `SALCKey` carries `SiteDecor`s, so
  `SALCBasis.fingerprint` was a function of the build (SCEFitting measured three
  environments, three values for the same content). No container in `src/` is
  iterated in hash order (the decor-keyed `Dict`s and `Set`s are
  membership-only), so no output changes.
- **An independent Cartesian projector oracle** for the decor engine's counts
  (test only): the invariant counts on the O_h single site and the D4h bond are
  re-derived by averaging the group action over the multilinear forms in the
  components of `ê` and `u` — no Clebsch–Gordan, no Wigner-D, none of the
  CountingOracle's machinery — and agree with the engine and with Burnside by
  hand (`(88 + 56)/16 = 9`, `48/24 = 2`).

### Added — the pin tier: change detectors over the SALC chain (2026-08-21)

- **`test/pin/`** — a byte-level pin over five real crystals (bcc Fe, B2 FeRh,
  hcp Co, wurtzite GaN, rocksalt MnO), with its own environment, its own driver,
  and a CI job that runs it at 4 and at 1 thread. **Change detectors, not
  correctness evidence** — every expected value was produced by this package;
  `test/unit/test_normalization.jl` is the correctness half of the pair.
  - Four layers, each with its own strictness and its own recapture rule:
    **L0** structure (integers and labels — exact on every platform, and the
    precondition for the two layers indexed by position), **L0′** the sign/support
    pattern of `folded`, **L1** every `folded` entry as a raw IEEE-754 bit
    pattern, **L2** `‖X_E‖_F²` / `r2` / `coef` / held-out energies.
  - **The L0′ threshold is measured**: over 12,040 tensor entries, 7,728 are
    exactly zero and the smallest survivor is 0.239, against a prune threshold of
    1e-10 — a **9.38-decade gap**, with `eps = 5e-6` at its geometric mean. That
    gap is what makes L0′ exact across platforms.
  - **L1 is not declared portable**: the chain runs through `eigen` and BLAS, so
    it is exact on the capture platform and `rtol = 1e-12` elsewhere, with the
    platform recorded in each pin's `[meta]`. Byte-stability across *thread
    counts* IS measured — captured at `-t 4`, exact at `-t 1`.
  - **The runner names the layer that moved and the rule that applies.** There is
    no branch protection and no pin-only-commit job, so that printout is the whole
    enforcement mechanism; the rules are in `test/pin/PIN.md`.
  - L2's targets are a seeded random vector, not energies pushed through this same
    basis: measured, that round trip cancels and leaves `coef` and `r2` numerically
    identical under a uniform rescale of the whole basis.

### Fixed — the benchmark suite could not run (2026-08-21)

- **`bench/bench_solver.jl` had been dead since the export/public split**: it
  calls `solve_coefficients`, which is `public` but not exported, so the script
  died at load with `UndefVarError`. Nothing runs `bench/` in CI, so nothing
  caught it. Fixed with an explicit `using <Pkg>: solve_coefficients`.
- **`bench/Manifest.toml` no longer resolved**: the package gained a `Printf`
  dependency after the manifest was last written (2026-07-30), so *every* bench
  script failed at load. Re-resolved.

### Added — benchmark baseline and regression rule (2026-08-21)

- **`bench/BENCH_LOG.md`** — the measured baseline for all six scripts at their
  stress defaults, plus an explicit rule for what counts as a regression: the two
  gate scripts are `bench_salcbasis` and `bench_design_matrix`; **allocation
  count is the primary gate at zero tolerance** (measured bit-identical across
  three consecutive runs, so the quantity is deterministic) and **wall time is
  secondary at 5%** (measured run-to-run spread at most 1.4%, so ~3.5x headroom).
  The other four scripts are context, with the reason each is not a gate.

### Added — absolute-normalization oracles for the SALC chain (2026-08-21)

- **`test/unit/test_normalization.jl`** — four closed-form gates over the stretch
  of the numeric chain that had no oracle: the Reynolds projector,
  `_canonical_basis`, `_canonicalize_members`, the `folded` contraction, and
  `evaluate_salc`. Everything that reached those stages before was gauge- or
  scale-invariant, so a uniform rescale of the whole basis passed the entire
  suite (measured: doubling every `folded` tensor left the orbit-sum invariance
  check AND a member-resolved invariance check green at 1e-14).
  - **O1** two sites, P1, `l₁ = l₂ = 1`: the Heisenberg channel is
    `Φ(e) = 2√3 (e₁·e₂)` — from the real-harmonic normalization
    (`test_harmonics.jl`), `‖C^{Lf}‖_F² = 2Lf+1` (`test_coupledbasis.jl`), the
    `(4π)^{N/2}` scale, and the `N!` ordering fold.
  - **O2a** one site, `l = 2`, P1: `Σ_b Φ_b(e) Φ_b(e′) = 5 P₂(e·e′)` (spherical
    addition theorem). Gauge-free — invariant under any orthogonal remixing of
    the five columns, so a legitimate `_canonical_basis` gauge change cannot
    break it.
  - **O2b** one pair member, P1, `l₁ = l₂ = l`: exactly `(2l+1)²` columns and
    `Σ_all Φ² = 4(2l+1)²` (Clebsch–Gordan completeness), measured for
    `l = 1…4`. This is the only gate in the suite that reaches `l ≥ 3`
    absolute normalization.
  - **O3** the C3v equilateral triangle: `Φ = 2√3 Σ_{i<j}(eᵢ·e_j)`, with the
    reference side built from the three geometric bonds and never from
    `s.members` — so member multiplicity is under test, which O1/O2 cannot see.
  - **Teeth measured**, not asserted: each oracle has a source mutation that
    kills it and leaves the other three green (table in the file header).

### Fixed — CI repair + moment-channel backlog (2026-08-20, post step 2)

- **CI repair (the bd03517 push failed CI in two jobs the local default suite
  does not run).** JET (`TEST_MODE=all`): two `read_extxyz` union splits were
  runtime-guarded but not inference-visible — `tryparse(Int, modes[1])` (the
  `!== nothing` narrowing does not survive a second `getindex`; bound local) and
  `crystal_fingerprint(reference)` inside the datum loop (`joint ⇒ reference isa
  Crystal` is a cross-branch fact; hoisted `ref_fp`). Docs (`:missing_docs`
  strict): the 14 moment-channel / extxyz docstrings were not on any manual
  page — `docs/src/api.md` gained an "Adiabatic site-moment channel" section.
- **`moment_resolvability` under-enumerated flat directions on a WIDE signature
  block** (more kept columns than signature rows): the economy SVD's `V` lists
  only `min(r, c)` directions, so the null report came back EMPTY at rank 20 of
  74 kept — the dataset door's "structurally dependent" disclosure silently
  missed all 54. Found by the new P1 face-(b) control fixture. The complement of
  `span(V)` is now read off a QR completion of `V` (never a full SVD — the row
  side can be huge and its full `U` is never needed). The report is asserted
  COMPLETE (`length(null_combinations) == length(kept) − rank`) and every
  combination is gated against numerical design annihilation.

### Added — moment-channel small backlog (2026-08-20)

- **Local-field diagnostics + the simple-feature nested floor** (the last small
  backlog item; script-level in step 2, now package API):
  - `moment_local_field(mb, configs; axes = configs)` / `(mb, data)` — per-row
    `|h₁| = |Σ ê_j|` and the collapse coordinate `ê·ĥ` over the pair-consistent
    neighbor set (`cutoff_pair` MinimumImage, tied images counted, `lmax_env = 0`
    species excluded). The `TrainingDatum` method resolves the row axis through
    `_moment_axis_matrix` — the mode rule extracted from the dataset constructor
    into ONE function so the two consumers cannot drift. A validating door
    (configs unit everywhere; axes unit on marked columns, exact zero allowed —
    the mode-1 undefined row, `edoth = NaN`).
  - `moment_coverage(train, new; q = 0.99)` — the M2-8 applicability monitor:
    fraction of new rows beyond the training `|h₁|` q-quantile + the
    anti-alignment fraction (`ê·ĥ < 0`, the measured collapse band).
  - `moment_simple_floor(f, data; lmax = 2)` — the M2-5 nested performance
    floor: per-orbit intercept + Legendre shell-sum OLS on the SALC fit's own
    kept rows. The nested bound `sigma_model ≤ sigma_floor` is CONDITIONAL and
    both conditions are returned, never assumed (review): per-feature
    `inclusion` (relative residual of an SVD range-basis projection onto the
    active kept design — never `X \ F`, which is rank-deficient by
    construction whenever columns vanish) and `nested_bound = f.estimator isa
    OLS` (a shrinkage fit legitimately trades training residual — the bound
    does not apply). An `lmax_env = 1` basis shows the P₂ feature at
    inclusion ≈ 0.98 — the disclosure case. `data` is paired to the fit loudly
    in two halves: bitwise target recomputation on every defined row AND a
    design-row replay of one fully-defined configuration (the target check is
    blind to a co-rotated environment — review-found). `n_rows`/`design_rank`
    disclose the saturated-design vacuous case.
  - `MomentBasis` now stores its build `tie_tol`; the diagnostics' neighbor
    enumeration reads it back (a widened tie band changed the tied-image
    multiplicity silently when it had to be re-passed by hand — review-found).

- **`moment_resolvability` default-`rtol` cache on the basis** (`MomentBasis`
  gained a trailing `resolvability::RefValue` field): the answer is a pure
  function of the basis and `MomentDataset` runs the gate at every construction,
  so a train/held-out pair paid the symbolic expansion twice. Cached calls
  return the SAME object (`===` is the tested contract); a non-default `rtol`
  always recomputes; an `UnclassifiableBasis` refusal is deliberately not cached
  (loud on every call).
- **`salc_groups(mb::MomentBasis)`**: group labels for group-adaptive shrinkage
  over pointed columns, keyed `(body, orbit_id, decors, mark class)` — the
  energy-side key folds stabilizer-inequivalent mark placements (an Fe–Ge pair
  marked on Fe vs on Ge predicts different atoms' moments) into one group;
  the mark class (read off the representative member's DISP-slot atoms AND
  site indices — review found the atom set alone merges two placements when a
  member carries two periodic images of one atom; regression-pinned) splits
  them while gauge blocks of one class still share a label. Design-side oracle:
  mixed-species mark classes have disjoint row support. Plus a
  `GroupAdaptiveRidge(mb; lambda)` convenience (unit weights — no MC cost story
  on this channel).
- **`fit(MomentFit, …)` reduces column-structured estimators to the active
  mask** (`_reduce_to_active`): with vanishing columns frozen, full-basis GAR
  labels used to die in `solve_coefficients` with a misleading
  "different SLCEBasis" `DimensionMismatch`. Group norms are exactly preserved
  (frozen coefficients are exact zeros); the group size `p_g` — hence the
  per-coefficient ε floor — drops by the frozen count, an O(ε) effect and a
  deliberate divergence from the energy side (ASR-frozen columns keep their
  `p_g`). Emptied groups are relabeled away.
- **Real end-to-end vanishing fixture** (no ctor injection): the energy side's
  face-(a) mechanism transplanted to the pointed basis — a four-fold WS tie on a
  3×3×6 two-atom cell partially fused by `m_y`, under `soc = true` (16 of 38
  columns cancel identically on cell-periodic data; design column norms are
  EXACTLY 0.0). Under `soc = false` only `Lf = 0` scalar spin couplings survive
  construction — image-odd content cannot exist, so pointed face (a) requires
  SOC. The fixture drives warn → store → exact-zero freeze → predict end to end.

### Added — moment channel: fast design path + band-profile diagnostic (2026-08-20, step 2 slice E)

- **`_design_moment(…; member_index = true)`**: the design build now skips, per
  row, every (member, term) whose mark does not sit on the row's atom — the
  granularity is the TERM (one member's terms can carry the mark on different
  sites), and skipping removes only exact-zero additions, so the fast path is
  value-identical to the full evaluation (gated elementwise-`==` in the tests;
  `member_index = false` keeps the full path as the oracle). Measured 22.7× on
  the FeGe 2×2×2 supercell (181 columns, 32 marked atoms).
- **`moment_band_profile(model, ds; nbins = 4)` / `(f::MomentFit)`**: the
  coverage-band residual profile (M2-8/L2-2) — per-configuration mean residuals
  over kept rows along the marked-sublattice order parameter `|⟨e⟩|` (now stored
  as `MomentDataset.order`), as equal-count bins plus the bin-free least-squares
  line (`slope`, `intercept`, Pearson `r`). A systematic band trend on held-out
  data is the basis-insufficiency signature; report it next to any σ.
- Deferred, recorded: the simple-feature nested lower-bound gate and the |h₁|
  feature-coverage check stay script-level for now; `salc_groups` mark-class key
  not yet added.

### Added — `MomentDataset` / `fit(MomentFit, …)` / `MomentModel`: the moment channel's regression layer (2026-08-20, step 2 slice D)

The pointed basis's counterpart of `SLCEDataset` + `fit` (`src/fitting/momentfit.jl`)
— an independent vertical slice with its own rows and coefficient vector; the
energy/torque/force row bookkeeping is untouched.

- **`MomentDataset(basis, data; gate_eps, coverage_floor = 0.5)`**: rows are
  (configuration, marked atom); the target `y = ê·M` reads the bare moment with
  the axis resolved by the **mode rule** (mode 4 → `directions`, so the
  marked-column substitution is the identity; mode 1 → `constraint_axes`; modes
  may be mixed). A mode-1 marked atom with an exactly-zero axis column has no
  defined target: excluded from every fit (`defined = false`, `y = NaN` — loud on
  raw use) and recorded per orbit. The **decomposability gate**
  `g = |M| − y²/|M| = |M| sin²θ ≤ gate_eps` keeps rows (`|M| = 0` passes with
  target 0; deliberately no `m_min` gate); `gate_eps` is a required keyword — the
  tolerance is a statement about the source calculation's constraint quality.
  Per marked-atom space-group orbit the constructor reports row counts, survival,
  and rms `‖M⊥‖`, and refuses loudly when survival drops below `coverage_floor`
  (before paying for the design build). `X` is built for ALL rows with
  `defined`/`keep` masks, so gated and ungated coefficients are both disclosable.
- **Doors and disclosures** (review round, same day): the constructor also
  refuses nonzero `displacements` (v1 expands `m_i(e)` at the reference geometry
  only) and mixed reference identities; it runs `moment_resolvability` before the
  design build — an unclassifiable basis propagates that refusal, identically
  vanishing columns land in `ds.vanishing`, structurally dependent combinations
  are warned loudly and stored in `ds.dependent`; the per-orbit report counts
  antiparallel mode-1 rows (`n_anti` — the gate is even in `y`, and the
  even-mark-rank columns assume the recorded axis is oriented with the converged
  moment); the gate/`‖M⊥‖` arithmetic uses the cancellation-free `M⊥ = M − y ê`
  form (non-negative `g` by construction).
- **`fit(MomentFit, ds, estimator = OLS())`**: no centering, no global intercept —
  the `l = 0` 1-body `[MARK]` columns are the per-orbit intercepts μ₀ (never
  shared across species; the `l = 2` 1-body columns are on-site ê anisotropies).
  Solves the gated rows and, for disclosure, all defined rows (`coeffs` /
  `coeffs_ungated`), passing `row_groups = row_config` so resampling estimators
  never split one configuration's marked-atom rows across folds; columns named in
  `ds.vanishing` are frozen to **exact zero** (the energy side's frozen-column
  discipline). `residuals` / `rmse_moment` take `gated = …`. Note: regularized
  estimators penalize the μ₀ columns like any other (v1 is OLS-first).
- **`MomentModel`** (basis + gated coefficients + provenance; deliberately no
  mode flag) and **`predict_moment(model, e; axes = e)`** — the runtime default
  axis is the spin direction itself, exactly the mode-4 identity substitution.
  `predict_moment` is a validating DOOR (the package's unit-norm rule for every
  spin-reading entry point): `e` must be unit columns throughout, `axes` unit on
  the MARKED columns only — unmarked axes columns are never read, so closed-form
  component extraction at `ê = x̂, ŷ, ẑ` stays legal.

Tests (`test/unit/test_momentfit.jl`): hand-arithmetic target/gate oracle,
mode-1 ≡ mode-4 identity equivalence (bitwise), planted-model recovery via
prediction equivalence (rank-robust; the FeGe primitive fixture's structural
flat direction is cross-named by `moment_resolvability`), gate keep/reject with
gated-vs-ungated disclosure, the `|M| = 0` pass, zero-axis exclusion, the
coverage-floor refusal, setup-uniformity and requirement doors, dataset-level
time reversal (bitwise), and μ₀ absorbing a per-orbit constant shift.

### Added — `TrainingDatum` carries the adiabatic-moment channel (2026-08-20, step 2 of the pointed moment expansion)

Three optional fields (all default `nothing`; the energy/torque/force paths are
bit-identical when absent — gated in `test_dftsource.jl`):

- **`moments_bare`** (`3 × n_atoms`, μ_B): the bare per-atom moment vectors (VASP
  `M_int`), target of the adiabatic site-moment channel `y_a = ê_a · M_a`.
  Finiteness is the only value constraint — the signed readout crossing zero is
  the point of storing the vector (the non-analytic `‖M‖` target was rejected at
  design time). Distinct from the smoothed `magmoms · directions` decomposition
  (`MW_int`, which the constraint acts on and the torque channel reads); their
  ratio is configuration-dependent, so both are stored.
- **`constraint_axes`** (`3 × n_atoms`): per-atom constraint axes, unit columns
  or exactly-zero ("no axis for this atom"); near-zero noise is refused, never
  normalized.
- **`constraint_mode`** (`1 | 4`): the physical class of the constrained-DFT
  scheme — `1` = transverse-penalty type (axis prescribed, sign free; VASP
  `I_CONSTRAINED_M = 1`), `4` = direction-pinning type. The moment channel's
  evaluation axis is keyed by this mode (mode 4 → `directions`, mode 1 →
  `constraint_axes`), **deliberately never by which fields happen to be
  present** — an availability-keyed fallback would silently drop a mode-1 datum
  with missing axes into the broken `ê_MW` coordinate (measured σ 2.1× on FeRh).
  Hence the ctor invariants: mode 1 requires `constraint_axes`; `constraint_axes`
  without a declared mode is refused.

`spin_datum` (both arities) and `joint_datum` pass the trio through unchanged, so
the construction paths cannot disagree. Consumers (extxyz I/O, `MomentBasis`,
the moment dataset/fit layer) land in subsequent slices.

### Added — `MomentBasis`: the pointed SALC basis for adiabatic site moments (2026-08-20, step 2 slice C)

The moment channel's counterpart of `SLCEBasis` (`src/basis/momentbasis.jl`):
per marked reference-cell atom the design row is the pointed SALC vector, and one
shared coefficient vector serves every symmetry-equivalent site. The MARK is the
displacement decor `SiteDecor(disp = (1, 0))` (the polar `|u|²` factor evaluated
on an indicator field), so the existing decor engine, evaluation kernels, and
canonical-member machinery carry the basis unchanged; `_orbit_salcs_decors`
gained one inert `admit` kwarg (a per-assignment predicate, mutually exclusive
with the species caps — byte-identical behavior when absent).

- **`MomentSpec`**: mark-aware truncation — the marked site's ê factor is
  allowed for ANY species (`lmax_mark`; an E-inactive species' induced moment is
  what the channel predicts), environment spin factors only for species the
  consumer samples (`lmax_env` + the required `sampled` claim, refused loudly on
  mismatch — M3-1 decision A). Time reversal keeps even-Σl labels only (the mark
  rank counts).
- **3-body stars are mark–environment-bond cut** (M2-5): both mark bonds
  minimum-image within `cutoff_star`, the environment–environment edge free (the
  triangle is pinned by the mark bonds). The enumeration expands every star to
  all 3! site orderings — `candidate_clusters`' multiplicity convention, without
  which the closed-star column came out at half the prototype's 6.0 geometric
  oracle (caught by the oracle, fixed at the source).
- **`_design_moment`** evaluates rows `(config, marked atom)` with the
  marked-COLUMN substitution: the spin matrix's marked column is replaced by the
  evaluation axis (identity for mode 4), exact because every pointed label
  carries exactly one mark.
- **`moment_resolvability`** (D9′, replaces nothing — the energy-side
  `unresolvable_columns` is deliberately not reused): symbolic signature
  expansion in the independent variables `(a, ê_a, e)` → vanishing columns,
  numerical rank, and null combinations that NAME the dependent columns; plus
  the mark-class census per cluster orbit (face-(b) hazard preregistration).
  Members putting two environment spin factors on one reference-cell atom (two
  periodic images of one neighbor) are refused as `UnclassifiableBasis` — the
  monomial signature would overcount the rank there (measured 108 symbolic vs
  98 actual on the FeGe primitive cell; harmonic products on one sphere reduce).

Gated in `test_momentbasis.jl` against the design-record prototype's independent
geometric references: star closed form = 6.0 × the geometry sum (spread 6e-14),
shell-sum normalization 2√3, G_i covariance including the axes, bitwise time
reversal, substitution locality, and signature rank ≡ independent random-design
rank with null combinations annihilating the actual design.

### Added — extended-XYZ container, axis gates, EMBSET pair reader (2026-08-20, step 2 slice B)

The canonical on-disk training-set format moves to extended-XYZ (ASE dialect),
one self-contained file per set (`src/io/extxyz.jl`):

- **`write_extxyz(path, data, crystal)` / `read_extxyz(path; reference)` /
  `ExtxyzFile`**: the structure (lattice + positions) is ALWAYS stored, even for
  spin-only data — self-containment removes the "which POSCAR pairs with which
  EMBSET" provenance-bug class. Per-atom columns exist only when observed
  (`species:pos:mw[:bcon][:mint][:mconstr][:forces]`, 1:1 with the datum's
  `Union{…,Nothing}` channels); numbers print shortest-round-trip so every stored
  value survives the file bit-exactly. **spin-only vs joint is measured from
  positions** (bitwise identical across frames → spin-only; differing positions
  need `reference::Crystal` and become displacements), and a `config_type` claim
  contradicting the measurement is a loud error, as are cross-frame drift of
  species/lattice/columns/mode/setup, `units_field ≠ eV/muB` (the "T" header
  mislabel), and `mconstr` without a declared mode.
- **`check_moment_gates`** (public unexported): the moment channel's
  axis-consistency gates, run at generation AND at every load — archived
  constraint axes are re-verified against the converged moment directions, never
  believed. Sign-consistency (mode 1, decomposable rows `|y| > 5e-3 μ_B` ≈
  10 σ_flip): `sign(ê_MW·ê_c) = sign(y)` exactly. Axis-angle p99 < 5° (both
  modes) catches small-angle staleness the sign gate cannot. A whole-axis flip
  in mode 1 flips `y` with it and correctly does NOT fire (the axis sign is a
  gauge; gated in `test_extxyz.jl`).
- **`read_embset_pair(mw_path, mint_path)`**: the legacy archive reader for
  EMBSET (MW) + EMBSET_mint (bare M) siblings. Loud pairing checks: config
  count, block shape, **field blocks bitwise identical**; energy lines are
  deliberately NOT compared (the two writers' conventions differ — measured
  ΔE = 0.148 eV on the FeRh archive). `constraint_mode`/`constraint_axes`
  attach what the format cannot carry, and the axis gates run.

### Added — `tie_tol` exposed; OLS warns on a deficient design (2026-08-13, back-port from SCEFitting.jl)

The MnTe(0001) slab cross-check (fixed first in the spin-only SCEFitting.jl)
showed that DFT-relaxed coordinates symmetric to less than the same-distance band
(slab residual ~2.4e-6 Å, tie splits ~2e-7 relative vs the 1e-8 band) split
symmetry-partner minimum-image ties differently, so the candidate set loses group
closure and the orbit builder's own refusal (d4660a7) fires with no remedy
available — the slab could not be built at all.

- **`SLCEBasis(...; tie_tol)`** exposes the relative same-distance band (default
  unchanged at `1e-8`, hard cap `1e-2`), threaded to both the neighbor list and
  the cluster-edge admission (`NeighborList.tol`), and carried by the TOML setup
  (`[interaction].tie_tol`, `read_setup`, `SLCEBasis(path)`) so a build that
  needed a widened band is reproducible from its own file. A widened band merges
  the near-tie shell; the resolvability layer (`unresolvable_columns` →
  `build_asr`) then freezes whatever the merged shell's aggregation cancels,
  exactly as it does for exact WS-boundary ties. (On the bulk MnTe + SOC
  cross-check fixture that layer already froze precisely the 14 aggregate-zero
  columns of the 51-column anisotropic basis — SCEFitting's defect ② never
  existed here; verified while porting.) The closure refusal now names the
  remedy.
- **`OLS` warns on a rank-deficient (or severely ill-conditioned) design**: the
  solve is unchanged (explicit pivoted QR, bitwise-identical to `X \ y` on
  rectangular designs), but a diagonal ratio below `1e-10` warns that the
  coefficients are non-unique. One-sided conditioning gate (cannot fire while
  `κ₂ < 1e10`). This is the backstop for what the resolvability classification
  cannot certify — notably the `asr = false` AllImages opt-out
  (`UnclassifiableBasis`), previously the one fit route with no loud gate at
  all — and for degenerate training data.
- The perturbed-honeycomb closure fixture (refusal + `tie_tol` heal, end to end
  through `SLCEBasis`) is ported into `test/unit/test_clusters.jl`; the closure
  gate itself had no regression test here.

### Fixed — the `atol` keyword can no longer bypass the direction-door cap (review 2026-08-11 M2)

`SLCEDataset(...; atol)` reached `_validate_direction` without `_check_atol`, so
the `_DIRECTION_ATOL_MAX = 1e-2` cap bound only the projecting doors
(`UnitVector3` / `SpinConfiguration`) — and the dataset door is the one that
validates WITHOUT projecting, so past the cap a moment-scaled vector (`‖e‖ = 0.6`
at `atol = 0.5`) entered the design matrix raw and biased every fitted `jϕ` by
`C_l·δ` (~98 % on the demonstration). `_validate_config` now enforces the cap;
`direction.jl`'s "widening no longer degrades any number" rationale is corrected
to name which doors that is true for. `TrainingDatum` now validates through the
family's `_validate_direction` (it hand-rolled the rule and omitted the
`|component| ≤ 1` half); `Harmonics._validate_unit`'s restated band gains a
behavioral tripwire pinning it to `_DIRECTION_ATOL`.

### Fixed — `effective_dof`/`gcv` refuse a `refit` result by name (review 2026-08-11 M3)

Both diagnostics reconstruct the FULL design from `dataset` + `reparam`, which is
not the problem a `refit` solved (it solves on a support, under a re-derived
sub-stage that is deliberately not stored) — `effective_dof(refit)` returned the
full-design rank (40.0 where ≈ 5 was honest) and `gcv` a silent `Inf` through the
`n − df` guard. `SLCEFit` now records the refit `support` (a new trailing field;
`nothing` for a direct fit) and both diagnostics refuse a refit result with the
remedy named. `refit` also refuses a `GroupAdaptiveRidge` up front (its
`column_groups` index the full design, not the chosen support — the old failure
was a `DimensionMismatch` blaming a "different SLCEBasis").

### Fixed — GCV accounting corrections (review 2026-08-11)

**Changes `gcv`/`effective_dof`/`select_fit` numbers in the named regimes.**

- The `+1` intercept is charged only when the energy block carries weight: at
  `torque_weight + force_weight == 1`, `j0` is estimated from rows GCV's
  `n_eff` does not count, inflating the score by `((n−df)/(n−df−1))²` — ~3 % at
  `n_eff = 72` and unbounded as `df → n_eff`.
- `select_fit`'s `:cv` score is the held-out SSE **per informative row**
  (`sse/neff`, not `sse/n`), putting it on the same objective scale as
  `cross_validate.pooled_score` and `select_support.score`.
- The frozen-weight optimism of the adaptive estimators' df (a lower bound —
  measured −14 % at small λ against a numerical `∂ŷ/∂y` trace) is now documented
  on `gcv`/`effective_dof`/`select_fit` with its direction and the `:cv` remedy;
  the math is the standard converged-weight treatment and is unchanged.
- `select_fit` re-checks the Pareto rule after the cold re-derivation mutates
  the selected row (warns if a knife-edge case ever moves the choice); the `:cv`
  fold path zeroes frozen columns like the full-data path; `_support_thresholds`
  refuses an empty group vector by name instead of dying in `log(0)`.

### Fixed — smaller review items

- `wignerD_real` asserts its least-squares residual (a degenerate Fibonacci
  sample matrix would have returned a non-representation silently; measured
  `cond(A) ≤ 3.82`, so this is latent-risk insurance).
- Stray leftover phonopy comment fragment removed from the alamode include line.

### Fixed — the displacement-radius guard's reference distance on non-reduced cells

`_min_reference_distance` scanned lattice shifts over a fixed `[-1, 1]³` box, so
on a strongly skewed (non-reduced) cell — where the shortest lattice translation
needs `|n| ≥ 2` — it over-estimated the shortest reference distance and the
`0.5·dmin` displacement-radius warning threshold was silently too permissive
(audit 2026-08-01 #2; measured: `a₂ = 2.5·a₁ + 0.1·ŷ` has its shortest lattice
vector `2a₂ − 5a₁` at `|n₁| = 5`, reported 1.0 for a true 0.2 — threshold 2.5×
too lax). The scan now uses the neighbor list's two-pass idiom: a `[-1, 1]³`
first pass bounds the minimum, `_sufficient_range` grows the box to a provably
sufficient range, and one rescan finds the true minimum. Affects only when the
warning fires — no fitted number moves. Gates: the hand-derived skewed cell
(0.2, where the fixed box says 1.0), a two-atom skewed cell against an in-test
brute force, the guard firing at the true threshold, and the slab / lone-atom
no-image edge cases.

### Changed — the checked harmonic entries validate by the family rule

`Zlm` / `grad_Zlm` (the checked entries; the kernels behind doors use the
`_unsafe` pair) validated the direction with a bare `1e-8` norm band — stricter
than every door in the package on the harmless axis and blind on the
load-bearing one. `_validate_unit` now applies the family rule: the `1e-6` band
plus the `max|component| ≤ 1` bound that establishes `dnPl`'s `|z| ≤ 1` domain,
so a near-pole direction `5e-9` off unit is refused with a named `ArgumentError`
instead of throwing a bare `DomainError` from inside the Legendre recursion
(audit 2026-08-01 #1, rank 4 — the last of the four ranks).

### Changed — the lattice-only entry is `nothing`, and only `nothing`

**Breaking**: `predict_energy` / `predict_force` / `affine_energy` / the
`EffectiveModel` energy no longer accept an all-zero `3 × n_atoms` matrix in the spin
slot. Callers who mean "there is no magnetic state" pass `nothing`, which they could
already do; callers who pass a matrix now have it validated as a magnetic state — unit
columns and `max|component| ≤ 1` — even on a lattice-only model where nothing reads it.

This closes the asymmetry the `SpinConfiguration` entry above left behind.
`_resolve_spins` (the derivative readouts) validated unconditionally and represented
the absence as `nothing`, while `_validate_config_pair` (the joint predictors and
`affine_energy`) still made an exception — shape only, no unit check — *so that* the
omission marker, an all-zero matrix, could pass through it. That exception is what made
the marker legal enough to travel: `_no_spins` handed a fabricated all-zero
configuration to a public door, and the door had to be weakened to let it back in. It
is now a LOCAL filler built at the `::Nothing` method itself
(`_spin_kernel_matrix(nothing, nat)`) and passed to an unvalidated kernel
(`_joint_energy` / `_joint_force` / `_affine_energy` / `_effective_energy`); nothing
stores it and no door ever sees it, so no door needs an exception for it.

The refusal and its message are one definition, `_require_spin_free` /
`_refuse_missing_spins` beside the `_basis_has_spin` predicate they turn on
(`slce/model.jl`, moved there from `slce/forceconstants.jl` so the predictors can read
it). Before this there were two, with two different messages and two spellings of the
same argument, and only one of them validated. `EffectiveModel` keeps its own copy of
the *question* — a re-expansion carries no `SALCKey`s, so "does anything here read a
spin?" is answerable only from its term list — but now shares the answer's shape.

Also fixes a JET failure introduced with the door validation: `magnon_phonon_vertices`
discharged the spin-free branch with a `::SpinConfiguration` typeassert, which
inference cannot discharge on the `spins = nothing` method. It now throws the shared
refusal directly, which infers as `Union{}` and narrows the argument.

Gates: `test_latticeonly.jl` "`spins` / `e` may be omitted only when there is none"
(the three all-zero refusals go red under a mutation restoring the `_basis_has_spin`
exception, verified) and `test_effective.jl`'s door test.

### Added — `UnitVector3` / `SpinConfiguration`: the unit-norm invariant is now a type

**Breaking**: `ForceConstantSet.spins` is `Union{SpinConfiguration,Nothing}` and
`MagnonPhononVertices.spins` is `SpinConfiguration`; the all-zero omission marker is
gone, and an all-zero matrix is refused wherever a magnetic state is expected.

Validating at every door (the entry above) fixes the audited defect but leaves the
shape that produced it: "this is a unit vector" is an invariant with no
representation, so it lives as a rule each entry point must remember, and the next
door added is free to forget it again. Nothing downstream would catch that — measured,
the acoustic modes of `D(0)` and `asr_residual` are completely blind to an off-unit
spin state, because the ASR's rows are keyed per spin monomial and `Aβ = 0` is an
identity in `e` that a wrong `e` satisfies too. So the invariant is now carried by a
type that cannot be obtained without establishing it.

**Two constructors, and the split is the design.** `UnitVector3(v)` validates and then
projects; `UnitVector3(v, Trusted())` validates and preserves the caller's bits. The
second is a correctness requirement, not an optimization: `normalize` is not bitwise
idempotent (measured, ~38 % of already-unit directions move by up to 4.4e-16 under a
second application), and SLCEDynamics' default integrator advances spins by a rigid
Rodrigues rotation with no renormalization step — so re-projecting a restored
checkpoint applies an operation the uninterrupted run never applied, which forks a
chaotic trajectory. That is exactly why `_config_verbatim` refuses to normalize, and
the refusal now has a door of its own instead of being a bypass around the rule.

**Validation order matters and changed a behaviour.** The projecting constructor asks
"is this a direction?" (finite, `|‖v‖ − 1| ≤ atol`) *before* projecting — that is the
half projection must not paper over, and it is what still rejects a moment-scaled
vector loudly — and asks the component bound of the *projected* result, because that
bound is a precondition on the value the kernels see. Consequence: a direction `5e-9`
off unit norm at the pole, which no tolerance could ever accept (`|z| > 1` breaks
`dnPl`'s domain), is now **repaired** rather than refused — projection lands `z`
exactly on `1.0`. It was the remedy the old error message told the caller to apply by
hand.

`atol` stays a public keyword and gains a hard cap of `1e-2`. With the projection in
place, widening it no longer degrades any number — the harmonics error budget it used
to buy is zero once the stored value is exactly unit — so it is now purely "how far
from unit may a thing be and still be called this direction", and there is real demand
for widening it (a MAGMOM written to four decimals is ~2e-5 off unit, which is a file
format, not a bug). The cap is what keeps that from becoming "accept a different
vector and silently project it onto the sphere".

`write_alamode`'s cross-set consistency key becomes honest as a side effect: two
lattice-only sets now agree because both carry `nothing`, not because both fabricated
the same zeros, and two sets at the same state agree because the constructor
canonicalized them rather than by luck of the caller's last ulp.

**Scope, stated plainly.** The guarantee is established at construction, and the
sampling packages mutate working state in place at a granularity no wrapper reaches
without entering their hot loops. Working buffers stay raw; the type sits at the
transport boundary. This reduces the places that must know the policy from ~40 kernel
call sites to a handful of named doors and makes "no magnetic state" type-checkable —
it does not make the invariant true by construction family-wide. `predict_energy` /
`predict_torque` / `predict_force` still take a bare matrix and still accept an
all-zero one on a lattice-only model; converting them, and promoting
SLCEMonteCarlo's `SpinConfig` alias to the same nominal type, is follow-on work.

### Fixed — the derivative readouts validate `spins` at the door, by the package's own rule

`_resolve_spins` — the funnel for `force_constants`, `strain_derivatives`,
`magnon_phonon_vertices` and `grid_strain_derivative` — checked the SHAPE of `spins`
and nothing else, while everything behind it called the CHECKED harmonics, whose band
is `1e-8`. So the readouts accepted a different set of magnetic states than the
dataset boundary, in both directions:

- **too strict**: a state `5e-7` off unit norm built a dataset, fitted and predicted,
  then `force_constants(model; spins = e)` threw `"direction must be a unit vector"`
  from inside the accumulation, naming neither the argument nor the atom;
- **too lax**: `‖e‖ = 1.7` succeeded *silently* whenever the requested order carried
  no spin-dressed term, and the bogus vector was stored on `ForceConstantSet.spins`,
  which `write_alamode` uses as its cross-set consistency key.

Each door now validates through the shared rule and the kernels below it use the
unchecked entries. Concretely: `_validate_config`'s per-column body is factored into
`_validate_direction` (finite, `|‖e‖ − 1| ≤ atol` at the package's `1e-6`, and
`max|component| ≤ 1`), `_resolve_spins` calls it — conditioned on `_basis_has_spin`,
exactly like `_validate_config_pair`, so the all-zero omission marker of the spin-free
entry path stays legal — and the two entry points that were their own doors and had no
check at all, `site_rows!` and `predict_energy(::EffectiveModel, e, du)`, now call it
too. The five inner call sites (`_fill_fcs_tensor!`, `magnonphonon.jl` ×2,
`effective.jl`, `rowlayout.jl`) moved to `Zlm_unsafe`/`grad_Zlm_unsafe`.

Two notes on why it is stated this way. The **component bound**, not the tolerance, is
what establishes the kernel's precondition: `Zlm` reaches `dnPl`, whose domain is
`|e_z| ≤ 1`, and near a pole any `δ > 0` can push past it — a column `5e-9` off norm
clears a `1e-8` band and still throws a bare `DomainError`. And `site_rows!` is
documented as the reference filler "which a consumer's own version must reproduce",
yet every mirror that exists (SLCEMonteCarlo's `_zlm_row!` and its device replica,
SLCETools' samplers) reaches `Zlm_unsafe` and validates nothing — so the reference was
the only implementation that threw, and at a tolerance no other door used. It now
states the same domain its consumers do.

Gates in `test/unit/test_forceconstants.jl` "spins is validated at the door": both
halves of the reported defect, the pole hazard, the message naming argument + atom +
readout, and the spin-free path still accepting both `zeros(3, nat)` and `nothing`.

### Fixed — `grad_Zlm`'s tangency is now an identity in the input, not an approximation

`Harmonics._grad_zlm_assemble` removed the radial component as `u(u·∂Z)` instead of
`û(û·∂Z)` — it never divided by `r² = ‖u‖²`. On an exactly unit `u` the two agree,
which is why every existing tangency test passed; off it they do not, and the
documented identity `u·∇Zₗₘ = 0` degraded linearly in the input's norm error:
`≈ 2·C_l·δ` with `C_l = √((2l+1)/4π)·l(l+1)/2`, measured 1.7e-7 at `δ = 1e-8` and
1.7e-5 at `δ = 1e-6` against a 4e-15 rounding floor. Dividing by `r²` makes the
tangency hold at any radius: measured `≤ 2.9e-15` for `δ` anywhere in `[0, 1e-3]`.

The module preamble already specified the fixed behaviour (`∇Z = ∂Z − r̂ (r̂·∂Z)`,
with the hat), so this is the code catching up to its own contract rather than a
convention change. Two downstream contracts written as exact statements were false
before it and are true now: `magnon_phonon_vertices`' `V·ê_b ≡ 0` ("No frame
convention is invented here" — the vertices are what makes the Cartesian return
value free of a local-frame convention) and SLCEMonteCarlo's `e_s · G[s] == 0`
"exactly", which a sweep's unrenormalized drift had been violating by `2·C_l·δ`.

**Not bit-neutral.** `r²` evaluates to exactly `1.0` for only ~50 % of normalized
directions, so gradient-derived numbers move by up to ~3.6e-15 relative. No pin in
any of the five suites moved (all green at unchanged counts), but the GPU device
replica `SLCEMonteCarlo/src/gpu/grad_device.jl` had to be changed in the same
commit — its bitwise host ≡ device gate is the guard on that coupled site, and it
runs on the KernelAbstractions CPU backend, so it bites without a GPU.

New gate `test/unit/test_harmonics.jl` "gradient tangency is independent of ‖u‖":
the oracle is the analytic identity (exact zero at every radius), the bound is the
rounding floor with headroom, and the mutation is resolved by 7–10 orders. Verified
by reverting the fix: only the new testset turns red, the pre-existing tangency test
stays green — the blindness being that it only ever samples exactly-unit input.

### Changed — internal names spelled out (no public surface touched)

The `STYLE_GUIDE.md` §1 naming contract's safe tier, applied: internal locals and
private helper functions now spell their words out. Nothing exported, nothing in the
`public` tier, no struct field and no persisted key changed, so this is invisible to
every caller and to every file on disk; the suites are green at the same counts.

Locals: `est` → `estimator`, `idx` → `index`, `cfg`/`cfgs` → `config`/`configs`.
Helpers: `_edof`/`_edof_ns` → `_effective_dof_gram`/`_effective_dof_nullspace`,
`_gar_weights!` → `_group_adaptive_weights!`, `_sm_*` → `_grid_*`, `_ex_*` →
`_exchange_*`, `_wig` → `_wigner_d`, `_frob` → `_frobenius_inner`, `_mfslice` →
`_mf_slice`, `_tclose` → `_translations_equal`, `_jnum` → `_toml_float`, `_mat3` →
`_matrix3`, `_ge0` → `_is_nonnegative`.

`STYLE_GUIDE.md` §1.9 records what was renamed and what deliberately was not.

### Fixed — writing an EMPTY force-constant set no longer does so in silence

`write_phonopy` on an empty `ForceConstantSet` produced a valid, all-zero
`FORCE_CONSTANTS` that phonopy reads happily and turns into an all-zero band structure.
`force_constants` documents that a pure-spin model yields an empty set and
`_warn_spin_blind` deliberately stays silent there (no displacement content at the order
at all), so nothing in the chain said a word — the package's own "plausible-looking
output from an empty computation" failure, which it refuses elsewhere. Both writers now
warn, naming the two ways to get an empty set. `write_alamode` checks per order.

### Changed — `dof` documents that a `refit` result reports the pre-refit count

`refit` solves on a support sub-matrix under its own sub-stage reparameterization but
stores the original one on the returned fit, so `dof(refit(f)) == dof(f)` however many
columns the support dropped. `identifiability` already states this caveat for itself;
`dof` promised "the column count of the reparameterization the fit actually solved
under" and did not. No number changes — the docstring now says which count it is.

### Changed — `residual_flat` is computed whenever the expansion ran

`_unresolvable_split` used to compute its leftover-flat-direction count only when tie
face (b) had fired, on the argument that whole-orbit granularity is the only thing that
can under- or over-shoot. That argument is sound and covers one of the two routes to a
leftover flat direction. The other needs no second orbit: two channels of a SINGLE orbit,
each individually nonzero, can go dependent once the tie collapses their arguments — face
(a) territory, where a per-column test is blind by construction and, until now, nothing
else looked either. The one diagnostic that could have caught it was switched off in
exactly the case it was for.

No basis has yet produced a nonzero value — about 70 were tried (real crystals and
hand-grouped fixtures, `N = 1..4`, degree ≤ 6, tie multiplicities 2/4/6/8) and `frozen`
equalled the sampled nullity every time. That is a reason to keep the check cheap, not to
skip it: `S` is already built by the time the count is taken, and the rank costs about 2 %
of building it.

### Fixed — the face-(b) argument was stated wrongly in six places

"A design column depends on which atoms a member joins and never on which image it
reached them through, so the two orbits are the same function" is false as written: a
member's tensors carry its own bond geometry, so the two orbits' columns are *not* equal
and evaluating any two of them shows it. What actually collapses is the **span** — every
member of either orbit reads its sites' displacements off the same reference-cell atoms,
so the orbits span one function space and the data fix only how much of that space is used
in total. The conclusion (freeze the sharing orbits outright, never split) is unchanged;
the justification is not. Corrected in `unresolvable_columns`' docstring, the
`basis/resolvability.jl` header, `build_asr`'s warning text, `theory/resolvability.md`,
`guide/lattice_dynamics.md` and `CLAUDE.md`.

Two facts recorded with it. The undetermined fraction is **not "half"** in general — it is
`length(frozen) − rank(S[:, frozen])`, so `1 − 1/k` for a `k`-fold tie the group separates
completely, and something between when the group fuses the images only partially, in which
case both faces occur in one basis. And the necessity of a tie is a **proof**, not a
measurement: under `MinimumImage` fixing site 1 at the origin fixes every other site's
image uniquely while those minimum images are unique, so two orbits over one atom multiset
force some edge to have a second equidistant image.

### Documented — at `N ≥ 3` the freeze covers congruent siblings only

The compact-cluster criterion admits a cluster only when all `C(N,2)` edges are
simultaneously minimum-image. Congruent siblings of a tie pass or fail that together, so
the freeze sees them; a sibling reached through an image that puts one of its *other*
edges on a longer shell is rejected there, and the tie then leaves no trace at all — no
tie reported, nothing frozen. That is aliasing, not indeterminacy: measured on a P1 cell
with a tied `(1,2)` edge, a 3-body degree-`(1,1,1)` sector spans the full trilinear space
(rank 27 = 3³, span identical to an independently built basis of all 27 monomials), so
every coupling is representable and `identifiability` reports `nullity = 0`. What the
aliasing costs is the `R` **label**: `force_constants` attributes the coupling to the
admitted geometry, which matters once cubic constants are exported and read as
`Φ(R₁, R₂)`.

### Added — three resolvability gates: both faces at once, and `N ≥ 3`

- **(H) both faces of a tie in one basis.** Every previous fixture had exactly one face,
  which left `_unresolvable_split`'s `!(j in vset)` guard — the single line deciding what
  happens where the faces intersect — untested: with face (a) alone nothing reads
  `undetermined`, with face (b) alone `vanishing` is empty, so deleting the guard failed
  no assertion. The new fixture makes a four-fold tie that `{E, m_y}` fuses only
  partially. Deleting the guard now fails it.
- **(I) face (b) on a three-body cluster.** The classifier is body-agnostic by
  construction and was never run at `N ≥ 3`; this gates a three-atom multiset reached by
  two orbits, end to end (exact recovery on the retained span, frozen coefficients exactly
  zero, loud failure on data containing the dropped interaction).
- **(J) the `N ≥ 3` scope above**, with the 27-monomial trilinear space built in the test
  as the independent oracle for "nothing is lost".

Gate (B) also now asserts `residual_flat == 0` on the face-(a)-only fixtures. That
assertion was previously passing on an untouched initial value; it has content only
because the count is no longer conditioned on face (b).

### Added — an integration tier: the whole pipeline, on crystals with names

`test/integration/` (own environment, own CI job) walks six named real crystals from
`Crystal` to a persisted model and back: build → data → ASR → fit → recovery → force
constants / phonons / effective model / `restrict` / strain / magnetoelastic /
published terms → persistence. The rows are bcc Fe in its conventional cell **and**
in a 2×2×2 tiling of it, B2 FeRh, hcp Co, wurtzite GaN, and rocksalt MnO; each is
there for a structural feature that changes which code path fires, not for another
cell shape.

This is not more unit tests. Every unit file gates one stage against one hand-built
fixture, and nothing gated the *combination* on a real system — which is how the
last several real defects in this package were found. It needs Spglib, which is
exactly why it is outside `Pkg.test()`: the core suite must never depend on a
symmetry backend, and a hand-assembled group would defeat the tier's purpose
(Spglib is its independent oracle for the space group).

The tier declares a **coverage matrix** — per row, which of the 17 columns run and,
with a reason in the file, which do not. The driver refuses a row that leaves a
column unaccounted for, and refuses to report success if a declared column never
executed. Every column's oracle is independent of the path it checks: the
International Tables, the crystal's own space group (`E(g·config) = E(config)`),
central differences of `predict_energy` and of `affine_energy`, bitwise round trips,
and the reparameterization's own ledger `m = q + rank + frozen`.

Three things it measured on its first runs, each of which had been assumed the other
way while writing it:

- **The freeze is the norm at minimal cell size, not an edge case.** bcc Fe freezes
  14 of its 22 joint columns, B2 FeRh 35 of 46, hcp Co 40 of 57, wurtzite GaN 15 of
  35. A primitive cell has one atom per orbit, so any coordination shell with
  multiplicity greater than one reaches the same atom through several images. The
  2×2×2 tiling of bcc Fe has the *same 22 columns* and freezes none of them, going
  from 4 free parameters to 10.
- **`asr = false` is not a control by itself.** Refitting centre-of-mass-free data
  whose target came from a feasible model returns the same coefficients either way
  (3e-15) — the constraint only bites when the data pull away from it. The tier now
  builds a deliberately violating truth, and separately lets the sample drift, which
  is what makes the violation observable in every row.
- **What the sum rule costs is a property of the row**, ranging from machine
  precision on the heavily frozen cells to 0.69 on the tie-free ones: with
  `Σ_a u_a = 0` the energy block barely sees the violating content while the force
  block does.

### Fixed — a basis with no columns was fitted as an intercept

`SLCEDataset` accepted a basis with **zero** SALC columns. Everything downstream then
reported on the intercept: `r2_energy` exactly `0.0`, an empty `coef`, and a
`predict_energy` returning the same number for every configuration — a silent
constant dressed as a model, with no warning anywhere.

It is reachable from an ordinary spec. The pair a minimum-image convention cannot
express is the same-atom one, so in rocksalt (where the whole magnetic problem is
cation–cation) a superexchange spec on the primitive cell builds nothing at all —
found by the integration tier's `rs-MnO` row on its first run. The dataset boundary
now refuses it and names the three causes (a cutoff below the first admissible
shell; an `lmax`/`pmax` that empties the channel; the same-atom pair, with the
remedy). The refusal is at that boundary and not at construction because
`restrict(model, :spin)` builds an empty basis *deliberately* — the clamped-ion
sub-model of a lattice-only model — and must keep working.

### Fixed — a Wigner–Seitz tie that symmetry does not fuse was invisible to the freeze

The unresolvable-column freeze classified **columns**, and a boundary tie has a second face
that is not a null column. When the point group permutes the tied images they share one
orbit whose sum weights them equally, and odd content cancels — that is the case the freeze
handled. In low symmetry (`P1`, monoclinic) no operation relates them, so they sit in
**different orbits** with independent couplings: every column is individually nonzero, but
the two orbits are the same function of any cell-periodic configuration, so only the SUM of
their couplings is determined and a whole *combination* is flat.

Measured on a `P1` three-atom cell whose pair sits exactly on the WS face: `p = 63`,
structural rank 54, so nine flat directions — all of them ASR-feasible, hence untouched by
`build_asr`. A 400-configuration force co-fit reached `rmse_energy = 4.2e-16` with a clean
`asr_residual`, `identifiability` reporting `nullity = 9` (gap 2.6e14), coefficients off by
1.37 against a truth whose largest was 2.32, `D(0)` exact to 1.2e-15 — and **`D(q)` 52 %
wrong**. `fit` emitted no warning, because its check is per column and each column was fine.

Fixes:

- `_has_boundary_tie` now looks **across** SALCs (is any atom multiset reached by two
  orbits?), not only inside one. The old within-SALC scan returned `false` on the fixture
  above, so the expansion never ran. Under `MinimumImage` the cross-orbit case can only
  arise from a tie — verified that widening the cutoff to admit a farther shell of the same
  pair does *not* produce one, since only the minimum image is enumerated.
- The tied orbits are frozen **whole**: `unresolvable_columns` now returns the union of the
  vanishing columns and every column of every orbit sharing an atom multiset.
  `_unresolvable_split` keeps the two reasons apart, and `build_asr` gives each its own
  message with its own remedy.
- There is no justified split of the determined sum — the two images share a phase only at
  `q = 0`, so equal division is an interpolation ansatz and not a measurement — so the
  interaction is dropped rather than divided. This **discards determined content on
  purpose**: on the fixture the 18 dropped columns are 9 determined sums plus 9
  undetermined differences, and a fit to data containing that shell now reports
  `r2_energy ≈ 0.70` instead of 1. The nonzero residual is the intended signal.
- A cheap structural pre-check still short-circuits the whole thing, and when the
  cross-orbit case does fire the remaining structural rank is verified and reported rather
  than assumed (`residual_flat`).

Standard cells are unaffected, and the reason is symmetry: with space groups from Spglib the
count of null columns already equals the full structural nullity on bcc Fe, B2 FeRh, hcp Co,
wurtzite GaN and rocksalt MnO.

Gate (G) in `test/unit/test_resolvability.jl`; worked example with the numbers in
`docs/src/theory/resolvability.md`.

### Fixed — five of the six read-outs were silenced by the freeze warning's `maxlog`

Julia keys `maxlog` by the log **statement**, so one `@warn` shared by six deliverables at
`maxlog = 1` speaks for whichever ran first. Measured on a B2 FeRh joint basis:
`strain_derivatives` warned, and `magnetoelastic_constants`, `magnon_phonon_vertices`,
`decorated_terms` and `force_constants` all returned frozen-channel results in silence. The
log id now carries the read-out name, so each says it once and none mutes another.

### Fixed — `magnetoelastic_constants` scored an empty problem as a perfect one

`residual = resid / max(‖dev‖, eps())` sent a model with no magnetization-dependent
ε-linear response to `residual = 0.0` — the *best* possible value — beside `B₁ = B₂ = 0`,
indistinguishable from a determined answer. That is the generic outcome on a standard cell
of a cubic magnet, because the ε-linear tier is odd under exchanging a bond's two ends and
the boundary tie annihilates it (measured: every ε-linear column frozen on B2 FeRh,
rocksalt MnO and L1₀ FePt). The degenerate case now reports `residual = NaN`, carries the
projected magnitude in a new `signal` field, and says out loud that zero is an absence
rather than a measurement.

`exchange_strain_derivatives`' `skipped` list was likewise computed *below* its
`jphi == 0.0 && continue`, making "content this view cannot show" a property of one fit's
values: a model whose non-representable channels sat at exactly zero reported
`skipped == []`, a false all-clear that `refit` undoes. Measured 150 channels against a
random model and 0 against the same model with those coefficients zeroed. The
classification now happens above the coefficient filter, as `bilinear_terms` already did.

### Fixed — the fold deal never reached the force channel

`_grouped_folds` spreads a class over folds by dealing consecutive indices, so a *channel*
is spread only if its classes are visited consecutively. Under the `2·torque + force`
packing the torque channel is `{3, 2}` — adjacent, safe by accident — while the force
channel is `{3, 1}`, which descending order splits around class 2. Measured on 1 both /
1 torque-only / 1 force-only / 3 plain: both force-bearing configs landed in the same fold
for every seed at `nf = 2` and `nf = 3`, so a training split had no force rows and `fit`
refused it with "the dataset has no force data" — about a dataset that has it. The silent
face was per-fold scores mixing objectives, with `wF·MSE_F` present in some folds and
absent in others.

`_grouped_folds` now takes a `class_order`, and the two-ragged case asks for `[2, 3, 1, 0]`,
in which both channels are consecutive. Every other occupancy pattern keeps the descending
deal bit-identically, so no recorded seed moves. `select_fit`'s `:cv` branch gained the same
per-channel fold caps (it had none, and being a Gram-downdating path it dropped a channel
from the training design *silently*), and its raggedness test no longer re-derives torque
row counts from `3 · n_atoms`.

### Changed — the objective is MSE at every weight, so `lambda` means one thing

`_assemble_problem` whitened the energy block by `√((1 − w_T − w_F)/n_E)` whenever a
derivative weight was positive and returned the centered design **unscaled** otherwise —
i.e. it minimized the energy SSE at `w_T = w_F = 0` and the energy MSE everywhere else,
while `fit`'s docstring states the MSE form throughout. OLS is unaffected, but every
penalized estimator saw a Gram jumping by `n_E` across that boundary: measured on 60
configurations, `Ridge(lambda = 1.0)` gave `rmse_energy = 0.0038` at `w_T = 0` and `0.086`
at `w_T = 1e-12`. An infinitesimal weight was not an infinitesimal change.

The energy-only branch now applies the same `1/√n_E`. **λ for an energy-only penalized fit
is `n_E` times smaller than before** — recorded λ paths must be rescaled; the test fixtures'
grids were divided by their configuration count.

### Fixed — a model could carry fewer coefficients than its basis has SALCs

`SLCEModel`'s four-field default constructor validated nothing, and `SLCEModel(f::SLCEFit)`
goes straight through it (the three-argument method did check). `SLCEDataset` did not check
`X_E`/`X_T` widths either — only the compact `X_F` was checked — and `_assemble_problem`
reads the design width off `dataset.X_E`. A narrower block therefore produced a model whose
`jphi` was shorter than its basis, and since every read-out loops `eachindex(model.jphi)`,
`predict_energy`, `decorated_terms`, `force_constants` and `effective_model` all silently
answered about a **prefix** of the basis; only `coeftable` noticed. Both widths are now
checked at their constructors. Two test fixtures were relying on the gap — one of them
(`test_sunny.jl`) expressed "only the first SALC" as a one-element `jphi` — and now spell
out full-width vectors.

Also fixed at the same boundary: `restrict(model, :lattice)` threw
`"pmax > 0 needs a sector table with displacement content"` whenever no spin-free
displacement sector existed — which is exactly the minimal magnetoelastic spec — naming a
keyword the caller had passed; the empty lattice sub-model is the right answer, and
`restrict(model, :spin)` already returned its empty counterpart without complaint. The
`BasisSpec` sector form now also rejects `pmax > 0` with no displacement content in any
sector, as the dense form always did: the unreachable cap made `_basis_has_disp` read
`true`, so `predict_energy(model, e)` demanded a displacement field and
`spin_multipole_terms` refused the model. And the three-argument `SLCEDataset` now refuses
empty `configs`, like the four- and five-argument forms.

### Fixed — the dead-column warning reported indices nobody could use

Two cases in `_warn_unidentified`, both silent-by-omission rather than wrong.

A reparameterization with **no constraint rows** is a pure freeze, so `Z` is exactly the
selection matrix of `free` and γ direction `k` is `jphi` column `free[k]`. The warning
reported the γ index anyway — a number indexing nothing the caller holds — and withheld the
`refit`-drops-them advice, which does apply there. It now maps back to β positions and says
so; with constraint rows present nothing changed, because there the two spaces genuinely
differ.

An **all-zero design** returned early ("nothing to rank"), so the one fit where nothing at
all was determined was the one fit that said nothing. Measured on a force-only fit of a bcc
joint basis whose single ASR-feasible direction is pure spin, hence invisible to forces:
every column dead, no warning. It now reports every column.

Gate (d) in `test_resolvability.jl` reads the log record's `columns` and `coordinates`
kwargs and checks both branches against a structural oracle (which columns are pure spin is
a property of the keys, and a force-only fit sees none of them).

### Added — the physical readouts name a channel the reference cell cannot resolve

The freeze is silent in the fit (the coefficient is held at exactly zero, which is the
honest value) and **loud in the readouts**, because they differentiate the individual
cluster members, where the boundary tie does not cancel. Until now nothing said so.

`_warn_unresolvable` reports both directions, once each, from `force_constants`,
`strain_derivatives`, `exchange_strain_derivatives`, `magnon_phonon_vertices` and
`decorated_terms` — the last being the Monte-Carlo hand-off, where the supercell resolves
the tie and the missing channel becomes real physics (`magnetoelastic_constants` inherits
the diagnostic through `strain_derivatives`):

* **coefficient zero** — the deliverable carries no contribution from that channel, and
  that is not zero physics. The remedy is a cell in which the offending pair's minimum
  image is unique, never a wider cutoff.
* **coefficient nonzero** (hand-built, or fixed from elsewhere) — the value reaches `Φ`
  and every `q ≠ 0` of `D(q)` although no data on this cell could have determined it.
  Legal, and now named.

The sharp part, gated in `test_resolvability.jl` (F): the two models — one with the frozen
coefficient at zero, one with it at 0.3 — are indistinguishable in the energy of every
cell-periodic configuration and have `D(0)` equal to 1e-12, while `D(q)` at `q = (0.3,
0.1, 0.25)` differs. `Σ_R Φ(R)` is the Hessian of exactly the energy the cell can express,
so a model can pass every Γ-point check and still carry an unconstrained dispersion.

`unresolvable_columns` gained a structural short-circuit so those call sites are affordable:
a tie is a *necessary* condition (rows are keyed by per-atom content, so members with
different atom multisets can never cancel each other, and a lone member's contribution is
nonzero by construction), and ruling it out costs one pass over the members instead of the
whole monomial expansion. Gate (A) now checks the fast path and the unconditional
expansion against the same evaluator verdict, on tied and untied fixtures.

### Fixed — cancellation residue could become a full-strength ASR constraint

The residue cut in the ASR builder was taken against `A`'s own maximum (per row, then
global). That works while *some* entry is real, and fails completely when none is: on a
Wigner–Seitz-tied cell the differentiated expansion cancels wherever the undifferentiated one
does, so `A` can be residue from end to end — and then the maximum is residue too, every row
looks full-strength relative to it, and `_asr_nullspace`'s row normalization promotes BLAS
rounding to unit-norm constraints.

Measured on the two all-frozen fixtures: a bcc `degree = 3` basis (`max|A| = 8.9e-16` against
a gross accumulation of 18.5) and a bcc spin × `degree = 1` one (`3.6e-15` against `82.1`).
Both kept 24 and 7 "constraints"; `asr_residual` returned **0.238** and **0.360** for
hand-built models on bases whose every column is identically zero on that cell — so the public
verifier that `force_constants`, `dynamical_matrix` and `strain_derivatives` all gate on was
refusing legal models over rounding noise. It now returns `0.0`, and `_asr_matrix` returns no
rows at all.

The fix is the classifier's rule, applied one granularity down: `_asr_expansion` reports the
gross mass `G[r, j] = Σ|contributions|` beside each accumulated entry, and `_prune_residue!`
zeroes an entry iff `|A[r, j]| ≤ _CANCELLATION_RTOL · G[r, j]` — *"did this entry cancel?"*,
never *"is this entry small?"*. One constant, `_CANCELLATION_RTOL`, now serves both readers.

One consequence needed handling: after the prune, an all-cancelling expansion reaches
`rank == 0`, which `build_asr` refused as a broken symbolic expansion. Depositing nothing and
depositing-then-cancelling are different statements — the second is a basis whose free columns
carry difference content only, where rank 0 is the right answer — so the refusal now reads
whether the expansion deposited anything at all, not the rank.

Gated in two places: `_prune_residue!` on hand-written `(A, G)` pairs (`test_asr.jl`), and
gate (E) in `test_resolvability.jl`, which counts real constraints through the *production*
gradient kernel's translation image and compares it to `rep.rank` over the free columns. That
gate carries its own illustration of the same hazard: over *all* columns, the standard relative
singular-value cut reports 8 and 5 constraints on those two identically-zero bases.

### Fixed — the freeze reached only some of the paths that needed it

Two parallel reviews of the freeze found four ways past it and one wrong justification. All
five were reproduced before fixing and are now gated.

**A staged fit re-freed frozen columns, and the value reached a deliverable.**
`sector_columns` resolves selectors from `SALCKey` content alone, so `_fit_stage` handed
back a column whose design entries are identically zero and `_stage_reparam` gave it a free
null-space direction — an unbounded solve. Measured on a two-stage bcc fit: `jphi` came back
with `3.7e14`, `asr_residual` still reported `0.0` (the column is in the ASR's null space, so
it violates nothing), and `force_constants` carried `max |Φ| = 3.0e14`. The stage's free set
is now intersected with the basis-level one, and a mask that names a frozen column says so.

**The pure-spin path never froze at all, and the two `asr` settings disagreed.** The
pure-spin `SLCEDataset` constructors hard-coded `asr = nothing` — correct while `build_asr`
always returned `nothing` for pure spin, wrong now that a pure-spin basis can carry
unresolvable columns (measured: 4 of 7 on a bcc `soc` pair basis). The polarity was
inverted: `asr = true` left ~1e-18 in those coefficients and reported `dof = 8`, while
`asr = false` — which rebuilds the freeze — gave exact zeros and `dof = 4`. ~1e-18 is not
zero for `select_support`'s alive rule or SLCEMonteCarlo's `t.coef != 0.0` prune, so those
columns would have bought site programs and become supercell physics.

**`asr = false` was mistaken for a staged fit.** `_resolve_asr_rep` builds the freeze-only
reparameterization, and `_is_staged` inferred staging from `reparam !== dataset.asr` — object
identity — so `select_support` refused an ordinary ablation fit with a message about staging
that was false. `SLCEFit` now records `staged` instead of inferring it.

**Classification broke the AllImages escape hatch.** `unresolvable_columns` ran before
`build_asr`'s early exits, so its repeated-atom refusal fired on two paths that used to
return immediately: `asr_residual` threw on a legal pure-spin `AllImages` model (was `0.0`),
and an `AllImages` joint dataset became unfittable by any route — `asr = true` and
`asr = false` both threw, the former advising the latter. The refusal is now a dedicated
`UnclassifiableBasis`, caught in `build_asr` (nothing frozen, said once — the honest
"unknown", not "none"), and `asr_residual` builds only the ASR matrix, which also takes the
classification off a public diagnostic's path (measured 40 ms / 70 MiB on a pure-spin basis
of 334 columns, previously free). `select_fit` likewise gates on constraint *rows* rather
than `rep === nothing`, so it no longer refuses every Wigner–Seitz-tied crystal; it pins the
frozen entries exactly, which costs no residual because the design is blind to them.

**The stated reason for keeping the columns was wrong for most of them.** "The same basis
functions are nonzero under a uniform strain" is false for the majority of frozen columns and
*structurally* false for every pure-spin one — such a SALC has no displacement slot for a
strain to act on, so `affine_energy` reduces to `predict_energy`, the annihilated orbit sum.
Measured with 200 draws each: strain-visible for 0 of 1 frozen columns on bcc harmonic
(the row the chapter led with), 0 of 4 pure-spin, 6 of 8 at degree 3, 4 of 10 on
spin × degree-1. The argument that holds unconditionally is the **supercell** one — tiling
maps the tied images onto distinct atoms — and all four sites now lead with it. The
`dJ/dr` / exchange-magnetostriction reading is dropped: where the strain response survives it
is the response to the relative bond-vector change `M·d`, whose transverse part rotates the
bond rather than stretching it. The exchange-parity account of ASR-dead pair channels is
qualified with its precondition (an operation must exchange the bond's ends, and both ends
must carry the same spin rank); on a heteroatomic bond the partner can be the same channel
with the displacement on the other end.

Known and not fixed here: `_asr_matrix` prunes cancellation residue relative to its own
global maximum, so an `A` that is *entirely* residue cannot be detected — row normalization
then promotes noise to unit-norm constraints, and `asr_residual` returns `0.299` for a
hand-built model on a bcc spin × displacement basis where `max |A| = 3.6e-15`. Pre-existing,
newly reachable, and the fix is the same gross-scale comparison the classifier uses.

### Added — the columns a reference cell cannot resolve are classified and frozen

`unresolvable_columns(basis)` names the design columns that are identically zero on every
cell-periodic configuration the reference cell can express, and `build_asr` now excludes
them from the reparameterization's free set, so every fit holds them at exactly zero. They
are **unidentifiable, not physically zero**: the same basis functions are nonzero under
`affine_energy` and in a Monte-Carlo supercell, which is why the columns stay in the basis
and why leaving them free was the actual hazard. Measured before the change, on a bcc
degree-2 basis: a fit hands such a column an arbitrary value, energies and forces vanish for
every trainable configuration and `D(0) ≡ 0` — so the acoustic-zero gate passes *vacuously* —
while `‖D(½,½,0)‖ = 91.4` with eigenvalues ±45.72. The freeze applies under `asr = false`
too: being unidentifiable on this cell is a property of the basis, not of what the fit
imposes.

The classification is structural — every SALC is expanded into one common monomial/symbol
basis (the machinery `_asr_matrix` differentiates, undifferentiated) and each column's
assembled norm is compared against the gross magnitude it accumulated, which asks "did this
cancel?" rather than "is this small?". It replaces `_identically_zero_salcs`, a
three-seeded-probe heuristic in the fitting layer, whose relative-to-the-largest-column cut
was also blind to the case where *every* column cancels. `fit`'s surviving dead-column
warning is now purely about data starvation.

Two consequences fall out. `build_asr` no longer throws "the symbolic expansion is broken"
on a basis whose only ASR-feasible direction was an identically-zero function (measured: bcc
harmonic got `Z = [0; 1]`, i.e. the only feasible model was a multiple of a null function);
it now reports the truth, that the surviving free columns admit no translation-invariant
content. And `group_costs`' structural discount stops pricing null columns at full
Monte-Carlo cost, because a frozen column has an all-zero `Z` row.

Gated in `test/unit/test_resolvability.jl` against the evaluator on six fixtures (hand-written
space groups, so the core test environment needs no symmetry backend), plus: the rank
deficiency of a sampled value matrix equals the number of classified columns on every fixture
— the property that makes freezing whole *columns* exact rather than approximate — exact
ε-linearity of a frozen column's strain response, the `2×2×2` remedy versus the insufficient
single-axis doubling, and byte-neutrality where nothing is frozen (`Z` bitwise unchanged).

### Changed — a Wigner–Seitz boundary tie is not independently resolvable

[Periodic resolvability](docs/src/theory/resolvability.md) claimed that the equidistant images
kept on the WS boundary are "each a genuinely different, independently resolvable geometry".
The first half is right and the second is backwards: all images of a tie join the *same two
atoms of the reference cell*, a `TrainingDatum` carries one spin and one displacement per
reference-cell atom, so every tied member sees identical arguments and whatever content is odd
under the operations permuting the ties can cancel in the orbit sum. A tie is necessary — with
a unique minimum image nothing can cancel — but which content dies is a property of the whole
group, not of the tie alone, so the chapter now states the measured table and points at the new
`unresolvable_columns`. Measured: the bcc cross pair (tie 8) loses 1 of 2 harmonic columns,
8 of 8 at degree 3, and 4 of 7 pure-spin `soc` columns; a pair at half a lattice vector (tie 2)
loses all 4 spin × degree-1 columns, which are exactly the odd-`Lf` ones, and none of its 6
harmonic columns, which are all even-`Lf`; an hcp pair loses 1 of 4. The two-fold pattern is
what bond reversal `(-1)^Lf` predicts, and the chapter says explicitly that this is a guide to
where to look rather than a rule — a larger tie removes even-`Lf` content too, and supplying a
subgroup instead of the crystal's own group leaves odd-`Lf` content standing.

Those coefficients are unidentifiable, not absent — the same functions are nonzero under
`affine_energy` (the relative, bond-stretch content survives a uniform strain) and in a
Monte-Carlo supercell. The chapter now states the cost, and states the remedy correctly: the
tie must be broken in *every* direction whose separation component is half the cell length. A
cell doubled along one axis is not enough — for bcc it only takes the corner tie from eight
images to four and the dead channels stay dead; `2×2×2` resolves it.

### Changed — the ASR diagnostics say who can fix them, and say it once

`build_asr` gained a `warn` keyword and both of its *diagnostics* (no translation-invariant
displacement content at all; individual structurally zeroed columns) now carry
`maxlog = 1`. `asr_residual` passes `warn = false`, because it **re-derives** the
constraint rather than constructing it — every derived quantity gates on it, so one true
statement about the truncation was being reprinted once per output (38 times on one
documentation page). The refusals are untouched: a broken symbolic expansion or a forbidden
band still throws, and `warn` cannot silence them.

The dead-column message also named the wrong remedy. It said the partners are "outside the
truncation", which invites widening the cutoff — correct for a spin-free displacement
channel, useless for a spin-dressed one. The sum rule holds separately in each spin sector,
so a spin-dressed column survives only if some term dresses the *same* spin invariant
differently (a displaced ligand, say). A *pair* channel whose exchange parity — the parity of
its spin factor times `(-1)^Lf` — comes out even can only couple to `u_a + u_b`, and no extra
pair orbit or wider cutoff changes that parity, so its partner has to come from another
sector: a displaced third atom. Either way the coefficient is excluded by the sum rule, not by
crystal symmetry; the basis function itself is generally nonzero (order 0.3 on the
magnetoelastic fixture in the strain guide, whose `L_S = 1` column is dead in exactly this
way, and still dead when the cell is doubled). Measured while writing this: a spin-dressed
`degree = 2` pair sector has 4 structurally dead columns per pair orbit with `sites = 2`
(12 on a three-orbit fixture) and **none** with `sites = 2:3`, i.e. once the ligand is in.

### Added — magnetoelastic coupling, with the B₁/B₂ convention pinned

Two deliverables on top of the ε-linear strain response, and deliberately none above it:

`magnetoelastic_constants(model; signs, tol)` returns the cubic constants as
`(; B1, B2, ion = :clamped, residual, volume)` in one pinned convention,

```
E_me / V = B₁ Σ_i ε_ii (α_i² − 1/3) + 2 B₂ Σ_{i<j} ε_ij α_i α_j
```

with **tensor** shear `ε_ij`, that summation range, that sign, and `E_me` an energy
*density*. Every one of those is a convention that differs between papers — a factor of
two on B₂ for engineering shear, another for `Σ_{i≠j}`, a shifted B₁ without the `−1/3`,
an overall sign — and none of them was pinned anywhere in the package before: the existing
magnetoelastic gate fixes the *span* of the block and says in its own comment that it
fixes no normalization. The pin is now gated (design record §12 gate (u)) against a closed
form written out by hand and evaluated through the production SALC evaluator, never
against the deliverable's own path.

`ion = :clamped` is a **field of the result**, not a sentence in the docstring: no
internal-strain relaxation `Λ = −Φ⁻¹Ξ` has been applied, the clamped-versus-relaxed
difference is routinely a factor ~2 and can flip a sign, and a bare clamped-ion B₂
compared against experiment is a wrong number rather than an approximate one.

The constants are obtained by **projection, not readout**: the exact ε-linear response is
computed at 19 magnetization directions and least-squares-fitted to the two-constant cubic
form with the α-independent part free. Two well-chosen directions would return numbers for
a model that is not of that form; the projection returns `residual` — the fraction of the
magnetization-dependent response the form does not explain — and warns when it is large. A
non-cubic crystal, `l = 4` spin content, or a magnetic state whose sublattices do not share
an axis all surface there instead of in a plausible-looking pair of constants.

`exchange_strain_derivatives(model)` keeps the resolution instead of compressing it:
`∂M_ab/∂ε` for every bond and `∂A_a/∂ε` for every single-ion matrix — the model's `dJ/dr`
resolved per bond and per strain component, in the form a spin Hamiltonian consumes. It is
gated by **reconstruction**: contracting every bond with a spin configuration rebuilds the
cell's total ε-linear response, which `strain_derivatives` computes without ever touching
the tesseral-to-Cartesian conversion — the fence `_reconstruct_energy` puts under
`bilinear_terms`, one derivative up.

The per-bond split needs a caveat the total does not. A strain moves each site by
`ε·(R_s − origin)`, and the acoustic sum rule cancels that origin from the *total*, not
bond by bond: a bond whose displacement content is not purely relative carries an absolute
position of its own. Symmetry usually rules this out — a bond orbit with a site-swap
operation admits only `u_b − u_a` — so the check passes silently on every fixture tried,
including the split-home chain that `strain_derivatives` refuses at `order = 2`. It is
measured on every call anyway, and a disagreement is refused.

Why the tier stops at first order, twice over: ε-linear content is Seth–Hill
measure-independent unconditionally (second-order elastic constants agree across measures
only at a stress-free reference, and a spin–lattice cell can be stress-free for at most one
magnetic state), and the sum rule buys origin independence only where the affine field is
periodic, which is `ε = 0`.

### Fixed — the term list depended on the coefficient values

`decorated_terms` and `spin_multipole_terms` skipped any SALC whose coefficient was exactly
zero. That makes the emitted list — and the index → SALC map a consumer addresses it by —
a function of the coefficient *values* rather than of the basis. It is harmless for a
reader and wrong for anything that rewrites coefficients in place: sparse estimators and
`refit` produce exact zeros routinely, so two models built on the *same* basis (two points
of a `StrainedModels` grid, an active-learning refit) can emit lists of equal length whose
maps are shifted relative to each other — and a coefficient hot-swap then writes each value
onto a neighbouring cluster while every length check passes.

Both surfaces now take `keep_zero`, which emits one term per SALC member unconditionally
and makes the index map a property of the SALC basis alone — the object a volume grid
already asserts identical across its points. The default is `false`, so every existing
consumer, byte-comparison and benchmark is unaffected; a consumer that will rewrite
coefficients must opt in. The test demonstrates the hazard next to the fix rather than
asserting the fix alone.

### Added — volume grids, and the acceptance gate that catches what they get wrong

`StrainedModels(models, scales)` holds a K(ε) grid: the same crystal at a set of isotropic
linear scales `s = 1 + η`, with `model_at(sm, s)` interpolating the coefficients and
`grid_strain_derivative(sm, s)` differentiating along the grid. Isotropic scaling is the one
strain family that preserves the point group, so the SALC keys survive and coefficients can
be interpolated at all.

The constructor refuses a grid that cannot be interpolated: cells that are not similar,
cutoffs that do not scale with `s` (an *absolute* cutoff lets a neighbour shell cross it as
the volume changes, and the key sets then diverge with nothing to say so), a `disp_scale`
that moves, a different key set, different member images — the *home-image condition*,
which the key set is blind to — and a different **SALC gauge**, compared on the folded
tensors. That last one is the quiet failure the others miss: the projector fixes each
invariant only up to a sign, and a coefficient interpolated against a sign-flipped basis
function is wrong while every key matches and every per-point fit stays perfect.

The acceptance gate is that the two strain derivatives on a grid agree — the intra-model
incremental one, taken inside a single model with its coefficients fixed, and the grid
finite difference, which also carries their drift. Building it caught two real things, both
of which the design record predicted this one test would find:

- **Basis truncation.** A grid's basis must be closed under *re-expansion*. A model at
  scale `s` is the reference re-expanded around the scaled geometry, and that shift is
  lower-triangular in degree — `degree = 2` content generates `degree = 1`, and a
  spin-dressed `degree = 1` sector generates a pure-spin term. Measured: 5% disagreement
  with the pure-spin sector missing, 1% with the lattice `degree = 1` sector missing, 4e-7
  with both present.
- **A mislabelled η is a factor of exactly `s`.** `dE/dη = s·dE/ds`, because `η` is measured
  from the reference the derivative is taken *at*. Dropping the factor agrees at the
  unstrained point and is off by the strain everywhere else.

The residual 4e-7 is the coefficient interpolation error and nothing else, which halving the
grid span demonstrates.

The interpolation **abscissa** is a modelling choice — and measuring it (leave-one-out
against a directly fitted point) settled what the default should be. The map relating two
grid points is a re-expansion, which is exactly polynomial in the affine displacement and
therefore in the *linear scale*, of the same degree as the displacement content. So
coefficients are polynomials in `s` and are interpolated exactly, while in the volume they
are polynomials in `s³`: 6e-14 in `:linear` against 2.5e-8 in `:volume` on a controlled
fixture. The default is `:linear`; `:volume` remains the right choice when the object of
interest is an equation of state rather than a coupling.

### Added — magnon–phonon vertices

`magnon_phonon_vertices(model; spins)` returns the mixed second derivative
`∂²E/∂u_{aα}(0) ∂e_{bβ}(R)` — the object that couples the two subsystems, where the force
constants differentiate twice in `u` and the bilinear couplings twice in the spin
directions. Contract one index with a phonon eigenvector and the other with a magnon
polarization and it is the one-magnon–one-phonon vertex.

The spin derivative is **tangential**: the radial part is projected out (the same
`grad_Zlm` the torque design matrix is built from), so `V·ê_b ≡ 0` identically. That is
what lets the result come back in plain Cartesian components without pinning anyone's
local-frame convention — project onto whichever `(f₁, f₂)` your magnon basis uses. The
pair `(a, b, R)` is *ordered*, `a` displaced and `b` magnetic, unlike the undirected bond
keys of `bilinear_terms`.

Only terms with exactly one degree-1 displacement factor *and* a spin factor contribute, so
a magnetoelastic sector declared at `disp = (degree = 2,)` produces no vertices at all —
that content feeds the spin-dependent force constants instead, which is the trap
`force_constants` warns about seen from the other side.

Gated against a mixed central finite difference of the production evaluator, plus the two
structural properties: tangency, and the translation sum rule (a rigid translation cannot
change the energy, hence cannot change its derivative with respect to any spin).
"Adiabatic" in the docstring is a scope statement rather than a hedge — these are
derivatives of the static energy surface, and retardation, phonon angular momentum and
spin-lattice relaxation are outside a static cluster expansion by construction.

### Added — the homogeneous-strain response, exactly

`strain_derivatives(model; spins, order, origin, symmetrize)` returns the exact
`∂ⁿE_cell/∂εⁿ` at the model's own reference geometry: `order = 1` is the reference
stress times the cell volume, `order = 2` the **clamped-ion** elastic tensor times the
volume, and both are functions of the magnetic state — which is what magnetoelasticity
is. The strain measure is Biot (Seth–Hill `m = 1`, `F = I + ε`), so `u_i − u_j = ε·d_ij`
is exact by definition rather than a linearization.

It is exact for the same reason the force constants are: every displacement site factor
is a homogeneous polynomial, so substituting the affine field turns a term of
displacement degree `n` into a degree-`n` polynomial in `ε`. The order-`n` derivative
therefore draws from degree-`n` terms alone, and the whole Taylor series holds at
**finite** strain — the acceptance gate contracts it against `affine_energy` at a
100%-scale affine map, where a missed factor cannot hide inside a truncation error.

**The acoustic sum rule is enforced here, not merely recommended.** A strain displaces
site `i` by `ε·(R_i − origin)`, so moving the origin changes the energy by
`ε : (Σ_i ∇_i E)`: without `A·β = 0` there is no origin-independent strain response at
all, and the answer is undefined rather than inaccurate. `strain_derivatives` throws, at
a tolerance tighter than the 1e-10 quoted elsewhere, because the strain path weights the
residual by `|R_i|`.

**And at `order ≥ 2` the sum rule is not sufficient — a new finding.** The ASR is the
identity `Σ_a ∂E/∂u_a ≡ 0` in the *atom* variables, i.e. on cell-periodic fields. At
`ε = 0` the affine field is zero, hence periodic, so the ε-linear response is
origin-independent unconditionally. One order out the field is not periodic, and the
stronger per-*site* identity would be needed. The gap is the same home-**image gauge**
the rotational diagnostic turned up, now affecting a physical output: the same dimer
chain described with its bonded partner in the home cell is origin-independent to
1e-13, and described with the bond crossing the cell edge is off by a factor of 8.
`strain_derivatives` therefore recomputes `order ≥ 2` at a shifted origin and **refuses**
a disagreement, naming the crystal description — not the fit — as the thing to change,
rather than returning a description-dependent elastic constant behind a caveat.

`force_constants` is unchanged numerically; it now shares its spin-resolution rule
(`_resolve_spins`) and its spin-blindness predicate (`_spin_blind_at_order`) with the
strain path, so the two cannot drift.

### Added — rotational invariance is now measurable (a diagnostic, not a constraint)

`affine_energy(model, spins, M; origin, base)` evaluates the model under an **affine**
displacement field `u(R) = M·(R − origin) + base`, resolved at each cluster site's own
equilibrium position (image shift included). This is the path `predict_energy` cannot
express: it reads a site's displacement as `u[:, atom]`, so the fields it accepts are
cell-periodic — and translation is the one affine field that is, which is exactly why
the ASR was testable through the ordinary predictors and rotation was not. At `M = 0`
the two agree bit-identically; on an ASR-satisfying model the result is independent of
`origin` to `1e-15`.

Two diagnostics ride on it. `rotational_residual` rotates the lattice rigidly by a
**finite** angle and reports the cost as a fraction of a same-size deformation
(design record §12 gate (q)); `rotation_transfer_residual` rotates spins and lattice
together and reports how much of the two halves fails to cancel — the SOC rotation
law `𝓡_U E = −𝓡_S E`, previously derived and unverified in code (gate (r)).

Nothing is imposed. Translation remains the only affine invariance this package
constrains; see `docs/design-notes.md` for why (a continuous rigid rotation is not a
space-group operation) and for the three findings the measurement produced — the
linearized test is *blind*, not weak; a truncated model's residual vanishes with `ω`
rather than being zero, so the decay rate is the signal; and on-site displacement
content carries a home-image gauge that periodic training data cannot fix.

### Fixed — constrained fits left numerical junk on forbidden directions

`fit` and `refit` lifted `γ` to `β` as `beta_p + Z·γ`. A column that no feasible model
can carry has a **numerically** zero row of `Z` (~1e-16 out of the SVD), not a
structurally zero one, so the product left ~1e-15 to 1e-16 on directions the constraint
forbids. Both lifts now go through one `_lift_gamma`, which snaps those rows back to the
particular solution — exactly `0.0` under a homogeneous reparameterization, and the
particular-solution value on an affine stage, where a `Z`-zeroed free column may
legitimately carry one.

This is not cosmetic, because two consumers test coefficients **exactly**:
`select_support`'s alive rule (`!= 0.0`) charged the group's full Monte-Carlo cost for
the junk, and SLCEMonteCarlo's term prune (`hamiltonian.jl`, `t.coef != 0.0`) would have
bought it a complete set of site programs. Measured on the D4h fixture, 171 of 198
realized supports carried at least one such column (worst |jϕ| = 3.8e-15), and 31 of 198
reported a different alive-group set than the honest one — at only `G = 4`. The
basis-level case is separate and was also live: `fit` itself left ~1e-16 on the four
columns `build_asr` warns are structurally zeroed for every support.

Found by review, not by the test suite — the front's own test re-derived the alive set
with the same predicate, so it was self-consistent by construction, and the fixture's
supports happened to land in the regime where the dead row *is* exactly zero (it is,
when the column's whole connected component of `A[:, S]` is full rank). Both cases are
now pinned: `fit`'s output on the basis-dead columns, and a no-junk invariant at every
point of a constrained front.

### Changed — `select_support` runs under an ASR, and prices it honestly

`select_support` no longer refuses a plain ASR-constrained fit. The refusal was
conservatism about the *reporting*, not about the fit: the engine it drives is `refit`,
which already re-derives the null space on each support and is already gated on
translation invariance and `Σf = 0` after the fact. What was actually wrong is that
`n_alive` and `cost` were computed from the *pre*-threshold magnitudes `m_g > t`; under
a constraint those over-report, because a support that splits a constraint-coupled
column set structurally zeroes some survivors. Both columns are now derived from the
**returned refits** — the convention the unconstrained front was already pinned to — so
the reported cost is the cost of the model handed back. The staged (`frozen` /
`sector_mask`) refusal stays; `select_fit`'s ASR refusal stays, because its λ path
genuinely does solve on an unconstrained Gram.

`group_costs` gains an opt-in `asr` keyword that drops **structurally infeasible
columns** before the entry union, so a group keeping some feasible columns loses only
the entries its dead ones contributed uniquely, and a group that is dead throughout costs
zero. Not a rounding error: on a `pmax = 1` spin+displacement truncation whose
displacement content the ASR kills outright, the dead group carried 94.7 % of `Σ_g c_g`.
`select_support` passes the fit's own constraint by default; a cost computed from the
basis alone still ignores it, so the plain call stays a property of the basis. An affine
(staged) reparameterization is refused rather than mispriced — there a `Z`-zeroed column
can still carry a particular-solution value, so dropping it would *under*-report.

Two claims made in an earlier draft of this entry were wrong and are corrected here. The
post-refit and pre-threshold rules agree only for an estimator that does not itself
produce exact zeros (OLS, the ridge family) — a sparse `estimator` legitimately zeroes
support columns, where the new rule is simply the better one. And `AdaptiveRidge` /
`GroupAdaptiveRidge` never return exact zeros at all, so a front de-biased with one of
them reports every support column alive and its `cost` column degenerates; that caveat is
now in the `select_support` docstring.

New public helper `SLCE.group_freedom(rep, column_groups)` = `s_g = ‖Z[g, :]‖_F²`, with
`Σ_g s_g ≡ q` exactly and `s_g = 0` ⟺ the group is structurally zeroed. It is
gauge-invariant despite being written in `Z` (it is a trace of the projector `Z·Z'`) and
needs no fit.

**Measurement that decided this, recorded because it closes two design questions.**
On five fixtures with `G` from 2 to 20: not one displacement-touched group has a feasible
subspace on its own (`dim null(A[:, cols_g]) = 0` in every case); the constraint's true
all-or-nothing atoms are the circuits of `A`'s column matroid, 2–3 columns each but
spanning several groups; and closing the groups under `A`'s connected components
collapses them to **two** clusters regardless of `G` (K/G 0.50 → 0.12 → 0.10 as G grows).
So (i) component closure is rejected — it destroys the cost axis rather than repairing
it, and (ii) the group axis is intact on the pure-spin side (identity blocks in `Z`,
untouched by the ASR) and close to binary on the displacement side. Both facts are now in
the `select_support` docstring and design record §13. A constrained λ path is *not*
scheduled off the back of this: §13's own postscript records that on production l044
nothing dies along the λ path at all, while the threshold front delivered 38 % of the
Monte-Carlo cost at a better held-out RMSE.

Also in this change: `_ASR_DEAD_ROW` is one named constant for the structurally-zeroed
column rule (`‖Z[j, :]‖ < 1e-12`) that four sites had been spelling as a literal, and
design record §13's cost definition is brought up to date — it still described the
pre-2026-07-28 `(body, orbit_id, ls)` grouping and an entry *count*, where the code
groups by `decors`, keys on slot labels, and sums slot counts; the lower-bound caveat and
a correction to the fixed-point argument (the IRLS descends `Σ_g v_g·log(‖β_g‖² + p_gε)`,
whose per-group price carries a log factor spanning ~0.3–18, not the flat `λ·v_g` the
majorizer suggests) went in with it.

### Added — the selection layer takes `force_weight`

`select_fit` and `cross_validate` gain `force_weight`; `select_support` reads it off
the fit and no longer refuses a force co-fit. All three score the fit's own three-block
objective `(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F`, and `rmse_force` joins
`rmse_energy` / `rmse_torque` on `SupportPath` and `CVResult` (both Tables sources —
their column tuples grew) and in their `show` methods. The three RMSE axes are in
different units (eV, eV, eV/Å) and are only comparable against themselves.

Three decisions worth stating, because each had a defensible alternative:

- **At `force_weight = 0`, a force-carrying dataset behaves exactly as one with the
  forces dropped — including the fold deal.** Force presence enters the fold strata
  only at `w_F > 0`, deliberately unlike torque (which stratifies at any weight, and is
  grandfathered). Stratifying on a zero-weight channel would have changed the deal, and
  therefore the score, of every force-carrying dataset at `force_weight = 0` —
  invalidating every recorded CV number to buy nothing but a more even `rmse_force`.
  A pinned test caught this and is what forced the decision.
- **A channel absent from a holdout fold is omitted from that fold's score, not counted
  as zero and not renormalized away.** Renormalizing would put a fold that lost a block
  on the same scale as one that kept it, making the per-fold column incomparable. The
  fold count is instead capped by each weighted channel's configuration count
  (`min(nfolds, n_torque, n_force)`) so the case stays rare and loud.
- **At `torque_weight + force_weight == 1` pure-spin groups cost nothing and are never
  alive.** The energy block has zero weight there, so those columns are structurally
  absent from the design — correct (a derivative-only fit genuinely cannot see them),
  but it means the front is over the derivative-visible model only. Documented in
  `select_support`, along with the fact that the relative alive floor spans blocks of
  different units in that regime.

`_grouped_folds` now takes integer stratum labels instead of a `Bool`, dealt in
**descending** class order with one running counter across classes (`cross_validate`
packs them as `2·torque + force`, scarcest channel high). Both properties are
load-bearing: a force-free dataset degenerates to the historical `(true, false)` deal
bit-identically, so recorded seeds keep their folds.

Also fixed as part of this: `select_support` assembled its per-group magnitudes with
`_assemble_problem(f.dataset, w)` while `refit` assembles with `(w, wF)`. Under a force
co-fit the energy block's whitening differs between the two, so the magnitudes — and
every threshold derived from them — were in different units from the threshold handed
to `refit`. `select_fit`'s local effective row count had the same shape of bug
(`1 - w` where the assembly uses `1 - w - wF`).

Still unimplemented: selection *under* an ASR reparameterization (dual β/γ assembly).

### Fixed — the group cost model priced the wrong Monte-Carlo program

`group_costs` counted one entry per nonzero tensor element. That is the size of
`SLCEMonteCarlo`'s **energy** program, which `total_energy` walks **once per run**.
What a sweep walks is the **site** programs, and `_push_term_programs!` emits one per
member site position — `nnz · length(slots)` entries, every sweep. The cost is now the
summed slot count over distinct entries.

The consequence was a live mis-ranking in the shipped cost-weighted selection: at equal
entry count a 3-body group was priced at 2/3 of its real sweep cost relative to a 2-body
group, so `select_fit` / `select_support` systematically preferred keeping high-body
groups. Absolute costs change; only ratios matter to selection, and those change too.

- **`group_costs` now accepts joint (displacement-decorated) bases.** The entry key
  moved from the pure-spin per-site `ls` to per-slot `(channel, site, k, l)` labels,
  which is what makes a joint basis well-defined at all: keying on `ls` drops every
  displacement slot, collapsing distinct lattice groups onto one key and destroying the
  additivity the selection relies on. The key mirrors the shape of SLCEMonteCarlo's
  `_reduced_key_type(::Type{DecoratedTerm})`.
- The slot factor is read **per entry**, off the key, not applied as a per-group
  constant: `column_groups` may be coarser than `salc_groups`, and only a per-entry
  factor stays additive under that.
- **Two claims in the docs were wrong and are corrected**, not just reworded. There is
  no adjacency merge on the entry key — `decorated_terms` emits one term per
  `(SALC, member, SALCTerm)` without merging, `TiledHamiltonian` tiles without merging,
  and `reduce_cell` buckets on the key then splits each bucket by `(coef, folded)`,
  merging *translation orbits*, not channels. And the one cross-package check on record
  (`Σc_g = 744,636` on l044) was against the term tensors' `Σnnz`, i.e. against the
  energy program — it confirmed internal consistency, not the sweep cost, and needs
  redoing.

`group_costs` now also states what it deliberately does **not** model: the per-visit
`O(nrows)` block (`nrows` comes from `row_layout` over basis *keys*, so it never shrinks
when a group dies — selection-invariant and unattributable to any group), and sweep
multiplicity across the spin / overrelaxation / displacement passes (run-time
`UpdatePlan` counts, with `site_has_*` predicates that depend on which *other* groups
survive, making the true cost supermodular).

### Fixed — `select_support` told unstaged callers their fit was staged

The staged-fit refusal keyed on `f.reparam === nothing`. But `fit` stores
`reparam = _resolve_asr_rep(dataset, asr)`, so **every** plain fit on a joint basis
carries one, and an ordinary `fit(SLCEFit, joint_ds, est)` was reported as a
"staged fit (frozen / sector_mask)". It was masked only because the ASR refusal above it
fired first. The distinction is now one named predicate, `_is_staged(f)` —
`f.reparam !== nothing && f.reparam !== f.dataset.asr` — with the three states
(unconstrained / plain ASR / staged) spelled out where it is defined.

### Added — `select_fit(...; asr)`, and joint selection when the fit is unconstrained

`select_fit` and `select_support` refused *any* ASR-carrying dataset. The λ path solves
on a cached unconstrained Gram and decides alive groups by a β-indexed magnitude rule,
so that refusal is right for a **constrained** fit — but a displacement basis carries an
ASR by default, which made every joint model unselectable even when the user had
deliberately opted out with `asr = false`.

- `select_fit` gains `asr::Bool = true`, threaded to `fit` exactly as `cross_validate`
  already threads it, including the cold re-solve of the selected λ. `asr = false`
  now selects a deliberately unconstrained joint model end to end.
- Both refusals now say *why* (unconstrained Gram, β-indexed alive rule) and *what to
  do*, instead of "not implemented yet".
- The refusal tests were pinned on exception **type** only, so the `select_support`
  fixture was a force co-fit and had been exercising the force refusal, not the ASR one,
  the whole time. The messages are now pinned by text.

Selection *under* the constraint (dual β/γ assembly, `nullspace` into the solve) and
`force_weight` in the selection scorers remain unimplemented.

### Changed — BREAKING: the fourth naming batch, single-package names

The audit's mid tier: names inside one package that say the wrong thing about what
they hold. Three of the candidates were dropped on inspection and are recorded here
so the question is not reopened.

- **`SlotRef` → `Slot`** (`public`, and `DecoratedTerm.slots` / SLCEMonteCarlo's
  adjacency builder read it). It is not a reference to a slot — it *is* the slot
  (site index + site factor), as its own docstring said. `Ref` additionally means
  a mutable box in Julia (`Base.Ref`), which this is not. SLCEMonteCarlo.jl moves
  with it.
- **`PrecomputedPilot` → `FixedCoefficients`.** The type holds a fixed coefficient
  vector and returns it from `solve_coefficients`; being an `AdaptiveLasso` pilot is
  where it is first plugged in, not what it is — and it is already used as a plain
  fixed-coefficient estimator in the fit, ASR and selection suites.
- **`SelectionPath` → `LambdaPath`.** `select_fit` and `select_support` are two
  selection paths; the λ sweep had claimed the generic word, leaving its sibling
  `SupportPath` looking like a different kind of thing. Now each is named for what it
  sweeps.
- **`sector_mask = :soc` → `:soc_only`.** `Sector(soc = true)` *admits* the
  spin-orbit channels — a superset that still contains the `L_S = 0` columns — while
  the mask selector picks the `L_S ≠ 0` columns *alone*. One word for the two
  opposite meanings, and picking the wrong one silently freezes the columns the
  caller meant to fit.
- **`BasisSpec.sectors` → `BasisSpec.sector_rules`.** The keyword `sectors = ...`
  takes `Sector` **sugar**; the field holds the resolved dense `SectorRule` rows. The
  shared name blurred exactly the resolution boundary the spec's design depends on.
  The keyword is unchanged, and so is the persisted document key `"sectors"` — no
  saved model is affected.
- **`restrict(model, channel)` → `restrict(model, sector)`** (parameter name only,
  not the call syntax). `:spin` / `:lattice` are the selector names
  `SLCE.sector_columns` already uses; calling them "channels" pointed at the
  `Channel` enum, whose members are `SPIN` / `DISP` / `OCC` and label a *site
  factor*, not a whole SALC.
- **`select_support(; thresholds = 25)` → `select_support(; npoints = 25)`**, with
  `thresholds` now taking the explicit vector only. One keyword meant "25 grid
  points" for an `Integer` and "these absolute thresholds" for a vector, so
  `thresholds = 10` — "sweep down to a magnitude of 10" — silently became a ten-point
  grid, distinguishable from `thresholds = [10.0]` only by the literal's type. It is
  now a `TypeError`.

- **`theta` → `cost_exponent`, `delta` → `score_rtol`** (`cost_weights`,
  `GroupAdaptiveRidge(basis; …)`, `select_fit`, `select_support`, and the
  `LambdaPath` / `SupportPath` fields and `show`). These were bare Greek letters
  carried straight out of the implementation plan into the public API. `theta` is
  purely the exponent in `v_g = √p_g·(c_g/c̄)^θ` — `cost_exponent = 0.5` tells the
  reader the functional form, `theta = 0.5` tells them nothing — and `delta` is the
  relative tolerance in `score ≤ (1 + δ)·min`, which is what this package spells
  `rtol` everywhere else (`_SAME_DIST_RTOL`, `_DEAD_COL_RTOL`). The symbols `θ` and
  `δ` stay in the formulas and in `docs/design-notes.md` §13, now with the
  keyword-name mapping written next to them: renaming the math would make the
  derivation unreadable.

Dropped after inspection, deliberately unchanged:

- `write_phonopy(; comment)` vs `write_alamode(; description)` — not the same field.
  `comment` is the POSCAR's comment line (what the VASP format calls it); `description`
  fills alamode's `<OriginalXML>` element. Each is named after its host format.
- `images = MinimumImage()` vs the type `AbstractImageSelection` — the call site reads
  better than `selection = MinimumImage()` would, and no name is wrong.

### Added — the documentation is published

- **<https://tomonori-tanaka.github.io/SLCE.jl/dev/>** — the Documenter site is
  now deployed to GitHub Pages by the `documentation build` CI job (`deploydocs`
  in `docs/make.jl`, `permissions: contents: write` on the job). It was being
  built on every push and then thrown away.
- **Per-line source links work**: `remotes = nothing` / `edit_link = nothing` are
  gone in favour of the real repository, so every docstring on the site links to
  its own lines on GitHub and each page has an "Edit on GitHub" link.
- README carries a docs badge, a CI badge and the site URL.

### Changed — BREAKING: the third naming batch, across the four-package family

The audit's cross-package tier: names that are only wrong when you look at two
packages at once, which is why none of the four suites could see them. Landed together
in the four repositories.

- **`MultipoleTerm` / `multipole_terms` → `SpinMultipoleTerm` /
  `spin_multipole_terms`.** This is the **pure-spin** term view — it refuses a
  displacement-decorated model — and its general successor is
  `DecoratedTerm` / `decorated_terms`. The unqualified name promised the general
  surface and delivered the restricted one, with the restriction announced only by an
  error message. Renamed as a pair: a `SpinMultipoleTerm` returned by
  `multipole_terms` would be half a rename. SLCEMonteCarlo.jl and SLCETools.jl move
  with it.
- **`KB_EV` and `resolve_kt` now live here** (`src/units.jl`, `public`, unexported).
  SLCEMonteCarlo and SLCETools each carried a private copy — character for character
  identical, which is what makes the duplication dangerous rather than obviously
  broken: two copies of a unit conversion can drift while both suites stay green.
  The fitting core has no temperature of its own; what it owns is the *convention*,
  and it is the one package all three samplers already depend on. SLCEMonteCarlo
  re-exports `KB_EV` unchanged, so its users see no difference.
- **The acronym is `SLCE`, expanded "spin–lattice cluster expansion"** — the name
  ratified in `docs/specs/spin-lattice-ce-design.md`. Prose across the family still
  said `SCE`, expanded three different ways, one of them ("symmetry-adapted cluster
  expansion", in SLCEMonteCarlo) simply wrong: *symmetry-adapted* describes the basis,
  not the expansion. Documentary only — no identifier changed. Literature citations
  (Drautz and Fähnle's *spin-cluster expansion*), the dated decision records under
  `docs/specs/`, and the `CHANGELOG` history keep their original wording; so does the
  theory page for the spin channel, which really is describing the spin-only formalism.

Also landing in the sibling packages: `SLCEMonteCarlo.has_disp` and
`SLCETools`' `natoms` fields, both folded into this package's generics
(`has_disp`, `n_atoms`); and `SLCEDynamics.run_llg_gpu` → `gpu_run_llg`.

### Changed — BREAKING: the second naming batch, and the joint datum finally has a name

The same audit's next tier: names that are wrong but were not *silently* wrong, plus
the one taxonomy hole it found. Again no aliases.

- **`Sector(; nbody)` → `Sector(; sites)`**, and `spin = (nbody = …)` →
  `spin = (sites = …)`; the resolved `SectorRule.nbody` field follows. `nbody` meant
  three things, two of them inside a single `Sector(...)` call: `BasisSpec(; nbody = 3)`
  is "body orders 1:3", `Sector(; nbody = 3)` was "**exactly** 3" (an `Int` resolves to
  `(3, 3)`), and `spin = (nbody = …)` was the spin-decorated **site** count. Carrying
  the `BasisSpec` habit into a sector row silently dropped every 1- and 2-body spin
  term — the fit still ran and still reported a good `R²` on a basis that could not
  express exchange. `sites` says what it counts, and the docstring now carries the
  `Int`-means-exactly rule as a warning.
- **`joint_datum(energy; moments, field, displacements, forces, reference, …)` — new.**
  The `*Datum` family named the two degenerate corners and left the joint case, the
  package's entire reason for existing, to a hand-written `TrainingDatum(; …)`. That
  was also the corner that had to hand-build a `DatumProvenance` for the reference
  stamp *and* carried a torque-qualifying field — the collision that produced the
  `torque_qualified` bug. `joint_datum` does both derivations in one call.
- **`SpinDatum` / `LatticeDatum` → `spin_datum` / `lattice_datum`.** Neither was ever a
  type — `SpinDatum` was one until the `TrainingDatum` merge, and removing it broke
  every `::SpinDatum` annotation. UpperCamelCase kept promising a type that is not
  there; snake_case matches what they are (and what `~/Packages/CLAUDE.md` says
  functions look like). SLCETools.jl's VASP reader moves with it.
- **`to_sunny(; spins)` → `(; spin_length)`.** `spins` is the spin *configuration*
  everywhere else in the family (`force_constants(; spins)`, `ForceConstantSet.spins`);
  here it was the effective spin length `S_eff = m/(gμ_B)`. Same word, same argument
  position, same receiver, different physical object.
- **`solve_coefficients(; groups)` → `(; row_groups)`**, and `select_support(; labels)`
  / `group_costs(basis, labels)` / `cost_weights(...).labels` → `column_groups`. The
  estimator-extension contract labelled **rows** with `groups` while
  `GroupAdaptiveRidge.column_groups` labelled **columns**; the docstring had to
  apologise for it. Feeding column labels to the row keyword ran and was silently
  discarded.
- **Persistence: schema tags `scefitting/sce-{basis,model}` → `slce/{basis,model}`,
  `schema_version` 5 → 6** (the sector table's `nbody` key is now `sites`). Documents
  written before the rename still load: unlike an API rename, refusing them would
  strand every model already saved to disk, so the old tags and the v5 sector key stay
  in the back-read.

### Changed — BREAKING: five names that misrepresented what they held

A five-lens naming audit over the package (public surface, struct fields and
keyword arguments, physics conventions, internals, and the four-package family)
found the internals healthy and the damage concentrated in names a user types.
These five are the subset where the wrong reading produces **plausible numbers
rather than an error**. No aliases or deprecation shims: an alias would preserve
exactly the reading each rename exists to remove.

- **`build_salc_basis(; isotropy)` → `(; scalar_only)`**, and the same for
  `SLCE.coupled_bases` and `AngularMomentum.build_real_bases`. `BasisSpec` has
  refused `isotropy` since the SOC rewrite *because the replacement inverted it*
  (`isotropy = true` ⇔ `soc = false`), yet the declared-`public` builder
  underneath still took the retired name with the retired polarity, bridged by
  `isotropy = !spec.soc`. A user following the builder's own docs — or a sed that
  finished the rename across the bridge — selected the **complementary** channel
  set; the basis still builds, still projects, still fits. `scalar_only` states
  the `Lf == 0` filter it performs, so the polarity is unmistakable and no live
  keyword carries the inverted reading. `isotropy` now survives only as the two
  deprecation errors and the persist legacy back-read (`soc = !isotropy`).
- **`identifiability(...).tol` → `.sigma_cut`.** The keyword is `rtol`
  (relative); the returned field was `rtol·σ_max` (absolute). Round-tripping
  `identifiability(f2; rtol = r.tol)` — the obvious thing to do with a field
  named `tol` — squared the cut, kept every singular value, and reported a
  rank-deficient design as identified, which is the one question the function
  exists to answer.
- **`EffectiveTerm.coef` → `.scaled_coef`.** `MultipoleTerm.coef` and
  `DecoratedTerm.coef` are the raw fitted `jϕ` with the `(4π)^{n_spin/2}` measure
  left to the consumer; this one has it — plus the SALC `folded` weight and the
  shifted polynomial coefficient — already folded in, necessarily, since one term
  merges many SALCs. A consumer migrating between the views now gets
  `has no field coef` instead of a silent `(4π)^{n_spin_slots/2}` over-count
  (12.6× on a two-spin term, `√(4π)` per slot beyond that).
- **`SCECoefficients` → `SLCECoefficients`.** The last public name still carrying
  the pre-rename prefix, next to `SLCEBasis` / `SLCEDataset` / `SLCEModel` /
  `SLCEFit`.
- **`dynamical_matrix(; masses)` now documents its units** (name unchanged —
  `masses` is what phonopy and ALAMODE call it). The docstring promised
  eigenvalues "`ω²`" and stated no unit anywhere in `src/` or `docs/`; they are
  `ω²` **in eV/Å²/amu** and in nothing else, so `sqrt.(eigvals(D))` read as THz
  is off by 15.633302. Both the docstring and the introspection guide now carry
  the conversions to THz and cm⁻¹. This was the only finding in the audit whose
  failure mode is a published phonon spectrum.

### Added — the anharmonic exit: ALAMODE's `anphon`

`write_phonopy` covers the harmonic channel, and phonopy stops there. Cubic and
quartic constants are precisely what SLCE can produce and nothing downstream could
consume, so `write_alamode(path, fcs...)` writes an ALAMODE `FCSXML` — harmonic plus
any anharmonic orders — and `anphon` turns them into relaxation-time thermal
conductivity, self-consistent phonons, Grüneisen parameters and isotope scattering.

The format is positional in three ways at once, so none of it was inferred from
documentation:

- `pair1` names a **primitive-cell** atom (`anphon` resolves it through
  `map_p2s[a][0]`), while `pairK` for `K ≥ 2` names a **supercell** atom. The
  supercell order is ours, but `Data.Symmetry.Translations` must agree with it and
  translation 1 must be the identity.
- `cell_s` indexes a fixed 27-entry table of **supercell** shifts (origin first, then
  `ix, iy, iz ∈ {-1,0,1}` skipping it, `iz` fastest). It is what lets a shift pointing
  out of the supercell keep its sign; writing the folded residue with `cell_s = 1`
  instead moves the frequencies by **5.2e-1** (mutation-tested).
- Values are Rydberg atomic units, converted from eV/Åⁿ.

Gated by running `anphon` and comparing its frequencies against `dynamical_matrix`
(`test/alamode/`, local-only like the oracle — anphon is a C++ binary wanting Boost,
FFTW, spglib and MPI). The comparison fits one scale across all modes at all q and
checks two separate things: the absolute residual after it (**3.7e-5 cm⁻¹**, bounded
below by anphon printing four decimals, which is why it is absolute — a relative
tolerance is meaningless on a soft mode) and that the fitted scale is 1 to
**5.5e-10**, which is the units claim. The cubic block is proven to parse and be
consumed (`GRUNEISEN = 1` refuses without cubic constants), not numerically verified;
the harmonic block pins units, sign and `cell_s`, and the cubic block is written by
the same code path. The core suite keeps what CI can defend without anphon — the
`cell_s` table rebuilt from its specification, and that every value parses as a bare
number, since `anphon` uses `boost::lexical_cast`, which rejects leading whitespace
and killed this exporter's first run on a padded `%25.15e`.


### Added — the lattice channel's exit, and its other restriction

- **`write_phonopy(dir, fcs)`** writes a `ForceConstantSet` as a phonopy calculation
  (`POSCAR` + `FORCE_CONSTANTS`), so band structures, DOS, thermal properties and
  thermal conductivity come from phonopy instead of from reimplementations here. The
  magnetic state is already in `fcs`: export two states and the difference between the
  band structures is the model's magnetoelastic content.

  `FORCE_CONSTANTS` is a **positional** format — a matrix over supercell atom indices,
  with the supercell built by phonopy from the unit cell and `--dim` — so a
  disagreement about its ordering yields a permuted force-constant matrix that still
  diagonalizes and still has three acoustic modes at Γ. The ordering was therefore not
  taken from documentation but read off phonopy, and it is gated by running phonopy and
  comparing its dynamical matrix against `dynamical_matrix` (`test/phonopy/`, its own
  environment and CI job; phonopy is never a dependency of `Pkg.test()`). Agreement is
  7.4e-16.

  Three properties of that gate's fixture are load-bearing, each found by mutation
  testing rather than by design:
  - the comparison is **mass-weighted**, with two species of different mass — phonopy's
    phase depends only on the lattice translation, so permuting basis atoms is a
    similarity transform. Dropping the species permutation moves an unweighted
    comparison by 8e-16 (i.e. not at all) and a mass-weighted one by 1.7e-2.
  - **species are interleaved** (`[Fe, Ni, Fe]`), so the POSCAR's species grouping is a
    real permutation. The `POSCAR` and `FORCE_CONSTANTS` are written together for
    exactly this reason: the former fixes the atom order of the latter.
  - **the lattice shifts are not symmetric under swapping the first two axes.** The
    first fixture tried had `R ∈ {(0,0,0), ±(1,1,0)}`, which is, and reversing the
    lattice-point axis order produced a *byte-identical* file — a gate that passed
    before and after the violation. With the shift set fixed, that mutation moves the
    comparison by 1.8e-1, and swapping the atom / lattice-point nesting by 3.2e-1.
- **`restrict(model, :lattice)`** — the spin-independent sub-model, the mirror of
  `restrict(model, :spin)`. Its predictions are independent of the magnetic state by
  construction, so it needs no `spins` anywhere, and
  `predict_energy(model, e, u) − predict_energy(lat, nothing, u)` is everything the
  magnetic state controls. It is *not* a spin average: a rank-0 spin factor is constant
  in `e` yet still carries a spin slot, so it lives in the remainder.


### Added — the lattice-only entry path

The same API review walked a phonons-only workflow end to end and counted where the
package demanded a magnetic state from a caller who has none. Four places, each
answerable only by inventing one — and a fabricated `+ẑ` ferromagnet is
indistinguishable from a real one once it reaches a basis that reads spins.

- **`LatticeDatum(energy; displacements, forces, reference, …)`** — `SpinDatum`'s
  counterpart for spin-free data. It fills the mandatory spin channel with the `ẑ`
  placeholder and **exactly zero** moment magnitudes, and passing `reference` builds
  the `reference_id`/`crystal_fingerprint` stamp the displacement channel requires.
  Zero is the load-bearing part: a lattice-only basis never reads the placeholder, and
  the same datum against a spin-carrying basis trips `SLCEDataset`'s existing
  zero-moment invariant by name.
- **`SLCEDataset`'s `use_torque` defaults to `nothing`**, resolved from the basis:
  `false` for a lattice-only expansion, which has no torque design block and for which
  the old `true` default was a requirement no correct call could satisfy. Resolved
  from the *basis*, never from whether the data carry torques — the latter would turn
  "the adapter dropped the constraining fields", today a loud error, into a silent
  energy-only fit.
- **`force_constants`, `predict_energy` and `predict_force` accept no spin state**
  (omitted, or `nothing`) when the basis carries none. The predicate is the new
  `_basis_has_spin` (spec ∪ surviving SALCs). It is **not** `is_soc_free`, which asks
  whether a label's total spin rank vanishes — true for most ordinary `soc = false`
  spin labels, so building the test on it would wave a spin-carrying model through and
  evaluate it at a fabricated state. The two predicates are now gated against each
  other on a basis where they disagree.
- **A hand-built `DatumProvenance` no longer suppresses the torque channel.** Two
  requirements collided: the displacement channel *requires* a provenance carrying the
  reference identity, and building one discarded the automatic `torque_qualified` that
  explicit torques (or a nonzero field) earn — so a joint datum with displacements AND
  torques failed the dataset build with "pass `use_torque = false`", the exact opposite
  of the caller's intent. Both datum constructors now **upgrade** the flag and never
  revoke it; withholding torques from a fit is `use_torque = false`'s job, at the
  dataset level where that decision belongs.

### Fixed — a dead design column blamed the data for a property of the basis

`fit`'s dead-column warning said "this fit's data say nothing about them", which is
advice that cannot be taken when the SALC is zero for *every* configuration. Measured
on a conventional bcc 2-atom cell: 1 of 3 lattice-only SALCs, and **14 of 23** joint
ones, cancel identically under minimum-image ties — so `n_salcs` overstates a small
cell's real freedom by a lot. The two cases now get separate messages. The classifier
evaluates the basis at three probe configurations (reached only once a dead column
exists, and seeded, because a diagnostic must not depend on ambient RNG state); it is
pinned against a 200-sample design matrix built through the other code path, where the
two index sets agree exactly.

### Added — the magnetic-symmetry contract of the force constants

A four-agent review of the user-facing API against two concrete workflows — a
collinear-antiferromagnet lattice-dynamics run in the ALAMODE mould, and a
magnetic-space-group calculation with SOC — found the package's headline physics
claim true, load-bearing, and written down nowhere. That is now fixed, together
with the two ways of losing it silently.

- **`force_constants` documents which group it imposes, and why the joint path is
  the right one.** Force constants are time-reversal *even*, so an antiunitary
  element `g·T` of the magnetic space group constrains `Φ` through its rotation
  part exactly as a unitary element does: the correct invariance group is
  `D ∪ D′`, not the unitary halving subgroup. The joint path lands on it for
  free — the SALCs are projected with the paramagnetic grey group `G × {1, T}`,
  and fixing `spins` reduces that to the magnetic stabilizer with nothing
  declared. On a stripe-AFM fixture the three candidate groups admit 7 (the
  paramagnetic group, i.e. a lattice-only basis — *too large*, it zeroes what the
  order breaks), **12** (`D ∪ D′`, correct), and 16 (`D` alone, which is what
  relabelling the sublattices as distinct species would impose — *too small*).
  The joint constants satisfy the unitary *and* the antiunitary half to 1.8e-15
  and are demonstrably not invariant under the rest. The whole ledger is now a
  test (`test/unit/test_forceconstants.jl`), built against a hand-assembled
  P4/mmm group so it needs no Spglib.
- **`force_constants` warns when the constants cannot depend on `spins`.** A basis
  carrying spin content and displacement terms of the requested `order`, but no
  term carrying both, produces `Φ` (and `D(q)`) bit-identical for every magnetic
  state — a "magnetic" phonon calculation that is not magnetic at all, with a
  perfect r² and no other signal. The usual way in is declaring the
  magnetoelastic sector at `disp = (degree = 1,)`, which feeds the *forces*; the
  harmonic constants need `degree = 2` under a spin-carrying sector. The check
  reads the basis, never `jphi` — a coefficient that fitted to zero is a property
  of one fit and `refit` moves it, whereas an empty channel is permanent.
  README's and the basis guide's magnetoelastic examples now say which
  deliverable each `degree` feeds.
- **`fit` warns when a dataset carries forces and `force_weight` is 0** (its
  default). Nothing downstream can tell that apart from a deliberate energy-only
  fit, and a force-first workflow — forces are *the* observable in ALAMODE, whose
  `DFSET` has no energy column at all — lands there by doing nothing wrong.

### Fixed — whole-package review

A nine-agent review over the whole package (a critical reviewer and a
software-engineering reviewer among them). Everything below was verified
independently before it was acted on, and every source fix carries a gate that
was mutation-tested — a gate that passes both before and after the violation is
worse than none.

**Numbers that were wrong.**

- **`Ridge(lambda = 0)` was not OLS.** It solved the normal equations, squaring
  the condition number for no benefit; `λ = 0` now routes to the pivoted-QR OLS
  path exactly. Both docstrings had it backwards (`OLS` claimed normal
  equations, `Ridge` claimed a QR) and now say what the code does.
- **`_rank_df` used `maximum(size(X))` in the rank tolerance** where
  `LinearAlgebra.rank`'s documented default is `minimum(size(X))`. On a tall
  design (`n ≫ p`) this inflates the cut by `n/p` and silently under-counts the
  degrees of freedom feeding GCV.
- **`Crystal` did not deliver the `[0, 1)` it documented.** `mod(-1e-18, 1.0)`
  is `1.0` in Float64 — the exact result falls below the float resolution near 1
  and rounds up. `build_neighbor_list`'s `AllImages` image box is exactly tight
  and its derivation needs `|Δf| < 1` *strictly*, so a coordinate stored at
  `1.0` drops the shell sitting on the cutoff sphere: measured 102 pairs against
  a brute force of 104 on a triclinic cell.
- **`dynamical_matrix`'s phase convention had no gate**, so the `2π` scale was
  free. Now gated by periodicity in reciprocal-lattice vectors. (The *sign* of
  the phase is not observable in the spectrum — `D` and `conj(D) = Dᵀ` share
  eigenvalues — so the scale is the real content.)
- **A spin component just outside `[-1, 1]` reached the Legendre recursion**,
  which has that as a hard domain, and threw a `DomainError` from inside the
  threaded design assembly naming neither the configuration nor the atom.
  Tightening the unit-norm `atol` does *not* close this: measured, `‖e‖ − 1 =
  5e-9` clears a `1e-8` band and still throws. The boundary now checks the
  components, which closes it and rejects nothing the tolerance was meant to
  allow.

**Invariants that could be bypassed.**

- **`SALCBasis` and `SLCEBasis` were plain structs**, so the exactly-field-typed
  call reached the default constructor and skipped every check — which
  persistence and `restrict` actually do. Both now validate in *inner*
  constructors, as `STYLE_GUIDE.md` already required. `SALCBasis` additionally
  derives its `fingerprint` from the keys instead of accepting one, so the
  summary cannot drift from what it summarizes.
- **A space group was taken on faith.** No integrality of the fractional
  rotations, no `|det W| = 1`, no Cartesian orthogonality, no closure, no
  duplicate check, and `map_sym` columns were *documented* as permutations
  without being verified to be one. Downstream a `SpaceGroup` is consumed as a
  group — orbits reduce by stabilizer counting, SALCs project with `(1/|G|)
  Σ_g` — so a merely plausible list of matrices does not fail, it produces wrong
  multiplicities and a non-idempotent projector. `_validate_ops` now checks all
  of it, with the full pairwise closure running up to 192 operations (48 point
  ops × 4 centerings, i.e. every conventional setting) and that cap stated
  rather than silent.
- **The neighbour list's `tol` was public and then contradicted**: every
  downstream "is this edge inside a radius" decision read a hard-coded constant,
  so a caller who widened the band got a list built one way and clusters
  admitted another. The band now travels on the `NeighborList`.

**Gates that could not fail.**

- Four `same_members` calls in `test_mixedsalc.jl` were bare expressions with no
  `@test` — the anti-drift comparison they belong to was reporting success
  without asserting anything.
- `runtests.jl` accepted an unknown `TEST_MODE` by running zero tests and
  reporting success, and the threaded-vs-serial gates were vacuous at one
  thread. Both now refuse.
- **The decor engine's axis-relabelling convention had no value-level gate.**
  See `test/unit/test_mixedsalc.jl` — the short version is that the existing
  fixtures pinned it by *shape* only, and an invariance check cannot catch it
  either (the wrong convention yields a differently-scaled invariant, not a
  non-invariant).
- **Two test fixtures declared a group that is not one**: `_triangle_c3v` and
  `_mx_triangle_c3v` put a 3-fold rotation in a *cubic* box, where its
  fractional matrix is not integral. Harmless only by accident — every cluster
  in those fixtures sits inside one cell, so no lattice shift is ever rotated.
  Both moved to a hexagonal cell, geometry and every assertion count unchanged.

**Documentation that had drifted.** The README was two milestones behind (no
mention of displacements, forces, the ASR, or the phonon deliverables); the
oracle harmonics agreement was described as "bit-for-bit" when it is
`atol = 1e-13, rtol = 1e-12`; the design record named an ASR solver
(column-pivoted QR) that is not the one implemented (per-component SVD), listed
two convenience presets as public API that were never built, and declared M3
closed on a gate (c) that does not exist — now recorded as not run, the way gate
(m) was, rather than quietly carried. `EffectiveTerm.coef` and
`force_constants(...; order = 1)` gained the convention statements they were
missing.

### Added — effective models at a displaced structure (M5 slice 1)

- **`effective_model(model; u0, atol = 0.0)` → `EffectiveModel`** (with
  `EffectiveTerm` and a `predict_energy(em, e, δu)` method). The same fitted
  coefficient set re-expanded around the displaced reference `R + u0`, satisfying
  `predict_energy(em, e, δu) == predict_energy(model, e, u0 .+ δu)` exactly.

  This is an exact change of expansion point, not a Taylor truncation: every
  displacement site factor `|u|^{2k} R_{lm}(u)` is a homogeneous polynomial, so the
  shift is a finite linear map on the coefficients — **lower triangular in total
  degree**, which is where the degree-0 and degree-1 content that the reference
  expansion does not carry comes from. It is built by binomially shifting each
  monomial of `SolidHarmonics.solid_harmonic_poly` (the same function the force
  constants and the ASR builder read) and multiplying the shifted polynomials across a
  term's displacement slots.

  Use it to reach a symmetry-broken distorted structure — a relaxed cell, a condensed
  soft mode — with no common-subgroup grid, and as the starting point for
  renormalizing coefficients onto a thermally displaced reference.

- **The result is deliberately not a `SLCEModel`.** The displaced structure's symmetry
  is the stabilizer of `u0` inside the reference group, generally a proper subgroup, so
  the reference SALCs cannot span it. `EffectiveModel` is the unsymmetrized
  decorated-monomial form instead: one term per (spin factors) × (displacement
  monomial), evaluable and differentiable, carrying no `SALCKey`s, not fittable and not
  persisted.

- **`δu = 0` is the `u0`-frozen point, not the clamped-ion one**, and that is the
  point of the utility rather than a wart. The re-expansion renormalizes the spin-only
  couplings in place (it does not add new ones) and generates the reference forces the
  displaced structure carries.

- A homogeneous strain is refused rather than silently accepted: the affine pattern
  `u_i = ε·R_i` is not cell-periodic and must never go through a periodic evaluator, so
  a `strain` keyword exists purely to throw with that explanation instead of failing as
  a `MethodError`.

- Gates (`test/unit/test_effective.jl`) run against the production evaluator: the
  exactness identity at small and large `u0`, `u0 = 0` reproducing the original
  surface, a central difference of the re-expanded surface reproducing the model's
  analytic `predict_force` at `u0`, the degree structure, the same-atom variable merge
  through an `AllImages` self-bond, `atol` pruning breaking exactness on purpose, and
  the refusal surface. Exactness is measured against the sample's **energy scale**, not
  pointwise: a random spin configuration can sit at a zero crossing of the energy,
  where the pointwise relative error reaches 7e-13 while the absolute error never
  exceeds 6e-14 (3e-15 scale-relative).

### Added — the sampler row-table contract (M4 slice 3a)

- **`SLCE.row_layout(model)` → `SLCE.RowLayout`** (public, unexported), with
  `SLCE.row_index` and `SLCE.site_rows!`. A sampler evaluates a term by gathering one
  number per tensor axis out of a per-site row table; this fixes that numbering ONCE
  so the model and the sampler's program builder cannot disagree about it.

  Blocks stack in `Channel`-enum order. The `SPIN` block is at offset 0 and is
  *verbatim* `Harmonics.lm_index` — not merely isomorphic to it — so a pure-spin
  model's layout is the one the spin-only consumers already use and the displacement
  channel can only ever append. (It keeps the `l = 0` row that no `SiteFactor`
  addresses, precisely so the numbering is `lm_index` itself.) The `DISP` block
  carries `2l + 1` rows per `(k, l)` the basis uses, holding `|u|^{2k} R_{l,m}(u)`.

  The layout is a property of the **support**, not the coefficients, so a consumer
  may keep its row tables across a coefficient hot-swap; `RowLayout` compares by
  value for exactly that check.

The gate is end-to-end (`test/unit/test_rowlayout.jl`): a miniature consumer that
tabulates rows with `site_rows!`, addresses them with `row_index`, and contracts
`decorated_terms` reproduces `predict_energy`. Index bookkeeping alone cannot catch a
numbering-vs-contents disagreement; that sum can.


### Added — force constants and the dynamical matrix (M4 slice 2)

- **`force_constants(model; spins, order = 2)` → `ForceConstantSet`** — the exact
  `order`-th derivatives of the energy with respect to atomic displacements at
  `u = 0`, **with the spins held at a given configuration**. That dependence is the
  point of a spin–lattice expansion: evaluate at two magnetic states and the
  difference is the magnetic contribution to the lattice dynamics.

  Exact, not finite differences. Every displacement site factor `|u|^{2k} R_{lm}(u)`
  is homogeneous of degree `2k + l`, so only terms whose degrees sum to `order`
  survive differentiation at the origin, and their contribution is read off the
  monomial coefficients `solid_harmonic_poly` returns — the same function the ASR
  builder uses, so the two cannot drift. `order = 3` gives the cubic anharmonic
  constants (gated against a third finite difference, which is exact for a cubic
  model: agreement at 1e-16 relative).

  Indexing follows lattice dynamics, not SALC members: `Φ[(a,0),(b,R)] =
  ∂²E_cell/∂u_a(0)∂u_b(R)`, one entry per ORDERED index tuple, anchored so the first
  index is in the home cell. The reverse ordering is a separate key equal to the
  transpose — a property the suite checks, never a storage trick.
- **`dynamical_matrix(fcs, q; masses = nothing)`** — `D_{aα,bβ}(q) = Σ_R Φ(R)
  exp(2πi q·R) / √(MₐM_b)`, with `q` in **fractional reciprocal coordinates** and
  rows/columns atom-major, Cartesian-minor. Hermitian, and `D(−q) = conj(D(q))`.
  Omit `masses` for the bare force-constant matrix.
- **The ASR pays off physically here.** A model fitted under the acoustic sum rule
  has exactly three zero eigenvalues of `D(0)` — the acoustic branches — and
  vanishing `Σ_{b,R} Φ_{aα,bβ}(R)`. One fitted with `asr = false` has neither, on the
  same basis and the same data. Both are gated, and both are shown in
  `guide/introspection.md`. (Zero acoustic frequencies are not the same as *stable*
  phonons: a negative eigenvalue is a real instability of the fitted model.)

Gates (`test/unit/test_forceconstants.jl`): the Γ-restricted sum `Σ_R Φ(R)` against a
finite-difference Hessian of the production evaluator, the transpose relation between
reverse-ordered keys, `D(q)` Hermiticity / conjugation / mass weighting, the acoustic
modes with an unconstrained model as the teeth, spin dependence (each configuration's
constants exact for *its* configuration), order 3 against a third derivative, and the
empty results for pure-spin, clamped-ion-restricted, zero, and under-truncated models.


### Added — the downstream term contract (M4 slice 1)

- **`decorated_terms(model)` → `Vector{DecoratedTerm}`** — the general successor of
  [`multipole_terms`](@ref). One term per cluster member and slot layout, of any
  channel content: each tensor axis is labelled with its own `SLCE.SlotRef`
  (which member site, and the `(channel, k, l)` factor on it), so a site carrying
  both a spin and a displacement factor is two axes. On a pure-spin model it
  reports exactly the terms `multipole_terms` does.
- **The consumer scale rule is now carried, not derived.** `DecoratedTerm.scale` is
  `(4π)^(n_spin_slots / 2)` — one `√(4π)` per SPIN slot, because the 4π is an
  artifact of the spin-sphere measure and the displacement kernel is normalized
  4π-free. The pure-spin-era shortcut `(4π)^(body/2)` (one factor per *site*) agrees
  only when every site carries exactly one spin factor and nothing else; on a
  force-constant term, whose sites carry no spin factor at all, it invents a factor
  per site. Both existing consumers derive it from `length(atoms)`, which is why:
- **`multipole_terms` refuses a displacement-decorated model**, and the message now
  names both hatches — `decorated_terms(model)` and
  `multipole_terms(restrict(model, :spin))`. The trigger is the **spec**
  (`_basis_has_disp`), not the surviving coefficients: a displacement sector whose
  couplings all fitted to zero was still built and fitted in a `p ≥ 1` setting, and
  reporting it as pure spin would fail open on the invariant the refusal exists for.
- **`restrict(model, :spin)`** — the exact clamped-ion sub-model: the pure-spin SALCs
  with their coefficients untouched, and a spec reduced honestly (`pmax` zeroed,
  sectors cut to their displacement-degree-0 row) so the pure-spin surfaces and
  persistence both accept the result. Gated on bitwise equality with the joint model
  at `u = 0`, for energy and torque.

  **`restrict` is not a refit**, and the new `guide/introspection.md` says so in a
  boxed warning plus a measured comparison: the clamped-ion couplings are the
  reference-geometry values, while a spin-only fit to the same data has no
  displacement coordinate to attribute anything to and absorbs the lattice's
  contribution into `J`. The gap is the physics the joint model exists to separate.


### Changed — test-suite audit: gates that could not fail

A review of the *tests* (not the code) for verification gaps. Nothing was found
broken — the oracle / Sunny / GLMNet suites, which had not run since the ASR,
identifiability and staged slices, are green (14063 / 26 / 27), and a mutation
check confirms gate (j) catches a force-sign flip applied at **both** sign sites
at once. What was missing was the ability to notice future breakage:

- **CI now runs the suite with `JULIA_NUM_THREADS: 4`.** The threaded-vs-serial
  gates compare a parallel result against a serial reference; at one thread
  `Threads.@threads` *is* that reference, so every such assertion passed while
  exercising nothing. A single-threaded CI cannot see a data race in the orbit
  loop or in a design-matrix builder.
- **The joint design builders got the serial reference the pure-spin ones had.**
  `_design_energy(basis, cfgs, disps)`, `_design_torque(…, tatoms)` and
  `_design_force` each thread over columns with a per-task `SALCScratch` plus two
  gradient buffers, and none of the three was compared against anything.
- **The ASR is now gated at 3-body**, where one constraint row couples three site
  blocks instead of a pair — the case third-order force constants (and M4's
  `force_constants` / `dynamical_matrix`) live in. Every previous `build_asr`
  fixture was a 2-body bond. Symbolic rank ≡ numerical translation-image rank
  (217), finite-`t` translation invariance and per-config `Σf = 0` through the
  production evaluator, after `fit` and after `refit`, with an unconstrained model
  as the teeth.
- **A displacement-decorated `SLCEModel` now round-trips through save/load with the
  numbers checked** — coefficient re-pairing by `SALCKey`, the three joint
  predicts, and `asr_residual` (recomputed from the basis, never persisted) before
  and after a reload. The model-level reload contract had been gated only on
  pure-spin keys, whose v5 form is a value-preserving relabel of the v4 one; the
  mixed-basis persistence test compares an `SLCEBasis` structurally.
- **`examples/` are CI'd.** Each self-gates with `@assert`s on a recovered coupling
  — the fence that caught the canonical-member J/2 regression — and nothing ran
  them: the docs job executes the tutorials, which are different files.
- `checkdocs = :public`: the unexported `public` surface (`build_asr`,
  `sector_columns`, `salc_groups`, `SolidHarmonics`, …) is API, and a missing
  docstring there now fails the strict docs build like any other.
- Pinned the documented behavior that `cross_validate` ignores the force block
  (identical result with the block dropped), and `rss_force`, the one exported
  diagnostic never called directly.
- Noted in `fitting/asr.jl` that the `(4π)^{n_spin/2}` column scale cannot move the
  null space — `_ASRRowKey` keys on the spin monomial, so the factor is constant
  along each row and the relative row normalization divides it out. Verified by
  mutation: dropping it leaves the whole suite bit-identically green. It stays
  because it is what makes `A`'s entries comparable to the design's.


### Added — staged (hierarchical) fitting, closing M3 (slice 6)

- **`fit(...; frozen, sector_mask)`** — fit a joint model in physical stages
  (exchange, then spin–lattice coupling, then force constants) instead of one shot.
  `sector_mask` selects the columns a stage fits (`SLCE.sector_columns`: `:all` /
  `:spin` / `:lattice` / `:coupled` — a partition by channel — and `:soc_free` /
  `:soc` — a crosscutting partition by `L_S`; also unions, explicit column lists,
  and `Bool` masks), `frozen` holds the rest at a previously fitted
  [`SLCEModel`](@ref)'s coefficients, matched by `SALCKey` and never positionally
  (a key the target basis lacks, carrying a nonzero coefficient, is an error rather
  than a silent drop). `j0` is never frozen. A frozen value on a free column is
  ignored — that column is being re-fitted.
- **The staging axis is not the truncation axis.** `Sector(soc = false)` decides
  what the model can express; `sector_mask = :soc_free` decides what a stage fits.
  Both now read ONE predicate (`SLCE.is_soc_free`), and the suite gates that the
  masked key set is exactly the key set a `soc = false` rebuild produces (design
  record §13 risk 4, "soc-vs-sector_mask drift").
- **Affine ASR (the `ASRReparam.beta_p` slot, until now a placeholder that threw)**:
  a stage's constraint is `A_free·β_free = −A_frozen·β_frozen`, solved exactly as
  particular solution + null space, so a staged model is translation-invariant **as
  a whole**. When the frozen part is itself ASR-satisfying — every stage of a chain
  is — the right-hand side vanishes and the stage stays homogeneous (the staged-fit
  theorem, §6 amendment 8); "satisfying" is judged by the same *relative* measure
  `asr_residual` reports, since a fitted stage leaves ~1e-16 and treating that as a
  violation would send every chained stage down the affine path, where a
  roundoff-sized right-hand side is generically infeasible. An infeasible freeze
  (a violation on constraint rows the free columns cannot balance) is refused with
  those rows named. Measured on the D4h fixture: **channel masks never straddle a
  constraint row** (A's rows are graded by spin content and displacement degree),
  while the `L_S` masks straddle 54 of 180 — which is precisely why the staging axis
  needs the affine machinery.
- **`SLCEFit.reparam`** now records the reparameterization the estimator actually
  solved under (the dataset's ASR, or the stage's). `refit`, `gcv`,
  `effective_dof`, `identifiability` and `dof` read it instead of `dataset.asr`, so
  a staged fit is never silently re-assembled as an unstaged one: `dof` is the
  stage's free-parameter count, and `refit` stays inside the stage (frozen
  coefficients are not thresholded away, the support is intersected with the free
  columns, and the constraint is re-derived for that sub-stage).
  `cross_validate` threads `frozen`/`sector_mask` to every fold;
  `select_support` rejects a staged fit (its frozen columns are not selectable).
- No staging request means the untouched path: `fit` without `frozen` and with
  `sector_mask = :all` returns bitwise-identical coefficients and keeps the
  dataset's own reparameterization object.

### Added — identifiability diagnostics and derivative-only recovery (M3 slice 5)

- **`identifiability(fit_or_dataset; rtol = nothing)`** — the rank diagnosis of the
  design the fit actually solves, in the coordinates it solves them in (`p`
  unconstrained, the ASR's `q = p − rank(A)` under the reparameterization). Returns
  `(; ncols, rank, nullity, sigma_min, sigma_max, tol, gap)`; `nullity > 0` means
  the data do not determine the model and the returned coefficients along those
  directions are the estimator's null-space convention (OLS's minimum norm, or
  whatever a penalty prefers), not estimates. `gap` (smallest kept / largest
  dropped singular value) makes an ambiguous rank decision visible — a diagnostic
  reports where `_asr_nullspace` refuses. The cut is `min(size(X))·eps` (Base's
  `rank` convention): a `max`-based one would grow with the row count while the
  `1/√n` block whitening keeps the spectrum sample-size-independent. The dataset
  method answers before fitting, for a `(torque_weight, force_weight, asr)` one is
  considering, and raises exactly the errors the corresponding `fit` would (shared
  validation).
- **Standing dead-column warning in `fit`** — the `O(n·q)` per-column half of the
  same question: design columns the data do not touch (every pure-spin column in a
  force-only fit) are reported, since their fitted values are artifacts. The test
  is the package's usual **relative** norm cut, not `iszero`: on COM-free samples
  the SALCs proportional to `Σ_a u_a` sit at ~1e-19 in the torque design, and
  centering leaves a structurally constant column at ~1e-17. The exact statement
  stays opt-in: a full rank test is `O(n·q²)`, the same order as the fit itself at
  production scale.
- **Recovery plan B (`test/unit/test_identifiability.jl`)**: forces and torques
  **alone** (zero energy weight — the energy data enter only through the analytic
  `j0`) recover a known joint model to ~1e-15 under the ASR, from
  center-of-mass-free displacement samples. Without the constraint the same data
  admit a materially different model (0.97 away in `β∞`) at machine-precision
  derivative residuals: the two agree on the sampled slice — energies, forces, and
  even `Σ_a f_a = 0` — and disagree the moment `u` drifts off it. The measured
  rank ledger (design record §6 amendment 9, corrected there): torque is blind to
  **all spin-independent** content, so its deficiency is
  `rank(A) + dim{spin-free feasible directions}` — equal to `rank(A)` exactly when
  every displacement-decorated SALC carries spin factors (the first fixture), and
  strictly larger, with a residue the ASR cannot cure, once a lattice-only sector
  is present (second fixture, p = 219: torque-only nullity 156 unconstrained, 6
  constrained — force constants are **not** identifiable from torques alone);
  force-only deficiency is smaller than `rank(A)` (`Σ_a f_a` *is* `−D E` at the
  sample, so forces see part of the violation) plus the structurally zero pure-spin
  columns; the plan-B co-fit is deficient unconstrained and **full rank `q`**
  constrained in *both* fixtures, with `rank([X; A]) = p` — the
  constraint rows supply exactly the missing information. Generic (drifting)
  displacements are full rank with no constraint at all: the deficiency is a
  property of the sampling protocol, and the ASR's payoff is on realistic COM-free
  data and in predictions off the sampled slice.

### Added — acoustic sum rule: exact translation-invariance constraints (M3 slice 4)

- **ASR as exact linear equality constraints** `A·β = 0` (energy invariance under
  rigid translations `u_a → u_a + t`), enforced by null-space reparameterization
  `β = Z·γ` — by construction, never approximately, never as a penalty. The
  constraint matrix is built **symbolically**: the translation generator
  `D = Σ_a ∇_{u_a}` applied to every displacement-decorated SALC, expanded in one
  common unsymmetrized monomial basis across orbits (spin factors as opaque
  symbols; displacement factors via `SolidHarmonics.solid_harmonic_poly`, an exact
  coefficient-space rerun of the evaluator's own recurrences, gated against the
  numeric kernel). `A` lives in design-column coordinates (the `(4π)^{n_spin/2}`
  evaluator scale included).
- **Rank policy**: row-normalized `A`, exact connected components, per-component
  SVD with cut `σ ≤ 1e-10·σ_max` and a **forbidden band** (`σ ∈
  [1e-12, 1e-8]·σ_max` ⇒ error — ambiguous rank refuses, never guesses); `Z` is
  orthonormal per component, identity on pure-spin columns. `rank(A)` is gated for
  exact equality against an **independent numerical translation image** (random
  points through the production `accumulate_grad!`) in the test suite. A
  truncation admitting NO translation-invariant displacement content (e.g. pair
  `(1,1)` splits without on-site partners under `pmax = 1`) is correct but loud
  (warning at build).
- **Placement** (`force_cols` discipline): `build_asr(basis)` runs once at
  `SLCEDataset` construction, stored on `dataset.asr` (`nothing` for pure-spin
  bases — the structural fast path; carried by slicing/`vcat`);
  `_assemble_problem` only applies it (each block right-multiplied by `Z` before
  stacking, the compact force block via `Z[force_cols, :]`). Nothing is
  persisted: `Z`'s gauge is factorization-dependent, `β` is gauge-invariant, and
  the public verifier `asr_residual(model)` recomputes `‖Aβ‖/(‖A‖·‖β‖)` from the
  basis (fingerprint precedent).
- **`fit(...; asr = true)`** (default ON — no joint users existed, so this is the
  free moment): the estimator solves in γ; `SLCEFit` records the applied flag and
  achieved residual (part of the re-assembly contract). Pure-spin fits are
  **bitwise identical** under either setting (gated). `asr = false` is for the
  §12 (k) violation demonstration and ablations — an unconstrained joint fit on
  noisy data demonstrably breaks translation invariance and `Σ_a f_a = 0`.
- **β-space penalties through `Z`**: `solve_coefficients` gains a `nullspace`
  kwarg. OLS/Ridge are unchanged (orthonormal `Z` ⇒ γ-space solve ≡ β-penalized
  constrained solve); AdaptiveRidge/GroupAdaptiveRidge evaluate their unchanged
  β-space weight maps at `β = Z·γ` and solve the compressed SPD system
  `X̃'X̃ + λ·Z'·D(β)·Z` (the log-sum MM descent survives restriction to a
  subspace). `PrecomputedPilot` and the GLMNet L1 estimators reject constrained
  solves with pointed messages (L1 under `Z` is a generalized lasso; L1 on γ is
  gauge-dependent).
- **`refit` re-derives the support null space** `null(A[:, S])` — an
  unconstrained refit would silently re-break translation invariance in the
  de-bias step. A support that splits a constraint-coupled column set structurally
  zeroes survivors (legal, warned); translation/`Σf = 0` gates run post-refit in
  the test suite.
- **Diagnostics in the reparameterized space**: `dof` = `p − rank(A) + 1` (free
  parameters), `effective_dof`/`gcv` whiten the compressed penalty by Cholesky
  congruence (`Z'DZ` is dense — the diagonal shortcut would be silently wrong),
  and GCV's `n` now excludes exactly-zero-weight rows (the energy block at
  `w_T + w_F = 1`; also fixed on the pure-spin `select_fit` path).
  `cross_validate` fits each fold under the same `asr` setting (kwarg).
- **Fences**: `select_fit`/`select_support` reject ASR-carrying (joint) bases —
  their λ path and group-cost model are pure-spin until the channel-split
  `group_costs` lands. The affine constraint slot (`ASRReparam.beta_p`, for
  frozen-stage offsets) exists but throws until the staged-fit slice.
- Review-round hardening (numerical + code review): **cancellation residue in
  `A` is pruned to exact zeros per row** and the row drops are relative — a
  `refit` column subselection can no longer promote ~1e-16 roundoff into a
  full-strength, BLAS-rounding-determined constraint (gated by an
  empty-a-row's-columns subselection test); `build_asr` refuses a rank-0 result
  on a displacement-active basis (physically impossible ⇒ broken expansion) and
  reports **basis-level** structurally-zero columns once, while `refit` warns
  only about columns its support additionally kills; the constrained IRLS
  convergence test moved to β (γ's ∞-norm is factorization-gauge-dependent);
  `_edof_ns` is gated against the dense hat matrix and validates positive
  weights; the GCV zero-weight-row test uses the assembly's own scale
  expression (a `wT + wF >= 1` sum test can disagree by one rounding);
  `vcat` refuses silently disagreeing `asr` fields; an AllImages self-image
  joint dataset is constructible with `asr = nothing` but `fit`'s default
  `asr = true` then errors (unconstrained joint fits are an explicit opt-out);
  the forbidden-band refusal is an `ArgumentError` (package convention); the
  component-grading and residue-free-row invariants are asserted in the suite.
- Deferred: a serialized joint prediction fixture (gate (c) touches only
  pure-spin bases). The recovery-plan-B rank-accounting gate landed in M3 slice 5
  (above), correcting §6 amendment 9's blanket "deficiency = rank(A)" to the
  measured per-channel ledger.

### Added — force design block and three-block co-fit (M3 slice 3)

- **Joint designs**: against a displacement-decorated basis,
  `SLCEDataset(basis, data::Vector{TrainingDatum})` now builds `X_E`/`X_T` at each
  configuration's `(e, u)` through the joint kernels (a spin-only datum contributes
  `u = 0` exactly; pure-spin columns stay bit-identical to the spin-only design
  path) and, with `use_force = true` (default, mirroring `use_torque`), the force
  design block `X_F` with targets `f = −∂E/∂u` (sign pinned per design record §6).
  `X_F` is stored **compact**: columns are only the displacement-active SALCs
  (`SLCEDataset.force_cols` — a pure-spin SALC has `∂Φ/∂u ≡ 0`; the zero columns
  exist only in the assembled fit), and rows only for atoms some SALC displacement
  slot reads (`_disp_referenced_atoms` — structurally zero rows are excluded, never
  padded, with a warning when their DFT forces are nonzero, e.g. a forgotten
  per-species `pmax`). Force rows follow the stored ragged-bookkeeping contract of
  the torque block with their own `force_config`; slicing and `vcat` re-label and
  re-offset it, and `vcat` refuses to mix displacement-bearing and
  displacement-free parts.
- **Three-block co-fit**: `fit(SLCEFit, dataset, est; torque_weight, force_weight)`
  minimizes `(1 − w_T − w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F` (each weight in
  `[0, 1]`, `w_T + w_F ≤ 1`; force block whitened by `√(w_F/n_F)` and scattered to
  full width in `_assemble_problem`). `SLCEFit` gains a `force_weight` field;
  `refit`, `gcv`, and `effective_dof` assemble with both weights. `w_F = 0`
  reproduces the previous energy(+torque) fit bitwise.
- **Joint prediction**: `predict_energy(model, e, u)`,
  `predict_torque(model, e, u)`, and the new exported `predict_force(model, e, u)`
  (`f = −∂E/∂u`, zero on pure-spin models and displacement-unreferenced atoms).
  The two-argument predict forms now **refuse** a displacement-decorated model
  instead of erroring deep in the spin-only kernel — silently assuming `u = 0`
  would hide forgotten displacement data.
- **Force diagnostics**: `has_force`, `residuals_force`, `rss_force`, `r2_force`,
  `rmse_force` (uncentered baselines, mirroring the torque block).
- **Selection layer**: unchanged and force-unaware by design (the group cost model
  is not channel-split yet — design record §6): `cross_validate`/`select_fit`
  score energy(+torque) only (documented), `select_support` rejects a force
  co-fit.
- Gate (j) at model level plus the synthetic recovery plan A land in
  `test/unit/test_jointdata.jl`: `predict_force ≡ −FD(predict_energy)`, joint
  torque tangent FD, `X_F` columns ≡ the raw gradient kernel, exact OLS recovery
  of a synthetic joint model through energy/torque/force channels at several
  `(w_T, w_F)`, ragged force bookkeeping, and the structural-zero exclusion.
- Fix: `_referenced_atoms` (the zero-moment guard's notion of "spin-referenced")
  is now channel-aware — in a mixed SALC a displacement-only site (spin-inactive
  ligand) no longer counts as spin-referenced, so legitimate `lmax = 0` ligand
  configurations pass; pure-spin bases are unaffected (every member atom is a spin
  site).
- Review-round hardening (numerical + code review): joint-path **torque rows are
  restricted to spin-referenced atoms** (a displacement-only ligand contributes
  exactly-zero `X_T` and `y_T` rows, which would only dilute the `√(w_T/n_T)`
  whitening; the pure-spin constructors keep their historical all-atom layout for
  oracle parity, and `_assemble_problem` reads per-config row counts off the stored
  `torque_config`/`force_config`, never off `3·n_atoms`); the **displacement-radius
  guard** landed at the dataset boundary (warn when an amplitude exceeds half the
  shortest reference interatomic distance — the un-minimum-imaged-adapter symptom;
  `_min_reference_distance`); degenerate force blocks warn (all-zero targets, and
  an identically zero `X_F`, e.g. all degree-≥2 displacement factors at `u = 0`);
  `vcat` takes `force_cols` from any part that carries them (slicing away all force
  rows no longer drops the column set); the `AbstractDFTSource` convenience
  constructor forwards `use_force`/`zero_moment_atol`; the `1 − w_T − w_F`
  complement is clamped at 0 against boundary-sum rounding; both coverage warnings
  report a `coverage` value.

### Changed — joint data layer: `TrainingDatum` (M3 slice 2, BREAKING)

- **`SpinDatum` the type is gone**; the name survives as convenience constructors
  returning the new **`TrainingDatum`** — one concrete, code-agnostic training
  record for the spin-lattice pipeline. Required channels: `energy` (every code
  emits one; VASP constrained runs carry exactly one penalty `E_p` in TOTEN —
  verified, negligible/subtractable) and the spin channel (`directions` unit
  columns — collinear/Ising `±ẑ` is legal input — plus **nonnegative** `magmoms`;
  the sign lives in the direction, enforced). Optional channels, `nothing` =
  **not observed** (distinct from an observed zero): `displacements` (`nothing`
  ≡ `u = 0` exactly), `forces` (`f = −∂E/∂u`, legal at `u = 0` without
  displacements — reference forces are real physics), `field`, `torques`
  (derived `τ = m × B` when the field is present; direct-`torques` callers must
  match the Landau–Lifshitz convention stated in the docstring). Breaking for
  type annotations only (`::SpinDatum`, `Vector{SpinDatum}`); call sites of the
  constructor keep working, and a new 2-arg `SpinDatum(energy, moments)` covers
  field-less sources (collinear runs, codes without constrained noncollinear
  output).
- **`DatumProvenance`** (keyword-constructed) rides on every datum:
  `torque_qualified` gates torque rows into `X_T` and is **derived** as
  `any(!iszero, field)` by the `SpinDatum` constructor — a zero field carries no
  constraint information, so its `τ = 0` rows are not admitted unless the caller
  explicitly asserts a converged unconstrained run (whose `τ = 0` is a genuine
  stationarity observation) via `DatumProvenance(torque_qualified = true)`;
  `setup_id`/`soc` enforce **one computational setup per dataset** (cross-setup
  energies share no common scale, and SOC-sector basis functions do not vanish
  on Ising configs, so a mixed regression corrupts exactly the smallest
  couplings — multi-fidelity use goes through staged fits, per-family datasets +
  `frozen`/`sector_mask`); `reference_id` + `reference_fingerprint` pin the
  clamped-ion reference.
- **`crystal_fingerprint(crystal)`**: stable (Julia-version-independent,
  hand-rolled FNV-1a) content fingerprint of a `Crystal` — coordinates quantized
  at `1e-10` with the fractional wrap boundary folded, `-0.0` normalized,
  length-prefixed strings. Same-lineage identity by design, not symmetry
  equivalence. `SLCEDataset` requires every datum's fingerprint to match
  `basis.crystal` whenever the basis carries `p ≥ 1` SALCs — **keyed off the
  basis, not the datum**: against a displacement-decorated basis even spin-only
  data assert "u = 0 at the reference", so unpinned legacy data are rejected
  (the double-counting protocol as an invariant); pure-spin bases keep the
  unpinned path, and displaced data against a pure-spin basis are rejected
  outright.
- **Mixed (ragged) datasets**: `SLCEDataset(basis, data::Vector{TrainingDatum})`
  admits torque rows exactly for the qualified configurations, so "many cheap
  energy-only configs + few constrained noncollinear ones" is one first-class
  dataset. Torque rows of unqualified configs are **excluded, never zero-padded**
  (padding would silently inflate the `√(w/n_T)` whitening). The per-row config
  index `SLCEDataset.torque_config` is stored and read by every consumer:
  slicing/`vcat` re-label and re-offset it (`vcat` now legally concatenates
  torque-bearing and energy-only parts), `_assemble_problem` builds its
  grouped-CV labels from it (replacing the uniform-block `div(n_T, n_E)`
  derivation, which silently mislabels ragged data) and errors on a torque-free
  split under `torque_weight > 0`, and `cross_validate` stratifies its folds by
  torque presence (torque-free holdout folds score energy-only, never `0 · NaN`;
  `w > 0` requires ≥ 2 torque-bearing configs and caps the fold count so every
  training split keeps torque rows). A five-argument
  `SLCEDataset(basis, configs, energies, torques, torque_sel)` exposes the mixed
  form directly; the four-argument form is unchanged (bit-identical designs).
- `read_embset` returns `Vector{TrainingDatum}` and gains a `setup_id` keyword
  (stamped into every datum; per-config `torque_qualified` derived from the
  field block). `AbstractTrainingDatum` is removed (nothing dispatched on it).
- Review-round hardening: `SLCEDataset` stores a dataset-level **identity
  summary** (`provenance`: setup/soc/reference) so `vcat` — the documented
  incremental-addition path — re-asserts the one-setup/one-reference invariants
  instead of silently bypassing them (slices carry it; the raw constructors
  accept it as a keyword); the keyword `TrainingDatum` constructor derives
  `torque_qualified` by the same rule as `SpinDatum` (explicit `torques` qualify;
  an all-zero field does not) so the two construction paths of one type cannot
  disagree on the gate; `_assemble_problem` warns once (`maxlog = 1`) when a
  mixed dataset's torque rows cover only part of the configurations (the
  torque-bearing minority carries the whole `torque_weight`); `select_fit`'s
  `:cv` branch stratifies its fold deal by torque presence exactly like
  `cross_validate`; `_basis_has_disp` keys off the spec's `pmax` as well as the
  surviving SALCs (a symmetry-annihilated `p ≥ 1` spec must still pin the
  reference); `crystal_fingerprint` documents the quantization-boundary
  limitation honestly (straddling values mismatch loudly — pinned by a test).
- Deferred to the joint-design slice (with `X_F`): displacement validation
  beyond finiteness (minimum-image consistency, the displacement-radius guard)
  and the `p ≥ 1` design-matrix build itself (`SLCEDataset` still refuses a
  displacement-decorated basis at the design boundary, *after* the reference
  invariants run).

### Added — joint gradient kernel (M3 slice 1)

- `accumulate_grad!(Ge, Gu, salc, e, u, weight[, scratch])`: the joint
  (spin + displacement) gradient of a SALC. Spin axes accumulate the
  tangent-projected direction gradient into `Ge` (the torque convention,
  bit-identical to the single-buffer spin-only path on pure-spin SALCs);
  displacement axes accumulate the plain Euclidean `∂(|u|^{2k}R_{lm})/∂u`
  into `Gu` with no projection — the force convention is `f_a = −∂E/∂u_a`
  (design record §6), so model forces are `−Σ_ϕ jϕ·Gu_ϕ`. Shares the joint
  evaluator's channel-dispatched value tables and `(4π)^(n_spin/2)` scale;
  passing one array as both buffers is refused (the conventions differ).
  The spin-only `accumulate_grad!` still refuses displacement-decorated
  SALCs, now naming the joint form. Gate (j) runs at engine level in
  `test/unit/test_jointgrad.jl` (finite differences incl. a degree-4-exact
  stencil over `(k, l) = (1, 2)/(2, 0)` factors and an AllImages self-bond
  repeated-column case, the `Ge ⊥ e` tangency invariant, bit-identity,
  `u = 0` exactness, error surface); the model-level force design block
  `X_F` arrives with the M3 data layer.

### Added — synthetic recovery tests, plan A (test-only)

- `test/unit/test_recovery.jl`: Cartesian ground-truth spin–lattice models →
  sampled energies → OLS on the SALC design → held-out energies exact and the
  Cartesian input constants recovered through probe configurations (in-span
  recovery is a theoretical guarantee, so drift = a conventions bug). These
  are function-space gates: they pin span, selection rules, projection, and
  `u` indexing — not the per-SALC normalization, which stays pinned by the
  oracle and gates (e)/(g). Models:
  (A1) exchange striction + ligand path on the bent Fe–O–Fe unit incl. the
  DMI negative/positive pair (soc = false irrecoverable, soc = true recovers
  `D` exactly); (A2) the B₁/B₂ shell constants on Fe(O)₆; (A3) the chirality
  twist with full coefficient discrimination (only the `(L_S = 1, Lf = 0)`
  column carries weight) and the `L_S = 0` negative; (A4) spin-dependent
  force constants `Φ⁰ + (ê₁·ê₂)Φ¹` at `p = 2`. Plan B (recovery from forces
  and torques) lands with the M3 data layer; plan C (bcc Fe literature
  parameters) is demo material.

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
