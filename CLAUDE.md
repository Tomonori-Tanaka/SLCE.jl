# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

**Before writing, reviewing, or renaming code here, read
[`STYLE_GUIDE.md`](STYLE_GUIDE.md).** Its §1 is the SLCE-family naming contract —
mirrored verbatim in all five packages, canonical copy in this one — and the
sections after it are this package's own deltas. **Read
[`ARCHITECTURE.md`](ARCHITECTURE.md)** when you need the dependency graph, the
include layering, or a reading order through the source.

## Project goal

Clean, extensible, Julia-native rebuild of `Magesty.jl`: fit a spin-cluster
expansion (SLCE) `E = j0 + Σφ Jφ Φφ({e_a})` to noncollinear DFT data. The numerical
core (tesseral harmonics, Clebsch–Gordan coupling, symmetry-adapted basis, design
matrix, regression) is **reimplemented from scratch**; `Magesty.jl` is used only
as a *pinned numerical oracle* in `test/oracle/`. Priority: numerical correctness
and reproducibility over stylistic concerns. See `SPEC.md` for the realized
architecture and `docs/design-notes.md` for the rationale.

Both SLCE observables are fitted: the **energy** `E` and the per-atom **torque**
`τ_a = −e_a × ∂E/∂e_a` (the physical / Landau–Lifshitz torque `m_a × B_eff,a`, the
analytic derivative of the same energy surface). An
energy+torque co-fit minimizes `L = (1−w)·MSE_E + w·MSE_T` for a `torque_weight`
`w ∈ [0,1]`. Cluster enumeration is **arbitrary body order** (`nbody = K`):
pairwise-within-cutoff cliques, with the SALC projection generalized to the combined
(ordering × coupling-path × `Mf`) space so that permutation-equivalent sites — which
at `N ≥ 3` mix coupling paths and, for unequal `l`, `l`-orderings — are handled.

Bit-for-bit agreement with Magesty is **not** a goal — refining methods so results
differ slightly is allowed and is the point of the exercise. Validation is
physical/intrinsic (symmetry invariance, finite differences, known-coupling
recovery); the oracle compares convention-fixed kernels and gauge-invariant
aggregates, never raw gauge-dependent SALC coefficients.

## Numerical / physics conventions

Easy to break silently — confirm before touching the algorithm.

- **Spin directions are unit vectors**; `directions[:, a]` (norm 1) is the
  direction, the moment magnitude `magmom[a]` is stored separately.
- **Spin layout `3 × n_atoms`** (rows x, y, z; columns atoms). Transposing breaks
  the pipeline.
- **Real (tesseral) spherical harmonics `Zₗₘ`** (Drautz, PRB 102 024104), per-site
  factor `(4π)^(−1/2)`; the design matrix carries `(4π)^(N/2)` so an N-body term
  cancels to O(1).
- **Time reversal**: even-`Σl_s` channel only (odd-`Σl` `ls`-assignments are not
  enumerated). The SALC projector rotates each **site** axis by `wignerD_real(l_i, R)`
  (full `R`, improper included) and reads the `Lf` action off by contraction against
  the orthonormal coupled tensors; the per-site `(−1)^{l_i}` parity with even `Σl`
  makes the improper handling automatic (no explicit proper-part case), so
  symmetry-forbidden odd-`Lf` channels are dropped and allowed ones kept.
- **Lattice / reciprocal**: `Lattice.vectors` columns are `aᵢ`; `reciprocal =
  inv(vectors)` rows are `bᵢ` with `aᵢ·bⱼ = δᵢⱼ` (no 2π). Interplanar spacing
  `dᵢ = 1/‖row_i(reciprocal)‖`. Fractional coords are wrapped to `[0,1)` on
  periodic axes (neighbor-list precondition).
- **Periodic resolvability (minimum image / Wigner–Seitz)**: a finite supercell under
  plain PBC can only resolve interactions whose displacement lies in the **Wigner–Seitz
  cell** of the (super)lattice — a polyhedron reaching the body-diagonal corner
  `(L/2,L/2,L/2)` at `√3·L/2`, **not** a sphere of radius `L/2`. A farther periodic image
  of an atom carries the *same spin* — and, since a datum's displacement field is
  cell-periodic (`3 × n_atoms` of the REFERENCE cell), the same displacement — so its
  interaction is collinear with (an alias of)
  the minimum-image one and is not independently fittable. The default `MinimumImage`
  selection enumerates exactly this set (boundary ties kept; `i==j` self-pairs and
  reused-atom clusters dropped); `cutoff = Inf` is the whole WS cell.
  **The `i==j` drop is invisible in the fit and visible in the READOUTS**, so state the
  condition positively wherever force constants are documented: a pair is representable
  iff its ends are distinct atoms *of the reference cell*, at their minimum-image
  separation within cutoff — which reaches the WS corner, so a cutoff cap is not the
  limit (origin ↔ body centre of a cubic cell, `0.866 L`, is representable). What is
  excluded is an atom with its own image at ANY distance, hence
  `Φ[(a,0),(a,R≠0)]` is never emitted by `force_constants` and the `q`-dependence it
  would carry is missing from `dynamical_matrix` — `write_phonopy`/`write_alamode` then
  export that silently. Measured 2026-07-30: a 1-atom cell yields body orders `[1]` only,
  `build_asr` warns "no translation-invariant displacement content", every displacement
  coefficient is constrained to zero and `D(q) ≡ 0` (so the failure is LOUD, not a
  plausible-looking dispersion); the same crystal as a 2-atom cell gives `q = 15` free
  dims, off-site `Φ`, and `‖D(q)−D(0)‖ = 3.4`. Advice is therefore "describe the crystal
  with a cell in which those atoms are separate atoms", never "raise the cutoff". Prose
  lives in `docs/src/theory/resolvability.md` §"Which interactions the enumeration can
  represent", the warning box in `docs/src/guide/lattice_dynamics.md`, and the
  `MinimumImage` / `force_constants` docstrings — all four move together. `AllImages`
  (every image, `R`-distinguished) is **never fittable** — `SLCEDataset` refuses a basis
  with self-image members (`_refuse_self_image_basis` → `UnclassifiableBasis`). It serves
  two non-fitting consumers: the future generalized-Bloch / spin-spiral path where
  `e^{iq·R}` resolves the images, and the **tiling template** a downstream consumer
  expands onto a supercell (a monatomic cell's NN bond can only be *written* as a
  self-image cluster; SLCEMonteCarlo's cubic-Heisenberg tutorial and SLCEDynamics'
  `examples/bloch_mt.jl` are the live cases, coefficients set by hand). For `cutoff < min_d dᵢ / 2` the
  two coincide. Do **not** "fix" a `> L/2` cutoff by folding aliases into a shorter shell
  (double-counts) — that regime is simply unresolvable from one supercell.
- **Energy units**: `Jφ` carry the DFT input unit (eV); `j0` is separate.
- **Torque**: `τ_a = −e_a × ∂E/∂e_a` (the physical / Landau–Lifshitz torque `m_a × B_eff,a`,
  matching the *General spin models* paper; the opposite sign of the energy-rotation-gradient
  `+e×∇E`), design-matrix entry `(4π)^(N/2)·(−e_a × ∂Φ/∂e_a)` = `(4π)^(N/2)·(∂Φ/∂e_a × e_a)`
  (same scale and `μ`-mapping as the energy kernel). The torque design matrix `X_T`
  has no `j0` column. Its rows are flattened config-major, then atom-major, then
  `xyz`. The co-fit whitens the (centered) energy block by `√((1−w)/n_E)` and the
  torque block by `√(w/n_T)`; `j0` stays an energy-only quantity.

- **Invariance layers — what a fitted model does and does NOT guarantee.** Two
  independent layers, and this package implements one and a half. **Layer 1**
  (space-group symmetry) is the SALC projection, and it is verifiably the *same*
  space as the textbook Cartesian force-constant reduction (identical subspace,
  all principal angles zero, on both a P4/mmm and a triclinic fixture) — so no
  symmetry deficit versus ALM/hiPhive, and **no numerical advantage either: never
  claim the irreducible basis conditions better**, because irreducible ↔ Cartesian
  is an orthogonal transform and the singular spectrum is identical (measured cond
  ratio 1.000000). **Layer 2** (affine invariance: uniform motions with the cell
  held fixed) is *not* a space-group property and cannot be projected out at
  basis-construction time — which is why `build_asr` is a separate constraint on
  coefficients. Of the three affine families only `u → u + t` (ASR) is
  implemented; **Born–Huang rotational (`ε_antisym`) and equilibrium/vanishing-stress
  (`ε_sym`) are NOT**, and they are independent of each other and of the Huang
  conditions (`rank[Huang; Q] = 21 = 15 + 6`). Benign at harmonic order with
  relaxed (`σ ≈ 0`) training data — the `ε_sym` six *are* `σ = 0` — but not
  automatic at anharmonic orders, and load-bearing in the M5 strain channel where
  `ε` is an explicit model variable. Read `docs/design-notes.md` §15 before adding
  any constraint layer: it records why it must sit next to `build_asr` rather than
  in the SALC builder, and the truncation-boundary hazard (the condition couples
  order `n` to `n+1`, so at the top retained order it degenerates to
  `L·Φ⁽ⁿ⁾ = 0` and over-constrains — kernels `1,1,3,6,15` generic / `1,0,1,0,1`
  fully symmetric, i.e. odd order is killed outright).

## Coupled ("linked") code sites — change one, check all

- `basis/Harmonics.jl` (`Zlm`, `grad_Zlm`) ↔ the on-sphere central-difference and
  closed-form agreement tests (`test/unit/test_harmonics.jl`). Normalization / sign
  drift silently biases `X`.
- **Unit-norm validation is a DOOR's job, never a kernel's** (`direction.jl`
  `UnitVector3`/`SpinConfiguration`/`_validate_direction` ↔ `slce/model.jl`
  `_validate_config` ↔ `slce/forceconstants.jl`
  `_resolve_spins` ↔ `slce/rowlayout.jl` `site_rows!` ↔ `slce/effective.jl`
  `predict_energy(::EffectiveModel, …)` ↔ `fitting/fit.jl` `_validate_config_pair` ↔
  `io/dftsource.jl` `TrainingDatum`): "this is a unit vector" is an invariant with no
  representation — it lives only as a rule each door must remember — so the rule is
  stated ONCE (`_validate_direction`: finite, `|‖e‖ − 1| ≤ atol` at the package's
  `1e-6`, and `max|component| ≤ 1`) and every door calls it — increasingly by
  constructing a `SpinConfiguration`, which cannot exist without it. **The
  `_DIRECTION_ATOL_MAX` cap binds EVERY door that takes an `atol`, projecting or
  not** (since the 2026-08-11 review): `_validate_config` is a non-projecting door
  whose raw columns enter the design matrix, so a widened band there buys a
  `C_l·δ` design bias directly — it used to accept any `atol`, which let
  `atol = 0.5` fit a ~98 %-biased design from moment-scaled columns. **The two
  constructors are not a convenience pair**: the projecting one is for data entering
  the package, the `Trusted` one for a value an evolution already placed on the
  sphere, and conflating them breaks a bit-identical resume, because `normalize` is
  not bitwise idempotent (~38 % of already-unit directions move) and SLCEDynamics'
  default integrator preserves `|e| = 1` by construction with no renormalization step.
  **Validation order is load-bearing**: ask "is this a direction?" before projecting
  (that is what still rejects a moment-scaled vector), and ask the component bound of
  the PROJECTED value (that is a precondition on what the kernels see) — asking it
  first would refuse the near-pole case projection exists to repair. Below a door,
  kernels
  use `Zlm_unsafe`/`grad_Zlm_unsafe`; the checked entries exist for callers who
  cannot guarantee `(l, m)`, which is a SEPARATE hazard (`|m| > l` reaches
  `_plm_norm`'s zero division and returns `NaN`) that no door discharges.
  **The component bound is the load-bearing half and the tolerance is not**: `Zlm`
  reaches `dnPl`, whose domain is `|e_z| ≤ 1`, and near a pole any `δ > 0` can push
  past it — measured, a column `5e-9` off norm clears a `1e-8` band and still throws
  a bare `DomainError` from inside the accumulation. Tightening `atol` therefore
  cannot establish the precondition; only the component bound can, and it rejects
  nothing harmless (a column off norm by `1e-7` whose largest component is `0.8`
  passes). **Every door validates unconditionally, including on a lattice-only model
  where nothing reads the columns**: the omission is `nothing` and only `nothing`, so
  passing a matrix is a claim that it IS a magnetic state. `_validate_config_pair` and
  `predict_energy(::EffectiveModel, …)` used to make an exception (shape only) so that
  the spin-free path's all-ZERO marker could pass through them, and that exception was
  what made the marker legal enough to reach a spin-carrying path. The marker is now
  a LOCAL filler produced at the `::Nothing` method itself
  (`_spin_kernel_matrix(nothing, nat)`, `slce/model.jl`) and handed to an unvalidated
  kernel (`_joint_energy` / `_joint_force` / `_affine_energy` / `_effective_energy`) —
  it never re-enters a door, so no door needs to know it exists.
  Adding a new entry point that reaches a spin factor means adding a door — the
  audited defect was exactly a door added without one, and NOTHING downstream can
  catch it: measured, the acoustic modes of `D(0)` and `asr_residual` are completely
  blind to an off-unit spin state (the ASR's rows are keyed per spin monomial, so
  `Aβ = 0` is an identity in `e` and holds for a wrong `e` too).
- **Energy kernel `evaluate_salc` ↔ gradient kernel `accumulate_grad!`** (`basis/salc.jl`):
  identical `μ = idx[i] − ls[i] − 1` mapping, `ls`, `folded`, and `(4π)^(N/2)` scale.
  The gate is the finite-difference self-consistency `predict_torque ≈ −e × ∇E_FD`
  (`test/unit/test_torque.jl`, `test_nbody.jl`): the torque must be the exact (negative
  rotation-) derivative of the energy surface. Change one kernel, re-check the other.
  Both spin-only forms REFUSE displacement-decorated SALCs (the refusal is the guard
  against silent mis-scaling); the joint forms are `evaluate_salc(salc, e, u)` and the
  two-buffer `accumulate_grad!(Ge, Gu, salc, e, u, weight)` (M3 slice 1 — spin axes
  tangent-projected into `Ge`, disp axes Euclidean into `Gu`, force sign `f = −∂E/∂u`
  applied by the caller). The joint pair shares `_fill_ztables_mixed!` and the
  `(4π)^(n_spin/2)` scale, and its gate is the engine-level finite-difference suite
  (`test/unit/test_jointgrad.jl`) plus bitwise identity with the spin-only gradient on
  pure-spin SALCs. The model level rides these two kernels: the joint designs
  (`_design_energy`/`_design_torque` with `disps` and the compact `_design_force`,
  `fitting/design.jl`) and the joint predicts (`predict_energy`/`predict_torque`/
  `predict_force(model, e, u)`, `fitting/fit.jl`). The force sign `f = −∂E/∂u` is
  applied in exactly two places — `_design_force` and `predict_force` — and gate (j)
  at model level (`test/unit/test_jointdata.jl`) fences both against the energy
  surface by finite differences; change either sign site and re-check it.
- **The fractional wrap ↔ the AllImages image box ↔ `crystal_fingerprint`**
  (`geometry/crystal.jl` inner constructor ↔ `geometry/neighborlist.jl` ↔
  `io/dftsource.jl` `_fp_quant`): `Crystal` wraps periodic coordinates into a
  **half-open `[0, 1)`**, and that is a precondition, not a nicety. `mod` alone does not
  deliver it — `mod(-1e-18, 1.0) === 1.0`, because the exact result falls below the
  float resolution near 1 — so the constructor snaps `>= 1.0` back to `0.0`. The
  `AllImages` box `nrange = ceil(cutoff·‖b_d‖)` is exactly tight and its derivation
  needs `|Δf_d| < 1` **strictly**; with a coordinate at `1.0` it silently drops the
  shell sitting on the cutoff sphere (measured 102 pairs against a brute force of 104
  on a triclinic cell). `_fp_quant`'s single `-= 1.0` fold covers the same boundary from
  the I/O side and is now defence in depth rather than the only guard — both halves are
  pinned (`test_geometry.jl`'s wrap testset, `test_dftsource.jl`'s fingerprint one), so
  removing the snap turns the geometry side red and removing the fold turns the I/O side
  red. `MinimumImage` survives a `1.0` only by the slack of `_sufficient_range`'s `+1`,
  i.e. by luck, not by its stated argument.
- **Image selection ↔ neighbor list ↔ cluster edges** (`geometry/neighborlist.jl`,
  `clusters/enumerate.jl`, `slce/model.jl`): `SLCEBasis` threads one `images` value to
  **both** `build_neighbor_list` and `candidate_clusters`/`build_clusters`; they must
  agree. `MinimumImage` keeps minimum-image pairs (no `i==j`) and admits a clique edge
  only at its atom-pair minimum-image distance with all atoms distinct; `AllImages` keeps
  every in-cutoff image and admits edges within the radial cutoff. The tie/cutoff
  tolerance is relative (`_SAME_DIST_RTOL`) on both sides so a degenerate WS-boundary
  shell is never split; it is user-facing as `SLCEBasis(...; tie_tol)` (default the
  same constant, hard cap `_TIE_TOL_MAX = 1e-2`), riding on `NeighborList.tol` which
  `candidate_clusters` reads back — one value, both sides. Widening it is the remedy
  for relaxed/noisy coordinates whose symmetry residual splits ties (the orbit
  builder's closure refusal names it; the MnTe(0001) slab case, fixed first in
  SCEFitting.jl); the resolvability layer then freezes whatever the merged shell's
  aggregation cancels, exactly as for exact WS ties. Gate: the perturbed-honeycomb
  closure testset in `test/unit/test_clusters.jl` (refuses at the default band,
  builds the ideal SALC keys end-to-end through `SLCEBasis(tie_tol = 1e-3)`).
  The minimum-image search box is adaptive — change it and re-check
  the skewed-cell test. `images` is **not** persisted (the full SALC basis is stored and
  reloaded verbatim), so only `read_setup`/`SLCEBasis` carry it — and the same rule
  holds for `tie_tol` (`[interaction].tie_tol` in the TOML schema): both change the
  emitted basis, so both must round-trip through the setup file. **At `N ≥ 3` the clique
  check is on the actual chosen images of *every* pair (not just the anchor edges): a
  cluster is admitted only when all `C(N,2)` edges sit at their atom-pair minimum image
  simultaneously** (the compact-cluster criterion). Having each pair individually
  minimum-image-resolvable is *not* sufficient — the images minimizing `i–j` and `i–k`
  may force `j–k` onto a longer image, which must reject the cluster. WS-boundary ties
  then multiply N-body clusters just as they do pairs. The whole count (candidates and
  symmetry orbits) is pinned against an independent brute force in
  `test/unit/test_ws_nbody.jl`; change the edge rule or the search box and re-check it.
  **Per-body × per-species-pair cutoffs** layer on top: the neighbor list is built at
  the element-wise max over body orders (each pair admitted against its own
  species-pair radius, tie band per pair), and `candidate_clusters` re-checks every
  edge against *that body order's* radius — so `neighbors` is a superset that the
  per-order matrices trim. The per-pair admission has its own brute-force gate in
  `test/unit/test_truncation.jl`; change either side and re-check both pins.
- **Pure-spin ↔ decor SALC engines** (`basis/salcbasis.jl`):
  `_project_and_fold`/`_transport_term`/`_enumerate_ls` (production, oracle-pinned
  bitwise) and `_project_and_fold_decors`/`_transport_term_decors`/
  `_orbit_salcs_decors` (mixed channels) are two implementations of ONE
  construction. The decor engine must reproduce the pure-spin engine's
  enumeration order exactly — orbit representatives discovered in colex
  (`Iterators.product`) order, lex-min representative, assignments
  `unique(rep[p])` — or `block` indices and the canonical gauge silently change
  and key-addressed coefficient re-pairing breaks. Gate: "engines agree on pure
  spin" in `test/unit/test_mixedsalc.jl`, incl. the Cs-triangle (2 ordering
  orbits) and C3v-triangle (3 assignments) shapes. Change either engine → re-run
  that gate + the oracle suite.
- **SALC construction ↔ the ground-truth invariance test** (`test/unit/test_salc.jl`,
  `test/unit/test_nbody.jl`, `test/oracle/runtests.jl`): every SALC must satisfy
  `Φ(g·e) = Φ(e)` (non-collinear spins, all `Lf`, **all body orders**) and
  `Φ(−e) = Φ(e)`. The combined-space projection action (`_project_and_fold`) and the
  member transport (`_transport_term`) must use the *same* rotation direction and
  axis-relabel convention (`invperm(perm)`); the eigenvalue-exactly-0/1 idempotency
  assertion and the invariance test are the gates. **The construction's last step is
  `_canonicalize_members`** (`basis/salc.jl`): the ordered, anchored images (one per
  site ordering — the space the projection needs) are folded to one member per
  physical instance (sites sorted by `(atom, shift)`, `shifts[1] == 0` re-anchored,
  tensors `permutedims`-aligned and summed per site→`l` assignment — an exact
  regrouping). Everything downstream (evaluate/gradient kernels, introspection,
  persistence, analytic test references like the oracle's Heisenberg
  `Φ = 2√3 Σ_undirected`) assumes the canonical form: one member per undirected
  instance, tensors carrying the whole (formerly per-image) weight. Gates:
  `check_canonical_members` / `split_roundtrip_exact` (shared helpers in
  `test/unit/testutils.jl`, applied by `test_salc.jl` and 3-body `test_nbody.jl`)
  and the pre-v4 fold test in `test_persist.jl`. At `N ≥ 3` also confirm SALCs are
  linearly independent (design-matrix rank = #SALC). **The orbit loop in
  `build_salc_basis` is threaded** (`Threads.@threads`, one task per orbit via
  `_orbit_salcs`): orbits are independent and the output is sorted by `SALCKey`, so the
  basis is byte-for-byte thread-count-independent. The precondition is that the Wigner-D
  cache is **precomputed serially** (`_build_wig_cache`, full bounded `(l, g)` grid) and
  **read-only** inside the loop — `_wig` is a pure lookup, never a `get!`. Any new shared
  mutable state in the per-orbit path (or a lazily-filled cache) reintroduces a race;
  keep it task-local. Gate: `test/unit/test_salc.jl` "build is deterministic / thread-safe"
  plus the oracle, run under `julia -t N>1`.
- **`SALCKey`/`SALC` field surface ↔ ALL test environments**: the unit suite is
  not the only consumer — `test/sunny/runtests.jl`, `test/glmnet/runtests.jl`,
  `test/oracle/runtests.jl`, and `examples/*.jl` read key/SALC fields and are
  NOT exercised by CI (`TEST_MODE=all` only). Rename a field → grep all of
  `test/` and `examples/` (the M2a rename missed sunny/glmnet/examples until
  review).
- **Design-matrix columns are identified by `SALCKey`** (`SALCBasis.keys`, sorted),
  not by construction order. The key must stay **injective**: `block` runs across all
  canonical `l`-orderings that share one sorted `ls` label (a proper-subgroup site
  stabilizer splits a degenerate multiset into several ordering orbits — see
  `test/unit/test_nbody.jl`). `SLCEModel` re-pairs `jphi` to a basis **by key** on any
  reload (positionally paired only within a session); `fingerprint = hash(sorted keys)`
  guards against cross-basis confusion. This reload is realized by persistence
  (`SLCE.load(SLCEModel, …)` rebuilds `jphi` in basis-key order from the
  per-`SALCKey` coefficients).
- **`write_phonopy`'s supercell ordering is an EXTERNAL convention** (`io/phonopy.jl`
  ↔ `test/phonopy/`): `FORCE_CONSTANTS` is a matrix over supercell atom indices that
  phonopy builds itself from the unit cell and `--dim`, so a disagreement produces a
  permuted force-constant matrix that still diagonalizes and still has three acoustic
  modes at Γ. **No self-contained round trip can gate it** — the core suite's checks
  (`test_latticeonly.jl`) pin the tiling and the wraparound, and pass under any
  *consistent* map. The convention was read off phonopy (`s(a,l) = (a−1)·∏d + l₁ +
  d₁(l₂ + d₂l₃) + 1`, atom slowest, `l₁` fastest) and only the phonopy suite defends
  it. Three fixture properties there are load-bearing and were each found by mutation,
  not by design: the comparison is MASS-WEIGHTED with two different masses (phonopy's
  phase depends only on the lattice translation, so permuting basis atoms is a
  similarity transform — the species-permutation bug moves an unweighted comparison by
  8e-16 and a weighted one by 1.7e-2); species are INTERLEAVED (the POSCAR grouping is
  a real permutation, which is why POSCAR and FORCE_CONSTANTS are written together);
  and the shift set is NOT symmetric under swapping the first two axes (the first
  fixture had `R ∈ {0, ±(1,1,0)}`, and reversing the lattice-axis order produced a
  BYTE-IDENTICAL file). Change the writer → re-run the phonopy suite, and if you
  change the fixture, re-verify all three mutations still bite.
- **`write_alamode` is the SECOND external convention, and a harder one**
  (`io/alamode.jl` ↔ `test/alamode/`, local-only like the oracle — `anphon` is a C++
  binary wanting Boost/FFTW/spglib/MPI). Three things are positional at once and none
  came from documentation: `pair1` names a PRIMITIVE atom (`anphon` resolves it via
  `map_p2s[a][0]`) while `pairK≥2` names a SUPERCELL atom; the supercell order is ours
  but `Data.Symmetry.Translations` must match it with translation 1 the identity; and
  `cell_s` indexes a fixed 27-entry table of SUPERCELL shifts (origin first, then
  `ix,iy,iz ∈ {-1,0,1}` skipping it, `iz` fastest) which is what keeps a shift's SIGN
  — folding to the residue with `cell_s = 1` moves frequencies by 5.2e-1
  (mutation-tested). Values are Rydberg atomic units. The gate fits ONE scale over all
  modes at all q and asserts the absolute residual after it (3.7e-5 cm⁻¹, bounded
  below by anphon printing four decimals — a RELATIVE tolerance is meaningless on a
  soft mode) AND that the scale is 1 to 5.5e-10; the two claims are separate, and a
  single `≈` loose enough to absorb the constants would pass a sign error. The cubic
  block is proven to PARSE and be consumed (`GRUNEISEN = 1` refuses without it), never
  numerically verified — the harmonic block pins units/sign/`cell_s` and the cubic
  block rides the same code path. **`anphon` uses `boost::lexical_cast`, which rejects
  leading whitespace**: a padded `%25.15e` in the value field aborts the reader, so
  the core suite checks every value parses bare.
- **The spin-free entry path shares ONE predicate AND one refusal** (`slce/model.jl`
  `_basis_has_spin` / `_require_spin_free` / `_resolve_spins` /
  `_spin_kernel_matrix` ↔ `io/dftsource.jl` `SLCEDataset`'s `use_torque = nothing` ↔
  the derivative readouts `force_constants` / `strain_derivatives` /
  `magnon_phonon_vertices` / `grid_strain_derivative` ↔ `fitting/fit.jl`
  `predict_energy`/`predict_force(model, nothing, u)` ↔ `slce/affine.jl`
  `affine_energy` / `rotational_residual` / `rotation_transfer_residual`): every place
  that lets a lattice-only caller
  omit a magnetic state asks the same question, and it is **not** `is_soc_free`.
  The rule and its message live beside the predicate in `model.jl` rather than at any
  one consumer — they were two copies with two different messages (and two spellings
  of the same argument) until the doors were unified. `EffectiveModel` is the one
  entry that cannot read it: a re-expansion carries no `SALCKey`s, so the question
  "does anything here read a spin?" is answerable only from its own term list, and
  `slce/effective.jl` states it there.
  `is_soc_free` asks whether a label's total spin rank `L_S` vanishes — true for most
  ordinary `soc = false` spin labels — so a "has no spin content" test built on it
  waves a spin-carrying model through and evaluates it at a fabricated state. The two
  are gated against each other on a basis where they disagree
  (`test/unit/test_latticeonly.jl`). `_basis_has_spin` reads the SPEC as well as the
  surviving SALCs, exactly like `_basis_has_disp` and for the same reason. The
  omission marker everywhere is all-ZERO, never a plausible `+ẑ`: `lattice_datum`'s
  zero `magmoms` is what makes `SLCEDataset`'s existing zero-moment invariant reject
  the datum if it ever meets a spin-carrying basis, so that guard is load-bearing for
  the convenience constructor and must not be relaxed. `use_torque` resolves from the
  BASIS and never from whether the data carry torques — the latter converts "the
  adapter dropped the constraining fields" from a loud error into a silent
  energy-only fit.
- **`torque_qualified` is upgraded by every datum constructor, never revoked**
  (`io/dftsource.jl` `TrainingDatum` keyword ctor ↔ `spin_datum` ↔ `joint_datum` ↔
  `io/provenance.jl`): a joint datum MUST carry a provenance (the displacement
  channel requires `reference_id`/`reference_fingerprint`), and a hand-built one
  carries the struct default `false` — which used to discard the qualification that
  explicit `torques` or a nonzero `field` earn, so a datum with displacements AND
  torques died at the dataset with "pass `use_torque = false`", the opposite of the
  caller's intent. Keep the two construction paths' rules identical (they are
  compared in `test_dftsource.jl` for exactly this reason) and keep the upgrade
  one-way: suppressing a torque channel is `use_torque = false`'s job at the dataset
  level. `joint_datum` exists to own exactly this combination — it stamps the
  reference AND derives the qualification, so the joint corner no longer needs a
  hand-built provenance at all; its gate is `test/unit/test_latticeonly.jl`, which
  asserts BOTH survive one call. The three convenience constructors
  (`spin_datum` / `lattice_datum` / `joint_datum`) are functions returning
  `TrainingDatum`, deliberately snake_case: `SpinDatum` was a type once, and the
  UpperCamelCase spelling kept promising a type that no longer exists.
- **The moment channel's evaluation axis is keyed by `constraint_mode`, never by
  field presence** (`io/dftsource.jl` `TrainingDatum` ctor invariants ↔
  `check_moment_gates` ↔ `io/extxyz.jl` reader/writer ↔ `io/embset.jl`
  `read_embset_pair` ↔ `fitting/momentfit.jl` `MomentDataset`/`predict_moment`): mode 4
  (direction-pinning type) reads `ê` from `directions`, mode 1
  (transverse-penalty type) from `constraint_axes` — an availability-keyed
  fallback ("use `mconstr` if present, else MW") would silently drop a mode-1
  datum with missing axes into the broken `ê_MW` coordinate (`‖M‖ → 0` folds the
  MW direction; measured σ 2.1× on FeRh), so the ctor REFUSES mode 1 without
  axes AND axes without a declared mode. The gates (`check_moment_gates`) run at
  every boundary an axis-carrying datum crosses — extxyz generation, extxyz
  load, the EMBSET pair reader — because archived axes are re-verified, never
  believed. Two subtleties the tests pin (`test_extxyz.jl`): a whole-axis flip
  in mode 1 flips `y` with it (the axis sign is a GAUGE — the sign gate must NOT
  fire, and "fixing" it to fire would refuse every legitimately re-gauged
  archive), and the angle gate is a PERCENTILE (p99 < 5°), because collapse rows
  legitimately carry large single-row angles (FeGe τ0.5 max 5.6° at p99 0.14°).
  The same never-trust-the-flag rule shapes the extxyz reader: spin-only vs
  joint is MEASURED from positions (bitwise across frames), `config_type` is
  only a cross-checked claim. `moments_bare` (bare `M_int`) and the smoothed
  `magmoms·directions` (`MW_int`) are BOTH stored and neither substitutes for
  the other: the constraint acts on MW (τ and the configuration coordinates),
  the projection target reads M_int, and their ratio is configuration-dependent
  (0.691 ± 0.017 on FeGe τ0.5). VASP vocabulary (OSZICAR/INCAR parsing, λ
  printing, SAXIS, sign conventions) stays in SLCETools' generator; the
  `constraint_mode` numbers follow `I_CONSTRAINED_M`, but the KEY is the
  physical class — another code's scheme maps onto class 1 or 4 at its adapter.
  The dataset layer (`fitting/momentfit.jl`) carries the rule's consequences:
  `predict_moment`'s runtime default `axes = e` IS the mode-4 identity
  substitution (change one and the train/predict coordinates split); a mode-1
  marked atom with an exactly-zero axis has NO defined target — its rows are
  `defined = false` with `y = NaN` (loud on raw use) and a placeholder design
  row, excluded from BOTH the gated and ungated solves, never imputed; and the
  fit does NO centering and adds NO global intercept because the l = 0 1-body
  `[MARK]` columns already are the per-orbit intercepts μ₀ — wiring the moment
  design into any estimator path that centers columns or appends an intercept
  column double-counts μ₀. `predict_moment` is a spin-reading entry point and
  therefore a validating DOOR (the unit-norm rule): `e` unit everywhere, `axes`
  unit on the MARKED columns only (unmarked axes columns are never read — a
  whole-matrix door would refuse the legitimate closed-form ê = x̂/ŷ/ẑ readout).
  `MomentDataset` runs `moment_resolvability` at construction and `fit` freezes
  the vanishing columns to EXACT zero — the same frozen-column discipline as the
  energy side, so `coef != 0` reads structure; weakening either half silently
  reintroduces arbitrary min-norm coefficients for columns no cell determines.
  Three follow-ups ride that discipline (2026-08-20). (1) `moment_resolvability`
  caches its default-`rtol` result ON the basis (`MomentBasis.resolvability`, a
  `Ref` — the answer is a pure function of the basis and the dataset door runs
  the gate at every construction); cached calls return the SAME object (`===` is
  the tested contract), non-default `rtol` recomputes, an `UnclassifiableBasis`
  refusal is never cached. (2) its null report must stay COMPLETE on a WIDE
  signature block: the economy SVD's `V` lists only `min(r, c)` flat directions
  and the complement of `span(V)` is read off a QR completion — dropping that
  branch silently empties the dependency disclosure exactly when columns
  outnumber signature rows (found at rank 20 of 74 kept, ZERO combinations
  reported). (3) `fit(MomentFit, …)` hands column-structured estimators through
  `_reduce_to_active`: full-basis `GroupAdaptiveRidge` labels must shrink with
  the vanishing-column freeze. Group NORMS are exactly preserved (frozen
  coefficients are exact zeros), but the group size `p_g` drops by the frozen
  count, so `w_g = v_g/(‖β_g‖² + p_g·ε)` moves at O(ε) — material only for a
  group already at the ε floor, and defensible there (ε is a per-coefficient
  floor; a frozen column carries no coefficient). This deliberately DIVERGES
  from the energy side, where ASR-frozen columns stay in `column_groups` and
  keep their `p_g` contribution. Emptied groups are relabeled away. And the
  pointed `salc_groups(mb)` key is `(body, orbit_id, decors, mark class)`, NOT
  the energy-side `(body, orbit_id, decors)`: pointed keys sort the decoration
  into a canonical multiset, so stabilizer-inequivalent mark placements — an
  Fe–Ge pair marked on Fe vs on Ge, channels predicting different atoms — share
  the energy key and differ only in `block`; folding them couples the adaptive
  shrinkage of distinct physical channels (mixed-species mark classes have
  provably disjoint design row support — the gate). The class fingerprint is
  the representative member's DISP-slot atom set AND site-index set — the atom
  set alone is NOT injective: a canonical member carrying two periodic images
  of one atom projects two distinct mark placements onto one atom set
  (review-reproduced on a 2-atom P1 cell at `nbody = 3`; regression-pinned).
  `MomentDataset` refuses a referenced atom (marked or environment,
  `_referenced_atoms(::MomentBasis)`) with `‖MW‖ ≤ zero_moment_atol` — its
  direction is the ẑ placeholder the reader fabricated at ITS atol, so a dataset
  built from `read_extxyz(...; zero_moment_atol = x)` must pass the same `x`
  (the same obligation `SLCEDataset` states). The MODE RULE has exactly one code
  statement, `_moment_axis_matrix`
  (`fitting/momentfit.jl`) — the dataset constructor AND the local-field
  diagnostics (`moment_local_field` / `moment_simple_floor`) resolve row axes
  through it; a second inline `mode == 4 ? directions : constraint_axes` is the
  drift hazard this extraction removed. The diagnostics' neighbor set is
  `_pair_neighbors` — the pair enumeration's own convention (MinimumImage at
  `cutoff_pair`, tied images each counted, `i == j` dropped, `lmax_env = 0`
  species excluded); `moment_simple_floor`'s nesting claim (`sigma_model ≤
  sigma_floor`) is conditional on the reported per-feature `inclusion`, never
  asserted unconditionally. Note for fixtures: no `soc = false` pointed
  face-(a) case is known — for
  `Lf = 0` the transport matrix is `D⁰ = 1`, so every image of an assignment
  carries an IDENTICAL folded weight and nothing of opposite sign exists to
  cancel under the periodic fold (an argument for single-assignment `Lf = 0`
  blocks, not a proven general theorem); use `soc = true` to build one.
- **The pointed moment basis rides the decor engine, and three conventions keep it
  honest** (`basis/momentbasis.jl` ↔ `basis/salcbasis.jl` `_orbit_salcs_decors`'s
  `admit` kwarg ↔ `clusters/orbits.jl` `_orbits_from_members` ↔
  `clusters/enumerate.jl` `candidate_clusters`): (1) **member multiplicity is the
  engine's all-orderings convention** — `candidate_clusters` lists every physical
  instance once per site ordering (3! for a 3-body), and the SALC value scales with
  that count, so a pointed enumeration emitting fewer orderings silently rescales
  its columns per orbit (measured: half the prototype's 6.0 star oracle) —
  `_pointed_star_candidates` therefore expands every translation class to all 3!
  re-anchored orderings, and any new candidate source must do the same. (2) an
  `admit` predicate handed to `_orbit_salcs_decors` is judged on the lex-min
  representative of each permutation orbit, so its verdict MUST be a
  permutation-orbit invariant — anything built from (decor, species,
  edge-lengths-from-site) data is, because stabilizer perms preserve species and
  are isometries; a rule reading raw site indices is not. (3) the marked-column
  substitution in `_design_moment` is exact only because every pointed label
  carries exactly one mark (a member not marked at the row's atom dies on its
  |u|² = 0 factor before reading the substituted column) — a future label with
  two marks breaks the argument, not just the numbers.
  (4) **a star never repeats a reference-cell atom**: two minimum images of one
  neighbor, or an environment on an image of the mark, put two spin factors on ONE
  sphere — under plain PBC both read the same `e_a`, the product is Clebsch–Gordan
  reducible, and the member is a lower-body function wearing an N-body label (the
  monomial signature would overcount the rank; measured 108 symbolic vs 98 actual on
  the FeGe primitive cell). `_pointed_star_candidates` refuses that shape at the
  enumeration, the same rule `candidate_clusters` applies under `MinimumImage`, so
  the cell keeps the stars it can resolve and says so through the empty-sector
  warning instead of losing every sound column to a whole-basis refusal.
  `moment_resolvability`'s `UnclassifiableBasis` stays as the BACKSTOP for a
  candidate source that skips the rule (`fold_members_onto_one_atom` in
  `test/unit/testutils.jl` is how the tests play that source). An `AllImages` /
  generalized-Bloch star WOULD resolve those members; there is no such path here. The gates in
  `test_momentbasis.jl` are INDEPENDENT references (a geometric triangle
  enumeration, the 2√3 shell sum, symbolic ≡ random-design rank) — keep them
  that way; a reference through the SALC machinery gates nothing.
- **Persistence schema ↔ the serialized structs** (`io/persist.jl`): `_to_doc` /
  `_from_doc` mirror the fields of `Crystal` / `Lattice` / `SpaceGroup` / `BasisSpec`
  / `SALCKey` / `SALCTerm` / `SALCMember` / `SALC` / `SLCEBasis` / `SLCEModel`. Add or
  rename a field on any of these and both halves (and `test/unit/test_persist.jl`'s
  round-trip) must follow; the space group is rebuilt via `_assemble_spacegroup` from
  the stored fractional ops, and the `UInt64` fingerprint is stored as a string and
  **recomputed** on load (never trusted — `hash` is Julia-version dependent). The TOML
  input reader (`io/input.jl`) mirrors only the *setup* structs (crystal + basis spec
  + symmetry), not the SALCs. Schema v3 stores `BasisSpec` **resolved and dense**
  (per-body `cutoff` matrices, `lsum` with `typemax(Int64)` = uncapped, labels) and
  `_spec_from` still expands legacy v2 scalar-`pair_cutoff` docs — keep that branch
  alive as long as v2 model files circulate. Schema v4 stores SALC members in the
  canonical duplicate-free form; `_basis_from_doc` folds pre-v4 members on load
  (`_canonicalize_members`) — keep that branch alive as long as v3 model files
  circulate, and never write a non-canonical basis. Schema v6 renamed the sector
  key `"nbody"` → `"sites"` and the schema TAGS `scefitting/sce-*` → `slce/*`;
  `_LEGACY_SCHEMA_TAGS` and the v5 sector-key fallback stay for as long as
  pre-rename files exist on disk. A persisted-document rename is NOT an API rename:
  breaking the API costs a call-site edit, breaking the format strands saved models,
  so the read path keeps compatibility even when the write path does not. The
  converse also holds and is why the document key stayed `"sectors"` when the struct
  field became `sector_rules`: a rename that buys clarity in the API buys nothing in
  a format nobody reads by hand, and costs a compatibility branch forever.
- **BasisSpec sugar resolution ↔ canonical consumers** (`slce/truncation.jl`,
  `slce/model.jl`, `io/input.jl`, **`basis/momentbasis.jl`**): the ergonomic forms (label keys, `"*"` wildcards,
  body-keyed tables, unordered `"A-B"` pair keys, specificity resolution) are expanded
  ONCE, in the `BasisSpec` keyword constructor; everything downstream —
  `SLCEBasis`'s fan-out (`_superset_cutoff` → `build_neighbor_list`,
  `cutoff` → `candidate_clusters` per-edge admission, `lsum` → `_enumerate_ls`),
  persistence, `show` — reads only the dense fields. Add a sugar form or change the
  specificity rule → update the TOML reader (`_cutoff_from_input` etc.), the BasisSpec
  docstring, and `test/unit/test_truncation.jl` together. **`_resolve_lsum` has two
  callers**: `BasisSpec` and `MomentSpec` (the two `lsum` keywords are the same
  spelling on purpose), so touching its accepted forms or error text moves both. The sector table follows the
  same discipline: `Sector` is unresolved sugar, `_resolve_sector(s)` (truncation.jl)
  produces the dense `SectorRule` rows stored on `BasisSpec.sector_rules`, and `nbody` /
  the per-body `cutoff` envelope
  are DERIVED from them — a new sector knob must be resolved there, persisted in
  `_sector_doc`/`_sector_from`, compared in `==`, and printed in `show`, or specs stop
  round-tripping. Renaming a spec keyword touches FOUR surfaces at once: the keyword
  constructor (deprecation error, `isotropy` precedent), the `[interaction]` TOML
  schema (`io/input.jl`), the persist spec doc (back-read branch on the NEW key,
  never on the version), and every BasisSpec call in test/ + examples/ + docs/.
  The SOC rule reaches the builder as `build_salc_basis(; scalar_only = !spec.soc)`
  (`model.jl`), and `scalar_only` is the literal `Lf == 0` filter all the way down
  through `coupled_bases` and `AngularMomentum.build_real_bases` — the polarity is
  in the name on purpose, because the retired `isotropy` spelling was the same
  filter under the opposite-sounding word and a blind sed across the bridge would
  invert the channel set silently. That chain is the PURE-SPIN builder, where
  `L_S ≡ Lf`. The decor engine (`_orbit_salcs_decors`) takes `soc` itself and
  screens on `L_S` (`is_soc_free`), handed to `build_real_bases(; keep)` as a path
  predicate before any tensor is built; on a mixed label the `Lf` and `L_S` screens
  accept DISJOINT path sets (`ls = (1,1,1)`, two spin slots: `L_S = 0 ⇒ Lf = 1`,
  `Lf = 0 ⇒ L_S = 1`), so never spell the decor engine's rule as `scalar_only`. `isotropy` now survives ONLY as (a) the
  `BasisSpec` / `[interaction]` deprecation errors and (b) the persist legacy
  back-read (`soc = !isotropy`); it is not a live keyword anywhere.
- **Example/tutorial hand-built ground truths ↔ canonical member semantics**
  (`examples/*.jl`, `docs/src/tutorials/*.md`, `docs/src/getting_started.md`,
  `README.md`): synthetic energies/torques written as sums over `salc.members` assume
  each physical cluster instance appears ONCE (the canonical duplicate-free form,
  persist v4). The pre-v4 examples were written against ordered-image lists (each bond
  twice) and silently encoded half the coupling after the member fold landed — and the
  first fix pass caught only `heisenberg_chain`, missing the SAME expression in
  `persist_and_input.jl`, `getting_started.md`, and `README.md` (grep the whole repo
  for the pattern, not the file you started from). Caught only by per-example
  `@assert`s (every J-recovery example must self-gate — `persist_and_input` gained
  one). Changing member multiplicity/canonicalization → re-run every example AND the
  executable tutorials (`docs/make.jl` runs them; README snippets run nowhere — review
  them by hand).
- **`coeftable` columns ↔ `SALCKey` fields** (`slce/coeftable.jl`): each result row is
  read straight off a `SALCKey` (`body` / `orbit_id` / `decors`→comma string via
  `_decor_string` / `L_S` / `Lf` / `block`) plus `jphi`; the `J` column pairs with
  `basis.salc_basis.keys` **positionally**
  (same order as the design matrix). Add or rename a `SALCKey` field → update the row
  builder, the `Tables.Schema`, and `test/unit/test_coeftable.jl`.
- **DFT training-torque target ↔ the model torque convention** (`io/dftsource.jl`): the
  training torque carried by a `TrainingDatum` is `τ_a = m_a × B_a` (`B` = constraining
  field), which must stay the *same* physical quantity, sign, and `3×n_atoms` config/atom/`xyz`
  layout as the model's `predict_torque = −e_a × ∂E/∂e_a` (the design-matrix convention) —
  **both** are the physical / Landau–Lifshitz torque `m × B_eff`. Flip one side only and the
  co-fit silently biases; flipping **both** (as done when the package moved from the `+e×∇E`
  energy-rotation-gradient to this `−e×∇E` Landau–Lifshitz convention) leaves `J` unchanged.
  The convention is enforced by code only on the field-derived path (`spin_datum` /
  `TrainingDatum(field = ...)` compute the cross product); a caller passing `torques`
  directly bypasses that funnel, so the `TrainingDatum` docstring restates it there.
  **Both sides are gated, independently, and the gates do not cancel.** Model side:
  `test_torque.jl` "predict_torque = −e × ∇E (finite differences, anisotropic)" builds its
  reference from `predict_energy` by central differences — an energy surface carries no
  torque convention, so flipping `predict_torque` (or `_design_torque`, caught by the
  `torque_weight = 1` recovery test in the same file) turns it red. Training side:
  `test_dftsource.jl` "torque convention: τ = m × B, closed form" pins a hand-written
  literal with the wrong sign named explicitly in the comment. A SIMULTANEOUS flip of both
  sides is a **gauge**, not a bug — `J` and `predict_energy` come out bit-identical — and it
  cannot pass anyway, since it would have to edit both a closed-form literal and an
  energy-derived reference. What is genuinely NOT gated here, and is the one thing to think
  about by hand, is the semantics of the external file: whether VASP's `lambda*MW_perp`
  block is `+B` or `−B`. That single bit is anchored only by SLCETools' `test/oracle/`
  (parsers vs Magesty), i.e. against a prior implementation rather than against physics.
  **DFT-code I/O is confined to `AbstractDFTSource` adapters in the SLCETools.jl package**
  (`SLCETools.VASP`), which produce `TrainingDatum`s (rotating moments / field from the `SAXIS`
  frame by `Rz(α)·Ry(β)` — spin channels only: displacements/forces stay in the lattice
  Cartesian frame); the core consumes only `TrainingDatum`/`SLCEDataset` and stays
  DFT-code-agnostic. The VASP parsers are cross-checked against Magesty in SLCETools's oracle.
  The torque sign defined here is the convention source the adapters must match. An adapter
  must NOT zero-fill a missing field block: `field = nothing` (not computed) and a present
  all-zero field (computed, vanishes) are different objects, and `torque_qualified` is
  derived from the latter — fabricated zeros re-open exactly the false-`τ = 0` hazard the
  optional channel exists to close.
- **Derivative-row bookkeeping is stored, never re-derived** (`slce/model.jl` `SLCEDataset` ↔
  `fitting/fit.jl` `_assemble_problem` ↔ `fitting/selection.jl` `cross_validate` /
  `_grouped_folds` ↔ `fitting/design.jl`): a mixed dataset's torque design `X_T` is ragged
  (rows exist only for torque-qualified configs — excluded, never zero-padded, or the
  `√(w/n_T)` whitening silently dilutes real torques), and `dataset.torque_config` (per-row
  config index, nondecreasing) is the ONE source every consumer reads: `dataset[idx]` and
  `vcat` re-label/re-offset it, `_assemble_problem` builds the grouped-CV `groups` from it
  (the old `div(n_T, n_E)` uniform-block derivation silently mislabels ragged data), and
  `cross_validate` derives its torque-presence strata from it (stratified folds + the
  `w > 0` fold cap keep every training split torque-bearing; a torque-free holdout fold
  scores energy-only, never `0 · NaN`), and `select_fit`'s `:cv` branch stratifies its
  per-unit fold deal the same way (both call `_grouped_folds(...; strata)`). Change the
  row layout in one place and all of them (plus the ragged tests in `test_dataset.jl` /
  `test_jointdata.jl`) move together. The force block follows the same contract with its
  own per-row `force_config` (sliced/re-offset by `dataset[idx]`/`vcat`, grouped by
  `_assemble_problem`) **plus two force-only twists**: `X_F` is stored COMPACT — columns
  are only `dataset.force_cols` (displacement-active SALCs; `_assemble_problem` scatters
  into full width at fit time, `residuals_force` indexes `jphi[force_cols]`) — and rows
  exist only for `_disp_referenced_atoms` (structurally zero rows excluded at build, with
  a warning if their DFT forces are nonzero). On the joint (`TrainingDatum`) path the
  torque block applies the same structural-zero exclusion via `_referenced_atoms`
  (spin-slot sites) — a disp-only ligand has exactly-zero `X_T` AND `y_T` rows; the
  pure-spin constructors deliberately keep the historical all-atom torque layout
  (oracle fit-parity: Nd2Fe14B B atoms are unreferenced), so per-config torque row
  counts differ between the two paths and consumers must read them off
  `torque_config` (`_assemble_problem` does, via its `tblock`/`fblock` searchsorted
  counts), never off `3·n_atoms`. **The selection layer is force-aware, and three of
  its rules are easy to break silently.** (1) `select_support` assembles with
  `_assemble_problem(f.dataset, w, wF)` — the SAME weights `refit` uses; drop `wF` and
  the per-group magnitudes `m_g` land in different units from the `threshold` handed to
  `refit`, so the reported alive/cost columns and the realized support disagree without
  erroring (gated by `test_jointdata.jl` asserting `refit`'s `jphi` equals the selected
  point's). (2) `_grouped_folds` deals stratum classes in DESCENDING label order with
  ONE running counter across classes; `cross_validate` packs the channels as
  `2·torque + force` so the scarcest is dealt first, and a force-free dataset degenerates
  to the historical `(true, false)` deal bit-identically. Change the order and every
  recorded seed re-deals. (3) **Force presence enters the strata only at `w_F > 0`**,
  deliberately unlike torque (grandfathered at any weight): stratifying on a zero-weight
  channel would change the fold deal — hence the score — of every force-carrying dataset
  at `force_weight = 0`, invalidating recorded results for nothing. The
  "at `w_F = 0`, identical to forces-dropped" contract is pinned in `test_jointdata.jl`.
  A channel missing from a holdout fold is OMITTED from the score, never counted as
  zero and never renormalized away — which is why the fold count is capped by each
  weighted channel's config count.
- **ASR constraint chain** (`fitting/asr.jl` builder ↔ `basis/SolidHarmonics.jl`
  `solid_harmonic_poly` ↔ `slce/model.jl` `SLCEDataset.asr`/`ASRReparam` ↔
  `fitting/fit.jl` `_assemble_problem`/`fit`/`refit` ↔ `fitting/estimators.jl`
  `solve_coefficients(...; nullspace)` ↔ `fitting/selection.jl` `gcv`/`effective_dof`):
  the constraint matrix `A` is the translation generator applied to the SAME
  homogeneous displacement polynomials the evaluator computes —
  `solid_harmonic_poly` reruns `_solid_harmonics_impl!`'s recurrences over
  coefficient dictionaries, so changing the displacement kernel's normalization or
  recurrence moves the poly function with it (the random-point agreement gate is the
  fence), and `A` carries the evaluator's `(4π)^{n_spin/2}` column scale (a future
  `disp_scale ≠ 1` must be folded in per degree — the builder asserts against it
  today). `Z` (orthonormal per connected component, identity on pure-spin columns,
  SVD rank with the forbidden-band refusal) is built ONCE at dataset construction
  and stored on `dataset.asr` (carried by slicing/`vcat` like `force_cols`;
  `nothing` on pure-spin bases — the bitwise-identity fast path is a gate);
  `_assemble_problem` only APPLIES it, and every consumer that re-derives a solve
  or a hat matrix — `refit`, `gcv`/`effective_dof` (Cholesky congruence of the
  dense `Z'DZ`, never the diagonal shortcut), a future constrained λ path — must
  apply the same reparameterization or silently diverge from `fit`. β (never γ/Z)
  is the public coefficient space: `SLCEFit.jphi`, the `refit` support rule,
  persistence, and fixtures are all β-indexed (β is factorization-gauge-invariant;
  Z is not — never persist or pin Z/γ), and **`refit` must re-derive the null
  space on its support columns** (`A[:, support]`) — a support splitting a
  constraint-coupled column set changes the feasible space (survivors may be
  structurally zeroed, warned). **γ → β is one function, `_lift_gamma` (asr.jl), with
  two readers** (`fit`'s basis-level lift and `refit`'s per-support sub-stage): the
  plain product `beta_p + Z·γ` leaves ~1e-15 on forbidden directions because a dead row
  of `Z` is NUMERICALLY zero (an SVD null row), not structurally zero, and that junk is
  load-bearing — `select_support`'s alive rule and SLCEMonteCarlo's term prune
  (`hamiltonian.jl`, `t.coef != 0.0`) are both EXACT tests, so it charges MC cost and
  buys site programs for a direction no feasible model can carry. Never "simplify" a
  lift back to the raw product, and never soften either exact test to a tolerance —
  the exactness of the consumers is what obliges the producer to emit exact zeros.
  The structurally-zeroed test is **one constant**,
  `_ASR_DEAD_ROW` (`‖Z[j, :]‖ < 1e-12`, absolute because `Z` is orthonormal), with four
  readers: `build_asr`'s basis-level warning, `refit`'s movable-column and
  split-coupled-set rules, and `group_costs`' structural discount.
  **The freeze must reach EVERY path that builds a reparameterization**, and two reviews
  found four that it did not: the pure-spin `SLCEDataset` constructors (they hard-coded
  `asr = nothing`, so a pure-spin cell's frozen columns were fitted freely and came back at
  ~1e-18 — not the exact zero the alive rule and the MC prune test), `_fit_stage`
  (`sector_columns` knows only key content, so a mask re-freed a frozen column and the
  unbounded solve put 3.7e14 into `jphi` and 3.0e14 into `force_constants` while
  `asr_residual` still read 0.0), `_is_staged` (object identity against `dataset.asr`, so
  the freeze-only rep the `asr = false` path builds read as "staged" — `SLCEFit.staged` is
  now RECORDED, never inferred), and `select_fit` (gated on `rep === nothing`, which would
  refuse every WS-tied crystal — it gates on constraint ROWS now). Classification must also
  stay off the paths that never needed it: it ran before `build_asr`'s early exits and its
  AllImages repeated-atom refusal broke `asr_residual` on a legal pure-spin model and made
  an AllImages joint dataset unfittable by any route, so that refusal is a dedicated
  `UnclassifiableBasis` (caught in `build_asr` → nothing frozen, said once) and
  `asr_residual` builds only `_asr_matrix`. **The `asr = false` escape hatch that fix
  restored is withdrawn again (2026-08-24), on a different argument:** it kept the route
  open but never restored identifiability — the self-image columns are redundant on the
  reference cell, so the fit returned an arbitrary representative. `SLCEDataset` now
  refuses at the door; `build_asr` keeps its catch because it is also an ANALYSIS entry
  point for hand-built / tiling-template models.
  **Why the columns are kept is the SUPERCELL, not the strain.** Tiling maps the tied images
  onto distinct atoms, which holds for every frozen column; `affine_energy` reaches only some
  (measured: 0 of 1 on bcc harmonic, 0 of 4 pure-spin, 6 of 8 at degree 3, 4 of 10 on
  spin × degree-1) and *structurally* none of the pure-spin ones — no displacement slot for a
  strain to act on. Never restore the `dJ/dr` reading.
  **A BOUNDARY TIE HAS TWO ALGEBRAIC FACES and only the first is a null column.** (a) the
  point group permutes the tied images, so they share ONE orbit whose sum weights them
  equally and the odd content cancels — the column is identically zero, equal weighting here
  is *symmetry* rather than a gauge, and the surviving even content is a determined coupling
  that stays. (b) in low symmetry no operation relates them, so they sit in DIFFERENT orbits
  with independent couplings: every column is nonzero and the two orbits' columns are not
  equal to each other either (a member's tensors carry its own bond geometry). What
  collapses is the SPAN — every member of either orbit reads its sites' displacements off
  the same reference-cell atoms — so the orbits span one function space and only the TOTAL
  is determined: a null COMBINATION, invisible to any per-column test. **Never restate this
  as "the two orbits are the same function"**; that wording was wrong and stood in six
  places until a review caught it (a column-equality claim is falsified by evaluating any
  two of them). The undetermined fraction is `length(frozen) − rank(S[:, frozen])`, which
  is `1 − 1/k` for a fully separated `k`-fold tie and NOT "half" in general — under partial
  fusion the two faces coexist in one basis (four-fold tie under `{E, m_y}`: 8 vanishing +
  10 undetermined, true flat dimension 13). Face (b) is the dangerous one and was missed until measured: on a
  P1 cell with the pair on the WS face, a force co-fit reached `rmse_E = 4.2e-16` with a clean
  `asr_residual` and `D(0)` exact to 1.2e-15 while `D(q)` was **52 % wrong**, nine flat
  directions all ASR-feasible, `fit` silent. So **`_has_boundary_tie` must scan ACROSS SALCs**
  (is any atom multiset reached by two orbits?) — the within-SALC scan is the hole, and under
  `MinimumImage` the cross-orbit case can only come from a tie — and that necessity is a
  PROOF, not a measurement: every edge sits at its atom pair's minimum image, so fixing
  site 1 at the origin fixes every other site's image uniquely while those minimum images
  are unique, giving one cluster per atom multiset up to translation.
  **At `N ≥ 3` the freeze covers CONGRUENT siblings only, and that is a scope statement to
  keep saying out loud.** The compact-cluster criterion admits a cluster only when all
  `C(N,2)` edges are simultaneously minimum-image; congruent siblings pass or fail
  together and the freeze sees them, but a sibling reached through an image that puts one
  of its OTHER edges on a longer shell is rejected there, and the tie then leaves no trace
  (`_has_boundary_tie` false, nothing frozen). That is aliasing, not indeterminacy — the
  admitted cluster absorbs it exactly, measured on a P1 cell with a tied `(1,2)` edge where
  a 3-body degree-`(1,1,1)` sector spans the FULL trilinear space (rank 27 = 3³, span
  identical to an independently built basis of all 27 monomials, `identifiability.nullity
  = 0`). The cost is the `R` LABEL: `force_constants` attributes the coupling to the
  admitted geometry, which matters once cubic constants are exported and read as
  `Φ(R₁, R₂)`. Do not "fix" this by loosening the compact criterion. **There is no justified split of the determined sum** — the images share a
  phase only at `q = 0`, so equal division is an interpolation ansatz, which is why route A's
  freeze-at-zero for face (a) is legitimate and a face-(b) split would not be — so the whole
  interaction is dropped: every column of every orbit sharing an atom multiset is frozen.
  That discards determined content ON PURPOSE (18 columns = 9 determined sums + 9
  undetermined differences on the fixture, `r2_energy ≈ 0.70` on data containing the shell),
  and the nonzero residual is the intended loud failure. Standard cells are untouched and the
  reason is symmetry: with real space groups the null-column count already equals the full
  structural nullity on bcc Fe, B2 FeRh, hcp Co, wurtzite GaN, rocksalt MnO. `residual_flat`
  reports any flat direction the orbit granularity leaves behind rather than assuming none.
  Gate (G) in `test_resolvability.jl`; the worked example is in `theory/resolvability.md`.
  **Two different reasons produce an all-zero `Z` row and they must not be conflated.**
  A *frozen* column (`unresolvable_columns`, `basis/resolvability.jl`) cannot be determined
  on this cell — no fit can reach it, the remedy is a different reference cell, and
  `build_asr` excludes it from `free` under `asr = false` too. A *structurally zeroed*
  free column is excluded by the sum rule — the remedy is a partner term. `build_asr`
  reports each with its own message and scans only `free` for the second; the counts that
  gate on translation invariance (`rank == ndisp`) count free displacement columns only, so
  a basis whose surviving columns admit no invariant content says so loudly instead of
  throwing "the symbolic expansion is broken". Classification is **structural, never
  sampled**: the undifferentiated twin of `_asr_matrix`'s monomial expansion, judged per
  column against its own gross accumulation (a global cut is blind to the all-columns-cancel
  case, which is exactly the bcc harmonic one). Its group-resolved
  form is `group_freedom` (`s_g = ‖Z[g, :]‖_F²`, `Σ_g s_g ≡ q`, gauge-invariant).
  **`_CANCELLATION_RTOL` (`basis/resolvability.jl`) is one constant with two readers, and
  the reading is "did this cancel?", never "is this small?"**: `unresolvable_columns` judges
  a COLUMN of the undifferentiated expansion, `_prune_residue!` (`asr.jl`) judges each ENTRY
  of the differentiated one against the gross mass `_asr_expansion` reports beside it. The
  differentiated expansion cancels wherever the undifferentiated one does, so a tied cell can
  produce an `A` that is residue end to end — and a cut against `A`'s own maximum (per row or
  global) cannot see it: every row then looks full-strength, the row normalization promotes
  BLAS rounding to unit constraints, and `asr_residual` read 0.36 on a bcc spin × displacement
  basis whose every column is identically zero (`max|A| = 3.6e-15` against a gross 82), i.e.
  the public verifier the force-constant / dynamical-matrix / strain paths gate on refused a
  legal model over noise. The prune is why `build_asr`'s broken-expansion refusal reads
  `visited` (any positive gross) rather than the rank alone: "deposited nothing" is a broken
  expansion, "deposited and cancelled" is a basis carrying difference content only, and after
  the prune the second one reaches `rank == 0` and must NOT be refused. Gates: the pure
  function on hand-written `(A, G)` in `test_asr.jl`, and in `test_resolvability.jl` gate (E)
  — `rep.rank` against the rank of the production gradient kernel's translation image over the
  FREE columns (over ALL columns that same count reads 8 and 5 on the two all-frozen fixtures,
  which is the identical blindness one layer out).
  **The ASR's granularity is NOT the group** (measured 2026-07-28, five fixtures,
  `G` 2→20): no displacement-touched group has a feasible subspace alone, the true
  atoms are matroid circuits of 2–3 columns that cross groups, and component closure
  collapses `G` to two clusters regardless of `G` — so closure is rejected and the
  cost axis is real only on the ASR-untouched pure-spin groups. Nothing is persisted:
  `asr_residual(model)`
  recomputes `‖Aβ‖/(‖A‖‖β‖)` from the basis (fingerprint precedent — recompute,
  never trust), and the physical consumers (M4 force constants / dynamical matrix,
  MC joint ingest) gate on it; hand-built violating models are legal (the gate-(k)
  violation demo requires them). The `‖Aβ‖` residual is architecturally ~eps under
  the reparameterization and passes even if `A` is WRONG — the real acceptance
  gates are the symbolic-vs-numerical rank equality (through `accumulate_grad!`'s
  `Gu` column sum) and finite-`t` translation invariance + per-config `Σf = 0`
  through the production evaluator, run after `fit` AND after `refit`
  (`test/unit/test_asr.jl`). `ASRReparam.beta_p` is the affine slot the staged fit
  fills (frozen-stage offsets); the staged-fit rule is "each stage fitted under its
  own ASR keeps the next stage's constraint homogeneous".
  **Who is allowed to WARN**: `build_asr(basis; warn = true)` owns the two truncation
  diagnostics; `asr_residual` calls it with `warn = false` because it re-derives, and any
  new re-derivation path must do the same or one true statement about the truncation gets
  reprinted once per derived output (measured: 38 lines on one docs page before this). Both
  diagnostics also carry `maxlog = 1` so a `StrainedModels` grid of structurally identical
  bases says it once. The REFUSALS (broken symbolic expansion, forbidden band) ignore
  `warn` — they say the answer would be wrong, not that the truncation is narrow. The
  dead-column advice is per channel and easy to state backwards: a spin-FREE displacement
  column wants more pair orbits (widen the cutoff), a spin-DRESSED one wants a term
  dressing the same spin invariant differently (a displaced ligand — `sites = 2` gives 4
  dead columns per pair orbit, `sites = 2:3` gives none), and a symmetry-forbidden channel
  (ε-linear pair coupling on a bond whose midpoint is an inversion centre) is never
  revived by any truncation. The doc fixtures on `guide/{introspection,lattice_dynamics,
  strain}.md` are built to that rule, so a change here shows up as new warnings in the
  published pages.
- **Periodic evaluator ↔ affine evaluator** (`fitting/fit.jl` ↔ `slce/affine.jl`):
  `predict_energy` resolves a site's displacement as `u[:, atom]`, so it can express
  ONLY cell-periodic fields — translation is the one affine field that is, which is why
  the ASR is testable through the ordinary predictors and rotation/strain are not.
  `affine_energy` is the missing path (same `SALC.members` sum, field resolved at each
  member site's own position **with its image shift**), and it is a RE-INDEXING of the
  evaluator, never a second one: the `M = 0` bit-identity gate
  (`affine_energy(m, e, zeros(3,3); base = u) === predict_energy(m, e, u)`) is what
  enforces that, so both kernels must keep sharing `_eval_term_mixed` and its loop
  order. Rotational invariance is **measured, never imposed** — translation is the only
  affine condition constrained (a continuous rigid rotation is not a space-group
  operation, so no basis projection can remove it). Three traps the gates pin
  (`test/unit/test_rotation.jl`): the LINEARIZED rotation test is blind, not weak (it
  misses exactly `½ω²F·(W²d)`, and returns exact zero on radial forces at every ω); a
  truncated model's residual VANISHES WITH ω rather than being zero, so the decay rate
  is the signal and a single-ω threshold conflates truncation with violation; and
  on-site displacement content carries a home-**image gauge** (`F·M(L)`, at `O(ω²)` —
  the same order as the content) that periodic training data cannot fix, so any future
  rotational constraint matrix is NOT a function of the model alone. In a SOC sector
  zero is the wrong expectation for `rotational_residual`; the sector-independent
  statement is `rotation_transfer_residual` (`𝓡_U E = −𝓡_S E`, gate (r)).
- **Staging axis ↔ truncation axis** (`fitting/staged.jl` ↔ `basis/salcbasis.jl` ↔
  `slce/truncation.jl`): `Sector(soc = …)` defines the model's SUPPORT (columns that
  are never built), `sector_mask`/`frozen` define what a fit STAGE moves (columns held
  at frozen values). They are never merged, and they share exactly one predicate —
  `is_soc_free` (`basis/salc.jl`), used by the basis builder's `soc || is_soc_free(L_S)`
  filter and by `sector_columns(basis, :soc_free)` — because the failure mode is silent
  drift between "what a SOC-less calculation can express" and "what a SOC-less stage
  fits" (design record §13 risk 4). The gate is set equality between the masked keys
  and a `soc = false` rebuild's keys (`test/unit/test_staged.jl`). A stage is realized
  as ONE affine `ASRReparam` (zero `Z` rows on frozen columns, `beta_p` = frozen values
  + particular solution), so the assembly/solve path is the ASR path verbatim; the
  identity `size(Z, 2) == p − rank` therefore holds only for a basis-level
  reparameterization, NOT for a stage. `SLCEFit.reparam` is what every re-derivation
  (`refit`/`gcv`/`effective_dof`/`identifiability`/`dof`) must read — reading
  `dataset.asr` instead silently re-assembles a staged fit as an unstaged one.
  **Asking "is this fit staged?" is `_is_staged(f)` (`fitting/fit.jl`), never
  `f.reparam !== nothing`** — `fit` stores the DATASET's reparameterization for an
  ordinary constrained fit and only substitutes a freshly built one for a stage, so the
  question is object identity against `f.dataset.asr` and the three states are
  unconstrained (`nothing`, including a deliberate `asr = false` on a joint basis) /
  plain ASR / staged. The `!== nothing` shortcut reads true for every plain joint fit;
  that is how `select_support` came to tell unstaged callers their fit was staged, and
  the regression is pinned by the error TEXT in `test/unit/test_asr.jl` (a type-only
  `@test_throws ArgumentError` cannot see it — the fixture there was a force co-fit
  hitting a different refusal entirely). Whether
  the frozen part counts as ASR-satisfying is the RELATIVE `asr_residual` measure, not
  `A·β == 0`: a fitted stage leaves ~1e-16, and treating that as a violation sends the
  next stage down the affine path where a roundoff-sized right-hand side is generically
  infeasible (the bug the chain gate caught).
- **Sunny export conversion ↔ the energy reconstruction** (`slce/bilinear.jl` — matrices /
  extraction / gate — plus `interop/sunny.jl` — primitive unfold — and
  `ext/SLCESunnyExt.jl`): `_l1_pair_matrix` / `_l2_onsite_matrix` must satisfy
  `eₐ'·M·e_b = Σ folded·Z·Z` (the gate is the `Z₁`/`Z₂` contraction test; the tesseral
  constants `N1`/`A2`/`B2` are defined once, in `basis/Harmonics.jl`); the per-bond
  matrix is `jϕ·(4π)^(N/2)·M` and the two directed members `(a,b,R)`/`(b,a,−R)` fold into
  one matrix on the canonical `a≤b` bond (reverse transposed). The whole chain is checked
  **without Sunny** by `_reconstruct_energy ≈ predict_energy − j0`; the extension then
  rescales by the effective spin (`scaling = :moment` sets `J = M/(SₐS_b)`; `:coupling`
  folds `S_eff` into the couplings at a placeholder `Moment`) so the `Sunny.System` energy
  matches. Only `ls=[1,1]`/`ls=[2]` are
  representable — every other channel must be **reported as skipped**, never silently
  dropped. Change a harmonic normalization or the `(4π)^(N/2)` scale → both the matrix
  formulas and the energy gate move together.
- **Force constants ↔ the displacement kernel ↔ the ASR** (`slce/forceconstants.jl`,
  `test/unit/test_forceconstants.jl`): the constants are EXACT derivatives, obtained
  by reading monomial coefficients off `SolidHarmonics.solid_harmonic_poly` — the same
  function `fitting/asr.jl` builds `A` from, so a change to the displacement kernel's
  normalization or recurrence moves the force constants, the ASR matrix, and the
  evaluator together. Only terms whose displacement degrees sum to `order` contribute
  (each site factor is homogeneous of degree `2k + l`). Indexing is the
  lattice-dynamics convention `Φ[(a,0),(b,R)]`, one entry per ORDERED tuple anchored on
  the home cell — NOT the SALC-member convention, where each undirected instance
  appears once; the reverse ordering is a separate key equal to the transpose. The
  acceptance gate is the Γ-restricted sum `Σ_R Φ(R)` against a finite-difference
  Hessian of the production evaluator (that is what pins the folding convention), and
  the ASR shows up here PHYSICALLY: an ASR-fitted model has exactly three zero
  eigenvalues of `D(0)`, an unconstrained one has none. `dynamical_matrix` takes `q` in
  FRACTIONAL reciprocal coordinates (the package's `reciprocal` carries no 2π, so the
  phase is written `exp(2πi q·R)` with an integer `R`) and lays rows out atom-major /
  Cartesian-minor like every other derivative block.
  **The unresolvable-column freeze is silent in the fit and LOUD here**, and that is what
  `_warn_unresolvable` (this file, beside `_warn_spin_blind`) exists to say — one
  definition, five readers: `force_constants`, `strain_derivatives`,
  `exchange_strain_derivatives` (its own path — `magnetoelastic_constants` inherits it
  through `strain_derivatives`), `magnon_phonon_vertices`, and `decorated_terms` (the MC
  hand-off, where the supercell resolves the tie and the missing channel becomes real
  physics). These readouts
  differentiate the individual cluster MEMBERS, where a Wigner–Seitz tie does not cancel,
  so a frozen column shows up as a missing channel when its coefficient is zero and as
  unconstrained `Φ`/`q ≠ 0` content when a caller supplies one (legal — hand-built and
  externally fixed models are). **The tie is invisible at `q = 0`**: `Σ_R Φ(R)` is the
  Hessian of exactly the energy the cell can express, so a model passes every Γ-point gate
  (including the acceptance gate above) and still carries an unconstrained dispersion —
  measured on the z-doubled bcc fixture, `D(0)` equal to 1e-12 and `D(0.3, 0.1, 0.25)`
  different, from a single frozen coefficient. Gate (F) in `test_resolvability.jl`; the
  advice is a different reference CELL, never a wider cutoff. **Both messages carry
  `maxlog = 1` AND a per-read-out `_id`**, and the `_id` is load-bearing: Julia keys `maxlog`
  by the log STATEMENT, so one shared `@warn` at `maxlog = 1` speaks for whichever
  deliverable ran first and silences the other five — measured on a B2 FeRh joint basis,
  `strain_derivatives` warned while `magnetoelastic_constants`, `magnon_phonon_vertices`,
  `decorated_terms` and `force_constants` all returned frozen-channel results in silence
  (`magnetoelastic.jl` records the same trap for its own residual warning). The call sites are
  affordable only because `unresolvable_columns` short-circuits on a structural pre-check
  (`_has_boundary_tie`): a tie is NECESSARY (rows are keyed by per-atom content, so content
  over different atom multisets never meets, and a lone member's contribution is nonzero by
  construction), so an untied cell never pays the expansion. The pre-check must scan ACROSS
  SALCs as well as within one — see the two faces of a tie in the ASR chain above; the
  within-SALC-only version let face (b) through silently. Gate (A) checks the equivalence by
  running BOTH `unresolvable_columns` and the unconditional `_unresolvable_expanded` against
  the evaluator; gate (G) pins face (b). Break the necessity claim and the freeze silently
  stops working.
  **The magnetic symmetry of the result is a consequence of the projection, not an
  input, and it is the package's headline physics claim** — say it wherever force
  constants are described. Force constants are time-reversal EVEN, so an antiunitary
  `g·T` constrains `Φ` through its rotation part like a unitary element: the correct
  group is `D ∪ D′`, and the joint path lands on it because the SALCs carry the
  paramagnetic grey group `G × {1, T}` and fixing `spins` reduces that to the magnetic
  stabilizer. Both neighbours are wrong and silent — a lattice-only basis imposes the
  paramagnetic group (too large), sublattice-relabelled species impose `D` alone (too
  small, and this is what ALAMODE's `MAGMOM` does: `alm/system.cpp` splits atom types
  by `(element, m_z)` and feeds that to spglib). The three counts 7 / 12 / 16 on the
  stripe-AFM fixture are the ledger, gated in `test_forceconstants.jl` against a
  hand-assembled P4/mmm group (no Spglib in the core env — `_basis_with_sg` calls the
  same three builders `SLCEBasis` does). `_warn_spin_blind` is the other half: a basis
  with spin content and displacement terms at `order`, but no term carrying both,
  yields `Φ` identical for every magnetic state. It reads the BASIS, never `jphi`
  (`spin_multipole_terms` precedent — a coefficient fitted to zero is one fit's property,
  `refit` moves it), and its `maxlog = 1` is why the gate checks the SILENT cases
  first: a silence assertion after the warning has fired asserts nothing.
- **Strain response ↔ the force-constant machinery ↔ the ASR**
  (`slce/strain.jl`, `test/unit/test_strain.jl`): `strain_derivatives` is the FOURTH
  consumer of `SolidHarmonics.solid_harmonic_poly` and the second consumer of the affine
  path — a homogeneous strain is not cell-periodic, so `predict_energy` cannot reach it.
  It is one chain rule on top of the force constants, not a second derivative engine:
  `∂ⁿE/∂ε^n = Σ_{s₁…s_n} [∂ⁿE/∂u_{s₁α₁}⋯] d_{s₁β₁}⋯` over MEMBER SITES with `d_s` that
  site's own position (image shift included), so `_accumulate_strain!` calls
  `_fill_fcs_tensor!` verbatim and only the final contraction is new. **Do not reroute it
  through `force_constants`' output**: the lattice-dynamics key is anchored on the home
  cell, which throws away precisely the absolute positions the affine field needs.
  Two rules are shared rather than restated — `_resolve_spins` (the spin-free entry
  rule) and `_spin_blind_at_order` (the predicate; the two callers write their own
  messages because the advice is opposite: `degree = 1` is what magnetoelasticity NEEDS
  and what harmonic constants must not stop at). The acceptance gate is the Taylor
  identity against `affine_energy` at FINITE strain — exact because a degree-`n` term is
  a degree-`n` polynomial in `ε`, and the two paths are independent implementations
  (monomial coefficients vs the production `_eval_term_mixed`). **The ASR is a
  precondition, enforced as a THROW at `_STRAIN_ASR_RTOL = 1e-12`** (tighter than the
  1e-10 elsewhere, because the strain path weights `Σ_i ∇_i E` by `|R_i|`): an origin
  shift adds the uniform translation `ε·t`, so on a violating model the quantity is
  undefined, not inaccurate. **But the ASR is sufficient only at order 1**: it is an
  identity in the ATOM variables, i.e. on cell-periodic fields, and at `ε = 0` the affine
  field is zero (periodic) — one order out it is not, and the per-SITE identity that would
  be needed is not implied. The gap is the SAME home-image gauge the rotation diagnostic
  found, now on a physical output (one dimer chain, two descriptions: 1.4e-13 vs a factor
  8; bcc-like ~25). So `order ≥ 2` **measures** origin independence on every call
  (`_STRAIN_ORIGIN_RTOL`, one probe shift — the dependence is affine in the shift, so one
  generic probe suffices) and refuses a disagreement, naming the crystal DESCRIPTION as
  the fix. Never soften that to a warning and never make `check_origin = false` a default:
  the failure is O(1), silent, and produces a publishable-looking elastic constant.
  `symmetrize = false` is the general-affine-map derivative;
  the antisymmetric part it keeps is `rotation_transfer_residual`'s content
  (`𝓡_U E = −𝓡_S E`), never noise to discard silently. Measure = Biot / Seth–Hill
  `m = 1` (design record §9e) — `order = 1` is measure-independent, `order = 2` is not,
  so any second-order output must record the measure AND the spin state.
- **The magnetoelastic tier ↔ the ε-LINEAR strain response ↔ the bilinear conversion**
  (`slce/magnetoelastic.jl`, `test/unit/test_magnetoelastic.jl`): both deliverables ride on
  `strain_derivatives(order = 1)` and **must stay there**. The tier is first-order for two
  independent reasons — measure independence (§13 risk 2: second-order elastic constants
  agree across the Seth–Hill family only at a stress-free reference, and a spin–lattice
  cell is stress-free for at most one magnetic state) and origin independence (the ASR
  covers `ε = 0` alone). Adding an `order = 2` arm to either one silently forfeits both.
  **The B₁/B₂ convention is pinned here and nowhere else** — `E_me/V = B₁ Σ_i ε_ii(α_i² −
  1/3) + 2B₂ Σ_{i<j} ε_ij α_iα_j`, tensor shear, that range, that sign, energy DENSITY —
  because gate (e2) fixes only the block's span ("C2 is the fit gauge"). Gate (u) compares
  it against a hand-written closed form through `evaluate_salc`, never against
  `strain_derivatives`' own output; if the convention ever moves, that test is the one
  place to change, and the docstring, the guide and `_me_form` in the test must move with
  it. `ion = :clamped` is a FIELD of the returned tuple, not prose — do not "simplify" the
  return value to `(B1, B2)`. The constants come from a PROJECTION over 19 directions with
  `residual` reported, not from two evaluations: a model outside the two-constant form must
  say so rather than return numbers. `exchange_strain_derivatives` shares the tesseral →
  Cartesian conversion with `bilinear_terms` (`_l1_pair_matrix` / `_l2_onsite_matrix`) but
  NOT its `(4π)^(body/2)` scale — the joint rule is `count(has_spin, decors)/2`, and the
  pure-spin shortcut is exactly the §7 consumer-scale trap. Its gate is the reconstruction
  identity against `strain_derivatives(symmetrize = false)`, which is why the unsymmetrized
  `(γ, δ)` convention is contract and not an accident. The per-bond split is
  origin-dependent whenever a bond's own displacement content is not relative — measured on
  every call in the same idiom, never assumed from the model-wide ASR.
- **`decorated_terms`' index map ↔ `keep_zero` ↔ an in-place coefficient rewrite**
  (`slce/introspect.jl`, `test/unit/test_introspect.jl`): both introspection surfaces prune
  exactly-zero coefficients by default, which makes the term list — and the index a
  consumer addresses it by — depend on the coefficient VALUES. Fine for a reader; wrong for
  a rewriter. **Anything that will later call `SLCEMonteCarlo.set_coefficients!` must build
  its term list with `keep_zero = true`**, so the map is a property of `salc_basis` (the
  object `StrainedModels` asserts identical across a grid) rather than of one fit. The
  failure is silent in the worst way: two models with the same NUMBER of exact zeros at
  different keys give equal-length lists with shifted maps, so every length check passes and
  each coefficient lands on a neighbouring cluster. **Do not change the default to `true`**
  — that would move every existing consumer's byte comparisons — and do not "simplify" the
  two surfaces to share one pruning rule with the MC's `keep_zero_terms`: they are different
  halves of the same hazard (upstream emission vs downstream support freezing), and the MC
  flag cannot resurrect a term upstream never emitted.
- **Volume grids ↔ re-expansion closure ↔ the SALC gauge** (`slce/strainedmodels.jl`,
  `test/unit/test_strainedmodels.jl`): the grid invariant is a SIMILARITY (`A_i = s_i·A₀`,
  fractional basis untouched), which is what licenses `_scaled_basis`' surgery — carrying
  the space group and the SALC basis over unchanged and scaling only the cell and the two
  cutoff surfaces. **Never scale `disp_scale` with it**: the whole grid's coefficients live
  in one frozen normalization. Two invariants beyond key equality are load-bearing and both
  are silent when broken: the **home-image condition** (same atoms at the same integer
  shifts) and the **SALC gauge** (the folded tensors themselves — the projector fixes each
  invariant only up to a sign, and a flipped one is interpolated against the others with
  every key matching and every per-point fit perfect). **A grid's basis must be closed under
  re-expansion** — `degree = 2` needs `degree = 1`, spin×`degree = 1` needs a pure-spin
  sector — because a grid point IS the reference re-expanded, and the shift is
  lower-triangular in degree; a fixture that forgets this fails the acceptance gate by
  percent, which is how it was found. **`dE/dη = s·dE/ds`**: η is always measured from the
  reference the derivative is taken at, matching `strain_derivatives`, so
  `grid_strain_derivative` carries the factor `s` — dropping it agrees at `s = 1` and is
  wrong by the strain everywhere else.
- **Magnon–phonon vertices ↔ `grad_Zlm` ↔ the torque path** (`slce/magnonphonon.jl`,
  `test/unit/test_magnonphonon.jl`): `magnon_phonon_vertices` is `_fill_fcs_tensor!` with
  one axis moved from "evaluate" to "differentiate" — the displacement axis still reads
  `solid_harmonic_poly`, the differentiated spin axis reads `Harmonics.grad_Zlm`, which is
  the SAME tangential gradient `_design_torque` is built from. **Do not replace it with a
  raw `∂Z/∂e`**: the projection is what makes `V·ê_b ≡ 0` and therefore what makes the
  Cartesian return value free of a local-frame convention. The `(a, b, R)` key is ORDERED
  (`a` displaced, `b` magnetic) — never canonicalize it the way `bilinear_terms` does its
  bonds. Contributing content is exactly one degree-1 displacement factor plus a spin
  factor, so a `degree = 2` magnetoelastic sector yields an EMPTY set, and that is
  `_warn_spin_blind`'s trap from the other side. No ASR precondition (no absolute positions
  enter), but the translation sum rule `Σ_{a,R} V = 0` is gated — if it starts failing, the
  bug is in the model or the ASR, not here.
- **Re-expansion ↔ the same displacement polynomial** (`slce/effective.jl`,
  `test/unit/test_effective.jl`): `effective_model(model; u0)` is the THIRD consumer of
  `SolidHarmonics.solid_harmonic_poly`, after the ASR builder and the force constants —
  it shifts each monomial binomially instead of differentiating it, so a change to the
  displacement kernel's normalization or recurrence moves all three together. Two
  things to keep straight. (1) **The output is not a `SLCEModel` and must never be made
  into one**: the displaced structure's symmetry is the stabilizer of `u0`, generally a
  proper subgroup, so reference SALCs cannot span the result; `EffectiveModel` carries
  no `SALCKey`s, is not fittable, and is not persisted. (2) **Monomial variables are
  per ATOM, not per slot** — displacements are cell-periodic, so an `AllImages`
  self-bond puts two displacement slots on one variable and their exponents must add.
  Keying by slot yields the SAME energy (evaluation multiplies either way) but a
  non-canonical term list: like terms stop merging, coefficients arrive split across
  duplicates, exact cancellations do not happen, and `atol` prunes halves of a term.
  Do not let a comment claim more than that. The acceptance gate is the identity
  `E_eff(δu) ≡ E(u0 + δu)` against the production evaluator, measured against the
  sample's ENERGY SCALE and never pointwise-relative — a random spin configuration can
  sit at a zero crossing of the energy, where the pointwise ratio reaches 7e-13 while
  the absolute error stays at 6e-14.
- **Introspection surfaces ↔ the consumer scale rule ↔ `restrict`**
  (`slce/introspect.jl`, `test/unit/test_introspect.jl`): the scale a consumer applies
  is `(4π)^{n_spin_slots/2}` — one `√(4π)` per SPIN slot, defined ONCE in
  `_slot_scale` and shipped as `DecoratedTerm.scale`. It is NOT `(4π)^{body/2}`: the
  two agree exactly when the SPIN-slot count equals the body order (every site carrying a
  spin factor, alone or beside a displacement one), so
  a force-constant term (sites with no spin factor at all) is where the pure-spin-era
  shortcut invents a factor out of nothing — that is the case the gate must contain, and
  a fixture whose sites all happen to carry spin makes the whole check vacuous (the
  first docs fixture did exactly that). `decorated_terms` is the general surface;
  `spin_multipole_terms` is its frozen p = 0 predecessor and REFUSES any displacement model,
  triggered on `_basis_has_disp` (the spec, not the surviving coefficients — a
  displacement sector whose couplings all fitted to zero is still a p ≥ 1 model), with
  the message naming both hatches. `restrict(model, :spin)` filters the pure-spin SALCs
  and rebuilds the spec through `_spin_spec` (pmax zeroed, sectors reduced to their
  degree-0 row): the spec has to be honest or `_basis_has_disp` re-refuses the very
  model `restrict` exists to produce, and persistence writes a spec that will not
  reload. Its gate is bitwise equality with the joint model at `u = 0`, and the
  `restrict ≠ refit` warning (docstring + `guide/introspection.md`, design record §13
  risk 5) is mandatory documentation, not a nicety.
  **`EffectiveTerm` (`slce/effective.jl`) is the third public term view and takes the
  OPPOSITE convention** — which is why its field is named `scaled_coef`, not `coef`,
  so the mistake is a `has no field` error rather than a silent over-count:
  `scaled_coef` has the `(4π)^{n_spin_slots/2}` scale, the SALC's
  `folded` weight, and the shifted polynomial coefficient ALREADY folded in, because
  one term merges contributions from many SALCs and there is no raw `jϕ` to hand back.
  A consumer migrating from `decorated_terms` and re-applying `_slot_scale`
  double-counts `4π` per spin slot. Say so in any new term-view docstring; the scale
  expression itself is also written out longhand in five kernels
  (`salc.jl` ×2, `asr.jl`, `forceconstants.jl`, `effective.jl`) rather than routed
  through `_slot_scale`, and is equivalent only because `SiteDecor` admits at most one
  SPIN factor per site.
- **Fitted-model introspection ↔ the per-term scale convention** (`slce/introspect.jl`,
  `test/unit/test_introspect.jl`): `spin_multipole_terms` is the **public, stable** view downstream
  packages (the `SLCETools.jl` mean-field samplers) read instead of `model.basis.salc_basis.salcs` /
  `SALCMember` / `SALCTerm`. It returns the **raw** fitted `jϕ` as `coef` and leaves the per-N
  scale `(4π)^(body/2)` to the consumer — the scale lives in exactly one place (the
  reconstruction gate `_energy_from_terms`), so do **not** also apply it inside
  `spin_multipole_terms`. `bilinear_terms` is a thin public wrapper of the general
  `_bilinear_terms` extraction (in `slce/bilinear.jl`), so its numerics move with the
  Sunny coupled-site above.
  Add or rename a `SpinMultipoleTerm` field → update the gate and the `SLCETools.jl` consumers
  (`mfa/bridge.jl` — renamed from `sce_bridge.jl`; grep the package, not this name alone).
- `solve_coefficients(est, X, y; row_groups)` receives a **column-centered** `X` (⇒ the
  solver adds no intercept; `j0` is recovered analytically in `fit`). Every estimator —
  in-tree or in an extension — must honor this. `row_groups` (optional) labels rows from the
  same physical sample (in a co-fit, a configuration's energy row and its
  torque-component rows share a label); a resampling estimator (CV-based `ElasticNet` /
  `Lasso` / `AdaptiveLasso` in `ext/SLCEGLMNetExt.jl`) must keep same-label rows
  in the same fold so CV does not leak within-configuration structure. The analytic / adapter
  estimators (`OLS` / `Ridge` / `AdaptiveRidge` / `FixedCoefficients`) ignore it. The GLMNet
  solve uses `intercept = false` + column `standardize` and selects λ by configuration-
  grouped, seeded CV (`:lambda_min`/`:lambda_1se`); change the centering/whitening in
  `fit` and the penalty scale (`λ·std`) moves with it. `AdaptiveLasso` runs its `pilot`
  through `solve_coefficients` (forwarding `row_groups`), then a weighted-L1 GLMNet solve with
  fixed `penalty_factor`; `AdaptiveRidge` is a pure-core reweighted-ridge loop sharing the
  centered-`X` contract. Validated in the separate `test/glmnet/` env (GLMNet-backed) and
  `test/unit/test_fit.jl` (core `AdaptiveRidge` / `FixedCoefficients` solves), never mixing
  the two (GLMNet absent in the core suite).
- **The objective is the energy MSE at EVERY weight setting** (`fitting/fit.jl`
  `_assemble_problem`): the energy block is whitened by `√((1 − w_T − w_F)/n_E)`, and the
  `w_T = w_F = 0` branch applies the same `1/√n_E` rather than returning the centered design
  unscaled. It used to return it unscaled, i.e. minimize the SSE there and the MSE everywhere
  else, which made the objective DISCONTINUOUS at zero: OLS is scale-invariant, but every
  penalized estimator's Gram jumped by `n_E`, so one `lambda` meant two things — measured on
  60 configurations, `Ridge(lambda = 1.0)` gave `rmse_energy` 0.0038 at `w_T = 0` and 0.086 at
  `w_T = 1e-12`. λ for an energy-only penalized fit is therefore `n_E` times smaller than in
  any record predating this; the `test_selection.jl` grids divide by their config count, and
  `test_asr.jl`'s hand-built assembly reference carries the `se` factor explicitly (which is
  what makes it a pin on the convention).
- **The penalty metric ↔ every weight map ↔ the fitting doors** (`fitting/metric.jl`,
  `fitting/estimators.jl`, `fitting/selection.jl`, `fitting/momentfit.jl`,
  `fitting/fit.jl`): a metric is per-column data carried on `Ridge` / `AdaptiveRidge` /
  `GroupAdaptiveRidge`, and it enters at SIX sites — `Ridge`'s solve, `AdaptiveRidge`'s
  iteration 0 and its weight update, `_solve_gar`'s cold start and
  `_group_adaptive_weights!`, and the three `_penalty_diagonal` methods behind
  GCV/`effective_dof`. Always in the DENOMINATOR of a weight map: outside it the
  adaptive estimators stop being scale invariant and the group-L0 fixed point reads
  `λ·v_g·⟨m⟩_g` instead of `λ·v_g`. Three consequences that are easy to miss when adding
  a site: (i) **`select_fit` resolves the metric ONCE** and passes it to every solve,
  every GCV weight, every fold, and the cold re-solve of the selected point — the
  returned fit and its score must describe the same estimator, which is why `est_sel` is
  rebuilt through the positional inner constructor (no default for `metric`, so a missed
  site is a `MethodError`, not a silent uniform fit); (ii) **any column selection
  reduces the metric with the design** — `refit`'s support and the pointed fit's
  vanishing-column freeze, through `_reduce_to_active`, which now has methods for
  `Ridge` / `AdaptiveRidge` / `AdaptiveLasso` (via its pilot) as well as
  `GroupAdaptiveRidge`; (iii) **`mⱼ = 0` means unpenalized**, so a new zero has to be a
  structural exemption (`_refuse_zero_metric`), the unpenalized block needs full column
  rank (`_check_free_block`), and the dof splits (`_effective_dof_free`). Under an ASR /
  freeze reparameterization the penalty compresses to `Z'·Diagonal(D)·Z` while `metric`
  stays indexed by the BASIS columns — same convention as `column_groups` — and a
  `metric === nothing` still takes the exact `λ·I` γ-space path, which is what keeps
  unweighted fits bit-identical. Scope: **pure spin only** (`penalty_metric(::SLCEBasis)`
  refuses a displacement basis, `_check_metric_provenance` refuses `force_weight > 0`);
  the joint reference ensemble is a separate spec. `MetricProvenance` is checked at
  `fit` / `refit` / `select_fit` / `cross_validate` / `fit(MomentFit, …)` — a metric
  built for the wrong channel, basis, or `torque_weight` is invisible to every numerical
  gate, since scale invariance holds for any `m ∝ c²`, right or wrong.
- **GCV ↔ `_assemble_problem` ↔ `islinear` ↔ the GAR weight map** (`fitting/selection.jl`,
  `fitting/estimators.jl`): `gcv`/`effective_dof` reassemble the design through
  `_assemble_problem` (change the centering/whitening and the score moves with `fit`),
  are gated by `islinear`, **refuse a `refit` result by name** (`SLCEFit.support`,
  recorded at refit since the 2026-08-11 review — the reconstruction is the FULL
  design, not the support the refit solved on, and the sub-stage is not stored),
  charge the `+1` intercept only when the energy block carries weight, and
  recompute the converged penalty diagonal through the
  **same** functions the solvers iterate — `_group_adaptive_weights!` (the single
  definition of `Dⱼ = mⱼ·v_g/(Σ_{k∈g} m_kβ_k² + p_g·ε)`, and what it returns is the
  penalty DIAGONAL, metric included; `_penalty_diagonal` has one method per linear
  estimator, and `AdaptiveRidge`'s `mⱼ/(mⱼβ² + ε)` must stay in sync with its solve
  loop). Change a
  weight formula in the solver and the `_penalty_diagonal` method, the design-notes §13
  derivation, and the dense-hat-matrix tests in `test/unit/test_selection.jl` move
  together. §13 states the derivation in `θ` and `δ`; the API spells those out as
  `cost_exponent` and `score_rtol`, and the mapping is written in §13 itself and in the
  `cost_weights` docstring — keep the formulas in Greek (a derivation written in
  `score_rtol` is unreadable) and the keywords in words, never the other way round. `select_fit`'s alive-group rule is the `refit` scaled-magnitude support rule
  (`|jϕⱼ|·‖X[:,j]‖ > threshold`) applied per group — change one side and the other (and
  the E2E cost-recomputation test) follows; `select_support` uses that rule only to build
  its threshold grid and to drive `refit` (column-wise), and reads its `n_alive`/`cost`
  columns back off the **returned refits** (groups with a nonzero de-biased coefficient).
  Do not "simplify" that back to `m_g > t`: the two agree without a constraint, but under
  an ASR a support that splits a constraint-coupled column set structurally zeroes some
  survivors, and `m_g > t` would then report a cost the returned model does not pay.
  `group_costs(...; asr)` additionally prices structurally infeasible groups at zero, and
  `select_support` passes the fit's own `reparam` — so a change to either the discount or
  the post-refit derivation must move the `test_asr.jl` front test with it. `select_support`'s evalset score and
  `cross_validate`'s holdout score share the prediction-space convention
  (`y_E − (j0 + X_E·jϕ)`, `y_T − X_T·jϕ`, combined as `(1−w)·MSE_E + w·MSE_T`) —
  change `fit`'s objective normalization and both scores must follow.
  `salc_groups`/`group_costs` assume sorted `SALCBasis.keys` and canonical members; the
  entry key is `(atoms, shifts, slotkeys, index)` and the cost is the summed **slot
  count** over distinct entries — change either and re-check the brute-force union test
  (`test_selection.jl` "group_costs: brute-force union, additivity, validation").
  **What that key does and does not mirror, established 2026-07-28 and previously
  recorded wrongly here.** It matches the shape of SLCEMonteCarlo's
  `_reduced_key_type(::Type{DecoratedTerm})` (`reduce.jl`), and the canonical slot order
  is the one `_align_reduced` reproduces — but there is **no adjacency merge on that
  key**: `decorated_terms` emits one term per `(SALC, member, SALCTerm)` without
  merging, `TiledHamiltonian` tiles without merging, and `reduce_cell` buckets on the
  key and then splits each bucket by `(coef, folded)`, i.e. it merges *translation
  orbits*, not channels. So the union is a genuine lower bound, not an identity, and
  any claim that it equals a realized entry count needs re-deriving.
  The **slot-count factor is load-bearing and was missing until 2026-07-28**:
  `_push_term_programs!` emits one **site** program per member site position
  (`nnz · length(slots)` entries, walked every sweep) plus one **energy** program
  (`nnz`, walked once per run by `total_energy`). Counting `nnz` priced the energy
  program, so groups were mis-ranked by a factor `body`. Do not "simplify" it back, and
  do not hoist it to a per-group constant — `column_groups` may be coarser than
  `salc_groups`, and only a per-entry factor stays additive under that.
  What the metric deliberately omits: the per-visit `O(nrows)` block (`nrows` comes from
  `row_layout` over basis *keys*, so it never shrinks when a group dies — selection-
  invariant and unattributable), and sweep multiplicity (spin / overrelaxation /
  displacement passes, whose counts are run-time `UpdatePlan` inputs and whose
  `site_has_*` predicates depend on which *other* groups survive, making the true cost
  supermodular).
  The cross-package check on record — `Σc_g = 744,636` on l044 — was against the term
  tensors' `Σnnz`, i.e. against the **energy** program, so it confirmed internal
  consistency and not the sweep cost; it needs redoing against site-program entries.
  There is no script, and SLCEMonteCarlo references neither `salc_groups` nor
  `group_costs`. Deliberately left manual: a mismatch costs prediction accuracy in the
  cost-weighted selection, not correctness of any fitted number.
- **`fit` ↔ `refit` share `_assemble_problem`** (`fitting/fit.jl`): the `(X, y, xbar, ybar,
  groups)` centering/whitening assembly lives in one helper so the two build identical
  designs — change the centering or whitening there and **both** move together (the oracle
  pins `fit`'s numerics). `refit` re-solves on the scaled-magnitude support
  `|jϕ_j|·‖X[:,j]‖ > threshold` of an existing fit (a column sub-matrix), so it rejects a
  `FixedCoefficients`-backed estimator (fixed-length pilot vector ≠ support length).
  **`identifiability` (`fitting/diagnostics.jl`) is a third consumer of that same
  assembly** — it reports the numerical rank of the design the fit solved (or would
  solve), so it must reassemble through `_assemble_problem` with the fit's own
  `(w_T, w_F, asr)` and share `fit`'s validation (`_validate_fit_request` /
  `_resolve_asr_rep`, extracted for exactly that reason: a pre-fit diagnostic must
  raise the fit's errors, not fail deeper in the assembly). `fit`'s standing
  dead-column warning is the cheap per-column half of the same question (a
  **relative** norm cut, `_DEAD_COL_RTOL` — the package's convention everywhere else,
  and an exact `iszero` test has a measured false negative: `Σ_a u_a` columns sit at
  ~1e-19 on center-of-mass-free samples) and leans on `refit`'s scaled-magnitude rule
  (`|jϕⱼ|·‖X[:,j]‖ > threshold`) to justify its "`refit` drops them" advice — change
  that rule and the message follows; the advice is deliberately dropped when the
  reparameterization has constraint ROWS, where the reported indices are γ directions and
  no SALC sits at index `j`. **Zero rows is the other case and it is NOT that one**: the
  rep is then a pure freeze, `Z` is exactly the selection matrix of `free`, so γ direction
  `k` IS `jphi` column `free[k]` — reporting the γ index there hands the caller a number
  indexing nothing they hold, so the message maps it back and gives the β-space advice
  (gate (d), `test_resolvability.jl`, which reads the log's `columns`/`coordinates`
  kwargs). An ALL-zero design is likewise not an exempt case but the loudest one — it used
  to return early ("nothing to rank"), so the only fit that determined nothing was the
  only fit that said nothing. The
  physical accounting the diagnostic exists for (which channel sees which
  translation-violating directions under center-of-mass-free sampling) is pinned in
  `test/unit/test_identifiability.jl` against the FIXTURE's exact numbers
  (p/rank(A)/q/pure-spin counts): change the basis fixture and re-derive them there,
  never relax the equalities.
  **`SLCEFit.residuals` is the energy-only residual** `y_E − (j0 + X_E·jϕ)` (not Magesty's
  combined whitened residual); the diagnostics report energy and torque blocks separately
  (`residuals_energy` returns the stored vector, `residuals_torque`/`rss_torque` recompute
  `y_T − X_T·jϕ` and validate `has_torque`; `r2_*`/`rmse_*` build on `rss_*`).
- **`src/units.jl` is the family's ONLY definition of `KB_EV` / `resolve_kt`.** This
  package has no temperature — a fitted model is a zero-temperature energy surface —
  but it owns the *convention*, because it is the one package SLCEMonteCarlo,
  SLCEDynamics and SLCETools all depend on. Two of them used to carry private copies,
  character for character identical; that is the shape a drift hazard has before it
  drifts, since neither suite can see the other's constant. A downstream package
  `using SLCE: KB_EV, resolve_kt` and re-exporting is correct; a second `const KB_EV`
  anywhere in the family is not.
- **Family generics are extended, never re-defined: `n_atoms`, `has_disp`.** A
  downstream package that wants the same question at its own granularity writes
  `import SLCE: has_disp` and adds a method (SLCEMonteCarlo does this for
  `has_disp(::TiledHamiltonian)` and `n_atoms(::ReducedCell)`, SLCETools for
  `n_atoms(::ExchangeModel)` / `n_atoms(::MultipoleModel)`). Defining a second generic
  of the same name compiles, passes both suites, and leaves a user who loads both
  packages with two functions that cannot both be called unqualified. When adding a
  predicate here that a sampler will plausibly want, `public` it so the downstream
  `import` is a supported move rather than a reach into internals.

## Downstream divergence: the pure-spin carve-out (SCEFitting.jl)

`SCEFitting.jl` is the pure-spin carve-out of this package and a **standalone**
package (it does not depend on this one). Files are kept close on both sides so
patches apply either way, but a set of divergences is deliberate — and several of
them are the same name with the OPPOSITE meaning, which is the kind that bites
silently.

**The authoritative table is `../SCEFitting.jl/CLAUDE.md`, "Upstream divergence
ledger".** Read it before porting anything in either direction; do not maintain a
second copy here, which would drift. The traps that matter when editing THIS package:

- `_orbit_salcs_decors`'s screen is `soc::Bool` **positional** here (`soc = true`
  keeps every `L_S`); downstream it is a required keyword `isotropy` with the
  opposite polarity. A verbatim copy of a call in either direction must be a
  `MethodError`, never a silently inverted screen — so never give either a default.
- `MomentSpec(; soc = false)` here is `MomentSpec(; isotropy = true)` there.
- Penalty-metric names: `_group_adaptive_weights!`, `_effective_dof_gram` /
  `_effective_dof_free`, `cost_exponent`, `FixedCoefficients`, and the `row_groups` /
  `nullspace` keywords are this package's spellings; downstream they are
  `_gar_weights!`, `_edof` / `_edof_free`, `theta`, `PrecomputedPilot`, `groups`.
  Port LOGIC, never signatures.
- Things that exist only here: the ASR / freeze reparameterization (so the penalty
  compresses to `Z'·Diagonal(D)·Z` and the metric is indexed by BASIS columns), the
  displacement channel, `SolidHarmonics`' gradient API, `CountingOracle`.
- Things that exist only downstream: the moment channel's λ-selection API
  (`cross_validate(::MomentDataset, …)`, `MomentCVResult`, `gcv` / `effective_dof`
  for `MomentFit`), the `[moment]` TOML section, the function-space reduction.

## Tests

| Command | Purpose |
|---|---|
| `julia -t 4 --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia -t 4 --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia -t 4 --project -e 'using Pkg; Pkg.test()'` | JET type-stability |
| `julia --project=test/integration test/integration/runtests.jl` | whole pipeline on six named real crystals |
| `julia --project=test/oracle test/oracle/runtests.jl` | from-scratch numerics vs pinned Magesty |
| `julia --project=test/sunny test/sunny/runtests.jl` | real `Sunny.System` energy vs SLCE (extension) |
| `julia --project=test/glmnet test/glmnet/runtests.jl` | GLMNet Lasso / elastic-net solve (extension) |
| `for f in examples/*.jl; do julia --project=examples $f; done` | the runnable examples' own `@assert` gates |

The core suite (`runtests.jl`) dispatches on the `TEST_MODE` env var
(`default`/`all`/`unit`/`aqua`/`jet`) and never depends on Magesty. The oracle,
Sunny, and GLMNet suites are separate environments that carry the heavy/optional
dependency (a pinned Magesty.jl / Sunny / GLMNet) the core deliberately omits.

**The integration tier (`test/integration/`, own CI job) is not more unit tests.**
Every unit file gates one stage against one hand-built fixture; nothing gated the
*combination* on a crystal with a name, and that is how the last several real
defects were found ([[feedback-concrete-system-bug-hunting]] in spirit: one real
system, end to end). It walks bcc Fe (conventional **and** 2×2×2), B2 FeRh, hcp Co,
wurtzite GaN and rocksalt MnO through build → data → ASR → fit → recovery → every
deliverable → persistence, and declares a **coverage matrix**: per row, which of
the 17 columns run and — with a reason in the file — which do not. The driver
refuses a row that leaves a column unaccounted for and refuses to report success if
a declared column never executed. It needs Spglib, which is precisely why it is not
in `Pkg.test()`: the core suite must never depend on a symmetry backend, and here a
hand-assembled group would defeat the purpose (Spglib is the tier's *independent*
oracle for the space group). Adding a column means adding it to `COLUMNS`, to
`CHECKS`, and to every row's `runs` or `skips` — that is the point.

**Run the core suite with `-t N>1`.** CI pins `JULIA_NUM_THREADS: 4` for exactly
this reason: the threaded-vs-serial gates (`test_threading.jl` for the pure-spin
AND joint design builders, the deterministic basis build in `test_salc.jl`)
compare a parallel result against a serial reference, and at one thread
`Threads.@threads` *is* the serial reference — the assertions pass while
exercising nothing. A hoisted scratch buffer or a lazily-filled cache in the
orbit loop is invisible to a single-threaded run.

The examples are CI'd too (their own job), because each one self-gates with
`@assert`s on a recovered coupling — the fence that caught the canonical-member
J/2 regression — and nothing else runs them: the docs job executes the
`docs/src/tutorials/*.md` `@example` blocks, which are different files. Their
`Manifest.toml` is untracked, so a stale local one (it survived the M0 rename
still naming `MagestyRebuild`) breaks the env, not the repo: `rm
examples/Manifest.toml && julia --project=examples -e 'using Pkg; Pkg.resolve()'`.

## Git (this rebuild)

Local git (`add` / `commit` / branch) is pre-authorized for this exploratory
rebuild — no per-action confirmation. Only remote operations (`push`,
registration) require confirmation. Commit directly to `main`; the `feat/v0-core`
working branch was retired once v0 landed (it tracked `main` one-to-one, so it
gave no isolation). Cut a short-lived topic branch only for genuinely risky or
experimental work you do not want on `main` yet.

## References

- `STYLE_GUIDE.md` — package-specific style deltas.
- `SPEC.md` — realized architecture, types, public API.
- `docs/design-notes.md` — why the rebuild diverges from Magesty (the refinements).
  **§15 is scoping, not history**: which invariance layers a fitted model satisfies,
  which it does not, and the two design constraints on ever adding the missing ones.
- `CHANGELOG.md` — what landed in the v0 slice.
- `examples/heisenberg_chain.jl` — runnable end-to-end (recovers `J`);
  `examples/kagome_threebody.jl` — 3-body / multi-term SALCs, energy+torque co-fit.
- `references/` — supporting literature (notes tracked, PDFs local-only).
