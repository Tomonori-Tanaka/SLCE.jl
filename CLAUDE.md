# CLAUDE.md

> Shared baseline (numerical-correctness priority, JP-conversation / EN-repo
> policy, Conventional Commits, Julia style, shared subagents) is inherited from
> `~/Packages/CLAUDE.md`. Only package-specific rules live here.

## Project goal

Clean, extensible, Julia-native rebuild of `Magesty.jl`: fit a spin-cluster
expansion (SCE) `E = j0 + Σφ Jφ Φφ({e_a})` to noncollinear DFT data. The numerical
core (tesseral harmonics, Clebsch–Gordan coupling, symmetry-adapted basis, design
matrix, regression) is **reimplemented from scratch**; `Magesty.jl` is used only
as a *pinned numerical oracle* in `test/oracle/`. Priority: numerical correctness
and reproducibility over stylistic concerns. See `SPEC.md` for the realized
architecture and `docs/design-notes.md` for the rationale.

Both SCE observables are fitted: the **energy** `E` and the per-atom **torque**
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
  of an atom carries the *same spin*, so its interaction is collinear with (an alias of)
  the minimum-image one and is not independently fittable. The default `MinimumImage`
  selection enumerates exactly this set (boundary ties kept; `i==j` self-pairs and
  reused-atom clusters dropped); `cutoff = Inf` is the whole WS cell. `AllImages`
  (every image, `R`-distinguished) is **only** for the future generalized-Bloch /
  spin-spiral path where `e^{iq·R}` resolves the images. For `cutoff < min_d dᵢ / 2` the
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
  shell is never split. The minimum-image search box is adaptive — change it and re-check
  the skewed-cell test. `images` is **not** persisted (the full SALC basis is stored and
  reloaded verbatim), so only `read_setup`/`SLCEBasis` carry it. **At `N ≥ 3` the clique
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
- **The spin-free entry path shares ONE predicate** (`slce/model.jl`
  `_basis_has_spin` ↔ `io/dftsource.jl` `SLCEDataset`'s `use_torque = nothing` ↔
  `slce/forceconstants.jl` `force_constants(; spins = nothing)` ↔ `fitting/fit.jl`
  `_no_spins` / `_validate_config_pair`): every place that lets a lattice-only caller
  omit a magnetic state asks the same question, and it is **not** `is_soc_free`.
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
  so the read path keeps compatibility even when the write path does not.
- **BasisSpec sugar resolution ↔ canonical consumers** (`slce/truncation.jl`,
  `slce/model.jl`, `io/input.jl`): the ergonomic forms (label keys, `"*"` wildcards,
  body-keyed tables, unordered `"A-B"` pair keys, specificity resolution) are expanded
  ONCE, in the `BasisSpec` keyword constructor; everything downstream —
  `SLCEBasis`'s fan-out (`_superset_cutoff` → `build_neighbor_list`,
  `cutoff` → `candidate_clusters` per-edge admission, `lsum` → `_enumerate_ls`),
  persistence, `show` — reads only the dense fields. Add a sugar form or change the
  specificity rule → update the TOML reader (`_cutoff_from_input` etc.), the BasisSpec
  docstring, and `test/unit/test_truncation.jl` together. The sector table follows the
  same discipline: `Sector` is unresolved sugar, `_resolve_sector(s)` (truncation.jl)
  produces the dense `SectorRule` rows, and `nbody` / the per-body `cutoff` envelope
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
  invert the channel set silently. `isotropy` now survives ONLY as (a) the
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
  read straight off a `SALCKey` (`body` / `orbit_id` / `ls`→comma string / `Lf` /
  `block`) plus `jphi`; the `J` column pairs with `basis.salc_basis.keys` **positionally**
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
  counts), never off `3·n_atoms`. The selection layer has NO `force_weight`
  yet: `cross_validate`/`select_fit` score energy(+torque) only (documented), and
  `select_support` rejects a force co-fit outright — extending any of them means adding
  force-presence strata AND the channel-split `group_costs` (design record §6).
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
  structurally zeroed, warned). Nothing is persisted: `asr_residual(model)`
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
  `dataset.asr` instead silently re-assembles a staged fit as an unstaged one. Whether
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
  (`multipole_terms` precedent — a coefficient fitted to zero is one fit's property,
  `refit` moves it), and its `maxlog = 1` is why the gate checks the SILENT cases
  first: a silence assertion after the warning has fired asserts nothing.
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
  two agree only when every site carries exactly one spin factor and nothing else, so
  a force-constant term (sites with no spin factor at all) is where the pure-spin-era
  shortcut invents a factor out of nothing — that is the case the gate must contain, and
  a fixture whose sites all happen to carry spin makes the whole check vacuous (the
  first docs fixture did exactly that). `decorated_terms` is the general surface;
  `multipole_terms` is its frozen p = 0 predecessor and REFUSES any displacement model,
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
  `test/unit/test_introspect.jl`): `multipole_terms` is the **public, stable** view downstream
  packages (the `SLCETools.jl` mean-field samplers) read instead of `model.basis.salc_basis.salcs` /
  `SALCMember` / `SALCTerm`. It returns the **raw** fitted `jϕ` as `coef` and leaves the per-N
  scale `(4π)^(body/2)` to the consumer — the scale lives in exactly one place (the
  reconstruction gate `_energy_from_terms`), so do **not** also apply it inside
  `multipole_terms`. `bilinear_terms` is a thin public wrapper of the general
  `_bilinear_terms` extraction (in `slce/bilinear.jl`), so its numerics move with the
  Sunny coupled-site above.
  Add or rename a `MultipoleTerm` field → update the gate and the `SLCETools.jl` consumers
  (`mfa/bridge.jl` — renamed from `sce_bridge.jl`; grep the package, not this name alone).
- `solve_coefficients(est, X, y; row_groups)` receives a **column-centered** `X` (⇒ the
  solver adds no intercept; `j0` is recovered analytically in `fit`). Every estimator —
  in-tree or in an extension — must honor this. `row_groups` (optional) labels rows from the
  same physical sample (in a co-fit, a configuration's energy row and its
  torque-component rows share a label); a resampling estimator (CV-based `ElasticNet` /
  `Lasso` / `AdaptiveLasso` in `ext/SLCEGLMNetExt.jl`) must keep same-label rows
  in the same fold so CV does not leak within-configuration structure. The analytic / adapter
  estimators (`OLS` / `Ridge` / `AdaptiveRidge` / `PrecomputedPilot`) ignore it. The GLMNet
  solve uses `intercept = false` + column `standardize` and selects λ by configuration-
  grouped, seeded CV (`:lambda_min`/`:lambda_1se`); change the centering/whitening in
  `fit` and the penalty scale (`λ·std`) moves with it. `AdaptiveLasso` runs its `pilot`
  through `solve_coefficients` (forwarding `row_groups`), then a weighted-L1 GLMNet solve with
  fixed `penalty_factor`; `AdaptiveRidge` is a pure-core reweighted-ridge loop sharing the
  centered-`X` contract. Validated in the separate `test/glmnet/` env (GLMNet-backed) and
  `test/unit/test_fit.jl` (core `AdaptiveRidge` / `PrecomputedPilot` solves), never mixing
  the two (GLMNet absent in the core suite).
- **GCV ↔ `_assemble_problem` ↔ `islinear` ↔ the GAR weight map** (`fitting/selection.jl`,
  `fitting/estimators.jl`): `gcv`/`effective_dof` reassemble the design through
  `_assemble_problem` (change the centering/whitening and the score moves with `fit`),
  are gated by `islinear`, and recompute the converged penalty diagonal through the
  **same** functions the solvers iterate — `_gar_weights!` (the single definition of
  `wⱼ = v_g/(‖β_g‖² + p_g·ε)`; `_penalty_diagonal` has one method per linear estimator,
  and `AdaptiveRidge`'s `1/(β² + ε)` must stay in sync with its solve loop). Change a
  weight formula in the solver and the `_penalty_diagonal` method, the design-notes §13
  derivation, and the dense-hat-matrix tests in `test/unit/test_selection.jl` move
  together. `select_fit`'s alive-group rule is the `refit` scaled-magnitude support rule
  (`|jϕⱼ|·‖X[:,j]‖ > threshold`) applied per group — change one side and the other (and
  the E2E cost-recomputation test) follows; `select_support` reuses the same rule for its
  per-point alive/cost columns while delegating each point's fit to `refit` itself
  (column-wise), so all three move together. `select_support`'s evalset score and
  `cross_validate`'s holdout score share the prediction-space convention
  (`y_E − (j0 + X_E·jϕ)`, `y_T − X_T·jϕ`, combined as `(1−w)·MSE_E + w·MSE_T`) —
  change `fit`'s objective normalization and both scores must follow.
  `salc_groups`/`group_costs` assume sorted
  `SALCBasis.keys` and canonical (v4) members; the entry key `(atoms, shifts, ls, index)`
  mirrors what the SLCEMonteCarlo adjacency merge folds — change either representation
  and re-check the brute-force union test (`test_selection.jl` "group_costs: brute-force
  union, additivity, validation"). The cross-package half — that `Σ_{g alive} c_g` really
  equals SLCEMonteCarlo's contraction-entry count — has been confirmed **by hand once**, on
  l044; there is no script and SLCEMonteCarlo references neither `salc_groups` nor
  `group_costs`. Deliberately left manual: a mismatch costs prediction accuracy in the
  cost-weighted selection, not correctness of any fitted number.
- **`fit` ↔ `refit` share `_assemble_problem`** (`fitting/fit.jl`): the `(X, y, xbar, ybar,
  groups)` centering/whitening assembly lives in one helper so the two build identical
  designs — change the centering or whitening there and **both** move together (the oracle
  pins `fit`'s numerics). `refit` re-solves on the scaled-magnitude support
  `|jϕ_j|·‖X[:,j]‖ > threshold` of an existing fit (a column sub-matrix), so it rejects a
  `PrecomputedPilot`-backed estimator (fixed-length pilot vector ≠ support length).
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
  that rule and the message follows; the advice is deliberately dropped in the ASR
  branch, where the reported indices are γ directions, not `jphi` positions. The
  physical accounting the diagnostic exists for (which channel sees which
  translation-violating directions under center-of-mass-free sampling) is pinned in
  `test/unit/test_identifiability.jl` against the FIXTURE's exact numbers
  (p/rank(A)/q/pure-spin counts): change the basis fixture and re-derive them there,
  never relax the equalities.
  **`SLCEFit.residuals` is the energy-only residual** `y_E − (j0 + X_E·jϕ)` (not Magesty's
  combined whitened residual); the diagnostics report energy and torque blocks separately
  (`residuals_energy` returns the stored vector, `residuals_torque`/`rss_torque` recompute
  `y_T − X_T·jϕ` and validate `has_torque`; `r2_*`/`rmse_*` build on `rss_*`).

## Tests

| Command | Purpose |
|---|---|
| `julia -t 4 --project -e 'using Pkg; Pkg.test()'` | unit + Aqua (default) |
| `TEST_MODE=all julia -t 4 --project -e 'using Pkg; Pkg.test()'` | unit + Aqua + JET |
| `TEST_MODE=jet julia -t 4 --project -e 'using Pkg; Pkg.test()'` | JET type-stability |
| `julia --project=test/oracle test/oracle/runtests.jl` | from-scratch numerics vs pinned Magesty |
| `julia --project=test/sunny test/sunny/runtests.jl` | real `Sunny.System` energy vs SCE (extension) |
| `julia --project=test/glmnet test/glmnet/runtests.jl` | GLMNet Lasso / elastic-net solve (extension) |
| `for f in examples/*.jl; do julia --project=examples $f; done` | the runnable examples' own `@assert` gates |

The core suite (`runtests.jl`) dispatches on the `TEST_MODE` env var
(`default`/`all`/`unit`/`aqua`/`jet`) and never depends on Magesty. The oracle,
Sunny, and GLMNet suites are separate environments that carry the heavy/optional
dependency (a pinned Magesty.jl / Sunny / GLMNet) the core deliberately omits.

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
