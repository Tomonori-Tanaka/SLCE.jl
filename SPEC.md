# SLCE.jl — specification and architecture

Work-in-progress rebuild of Magesty.jl. This file tracks the **realized**
architecture as milestones land; the full design rationale lives in the
implementation plan.

## Pipeline

```
geometry → symmetry → clusters → basis (SALC) → design matrix → fit → predict
```

## Module layout (`src/`)

Single top-level `module SLCE`, small files included in dependency
order; the two self-contained numeric kernels (`Harmonics`, `AngularMomentum`)
are nested submodules. Heavy / optional dependencies (Spglib, GLMNet, Sunny)
live in `ext/` package extensions; the core loads without them. Persistence and
input files use the stdlib `TOML` (no external dependency).

```
src/
├── SLCE.jl        top-level module: includes, exports, `public` tier
├── units.jl             KB_EV / resolve_kt — the family's one kelvin ↔
│                        model-energy conversion (this package has no
│                        temperature; it owns the convention)
├── geometry/            lattice.jl, crystal.jl, neighborlist.jl
├── symmetry/            types.jl, backend.jl (Spglib method in ext/)
├── basis/               Harmonics.jl, AngularMomentum.jl (submodules),
│                        coupledbasis.jl, salc.jl, salcbasis.jl
├── clusters/            enumerate.jl, orbits.jl
├── fitting/             estimators.jl, design.jl, fit.jl, diagnostics.jl,
│                        selection.jl (MC-cost groups, GCV, λ-path + Pareto)
├── slce/                 model.jl (pipeline types), coeftable.jl,
│                        bilinear.jl (tesseral → Cartesian extraction),
│                        introspect.jl (public introspection)
├── interop/             sunny.jl (primitive unfold + `to_sunny` entry point)
└── io/                  persist.jl, input.jl, dftsource.jl
```

The include order in the SLCE layer is `slce/coeftable.jl` → `slce/bilinear.jl` →
`slce/introspect.jl` → `interop/sunny.jl`: the bilinear extraction is a core
capability consumed by both the introspection and the Sunny interop.

## Extension seams (contracts)

- **DFT sources**: `read_configs(src::AbstractDFTSource) -> Vector{TrainingDatum}`.
- **Estimators**: subtype `AbstractEstimator` + `solve_coefficients(est, X, y; row_groups)
  -> jphi` (the `(X, y)` is already column-centered and row-scaled — add no intercept,
  do not re-weight; `row_groups` labels rows from the same sample for grouped resampling).
  Types live in core; solver methods needing GLMNet live in `ext/`.
- **Symmetry**: `analyze_symmetry(backend::AbstractSymmetryBackend, crystal; tol) -> SpaceGroup`.

## Realized so far

### geometry (M1)
- `Lattice(vectors; pbc)` — columns are lattice vectors; `reciprocal = inv`,
  `interplanar_spacing(lat, i) = 1/‖row_i(reciprocal)‖`.
- `Crystal(lattice, frac_positions, species, species_labels)` — fractional coords
  wrapped to `[0,1)` on periodic axes (inner constructor); `n_atoms`,
  `cartesian_positions`.
- `build_neighbor_list(crystal, cutoff) -> NeighborList` — cutoff-driven image
  range `N_d = ceil(cutoff·‖b_d‖)`; `NeighborPair` retains the integer lattice
  `shift` (R) for later reciprocal-space / spin-spiral rows. Validated against an
  independent over-large-shell brute force (cubic multi-shell + sheared triclinic).
- `build_neighbor_list(crystal, cutoff, selection)` — periodic-image selection
  (`AbstractImageSelection`): `MinimumImage()` (the SLCE-fitting default) keeps only the
  minimum-image, plain-PBC-resolvable pairs of the Wigner–Seitz cell — with boundary
  ties (`L/2` faces / edges / `(L/2,L/2,L/2)` corners) and **no `i==j` self-pairs**
  (same spin ⇒ not an independent pair) — over an adaptive, skew-safe image box;
  `cutoff` may be `Inf` (whole WS cell). `AllImages()` is the every-image enumeration
  above (finite cutoff only), the generalized-Bloch / spin-spiral seam.

### basis — `Harmonics` submodule (M2)
- `Harmonics.Zlm(l, m, u)` — real tesseral harmonic (Drautz convention, per-site
  `(4π)^(−1/2)`); `Harmonics.grad_Zlm(l, m, u)` — tangent-projected Cartesian
  gradient `∂Z − u(u·∂Z)`. `lm_index(l, m) = l²+l+m+1`, `num_lm(lmax)`. Legendre
  primitive from `LegendrePolynomials.dnPl`. The tesseral `l ≤ 2` Cartesian-conversion
  constants `N1 = √(3/4π)`, `A2 = √(15/16π)`, `B2 = √(5/16π)` are defined once here
  (used by `slce/bilinear.jl` and downstream consumers).
- Validated by: closed-form standard solid harmonics (`l ≤ 2`), gradient tangency
  + on-sphere central difference, and agreement with Magesty's `TesseralHarmonics`
  (oracle) to `atol = 1e-13, rtol = 1e-12` on values and `1e-12 / 1e-11` on
  gradients — *not* bit-for-bit, as this line used to claim: the two use different
  Legendre primitives, so the last few ulp differ by construction.

### basis — `SolidHarmonics` submodule (joint spin–lattice M1)
- The displacement-channel kernel: real solid harmonics `Rₗₘ(u)` evaluated as
  homogeneous polynomials in the Cartesian components (homogenized
  associated-Legendre recurrence — regular and exact at `u = 0`, the densest
  sampling point of a reference-structure expansion), plus the **Euclidean**
  gradient `∂Rₗₘ/∂(x,y,z)` (no tangent projection — what force rows need; the
  spin-side on-sphere `grad_Zlm` is a different object and neither kernel
  substitutes for the other).
- **Normalization is 4π-free (Racah-type)**: `Rₗₘ(û) = √(4π/(2l+1))·Zₗₘ(û)` on
  the unit sphere, so `R₁₋₁, R₁₀, R₁₁ = y, z, x` exactly and only spin sites
  contribute to the per-term `(4π)^(n_spin/2)` design-matrix scale
  (`docs/specs/spin-lattice-ce-design.md` §3).
- API: batch `solid_harmonics`/`solid_harmonics_grad` (+ allocation-free `!`
  forms), single-`(l,m)` `Rlm`/`grad_Rlm`, `solid_harmonic_index` (same layout
  as `Harmonics.lm_index`).
- Validated by: the explicit-factor cross-check against `Harmonics.Zlm` up to
  `l = 16`, homogeneity `R(λu) = λˡR(u)`, exact `u = 0` values/gradients, and
  central-difference gradients (`test/unit/test_solidharmonics.jl`).
- Test support (`test/support/countingoracle.jl`): the **CountingOracle** — the
  independent decorated-cluster invariant-counting and Reynolds-projection
  oracle (polynomial-composition slot matrices, cycle-wise characters with
  pair-orientation swap signs, plethysm `Sym^p`), ported from the prototype
  with its review blockers fixed (per-slot swap flags, slot quantum-number
  validation, repeated-slot rejection, closure/integrality guards). Scope: no
  time reversal (Σl_spin-even comparisons only), no translation folding, no
  strain slot. Gate infrastructure for count ≡ projector-rank comparisons
  (`test/unit/test_countingoracle.jl`: the cubic 2-invariant count, the
  axial-vs-polar inversion kill-shot, the ⟨C4z⟩ two-pair swap regression, the
  bent/collinear ligand pair, the chirality-twist decoration).

### basis — `AngularMomentum` submodule (M3, M4)
- `clebsch_gordan` (Racah formula); `wignerD_real(l, R)` (real Wigner-D, built
  from the package's own `Zₗₘ` by an exact least-squares fit — handles improper
  rotations); `coupling_paths`, `coeff_tensor_complex`, `c2r_matrix`,
  `complex_to_real_tensor`, `build_real_bases`. `CoupledBasis{R}` / `coupled_bases`.
- Validated: CG vs WignerSymbols + orthonormality; Wigner-D functional identity +
  special cases + Magesty Δl; coupled-tensor realness, Frobenius norm² = 2Lf+1,
  rotational equivariance `f(R·e) = Δ^Lf(R)·f(e)`, and Magesty `build_all_real_bases`.

### symmetry (M5)
- `AbstractSymmetryBackend` + `analyze_symmetry(backend, crystal; tol) -> SpaceGroup`;
  in-tree `NoSymmetry` (P1); `SpglibBackend` type in core, method in
  `ext/SLCESpglibExt`. `map_sym` derived in-tree (shared by all backends).

### clusters (M6) — arbitrary body order
- `ClusterMember` (atoms + per-site lattice `shift` R), `ClusterOrbit`, `ClusterSet`;
  `candidate_clusters` enumerates `N`-body **edge-admissible cliques** (`N = 2` is the
  directed neighbor pairs), `build_clusters` reduces them to symmetry orbits via a
  canonical key built from the site-image map `(b, τ + W·R)`. An edge is admissible
  under `MinimumImage` iff it sits at its atom-pair minimum-image distance (and the
  clique's atoms are distinct, so a cluster never reuses an atom's image — that would
  alias a lower-body term); under `AllImages` iff within the radial cutoff. The
  `selection` is threaded from `SLCEBasis` so the neighbor list and clusters agree.

### SALC basis (M7) — arbitrary body order
- `build_salc_basis` projects each orbit's representative onto the trivial irrep of
  its site stabilizer. At `N ≥ 3` a stabilizer op can permute equivalent sites, mixing
  coupling paths and (for unequal `l`) `l`-orderings, so the projection runs over the
  **combined (ordering × path × `Mf`) space**: each op acts by rotating every site axis
  (`wignerD_real(l_i, R)`) and relabeling axes by `invperm(perm)`, with the action
  matrix read off by contraction against the orthonormal coupled tensors. Deterministic
  axis-pivoted gauge; per-ordering fold into a multi-term SALC (one `SALCTerm` per
  ordering). `SALCKey` (canonical, injective column address — `block` runs across split
  ordering orbits) + `SALCBasis` (sorted keys + fingerprint). `evaluate_salc(salc, e)`.
  Since M2a the key carries the **joint decoration label**: a sorted
  `Vector{SiteDecor}` (per-site `isbits` combination of at most one factor per
  channel — `SiteFactor(SPIN, 0, l)` / `SiteFactor(DISP, k, l)`, `Channel` order
  `SPIN < DISP < OCC` with `OCC` reserved; `basis/decor.jl`) plus the total spin
  rank `L_S` (spin-first coupling; a good quantum number of the projection).
  The pure-spin construction emits `decors = spin_decors(ls)` with `L_S = Lf`
  (the v4 shape); `spin_ls(key)` reads the `ls` label back. `SALCTerm` axes are
  `Slot`s (member-site index + `SiteFactor`, canonical order SPIN axes before
  DISP axes each by site) — the slot → site map that admits several axes on one
  site; pure-spin terms are the identity list `spin_slots(ls)`.
- **Mixed-channel engine (M2b-2)**: `_orbit_salcs_decors(crystal, sg, N,
  orbit_id, O, labels, soc, wcache)` projects explicit decoration labels on a
  cluster orbit — multiset arrangements split into site-permutation orbits,
  spin-first slot coupling (`L_S` per path via `_path_LS`), per-`(L_S, Lf)`
  block projection (`_project_and_fold_decors`; the blocks are exactly the
  L_S-block-diagonal structure of the grey-group projector), canonical-slot
  transport, Σl_spin-even label validation, `soc = false` ⇔ `L_S = 0` only.
  Both channels rotate through the single polar Wigner cache (parity theorem,
  design record §4); `l = 0` trace axes skip rotation. Joint evaluation
  `evaluate_salc(salc, e, u)`: spin axes `Z_{lm}(ê)`, displacement axes
  `|u|^{2k}R_{lm}(u)`, scale `(4π)^(n_spin/2)`; displacement-decorated SALCs
  vanish exactly at `u = 0`. Gates in `test/unit/test_mixedsalc.jl`:
  engines-agree on pure spin (anti-drift), gate (e) cubic 2/0, gate (g)
  count ≡ CountingOracle 9 on the doubly-decorated bond, gate (g2)
  chirality-twist `L_S = 1` sector, gate (i) core, mixed space-group
  invariance + grey-group T. The sector-spec surface driving this engine from
  `BasisSpec` is M2b-3. Normalization note: with the `(4π)^(n_spin/2)` scale a
  spin axis evaluates as `√(2l+1)·C_lm(ê)` while a Racah disp axis is
  `C_lm(u)|u|^{2k}` — "disp factors are scale-free" holds in the 4π sense only,
  and the per-slot `√(2l+1)` channel asymmetry biases un-standardized
  ridge/GAR penalties across channels (standardize columns, or fold the factor
  into per-sector penalty weights, when mixed fitting lands in M3).
- **Sector-table spec surface (M2b-3a)**: the exported `Sector(; spin, disp,
  soc, cutoff, sites)` sugar declares one truncation-table row (theory note
  §3/§7); `BasisSpec(labels; sectors, lmax, pmax, ...)` resolves it once into
  the dense canonical `SectorRule` (`spin_mode ∈ :none/:explicit/:any`, range
  tuples, resolved per-pair cutoff — validated, comparable, persisted), stored on
  the field `BasisSpec.sector_rules` (named apart from the `sectors` keyword so the
  sugar/resolved boundary is visible; the persisted key stays `"sectors"`). The
  admitted label set is the union of the rows ∩ the global per-species
  `lmax`/`pmax` and per-body `lsum` caps; `nbody` and the per-body `cutoff`
  envelope are derived (elementwise max over the sectors admitting each body
  order). `soc` is per sector (`false` ⇔ enumerate only `L_S = 0`; exact
  together with the even-`Σl_spin` screen). The dense pure-spin form replaces
  `isotropy` with the inverted `soc` flag (deprecation errors at the keyword,
  the TOML input key, and nowhere silently); `disp_scale` is fixed in the spec
  and persisted.
- **Sector-driven construction (M2b-3b)**: `build_salc_basis(crystal, sg,
  clusters, spec; neighbors, selection)` (`basis/sectorbasis.jl`) expands the
  resolved table per cluster orbit — sector cutoff re-admission with EXACTLY
  `candidate_clusters`'s banded semantics (MinimumImage gates the pair's
  minimum-image distance, AllImages each edge), label generation
  (`_spin_multisets` × `_disp_multisets` per total degree — the `(k, l)`
  enumeration IS the Sym^p restriction — married onto the orbit's sites by
  `_marry_multisets`, shared sites carrying both channels), per-label
  effective `soc` = OR over the admitting sectors, then the decor engine with
  per-species `lmax`/`pmax` permutation-orbit filtering (`_admit_assignment` —
  species are stabilizer-invariant, so the canonical representative decides,
  keeping `block` indices aligned with `_enumerate_ls`). Key uniqueness of the
  union is asserted (the key-union invariant). `SLCEBasis` routes: empty
  sectors → the pure-spin engine (bit-identical to before), else the sector
  path. The spin-configuration `SLCEDataset` path refuses a
  displacement-decorated basis at the boundary (the joint path goes through
  `TrainingDatum` vectors — M3 slice 3). Gates
  (`test/unit/test_sectorbasis.jl` + the build testset in
  `test_mixedsalc.jl`): sector-expressed pure spin ≡ legacy dense bitwise
  (incl. fingerprint, `lsum` global and sector-local, `soc` both ways), the
  builder ≡ engine bitwise on the gate-(g) bond content (9 SALCs), overlap
  soc=false ∪ soc=true idempotence, per-sector cutoff shell separation,
  ligand `pmax` slot placement, u = 0 exactness over a mixed build, and a
  mixed-basis persist round-trip.
- **Channel representation trait (M2b-3c)**: `rep_scale(channel, detR, l)`
  (`basis/decor.jl`, public unexported) is the single declared source of the
  per-channel O(3) action relative to the polar Wigner matrix — SPIN axial
  `det(R)^l`, DISP polar `1`, OCC reserved (throws). Production never applies
  it (the even-`Σl_spin` screen ⇒ `det(R)^{Σl_spin} ≡ +1`; design record §4);
  it seats the gate (o) representation pins and mutation teeth (M2d) and the
  oracle's independent derivation.
- **M2d verification gates** (test-only; closes the design record's §12 M2
  battery — every M2 gate (a)/(d)/(e)/(e2)/(f)/(g)/(g2)/(h)/(i)/(n)/(o)/(p)
  now runs in the suite):
  - gate (o) (`test_countingoracle.jl` + `test_mixedsalc.jl`): the op-by-op
    identity `D_spin(l, R) = rep_scale(SPIN, det R, l)·D_polar(l, R)` against
    the oracle's polynomial-derived matrices over O_h, the inversion pins
    (spin `+I` ∀l, disp `(−1)^l I`) on both the oracle and the production
    Wigner kernel, and two `rep_scale`-built mutation teeth, stated as
    effective reps on the polar product: `det^{Σl_all}·⊗D_polar` (every slot
    axial — identically the "global `det^{Σl_all}`" AND the "disp-as-axial"
    prose rules, since `det^{Σl_spin}·det^{Σl_disp} ≡ det^{Σl_all}`) and
    `det^{Σl_disp}·⊗D_polar` (spin polar × disp axial). Both shift the bent
    pair+ligand count 7 → 5; the all-axial rule also resurrects the
    kill-shot (0 → 1) while the other is blind there; the 9-bond
    (`Σl_spin` and `Σl_disp` both even) is blind to both. The
    production-relevant parity is the even-`Σl_spin` screen
    (`det^{Σl_spin} ≡ +1`), which is also why the two teeth coincide on every
    production-reachable label;
  - gate (p) (`test_mixedsalc.jl`): the full grey projector over every
    `(L_S, Lf)` path and every per-op `D(g)` are `L_S`-block-diagonal
    (< 1e-9), with full-space rank ≡ engine SALC count;
  - gate (e2) (`test_sectorbasis.jl`): THE B₁/B₂ gate — on an octahedral
    Fe(O)₆ unit under the full 48-op O_h group, the `l = 2` spin ×
    neighbor-shell `p = 1` sector emits exactly 2 SALCs (both `L_S = 2`,
    `Lf ∈ {1, 3}`, 6 physical bonds each); the affine substitution
    `u_j = ε·d_j` lands them EXACTLY (rel. residual < 1e-10) in the span of
    the two-constant cubic magnetoelastic forms
    `Σ_i ε_ii (n_i² − 1/3)` / `Σ_{i≠j} ε_ij n_i n_j` with both invariant
    directions realized independently (the B₁/B₂ normalization itself is a
    fit gauge, not pinned), and uniform shell translations and rigid
    rotations (the antisymmetric substitution) vanish identically;
  - gate (f) (`test_sectorbasis.jl`): on a mixed spec (soc-false ligand +
    bond-stretch sectors, soc-true doubly-decorated pair sector) the ligand
    sector emits exactly the one superexchange-path invariant
    `(ê₁·ê₂)(u_O)_y` (oracle-pinned 1/2/4 L_S-resolved), the dJ/dr
    bond-stretch sector exactly the oracle-pinned 2 SALCs spanning
    `(ê₁·ê₂)(u₁−u₂)_x` / `(ê₁·ê₂)(u₁+u₂)_y`, every `L_S = 0` SALC is
    global-spin-rotation invariant, every `L_S ≥ 1` SALC violates it (no
    DMI, operationally), and the `soc = true` flip adds exactly the
    `L_S ≥ 1` blocks bitwise-preserving `L_S = 0`;
  - gate (n) (`test_sectorbasis.jl`): masking the all-soc-true mixed build to
    `L_S = 0` ≡ the all-soc-false rebuild, bitwise;
  - gates (d)/(a) (`test_sectorbasis.jl`): dense ≡ `p = 0`-restricted sector
    build bitwise, down to `==` energy/torque design matrices and
    field-identical `spin_multipole_terms` (the MC consumption surface);
  - gate (i) full (`test_sectorbasis.jl`): the pure-spin subset of a mixed
    build IS the dense spin basis, `u = 0` joint evaluation `===` the dense
    evaluation, displacement-decorated SALCs exactly zero.
- **Synthetic recovery, plan A** (`test_recovery.jl`; engine level, zero core
  changes): Cartesian ground-truth spin–lattice models → sampled energies →
  OLS on the SALC design → held-out energies exact (< 1e-9 relative) AND the
  Cartesian input constants recovered through probe configurations. These are
  function-space gates — they pin the span, the selection rules, the
  projection, and the atom/component indexing, but NOT the per-SALC
  normalization (invariant under column re-mixing; the `(4π)^(n_spin/2)` /
  Racah scales stay pinned by the oracle and gates (e)/(g)). Four models: (A1) exchange striction `[J + a_s(u₁−u₂)_x + a_y(u₁+u₂)_y +
  a_O u_{O,y}]·(ê₁·ê₂)` on the bent Fe–O–Fe unit, with the DMI negative
  (planting `D(ê₁×ê₂)_z` makes the soc = false fit fail loudly, and the
  soc = true refit of the same data recovers `D` exactly); (A2) the B₁/B₂
  shell constants on Fe(O)₆ (per-bond C4v pair transported over the O_h
  shell); (A3) the chirality twist `K(ê₁×ê₂)·(u₁×u₂)` — all fitted weight on
  the `(L_S = 1, Lf = 0)` column (others < 1e-8 relative), `K` recovered via
  the twist ratio, and the `L_S = 0` basis demonstrably cannot express it;
  (A4) spin-dependent force constants `u_r·Φ⁰·u_r + (ê₁·ê₂)(u_r·Φ¹·u_r)` at
  `p = 2` with aligned/antialigned probes separating `Φ⁰` from `Φ¹`. Plan B
  (forces/torques-only recovery) lands with M3; plan C (bcc Fe literature
  parameters) is demo material.
- **Joint gradient kernel (M3 slice 1)**: the two-buffer
  `accumulate_grad!(Ge, Gu, salc, e, u, weight[, scratch])` accumulates
  `weight·∂Φ/∂e_a` (spin axes, tangent-projected — the torque convention) and
  `weight·∂Φ/∂u_a` (displacement axes, plain Euclidean — the force convention
  `f_a = −∂E/∂u_a`, design record §6) with the `(4π)^(n_spin/2)` scale of the
  joint `evaluate_salc`, sharing its channel-dispatched value tables (the
  scale expression, written in both kernels, is fenced by gate (j)). A
  displacement axis `|u|^{2k} R_{lm}` gets
  `|u|^{2k}∇R_{lm} + 2k|u|^{2(k−1)} R_{lm} u` (polynomial-exact, smooth at
  `u = 0`). On a pure-spin SALC `Ge` is bit-identical to the single-buffer
  spin-only path and `Gu` stays untouched; the spin-only path still refuses
  decorated SALCs, naming the joint form; passing one array as both buffers
  is refused. Gate (j) at engine level (`test_jointgrad.jl`):
  central-difference `∂Φ/∂u` (3-point where per-column degree ≤ 2, a
  degree-4-exact 5-point stencil for the quartic fixture that makes BOTH
  halves of the displacement product rule live — `(k, l) = (1, 2)/(2, 0)`
  factors) and tangent-directional on-sphere `∂Φ/∂e` FD, over the mixed
  9-SALC bond family, the `|u|²` trace channel, the 6-member Fe(O)₆ orbit
  (member transport), and an AllImages self-bond (both slots of each channel
  on ONE atom column — the repeated-column chain rule); the explicit tangency
  invariant `Ge[:, a] ⊥ e[:, a]`; pure-spin bit-identity; exact `u = 0`
  behavior (all-degree-1-pairs ⇒ both gradients ≡ 0; single degree-1 factor ⇒
  constant nonzero `Gu`); weight-0 fast path, accumulation additivity, shared
  scratch bit-identity, and the full error surface.
- **Force design block + three-block co-fit (M3 slice 3)**: against a
  displacement-decorated basis, the `TrainingDatum` dataset path evaluates
  `X_E`/`X_T` jointly at each config's `(e, u)` (spin-only datum ⇒ `u = 0`
  exactly, stored in `SLCEDataset.disps`; pure-spin columns bit-identical to
  the spin-only design) and builds the **compact** force block `X_F`: columns
  only the displacement-active SALCs (`force_cols`; `∂Φ/∂u ≡ 0` on pure-spin
  columns — the zeros are scattered in at assembly, never stored), rows only
  for force-bearing configs (`force_config`, the torque-block ragged
  contract) and displacement-referenced atoms (`_disp_referenced_atoms`;
  structurally zero rows excluded with a warning when their targets are
  nonzero). Entries are `−∂Φ/∂u` (the pinned force sign, applied only there
  and in `predict_force`). `fit(...; torque_weight, force_weight)` minimizes
  `(1−w_T−w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F` (`√(w_F/n_F)` whitening; `j0`
  energy-only; `w_F = 0` bitwise-identical to before); `refit`/`gcv`/
  `effective_dof` assemble with both weights; force diagnostics
  (`has_force`, `residuals_force`, `rss_force`, `r2_force`, `rmse_force`)
  mirror the torque block with uncentered baselines. Joint predicts
  `predict_energy/torque/force(model, e, u)`; the 2-arg forms refuse a joint
  model (no silent `u = 0`). The selection layer is force-aware: `select_fit` /
  `cross_validate` take `force_weight`, `select_support` reads it off the fit, and
  all three score `(1−w_T−w_F)·MSE_E + w_T·MSE_T + w_F·MSE_F`. Fold strata pack
  both derivative channels (`2·torque + force`, dealt in descending class order,
  so a force-free dataset degenerates to the torque-only deal bit-identically), the
  fold count is capped by each weighted channel's config count, and force presence
  enters the strata only at `w_F > 0` — at `w_F = 0` results are bit-identical to
  the pre-force-channel behaviour. Gate (j) at model level + synthetic recovery plan A:
  `test/unit/test_jointdata.jl`.
- **Periodic resolvability — the freeze (`basis/resolvability.jl`)**:
  `unresolvable_columns(basis)` names the coefficients no data on this reference cell can
  determine, and `build_asr` holds them at exactly zero (excluded from `free`, under
  `asr = false` too). A Wigner–Seitz boundary tie has **two faces** and both are frozen:
  (a) the point group permutes the tied images, so they share one orbit whose sum weights
  them equally and the odd content cancels — an identically zero column (`vanishing`);
  (b) in low symmetry no operation relates them, so they sit in *different* orbits with
  independent couplings, every column is nonzero, and only the SUM is determined — the whole
  interaction is dropped (`undetermined`: every column of every orbit sharing an atom
  multiset). Face (b) admits no justified split (the images share a phase only at
  `q = 0`), so dropping discards determined content on purpose: measured on a P1 fixture,
  18 columns = 9 determined sums + 9 undetermined differences, and `r2_energy ≈ 0.67` on
  data containing the shell — the intended loud failure. Classification is structural
  (`_signature_matrix`, the undifferentiated twin of the ASR expansion, judged per column
  against its own gross accumulation), never sampled, and short-circuits on a cheap
  cross-SALC pre-check (`_has_boundary_tie`); when face (b) fires the remaining structural
  rank is verified (`residual_flat`) rather than assumed. Real standard cells are
  unaffected — with space groups from Spglib the null-column count already equals the full
  structural nullity on bcc Fe / B2 FeRh / hcp Co / wurtzite GaN / rocksalt MnO. Gates (A)
  and (G), `test/unit/test_resolvability.jl`.
- **ASR — exact translation-invariance constraints (M3 slice 4)**: `A·β = 0`
  enforced by null-space reparameterization `β = Z·γ` (design record §6 +
  its 2026-07-26 amendments). Builder (`fitting/asr.jl`): translation
  generator applied symbolically to every displacement-decorated SALC —
  spin factors opaque symbols, displacement factors expanded to exact
  monomials by `SolidHarmonics.solid_harmonic_poly` (a coefficient-space
  rerun of the evaluator recurrences, kernel-gated) — collected in one
  common monomial basis across orbits, in design-column coordinates.
  Rank policy: row-normalized, exact connected components, per-component
  SVD with cut `1e-10·σ_max` and a forbidden band `[1e-12, 1e-8]·σ_max`
  (ambiguous rank errors); `Z` orthonormal, identity on pure-spin columns;
  `rank(A)` gated for exact equality against the numerical translation
  image through `accumulate_grad!`. The lift `γ → β` is one function
  (`_lift_gamma`, used by both `fit` and `refit`) and emits **exact zeros**
  on columns the constraint forbids: a dead row of `Z` is numerically
  (~1e-16), not structurally, zero, and the plain product would leave junk
  that `select_support`'s alive rule and SLCEMonteCarlo's `t.coef != 0.0`
  prune — both exact tests — read as a live term. Built once at `SLCEDataset`
  construction (`dataset.asr`, the `force_cols` discipline; `nothing` on
  pure-spin bases — bitwise-identity fast path, gated);
  `_assemble_problem` applies it per block. `fit(...; asr = true)` default;
  `SLCEFit` records the applied flag + achieved residual; nothing persisted
  (`asr_residual(model)` recomputes — the public verifier physical
  consumers gate on). β-space penalties compress to `Z'·D(β)·Z`
  (`solve_coefficients(...; nullspace)`; L1/GLMNet and `FixedCoefficients`
  reject); `refit` re-derives `null(A[:, S])` on its support;
  `dof = p − rank(A) + 1`, `effective_dof`/`gcv` whiten by Cholesky
  congruence, GCV's `n` excludes zero-weight rows;
  `select_fit`/`select_support` refuse to run *under* a reparameterization
  (unconstrained cached Gram + β-indexed alive rule), but `select_fit(...; asr =
  false)` selects a deliberately unconstrained joint model end to end.
  Gates (k): `test/unit/test_asr.jl` — symbolic ≡ numerical rank +
  subspace agreement, forbidden band, translation invariance and per-config
  `Σ_a f_a = 0` at physical `t` after `fit` AND after `refit`, the
  unconstrained-violation demonstration, zero-nullity truncation warning
  (`pmax = 1` pair splits), pure-spin bitwise identity, AllImages
  self-image refusal, and the same rank / invariance / `Σf = 0` battery at
  **3-body**, where one constraint row couples three site blocks (the
  third-order force-constant case; every other fixture is a 2-body bond).
- **Rotational invariance — measured, never imposed** (`slce/affine.jl`, design
  record §12 gates (q) + (r)). `affine_energy(model, spins, M; origin, base)` evaluates
  the model under `u(R) = M·(R − origin) + base[:, atom]` with `R` each cluster site's
  own equilibrium position, **image shift included** — the path `predict_energy` cannot
  express, since it resolves a site as `u[:, atom]` and so accepts only cell-periodic
  fields (translation is the one affine field that is; that asymmetry is why the ASR was
  testable through the ordinary predictors and rotation was not). Reduces to
  `predict_energy` **bit-identically** at `M = 0` — the gate that keeps it from becoming
  a second evaluator — and is `origin`-independent to `1e-15` under the ASR **to first
  order in `M` only**: the ASR is an identity in the atom variables, i.e. on cell-periodic
  fields, and at finite `M` the affine field is not one. Measured relative spread over four
  origins on ASR-clean models at `‖M‖ ~ 0.01`: 1.08 (wurtzite GaN), 1.69 (rocksalt MgO,
  where the rigid-rotation energy changes sign), 1.04 (B2 FeRh) — so
  `rotational_residual` is a function of `(model, origin)` and a single default-origin
  value can be a false negative (GaN: `5.5e-16` at the default against `1.27` about
  `(7, −3, 2.5)`, where the rotation genuinely costs `ΔE = 0.425`). The gap is the same
  home-image gauge `strain_derivatives` measures at `order ≥ 2`; the suite's dimer fixture
  (bond inside the cell) is the special case where it vanishes.
  `rotational_residual(model, spins;
  omega, axis, origin, u0)` = `|ΔE_rot| / RMS_6(ΔE_strain)` over the six
  Frobenius-normalized symmetric directions of the same `‖O − I‖_F`;
  `rotation_transfer_residual` = `|ΔE_joint| / (|ΔE_lattice| + |ΔE_spin|)`, which needs
  no external scale because the halves need not share their leading power of `ω`.
  Nothing is constrained: a continuous rigid rotation is not a space-group operation, so
  the SALC projection cannot remove it, and Born–Huang is independent of both the ASR and
  the 15 Huang conditions. Gates (`test/unit/test_rotation.jl`): the `M = 0` bit-identity
  and error surface; ASR origin-independence *with* its unconstrained counterexample;
  the **blind linearized test** pinned as an exact zero against the hand-derived
  `ΔE = 1 − cos ω` of a radial-force dimer; a central-pair-potential positive control
  (`~10⁻³`, falling with `ω`) against a random ASR-feasible model (`~1.7`, flat); the
  home-**image gauge** (same crystal, same periodic data, 1349× the residual on a stressed
  dimer — so a future constraint matrix is not a function of the model alone); and gate
  (r) on a pseudo-dipolar dimer, lattice-only `1.67` vs joint `1.5·10⁻⁴`.
- **Downstream term contract — M4 slice 1**: `DecoratedTerm` / `decorated_terms(model)`
  (`slce/introspect.jl`) — one term per member and slot layout, each tensor axis
  labelled by a `SLCE.Slot` (member-site index + `(channel, k, l)` factor), with
  the consumer scale `(4π)^{n_spin_slots/2}` carried as a field (single definition
  `_slot_scale`; NOT `(4π)^{body/2}`, which agrees only when every site holds exactly
  one spin factor). `spin_multipole_terms` stays the frozen p = 0 view and refuses any
  displacement model on the SPEC trigger `_basis_has_disp`, naming both hatches.
  `restrict(model, :spin)` = the clamped-ion sub-model (pure-spin SALCs + `_spin_spec`:
  `pmax` zeroed, sectors cut to their degree-0 row).
  **`keep_zero` — added 2026-07-29, and it is a correctness keyword, not a convenience.**
  Both surfaces prune SALCs whose coefficient is exactly zero, so the emitted list — and
  the index → SALC map a consumer addresses it by — is a function of the coefficient
  VALUES, not of the basis. Harmless for a reader, wrong for a REWRITER: sparse
  estimators and `refit` produce exact zeros routinely, so two models on ONE basis (two
  points of a `StrainedModels` grid, an active-learning refit) can emit equal-length lists
  whose maps are shifted, at which point `SLCEMonteCarlo.set_coefficients!` writes each
  coefficient onto a neighbouring cluster with every length check passing. This is the
  other half of the M5-4 blocker whose MC-side half (`keep_zero_terms`) was resolved
  2026-07-27 — the flag froze the support of an ALREADY-pruned list. `keep_zero = true`
  emits one term per SALC member unconditionally, making the map a property of
  `salc_basis`, which is exactly what a volume grid asserts identical across its points.
  Default `false`: every existing consumer, byte-comparison and benchmark is untouched.
  Gates
  (`test/unit/test_introspect.jl`): independent slot-by-slot reconstruction of
  `predict_energy(model, e, u) − j0`, the shortcut-scale reconstruction FAILING on the
  same fixture (teeth), pure-spin agreement with `spin_multipole_terms`, the refusal surface
  incl. the all-zero-displacement-coefficient case, and bitwise `restrict` ≡ joint at
  `u = 0` plus a persistence round-trip. Docs: `guide/introspection.md` (scale table +
  the mandatory `restrict ≠ refit` box with a measured comparison, design record §13
  risk 5).
- **Lattice dynamics — M4 slice 2** (`slce/forceconstants.jl`):
  `force_constants(model; spins, order = 2)` → `ForceConstantSet` — exact order-`n`
  displacement derivatives at `u = 0` with the spins fixed (only terms whose
  displacement degrees sum to `n` survive; contributions read off
  `SolidHarmonics.solid_harmonic_poly`, shared with the ASR builder). Keys are the
  lattice-dynamics convention `Φ[(a,0),(b,R),…]`, one per ORDERED index tuple,
  anchored on the home cell; the reverse ordering is a separate key equal to the
  transpose. `dynamical_matrix(fcs, q; masses)` = `Σ_R Φ(R) exp(2πi q·R)/√(MₐM_b)`
  with `q` in fractional reciprocal coordinates, rows atom-major / Cartesian-minor.
  Gates (`test/unit/test_forceconstants.jl`): Γ sum ≡ finite-difference Hessian,
  transpose relation, `D(q)` Hermiticity + `D(−q) = conj D(q)` + mass weighting,
  **three zero eigenvalues of `D(0)` for an ASR model and none for an unconstrained
  one** (the physical payoff of gate (k)), spin dependence, order 3 ≡ third
  derivative, and the empty-result cases.
- **Homogeneous-strain response — M5-2 slice 1** (`slce/strain.jl`):
  `strain_derivatives(model; spins, order = 1, origin, symmetrize = true)` → the exact
  `∂ⁿE_cell/∂ε_{α₁β₁}⋯` at the model's own reference `ε = 0`, a rank-`2n` array indexed
  with the Cartesian pairs adjacent. The **fourth** consumer of
  `SolidHarmonics.solid_harmonic_poly` (after the ASR builder, the force constants and
  `effective_model`), and the second consumer of the affine path: a strain field is not
  cell-periodic, so `predict_energy` cannot reach it. The mechanism is one chain rule
  applied to the force-constant machinery —
  `∂ⁿE/∂ε^n = Σ_{s₁…s_n} [∂ⁿE/∂u_{s₁α₁}⋯] d_{s₁β₁}⋯` over MEMBER SITES with
  `d_s = R_s − origin` that site's own position, image shift included — so
  `_accumulate_strain!` reuses `_fill_fcs_tensor!` verbatim and only the last step (contract
  with positions, instead of storing under a lattice-dynamics key) is new. Because a term of
  displacement degree `n` becomes a degree-`n` polynomial in `ε` exactly, the order-`n`
  derivative draws from degree-`n` terms alone and the Taylor series is exact at FINITE
  strain, which is what makes the gate sharp.
  Measure = **Biot / Seth–Hill m = 1** (`F = I + ε`, design record §9e); `order = 1` is
  `V σ` (measure-independent), `order = 2` the **clamped-ion** `V C` (not). Both depend on
  `spins` — the trap where they cannot is warned about through `_spin_blind_at_order`,
  the predicate now shared with `force_constants`' `_warn_spin_blind` (one predicate,
  two messages, because the advice is opposite: `degree = 1` is what magnetoelasticity
  NEEDS and what harmonic constants must not stop at). `_resolve_spins` is likewise
  shared, so the spin-free entry rule stays one rule.
  **The ASR is a hard precondition (§9d, gate (t)):** an origin shift adds the uniform
  translation `ε·t`, changing the energy by `ε : (Σ_i ∇_i E)`, so without `A·β = 0` the
  quantity is *undefined* rather than inaccurate — `strain_derivatives` **throws**, at
  `_STRAIN_ASR_RTOL = 1e-12`, deliberately tighter than the 1e-10 used elsewhere because
  the strain path weights the residual by `|R_i|`. `symmetrize = false` returns the raw
  derivative with respect to a general affine map, whose discarded antisymmetric part is
  measured content (`rotation_transfer_residual`, gate (r)), not noise.
  **…and at order ≥ 2 it is not sufficient.** The ASR is an identity in the ATOM
  variables, i.e. on cell-periodic fields; at `ε = 0` the affine field is zero, hence
  periodic, so order 1 is origin-independent unconditionally, but one order out the
  per-SITE identity would be needed and is not implied. The gap is gate (q)'s
  **home-image gauge**, now with teeth on a physical output: content anchored at an
  atom's home representative cannot cancel against partners the clusters reach at a
  different image, leaving the per-cell energy with a piece linear in the cell's
  position. Measured on ONE dimer chain in two descriptions — bonded partner at home ⇒
  1.4e-13, bond crossing the cell edge ⇒ factor 8; a bcc-like cell ⇒ ~25. So order ≥ 2
  **measures** it (recompute at a probe-shifted origin, `_STRAIN_ORIGIN_RTOL = 1e-8`,
  refuse a disagreement naming the crystal DESCRIPTION as the fix; `check_origin = false`
  for diagnosis only) instead of shipping a description-dependent elastic constant behind
  a caveat.
  Gates (`test/unit/test_strain.jl`): the Taylor identity
  `E(0) + D¹:M + ½D²:M:M ≡ affine_energy(model, e, M)` at `t` up to `−3.0` (two
  independent implementations — monomial coefficients vs. the production
  `_eval_term_mixed` — agreeing to 1e-12 at 100%-scale affine maps, not asymptotically);
  gate (t) origin invariance at three origins to `1e-11·‖D‖` **plus** the refusal on an
  unconstrained model, paired with a demonstration that `affine_energy` itself moves with
  the origin there; the home-image gauge at order 2, as the SAME chain in two
  descriptions (one answers, one is refused by name, and the escape hatch shows the
  refused number differs by > 10%); the degree filter (`degree = 2` only ⇒ `D¹ ≡ 0`, `degree = 1` only ⇒
  `D² ≡ 0`, pure spin ⇒ both); pair/mixed-partial symmetry of `D²`; the magnetic-state
  dependence and its silent-trap warning; and the error surface.
  **What it is not, yet:** at `ε = 0` there is one strain derivative. On a volume grid
  there are two that must agree — this intra-model incremental one and the grid finite
  difference — and their agreement is `StrainedModels`' acceptance gate (M5-3), which
  cannot run until a grid exists.
- **The ε-linear magnetoelastic tier — M5-2 slice 2** (`slce/magnetoelastic.jl`): the two
  deliverables that ride on `strain_derivatives(order = 1)`, and on nothing above it. The
  tier is ε-linear for two INDEPENDENT reasons and the file says both: §13 risk 2 (only
  ε-linear content is Seth–Hill measure-independent unconditionally; second-order elastic
  constants agree across measures only at a stress-free reference, which a spin–lattice
  cell can be for at most one magnetic state) and the slice-1 finding (the ASR buys origin
  independence only where the affine field is periodic).
  - `magnetoelastic_constants(model; signs, tol)` → `(; B1, B2, ion = :clamped, residual,
    volume)`. **This is where §12 gate (u) closes.** The pinned convention, stated in the
    docstring and gated in `test/unit/test_magnetoelastic.jl`:
    `E_me/V = B₁ Σ_i ε_ii(α_i² − 1/3) + 2B₂ Σ_{i<j} ε_ij α_i α_j` — TENSOR shear, that
    range, that sign, `E_me` an energy DENSITY (so B carries energy/volume). None of those
    was pinned anywhere before: gate (e2) fixes the magnetoelastic block's SPAN and says in
    its own comment that "C2 is the fit gauge — no B₁/B₂ normalization or sign convention
    is pinned here". `ion = :clamped` is a FIELD, not prose, so `result.B2` cannot be
    quoted without it.
    Extraction is a **projection, not a readout**: the exact `D⁽¹⁾(α)` is computed at 19
    deterministic directions (3 axes + 6 face + 4 body diagonals + 6 generic — the generic
    ones are what `l = 4` content fails on, since a high-symmetry-only set can be blind to
    it) and least-squares-fitted to
    `D⁰_ij + V B₁ δ_ij(α_i² − 1/3) + V B₂(1 − δ_ij) α_iα_j` with `D⁰` free. Reading two
    directions would return a number for a model that is not of this form; the projection
    returns `residual` instead — the fraction of the α-DEPENDENT response left unexplained
    (normalized by the α-dependent part, never by `‖D‖`, since the reference stress can
    dwarf the magnetoelastic signal). The warning above `tol` is deliberately NOT
    `maxlog`-limited (the `fit.jl:180` reasoning: it reports one model's property).
  - `exchange_strain_derivatives(model; origin, check_origin)` → `ExchangeStrainDerivatives`
    (`pairs[(a,b,R)][α,β,γ,δ] = ∂M_ab^{αβ}/∂ε_{γδ}`, `onsites[a]` likewise, `skipped`,
    `origin`). The model's `dJ/dr` resolved per bond and per strain component — the
    ε-linear derivative of exactly the matrices `bilinear_terms` reports, from terms with
    two `l = 1` spin factors (or one `l = 2`) plus exactly one `degree = 1` displacement
    factor. `γ, δ` are UNSYMMETRIZED (a general affine map), matching
    `strain_derivatives(symmetrize = false)`. `skipped` reports only what a user would
    actually miss — SALCs with ε-linear displacement content whose spin part is not
    bilinear-representable; pure-spin SALCs and `degree = 2` factors are absent from a
    first derivative BY DEFINITION and are not losses. The `(4π)` scale is
    `count(has_spin, decors)/2`, never `_bilinear_terms`' pure-spin `body/2` shortcut.
    **The per-bond split is origin-dependent in general and is therefore measured.** The
    ASR cancels the origin from the TOTAL, not bond by bond: a bond whose displacement
    content is not purely relative carries an absolute position. Symmetry usually rules it
    out (a bond orbit with a site-swap operation admits only `u_b − u_a`), which is why the
    check passes silently on every fixture tried, including the split-home `±1/6` chain —
    but it is recomputed at a probe-shifted origin and refused on disagreement, the same
    idiom `strain_derivatives` uses at order ≥ 2.
  Gates (`test/unit/test_magnetoelastic.jl`): gate (u) against a closed form written out
  by hand and evaluated through `evaluate_salc`/`_eval_term_mixed` on the (e2) octahedron
  — two implementations, one number; the convention itself re-derived longhand
  (`2·Σ_{i<j}` typed as a literal) against the production energy at finite (α, ε), with
  shear-only and hydrostatic special cases isolating the factor 2 and the `−1/3`; the
  `l = 2` time-reversal parity under a sublattice flip; `residual` > 1e-3 with the warning
  on the SAME crystal stretched to tetragonal; and for the bond derivatives the
  RECONSTRUCTION identity — contracting every bond with a spin configuration rebuilds
  `strain_derivatives(order = 1, symmetrize = false)` to 1e-11, the `_reconstruct_energy`
  fence one derivative up.
  The (e2) fixture needed two changes to carry a MODEL: the centre is displacement-active
  and a pure-lattice `degree = 2` sector rides along, because a basis whose every SALC is
  individually translation-invariant makes `A ≡ 0`, which `build_asr` refuses as a broken
  expansion — a narrow but real over-strictness of that guard, recorded here rather than
  fixed under the milestone.
- **Magnon–phonon vertices — M5-2 slice 3** (`slce/magnonphonon.jl`):
  `magnon_phonon_vertices(model; spins)` → `MagnonPhononVertices`
  (`vertices[(a, b, R)][α, β] = ∂²E_cell/∂u_{aα}(0) ∂e_{bβ}(R)`), §7's third named
  deliverable. Structurally `_fill_fcs_tensor!` with one axis moved from the "evaluate"
  column to the "differentiate" column: the single `degree = 1` displacement factor gives
  `α` through the same monomial coefficients, and each spin slot in turn is differentiated
  by `Harmonics.grad_Zlm` (product rule — a term with two spin factors on one atom
  contributes twice) while the rest evaluate to numbers.
  **The spin index is Cartesian AND unambiguous**, because `grad_Zlm` is the TANGENTIAL
  gradient (the same object the torque design matrix is built from): `V·ê_b ≡ 0` holds
  identically, so no local-frame `(f₁, f₂)` convention is invented and the caller projects
  onto their own magnon basis. The pair is ORDERED — `a` displaced, `b` magnetic — unlike
  `bilinear_terms`' undirected bonds, so `(a,b,R)` and `(b,a,−R)` mean different things.
  Contributing content is exactly "one degree-1 displacement factor + at least one spin
  factor": a magnetoelastic sector declared at `degree = 2` yields NO vertices (it feeds
  the spin-dependent force constants), which is `_warn_spin_blind`'s trap from the other
  side. No ASR precondition — no absolute position enters — but the translation sum rule
  `Σ_{a,R} V = 0` holds for a constrained model and is gated.
  "Adiabatic" (the §7-mandated docstring word) is a SCOPE statement: derivatives of the
  static surface, so retardation, the Berry-phase phonon-angular-momentum term and
  spin-lattice relaxation are outside a static CE by construction — the same free scope
  line §7 draws for the multipole readout.
  Gates (`test/unit/test_magnonphonon.jl`): the Γ-folded vertices against a MIXED central
  finite difference of `predict_energy` (displacement component × tangential spin
  direction, all `a, b, α`); tangency `‖T·ê_b‖ < 1e-12`; the translation sum rule; the
  degree-2 ⇒ empty / lattice-only ⇒ throw content rules; and the shared `_resolve_spins`
  error surface.
- **Volume grids — M5-3** (`slce/strainedmodels.jl`): `StrainedModels(models, scales;
  abscissa = :volume, degree, tol)` + `model_at(sm, s)` + `grid_strain_derivative(sm, s;
  spins)`, the K(ε) container of design record §9a. One linear scale `s = 1 + η` labels a
  point, `A_i = s_i·A₀` with the fractional basis untouched — a SIMILARITY, which is what
  keeps the space group (`A R_frac A⁻¹` is invariant under `A → sA`), the orbits, the keys
  and the folded tensors identical. `scales` is REQUIRED and not inferred: the volumes give
  only ratios, and nothing in a set of cells says which one is `η = 0`.
  Six construction-time refusals, the last two beyond what the record listed: non-similar
  cells; cutoffs that do not scale (**both** `BasisSpec.cutoff` and every
  `SectorRule.cutoff` — §9a's key-stability pin); a moving `disp_scale` (asserted now
  though `≠ 1` is still refused elsewhere, so it stays true the day that lifts); a different
  `SALCKey` set; a different MEMBER (atom, shift) structure — **the home-image condition**,
  which the key set is blind to (slice-1 consequence (α)); and a different **SALC GAUGE**,
  compared on the folded tensors, because the projector fixes each invariant only up to a
  sign and a flipped one would be interpolated against the others silently, with every key
  matching and every per-point fit perfect.
  `model_at` builds the basis at an interpolated scale by SURGERY (scale the cell and both
  cutoff surfaces, carry the space group and the SALC basis over) — licensed by the
  similarity, gated against a basis BUILT at that scale (cell, keys, members, folded
  tensors). Coefficients are interpolated by a centered/scaled Vandermonde (a raw-volume one is
  unusable by degree 3) in `abscissa ∈ (:linear, :volume, :logvolume)` — a MODELLING choice
  (gate (w)), hence a field — at `degree = n-1` (exact through the nodes) or lower (least
  squares).
  **Gate (w) closed 2026-07-29, and it moved the default.** Leave-one-out against a directly
  fitted point at the omitted scale: `:linear` 6e-14, `:logvolume` 3e-10, `:volume` 2.5e-8.
  Not a tie and not a fixture accident — the map relating two grid points is the §9d
  re-expansion, exactly polynomial in the affine displacement and hence in `s`, of the same
  degree as the displacement content. So on a CE-representable surface the coefficients ARE
  polynomials in `s` (degree 2 already interpolates exactly on the fixture, and raising the
  degree buys nothing) while in the volume they are polynomials in `s³`, where the error
  falls monotonically with degree — the two abscissas make the point from opposite sides,
  and both arms are gated. Default therefore `:linear`; `:volume` is the right choice when
  the object of interest is an equation of state rather than a coupling.
  **The acceptance gate, and the two things building it caught** (`test/unit/test_strainedmodels.jl`).
  The gate is `strain_derivatives(model_at(sm, s); order = 1)` traced ≡
  `grid_strain_derivative(sm, s)`. (i) **Basis truncation, exactly as §14 predicted.** A
  grid's basis must be CLOSED UNDER RE-EXPANSION: a point at scale `s` is the reference
  re-expanded around the scaled geometry, and that shift is lower-triangular in degree, so
  `degree = 2` content needs a `degree = 1` sector and spin-dressed `degree = 1` content
  needs a PURE-SPIN one. Measured on the fixture: 5% off with the pure-spin sector missing,
  1% with the lattice `degree = 1` missing, 4e-7 with both. (ii) **A mislabelled η is a
  factor of exactly `s`.** `dE/dη = s·dE/ds`, because η is measured from the reference the
  derivative is taken AT (`strain_derivatives`' own convention — a model knows only its own
  reference) while the grid parametrizes total scale. Dropping it agrees at `s = 1` and is
  off by the strain everywhere else. The residual 4e-7 is the coefficient interpolation and
  nothing else, established by halving the grid span and watching it fall.
- **Effective models at a displaced structure — M5 slice 1** (`slce/effective.jl`):
  `effective_model(model; u0, atol = 0.0)` → `EffectiveModel` — the same coefficient
  set re-expanded around `R + u0`, EXACTLY (design record §9d). Every displacement
  site factor is homogeneous, so the shift `u → u0 + δu` is a finite linear map on
  the coefficients, realized by binomially expanding each monomial of
  `SolidHarmonics.solid_harmonic_poly` (the same function the force constants and the
  ASR builder read) and multiplying the shifted polynomials across a term's
  displacement slots. The result is deliberately **not** a `SLCEModel`: the displaced
  structure's symmetry is the stabilizer of `u0`, generally a proper subgroup, so the
  reference SALCs cannot span it — `EffectiveModel` is the unsymmetrized
  decorated-monomial form the design record permits, one `EffectiveTerm` per (spin
  factors) × (displacement monomial), evaluated by
  `predict_energy(em, e, δu)`. **`EffectiveTerm.scaled_coef` takes the OPPOSITE scale
  convention from the other two public term views**, which is why it is not spelled
  `coef`: `SpinMultipoleTerm.coef` and
  `DecoratedTerm.coef` are the raw fitted `jϕ` (scale left to the consumer, or shipped
  in `DecoratedTerm.scale`), while `EffectiveTerm.scaled_coef` has `(4π)^{n_spin_slots/2}`,
  the SALC `folded` weight and the shifted polynomial coefficient already folded in —
  necessarily, since one term merges contributions from many SALCs and there is no raw
  `jϕ` to return. Re-applying `_slot_scale` to it double-counts `4π` per spin slot. **Variables are per ATOM, not per slot** (displacements
  are cell-periodic, so an `AllImages` self-bond puts two slots on one variable);
  keying by slot would give the same energy but a non-canonical term list. Gates
  (`test/unit/test_effective.jl`): the identity `E_eff(δu) ≡ E(u0 + δu)` at small AND
  large `u0` measured against the sample's ENERGY SCALE (a random configuration can
  sit at a zero crossing, where pointwise relative accuracy does not exist —
  measured 3e-15 scale-relative against 7e-13 pointwise), `u0 = 0` reproducing the
  original surface, `δu = 0` being the `u0`-frozen and NOT the clamped-ion point, a
  central difference of the re-expanded surface reproducing the model's analytic
  `predict_force` at `u0`, the degree structure (lower triangular: `{0,2} → {0,1,2}`,
  never above the original maximum, and degree 1 is precisely the `u0` signature),
  the spin-only coefficients being RENORMALIZED rather than added to (the SCP
  template), canonical/sorted/reproducible terms, the same-atom merge, `atol`
  pruning breaking exactness on purpose, and the refusal surface — including
  `strain`, which throws rather than accepting the non-cell-periodic affine pattern.
- **Staged (hierarchical) fitting — M3 slice 6 (closes M3)**: `fit(...; frozen,
  sector_mask)` (`fitting/staged.jl`). `SLCE.sector_columns(basis, selector)` —
  `:all` / `:spin` / `:lattice` / `:coupled` (channel partition) and `:soc_free` /
  `:soc_only` (`L_S` partition), unions, explicit columns, `Bool` masks; `:soc_free`
  shares `SLCE.is_soc_free` with the `Sector(soc = false)` truncation (anti-drift
  gate: masked keys ≡ SOC-less rebuild's keys). `frozen::SLCEModel` matched by
  `SALCKey` (orphan nonzero key ⇒ error); free-column frozen values ignored; `j0`
  never frozen. One affine reparameterization carries both: `Z` zero-rowed on
  frozen columns (a plain selection matrix when there is no ASR — still
  orthonormal, so the estimator contract holds), `beta_p` = frozen values +
  particular solution of `A_free·β_free = −A_frozen·β_frozen`; homogeneity judged
  by the relative `asr_residual` measure (`_STAGE_HOMOGENEOUS_RTOL`), infeasible
  freeze refused with the straddling rows named. `_assemble_problem` applies the
  offset target-side (`ỹ = y − X_β·beta_p`) per block, before centering.
  `SLCEFit.reparam` records what was solved under; `dof` = stage free parameters,
  `refit` stays inside the stage, `cross_validate` threads the plan,
  `select_support` rejects staged fits. Gates: `test/unit/test_staged.jl`
  (selectors + anti-drift, mask-only ≡ untouched bitwise, staged ≠ joint, exact
  recovery when the frozen values are exact, chain homogeneity + whole-model
  translation invariance, affine path with a hand-frozen violating coefficient,
  channel masks straddle 0 rows vs `L_S` masks 54/180, consumer follow-through).
- **Identifiability diagnostics + derivative-only recovery (M3 slice 5)**:
  `identifiability(fit_or_dataset; rtol) -> (; ncols, rank, nullity, sigma_min,
  sigma_max, sigma_cut, gap)` — the numerical rank of the assembled design in the
  coordinates the estimator solves (`q` under the ASR) at `min(size)·eps·σ_max`,
  with `gap` (smallest kept / largest dropped σ) exposing an ambiguous decision;
  `O(n·q²)` and opt-in; the dataset method validates exactly like `fit` (shared
  `_validate_fit_request` / `_resolve_asr_rep`). `fit` carries the `O(n·q)`
  per-column half as a standing warning on design columns whose norm is
  negligible (relative cut `1e-12`, the ASR convention — `iszero` has measured
  false negatives). Recovery plan B (`test/unit/test_identifiability.jl`):
  forces + torques alone (zero energy weight) recover a known joint model to
  ~1e-15 under the ASR from center-of-mass-free samples, versus a 0.97-away
  model at equal derivative residuals without it; measured rank ledger —
  torque-only deficiency `= rank(A) + dim{spin-free feasible}` (`= rank(A)` only
  when all displacement content is spin-decorated; a lattice-only sector leaves
  a residue the ASR cannot cure, gated on a second fixture), force-only smaller
  (plus the zero pure-spin columns), plan-B co-fit deficient unconstrained /
  full rank `q` constrained with `rank([X; A]) = p`, and no deficiency at all
  under drifting displacements (design record §6 amendment 9, corrected there).
- Validated by the ground-truth tests with non-collinear spins, **all `Lf`, all body
  orders**: space-group invariance `Φ(g·e)=Φ(e)`, time-reversal evenness, linear
  independence; projector eigenvalues exactly 0/1. Improper-op parity is handled
  automatically (site-axis rotation by full `R` + even `Σl`). Cross-validated against
  Magesty: per-`(body, ls, Lf)` invariant-subspace dimensions agree through 3-body.

### fitting + SLCE API (M8, M9)
- `BasisSpec` (validated: `nbody ≥ 1`, symmetric per-body `cutoff` matrices with
  entries `≥ 0` or `Inf`, `lsum ≥ 0` per body order, nonempty `lmax`/`pmax`
  with entries ≥ 0, `disp_scale` finite positive, sector-rule consistency —
  see the M2b-3a bullet for the sector-table form and the `isotropy → soc`
  replacement), `SLCEBasis` (carries its spec in the `spec` field), `SLCEDataset`
  (energy design matrix `X_E`, and the
  torque design matrix `X_T` via the four-argument [all configs] or five-argument
  [mixed: `torque_sel` names the torque-bearing configs, rows for the rest are
  **excluded**, never zero-padded] forms; the per-row config index `torque_config`
  is stored and read by every consumer — slicing, `vcat`, `_assemble_problem`'s
  grouped-CV labels — instead of a uniform-block assumption; the force block
  `X_F`/`y_F`/`force_config`/`force_cols` follows the same stored contract in
  compact column form (M3 slice 3 bullet above); supports `length`,
  configuration slicing `dataset[idx]` — integer/`Bool`/`:` — and `vcat` of
  same-fingerprint parts including mixed torque/force presence, all without
  recomputing design rows; the
  `TrainingDatum`/source path rejects a zero moment on a basis-referenced atom —
  spin-referenced in the channel-aware sense: a displacement-only ligand site is
  exempt),
  `SLCEModel`/`SLCEFit`
  (plus the public constructor `SLCEModel(basis, j0, jphi)` for synthetic models —
  keys filled in from the basis),
  `fit(SLCEFit, dataset, estimator; torque_weight, force_weight)`,
  `refit(f, estimator; threshold)`
  (re-solve on the scaled-magnitude support of `f` — the de-biasing step after a sparse
  fit; shares `_assemble_problem` with `fit`),
  `predict_energy`/`predict_torque`/`predict_force`,
  `coef`/`intercept`/`nobs`/`dof`/`r2_energy`/`rmse_energy`/`r2_torque`/`rmse_torque`/
  `rss_energy`/`rss_torque`/`residuals_energy`/`residuals_torque`/`has_torque` (energy and
  torque blocks reported separately; `SLCEFit.residuals` stores the energy residual).
  The generics `coef`/`fit`/`nobs`/`dof`/`coeftable`/`islinear`/`predict`/`residuals`/`r2`
  **extend StatsAPI** (imported, not shadowed); `predict`/`residuals`/`r2` are thin
  wrappers defaulting to the energy block (`predict_energy`/`residuals_energy`/`r2_energy`).
  `AbstractEstimator` with `OLS`/`Ridge` (validated: `lambda` finite and ≥ 0)/`AdaptiveRidge`
  (analytic `j0`; centered-`X` `solve_coefficients` contract).
- Validated: OLS recovers an in-span target (R²=1); **a Heisenberg chain fit
  recovers `J = 2√3·jphi` to rtol 1e-8** (the v0 done-line, oracle); torque is the
  exact derivative of the energy surface (on-sphere finite differences, equivariance,
  Heisenberg closed form), energy+torque co-fit recovers an in-span model.

### persistence + TOML input (M10)
- **Persistence** (`io/persist.jl`): `SLCE.save(path, x)` and
  `SLCE.load(SLCEBasis | SLCEModel, path)` serialize a self-contained,
  human-readable **TOML** document — the crystal, the space-group ops, the basis spec
  (document key `"spec"`; schema version 2 renamed it from `"interaction"`, version 3
  stores the resolved truncation — per-body `cutoff` matrices, `lsum`, labels — and
  still reads v2's scalar `pair_cutoff`; version 4 stores SALC members in the
  canonical duplicate-free form — one member per physical cluster instance, up to
  `N!`× smaller — and folds v2/v3 members on load; version 5 also stores the
  sector-capable spec layout — `soc`, `pmax`, `disp_scale`, `sectors` — and
  reads legacy `isotropy`-keyed spec docs via `soc = !isotropy`),
  and the *full* SALC basis (every member / term / folded tensor); a model adds `j0`
  and per-`SALCKey` coefficients. Reload reconstructs the basis verbatim (no
  re-projection) and re-pairs coefficients to the basis **by key**, not by position.
  The `struct ⇄ Dict` schema (`_to_doc` / `*_from_doc`) is format-agnostic and tested
  without any serializer; TOML is stdlib (no dep) and round-trips `Float64` exactly.
  `save` / `load` are **unexported** (call qualified) to avoid clashing with
  `FileIO`/`JLD2`.
- **TOML input** (`io/input.jl`): `read_setup(path) -> (; crystal, spec, backend, tol, images)`
  and `SLCEBasis(path::AbstractString; backend, tol, images)` build a basis from a human-authored
  `input.toml` (`[structure]` inline crystal, `[interaction]` with optional `images`
  (`"minimum_image"` default / `"all_images"`) and `cutoff = inf` for the full WS
  cell, optional `[symmetry]`);
  keyword arguments override the file's backend/tol. Training data and the estimator
  stay in Julia (mirrors the basis/data separation).
- **Tabular results** (`slce/coeftable.jl`): `coeftable(fit | model) -> SLCECoefficients`
  is a **Tables.jl** source — one row per SALC (`body`, `orbit_id`, `ls` as a comma
  string, `Lf`, `block`, `J`) — so it drops into `DataFrame` / `CSV.write` /
  `Arrow.write`. The library owns the internal-storage → labeled-row mapping; the caller
  brings the table/IO package. `j0` is the intercept (`intercept(c)`), not a row.
  Tables.jl is a lightweight core dep.
- **DFT data sources** (`io/dftsource.jl`): the **code-agnostic boundary only** —
  `TrainingDatum` (per-configuration observables: required energy + spin channel,
  optional displacements / forces / constraining field with the derived torque
  target `τ_a = m_a × B_a`, the physical / Landau–Lifshitz torque; `nothing` =
  not observed, distinct from an observed zero) with its `DatumProvenance`
  (`torque_qualified` gates X_T rows, auto-derived from the field; `setup_id`/`soc`
  enforce one computational setup per dataset; `reference_id` +
  `reference_fingerprint` = `crystal_fingerprint`, a hand-rolled stable FNV-1a over
  the canonical crystal serialization, pin the clamped-ion reference for any
  displacement-decorated basis — the double-counting protocol as an invariant).
  `spin_datum(energy, moments[, field])` survives as the spin-only convenience
  constructor (the 2-arg form covers collinear/Ising and torque-less codes), and
  `read_configs(src::AbstractDFTSource) -> Vector{TrainingDatum}`, with `SLCEDataset(basis, src)`
  going source → dataset. Against a displacement-decorated basis the
  `TrainingDatum` path is the joint entry point (`use_torque`/`use_force`
  channel switches; M3 slice 3 bullet above). The **concrete per-code adapters live in the `SLCETools.jl`
  package** (`SLCETools.VASP`: `read_poscar`/`write_poscar`, `Oszicar` with SAXIS rotation,
  and the INCAR writer), not in the core. Adding a DFT code is one sibling adapter there —
  neither the core nor its export list changes; the VASP parsers are cross-checked bit-for-bit
  against Magesty in SLCETools's oracle. The one in-core concrete format is Magesty's
  code-agnostic EMBSET training set (`io/embset.jl`): `read_embset` + the `EmbsetFile`
  source, cross-checked against `Magesty.read_embset` in this package's oracle.
- Validated: basis / model / fit round-trips (predictions bit-identical, coefficients
  re-paired by key under scrambled order, multi-op space-group ops, empty basis), input
  parsing + defaults + keyword overrides + error paths.

### Sunny.jl export (M11)
- **Bilinear extraction core** (`slce/bilinear.jl`, dependency-free): `_l1_pair_matrix` /
  `_l2_onsite_matrix` turn a folded tesseral tensor into a Cartesian exchange / single-ion
  matrix (`eₐ'·M·e_b = Σ folded·Z·Z`, via the `Harmonics.N1`/`A2`/`B2` tesseral constants);
  `_classify_salc` keeps only `ls=[1,1]` pairs and `ls=[2]`
  single-ion (`ls=[0…]` → `j0`, the rest skipped + reported). `_bilinear_terms`
  folds the directed members into one `BilinearTerms` matrix per undirected supercell bond.
  `_reconstruct_energy ≈ predict_energy − j0` gates the whole chain **without Sunny**.
- **Sunny-specific core** (`interop/sunny.jl`, dependency-free): `_sunny_primitive` unfolds
  the supercell matrices onto the chemical primitive cell recovered from the
  pure translations (one Sunny bond per primitive bond, a `clean` flag for the fallback),
  plus the `to_sunny` entry point (a friendly "load Sunny" error without the extension).
- **Extension** (`ext/SLCESunnyExt`, loaded by `using Sunny`):
  `to_sunny(model; spin_length, g = 2, mode = :auto, scaling = :auto, placement = :auto)
  -> Sunny.System` builds a real `System`
  (`set_exchange_at!`/`set_onsite_coupling_at!` on the supercell, or
  `set_exchange!`/`Bond` on the primitive cell); its classical
  energy reproduces `predict_energy − j0`. `scaling` chooses how the effective spin
  `S_eff` enters: `:moment` puts `S_eff` into Sunny's `Moment` and rescales
  `J = M/(SₐS_b)` (energy and dispersion exact; half-integer `S_eff` only), `:coupling`
  keeps a placeholder `Moment` (`s₀ = 1`) and folds `S_eff` into the couplings (any
  `S_eff`; dispersion-exact, static energy rescaled), `:auto` picks `:moment` for
  half-integer spins, else `:coupling`. `placement = :explicit` (exact, folded
  dispersion) / `:primitive` (unfolded) / `:auto`.
- Validated: Sunny-free conversion + primitive-fold tests (`test/unit/test_sunny.jl`,
  main suite) and a separate `test/sunny/` environment (real `Sunny.System` energy vs SLCE
  for both routes, the primitive system reshaped back to the supercell, mode/spin handling,
  the skip warning).

### Fitted-model introspection (M12)
- **Public, stable contract** (`slce/introspect.jl`): `spin_multipole_terms(model) ->
  Vector{SpinMultipoleTerm}` is the code-neutral view downstream packages (the `SLCETools.jl`
  mean-field samplers) read **instead of** the SALC-basis internals (`SALCMember` /
  `SALCTerm`). Each `SpinMultipoleTerm` carries the raw fitted `coef = jϕ`, the cluster
  `atoms` / `shifts`, the per-site `ls`, and the `folded` tensor; the per-`N` scale
  `(4π)^(body/2)` is **left to the consumer** (applied in exactly one place, the
  `_energy_from_terms` reconstruction gate). `bilinear_terms(model)` is the thin public
  wrapper of the general Cartesian bilinear / single-ion extraction `_bilinear_terms`
  (`slce/bilinear.jl`).
- Validated: `test/unit/test_introspect.jl` (the terms reconstruct `predict_energy − j0`).

### Regularized estimators (M12)
- **Analytic, in-core** (`fitting/estimators.jl`, no extension): `AdaptiveRidge(; lambda,
  epsilon, max_iter, tol)` — iterative reweighted ridge (Frommlet & Nuel 2016), an L0
  approximation that refits `(X'X + λ·Diagonal(w)) \ X'y` with `wⱼ = 1/(βⱼ² + ε)` until the
  ∞-norm change drops below `tol`; `lambda = 0` ⇒ OLS. `islinear` ⇒ `true` (a linear
  smoother in the converged-weight sense). `FixedCoefficients(beta)` — adapter returning a
  fixed coefficient vector (length-checked against `size(X, 2)`), for reuse as an
  `AdaptiveLasso` pilot.
- **GLMNet-backed types in core, solve in extension** (`ext/SLCEGLMNetExt`,
  loaded by `using GLMNet`): `ElasticNet(; alpha, lambda, standardize, nfolds, select,
  seed, nlambda)`, `Lasso(; …)` (= `alpha = 1`), and `AdaptiveLasso(; pilot, lambda, gamma,
  epsilon, standardize, …)`. Named and dispatched on without the dependency; argument
  validation lives in core. GLMNet minimizes `(1/2n)‖y−Xβ‖² + λ[(1−α)/2‖β‖₂² + α‖β‖₁]` on
  the column-centered `(X, y)` with `intercept = false` (so `j0` stays analytic) and
  `standardize` (penalty acts per-column at `λ·std`). `lambda = nothing` ⇒ K-fold CV over
  GLMNet's λ path, `select` = `:lambda_min`/`:lambda_1se`; folds are **grouped by
  configuration** (via `row_groups`) and seeded deterministically (`hash`-ranked, no RNG
  dependency). A numeric `lambda` skips CV.
- `AdaptiveLasso` (Zou 2006) runs `pilot` (any estimator; default `OLS`) for `β̂`, then a
  weighted Lasso with per-column penalty factor `wⱼ = 1/max(|β̂ⱼ|, ε)^γ` (held fixed across
  the fixed-λ or CV path). `gamma = 0` reduces exactly to a plain Lasso. The pilot's own
  solve receives the co-fit `row_groups`.
- Validated in a separate `test/glmnet/` environment: tiny-λ ≈ OLS, the analytic-`j0`
  centering invariant, CV support recovery + sparsity, `:lambda_1se` shrinkage,
  reproducibility, ElasticNet ≠ Lasso, AdaptiveLasso support recovery, `γ = 0` ≡ plain
  Lasso, pilot pluggability (`Ridge`/`FixedCoefficients`), and energy+torque grouped-CV
  co-fits. Core-only construction / validation / deferred-backend error, plus the full
  `AdaptiveRidge` / `FixedCoefficients` solves, in `test/unit/test_fit.jl`.

### Cost-weighted group selection (M13)
- **Estimator** (`fitting/estimators.jl`, in-core): `GroupAdaptiveRidge(column_groups,
  group_weights; lambda, epsilon, max_iter, tol)` — the group extension of
  `AdaptiveRidge`, approximating the weighted group-L0 `λ·Σ_g v_g·1{β_g≠0}` by iterating
  `wⱼ = v_g/(‖β_g‖² + p_g·ε)` (all columns of a group share one weight; a surviving
  group's converged penalty is exactly `λ·v_g`). Exact degeneration to `AdaptiveRidge`
  for singleton groups with unit weights; `lambda = 0` ⇒ OLS; `islinear` ⇒ `true`.
  `column_groups` labels **columns** (contiguous `1:G`, validated in the inner
  constructor) — unrelated to the per-row `row_groups` kwarg of `solve_coefficients`. The
  weight map `_gar_weights!` is the single definition shared with the GCV diagnostics.
- **Basis helpers** (`fitting/selection.jl`; public, unexported): `salc_groups(basis)`
  — column → group labels by `(body, orbit_id, decors)`, the granularity at which MC
  contraction entries vanish; `group_costs(basis, column_groups)` — per-group summed
  **slot count** over the distinct entries `(atoms, shifts, slotkeys, index)` in the
  union over canonical members (additive across the `salc_groups` partition, and under
  any coarser one). The slot factor is the number of **site** programs
  `_push_term_programs!` emits per entry (`nnz · length(slots)`, walked every sweep);
  the bare entry count is the size of the separate **energy** program, walked once per
  run, so counting it mis-ranked across body orders. Joint (displacement-decorated)
  bases are supported — the slot-keyed entry is what makes them well-defined;
  `cost_weights(basis; cost_exponent)` — `v_g = √p_g·(c_g/c̄)^θ` with `θ = cost_exponent
  ∈ [0, 1]`, tilting the
  penalty from cost-blind to cost-proportional. Convenience constructor
  `GroupAdaptiveRidge(basis; lambda, cost_exponent, …)` bundles them.
- **GCV / effective dof** (exported): `effective_dof(f)` = `tr(X(X'X+λD)⁻¹X') + 1` with
  the converged penalty diagonal recomputed from the fitted coefficients
  (`_penalty_diagonal`, one method per linear estimator); `gcv(f)` = `n·RSS/(n−df)²` on
  the assembled problem, `Inf` in the near-interpolating regime `df → n`. The dof trace
  is an eigenproblem on the smaller Gram side (`p ≤ n`: weighted `X'X`, reusing the
  path's cached Gram; `n < p`: the `n×n` dual) — never an `n×p` SVD. Linear estimators
  only; on torque co-fits GCV is optimistic (correlated within-configuration rows) —
  grouped CV is the ground truth there (documented, not an error).
- **λ path + Pareto** (exported): `select_fit(dataset, est; lambdas, torque_weight,
  criterion = :gcv|:cv, score_rtol, costs, threshold, nfolds, seed) -> LambdaPath` —
  descending warm-started path on a once-assembled Gram; per-λ score, effective dof,
  alive groups (the `refit` scaled-magnitude rule, any-column-per-group) and predicted
  MC cost `Σ_{g alive} c_g`; selection = cheapest λ within `(1 + score_rtol)` of the minimum score
  (the cost-aware generalization of `:lambda_1se`; `Inf` scores never eligible, cost
  ties → larger λ). The adaptive iteration never yields exact zeros, so the default
  `threshold = nothing` is a per-λ **relative** alive floor (`_ALIVE_RTOL = 1e-6` of
  that λ's largest scaled magnitude; an absolute number reproduces `refit`'s rule,
  `0.0` degenerates to all-alive). `criterion = :cv` is configuration-grouped K-fold in
  core (deterministic seeded folds; per-fold Gram downdate; fold reduction warns). The
  selected fit is re-solved cold — its path row (`n_alive`/`cost`) is re-derived from
  that cold solve and the effective absolute threshold is returned as
  `path.threshold`, so `refit(path.fit; threshold = path.threshold)` realizes exactly
  the reported support; `LambdaPath` is a Tables.jl source.
- **Threshold front** (exported): `select_support(f; npoints = 25, thresholds, score_rtol, column_groups,
  costs, evalset = f.dataset, estimator = OLS()) -> SupportPath` — the second knob:
  sweeps the alive threshold at a fixed fit (auto grid = log-rank-spaced points on
  the per-group scaled-magnitude spectrum + the full-support anchor, or an explicit
  vector), de-biases with `refit` per point, scores each refit by the fit's own
  `(1−w)·MSE_E + w·MSE_T` objective on `evalset` (pass a held-out slice for an honest
  axis; fingerprint-checked against the training basis), and applies the same
  Pareto rule. Needed because real-data group-magnitude spectra are continuous (no
  alive/dead gap for the λ path to expose). `SupportPath` is a Tables.jl source
  (`threshold`/`n_alive`/`cost`/`score`/`rmse_energy`/`rmse_torque`/`rmse_force`/
  `selected`).
  `n_alive`/`cost` are read off the **returned refits**, not off the pre-threshold
  magnitudes — the two agree without a constraint, but under an ASR a support that
  splits a constraint-coupled column set structurally zeroes some survivors, so the
  pre-threshold form over-reports. Runs under a plain ASR (staged fits still refused).
- **Constraint-aware costing** (`fitting/selection.jl`): `group_costs(basis, labels;
  asr)` prices at zero any group every column of which the constraint annihilates (no
  translation-invariant model can carry it); `select_support` passes the fit's own
  `reparam` by default. `group_freedom(rep, labels)` (public, unexported) =
  `s_g = ‖Z[g, :]‖_F²`, with `Σ_g s_g ≡ q` exactly, `0 ≤ s_g ≤ p_g`, `s_g = 0` ⟺
  structurally zeroed, and gauge-invariance (a trace of the projector `Z·Z'`).
  **Groups are not the ASR's granularity**: measured over five fixtures (`G` 2→20), no
  displacement-touched group has a feasible subspace alone, the constraint's atoms are
  matroid circuits of 2–3 columns spanning several groups, and closing groups under
  `A`'s connected components collapses them to two clusters regardless of `G` — closure
  is therefore rejected, and the cost axis is real on the ASR-untouched pure-spin
  groups and near-binary on the displacement side.
- **Generic CV** (exported): `cross_validate(dataset, estimator; torque_weight,
  nfolds = 5, seed = 1) -> CVResult` — configuration-grouped K-fold assessment of any
  `fit` call: each fold refits from scratch (fold-local centering/whitening, no
  leakage) and scores the held-out configurations in prediction space. Reports the
  per-fold and pooled out-of-fold energy **and** torque RMSEs independently of
  `torque_weight`, plus the `(1−w)·MSE_E + w·MSE_T` score. Deterministic seeded
  folds (`_grouped_folds`); fold reduction warns, `< 6` configs errors; a
  `FixedCoefficients` is rejected (fold-independent coefficients would leak).
  `CVResult` is a Tables.jl source. Unlike `select_fit(criterion = :cv)` (global
  whitening, λ ranking only), this is the honest generalization-error estimate.
- Validated in `test/unit/test_selection.jl`: construction/validation, exact
  `AdaptiveRidge` degeneration, group-sparse recovery + weight monotonicity, label/cost
  hand counts + additivity, `θ` endpoints, dense-hat-matrix trace agreement, the
  underdetermined regime + guards, warm/cold path consistency, the Pareto rule, and the
  end-to-end select → refit workflow.

## Not yet implemented (v0 follow-ups)
- The v0 slice is feature-complete; no estimator/observable/IO follow-ups outstanding.
- **Joint (spin + lattice) training data — the gate on every M4/M5 physics
  validation.** All production data today is pure spin (`l044` = Nd₂Fe₁₄B,
  `lmax = [4,4,2,…]`, `soc = false`, 3-body), so the displacement channel and every
  diagnostic that reads it (`force_constants`, `dynamical_matrix`,
  `rotational_residual`, `rotation_transfer_residual`) has only fixture-scale
  evidence. What is needed is a **mixed** dataset — `u = 0` spin-only configurations
  (pure-spin terms), `ê`-fixed displaced ones (clamped-ion force constants), and
  configurations where **both** are displaced (the cross terms, i.e. the
  magnetic-state dependence of the force constants — the flagship deliverable, and
  the only part no existing dataset touches). Four constraints on how it is
  generated, each already measured rather than assumed:
  1. **Relaxed reference (`F ≈ 0`).** At a stressed reference the rotational
     residual depends on the home-cell image choice (measured 1349×); at `F = 0` it
     does not. Same fact as the `ε_sym`/vanishing-stress escape.
  2. **COM-free displacement sampling.** Recovery from forces + torques alone is
     `~1e-15` under the ASR with COM-free sampling and lands on a different model
     without it (design record amendment 9).
  3. **Forces are mandatory.** Torque is blind to every spin-independent direction,
     so force constants are *not* identifiable from torques alone.
  4. **`torque_qualified` provenance** on any datum whose torques are used.
  Open decisions: the system (bcc Fe is cheapest, Nd₂Fe₁₄B is the target at 68
  atoms), how the displacement patterns are generated (`suggest_displacements`, the
  ALAMODE `MODE = suggest` equivalent, is also unimplemented), and how they are
  crossed with the spin configurations.

## Oracle environment (`test/oracle/`)

A separate Julia env (`[sources]`-deving both `SLCE` and a pinned
`Magesty.jl`) cross-checks convention-fixed kernels and gauge-invariant
aggregates / predictions against Magesty. The core suite never depends on
Magesty. Run: `julia --project=test/oracle test/oracle/runtests.jl`.

(Sections for M3–M10 added as they land.)
