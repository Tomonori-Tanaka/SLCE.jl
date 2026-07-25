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

The include order in the SCE layer is `slce/coeftable.jl` → `slce/bilinear.jl` →
`slce/introspect.jl` → `interop/sunny.jl`: the bilinear extraction is a core
capability consumed by both the introspection and the Sunny interop.

## Extension seams (contracts)

- **DFT sources**: `read_configs(src::AbstractDFTSource) -> Vector{<:AbstractTrainingDatum}`.
- **Estimators**: subtype `AbstractEstimator` + `solve_coefficients(est, X, y; groups)
  -> jphi` (the `(X, y)` is already column-centered and row-scaled — add no intercept,
  do not re-weight; `groups` labels rows from the same sample for grouped resampling).
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
  (`AbstractImageSelection`): `MinimumImage()` (the SCE-fitting default) keeps only the
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
  + on-sphere central difference, and bit-for-bit agreement with Magesty's
  `TesseralHarmonics` (oracle).

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
  `SlotRef`s (member-site index + `SiteFactor`, canonical order SPIN axes before
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
  soc, cutoff, nbody)` sugar declares one truncation-table row (theory note
  §3/§7); `BasisSpec(labels; sectors, lmax, pmax, ...)` resolves it once into
  the dense canonical `SectorRule` (`spin_mode ∈ :none/:explicit/:any`, range
  tuples, resolved per-pair cutoff — validated, comparable, persisted). The
  admitted label set is the union of the rows ∩ the global per-species
  `lmax`/`pmax` and per-body `lsum` caps; `nbody` and the per-body `cutoff`
  envelope are derived (elementwise max over the sectors admitting each body
  order). `soc` is per sector (`false` ⇔ enumerate only `L_S = 0`; exact
  together with the even-`Σl_spin` screen). The dense pure-spin form replaces
  `isotropy` with the inverted `soc` flag (deprecation errors at the keyword,
  the TOML input key, and nowhere silently); `disp_scale` is fixed in the spec
  and persisted. Driving `_orbit_salcs_decors` from the resolved table is
  M2b-3b; until then a sector spec refuses to build an `SLCEBasis`.
- Validated by the ground-truth tests with non-collinear spins, **all `Lf`, all body
  orders**: space-group invariance `Φ(g·e)=Φ(e)`, time-reversal evenness, linear
  independence; projector eigenvalues exactly 0/1. Improper-op parity is handled
  automatically (site-axis rotation by full `R` + even `Σl`). Cross-validated against
  Magesty: per-`(body, ls, Lf)` invariant-subspace dimensions agree through 3-body.

### fitting + SCE API (M8, M9)
- `BasisSpec` (validated: `nbody ≥ 1`, symmetric per-body `cutoff` matrices with
  entries `≥ 0` or `Inf`, `lsum ≥ 0` per body order, nonempty `lmax`/`pmax`
  with entries ≥ 0, `disp_scale` finite positive, sector-rule consistency —
  see the M2b-3a bullet for the sector-table form and the `isotropy → soc`
  replacement), `SLCEBasis` (carries its spec in the `spec` field), `SLCEDataset`
  (energy design matrix `X_E`, and the
  torque design matrix `X_T` via the four-argument form; supports `length`,
  configuration slicing `dataset[idx]` — integer/`Bool`/`:` — and `vcat` of
  same-fingerprint parts, all without recomputing design rows; the
  `SpinDatum`/source path rejects a zero moment on a basis-referenced atom),
  `SLCEModel`/`SLCEFit`
  (plus the public constructor `SLCEModel(basis, j0, jphi)` for synthetic models —
  keys filled in from the basis),
  `fit(SLCEFit, dataset, estimator; torque_weight)`, `refit(f, estimator; threshold)`
  (re-solve on the scaled-magnitude support of `f` — the de-biasing step after a sparse
  fit; shares `_assemble_problem` with `fit`), `predict_energy`/`predict_torque`,
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
- **Tabular results** (`slce/coeftable.jl`): `coeftable(fit | model) -> SCECoefficients`
  is a **Tables.jl** source — one row per SALC (`body`, `orbit_id`, `ls` as a comma
  string, `Lf`, `block`, `J`) — so it drops into `DataFrame` / `CSV.write` /
  `Arrow.write`. The library owns the internal-storage → labeled-row mapping; the caller
  brings the table/IO package. `j0` is the intercept (`intercept(c)`), not a row.
  Tables.jl is a lightweight core dep.
- **DFT data sources** (`io/dftsource.jl`): the **code-agnostic boundary only** —
  `SpinDatum` (energy + spin directions + magmoms + constraining field + the derived
  torque target `τ_a = m_a × B_a`, the physical / Landau–Lifshitz torque) and
  `read_configs(src::AbstractDFTSource) -> Vector{SpinDatum}`, with `SLCEDataset(basis, src)`
  going source → dataset. The **concrete per-code adapters live in the `SLCETools.jl`
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
  `to_sunny(model; spins, g = 2, mode = :auto, scaling = :auto, placement = :auto)
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
  main suite) and a separate `test/sunny/` environment (real `Sunny.System` energy vs SCE
  for both routes, the primitive system reshaped back to the supercell, mode/spin handling,
  the skip warning).

### Fitted-model introspection (M12)
- **Public, stable contract** (`slce/introspect.jl`): `multipole_terms(model) ->
  Vector{MultipoleTerm}` is the code-neutral view downstream packages (the `SLCETools.jl`
  mean-field samplers) read **instead of** the SALC-basis internals (`SALCMember` /
  `SALCTerm`). Each `MultipoleTerm` carries the raw fitted `coef = jϕ`, the cluster
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
  smoother in the converged-weight sense). `PrecomputedPilot(beta)` — adapter returning a
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
  configuration** (via `groups`) and seeded deterministically (`hash`-ranked, no RNG
  dependency). A numeric `lambda` skips CV.
- `AdaptiveLasso` (Zou 2006) runs `pilot` (any estimator; default `OLS`) for `β̂`, then a
  weighted Lasso with per-column penalty factor `wⱼ = 1/max(|β̂ⱼ|, ε)^γ` (held fixed across
  the fixed-λ or CV path). `gamma = 0` reduces exactly to a plain Lasso. The pilot's own
  solve receives the co-fit `groups`.
- Validated in a separate `test/glmnet/` environment: tiny-λ ≈ OLS, the analytic-`j0`
  centering invariant, CV support recovery + sparsity, `:lambda_1se` shrinkage,
  reproducibility, ElasticNet ≠ Lasso, AdaptiveLasso support recovery, `γ = 0` ≡ plain
  Lasso, pilot pluggability (`Ridge`/`PrecomputedPilot`), and energy+torque grouped-CV
  co-fits. Core-only construction / validation / deferred-backend error, plus the full
  `AdaptiveRidge` / `PrecomputedPilot` solves, in `test/unit/test_fit.jl`.

### Cost-weighted group selection (M13)
- **Estimator** (`fitting/estimators.jl`, in-core): `GroupAdaptiveRidge(column_groups,
  group_weights; lambda, epsilon, max_iter, tol)` — the group extension of
  `AdaptiveRidge`, approximating the weighted group-L0 `λ·Σ_g v_g·1{β_g≠0}` by iterating
  `wⱼ = v_g/(‖β_g‖² + p_g·ε)` (all columns of a group share one weight; a surviving
  group's converged penalty is exactly `λ·v_g`). Exact degeneration to `AdaptiveRidge`
  for singleton groups with unit weights; `lambda = 0` ⇒ OLS; `islinear` ⇒ `true`.
  `column_groups` labels **columns** (contiguous `1:G`, validated in the inner
  constructor) — unrelated to the per-row `groups` kwarg of `solve_coefficients`. The
  weight map `_gar_weights!` is the single definition shared with the GCV diagnostics.
- **Basis helpers** (`fitting/selection.jl`; public, unexported): `salc_groups(basis)`
  — column → group labels by `(body, orbit_id, ls)`, the granularity at which MC
  contraction entries vanish; `group_costs(basis, labels)` — per-group distinct-entry
  union count over canonical members (additive across the `salc_groups` partition);
  `cost_weights(basis; theta)` — `v_g = √p_g·(c_g/c̄)^θ`, `θ ∈ [0, 1]` tilting the
  penalty from cost-blind to cost-proportional. Convenience constructor
  `GroupAdaptiveRidge(basis; lambda, theta, …)` bundles them.
- **GCV / effective dof** (exported): `effective_dof(f)` = `tr(X(X'X+λD)⁻¹X') + 1` with
  the converged penalty diagonal recomputed from the fitted coefficients
  (`_penalty_diagonal`, one method per linear estimator); `gcv(f)` = `n·RSS/(n−df)²` on
  the assembled problem, `Inf` in the near-interpolating regime `df → n`. The dof trace
  is an eigenproblem on the smaller Gram side (`p ≤ n`: weighted `X'X`, reusing the
  path's cached Gram; `n < p`: the `n×n` dual) — never an `n×p` SVD. Linear estimators
  only; on torque co-fits GCV is optimistic (correlated within-configuration rows) —
  grouped CV is the ground truth there (documented, not an error).
- **λ path + Pareto** (exported): `select_fit(dataset, est; lambdas, torque_weight,
  criterion = :gcv|:cv, delta, costs, threshold, nfolds, seed) -> SelectionPath` —
  descending warm-started path on a once-assembled Gram; per-λ score, effective dof,
  alive groups (the `refit` scaled-magnitude rule, any-column-per-group) and predicted
  MC cost `Σ_{g alive} c_g`; selection = cheapest λ within `(1+δ)` of the minimum score
  (the cost-aware generalization of `:lambda_1se`; `Inf` scores never eligible, cost
  ties → larger λ). The adaptive iteration never yields exact zeros, so the default
  `threshold = nothing` is a per-λ **relative** alive floor (`_ALIVE_RTOL = 1e-6` of
  that λ's largest scaled magnitude; an absolute number reproduces `refit`'s rule,
  `0.0` degenerates to all-alive). `criterion = :cv` is configuration-grouped K-fold in
  core (deterministic seeded folds; per-fold Gram downdate; fold reduction warns). The
  selected fit is re-solved cold — its path row (`n_alive`/`cost`) is re-derived from
  that cold solve and the effective absolute threshold is returned as
  `path.threshold`, so `refit(path.fit; threshold = path.threshold)` realizes exactly
  the reported support; `SelectionPath` is a Tables.jl source.
- **Threshold front** (exported): `select_support(f; thresholds = 25, delta, labels,
  costs, evalset = f.dataset, estimator = OLS()) -> SupportPath` — the second knob:
  sweeps the alive threshold at a fixed fit (auto grid = log-rank-spaced points on
  the per-group scaled-magnitude spectrum + the full-support anchor, or an explicit
  vector), de-biases with `refit` per point, scores each refit by the fit's own
  `(1−w)·MSE_E + w·MSE_T` objective on `evalset` (pass a held-out slice for an honest
  axis; fingerprint-checked against the training basis), and applies the same
  Pareto rule. Needed because real-data group-magnitude spectra are continuous (no
  alive/dead gap for the λ path to expose). `SupportPath` is a Tables.jl source
  (`threshold`/`n_alive`/`cost`/`score`/`rmse_energy`/`rmse_torque`/`selected`).
- **Generic CV** (exported): `cross_validate(dataset, estimator; torque_weight,
  nfolds = 5, seed = 1) -> CVResult` — configuration-grouped K-fold assessment of any
  `fit` call: each fold refits from scratch (fold-local centering/whitening, no
  leakage) and scores the held-out configurations in prediction space. Reports the
  per-fold and pooled out-of-fold energy **and** torque RMSEs independently of
  `torque_weight`, plus the `(1−w)·MSE_E + w·MSE_T` score. Deterministic seeded
  folds (`_grouped_folds`); fold reduction warns, `< 6` configs errors; a
  `PrecomputedPilot` is rejected (fold-independent coefficients would leak).
  `CVResult` is a Tables.jl source. Unlike `select_fit(criterion = :cv)` (global
  whitening, λ ranking only), this is the honest generalization-error estimate.
- Validated in `test/unit/test_selection.jl`: construction/validation, exact
  `AdaptiveRidge` degeneration, group-sparse recovery + weight monotonicity, label/cost
  hand counts + additivity, `θ` endpoints, dense-hat-matrix trace agreement, the
  underdetermined regime + guards, warm/cold path consistency, the Pareto rule, and the
  end-to-end select → refit workflow.

## Not yet implemented (v0 follow-ups)
- The v0 slice is feature-complete; no estimator/observable/IO follow-ups outstanding.

## Oracle environment (`test/oracle/`)

A separate Julia env (`[sources]`-deving both `SLCE` and a pinned
`Magesty.jl`) cross-checks convention-fixed kernels and gauge-invariant
aggregates / predictions against Magesty. The core suite never depends on
Magesty. Run: `julia --project=test/oracle test/oracle/runtests.jl`.

(Sections for M3–M10 added as they land.)
