# Design notes

Why `SLCE` is built the way it is — the deliberate refinements over
`Magesty.jl`. Bit-for-bit agreement with Magesty is **not** a goal; these are
cleaner constructions that are validated to be *physically* correct (symmetry
invariance, finite differences, known-coupling recovery), and may differ from
Magesty by a gauge or by rounding.

## 1. Generalized cutoff neighbor list (vs. a fixed 27-cell grid)

Magesty enumerates neighbors over a hard-coded 3×3×3 image grid, which silently
caps any interaction range at one lattice translation and stores a dense
`O(n²)` `min_distance_pairs` matrix. The rebuild (`geometry/neighborlist.jl`)
drives the image range by the cutoff: along each periodic axis,
`N_d = ceil(cutoff · ‖b_d‖)` with `b_d` the reciprocal row, which the projection
bound `|n_d + Δf_d| ≤ cutoff/spacingᵈ` proves encloses every in-cutoff image —
correct for triclinic cells and arbitrary cutoffs, and a sparse `Vector` of pairs.
Each `NeighborPair` keeps the integer lattice translation `R` (`shift`), which the
orbit reduction propagates; this is the hook for future reciprocal-space /
generalized-Bloch `E(q)` training data. Validated against an *independent*
over-large-shell brute force (the production tight range is never checked against
itself), including a strongly sheared triclinic cell.

## 1b. Minimum-image resolvability and the Wigner–Seitz cell

A spherical cutoff is the wrong primitive for *what a finite supercell can resolve*.
Under plain periodic boundary conditions (the usual DFT, excluding the generalized-Bloch
spin-spiral case), every periodic image of an atom carries the **same spin**. So if atom
`A` reaches both the minimum image of `B` (at distance `d`) and a farther image of `B`
(at distance `d' > d`), the two "interactions" are the *same* `e_A · e_B` in every
training configuration — perfectly collinear design-matrix columns. A supercell can only
independently resolve the **minimum-image** set: the displacements inside the **Wigner–
Seitz cell** of the (super)lattice.

The WS cell is a polyhedron, not a ball. For a cubic cell of side `L` it is the cube
`[−L/2, L/2]³`: its inscribed sphere has radius `L/2` (face centres), but the furthest
resolvable pair is the body-diagonal corner `(L/2, L/2, L/2)` at `√3·L/2 ≈ 0.866 L`.
A spherical cutoff therefore cannot express "all resolvable pairs": to reach the corners
it must exceed `L/2` along the faces, which is exactly where it starts sweeping in the
aliased (non-resolvable) images. This face-vs-corner mismatch is the classic source of
`L/2`-plane double-counting errors.

The rebuild makes the selection explicit (`AbstractImageSelection`):

- **`MinimumImage`** (default) keeps, per atom pair, only the minimum-image
  displacement(s) — the WS representative — with **boundary ties kept as distinct
  members** (the `L/2` faces are 2-fold, edges 4-fold, corners 8-fold; e.g. a body-centred
  cubic pair has all eight `(±L/2,±L/2,±L/2)` images at once). `i==j` self-pairs are
  dropped: in plain PBC both ends share `e_i`, so the term is a constant (`Lf=0`) or a
  1-body alias (`Lf>0`), never an independent pair; likewise an `N`-body cluster must use
  distinct atoms, since a reused atom image aliases a lower-body term. The radial cutoff
  merely trims this set, and `cutoff = Inf` keeps the whole WS cell (Magesty spells
  this `cutoff = -1`). The minimum-image search box is grown adaptively to a provably
  sufficient range (`|n_d| ≤ ‖b_d‖·d_min + 1`), so heavily skewed / non-reduced cells
  cannot silently return a wrong "minimum".
- **`AllImages`** is the every-image-within-cutoff enumeration, retained as the
  **generalized-Bloch / spin-spiral seam**: there the spiral makes `e^{iq·R}` distinguish
  the images, so a single primitive cell plus a `q`-mesh resolves arbitrary range and the
  `L/2` wall does not apply. This is *why* `NeighborPair` carries `R` — for plain PBC it is
  redundant (folds to the minimum image), but the spin-spiral path needs it.

Why not just adopt Magesty's minimum-image filter wholesale? It silently *caps* the range
at the WS cell (drops farther images without comment); the rebuild instead keeps the
explicit `MinimumImage`/`AllImages` choice and the resolvable boundary visible. The two
agree exactly below half the smallest perpendicular width (`cutoff < min_d d_i / 2`),
where each in-cutoff image is already the minimum — so the Heisenberg `J = 2√3·jϕ`
recovery and the kagome 3-body co-fit are unchanged, and the only deliberate divergence
from Magesty is the degenerate single-atom-cell self-pair (one orbit there, zero here).
Full-WS bases under symmetry are checked to have full design-matrix column rank (no
collinear columns survive).

**The third edge at `N ≥ 3`.** For pairs the WS boundary only multiplies tied images;
for clusters of three or more atoms it adds a genuinely new constraint. A triangle
`{i, j, k}` is admissible only if **all three edges sit at their atom-pair minimum image
simultaneously** (the compact-cluster criterion) — and the clique check enforces this on
the *actual chosen images* of every pair, not just the two anchor edges `i–j`, `i–k`. It
is not enough for each pair to be individually minimum-image-resolvable: the images that
make `i–j` and `i–k` minimal can force `j–k` onto a longer, non-minimum image, which must
reject the cluster. (Concretely, three atoms equally spaced around a 1-D ring have every
pair minimum-image at the same distance, yet admit *zero* compact triangles — you cannot
realize all three minimal edges at once, just as an equilateral triangle does not embed on
a ring.) On the WS boundary the tied images then multiply the compact N-body clusters the
way they multiply pairs, and the symmetry reduction must group these without over- or
under-merging. Because this combinatorics is easy to get subtly wrong, the full count —
the candidate set for `N = 2, 3, 4` (including `cutoff = Inf`, the whole WS cell) and
the symmetry-orbit partition — is pinned in `test/unit/test_ws_nbody.jl` against an
**independent brute-force enumeration** of all compact clusters, on cells deliberately
seeded with face / edge / corner ties (cubic face-atoms, fcc, skewed hexagonal). The
production enumeration matches it exactly, and every emitted member is re-verified to have
all edges at the brute-force minimum-image distance.

## 2. Real Wigner-D from the package's own `Zₗₘ` (vs. Euler angles + complex D + c2r)

Magesty builds the real Wigner-D as `conj(C)·wignerD(l,α,β,γ)·transpose(C)` —
Euler-angle decomposition, the complex Wigner-D (a dependency), and the complex→
real matrix `C`. The rebuild (`AngularMomentum.wignerD_real`) instead **fits** the
matrix directly: sample the sphere (Fibonacci points), require
`Zₗ,m(R·r̂) = Σ_{m'} Δ[m,m'] Zₗ,m'(r̂)`, and solve the (exact, oversampled) linear
system. Advantages: it is consistent with *our* `Zₗₘ` convention by construction
(no transpose/phase bookkeeping), uses real arithmetic only, needs no `WignerD`
dependency, and handles improper rotations (reflections, inversion) with no special
case. Validated by the functional identity on fresh points, orthogonality,
`l=0/1/inversion` special cases, and agreement with Magesty's `Δl`.

## 3. Orbit–stabilizer SALC projection (vs. a whole-supercell Reynolds average)

Magesty averages over the whole space group acting on explicit supercell atoms.
The rebuild (`basis/salcbasis.jl`) uses the orbit–stabilizer theorem: project the
representative cluster's coefficient onto the trivial irrep of its **site
stabilizer**, gauge-fix, then transport to the orbit members. The invariant count
is identical, but the projectors are small and the cluster model stays
primitive-cell + integer `R`.

For **arbitrary body order** the projection runs over the combined
`(ordering o, coupling path p, Mf)` space (the next section explains why). The
key construction choice: each stabilizer operation acts by rotating every **site**
axis by `wignerD_real(l_i, R)` (full `R`) and relabeling axes by the induced
permutation, and the action matrix is read off by **contracting against the
package's own orthonormal coupled tensors** — no 6j/9j recoupling, in the same
spirit as building `wignerD_real` itself. Because the orthonormal coupled tensors
make the action's matrix exact, `P = mean_g M_g` is a true projector and the
eigenvalues come out exactly `0`/`1` (an in-band idempotency assertion guards this).

A subtlety found earlier (an adversarial review) is **parity**: an *improper* `R`
carries the harmonic parity `(−1)^l` per site, while the time-reversal-even sector
has total product parity `(−1)^{Σl} = +1`. A first version that represented an op on
the final multiplet directly by `wignerD_real(Lf, R)` disagreed for odd `Lf` and
broke invariance on centrosymmetric cells (it needed an explicit proper-part fix).
The site-axis construction above **handles this automatically**: rotating each site
axis by the full `R` carries the genuine `(−1)^{l_i}` per site, and even `Σl` makes
the net improper action correct — so symmetry-forbidden odd-`Lf` channels (e.g. a
Dzyaloshinskii–Moriya-like term on a bond with an inversion center) drop out with no
special case. The ground-truth gate is the invariance test `Φ(g·e) = Φ(e)` with
*non-collinear* spins for all `Lf` and all body orders (collinear test spins make
every odd-`Lf` invariant vanish and would silently hide such bugs), plus a
linear-independence check (design-matrix rank = #SALC).

## 3b. Combined-space projection and multi-term SALCs (`N ≥ 3`)

At `N ≥ 3` a stabilizer operation can **permute symmetry-equivalent sites**. That
permutation does two things a per-multiplet projector cannot express:

1. It mixes **coupling paths** — different intermediate `L` sequences that share the
   same final `Lf` — so the invariant lives in a span of paths, folded into one
   tensor (this already bites equal-`l` channels like `(2,2,2)`).
2. When the permuted sites carry **unequal `l`** (e.g. `(1,1,2)` on an equilateral
   triangle), it mixes **`l`-orderings**: the invariant is a combination of
   `(1,1,2)`, `(1,2,1)`, `(2,1,1)` that no single ordering contains.

So a SALC is a sum over orderings, each with its own per-site `ls` — a
**multi-term** object (`SALCTerm`); `evaluate_salc` and the torque gradient loop over
terms. When the site permutations are a *proper* subgroup of `Sₙ`, a degenerate
multiset can split into more than one ordering orbit (e.g. a mirror-only triangle
puts `l = 2` on the apex vs. on a base site — two distinct SALCs that share the
sorted label `[1,1,2]`); `SALCKey` keeps `block` running across them so it stays
injective (the by-key column contract, §4).

**Canonical member folding.** The ordered-image space is the right place to *run*
the projection (a stabilizer op is a plain axis permutation there), but it is a
redundant place to *leave the output*: transporting to every ordered, anchored image
delivers each physical cluster instance `N!` times (once per site ordering at `N`
distinct sites), each copy an axis-permuted tensor. Since the contraction
`Σ_μ folded[μ] ∏ᵢ Z_{lsᵢμᵢ}` is invariant under jointly permuting site slots and
tensor axes, the construction's last step (`_canonicalize_members`) folds the copies
exactly: sites sorted by `(atom, shift)`, shifts re-anchored to `shifts[1] = 0`,
tensors `permutedims`-aligned and summed per site→`l` assignment. This is a pure
regrouping — `Φ`, gradients, invariance, keys, and fingerprints are unchanged (up to
last-ulp reassociation) — but everything downstream of the basis (design matrices,
torque assembly, model files, Monte-Carlo contraction programs) shrinks by the
ordering multiplicity: measured 5.7–5.9× on the Nd₂Fe₁₄B `l044` production model
(this is also exactly Magesty's sorted-cluster bookkeeping, recovered here as a
single post-pass instead of being woven through the construction). Pre-v4 persisted
documents carry the redundant members and are folded on load (§ persist v4).

This was **cross-validated against Magesty**: for a kagome `P6/mmm` cell the number
of independent invariants per `(body, ls, Lf)` channel — a convention-free integer —
matches exactly through 3-body (all 42 channels, including the multi-term and
multi-path ones), and Magesty's own SALCs independently pass the invariance test. Two
from-scratch constructions agreeing on every invariant-subspace dimension is strong
mutual evidence that both are correct.

## 4. Canonical `SALCKey` column addressing (vs. implicit construction order)

In Magesty the design-matrix column order is the SALC iteration order, an implicit
"load-bearing" coupling that silently invalidates saved models if reordered. The
rebuild gives every SALC a structural `SALCKey` `(body, orbit_id, sorted ls, Lf,
block)`; the basis is sorted by key and the keys travel with the coefficients
(`SALCBasis.keys`, a gauge-independent `fingerprint`). A model re-pairs `jphi` to a
freshly built basis *by key*, not by position. (`orbit_id` is a stable ordinal
within a fixed `(crystal, spacegroup, cutoff)`, protected by the fingerprint.)

## 5. Pluggable backends via package extensions

Symmetry analysis and regularized estimators are behind abstract types
(`AbstractSymmetryBackend`, `AbstractEstimator`). The *types* live in the core
(`NoSymmetry`/`SpglibBackend`, `OLS`/`Ridge`/`Lasso`/`ElasticNet`), but heavy methods
live in extensions: `analyze_symmetry(::SpglibBackend, …)` is in `ext/` and lights up
only on `import Spglib`, just as `solve_coefficients(::ElasticNet, …)` lights up on
`using GLMNet` (see §12). The core thus loads with no heavy dependencies, and calling an
unloaded backend hits a friendly "load the backend" error on the abstract
supertype rather than a bare `MethodError`. This also removes Magesty's habit of
force-loading Spglib at `using` time.

## 6. Torque as the energy surface's exact derivative

The torque `τ_a = −e_a × ∂E/∂e_a` (the physical / Landau–Lifshitz torque
`m_a × B_eff,a`) is the SCE's second observable. Rather than
treat the torque design matrix as an independently-derived object (a place a sign
or normalization can silently drift out of step with the energy kernel), the
rebuild builds the per-site gradient `accumulate_grad!` from the *same*
`μ = idx − ls − 1` mapping, `ls`, `folded`, and `(4π)^(N/2)` scale as the energy
kernel `evaluate_salc` — only the innermost `Zₗᵢμᵢ` of each product is swapped for
`∇Zₗᵢμᵢ` (product rule, summed over the cluster's sites). The consequence is that
`predict_torque` is *by construction* the analytic (negative rotation-) gradient of
the surface `predict_energy` evaluates, and the gate is exactly that: an on-sphere
finite-difference check `predict_torque ≈ −e × ∇E_FD`, plus the Heisenberg closed
form `τ_a = J (Σ_{nn} e_j) × e_a`. This is convention-independent ground truth — it
would catch a self-consistent wrong convention that every gauge-invariant
comparison misses. (Magesty's reverse-mode, cluster-major gradient accumulator is
an optimization the v0 deliberately forgoes: the simple `O(N²)`-per-multi-index
leave-one-out product is clearer and fast enough for 2-body, and shares its inner
loop with `evaluate_salc` so the two cannot diverge.)

The co-fit follows Magesty: minimize `L = (1−w)·MSE_E + w·MSE_T` by whitening each
block (`√((1−w)/n_E)`, `√(w/n_T)`), with `j0` profiled out of the energy block
analytically — the torque block carries no intercept, so it never sees `j0`.

## 7. Persistence and input files (TOML, not XML/JSON)

Magesty persists the basis/model as **XML** (EzXML) and takes run setup from an
**`input.toml`**. The rebuild keeps the two-artifact split but unifies on one
zero-dependency format:

- **Input** (`io/input.jl`): a human-authored `input.toml` — `[structure]` (inline
  crystal: `lattice` as a list of the three lattice vectors, fractional `positions`,
  per-atom `species`, `species_labels`), `[interaction]` (`nbody`, `cutoff`, `lsum`,
  per-species `lmax`, `soc`), and optional `[symmetry]` (`backend`, `tol`).
  `SLCEBasis("input.toml")` builds the basis; keyword arguments override the file's
  backend/tol. Like Magesty, training data and the estimator are kept **out** of this
  file (loaded / chosen in Julia) — `input.toml` specifies only what defines the basis.
- **Persistence** (`io/persist.jl`): a **self-contained** TOML document — the crystal,
  the space-group ops, the basis spec, and the *full* SALC basis (every member, term,
  and folded tensor), plus (for a model) `j0` and per-`SALCKey` coefficients. Reload
  reconstructs the basis **verbatim** (no re-projection), so a saved model keeps working
  even if the construction code's gauge later changes — the file is the ground truth.
  Coefficients re-pair to the basis **by key** (§4), not by position; a scrambled
  on-disk coefficient order still reloads correctly.

Why TOML for both (the format was deliberated): the persistence artifact is a deep
nested numeric structure, which first suggested JSON. But Julia's **stdlib** `TOML`
round-trips `Float64` *exactly* (verified, including `-0.0` and `nextfloat(1.0)`) and
expresses the full nested SALC document, so a single zero-dependency format serves
both the human-written input and the machine-written dump — the most Julia-native
outcome. The `struct ⇄ Dict` schema layer (`_to_doc` / `_from_doc`) is kept
**format-agnostic** and is unit-tested with no serializer at all, so a JSON (or other)
backend would be a thin future addition rather than a rewrite. Two robustness details:
the `UInt64` structural fingerprint is stored as a *string* (TOML/JSON numbers lose
precision past `2^53`) and is **recomputed** on load rather than trusted (`hash` is
Julia-version dependent); and `-0.0` is normalized to `+0.0` on write so two builds of
the same object serialize byte-identically.

## 8. Result access as a Tables.jl source

Turning the fitted coefficients into a labeled table is the **library's** job, not the
caller's: only the package knows how to map internal storage (a `SALCKey` plus the
position of its coefficient in the `jphi` vector) to meaningful rows
(`body`, `orbit_id`, `ls`, `Lf`, `block`, `J`) — forcing a user to reach into
`f.dataset.basis.salc_basis.keys` would leak that. So `coeftable(fit | model)` returns an
`SCECoefficients` that implements the **Tables.jl** interface: it is a *data source*
that drops into whatever *sink* the caller chooses (`DataFrame`, `CSV.write`,
`Arrow.write`). The package depends only on the lightweight Tables.jl contract, never on
a table or IO package — the responsibility split (library owns the result semantics;
caller owns formatting/IO/plotting) falls exactly on the Tables.jl seam. The intercept
`j0` is exposed separately (`intercept`), not as a row, because it is not a per-cluster
quantity. The same contract is the natural entry point for the reverse direction
(ingesting tabular training data) if that lands later.

## 9. DFT-code-agnostic data boundary

The SCE pipeline must not care which DFT code produced its training data. So the
code-specific I/O is confined to a single boundary: the only objects the fitting
machinery consumes are `TrainingDatum` (per-configuration observables: energy + spin
directions + moment magnitudes, plus optional displacements / forces / constraining
field with the derived torque target, and a `DatumProvenance`) and the `SLCEDataset`
built from them; `SpinDatum(...)` survives as the spin-only convenience constructor.
Each DFT code is an `AbstractDFTSource` *adapter* implementing
`read_configs(src) -> Vector{TrainingDatum}`; the concrete adapters live **outside the
core**, as namespaced submodules of the companion SLCETools.jl package
(`SLCETools.VASP`, …), so nothing code-specific reaches the
core or its export list. Adding a code is one sibling submodule there — the core's public
surface does
not grow as codes multiply, and "once you hold the training data, its origin is
irrelevant" is enforced by the type structure, not merely by convention. (The same seam
is where a future third-party I/O package, or a format extension needing a dependency
such as `vasprun.xml` → EzXML, would plug in.) The VASP adapter mirrors Magesty's proven
POSCAR/OSZICAR conventions — including the torque target `τ_a = m_a × B_a` from the
constraining field (the same physical / Landau–Lifshitz torque as the model's
`predict_torque = −e_a × ∂E/∂e_a`) and the `Rz(α)·Ry(β)` SAXIS rotation — and is
cross-checked
bit-for-bit against Magesty's parsers in SLCETools's oracle, since real VASP outputs are
not vendored.

## 10. Oracle methodology

`test/oracle/` is a separate environment that `dev`s a pinned `Magesty.jl`; the
core suite never depends on Magesty. The oracle compares only **convention-fixed
kernels** bit-for-bit (`Zₗₘ`, Clebsch–Gordan vs. `WignerSymbols`, real Wigner-D,
coupled tensors) and **gauge-invariant aggregates** (space-group/spacegroup
numbers, orbit counts, held-out predictions, the recovered Heisenberg `J`). Raw
SALC coefficients and individual design-matrix columns are *never* compared — they
are gauge-dependent. Absolute conventions (signs, normalization) are pinned
independently by closed forms and finite differences, because a self-consistent
wrong convention would pass every gauge-invariant comparison.

## 11. Sunny export: conversion in the core, assembly in the extension

Exporting a fitted model to Sunny.jl (for linear spin-wave theory) is the one place an
SCE coefficient becomes a *physics-package object*, so a wrong factor would silently
corrupt a downstream Hamiltonian. Two decisions keep this safe.

**The conversion math lives in the core, the Sunny object in the extension.** Sunny is
a heavy optional dependency, so the `Sunny.System` assembly is a package extension
(`ext/SLCESunnyExt`, loaded by `using Sunny`, like the Spglib backend). But the
*numerically delicate* part — turning a folded tesseral tensor into a Cartesian matrix
with the right normalization (`_l1_pair_matrix = (3/4π)·folded` reproducing
`eₐ'·M·e_b = Σ folded·Z₁·Z₁`; the traceless-symmetric `_l2_onsite_matrix`), folding the
two directed cluster members into one matrix per undirected bond, and unfolding the
supercell onto the primitive cell — all sits in the **core** (the bilinear extraction in
`slce/bilinear.jl`, the primitive unfold in `interop/sunny.jl`), with no
Sunny dependency. It is gated by reconstructing the energy
(`_reconstruct_energy ≈ predict_energy − j0`) in the main test suite, so the conversion is
proven correct without ever loading Sunny; the extension only places the core-computed
matrices and is checked end-to-end (Sunny's own `energy`) in a separate environment.

**Only what Sunny can represent is exported — the rest is reported, not dropped.** Sunny's
bilinear-exchange-plus-single-ion model covers exactly the `ls=[1,1]` and `ls=[2]` SCE
channels; `ls=[0…]` is a constant (absorbed in `j0`), and every other SALC (3-body and up,
higher `l`) is collected into a `skipped` list and surfaced via `@warn`. Silently dropping
them would make an exported model look complete when it is not.

**Two cell routes.** The *supercell* route places the fitted matrices directly on the
training supercell (`set_exchange_at!`/`set_onsite_coupling_at!`, P1, inhomogeneous): exact
for any model, but the magnon dispersion is folded into the supercell Brillouin zone. The
*primitive* route recovers the chemical primitive cell from the space group's pure
translations, groups supercell atoms into sublattices, and places **one Sunny bond per
primitive bond without multiplicity** — Sunny's periodic replication then restores the
supercell multiplicity — giving the unfolded dispersion. This is exact only when the model
genuinely lives on the primitive cell (the interaction range stays below the supercell
boundary, the same `L/2` resolvability limit as §1b); a `clean` flag detects when it does
not and falls back to the exact supercell route. Spin enters only at assembly: the SCE
couplings are fit with unit directions, so the `scaling = :moment` route sets
`J = M/(SₐS_b)` to make Sunny's length-`S` dipoles
reproduce the unit-vector energy (half-integer `S_eff` only — Sunny's `Moment`
constraint), while `scaling = :coupling` holds `Moment` at a placeholder `s₀ = 1` and
folds `S_eff` into the couplings instead (any positive `S_eff`; dispersion-exact, static
energy rescaled). The single-ion term carries the classical
(`:dipole_uncorrected`) or quantum rank-2 (`:dipole`) rescaling.

## 12. GLMNet estimators: sparse selection inside the centered-`X` contract

An SCE basis with many channels (high `l`, several body orders) easily out-runs the
available DFT configurations, so a sparse / shrinkage estimator that selects which
couplings matter is the natural next estimator after `OLS`/`Ridge`. GLMNet (the Fortran
elastic-net) is a heavy, binary-backed dependency, so it follows the §5 pattern exactly:
the `ElasticNet` / `Lasso` *types* and their argument validation live in the core
(named, dispatched on, tested without the dependency), and only `solve_coefficients(::
ElasticNet, …)` lives in `ext/SLCEGLMNetExt` behind `using GLMNet`.

The subtle part is the existing estimator contract: `fit` hands `solve_coefficients` a
**column-centered** `X` and centered `y` and recovers `j0` analytically afterwards, so
the solver must add no intercept of its own. GLMNet is therefore called with
`intercept = false` — on already-centered data the penalized minimizer is identical
with or without a modelled intercept, and `intercept = false` keeps the contract literal
(verified: recovered `a0 ≈ 1e-16`, `max|Δβ| = 0`). Column `standardize` is left on (the
conventional Lasso default) so the L1 penalty is even-handed across SALC columns of very
different scale; GLMNet returns β on the original scale, and the penalty effectively acts
per-column at `λ·std(colⱼ)`. This `λ` is on GLMNet's `(1/2n)·RSS + λ·penalty` scale,
**not** comparable to `Ridge`'s plain `λ‖β‖₂²`.

Two refinements make the default (`lambda = nothing`, cross-validated) path trustworthy
for *this* problem rather than generic tabular data:

- **Configuration-grouped CV.** A co-fit stacks each configuration's energy row and its
  `3·n_atoms` torque-component rows; all are deterministic functions of the same spins
  and the same `β`. Plain row-wise CV folds would scatter them, so holding out part of a
  configuration while training on the rest leaks within-configuration structure and makes
  the CV error optimistically low — biasing λ selection toward under-regularization on
  the package's headline use case. `fit` therefore passes per-row `groups` labels (a
  configuration's energy and torque rows share one label) and the extension folds over
  whole groups, never splitting a configuration. The contract gains one optional
  keyword; `OLS`/`Ridge` ignore it.
- **Reproducible folds without a `Random` dependency.** An extension may only use the
  package's own (weak)deps, and pulling in `Random` just to seed `MersenneTwister` is
  unwarranted. Folds are instead assigned by ranking the resampling units by a seeded
  `hash` and dealing the ranks round-robin — balanced, seed-controlled, deterministic
  within a Julia session/version (cross-version reproducibility is not promised, and is
  not needed, since folds are never persisted).

Because the conversion is a pure β-passthrough (no sign, unit, or layout convention is
touched), correctness reduces to two intrinsic checks in the separate `test/glmnet/`
env: the analytic-`j0` centering invariant (`mean(predict) = ȳ`, estimator-independent)
and support recovery — a 1-sparse signal in a 44-column basis is recovered (the signal
column dominates, the model is genuinely sparse), with `:lambda_1se` shrinking more than
`:lambda_min`.

## 13. Cost-weighted group selection: Group Adaptive Ridge + GCV + a Pareto λ rule

**The objective is a weighted group-L0, so approximate that — not a surrogate.** A
Monte-Carlo sweep over a fitted SCE pays per surviving *contraction entry*, and an
entry is shared by every SALC of one `(body, orbit_id, ls)` group (the MC engine folds
their coefficients into a single contraction weight). The entry vanishes only when the
*whole group* is zero, so the real objective is

    ‖y − Xβ‖² + λ·Σ_g c_g·1{β_g ≠ 0}

with `c_g` the group's a-priori cost — the union count of its distinct
`(member sites, l-assignment, nonzero tensor index)` entries over the canonical (v4)
members. Distinct groups never share an entry key, so the costs are additive and the
"predicted MC cost" of any support is a plain sum. Neither an L1 group penalty (the
convex relaxation) nor per-column selection matches this: shrinking a surviving group
saves nothing, and killing single columns of a group saves nothing either.

**Why iterated reweighted ridge in core.** `GroupAdaptiveRidge` extends the in-tree
`AdaptiveRidge` (Frommlet & Nuel 2016; the group form follows Goepp–Bouaziz–Nuel's use
for knot selection) rather than adding a block-coordinate-descent group-lasso solver:
every step is the same analytic SPD solve `(X'X + λ·Diagonal(w))\X'y` the package
already trusts, it needs no groupwise orthonormalization and no GLMNet, it
warm-starts trivially along a λ path (the Gram is formed once), and — decisively — it
targets the group-L0 objective directly instead of its convex relaxation. The weight
map is

    wⱼ = v_g / (‖β_g‖² + p_g·ε)

and the denominator form is deliberate. At a fixed point a surviving group's penalty
contribution is `λ·v_g·‖β_g‖²/(‖β_g‖² + p_g ε) → λ·v_g`: the fixed multiplier `v_g`
**is** the group-L0 weight, so pricing groups by cost means simply setting
`v_g = √p_g·(c_g/c̄)^θ` (`√p_g` is the Yuan–Lin group-size factor; `θ` tilts from
cost-blind to cost-proportional and changes the *order* in which groups die). Writing
the denominator as `p_g·(mean_j βⱼ² + ε)` also keeps `ε` a per-coefficient magnitude
floor independent of group size — the same calibration as `AdaptiveRidge`'s `βⱼ² + ε`,
to which the update degenerates exactly for singleton groups with unit weights (pinned
by test). The trade-off versus a group lasso is theory: no convexity, selection
consistency is empirical. That is the same trade already accepted for `AdaptiveRidge`.

**GCV from the closed-form hat matrix — computed on the smaller Gram side.** The
converged fit is linear in `y` with the weights frozen (`islinear`), so
`df = tr(X(X'X + λD)⁻¹X')` is exact in that standard converged-weight sense and GCV
`n·RSS/(n − df)²` needs no refitting. The trace is evaluated as `Σᵢ sᵢ/(sᵢ + λ)` over
the eigenvalues of the weighted Gram `X̃'X̃` (`p ≤ n`, reusing the λ-path's cached
`X'X`) or its `n×n` dual (`n < p`) — never an `n×p` SVD, which would dominate the path
cost in the co-fit regime. Two honesty guards: the score is `Inf` once `df → n` (a
near-interpolating fit has no GCV-selectable error), and on torque co-fits GCV is
*optimistic* because the energy and torque rows of one configuration are correlated
while GCV treats rows as exchangeable — the same leak configuration-grouped CV folds
exist to prevent, so `select_fit(...; criterion = :cv)` (grouped K-fold in core,
deterministic seeded folds, per-fold Gram downdates) is the ground truth there and the
GCV docstring says so.

**Selection is a Pareto rule, not CV-min.** Minimizing the score alone answers the
wrong question — it ignores what the surviving model costs to *run*. `select_fit`
therefore reports the whole path (score, effective dof, alive groups, predicted cost
per λ) and selects the **cheapest** λ whose score is within `(1 + δ)` of the path
minimum: the direct cost-aware generalization of the conventional `:lambda_1se`
one-standard-error rule, with `δ` an explicit accuracy tolerance instead of a
fold-noise estimate. Together with `θ` this exposes the full trade surface — sweep `θ`,
take the lower envelope of the per-θ paths, and the (cost, error) Pareto front is
explicit rather than implicit in a penalty default. The selected fit is re-solved cold
at the chosen λ so that `fit(SLCEFit, dataset, estimator)` reproduces it verbatim, and
the de-biasing `refit` closes the workflow.

**Postscript: the threshold is the second knob (`select_support`).** The clean
alive/dead magnitude gap that makes the λ path's relative floor sufficient exists on
group-sparse *synthetic* targets; on real DFT data the per-group magnitude spectrum
is continuous (production l044: the sorted spectrum decays smoothly over four orders
with no knee), so along the λ path nothing ever dies and GCV simply prefers the full
model. The trade the λ path cannot see is exposed by sweeping the support threshold
directly at a fixed fit: each point is one cheap de-biasing `refit`, scored on a
held-out slice, with the same "cheapest within (1+δ)" rule — `select_support`. On
l044 this front offered 38 % of the Monte-Carlo cost at a held-out torque RMSE
*better* than the full model (de-biasing beats the interpolating tail) and 3 % of the
cost at +19 %. The λ path (`select_fit`) remains the tool that *shapes* coefficients
group-wise before thresholding; the threshold front is where the cost is actually
harvested.

## 14. Sector tables: one union, one projector, per-sector SOC

The joint spin–lattice truncation is a **union of sectors** (`Sector` rows →
resolved `SectorRule`s), not a product grid: the physically meaningful models —
force constants ∪ pure-spin SCE ∪ a handful of coupled rows — are sparse in the
(spin content × displacement degree × body order × radius) product, and a
product-grid spec would either admit the whole block or force per-cell
exceptions. Three decisions carry the design:

- **`soc` is a per-sector truncation rule, not a second group action.** There
  is exactly one projection (the grey magnetic group). `soc = false` keeps the
  `L_S = 0` columns of that one basis — a theorem-backed *subset*, exact
  together with the always-on even-`Σl_spin` screen (the screen is what kills
  the T-odd scalar chirality `ê·(ê×ê)`, which is `L_S = 0` but Σl-odd). This is
  also why the dense form's old `isotropy` keyword (an `Lf = 0` filter) had to
  be *replaced* rather than generalized: at `p = 0` the two coincide
  (`L_S ≡ Lf`), but a dJ/dr-type sector is `L_S = 0, Lf ≠ 0` and an `Lf`
  filter would wrongly kill it. The rename carries an inversion
  (`isotropy = true` ⇔ `soc = false`), so both the keyword and the TOML key
  fail loudly with the mapping in the message.
- **The union is a set-union on labels, projected once.** Per orbit, the
  admitting sectors' label sets are merged with an OR over their `soc` flags
  before projection; a label contributed by several sectors is projected
  exactly once, so overlapping rows can never produce duplicate — hence
  exactly collinear — design columns. `build_salc_basis` asserts key
  uniqueness after the sort (the key-union invariant).
- **Cutoffs: one envelope for the cluster set, per-sector re-admission.** The
  cluster orbits are built once at the per-body elementwise-max envelope; each
  sector then re-admits an orbit by comparing the same banded gate distances
  `candidate_clusters` uses (MinimumImage: the pair's minimum-image distance;
  AllImages: each edge), so a WS-boundary tie shell is kept or dropped whole
  by every sector, never split.

Species-resolved placement is deliberately split between generation and
projection: a multiset label does not know which site carries which decor, so
the generators cap by the loosest species present, and the decor engine's
per-permutation-orbit filter (`_admit_assignment`, checked on the canonical
representative — species are stabilizer-invariant) makes the per-site
`lmax`/`pmax` decision exactly where `_enumerate_ls` would have made it,
keeping `block` indices aligned between the pure-spin and decor engines.

## 15. Invariance layers: what the SALC projection does and does not enforce

There are **two independent layers** of invariance in a lattice-dynamics model,
and this package implements one and a half of them. Confusing the layers is the
single easiest way to over-claim what a fitted `SLCEModel` guarantees.

> **On the numbers in this section.** Every figure below — the principal-angle
> cosines, the condition-number ratio, `rank[Huang; Q] = 21`, the `1e-14`
> stress-condition residual — was measured externally, in the phonon paper's
> scratch scripts, and **none of them is gated by a test in this repository.**
> They are a record of those experiments, not a guarantee the suite maintains.
> The one physical consequence that *is* gated here is the acoustic-zero check on
> `dynamical_matrix` (`test/unit/test_forceconstants.jl`).

**Layer 1 — space-group symmetry. Implemented by the SALC projection, and it is
the standard construction, not a variant.** The symmetry-adapted space of
harmonic force constants reachable through
`force_constants(model; spins, order = 2)` is *bit-for-subspace* the same
as the textbook Cartesian reduction — impose covariance
`Φ(g·pair) = D(g) Φ(pair) D(g)ᵗ` plus the transpose relation
`Φ^{αβ}((a,0),(b,R)) = Φ^{βα}((b,0),(a,−R))` and take the null space. The union of
the two spaces has the same dimension as each, and every principal-angle cosine is
`1.000000000000`, on both fixtures below (a displacement-only sector,
`Sector(disp = (degree = 2,), nbody = 2, cutoff = c)` ∪ its `nbody = 1` partner,
`SpglibBackend`, pair set taken from `force_constants` so both sides use exactly
the same index tuples):

| fixture | space group | SALCs | SLCE image | Cartesian null space | union |
|---|---|---|---|---|---|
| `Crystal(Lattice(3.0·I), [1/6 −1/6; 0 0; 0 0], [1,1], ["Fe"])`, `c = 1.1` | P4/mmm (#123), 16 ops | 4 | 4 | 4 | **4** |
| `Crystal(Lattice([3.0 0.4 −0.2; 0 3.3 0.5; 0 0 2.9]), [0 0.37; 0 0.21; 0 0.44], [1,1], ["Fe"])`, `c = 3.0` | P-1 (#2), 2 ops | 12 | 12 | 12 | **12** |

Consequence, and it cuts both ways:

- The SALC basis has **no symmetry-handling deficit** relative to ALM / hiPhive /
  phonopy-style Cartesian reduction. It is a different gauge of the same space.
- It has **no numerical advantage either, and this must never be claimed.**
  Singular values are invariant under an orthogonal change of basis, and the
  angular part of the irreducible basis (`3 ⊗ 3 = D⁰ ⊕ D¹ ⊕ D²`, real spherical,
  Frobenius-orthonormal) *is* orthogonal — measured cond ratio
  (Cartesian frame / irreducible frame) `= 1.000000` on an ASR + Born–Huang
  system. The full SALC map additionally mixes orbit members and carries radial
  factors, so it is orthogonal *times a diagonal*; a diagonal does move the
  condition number, but that is column scaling, which equilibration fixes in any
  basis. Any apparent conditioning win is a normalization artifact.

**Layer 2 — affine invariance. This is NOT a space-group property, and only the
translation half is implemented.** The relevant operations are uniform motions of
all atoms *with the cell held fixed*:

| operation | condition | in this package |
|---|---|---|
| `u → u + t` | acoustic sum rule | **`build_asr`** (+ `asr_residual`) |
| `u → u + ε_antisym·R` | Born–Huang rotational | **not implemented** |
| `u → u + ε_sym·R` | equilibrium / vanishing stress | **not implemented** |

None of the three follows from the SALC projection, and that is not an oversight
in the projection: a rigid rotation of the whole crystal by an arbitrary
continuous angle is not an operation of the crystal's space group, so it cannot
be projected out at basis-construction time. That is exactly why the ASR lives as
a *separate* linear constraint on coefficients (`ASRReparam`) rather than inside
`build_salc_basis`. **O(3)-equivariance of the displacement expansion is not
physical rotational invariance** — the SALC basis is invariant under the
*crystal's* point group and says nothing about the energy cost of a rigid
rotation.

The two missing families are genuinely independent of each other and of the
15 Huang (pair-exchange) conditions: in the harmonic Cartesian setting
`rank[Huang; Q] = 21 = 15 + 6`, where `Q` is the rigid-rotation quadratic form.
The minimal witness needs no rank computation — simple cubic, nearest neighbours,
`Φ_d = 1`: the Huang residual is identically zero while `Q = 4·1 ≠ 0`.

This is not new physics — it is **Gazis & Wallis, Phys. Rev. 151, 578 (1966)**,
whose abstract states that the Born–Huang conditions "are necessary but not
sufficient for rotational invariance" and whose Eq. (12) is the `Q = 0` family.
But it is contested in the literature in a way that matters here, and the
resolution is domain-dependent. **Sarkar & Sengupta, phys. stat. sol. (b) 83, 263
(1977)** argue the extra conditions are "nothing new" — merely Born–Huang plus
vanishing stress — and modern reviews inherit that (Lin, Poncé & Marzari, npj
Comput. Mater. 8, 236 (2022)). Measured: on a **central-force-parametrized** model
their claim is exactly right, including their count — the 15 Huang functionals
collapse to rank 5 there, the anisotropic stress lies in that span, and only the
isotropic pressure escapes (which S&S themselves except). On a **freely fitted**
`Φ_d` — a general symmetric 3×3 per shell, which is what this package and every
other force-constant fitter produces — the Huang rank returns to 15 and both parts
of `Q` leave its span. **So the "Born–Huang is complete" result does not cover the
regime this package operates in.** That is the concrete reason the layer-2 gap
above is a real gap and not a bookkeeping one.

The freely-fitted regime is the "general tensor force" model of the older
literature, and the warning is explicit there — **Ramamurthy, Phys. Rev. B 57,
13554 (1998)**: the GTF model "is not at all invariant with respect to rigid body
rotations even in the case of high symmetry cubic crystals… the harmonic force
constants of the GTF model comply with Born–Huang conditions so long as the
crystal is in equilibrium, but they fail to satisfy the additional conditions that
reduce the change in potential energy to zero", with two consequences he names:
elastic inconsistency (two different expressions for `C₄₄` in noncubic crystals)
and the Cauchy discrepancies vanishing once all invariance conditions are imposed.
Both are reproducible on this package's own fitted models. Cite that paper, not a
derivation, if the gap ever needs justifying to a reviewer.

**When the gap is benign, and when it is not.**

- **Harmonic order, training data from a relaxed structure: benign.** The six
  `ε_sym` conditions *are* the vanishing-stress conditions — for central-force
  models `Q_ij = σ̃_ij − δ_ij tr σ̃` holds exactly (verified to `1e-14`), and that
  map is invertible, so `Q = 0 ⇔ σ = 0`. A DFT relaxation to `σ ≈ 0` already
  enforces them; nothing needs to be imposed.
  **In a spin–lattice model that escape is narrower than it looks, and it is worth
  being explicit about why** (added 2026-07-27, from the M5-0 review): the reference
  stress is a function of the spin configuration, `σ = σ(ê)`. A cell relaxed in one
  magnetic state is *stressed* in every other, so the data enforce the six conditions
  for that one state and for no other — while the flagship deliverable of the joint
  expansion is precisely the magnetic-state DIFFERENCE of the force constants
  (`force_constants` at two configurations). The same fact is why second-order
  elastic constants are Seth–Hill-measure-independent only at a stress-free
  reference and therefore never unconditionally so here; see design record §13
  risk 2.
- **Anharmonic orders: not automatic.** The rotational condition ties `Φ⁽ⁿ⁾` to
  `Φ⁽ⁿ⁺¹⁾`, so it is not satisfied order by order.
- **The strain channel, where `ε` is an explicit model variable: this is where it
  bites.** There the affine field is the model's own coordinate, so the layer-2
  conditions become statements about the fitted couplings rather than about the
  training geometry. **Which part of M5 that is, precisely** (sharpened
  2026-07-27): NOT the v0 strain work. In design record §9a the K(ε) grid is a
  family of separately fitted models and `ε` is a *label* of a grid point, never a
  fit coordinate; in §9e it is pinned as a Biot measure applied to build geometries.
  The layer-2 conditions bite as a *constraint* question only in §9b, the explicit
  global strain channel — which is exactly why §9b is gated on resolving its
  rotational-invariance hole first. For the v0 deliverables (`force_constants`,
  `exchange_strain_derivatives`, `magnetoelastic_constants`) the missing layer
  shows up as a **diagnostic on a derived quantity**: the symmetrized `∂J/∂ε` that
  gets reported is unaffected, while the antisymmetric part it drops is the
  rotational content, which a fitted model is free to get wrong.

**If it is ever implemented**, three constraints on the design, in decreasing
order of how easy they are to get wrong:

1. It goes **next to `build_asr`**, as a second constraint layer feeding the same
   `ASRReparam` mechanism (or a generalization of it) — never into the SALC
   builder, for the reason above.
2. **The truncation boundary over-constrains.** Because the condition couples
   order `n` to `n + 1`, at the top retained order the `Φ⁽ⁿ⁺¹⁾` term is absent and
   the constraint degenerates to `L·Φ⁽ⁿ⁾ = 0`, whose kernel is only the
   SO(3)-invariant subspace: dimensions `1, 1, 3, 6, 15` for `n = 2…6` in the
   generic index-symmetry type and `1, 0, 1, 0, 1` in the fully symmetric
   (on-site) type — i.e. **at odd order the fully symmetric part is forced to
   vanish entirely.** This is the exact, channel-resolved form of ALAMODE's
   `ICONST = 3` warning. Either impose from the top order downward
   (Esfarjani–Stokes, PRB 77, 144112) or exclude the top order.
3. The physical acceptance gate already exists in spirit: an ASR-fitted model has
   exactly three zero eigenvalues of
   `dynamical_matrix(force_constants(model; spins), [0, 0, 0])` (`dynamical_matrix`
   takes a `ForceConstantSet` and a three-component fractional `q`, not a model and
   a scalar). The rotational analogue is that a rigid rotation costs zero energy —
   testable directly through the evaluator at **finite** `ω`, the same way
   `test_asr.jl` tests finite-`t` translation invariance. There is no
   `rotational_residual` today; that is the shape it would take, and design record
   §12 gate (q) now reserves it.

Cross-reference: the derivation and the numerics behind this note live outside
this repo, in the phonon paper's working document
(`~/Documents/research/papers/writing/SphericalTensorForceConstants/literature.md`,
§7 seventh and eighth status updates). The spin-side counterpart — that in a
SOC sector the lattice's axial `l = 1` block does not vanish but *transfers* to
the spin channel, `𝓡_U E = −𝓡_S E`, block diagonal in `L_S` — is derived in the
SLCE paper; it is a **diagnostic**, not an imposed constraint.

**Correction (2026-07-27): that transfer is NOT gated by `test_sectorbasis.jl`, and
reading it as gated is the trap this section exists to prevent.** What that test
asserts (`:409-413`) is `evaluate_salc(s, e, ucfg(W)) < 1e-12` for antisymmetric
`W` — the LINEAR part of a rotation only, on an O_h fixture, and its own comment
says the kill is "an exact O_h statement". Both sides of the identity vanish there
for a reason that does not generalize: cubic symmetry forbids `l = 2` single-ion
anisotropy, so `𝓡_S E = 0` and hence `𝓡_U E = 0`. The identity acquires content
only where a rank-two on-site anisotropy is allowed (a D₄h-class fixture), which is
design record §12 gate (r); the finite-`ω` half is gate (q). Until those exist,
the honest statement is that the transfer is derived and unverified in code.
