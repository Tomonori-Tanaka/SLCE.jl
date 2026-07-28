# Data and fitting

```@meta
CurrentModule = SLCE
```

With a basis in hand, the second half of the workflow pairs it with DFT data, fits the
coefficients, and reports how well the fit did. The objects are [`SLCEDataset`](@ref),
[`fit`](@ref) with a pluggable estimator, and [`SLCEFit`](@ref) / [`SLCEModel`](@ref).

## Datasets

A [`SLCEDataset`](@ref) materializes the energy design matrix `X_E[config, salc] =
Φ_salc(config)` from a list of spin configurations (each `3 × n_atoms`, unit columns) and
their energies:

```julia
dataset = SLCEDataset(basis, configs, energies)
```

The four-argument form additionally takes per-configuration torques and builds the torque
design matrix for an energy + torque co-fit:

```julia
dataset = SLCEDataset(basis, configs, energies, torques)   # torques: each 3 × n_atoms
```

You can also go straight from a DFT source (see [Persistence and I/O](io.md)):
`SLCEDataset(basis, src)`. On that path every atom the SALC basis references must
carry a nonzero magnetic moment in every configuration — a quenched moment would
enter the fit through a placeholder direction and silently bias it, so it is an
error. Species that are genuinely non-magnetic belong outside the basis
(`lmax = 0`), and their moments are then never consulted.

### Slicing and concatenation

Datasets slice by configuration and concatenate without recomputing design
matrices, which makes train/test splits, filtering, and incremental data addition
cheap:

```julia
train, test = dataset[1:80], dataset[81:end]   # ranges, index vectors, Bool masks, :
f = fit(SLCEFit, train, OLS())
rmse_holdout = sqrt(sum(abs2, predict_energy(f, test.configs) .- test.y_E) / length(test))

more = SLCEDataset(basis2, new_configs, new_energies)
dataset = vcat(dataset, more)     # basis2 must be the same basis (fingerprint-checked)
```

`vcat` accepts parts built on a persisted-and-reloaded basis (the check is the
SALC-basis fingerprint, not object identity); mixing torque-bearing and
energy-only datasets is an error.

## The fit and the analytic intercept

[`fit`](@ref) column-centers the design matrix, so the reference energy `j0` is recovered
analytically as `mean(y_E − X_E·jϕ)` — *independent of the estimator* — and the centered
problem is handed to [`solve_coefficients`](@ref). Every estimator therefore returns only
the slope coefficients and adds no intercept of its own.

```julia
f  = fit(SLCEFit, dataset, OLS())
J  = coef(f)         # the fitted coefficients jϕ, in SALCKey order
j0 = intercept(f)    # the reference energy
```

## Energy + torque co-fit

The SCE's second observable is the per-atom torque
``\boldsymbol\tau_a = -\hat{\boldsymbol e}_a \times \partial E / \partial \hat{\boldsymbol e}_a``
(the Landau–Lifshitz / physical torque ``\boldsymbol m_a \times \boldsymbol B_{\mathrm{eff},a}``).
Because [`predict_torque`](@ref) is the analytic gradient of the same surface
[`predict_energy`](@ref) evaluates, the two are consistent by construction. A
`torque_weight ∈ (0, 1]` runs a co-fit that minimizes

```math
L = (1 - w)\,\mathrm{MSE}_{\text{energy}} + w\,\mathrm{MSE}_{\text{torque}},
```

implemented by whitening: the centered energy block is row-scaled by ``\sqrt{(1-w)/n_E}``
and the torque block by ``\sqrt{w/n_T}``, then stacked. `j0` does not enter the torque
block, so it stays an energy-only quantity.

```julia
f = fit(SLCEFit, SLCEDataset(basis, configs, energies, torques), OLS(); torque_weight = 0.5)
(r2_energy(f), r2_torque(f))
```

The [Getting started](../getting_started.md#Add-the-torque) page runs a full co-fit
end-to-end.

## Forces: the three-block co-fit

Against a displacement-decorated (joint) basis a third observable enters: the
per-atom force ``\boldsymbol f_a = -\partial E/\partial \boldsymbol u_a``
([`predict_force`](@ref), again the analytic derivative of the same surface). A
dataset built from `TrainingDatum` vectors with force channels carries the compact
force design block (see [`SLCEDataset`](@ref)), and

```julia
f = fit(SLCEFit, dataset, OLS(); torque_weight = 0.2, force_weight = 0.5)
(r2_energy(f), r2_torque(f), r2_force(f))
```

minimizes ``L = (1 - w_T - w_F)\,\mathrm{MSE}_E + w_T\,\mathrm{MSE}_T +
w_F\,\mathrm{MSE}_F`` (each weight in ``[0, 1]``, ``w_T + w_F \le 1``; the force
block is whitened by ``\sqrt{w_F/n_F}``). Note the three blocks carry different
units (eV², eV², (eV/Å)²), so the weights are **not** scale-free — a force RMS much
larger than the energy spread makes even a modest ``w_F`` dominate the fit; compare
`rmse_energy`/`rmse_force` at a trial weight before committing to one. Force rows
exist only for force-bearing configurations and displacement-referenced atoms —
never zero-padded — and the selection layer ([`select_fit`](@ref) /
[`cross_validate`](@ref) / [`select_support`](@ref)) does not take a `force_weight`
yet: it scores the energy(+torque) objective only.

## The acoustic sum rule (translation invariance)

A joint model's energy must be invariant under a rigid translation of the whole
crystal, ``\boldsymbol u_a \to \boldsymbol u_a + \boldsymbol t`` — the acoustic
sum rule (ASR; equivalently ``\sum_a \boldsymbol f_a = 0``). The site-factorized
basis does not guarantee this by itself, so `fit` enforces it **exactly** by
default (`asr = true`): the coefficients are solved in the null space of the
translation-constraint matrix (`β = Z·γ`, `A·β = 0` by construction — never a
penalty). Pure-spin bases have no displacement content and are unaffected
(bitwise). The residual actually achieved is stored on the fit
(`f.asr_residual`), and any model — including a hand-built or reloaded one — can
be verified with [`asr_residual`](@ref)`(model)`; downstream physical consumers
(force constants, Monte-Carlo ingest of joint models) gate on that value rather
than trusting a flag. `asr = false` fits unconstrained, for ablations only: on
noisy data the resulting model demonstrably gains energy under rigid
translations.

The constraint is not only about physical propriety — it is often what makes the
fit *identifiable at all*. Displacement samples that leave the center of mass
fixed (the natural DFT protocol) cannot see the translation-violating directions:
the torque channel is blind to every spin-independent direction (which, for a
basis whose displacement content is all spin-decorated, is exactly the violating
subspace — a lattice-only sector adds a residue no constraint can cure, so force
constants are never identifiable from torques alone), and the force channel sees
only the part whose `Σ_a f_a` does not vanish on the sampled slice. A
derivative-only fit (`torque_weight + force_weight = 1`, so the energy data enter
only through the analytic `j0`) is therefore rank-deficient without the ASR and of
full rank with it: the constrained fit recovers a known model to machine
precision, while the unconstrained one lands on a materially different coefficient
vector that reproduces every training force and torque just as well — and then
disagrees as soon as a configuration drifts off the sampled slice. Use
[`identifiability`](@ref) to see this ledger for your own dataset before or after
fitting.

Two practical notes. A too-tight truncation can admit **no** translation-invariant
displacement content at all (e.g. bond pair terms without their on-site
partners): the fit then warns and constrains every displacement coefficient to
zero — widen the per-species `pmax` or the displacement sectors. And L1
estimators (`Lasso`/`ElasticNet`/`AdaptiveLasso`) cannot solve the constrained
problem (they error, pointing at [`GroupAdaptiveRidge`](@ref)); the quadratic
and adaptive-ridge estimators work unchanged, with their penalties still defined
on the physical β.

## Estimators

The estimator is the regression strategy, dispatched on [`AbstractEstimator`](@ref). Four
are in-tree (closed form, no dependencies):

| Estimator | Penalty | Notes |
|-----------|---------|-------|
| [`OLS`](@ref) | none | ordinary least squares (QR) |
| [`Ridge`](@ref) | ``\lambda\lVert\beta\rVert_2^2`` | L2, closed form |
| [`AdaptiveRidge`](@ref) | ``\lambda\sum_j w_j\beta_j^2`` | iterative reweighted ridge, an L0 approximation |
| [`GroupAdaptiveRidge`](@ref) | as above, ``w`` shared per group | group-L0, fixed cost weights |

[`AdaptiveRidge`](@ref) (Frommlet & Nuel 2016) repeatedly refits a per-coefficient
weighted ridge with ``w_j = 1/(\beta_j^2 + \varepsilon)``, so large coefficients get a
light penalty and small ones a heavy penalty — iterating drives the small ones toward
zero. Each subproblem is the analytic weighted ridge, so it needs no extension:

```julia
fit(SLCEFit, dataset, AdaptiveRidge(lambda = 1e-3))     # L0-like selection, closed form
```

[`GroupAdaptiveRidge`](@ref) is its group extension: all columns of a group share one
weight ``w_j = v_g/(\lVert\beta_g\rVert^2 + p_g\varepsilon)``, so whole groups — not
individual columns — are driven to zero, and the fixed multiplier ``v_g`` prices each
group (at convergence a surviving group pays exactly ``\lambda v_g``). This is the
estimator behind the Monte-Carlo-cost-aware selection workflow below.

The penalized-path estimators — the Lasso, the elastic net, and the adaptive Lasso — are
provided by a **GLMNet extension** that lights up under `using GLMNet`:

```julia
using GLMNet                               # activates the estimator extension

fit(SLCEFit, dataset, Lasso())                          # CV-selected λ, sparse model
fit(SLCEFit, dataset, Lasso(select = :lambda_1se))      # the parsimonious 1-SE model
fit(SLCEFit, dataset, ElasticNet(alpha = 0.5))          # an L1/L2 mix
fit(SLCEFit, dataset, Lasso(lambda = 1e-3))             # a fixed penalty (no CV)
fit(SLCEFit, dataset, AdaptiveLasso())                  # data-driven reweighted Lasso
```

[`ElasticNet`](@ref) minimizes GLMNet's
``\tfrac{1}{2n}\lVert y - X\beta\rVert^2 + \lambda[(1-\alpha)/2\,\lVert\beta\rVert_2^2 +
\alpha\,\lVert\beta\rVert_1]`` on the centered design (so `j0` stays analytic), with
column standardization. With `lambda = nothing` the penalty is chosen by `nfolds`-fold
cross-validation (`select = :lambda_min` or `:lambda_1se`); a numeric `lambda` fits at
exactly that penalty. For a co-fit the CV folds are **grouped by configuration** so that a
configuration's energy and torque rows never split across folds.

[`AdaptiveLasso`](@ref) (Zou 2006) runs a `pilot` estimator first, then a weighted Lasso
with per-column penalty factors ``w_j = 1/\max(|\hat\beta_j^{\text{pilot}}|,
\varepsilon)^\gamma`` — penalizing columns the pilot found small and sparing those it
found large, which gives it oracle selection. `gamma = 0` reduces to a plain Lasso; the
pilot defaults to [`OLS`](@ref) but any estimator works, including a
[`PrecomputedPilot`](@ref) that reuses a prior fit's coefficients:

```julia
fit(SLCEFit, dataset, AdaptiveLasso(gamma = 1.0))                       # OLS pilot (Zou 2006)
fit(SLCEFit, dataset, AdaptiveLasso(pilot = Ridge(lambda = 1e-4)))      # for ill-conditioned designs
fit(SLCEFit, dataset, AdaptiveLasso(pilot = PrecomputedPilot(coef(prior)), lambda = 1e-3))
```

It shares `lambda` / `standardize` / CV behavior with [`ElasticNet`](@ref) (`lambda =
nothing` selects λ by configuration-grouped CV with the adaptive weights held fixed).

Implementing your own estimator is one method:

```julia
struct MyEstimator <: AbstractEstimator end
SLCE.solve_coefficients(::MyEstimator, X, y; groups = nothing) = X \ y  # centered (X, y)
```

## Staged (hierarchical) fits

A joint model can be built in physical stages instead of one shot: fit the
exchange first, then the spin–lattice coupling against the frozen exchange, then
the force constants. `sector_mask` says which columns a stage fits and `frozen`
supplies the coefficients of the previous stage:

```julia
f1 = fit(SLCEFit, ds, OLS(); sector_mask = :spin)
f2 = fit(SLCEFit, ds, OLS(); sector_mask = :coupled, frozen = SLCEModel(f1),
         torque_weight = 0.3, force_weight = 0.3)
f3 = fit(SLCEFit, ds, OLS(); sector_mask = :lattice, frozen = SLCEModel(f2),
         torque_weight = 0.3, force_weight = 0.3)
```

Selectors are `:all`, `:spin`, `:lattice`, `:coupled` (a partition by channel),
and `:soc_free` / `:soc` (a crosscutting partition by `L_S`); a collection of them
is their union, and an explicit column list or `Bool` mask also works. Inspect a
plan before running it with [`SLCE.sector_columns`](@ref). Frozen coefficients are
matched by `SALCKey`, never positionally, and a frozen value on a column the mask
leaves free is ignored — that column is being re-fitted. `j0` is never frozen.

Staging is **not** the same as truncation. `Sector(soc = false)` decides what the
model can express (it never builds those columns); `sector_mask = :soc_free`
decides what *this stage* fits, leaving the rest at their frozen values. The two
share one predicate, so they always name the same content.

Nor is a chain of stages the same as one joint fit: an earlier stage absorbs
whatever the later columns would have explained, so the two agree only when the
blocks are orthogonal in the design. Stages buy control and conditioning, not the
same optimum.

What the ASR contributes here is exactness across the chain. A stage's constraint
becomes affine, `A_free·β_free = −A_frozen·β_frozen`, and is solved as such — so
the staged model is translation-invariant **as a whole**, not stage by stage. When
the frozen part was itself fitted under the ASR (any stage of a chain), the
right-hand side vanishes and the stage stays homogeneous, which is why a chain
costs nothing in exactness. Freezing an externally supplied model that violates
the ASR is legal and takes the affine path; if the violation lives on constraint
rows the free columns cannot balance, the fit refuses and names those rows —
widen the mask, or freeze a model whose [`asr_residual`](@ref) is small.

[`refit`](@ref) stays inside the stage (frozen coefficients are never thresholded
away), and `dof` / [`gcv`](@ref) / [`identifiability`](@ref) all report the
stage's own parameter count.

## Refitting on a selected support

After a sparse fit (`Lasso` / `AdaptiveLasso` / `AdaptiveRidge`), the surviving
coefficients are shrunk toward zero by the penalty. [`refit`](@ref) removes that bias: it
keeps the support of an existing fit and re-solves on just those columns — by default with
[`OLS`](@ref), the textbook de-biasing step.

```julia
fsparse = fit(SLCEFit, dataset, Lasso())     # selects a support (some jϕ exactly zero)
fdebias = refit(fsparse)                     # OLS on that support — unshrunk survivors
```

A column survives when its scaled-magnitude contribution `|coef(f)[j]|·‖X[:, j]‖` exceeds
`threshold` (default `0`, i.e. exactly the nonzero support); pass a positive `threshold` to
prune further. `refit` reuses the dataset and `torque_weight` of the input fit, so the
co-fit whitening is identical.

## Cost-weighted group selection

A Monte-Carlo sweep over a fitted SCE pays per **contraction entry**, and an entry
vanishes only when *every* SALC of its `(body, orbit, l-multiset)` group has a zero
coefficient — so the quantity to minimize alongside the fit error is the summed cost of
the surviving groups, not the coefficient count. The workflow prices each group up
front and lets the penalty act at exactly that granularity:

```julia
est  = GroupAdaptiveRidge(basis; lambda = 1.0, theta = 1.0)   # cost-proportional weights
path = select_fit(dataset, est; lambdas = 10.0 .^ range(2, -8; length = 25))
fbest = refit(path.fit; threshold = path.threshold)   # de-bias exactly the alive support
```

(The reweighted ridge crushes dead groups to tiny — not exactly zero — values, so
"alive" is decided by a relative floor on the scaled magnitudes; `path.threshold` is
the effective absolute value at the selected λ, and passing it to [`refit`](@ref)
realizes exactly the support the path reported.)

The convenience constructor bundles `SLCE.salc_groups` (the column → group
labels) with `SLCE.cost_weights`, which sets the fixed per-group weights to
``v_g = \sqrt{p_g}\,(c_g/\bar c)^\theta`` — ``c_g`` being the group's a-priori
Monte-Carlo cost (its distinct contraction entries, `SLCE.group_costs`). Two
independent knobs shape the cost–accuracy trade:

- **`theta ∈ [0, 1]`** tilts the *penalty*: `theta = 0` ignores cost (plain group
  selection), `theta = 1` makes an expensive group earn its keep with a
  correspondingly larger error reduction. Different `theta` change the *order* in
  which groups die along the λ path.
- **`delta ≥ 0`** tilts the *selection*: [`select_fit`](@ref) scores every λ (GCV by
  default, or configuration-grouped CV with `criterion = :cv`) and picks the
  **cheapest** λ whose score is within `(1 + delta)` of the path minimum — the
  cost-aware generalization of the conventional `:lambda_1se` rule.

The returned [`SelectionPath`](@ref) is a Tables.jl source with one row per λ
(`lambda`, `score`, `edof`, `n_alive`, `cost`, `selected`); sweeping `theta` and taking
the lower envelope of the per-θ paths traces the full (cost, error) Pareto front. For
a torque co-fit prefer `criterion = :cv` — see the caveat in [`gcv`](@ref).

On real data the group-magnitude spectrum is usually **continuous** — there is no
clean alive/dead gap for the λ path to expose, and most of the cost–error trade
lives in the support threshold itself. [`select_support`](@ref) is the second knob:
it sweeps the alive threshold at a fixed fit, de-biases with [`refit`](@ref) at each
point (one cheap OLS per point — no re-solving the penalty), scores each refit on an
evaluation dataset, and applies the same Pareto rule:

```julia
train, held = dataset[1:80], dataset[81:100]        # dataset slicing
f     = fit(SLCEFit, train, GroupAdaptiveRidge(basis; lambda = 1e-5, theta = 1.0))
front = select_support(f; thresholds = 25, evalset = held, delta = 0.05)
front.fit                                           # the selected de-biased refit
```

Pass a held-out `evalset` for an honest error axis (the default is the in-sample
training set). On a production Nd₂Fe₁₄B model this front offered, e.g., 38 % of the
Monte-Carlo cost at a held-out torque RMSE *better* than the full model, and 3 % of
the cost at +19 % — trades the λ path alone cannot see.

## Cross-validation

[`cross_validate`](@ref) is the generic, honest assessment behind all of the above:
configuration-grouped K-fold CV of any `fit` call, refitting each fold from scratch
(centering and torque whitening stay inside the training fold — nothing leaks) and
scoring the held-out configurations in prediction space:

```julia
cv = cross_validate(dataset, GroupAdaptiveRidge(basis; lambda = 1e-5);
                    torque_weight = 1.0, nfolds = 5)
cv.pooled_rmse_energy, cv.pooled_rmse_torque   # both error axes, out-of-fold
cv.score                                       # per-fold (1−w)·MSE_E + w·MSE_T
```

Both RMSEs are reported whenever the dataset carries torque data, **independent of
`torque_weight`** — an energy-only fit still gets its torque error measured, which
is exactly what comparing `torque_weight` settings needs. The `pooled_*` fields
aggregate the out-of-fold residuals (every configuration is held out exactly once);
the per-fold columns give the spread. Use it where a single train/holdout split is
too noisy — e.g. to back a [`select_support`](@ref) point with a K-fold error bar,
or to rank estimators on an equal footing. It differs from
`select_fit(criterion = :cv)`, which whitens globally and only *ranks* a λ path;
`cross_validate` is the generalization-error estimate.

## Diagnostics

A fitted [`SLCEFit`](@ref) answers the usual questions:

```julia
r2_energy(f);  rmse_energy(f)        # in-sample energy R² / RMSE
r2_torque(f);  rmse_torque(f)        # torque equivalents (need a co-fit dataset)
nobs(f)                              # number of energy observations
dof(f)                               # degrees of freedom: length(coef(f)) + 1
rss_energy(f);  rss_torque(f)        # residual sums of squares
residuals_energy(f);  residuals_torque(f)   # the raw residual vectors
effective_dof(f)                     # hat-matrix trace + 1 (linear estimators)
gcv(f)                               # generalized cross-validation score
```

For a *linear* estimator (`islinear`: `OLS` / `Ridge` / `AdaptiveRidge` /
`GroupAdaptiveRidge`) two closed-form model-selection diagnostics come for free:
[`effective_dof`](@ref) is the trace of the hat matrix plus the intercept — the
*effective* parameter count a penalized fit actually spends, as opposed to
[`dof`](@ref)'s raw count — and [`gcv`](@ref) is the generalized cross-validation
score `n·RSS/(n − df)²` built on it, the fast λ-selection criterion used by
[`select_fit`](@ref).

[`identifiability`](@ref) answers a different question — not *how well* the fit
explains the data, but *whether the data determine the fit at all*:

```julia
identifiability(f)                        # the design the fit solved
identifiability(ds; torque_weight = 0.4, force_weight = 0.6)   # before fitting
# (; ncols, rank, nullity, sigma_min, sigma_max, sigma_cut, gap)
```

`nullity > 0` means the objective is exactly flat along that many coefficient
directions: different models reproduce the training observables identically, and
what comes back along those directions is the estimator's null-space convention
(`OLS`'s minimum norm), not an estimate. `gap` — the smallest kept over the
largest dropped singular value — says how clean that count is; a value within a
few orders of 1 means the spectrum has no break and the report is inconclusive
(raise or lower `rtol` and look again). `fit` always warns about the per-column
case — columns the data do not touch at all, e.g. every pure-spin column in a
force-only fit — but a flat direction is generally a *combination* of columns, so
the exact test is this opt-in one (a full SVD of the design; `O(n·q²)`). The canonical cure is more informative data or, for the
translation-violating directions, the [ASR](#The-acoustic-sum-rule-(translation-invariance)).

The energy and torque blocks are reported separately throughout (the rebuild does not fold
them into one combined residual): `residuals_energy(f)` is `y_E − (j0 + X_E·jϕ)` and
`residuals_torque(f)` is `y_T − X_T·jϕ` over the flattened torque components.

The generic names — [`predict`](@ref), [`residuals`](@ref), [`r2`](@ref), plus `coef` /
`fit` / `nobs` / `dof` / `coeftable` / `islinear` — **extend StatsAPI** (imported, not
shadowed) and default to the energy block, so they compose with `using StatsBase` /
`using GLM` instead of clashing.

To inspect the coefficients as a table, use [`coeftable`](@ref) — a Tables.jl source with
one row per SALC. See [Persistence and I/O](io.md#Tabular-coefficients).

Next: [Persistence and I/O](io.md).
