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

`vcat` accepts parts built on a persisted-and-reloaded basis (the check is the SALC-basis
fingerprint, not object identity), and a torque-bearing part concatenates with an
energy-only one into a **mixed** dataset — each torque row keeps its own configuration
through the re-offset `torque_config`. What the parts must agree on is (1) their
provenance, i.e. one computational setup and one clamped-ion reference, (2) whether they
carry displacements at all, (3) their force columns, and (4) their ASR reparameterization.
Each disagreement is a hard error, because each would otherwise reintroduce the bias the
invariant exists to prevent — a fabricated geometry, or a family-correlated energy offset.

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

The SLCE's second observable is the per-atom torque
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
never zero-padded. The selection layer takes `force_weight` too:
[`select_fit`](@ref) and [`cross_validate`](@ref) accept it directly and
[`select_support`](@ref) reads it off the fit, all scoring the same three-block
objective. Two consequences worth knowing:

- At `force_weight = 0` a force-carrying dataset is treated exactly as one with the
  forces dropped — including the fold deal, which deliberately does **not** stratify on
  a zero-weight channel, so scores recorded before the force channel existed still mean
  what they meant. `rmse_force` is still reported.
- At `torque_weight + force_weight == 1` the energy block has zero weight, so
  pure-spin columns are structurally absent from the design: they are never alive and
  cost nothing. The resulting front is over the derivative-visible model only.

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

`asr` threads through [`cross_validate`](@ref) and [`select_fit`](@ref) to every fit
they run. [`select_fit`](@ref) refuses to run *under* a reparameterization: its λ path
solves on a cached **unconstrained** Gram, so it would report a support the constrained
solve never chose. Selecting with it therefore means selecting a deliberately
unconstrained model (`select_fit(...; asr = false)`); the selected fit is re-solved cold
with the same `asr`, so a plain `fit` call reproduces it.

[`select_support`](@ref) has no such restriction — it drives [`refit`](@ref), which
re-derives the null space on each support, and reads its alive/cost columns back off the
refits it returns. Be aware of what the constraint does to the *cost axis*, though: the
sum rule ties columns across groups, and on every fixture measured no displacement-
touched group has a feasible subspace on its own, so the displacement side of the model
moves close to all-or-nothing while the pure-spin groups (which the constraint leaves
untouched) keep full granularity. `SLCE.group_freedom` reports how much of the null
space the constraint leaves each group, and `SLCE.group_costs(basis, labels; asr = ...)`
prices at zero any group the constraint kills outright — worth checking on a new
truncation, since a `pmax` too small to form difference invariants can leave a dead
group holding most of the nominal cost.

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

## Rotational invariance (measured, not imposed)

Translation is the *only* affine invariance this package imposes. The other two —
rigid rotation (Born–Huang) and vanishing stress — are **not constraints here**, and
the reason is structural: a rigid rotation by an arbitrary angle is not an operation
of the crystal's space group, so the SALC projection cannot remove it, and the
Born–Huang conditions are independent of both the ASR and the pair-exchange (Huang)
conditions. What the package gives you instead is a way to *measure* the violation.

```julia
rotational_residual(model, spins; omega = 0.05, axis = (0, 0, 1))
```

rotates the lattice rigidly by a **finite** angle and reports the energy that costs
as a fraction of what a real deformation of the same size costs. Zero is invariance;
`≈ 1` means the model cannot tell a rotation from a strain. The underlying evaluation
path is [`affine_energy`](@ref), which applies a displacement field
`u(R) = M·(R − origin) + base` at each cluster site's own position — image shift
included. [`predict_energy`](@ref) cannot express such a field at all: it resolves a
site's displacement as `u[:, atom]`, so the fields it accepts are cell-periodic, and
translation is the one affine field that *is*. On an ASR-satisfying model the answer
does not depend on `origin`.

The angle has to be finite. Linearizing the rotation to `u = ωWd` reproduces every
term of `ΔE` except `½ω²·F·(W²d)` — and on a model whose forces follow the bond
directions the linearized test returns *exactly* zero however badly the model
violates invariance, because `d·(Wd) = 0` for antisymmetric `W`.

In a SOC sector zero is the wrong expectation for the lattice half: the rotational
response *transfers* to the spin channel rather than vanishing (`𝓡_U E = −𝓡_S E`).
The statement that holds in every sector is [`rotation_transfer_residual`](@ref),
which rotates spins and lattice together and reports how much of the two halves fails
to cancel. On a pseudo-dipolar model — invariant under the joint rotation and under
neither half alone — the lattice-only residual is `1.67` while the joint one falls to
`1.5·10⁻⁴`, so the cancellation is not two small numbers agreeing. The gate pins the
two sides separately (`> 1.0` and `< 10⁻²`) and checks that a random ASR-feasible model
of the same basis shows no such cancellation.

Two things worth knowing before reading a number:

- **A truncated model is never exactly invariant.** The rotational condition ties
  order `n` to order `n + 1`, so the best a fit to a genuinely invariant potential
  achieves is a residual that *vanishes with `ω`* — that decay, not the value at one
  `ω`, is what distinguishes truncation from a real violation.
- **On-site displacement content carries an image gauge.** A model with 1-body
  (on-site) force content attaches it to the home-cell representative of each atom.
  Two descriptions of the same crystal that differ in which periodic image is called
  "home" fit the same periodic training data equally well and have *different*
  rotational residuals — measured 1349× apart on a stressed dimer, and identical
  once the reference is relaxed (`F = 0`). Periodic data cannot choose between them;
  this diagnostic can.

## Estimators

The estimator is the regression strategy, dispatched on [`AbstractEstimator`](@ref). Four
are in-tree (closed form, no dependencies):

| Estimator | Penalty | Notes |
|-----------|---------|-------|
| [`OLS`](@ref) | none | ordinary least squares (QR) |
| [`Ridge`](@ref) | ``\lambda\sum_j m_j\beta_j^2`` | L2, closed form |
| [`AdaptiveRidge`](@ref) | ``\lambda\sum_j D_j\beta_j^2`` | iterative reweighted ridge, an L0 approximation |
| [`GroupAdaptiveRidge`](@ref) | as above, ``w`` shared per group | group-L0, fixed cost weights |

[`AdaptiveRidge`](@ref) (Frommlet & Nuel 2016) repeatedly refits a per-coefficient
weighted ridge with ``D_j = m_j/(m_j\beta_j^2 + \varepsilon)``, so large coefficients
get a light penalty and small ones a heavy penalty — iterating drives the small ones
toward zero. Each subproblem is the analytic weighted ridge, so it needs no extension:

```julia
fit(SLCEFit, dataset, AdaptiveRidge(lambda = 1e-3))     # L0-like selection, closed form
```

[`GroupAdaptiveRidge`](@ref) is its group extension: all columns of a group share one
weight ``w_j = v_g/(\sum_{k\in g} m_k\beta_k^2 + p_g\varepsilon)``, so whole groups —
not individual columns — are driven to zero, and the fixed multiplier ``v_g`` prices
each group (at convergence a surviving group pays exactly ``\lambda v_g``). This is the
estimator behind the Monte-Carlo-cost-aware selection workflow below.

## The penalty metric

``m_j`` above is the **penalty metric**: a per-column scale, `nothing` (uniform) or a
vector from [`penalty_metric`](@ref). It is on by default whenever an estimator is
built from a basis.

``\lambda\lVert\beta\rVert^2`` is not invariant under rescaling a design column, and
SALC column norms are set by basis conventions — an orbit's member count, and the
ordering multiplicity the member fold absorbs — not by physics. A column with a larger
norm carries a smaller coefficient at the same physical effect, so it is shrunk *less*:
the unweighted penalty quietly prefers large orbits and high body order. The metric

```math
m_j(w) = (1-w)\,\mathrm{Var}[\Phi_j]
       + w\,\frac{1}{3 n_\text{atoms}}\,
         \mathbb{E}\Bigl[\sum_a \lVert (\partial\Phi_j/\partial e_a)\times e_a\rVert^2\Bigr]
```

— the reference norm of the column as the estimator sees it, over uniform-random spin
configurations — removes that, leaving only the prior you state deliberately through
`cost_exponent`.

```julia
est = GroupAdaptiveRidge(basis; lambda = 1e-5, cost_exponent = 1.0)  # metric attached
m   = penalty_metric(basis; torque_weight = 0.3)                     # or build it yourself
est_w = GroupAdaptiveRidge(basis; lambda = 1e-5, torque_weight = 0.3)
fit(SLCEFit, dataset, est_w; torque_weight = 0.3)                    # weights must agree
est_plain = GroupAdaptiveRidge(lab, weights; lambda = 1e-5)          # uniform penalty

# a λ sweep: build the metric ONCE and move it along the path
fits = [fit(SLCEFit, dataset, SLCE.with_lambda(est, l)) for l in lambdas]
```

Use `SLCE.with_lambda` rather than rebuilding the estimator by hand. A hand rebuild
(`GroupAdaptiveRidge(est.column_groups, est.group_weights; lambda = l)`) silently drops
the metric, and a dropped metric is indistinguishable from a deliberate uniform one —
no door will complain. Rebuilding through the basis-aware constructor instead re-runs
`penalty_metric` at every point, which is the expensive half: the metric is a
Monte-Carlo average, ~3.3 % relative standard error per column at the default
`metric_nconfig = 2048` (5.3 % on the worst column, `1/√nconfig` from there). The λ
path itself is unaffected — the metric is one extra multiply per column per iteration.

Four things follow from that definition and are worth knowing:

- **The metric is pure-spin only.** Its reference ensemble is uniform random spin
  directions, and a displacement factor ``|u|^{2k}R_{lm}(u)`` has no value on one: the
  reference distribution of ``u`` is a modelling decision (which amplitude? which
  temperature?) that has to be written down before it can be sampled. On a joint basis
  [`penalty_metric`](@ref) and the basis-aware constructors refuse by name; pass
  `metric = nothing` for the unweighted penalty there, and expect the same refusal if
  you fit with `force_weight > 0`.
- **`torque_weight` is part of the metric.** The assembled design mixes the energy and
  torque blocks by ``w``, so the column scales move with it. Build the metric at the
  weight you will fit at; [`fit`](@ref), [`select_fit`](@ref) and
  [`cross_validate`](@ref) refuse a mismatch, along with a metric built on a different
  basis. (A metric you assembled by hand carries no provenance and is never
  second-guessed.)
- **It is a property of the basis, not of the training data.** That is deliberate: λ
  becomes comparable between cells, cross-validation needs no per-fold recomputation to
  stay leak-free, and the prior sits on the function space rather than on how strongly
  one training set happened to excite each column. The price is the converse — a column
  the reference ensemble excites weakly but your data drives hard is effectively
  under-penalized — worth keeping in mind when your configurations are far from uniform
  (near-collinear low-temperature states, say).
- **The support rule is a separate scale, and already invariant.** [`refit`](@ref) and
  [`select_support`](@ref) threshold ``|\beta_j|\cdot\lVert X[:,j]\rVert``, which does
  not move under a column rescaling at all, so the metric does not touch it. The two
  knobs are independent by construction rather than by tuning.

An entry of exactly `0` marks a column **unpenalized**. That is how the pointed moment
channel's ``\mu_0`` intercepts are kept out of the penalty — `Ridge(mb; lambda)` and its
siblings carry `penalty_metric(mb)`, which zeroes them — for every estimator rather
than only the group form. Under an ASR / freeze reparameterization the penalty is
compressed as ``Z'\,\mathrm{diag}(D)\,Z``, so the metric keeps its basis-column
indexing exactly as `column_groups` does.

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
[`FixedCoefficients`](@ref) that reuses a prior fit's coefficients:

```julia
fit(SLCEFit, dataset, AdaptiveLasso(gamma = 1.0))                       # OLS pilot (Zou 2006)
fit(SLCEFit, dataset, AdaptiveLasso(pilot = Ridge(lambda = 1e-4)))      # for ill-conditioned designs
fit(SLCEFit, dataset, AdaptiveLasso(pilot = FixedCoefficients(coef(prior)), lambda = 1e-3))
```

It shares `lambda` / `standardize` / CV behavior with [`ElasticNet`](@ref) (`lambda =
nothing` selects λ by configuration-grouped CV with the adaptive weights held fixed).

Implementing your own estimator is one method, and it must accept both optional
keywords — [`fit`](@ref) passes them unconditionally:

```julia
struct MyEstimator <: AbstractEstimator end
SLCE.solve_coefficients(::MyEstimator, X, y;
                        row_groups = nothing, nullspace = nothing) = X \ y  # centered (X, y)
```

`row_groups` labels rows belonging to one physical sample (only a resampling estimator
needs it). `nullspace` is the ASR reparameterization: when it is not `nothing`, `X` is
already the compressed design ``X_\beta Z`` and the returned vector is ``\gamma``, with
``\beta = Z\gamma`` lifted by the caller. Penalties stay defined on ``\beta`` — an
estimator whose weights depend on the coefficients must evaluate them at ``Z\gamma``,
which is what `AdaptiveRidge` / `GroupAdaptiveRidge` do. Ignoring the keyword is fine for
an unpenalized solve like the one above, since ``Z`` is orthonormal.

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
and `:soc_free` / `:soc_only` (a crosscutting partition by `L_S`); a collection of them
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

A Monte-Carlo sweep over a fitted SLCE pays per **contraction entry**, and an entry
vanishes only when *every* SALC of its `(body, orbit, l-multiset)` group has a zero
coefficient — so the quantity to minimize alongside the fit error is the summed cost of
the surviving groups, not the coefficient count. The workflow prices each group up
front and lets the penalty act at exactly that granularity:

```julia
est  = GroupAdaptiveRidge(basis; lambda = 1.0, cost_exponent = 1.0)   # cost-proportional weights
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
Monte-Carlo sweep cost (`SLCE.group_costs`: over its distinct contraction entries, the
summed number of site programs each one costs the sweep), and ``\theta``
the `cost_exponent` keyword. Two independent knobs shape the cost–accuracy trade:

- **`cost_exponent ∈ [0, 1]`** (the ``\theta`` above) tilts the *penalty*: `0` ignores
  cost (plain group selection), `1` makes an expensive group earn its keep with a
  correspondingly larger error reduction. Different values change the *order* in
  which groups die along the λ path.
- **`score_rtol ≥ 0`** tilts the *selection*: [`select_fit`](@ref) scores every λ (GCV by
  default, or configuration-grouped CV with `criterion = :cv`) and picks the
  **cheapest** λ whose score is within `(1 + score_rtol)` of the path minimum — the
  cost-aware generalization of the conventional `:lambda_1se` rule.

The returned [`LambdaPath`](@ref) is a Tables.jl source with one row per λ
(`lambda`, `score`, `edof`, `n_alive`, `cost`, `selected`); sweeping `cost_exponent`
and taking the lower envelope of the paths traces the full (cost, error) Pareto front. For
a torque co-fit prefer `criterion = :cv` — see the caveat in [`gcv`](@ref).

On real data the group-magnitude spectrum is usually **continuous** — there is no
clean alive/dead gap for the λ path to expose, and most of the cost–error trade
lives in the support threshold itself. [`select_support`](@ref) is the second knob:
it sweeps the alive threshold at a fixed fit, de-biases with [`refit`](@ref) at each
point (one cheap OLS per point — no re-solving the penalty), scores each refit on an
evaluation dataset, and applies the same Pareto rule:

```julia
train, held = dataset[1:80], dataset[81:100]        # dataset slicing
f     = fit(SLCEFit, train, GroupAdaptiveRidge(basis; lambda = 1e-5, cost_exponent = 1.0))
front = select_support(f; npoints = 25, evalset = held, score_rtol = 0.05)
front.fit                                           # the selected de-biased refit
```

Pass a held-out `evalset` for an honest error axis (the default is the in-sample
training set). On a production Nd₂Fe₁₄B model the front exposed trades the λ path alone
cannot see: a large fraction of the Monte-Carlo cost removed at a held-out torque RMSE
*better* than the full model (de-biasing beats the interpolating tail), and an
order-of-magnitude cheaper model at a modest error penalty.

!!! note "The published figures predate the current cost metric"
    The specific percentages recorded for that run (38 % of the cost at better holdout,
    3 % at +19 %) were measured when `group_costs` priced a group by its nonzero
    contraction entries. It now prices each entry by its **slot count**, because the
    sweep walks one site program per member site position — the old metric priced the
    once-per-run energy program and mis-ranked groups by a factor of the body order. The
    shape of the front is unchanged; the numbers have not been re-derived, so treat them
    as illustrative rather than as a benchmark.

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
r2_force(f);   rmse_force(f)         # force equivalents (need a force block)
nobs(f)                              # number of energy observations
dof(f)                               # FREE parameters + 1 (see below)
rss_energy(f);  rss_torque(f);  rss_force(f)        # residual sums of squares
residuals_energy(f);  residuals_torque(f);  residuals_force(f)   # residual vectors
effective_dof(f)                     # hat-matrix trace + 1 (linear estimators)
gcv(f)                               # generalized cross-validation score
```

[`dof`](@ref) counts the parameters the fit was actually **free** to move, plus `j0`:
`length(coef(f)) + 1` for an unconstrained fit, `p − rank(A) + 1` under the ASR (the
default on a joint basis, where `rank(A)` equalities are enforced exactly rather than
fitted), and a stage's own free-column count for a staged fit. All three are one
expression — the column count of the reparameterization the fit solved under.

For a *linear* estimator (`islinear`: `OLS` / `Ridge` / `AdaptiveRidge` /
`GroupAdaptiveRidge`) two closed-form model-selection diagnostics come for free:
[`effective_dof`](@ref) is the trace of the hat matrix plus the intercept — the
*effective* parameter count a penalized fit actually spends, as opposed to
[`dof`](@ref)'s parametric count — and [`gcv`](@ref) is the generalized cross-validation
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
