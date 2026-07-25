# Oracle test environment

Validates `SLCE`'s from-scratch numerics against `Magesty.jl` used as a
numerical oracle. Kept in a **separate environment** so the core test suite never
depends on Magesty.

## Run

```bash
julia --project=test/oracle test/oracle/runtests.jl
```

First run resolves Magesty's full dependency tree (`Pkg.instantiate()` is implied
by `--project`; run it explicitly once if needed). `Manifest.toml` here is
gitignored.

## What it checks

- **Convention-fixed kernels** (bit-for-bit): tesseral harmonics `Zₗₘ` / `∇Zₗₘ`,
  Clebsch–Gordan, real Wigner-D, coupled tensors (up to a known global phase).
- **Gauge-invariant aggregates**: SALC counts, projector eigenvalues/rank, orbit
  multiplicities.
- **Fit parity** (end-to-end): the same EMBSET through both packages, bases built
  with the same channel content (the truncation semantics differ — see the
  testset comment for the `lmax`/`lsum` mapping), OLS at `torque_weight = 0` and
  `1`; held-out energy predictions must agree, and held-out torque predictions
  agree **up to the known global sign convention** (SLCE reports the
  physical / Landau–Lifshitz `τ = m × B`; Magesty reports `τ = −m (e × B)` —
  both sides flip target and predictor together, so the fitted function is
  identical).

Raw SALC coefficients and individual design-matrix columns are **not** compared
directly — they are gauge-dependent.

## Pinning

Both packages are `[sources]`-deved from local paths. The pinned Magesty rev is
recorded in `Project.toml`; keep the local `../../../Magesty.jl` checkout at that
rev for reproducible comparisons.
