# SLCE.jl benchmarks

Performance benchmarks for the core hotspots. This is **separate from
`test/oracle/`** — the oracle checks *numerical* agreement with Magesty.jl; these
scripts measure *speed and allocations*. Each script is standalone and runs in the
`bench/` environment (its own `Project.toml`/`Manifest.toml`, which develops the
package from `..`).

## Setup (once)

```bash
julia --project=bench -e 'using Pkg; Pkg.develop(path="."); \
    Pkg.add(["BenchmarkTools", "Spglib", "Statistics", "Printf", "Random", "LinearAlgebra"]); \
    Pkg.instantiate()'
```

## Run

```bash
julia --project=bench bench/bench_salcbasis.jl     [n] [lmax] [cutoff]    # SALC projection (the hotspot)
julia --project=bench bench/bench_clusters.jl      [n] [nbody] [cutoff]   # neighbour list + orbit reduction
julia --project=bench bench/bench_design_matrix.jl [n] [m] [lmax] [cutoff] # energy + torque design matrices
julia --project=bench bench/bench_solver.jl                               # OLS vs Ridge solve, size sweep
julia --project=bench bench/bench_end_to_end.jl    [n] [m] [lmax] [cutoff] # SLCEBasis build + fits
julia --project=bench bench/bench_nd2fe14b.jl      [nbody] [m] [cutoff]   # realistic multi-species case
```

Positional arguments are optional. **Defaults are the recorded stress baselines**
(seconds-scale, so regressions are visible above noise), not smoke sizes:

- **bcc Fe scripts**: `n = 4` (128 atoms; `n` is the supercell size, `2n³` atoms),
  multi-shell `cutoff = 6.0` Å (`bench_salcbasis` defaults to `lmax = 3`,
  `bench_clusters` to `nbody = 3`; `bench_design_matrix`/`bench_end_to_end` to
  `lmax = 2`, `m = 100` configs). For a quick smoke run pass the old sizes
  explicitly, e.g. `bench_salcbasis.jl 2 2 2.6` (16 atoms, first shell).
- **`bench_nd2fe14b.jl`**: the 68-atom Nd₂Fe₁₄B cell (`assets/nd2fe14b.toml`) —
  9 sublattice species, 16 symmetry ops, per-species `lmax = [4,4,2,2,2,2,2,2,0]`
  (B non-magnetic), `soc = false`, `nbody = 3` with `cutoff = 4.0` Å (compact
  cliques — like the real l044 run, which SLCE bounds by cutoff instead of
  Magesty's `lsum`), `m = 103` configs with torques (the real training-set size).
  Unlike the high-symmetry bcc fixture it stresses the many-orbit / few-ops regime
  and the co-fit solve, end to end through `fit`. Pass `2 103 inf` for the
  every-resolvable-pair variant (205 pair orbits).

## Fixtures and harness

`fixtures.jl` (shared, `include`d by every script) provides:

- `bcc_fe(n; a = 2.87)` — an `n×n×n` bcc Fe supercell (`Crystal`),
- `rand_configs(crystal, m; seed)` — `m` random unit-column spin configs,
- `basis_spec(; nbody, cutoff, lmax, soc)` — the matching `BasisSpec`,
- `bench_one(label, f; ntrials)` — warm-up + `@timed`/`@allocations` (median/min ms,
  alloc count, MiB), for hot paths too heavy for `@benchmark` sampling,
- `bench_header(title)`, `argn(i, default)`, `argf(i, default)`.

`assets/` holds structure-only TOML inputs (SLCE's `read_setup` schema) for
the realistic fixtures. Only the crystal structure lives there — training data is
always synthetic (`rand_configs` + random targets), since timing does not depend
on data values, and numerical agreement is `test/oracle/`'s job.

## Recording results

Append a before/after entry to [`../.claude/bench_log.md`](../.claude/bench_log.md)
when you touch a hot path (`basis/salcbasis.jl`, `clusters/`, `slce/model.jl` design
kernels, `fitting/estimators.jl`, `basis/Harmonics.jl`). Note machine, Julia version,
thread count, and the fixture size (`n`, `lmax`, `m`).
