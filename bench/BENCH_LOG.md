# Benchmark log — SLCE.jl

> Entries dated before the 2026-08-11 carve-out were written when this package
> was still named SCEFitting.jl; they are this package's own history, and the
> revived SCEFitting.jl (new UUID, cut from `698a841`) shares them as an
> ancestor. Its log starts at the carve-out and points back here.

A running record of performance numbers for the core hotspots, kept so refactors can
be checked for regression after the fact. Scripts live in [this directory](.); the
convention mirrors `Magesty.jl/.claude/bench_log.md`.

**How to log.** When you touch a hot path (`basis/salcbasis.jl`, `clusters/`, the
`sce/model.jl` design kernels, `fitting/estimators.jl`, `basis/Harmonics.jl`), run the
relevant `bench/` script before and after and append an entry below:

- a one-line context header (date, branch/commit, machine, Julia version, threads,
  fixture size `n`/`lmax`/`m`),
- a **Before / After** table (median time, MiB, allocs),
- a short note (wall-time vs alloc trade-off, downstream impact, follow-ups).

Entries are append-only history — keep them after merging.

---

## Baseline — 2026-06-29 (smoke, n = 2)

**Context**: 2026-06-29 · branch `main` · local macOS (darwin 24.6) · julia 1.12.6 ·
threads 1.
**Fixture**: bcc Fe (`bcc_fe(2)` → 16 atoms), single species, `nbody = 2`,
`pair_cutoff = 2.6`, `lmax = 2`, `m = 30` configs. Median over 3 trials (warm).

| Script / target | n (atoms) | lmax | m | median | MiB | allocs |
|---|---|---|---|---|---|---|
| `build_salc_basis`        | 2 (16)  | 2 | — | 16.5 ms | 140.6 | 703,067 |
| `build_neighbor_list`     | 2 (16)  | — | — | 0.06 ms | 0.01 | 12 |
| `build_clusters`          | 2 (16)  | — | — | 91.4 ms | 120.9 | 1,980,946 |
| `_design_energy`          | 2 (16)  | 2 | 30 | 76.6 ms | 109.4 | 3,967,082 |
| `_design_torque`          | 2 (16)  | 2 | 30 | 142.2 ms | 264.9 | 11,811,845 |
| `SCEBasis` (end-to-end)   | 2 (16)  | 2 | — | 109.1 ms | 262.0 | 2,687,360 |
| `fit (OLS)` (solve only)  | 2 (16)  | 2 | 30 | 0.002 ms | 0.07 | 41 |
| `solve_coefficients` OLS  | M=5000,P=500 | — | — | 125.1 ms | — | — |
| `solve_coefficients` Ridge| M=5000,P=500 | — | — | 26.6 ms | — | — |

**Notes**: at this tiny size `build_clusters` (symmetry-orbit reduction) and the
design-matrix kernels dominate, not the SALC build; `fit` itself is just the solve
(the design matrix is materialized once in `SCEDataset`). Alloc counts are high
across the board — the obvious first optimization targets. Scale `n` up (3 → 54,
4 → 128 atoms) for a stress baseline; the SALC build at `n = 4, lmax = 2` is the
minutes-scale case.

---

## #1: SCEBasis build — clusters + SALC hot paths (2026-06-29)

**Context**: 2026-06-29 · branch `main` · local macOS (darwin 24.6) · julia 1.12.6 ·
threads 1. **Fixture**: bcc Fe `bcc_fe(4)` = **128 atoms** (the `Fe444` tutorial cell),
`nbody = 2`, `lmax = 2`. Validated by the full unit suite (23585) + the Magesty oracle
(13873, gauge-invariant) — **output byte-identical** (same orbits, same `n_salcs`).

The 128-atom basis build was minutes-scale. Three changes, all numerics-preserving:

1. **`clusters/orbits.jl` `_canonical_key`** — the per-image sorted site list was a heap
   `Array` + `Tuple` per (candidate × op × anchor). Rewrote with a `Val(N)` barrier so the
   site tuples are statically sized and stack-allocated.
2. **`build_clusters` orbit-growing** — was an `O(n_candidates × n_ops)` canonical key per
   candidate; now grows each orbit from one representative by applying every op and matching
   images back by translation signature, so the canonical key runs once per orbit
   (`O(n_orbits × n_ops)`).
3. **`basis/salcbasis.jl` `_connect_all`** — replaced the per-member `_connect` (an
   `O(n_ops)` scan each) with one `O(n_ops)` sweep mapping `g·rep` images to members.
4. **Wigner-D memoization** — `_transport_term` recomputed `wignerD_real(l, R_g)` per
   member/term/site; now memoized by `(l, g)` across the whole build (shared with
   `_project_and_fold`).

### Before / After (bcc Fe 128 atoms, lmax = 2)

| Stage | cutoff | Before | After |
|---|---|---|---|
| `build_clusters` | 2.6 (1 NN) | 5.69 s, 126.6 M allocs, 7.5 GiB | **0.03 s, ~24 k allocs** |
| `build_clusters` | 6.0 (multi-shell) | ~45 s (extrapolated, alloc-bound) | **0.03 s** |
| `build_salc_basis` | 6.0 | 6.2–9.9 s, 86 M allocs, 13.6 GiB | **1.03 s, 12.2 M allocs, 0.79 GiB** |
| **`SCEBasis` total** | 6.0 | **~15 s** | **~1.5 s** |

**所感**: clusters の支配要因はアロケーション（GiB 級の一時オブジェクト）と
`O(候補×操作)` の二重で、(1)(2) で両方を解消し実質ゼロに。SALC は `_connect` の
`O(メンバ×操作)` 走査と `wignerD_real` 再計算が主因で、(3)(4) で 6–10× / メモリ 17×減。
残る build_salc_basis 1 s は転送カーネル（`nmode_mul`/`permutedims` の一時確保、出力サイズ
にほぼ比例）— さらに削るならバッファ再利用だが、数値的に最も繊細な領域なので保留。
`analyze_symmetry`(spglib, ~0.4 s) が次点。

---

## Stress baseline — 2026-07-14 (new heavy defaults + Nd2Fe14B)

**Context**: 2026-07-14 · branch `main` · local macOS (darwin 24.6) · julia 1.12.6 ·
threads 1. The #1 speedup left the old recorded case (128 atoms, lmax 2, cutoff 6.0)
sub-second, so the script defaults were promoted to seconds-scale stress sizes and a
realistic low-symmetry fixture (Nd₂Fe₁₄B, `bench/assets/nd2fe14b.toml`) was added.
These are the **current baselines at the new defaults** (median over 3 warm trials).

### bcc Fe (128 atoms, `n = 4`)

| Script / target | defaults | median | MiB | allocs |
|---|---|---|---|---|
| `build_salc_basis`          | lmax 3, cutoff 6.0 | 1.70 s | 3,536 | 54.3 M |
| `build_clusters` (nbody 3)  | cutoff 6.0 | 0.51 s | 537 | 10.4 M |
| `_design_energy`            | lmax 2, cutoff 6.0, m 100 | 1.96 s | 6,083 | 176 M |
| `_design_torque`            | lmax 2, cutoff 6.0, m 100 | 4.60 s | 15,161 | 401 M |
| `SCEBasis` (end-to-end)     | lmax 2, cutoff 6.0 | 0.76 s | 1,427 | 14.4 M |
| `SCEDataset` (energy+torque)| lmax 2, cutoff 6.0, m 100 | 6.84 s | 21,245 | 578 M |
| `fit` (OLS co-fit w=0.5)    | 38,400 torque rows × 46 cols | 24.6 ms | 42 | 65 |

### Nd₂Fe₁₄B (68 atoms, 9 species, 16 ops; lmax [4,4,2,2,2,2,2,2,0], isotropy)

| Target | nbody 3, cutoff 4.0 (default) | nbody 2, cutoff inf |
|---|---|---|
| orbits (2-body / 3-body) | 44 / 74 | 205 / — |
| `build_salc_basis` (n_salcs) | 0.36 s (397) | 46 ms (372) |
| `_design_energy` (m 103)  | 1.31 s | 0.24 s |
| `_design_torque` (m 103)  | 3.25 s | 0.63 s |
| `fit` (OLS co-fit, 21,012 torque rows) | 0.47 s | 0.43 s |
| `fit` (Ridge co-fit) | 0.11 s | 0.09 s |

**所感**: bcc Fe は高対称の極端ケース（多操作・少軌道）、Nd₂Fe₁₄B は逆の少操作・
多軌道 + 多種 + 非磁性種 (B, lmax 0) を張る。現時点の支配コストは設計行列カーネル
（特にトルク、~GiB/s 級のアロケーション）で、SALC 構築は #1 以降二番手。co-fit の
OLS solve (0.4–0.5 s) は列数 ~400 で正規方程式が効き始める領域。

---

## #2: cache-threaded dnPl in the harmonics / design kernels — 2026-07-15

**Context**: 2026-07-15 · branch `main` · local macOS (darwin 24.6) · julia 1.12.6 ·
threads 1. **Change**: LegendrePolynomials の `dnPl` はデフォルト引数で毎回
`zeros(l−n+1)` を確保する — `Zlm_unsafe`/`grad_Zlm_unsafe` に cache 受け取りの
4 引数メソッドを追加し(値はビット同一、NaN-poison ゲート)、`evaluate_salc`/
`accumulate_grad!` → design 行列ループ(task-local)→ predict 経路に配線。
"Before" = Stress baseline (2026-07-14)。

| Target | Before | After |
|---|---|---|
| bcc Fe `_design_energy` (m 100) | 1.96 s / 176 M allocs | **1.10 s / 45 M** |
| bcc Fe `_design_torque` (m 100) | 4.60 s / 401 M allocs | **2.35 s / 68 M** |
| 2141 `_design_energy` (m 103, nbody 3) | 1.31 s | **0.76 s** |
| 2141 `_design_torque` (m 103, nbody 3) | 3.25 s | **1.77 s** |
| 2141 `build_salc_basis` | 0.36 s | 0.36 s(不変 — 対象外) |

**所感**: 設計行列がほぼ半減。残る ~45–68 M allocs は `_site_ztables`/`_site_gtables`
がサイト×項ごとに作る小テーブル配列(comprehension)そのもの — 次に削るなら
テーブルの事前確保(term ごとの最大 l で確保して再利用)だが、kernel の見通しと
のトレードオフなので必要になってから。SALC 構築(Wigner-D 側)は別経路で不変。

---

## Baseline for the pointed-moment backport — 2026-08-21 (phase 0, S-bench)

**Why this entry exists.** The backport plan needs a performance baseline that
predates the work, plus an explicit rule for what counts as a regression. This
entry is that baseline; the rule is below it. Nothing here is a claim about
correctness.

**Context**: 2026-08-21 · branch `main` · local macOS (darwin 24.6, aarch64) ·
julia 1.12.6 · **threads = 1** (the scripts' default) · all six scripts at their
**stress defaults** (no positional arguments).

### The regression gates — and why exactly these

| script | role | why |
|---|---|---|
| `bench_salcbasis` | **GATE** | the projector / fold hot path — the code the backport touches |
| `bench_design_matrix` | **GATE** | `evaluate_salc` per (row × column) — the other half of the same chain |
| `bench_clusters` | context | upstream of the basis; the backport does not touch orbit enumeration |
| `bench_end_to_end` | context | a sum of the two gates plus the solve; moves for reasons the gates already explain |
| `bench_nd2fe14b` | context | realistic many-orbit / few-ops shape — a sanity read, too coarse to gate |
| `bench_solver` | context | pure BLAS; independent of everything the backport changes |

**How to judge.** Run each gate script **five times** and take the median of the
reported medians.

- **Allocation count is the primary gate, at zero tolerance.** Measured
  2026-08-21: three consecutive runs of each gate reported allocation counts and
  MiB that were *bit-for-bit identical* — the quantity is deterministic, so any
  change at all is a real change and must be explained in the entry that causes
  it.
- **Wall time is the secondary gate, at 5%.** Measured run-to-run spread over
  three runs was at most **1.4%** (salcbasis) and under 1.1% (design matrices),
  so 5% carries roughly 3.5x headroom over this machine's noise. A threshold at
  the noise floor would be a coin flip, not a gate.
- A gate that trips is not automatically a veto — it is a required line in the
  commit's log entry, saying what was traded for what.

### Measured — SCEFitting.jl @ `116c4fd`, SLCE.jl @ `a132928`

Same machine, same session, same fixtures. Both packages emit **the same
`n_salcs`** on every fixture below, so these are like-for-like.

| script / target | fixture | SCEFitting med | SLCE med | SCEFitting allocs | SLCE allocs |
|---|---|---|---|---|---|
| `build_salc_basis` | bcc Fe 4x4x4 (128 at.), lmax 3, cutoff 6.0 | **2803 ms** | **1998 ms** | 70,872,871 | 60,778,374 |
| `_design_energy` | bcc Fe 4x4x4, 100 cfg, lmax 2 | 466 ms | 474 ms | 11,299,224 | 11,299,408 |
| `_design_torque` | bcc Fe 4x4x4, 100 cfg, lmax 2 | 991 ms | 1026 ms | 7,527,192 | 7,527,376 |
| `build_neighbor_list` | bcc Fe 4x4x4, nbody 3, cutoff 6.0 | 3.66 ms | 3.71 ms | 24 | 24 |
| `build_clusters` | bcc Fe 4x4x4, nbody 3, cutoff 6.0 | 450 ms | 483 ms | 10,390,312 | 10,390,312 |
| basis build (end-to-end) | bcc Fe 4x4x4, lmax 2 | 934 ms | 846 ms | 18,103,347 | 16,419,513 |
| dataset (energy) | bcc Fe 4x4x4, 100 cfg | **465 ms** | **802 ms** | 11,300,233 | 17,759,959 |
| dataset (energy+torque) | bcc Fe 4x4x4, 100 cfg | **1452 ms** | **1845 ms** | 18,827,426 | 25,287,343 |
| `fit` (OLS, energy) | 46 columns | 0.20 ms | 0.15 ms | 320 | 326 |
| `fit` (OLS, co-fit w=0.5) | 46 columns, 38400 torque rows | 25.6 ms | 29.1 ms | 341 | 351 |
| `build_salc_basis` | Nd2Fe14B 68 at., nbody 3, cutoff 4.0 | 386 ms | 347 ms | 13,408,478 | 12,434,646 |
| `_design_energy` | Nd2Fe14B, 103 cfg | 120 ms | 121 ms | 1,723,810 | 1,725,398 |
| `_design_torque` | Nd2Fe14B, 103 cfg | 275 ms | 299 ms | 1,099,600 | 1,101,188 |
| `solve_coefficients` OLS | M=10000, P=800 | 2236 ms | 2117 ms | — | — |
| `solve_coefficients` Ridge | M=10000, P=800 | 126.6 ms | 126.7 ms | — | — |

**Two differences are large enough to matter to the backport route choice:**

1. **`build_salc_basis` is ~29% faster in SLCE** (1998 vs 2803 ms; 60.8M vs 70.9M
   allocations) at identical output (`n_salcs = 129` both, no drops on this
   fixture), and ~10% faster on Nd2Fe14B. Whatever SLCE gained since the carve-out
   was never backported. Under a facade route this comes for free; under a
   verbatim-port route it has to be ported or written off.
2. **Dataset construction is ~1.7x SLOWER in SLCE** (802 vs 465 ms energy-only).
   SLCE's dataset does strictly more work — the resolvability classification and
   the ASR reparametrisation, which SCEFitting does not have. This is a real cost
   of the facade route, and it is paid once per dataset, not per fit.

### Repairs made while taking the baseline

- **`bench/bench_solver.jl` was broken in BOTH packages**: it calls
  `solve_coefficients`, which is `public` but *not* exported, so `using SLCE` /
  `using SCEFitting` alone left it undefined and the script died on line 13.
  Fixed with an explicit `using <Pkg>: solve_coefficients`. The script had been
  dead since the export/public split; nothing runs `bench/` in CI, so nothing
  caught it.
- **`bench/Manifest.toml` no longer resolved** in SLCE.jl: the package gained a
  `Printf` dependency after the manifest was last written (2026-07-30), so every
  bench script failed at load with `Package SLCE does not have Printf in its
  dependencies`. Re-resolved.
