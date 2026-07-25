# Tanaka & Gohda — Magesty.jl methods paper (the formal SCE spec)

**Citation.** T. Tanaka and Y. Gohda, "Magesty.jl: A Julia package for the
efficient construction of spin-cluster expansion models" (manuscript, in
preparation). Builds on the foundational SCE / basis-function literature:
R. Drautz & M. Fähnle, *Phys. Rev. B* **69**, 104404 (2004) (SCE origin);
S. Singer & M. Fähnle (2006) and Dietermann *et al.* (2011) (symmetry-adapted
basis construction); and the related T. Tanaka & Y. Gohda, *Phys. Rev. Research*
**8**, 023300 (2026).

Draft source (user-local, not part of this repo):
`~/Documents/research/papers/writing/Magesty/main.tex`.

## Why it matters here

This is the **formal specification** of the construction `SLCE`
reimplements from scratch. Reading it independently confirmed that every numerical
convention in the rebuild matches the published formalism (not just the Magesty.jl
code) — a second, authoritative anchor beyond the oracle.

## Conventions confirmed to match (paper ⇒ rebuild)

- **Product basis** `Φ^prod = (4π)^{n/2} ∏_a Z_{l_a}^{m_a}`, with `1 ≤ l_a ≤ l_max`
  (paper Eq. 5) ⇒ the `(4π)^{N/2}` design scale and `l_s ≥ 1`.
- **Left-coupling tree** `l_1⊗l_2→L_{12}→…→L_f`, path label
  `L_seq = (L_{12},…)` (Eq. 8–11) ⇒ `AngularMomentum.coupling_paths` / `Lseq`.
- **Complex coupling tensor** by Clebsch–Gordan chaining (Eq. C2), with the running
  `M` fixed by the CG selection rule ⇒ `coeff_tensor_complex`.
- **Complex→real unitary** `U^{(l)}` (Eq. B2): `U_{00}=1`, `U_{+m,+m}=(−1)^m/√2`,
  `U_{+m,−m}=1/√2`, `U_{−m,−m}=i/√2`, `U_{−m,+m}=−i(−1)^m/√2` ⇒ matches
  `AngularMomentum.c2r_matrix` **entry for entry**.
- **Global reality phase** `exp[−i(π/2)(Σ_a l_a − L_f)]` (Eq. C5) ⇒ exactly the
  factor in `complex_to_real_tensor`.
- **Torque** — the methods paper Eq. 15 writes `τ_i = e_i × ∇_{e_i} E` (the
  energy-rotation-gradient) and notes "the Landau–Lifshitz dynamical torque has the
  opposite sign". The rebuild deliberately uses the **Landau–Lifshitz / physical**
  convention `τ_i = −e_i × ∇_{e_i} E = m_i × B_eff,i` instead (matching the published
  *General spin models*, PRR 8 023300 (2026), torque), so `predict_torque` /
  `_design_torque` carry the **opposite sign of Eq. 15**. This is sign-only: the DFT
  training target `m × B` is flipped to match, the co-fit objective is invariant, and the
  fitted `J` are unchanged — see the "deliberate difference" note below.
- **Combined loss** `(1−w)/N_c‖·‖² + w/N_T‖·‖²` with row-whitening
  `√((1−w)/N_c)`, `√(w/N_T)` (Eq. 17, 21) and **analytic `j0`**
  `J_0 = ȳ_E − x̄_E·J` (Eq. 19) ⇒ exactly `fit`.
- **Isotropic restriction** = keep only `L_f = 0`: this is the *no-SOC* physics
  (global spin-rotation invariance forces `L_f ≥ 1` coefficients to zero; Eq. consequence
  (a)). The rebuild's `isotropy = true` is this restriction; `L_f ≥ 1` channels are
  for the SOC case.

## One deliberate difference (labeling, not physics)

The paper labels each SALC `ν = (O, l, L_seq, L_f, k)` — it keeps the seed coupling
path `L_seq` in the label. The rebuild's `SALCKey = (body, orbit_id, ls, Lf, block)`
**folds `L_seq` (and, at `N ≥ 3`, the `l`-ordering) into `block`**, because its
projection runs over the *combined* (ordering × path × `Mf`) space rather than
per-path. Both yield the same invariant-subspace dimension per `(O, l, L_f)` —
verified to agree exactly with Magesty through 3-body (see
`docs/design-notes.md` §3b). `block` is kept injective across the split orderings.

(PDF, if copied, would live at `papers/tanaka-magesty-methods.pdf` — local only.)
