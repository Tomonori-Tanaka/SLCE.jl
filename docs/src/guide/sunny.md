# Sunny export

```@meta
CurrentModule = SLCE
```

A fitted SCE model is an energy surface; for linear spin-wave theory you want it as a
[Sunny.jl](https://sunnysuite.github.io/Sunny/) `System`. [`to_sunny`](@ref) builds a real
`Sunny.System` from the Sunny-representable channels of a model. It is provided by an
extension that activates under `using Sunny`.

```julia
using Sunny
sys = to_sunny(model; spin_length = 3/2)        # a real Sunny.System
```

## What is exported

Sunny represents two channels, and those are exactly what the conversion keeps:

| SCE channel | Sunny term |
|-------------|------------|
| `ls = [1,1]` 2-body (bilinear) | a `3 × 3` exchange matrix (Heisenberg + Dzyaloshinskii–Moriya + symmetric Γ in one) |
| `ls = [2]` 1-body | a traceless-symmetric single-ion anisotropy tensor |
| `ls = [0…]` | folded into the constant `j0` |

Every other channel — three-body and higher, higher `l` — **cannot** be represented by a
Sunny `System` and is **reported as skipped**, never silently dropped:

```julia
sys = to_sunny(model; spin_length = 3/2)        # @warn lists any skipped channels
```

## Spin length, scaling, and mode

Spins enter only at assembly. The SCE couplings are fit with *unit* directions, so they
absorb the moment magnitude; `spin_length` supplies the physical effective spin
``S_{\text{eff}} = m/(g\mu_B)`` (a number, or a per-species `label => S` mapping), and the
`scaling` keyword chooses how it enters:

- **`scaling = :moment`** — put ``S_{\text{eff}}`` directly into Sunny's `Moment` and
  rescale each bilinear bond `J = M / (Sₐ S_b)`. Both the static energy *and* the magnon
  dispersion are exact — but Sunny's `Moment` accepts only **half-integer** spins, so
  ``S_{\text{eff}}`` must be one.
- **`scaling = :coupling`** — keep `Moment` at a placeholder ``s_0 = 1`` and fold
  ``S_{\text{eff}}`` into the couplings instead. Works for **any** positive
  ``S_{\text{eff}}`` (itinerant / non-half-integer moments); the magnon *dispersion* is
  exact, while the represented static energy is rescaled (the dispersion is invariant
  under an overall spin rescale).
- **`scaling = :auto`** (default) — `:moment` when every ``S_{\text{eff}}`` is a
  half-integer, else `:coupling`.

The single-ion term carries the classical (`mode = :dipole_uncorrected`,
factor ``1/s^2``) or quantum rank-2 (`mode = :dipole`, factor ``2/(s(2s-1))``, requires
``s > 1/2``) rescaling; `mode = :auto` picks `:dipole` for half-integer spins.

```julia
sys = to_sunny(model; spin_length = 1.5, g = 2, mode = :dipole)    # :auto → scaling = :moment
sys = to_sunny(model; spin_length = 1.1)                           # :auto → scaling = :coupling
sys = to_sunny(model; spin_length = Dict("Fe" => 3/2, "O" => 1))   # per-species
```

## Two placements: exact vs. unfolded

```julia
to_sunny(model; spin_length, placement = :explicit)   # exact, on the training supercell
to_sunny(model; spin_length, placement = :primitive)  # unfolded onto the chemical primitive cell
to_sunny(model; spin_length, placement = :auto)       # primitive if clean, else fall back (default)
```

- **`:explicit`** places the couplings directly on the training supercell. The classical
  energy reproduces `predict_energy − j0` exactly, but the magnon dispersion is folded
  into the supercell Brillouin zone.
- **`:primitive`** recovers the chemical primitive cell from the space group's pure
  translations, groups supercell atoms into sublattices, and places one Sunny bond per
  *primitive* bond (no multiplicity — Sunny's periodic replication restores it), giving the
  *unfolded* dispersion. This is exact only when the model genuinely lives on the primitive
  cell; a `clean` check detects when it does not (the interaction range reaches the
  supercell boundary) and falls back to the exact supercell route.
- **`:auto`** (the default) uses `:primitive` when clean, otherwise `:explicit`.

## How it is validated

The numerically load-bearing conversion math — the harmonic-to-Cartesian matrices, the
directed-member folding, the primitive unfold — lives in a **dependency-free core layer**
and is gated *without Sunny* by reconstructing the energy
(`_reconstruct_energy ≈ predict_energy − j0`). The thin extension only assembles the
`Sunny.System` from the core-computed matrices, and a separate test environment checks the
real `Sunny.System` energy against the SCE energy for both placements. The rationale is in
[Architecture](../theory/architecture.md#The-core/extension-split).
