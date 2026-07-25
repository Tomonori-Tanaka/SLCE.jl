# Getting started

## Installation

SLCE.jl is an exploratory package and is not registered. Add it from its
local path (or a git URL) in the Julia package manager:

```julia
using Pkg
Pkg.develop(path = "/path/to/SLCE.jl")
```

The core has only lightweight dependencies. Optional features live in package
extensions that light up when you load their backend:

| Load this | Activates |
|-----------|-----------|
| `import Spglib` | real space-group symmetry (`SpglibBackend`) |
| `using GLMNet` | the `Lasso` / `ElasticNet` estimators |
| `using Sunny` | [`to_sunny`](@ref) export to a `Sunny.System` |

```julia
using Pkg
Pkg.add(["Spglib", "GLMNet", "Sunny"])   # only the ones you need
```

## A first fit: recover a Heisenberg coupling

The shortest meaningful end-to-end: build a basis for a spin chain, generate synthetic
Heisenberg energies ``E = J\sum_{\langle ij\rangle}\hat{\boldsymbol e}_i\cdot\hat{\boldsymbol e}_j``,
fit, and recover ``J``.

```@example gs
using SLCE
import Spglib                       # activate the SpglibBackend extension
using LinearAlgebra, Random

# A 4-atom chain of magnetic atoms along z.
lat   = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
frac  = [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75]
chain = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])

# Nearest-neighbor, 2-body, isotropic (Heisenberg) channel only.
interaction = BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], soc = false)
basis       = SLCEBasis(chain, interaction; backend = SpglibBackend())

(basis.spacegroup.symbol, n_salcs(basis))     # space group, number of SALC basis functions
```

There is a single SALC — the nearest-neighbor Heisenberg invariant. Now synthesize
training data from a known coupling and fit it:

```@example gs
rng = MersenneTwister(2026)
randcfg(nat) = mapreduce(_ -> (v = randn(rng, 3); v / norm(v)), hcat, 1:nat)

heis    = SLCE.salcs(basis)[1]   # the Heisenberg SALC (public-unexported: qualify)
J_true  = 0.0137
configs = [randcfg(4) for _ = 1:40]
E = [J_true * sum(dot(c[:, m.atoms[1]], c[:, m.atoms[2]]) for m in heis.members)
     for c in configs]

f = fit(SLCEFit, SLCEDataset(basis, configs, E), OLS())
(; r2 = r2_energy(f), J = 2 * sqrt(3) * coef(f)[1], J_true)
```

The fit is exact (``R^2 = 1``) and recovers the coupling: the SALC normalization makes
``J = 2\sqrt{3}\,j_\varphi``, a fixed relation between the physical coupling and the
fitted coefficient.

`fit` returns an [`SLCEFit`](@ref) — the heavyweight result that keeps the dataset and
answers diagnostics. When you only need to predict or persist, convert it to the
lightweight [`SLCEModel`](@ref) with `SLCEModel(f)` (see [Persistence and I/O](guide/io.md)).

## Add the torque

The same coefficients fix the per-atom torque
``\boldsymbol\tau_a = -\hat{\boldsymbol e}_a \times \partial E/\partial\hat{\boldsymbol e}_a``
(the Landau–Lifshitz / physical torque ``\boldsymbol m_a \times \boldsymbol B_{\mathrm{eff},a}``).
Pass per-configuration torques and a `torque_weight` to run an energy + torque co-fit:

```@example gs
# analytic Heisenberg torque for the synthetic data: τ = ∇E × e = −e × ∇E
function heis_torque(c, J)
    nat = size(c, 2); G = zeros(3, nat)
    for m in heis.members
        i, j = m.atoms[1], m.atoms[2]
        G[:, i] .+= J .* c[:, j]
        G[:, j] .+= J .* c[:, i]
    end
    reduce(hcat, cross(G[:, a], c[:, a]) for a = 1:nat)
end
torques = [heis_torque(c, J_true) for c in configs]

fc = fit(SLCEFit, SLCEDataset(basis, configs, E, torques), OLS(); torque_weight = 0.5)
(; r2_energy = r2_energy(fc), r2_torque = r2_torque(fc))
```

`predict_torque(fc, config)` now returns the analytic derivative of the same energy
surface `predict_energy(fc, config)` evaluates — the two are consistent by construction.

## Where to go next

- [Building the basis](guide/basis.md) — geometry, interaction range and periodic
  resolvability, symmetry backends, body order beyond pairs.
- [Data and fitting](guide/fitting.md) — the co-fit objective, the estimators
  (`OLS`/`Ridge`/`Lasso`/`ElasticNet`), and the fit diagnostics.
- [Tutorials](tutorials/index.md) — fuller, narrated runs.
