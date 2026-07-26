# API reference

```@meta
CurrentModule = SLCE
```

```@docs
SLCE
```

The public API, grouped by pipeline stage. It comes in two tiers: the **exported**
names (available after `using SLCE`) and the **public but unexported** names —
declared with the `public` keyword and reached by qualification
(`SLCE.salcs(basis)`, `SLCE.build_neighbor_list(...)`, …); the
[`SLCEBasis`](@ref) constructor drives the latter for you. The headline workflow is
built from `Crystal` + `BasisSpec` → [`SLCEBasis`](@ref) → [`SLCEDataset`](@ref) →
[`fit`](@ref) → [`SLCEModel`](@ref).

```@index
```

## Geometry

```@docs
Lattice
Crystal
n_atoms
cartesian_positions
interplanar_spacing
```

## Neighbor list and periodic image selection

```@docs
NeighborPair
NeighborList
build_neighbor_list
AbstractImageSelection
MinimumImage
AllImages
```

## Symmetry

```@docs
SymOp
SpaceGroup
AbstractSymmetryBackend
NoSymmetry
SpglibBackend
analyze_symmetry
n_ops
```

## Cluster orbits

```@docs
ClusterMember
ClusterOrbit
ClusterSet
build_clusters
```

## Decoration labels (joint spin–lattice keys)

The `isbits` value labels of the joint spin–lattice basis: a
[`SiteFactor`](@ref) is one per-site decoration factor (channel + `(k, l)`
labels), a [`SiteDecor`](@ref) the combined decoration of one site (at most one
factor per channel), and a [`SALCKey`](@ref) carries the sorted `SiteDecor`
multiset plus the total spin rank `L_S`. All public but unexported
(`SLCE.SiteDecor` etc.; `SLCE.Channel` would shadow `Base.Channel` if
exported).

```@docs
SLCE.Channel
SLCE.SiteFactor
SLCE.SiteDecor
SLCE.has_spin
SLCE.has_disp
SLCE.spin_rank
SLCE.disp_degree
SLCE.factors
SLCE.is_pure_spin
SLCE.spin_decors
SLCE.spin_ls
SLCE.rep_scale
```

## SALC basis

```@docs
SALCKey
SALC
SALCBasis
build_salc_basis
evaluate_salc
```

## BasisSpec, basis, dataset, model

`SLCEModel(fit)` extracts the lightweight predictor from a fit;
`SLCEModel(basis, j0, jphi)` assembles a synthetic model directly from a basis and
hand-set coefficients (both are documented under [`SLCEModel`](@ref)).

```@docs
BasisSpec
Sector
SLCE.SectorRule
SLCEBasis
SLCEDataset
SLCEModel
SLCEFit
n_salcs
salcs
read_setup
```

Datasets slice and concatenate without recomputing design rows (see
[Slicing and concatenation](guide/fitting.md#Slicing-and-concatenation)):

```@docs
Base.length(::SLCEDataset)
Base.getindex(::SLCEDataset, ::AbstractVector{<:Integer})
Base.vcat(::SLCEDataset, ::SLCEDataset...)
```

## Fitting

`fit` and `islinear` extend the [StatsAPI](https://github.com/JuliaStats/StatsAPI.jl)
generics of the same name (`islinear` is public but unexported).

```@docs
fit
refit
AbstractEstimator
OLS
Ridge
ElasticNet
Lasso
AdaptiveLasso
AdaptiveRidge
GroupAdaptiveRidge
PrecomputedPilot
islinear
solve_coefficients
```

## Prediction

```@docs
predict_energy
predict_torque
has_torque
```

## Diagnostics

The `*_energy` / `*_torque` accessors are the full, per-observable surface. The generic
names `coef` / `nobs` / `dof` / `predict` / `residuals` / `r2` **extend StatsAPI**
(imported, not shadowed) and default to the energy block, so they compose with the
StatsBase / GLM ecosystem.

```@docs
coef
intercept
nobs
dof
predict
residuals
r2
r2_energy
rmse_energy
r2_torque
rmse_torque
rss_energy
rss_torque
residuals_energy
residuals_torque
```

## Model selection

The fit-accuracy-vs-Monte-Carlo-cost workflow: `salc_groups` / `group_costs` /
`cost_weights` (public, unexported — call as `SLCE.salc_groups` etc.) build the
per-group cost weights of a [`GroupAdaptiveRidge`](@ref); `gcv` / `effective_dof` are
the closed-form hat-matrix diagnostics of the linear estimators; `select_fit` drives
the λ path and applies the cost-aware Pareto rule; `cross_validate` is the generic
configuration-grouped K-fold assessment for comparing any estimators or
`torque_weight` settings on both error axes.

```@docs
gcv
effective_dof
select_fit
SelectionPath
select_support
SupportPath
cross_validate
CVResult
SLCE.salc_groups
SLCE.group_costs
SLCE.cost_weights
```

## Tabular coefficients

`coeftable` extends `StatsAPI.coeftable`.

```@docs
coeftable
SCECoefficients
```

## Sunny export

```@docs
to_sunny
```

## Fitted-model introspection

A code-neutral view of a fitted [`SLCEModel`](@ref)'s multipole terms, the stable public
contract downstream packages (e.g. the mean-field samplers in
[`SLCETools.jl`](https://github.com/Tomonori-Tanaka/SLCETools.jl)) read instead of the
SALC-basis internals.
`multipole_terms` is the general per-term dump; `bilinear_terms` is the bilinear (`ls=[1,1]`)
and single-ion (`ls=[2]`) extraction as Cartesian `3×3` matrices (the same validated
extraction the Sunny export consumes). The tesseral spherical-harmonic kernel
[`SLCE.Harmonics`](@ref Harmonics) is the stable submodule those consumers pair it
with — see the next section.

```@docs
MultipoleTerm
multipole_terms
bilinear_terms
```

## Harmonics kernel

The `Harmonics` submodule (public, unexported — call as `SLCE.Harmonics.Zlm` etc.)
is the tesseral spherical-harmonic kernel, and a **stable surface for downstream
packages**: `Zlm` / `Zlm_unsafe`, `grad_Zlm` / `grad_Zlm_unsafe`, `lm_index`, `num_lm`,
and the tesseral normalization constants `Harmonics.N1 = √(3/4π)`,
`Harmonics.A2 = √(15/16π)`, `Harmonics.B2 = √(5/16π)` (the single definition of the
`l ≤ 2` tesseral ↔ Cartesian conversion factors, shared by the core's bilinear
extraction and downstream exchange mappings so the forward and inverse conversions
cannot drift apart).

The `_unsafe` variants skip input validation on the **unit-direction contract**: the
caller guarantees `u` is a unit 3-vector. SLCETools.jl's hot paths (the mean-field
samplers) call `Zlm_unsafe` under exactly this contract.

```@docs
Harmonics
Harmonics.Zlm
Harmonics.Zlm_unsafe
Harmonics.grad_Zlm
Harmonics.grad_Zlm_unsafe
Harmonics.lm_index
Harmonics.num_lm
```

## SolidHarmonics kernel

The `SolidHarmonics` submodule (public, unexported — call as
`SLCE.SolidHarmonics.Rlm` etc.) is the displacement-channel kernel: real solid
harmonics `Rₗₘ(u)` evaluated as homogeneous polynomials in the Cartesian
components of `u` (regular and exact at `u = 0`) together with their
**Euclidean** gradients — no tangent projection, in contrast to the spin-side
`Harmonics.grad_Zlm`; the two kernels are distinct objects and neither
substitutes for the other.

The normalization is **4π-free (Racah-type)**: on the unit sphere
`Rₗₘ(û) = √(4π/(2l+1)) · Harmonics.Zlm(l, m, û)`, and the rank-1 factors are
literally the Cartesian components `R₁₋₁, R₁₀, R₁₁ = y, z, x`. Only spin sites
carry the per-factor `(4π)^(−1/2)`, so the per-term design-matrix scale stays
`(4π)^(n_spin/2)` with displacement factors scale-free.

```@docs
SolidHarmonics
SolidHarmonics.solid_harmonics
SolidHarmonics.solid_harmonics!
SolidHarmonics.solid_harmonics_grad
SolidHarmonics.solid_harmonics_grad!
SolidHarmonics.Rlm
SolidHarmonics.grad_Rlm
SolidHarmonics.solid_harmonic_index
SolidHarmonics.num_solid_harmonics
```

## AngularMomentum kernel

The `AngularMomentum` submodule (public, unexported) carries the angular-momentum
recoupling machinery behind the SALC construction: Clebsch–Gordan coefficients, the
real (tesseral) Wigner-D representation, the complex↔real change of basis, and the
coupled real tensor bases over an `ls` multiset. Like `Harmonics` it is a
self-contained numeric kernel; unlike `Harmonics` it is not a declared downstream
surface — the SALC builder drives it.

```@docs
AngularMomentum
AngularMomentum.clebsch_gordan
AngularMomentum.wignerD_real
AngularMomentum.c2r_matrix
AngularMomentum.coupling_paths
AngularMomentum.build_real_bases
AngularMomentum.coeff_tensor_complex
AngularMomentum.complex_to_real_tensor
```

## DFT data sources

The **code-agnostic boundary** of the training-data input: the SCE pipeline only ever sees
[`TrainingDatum`](@ref) / [`SLCEDataset`](@ref). Concrete DFT-code adapters (the VASP
POSCAR / OSZICAR reader and INCAR writer) live in the companion
[SLCETools.jl](https://github.com/Tomonori-Tanaka/SLCETools.jl) package
(`SLCETools.VASP.read_poscar` / `Oszicar` / `write_inputs`), so adding a code touches neither
the core nor its exports.

```@docs
AbstractDFTSource
TrainingDatum
DatumProvenance
SpinDatum
crystal_fingerprint
read_configs
```

The one in-core concrete format is Magesty's (code-agnostic) EMBSET training set,
for legacy-data reuse:

```@docs
EmbsetFile
read_embset
```

## Persistence

`save` / `load` are intentionally **not exported** (the names clash with FileIO / JLD2 /
CSV); call them as `SLCE.save` / `SLCE.load`.

```@docs
SLCE.save
SLCE.load
```
