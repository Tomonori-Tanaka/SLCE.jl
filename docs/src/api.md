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
FixedCoefficients
islinear
solve_coefficients
```

## Prediction

```@docs
predict_energy
predict_torque
predict_force
has_torque
has_force
asr_residual
```

## Acoustic sum rule (translation invariance)

`fit` enforces the ASR by default on displacement-decorated bases (see the
[fitting guide](guide/fitting.md#The-acoustic-sum-rule-(translation-invariance)));
the machinery is public but unexported.

```@docs
SLCE.ASRReparam
SLCE.build_asr
```

## Rotational invariance (a diagnostic, not a constraint)

Translation is the only affine invariance the package *imposes*. Rigid rotation and
vanishing stress are measured instead — see the
[fitting guide](guide/fitting.md#Rotational-invariance-(measured,-not-imposed)).

```@docs
affine_energy
rotational_residual
rotation_transfer_residual
```

## Staged (hierarchical) fitting

`fit`'s `frozen` / `sector_mask` keywords fit a model in physical stages (see the
[fitting guide](guide/fitting.md#Staged-(hierarchical)-fits)); the selector table
is public but unexported.

```@docs
SLCE.sector_columns
SLCE.is_soc_free
```

## Diagnostics

The `*_energy` / `*_torque` / `*_force` accessors are the full, per-observable
surface. The generic names `coef` / `nobs` / `dof` / `predict` / `residuals` / `r2`
**extend StatsAPI** (imported, not shadowed) and default to the energy block, so they
compose with the StatsBase / GLM ecosystem.

```@docs
coef
intercept
nobs
dof
identifiability
predict
residuals
r2
r2_energy
rmse_energy
r2_torque
rmse_torque
r2_force
rmse_force
rss_energy
rss_torque
rss_force
residuals_energy
residuals_torque
residuals_force
```

## Model selection

The fit-accuracy-vs-Monte-Carlo-cost workflow: `salc_groups` / `group_costs` /
`cost_weights` (public, unexported — call as `SLCE.salc_groups` etc.) build the
per-group cost weights of a [`GroupAdaptiveRidge`](@ref), and `group_freedom` reports
how much of the null space an ASR leaves each group; `gcv` / `effective_dof` are
the closed-form hat-matrix diagnostics of the linear estimators; `select_fit` drives
the λ path and applies the cost-aware Pareto rule; `cross_validate` is the generic
configuration-grouped K-fold assessment for comparing any estimators or
`torque_weight` settings on both error axes.

```@docs
gcv
effective_dof
select_fit
LambdaPath
select_support
SupportPath
cross_validate
CVResult
SLCE.salc_groups
SLCE.group_costs
SLCE.cost_weights
SLCE.group_freedom
```

## Tabular coefficients

`coeftable` extends `StatsAPI.coeftable`.

```@docs
coeftable
SLCECoefficients
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
`decorated_terms` is the general per-term dump and the surface new consumers should
read: it accepts joint (spin + displacement) models, labels every tensor axis with its
own `(channel, k, l)`, and carries the `(4π)^{n_spin_slots/2}` scale as a field — take
it from there, never from the cluster shape. `spin_multipole_terms` is its frozen pure-spin
predecessor, kept bit-identical for the consumers written against it and refusing any
displacement-decorated model; `restrict(model, :spin)` is the bridge that turns a joint
model into one it accepts (the clamped-ion sub-model — read its warning, it is not a
refit). `bilinear_terms` is the bilinear (`ls=[1,1]`) and single-ion (`ls=[2]`)
extraction as Cartesian `3×3` matrices (the same validated extraction the Sunny export
consumes). The tesseral spherical-harmonic kernel
[`SLCE.Harmonics`](@ref Harmonics) is the stable submodule those consumers pair it
with — see the next section.

```@docs
DecoratedTerm
decorated_terms
restrict
SpinMultipoleTerm
spin_multipole_terms
bilinear_terms
SLCE.Slot
```

## Lattice dynamics

The displacement channel's physics deliverables: the exact force constants of a
fitted model **at a given spin configuration** (that dependence is the point of a
spin–lattice expansion), and their reciprocal-space form. An acoustic sum rule the
model actually satisfies shows up here as three zero eigenvalues of `D(0)` — see
[`asr_residual`](@ref).

```@docs
ForceConstantSet
force_constants
dynamical_matrix
write_phonopy
write_alamode
```

## Homogeneous strain

The same monomial coefficients, contracted with site positions instead of being
differentiated: `strain_derivatives` returns the exact `∂ⁿE_cell/∂εⁿ` at the model's
own reference. `order = 1` is the reference stress times the volume, `order = 2` the
**clamped-ion** elastic tensor times the volume; both are functions of the magnetic
state, which is what magnetoelasticity is.

The strain measure is Biot (Seth–Hill `m = 1`): the deformation is *defined* as
`F = I + ε` with `ε` symmetric, so `u_i − u_j = ε·d_ij` is exact rather than a
linearization. `order = 1` is measure-independent across the Seth–Hill family;
`order = 2` and beyond are not.

!!! warning "The acoustic sum rule is a precondition here, not a caveat"
    A homogeneous strain displaces site `i` by `ε·(R_i − origin)`, so shifting the
    origin changes the energy by `ε : (Σ_i ∇_i E)`. Without the ASR the strain
    response is *undefined* — the origin is unbounded — so `strain_derivatives`
    throws rather than warns, at a tolerance tighter than the one
    [`asr_residual`](@ref) is quoted against elsewhere.

    At `order ≥ 2` the sum rule is necessary but **not sufficient**: it is an identity
    on cell-periodic fields, and only at `ε = 0` is the affine field one. The residue
    is the home-image gauge, which makes the clamped-ion elastic tensor a function of
    the crystal *description* as well as of the fit. That order therefore re-measures
    origin independence on every call and refuses a disagreement — see the
    [introspection guide](guide/introspection.md#The-acoustic-sum-rule-is-required,-and-at-order-≥-2-it-is-not-enough).

```@docs
strain_derivatives
```

## Magnetoelastic coupling (ε-linear)

Two deliverables ride on the ε-**linear** strain response, and only on it. That tier is
safe twice over: ε-linear content is Seth–Hill measure-independent unconditionally, and
the sum-rule argument above buys origin independence exactly where the affine field is
periodic, which is first order and no further.

`magnetoelastic_constants` compresses it into the two cubic constants, in a convention
this package pins and gates:

```
E_me / V = B₁ Σ_i ε_ii (α_i² − 1/3) + 2 B₂ Σ_{i<j} ε_ij α_i α_j
```

with **tensor** shear `ε_ij`, the `Σ_{i<j}` range, that sign, and `E_me` an energy
density. `exchange_strain_derivatives` keeps the resolution instead: `∂M_ab/∂ε` per bond
and `∂A_a/∂ε` per site, the `dJ/dr` of the model in the form a spin Hamiltonian consumes.

!!! warning "Clamped ion — and the qualifier rides in the return value"
    Neither deliverable applies the internal-strain relaxation `Λ = −Φ⁻¹Ξ`. The
    clamped-vs-relaxed difference is routinely a factor ~2 and can flip a sign, so
    `magnetoelastic_constants` returns `ion = :clamped` as a field rather than only
    saying so in prose: `result.B2` cannot be quoted without it.

```@docs
magnetoelastic_constants
exchange_strain_derivatives
ExchangeStrainDerivatives
```

## Magnon–phonon vertices

The force constants differentiate twice in `u`, the bilinear couplings twice in the spin
directions; `magnon_phonon_vertices` is the **mixed** derivative `∂²E/∂u ∂e` that couples
the two — a phonon eigenvector on one index, a magnon polarization on the other. The spin
derivative is **tangential** (the radial part is projected out, so `V·ê_b ≡ 0`), which is
why the result can be returned in Cartesian components without pinning anybody's
local-frame convention.

"Adiabatic" is a scope statement: these are derivatives of the *static* energy surface, so
retardation, the Berry-phase term that gives phonons angular momentum, and spin-lattice
relaxation are outside a static cluster expansion by construction.

```@docs
magnon_phonon_vertices
MagnonPhononVertices
```

## Effective models at a displaced structure

The same fact the force constants ride on — every displacement site factor is a
homogeneous polynomial — used to MOVE the expansion point instead of differentiating
at it. `effective_model(model; u0)` rewrites a fitted model as an exact expansion
around `R + u0`, which reaches a symmetry-broken distorted structure (a relaxed cell,
a condensed soft mode) with no common-subgroup grid, and is the starting point for
renormalizing coefficients onto a thermally displaced reference.

The result is deliberately **not** a [`SLCEModel`](@ref): the symmetry of the
displaced structure is the stabilizer of `u0`, generally a proper subgroup, so the
reference SALCs cannot span it.

```@docs
EffectiveModel
EffectiveTerm
effective_model
predict_energy(::EffectiveModel, ::AbstractMatrix{<:Real}, ::AbstractMatrix{<:Real})
```

## Sampler row tables

The contract a sampler builds its gather programs against: which `(channel, k, l, m)`
basis rows a site carries, in what order, and what goes in them. Blocks stack in
`Channel`-enum order with `SPIN` first at offset 0 — verbatim
`SLCE.Harmonics.lm_index`, so adding the displacement channel never moves a row a
spin-only consumer already addresses.

```@docs
SLCE.RowLayout
SLCE.row_layout
SLCE.row_index
SLCE.site_rows!
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

The **code-agnostic boundary** of the training-data input: the SLCE pipeline only ever sees
[`TrainingDatum`](@ref) / [`SLCEDataset`](@ref). Concrete DFT-code adapters (the VASP
POSCAR / OSZICAR reader and INCAR writer) live in the companion
[SLCETools.jl](https://github.com/Tomonori-Tanaka/SLCETools.jl) package
(`SLCETools.VASP.read_poscar` / `Oszicar` / `write_inputs`), so adding a code touches neither
the core nor its exports.

```@docs
AbstractDFTSource
TrainingDatum
DatumProvenance
spin_datum
lattice_datum
joint_datum
crystal_fingerprint
read_configs
```

The one in-core concrete format is Magesty's (code-agnostic) EMBSET training set,
for legacy-data reuse:

```@docs
EmbsetFile
read_embset
```

## Units

A fitted model is a zero-temperature energy surface, so nothing in this package takes a
temperature. What lives here is the *convention* the family's samplers
(`SLCEMonteCarlo`, `SLCEDynamics`, `SLCETools`) share for crossing the kelvin ↔
model-energy boundary — one definition, so two copies cannot drift apart.

```@docs
SLCE.KB_EV
SLCE.resolve_kt
```

## Persistence

`save` / `load` are intentionally **not exported** (the names clash with FileIO / JLD2 /
CSV); call them as `SLCE.save` / `SLCE.load`.

```@docs
SLCE.save
SLCE.load
```
