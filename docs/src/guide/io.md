# Persistence and I/O

```@meta
CurrentModule = SLCE
```

Building a SALC basis is the expensive step, and a fitted model is worth keeping. This
page covers three I/O seams: saving and reloading models, building a basis from a
human-authored `input.toml`, tabulating coefficients, and reading DFT training data
through the code-agnostic source interface.

## Saving and reloading a model

`SLCE.save` / `SLCE.load` serialize a basis or a model to a
self-contained, human-readable **TOML** document. (They are intentionally *not* exported —
the names clash with FileIO / JLD2 / CSV — so qualify them.)

```julia
SLCE.save("model.toml", SLCEModel(f))      # or save("basis.toml", basis)
model = SLCE.load(SLCEModel, "model.toml")
predict_energy(model, configs)
```

The document stores the crystal, the space-group operations, the basis spec, the full
SALC basis, and (for a model) `j0` and the per-`SALCKey` coefficients. On reload the basis
is rebuilt *verbatim* (no re-projection), and coefficients are re-paired to it **by key**,
not by position — so a reloaded model predicts identically even if the basis were rebuilt
in a different order. TOML is chosen over JSON deliberately: stdlib `TOML` round-trips
`Float64` exactly and expresses the deep nested SALC document, so input and dump share one
zero-dependency format.

## A human-authored `input.toml`

Instead of constructing `Crystal` / `BasisSpec` in Julia, you can describe the *setup*
(crystal + interaction + optional symmetry) in a TOML file and build the basis from it.
Training data and the estimator stay in Julia.

```toml
# input.toml
[structure]
lattice   = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]]   # each entry = one cell vector
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75]]
species   = [1, 1, 1, 1]
species_labels = ["Fe"]

[interaction]
nbody    = 2
cutoff   = 2.6              # or `inf` for the whole Wigner–Seitz cell
lmax     = [1]
soc      = false

[symmetry]
backend = "spglib"          # or "none"
tol     = 1.0e-5
```

```julia
basis = SLCEBasis("input.toml")     # reads [symmetry] backend/tol from the file
```

The `[interaction]` section also takes the label-keyed / per-body forms (see
[The interaction specification](basis.md#The-interaction-specification)) — species
tables with a `"*"` fallback, a per-body-order `lsum`, and cutoffs per body order
and species pair:

```toml
[interaction]
nbody    = 3
soc      = false

[interaction.lmax]
"*" = 3
B   = 0

[interaction.lsum]          # keys = body orders; omitted orders are uncapped
1 = 0
2 = 4
3 = 4

[interaction.cutoff]        # body-keyed scalars ...
2 = 8.0

[interaction.cutoff.3]      # ... or a species-pair table per order
"Fe-*" = 6.0
"*-*"  = 8.0
```

[`read_setup`](@ref) returns the parsed setup (including the image selection) if you want
to inspect it before building.

## Tabular coefficients

[`coeftable`](@ref) returns an [`SLCECoefficients`](@ref) — a **Tables.jl** source with one
row per SALC. The columns are read straight off each [`SALCKey`](@ref) (the stable
design-matrix-column identity), plus the fitted coefficient:

| Column | Meaning |
|---|---|
| `body` | body order ``N`` of the cluster (2 = pair, 3 = triplet, …) |
| `orbit_id` | index of the cluster symmetry orbit at that body order |
| `ls` | per-site angular momenta ``(l_1,\dots,l_N)``, as a comma-joined string |
| `Lf` | final coupled angular momentum ``L_f`` of the invariant |
| `block` | disambiguates independent ``l``-orderings / coupling paths sharing the same `(body, orbit_id, ls, Lf)` |
| `J` | the fitted coefficient ``j_\varphi`` for this SALC (DFT energy unit, e.g. eV) |

The rows are in design-matrix column order, the same order as [`coef`](@ref) and
`basis.salc_basis.keys`. It drops straight into any table or IO package:

```julia
using DataFrames
df = DataFrame(coeftable(f))        # or CSV.write("J.csv", coeftable(f))
intercept(f)                        # the reference energy j0 (not a row)
```

The boundary is deliberate: the library owns the mapping from internal storage
(`SALCKey` + coefficient position) to meaningful, labeled rows; the caller brings the
table / IO / plotting package.

## Reading DFT data

DFT-code I/O is isolated at the *training-data boundary*. The core owns only the
**abstract seam**: a source implements [`read_configs`](@ref)`(src) -> Vector{TrainingDatum}`,
and the SLCE pipeline only ever sees the code-agnostic [`TrainingDatum`](@ref) /
[`SLCEDataset`](@ref) — once you have the data, the originating code is irrelevant.

```julia
basis   = SLCEBasis(crystal, interaction)
src     = some_source                                    # any AbstractDFTSource
dataset = SLCEDataset(basis, src)                         # read_configs(src) under the hood
fit(SLCEFit, dataset, OLS(); torque_weight = 0.5)
```

A [`TrainingDatum`](@ref) is one configuration's observables: the spin channel
(directions + moment magnitudes) plus energy are required, everything else —
displacements, forces, constraining field/torques — is optional, with `nothing`
meaning *not observed* (distinct from an observed zero). Spin-only data are built
with the [`spin_datum`](@ref) convenience constructor; `spin_datum(energy, moments)`
(no field) covers collinear/Ising configurations and codes without constrained
noncollinear output, which contribute energy rows only. Each datum carries a
[`DatumProvenance`](@ref): `torque_qualified` gates whether its torque rows enter a
co-fit (auto-derived — a zero field does not qualify unless you explicitly assert a
converged unconstrained run), `setup_id`/`soc` pin one computational setup per
dataset, and `reference_id`/`reference_fingerprint` (see
[`crystal_fingerprint`](@ref)) pin the clamped-ion reference for displacement data.
A dataset may freely **mix** torque-bearing and energy-only configurations from one
setup — the torque design keeps rows exactly for the qualified ones.

Against a **displacement-decorated basis** the same `TrainingDatum` path is the
joint entry point: designs are evaluated at each configuration's ``(e, u)`` (a datum
without displacements contributes ``u = 0`` exactly — atoms at the pinned
reference), and with `use_force = true` (the default, mirroring `use_torque`) the
per-atom forces ``\boldsymbol f_a = -\partial E/\partial \boldsymbol u_a`` of every
force-carrying configuration enter the force design block for a three-block co-fit
(`fit(...; torque_weight, force_weight)` — see the
[fitting guide](fitting.md#Forces:-the-three-block-co-fit)).

### Lattice-only data

At the other end of the joint expansion sits the user with no magnetic state at all —
force constants and phonons, nothing spin-dependent. [`lattice_datum`](@ref) is
[`spin_datum`](@ref)'s counterpart for them, and the whole chain then runs without a
spin argument anywhere:

```julia
data = [lattice_datum(E[i]; displacements = u[i], forces = F[i], reference = crystal)
        for i in eachindex(E)]
ds  = SLCEDataset(basis, data)             # use_torque resolves to false by itself
f   = fit(SLCEFit, ds, OLS(); force_weight = 1.0)
fcs = force_constants(SLCEModel(f); order = 2)     # `spins` not required either
```

### Both channels at once

The joint case — a magnetic state *and* a displaced structure, the expansion's whole
reason for existing — is [`joint_datum`](@ref). It does the spin-side derivation of
[`spin_datum`](@ref) and the reference stamp of [`lattice_datum`](@ref) in one call,
which is exactly the combination that a hand-written `TrainingDatum(; ...)` has to get
right twice:

```julia
data = [joint_datum(E[i]; moments = m[i], field = B[i],
                    displacements = u[i], forces = F[i], reference = crystal)
        for i in eachindex(E)]
ds = SLCEDataset(basis, data)
f  = fit(SLCEFit, ds, OLS(); torque_weight = 0.3, force_weight = 0.3)
```

`lattice_datum` fills the mandatory spin channel with the `ẑ` placeholder direction and
**exactly zero** moment magnitudes. Zero is the load-bearing part: a lattice-only basis
reads no spin site, so the placeholder is never touched, and feeding the same data to a
basis that *does* carry spin content trips the zero-moment invariant by name instead of
fitting a fabricated ferromagnet. `use_torque` (and the omission of `spins`/`e`) resolve
from the **basis**, never from what the data happen to contain — a lattice-only basis
has no torque block to build, while "the adapter dropped the constraining fields" must
stay the loud error it is today.

The torque target from a constrained calculation is
``\boldsymbol\tau_a = \boldsymbol m_a \times \boldsymbol B_a`` (the physical /
Landau–Lifshitz torque), the *same* physical quantity, sign, and layout as the model's
[`predict_torque`](@ref) ``= -\hat{\boldsymbol e}_a \times \partial E/\partial\hat{\boldsymbol e}_a``,
so the co-fit is consistent.

The **concrete DFT-code adapters** live in the companion
[SLCETools.jl](https://github.com/Tomonori-Tanaka/SLCETools.jl) package, not in the core. For
VASP:

```julia
using SLCE, SLCETools
using SLCETools.VASP: read_poscar, Oszicar

crystal = read_poscar("POSCAR")                          # → Crystal
basis   = SLCEBasis(crystal, interaction)
src     = Oszicar(["run1/OSZICAR", "run2/OSZICAR"])      # an AbstractDFTSource (constrained NCL)
dataset = SLCEDataset(basis, src)
fit(SLCEFit, dataset, OLS(); torque_weight = 0.5)
```

Adding another code is one more sibling adapter in SLCETools — the core and its exports do
not change. SLCETools.VASP also writes the *inverse* direction (sampled configurations →
constrained-noncollinear INCAR / input sets) for generating new training data.

### Legacy Magesty training sets (EMBSET)

The one concrete format the core ships is Magesty's **EMBSET** training set — it is
DFT-code-agnostic (energies plus per-atom moment and constraining-field vectors, exactly
what a spin-only [`TrainingDatum`](@ref) stores), so an existing Magesty data set drops
straight in:

```julia
dataset = SLCEDataset(basis, EmbsetFile("EMBSET"))        # or read_embset("EMBSET")
```

See [`read_embset`](@ref) for the format and its validation rules (the reader is
cross-checked against Magesty's own in the oracle suite).

Next: [Sunny export](sunny.md).
