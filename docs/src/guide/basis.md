# Building the basis

```@meta
CurrentModule = SLCE
```

The first half of the workflow turns a crystal and an interaction specification into a
symmetry-adapted SALC basis — the fixed set of invariants ``\Phi_\varphi`` whose
coefficients the fit will recover. This page covers the four ingredients:
[`Crystal`](@ref), [`BasisSpec`](@ref), the periodic image selection, and the symmetry
backend, all assembled by [`SLCEBasis`](@ref).

## Geometry: `Lattice` and `Crystal`

A [`Lattice`](@ref) holds the cell vectors as columns; a [`Crystal`](@ref) adds
fractional atomic positions, integer species labels, and per-species names.

```@example basis
using SLCE
import Spglib

lat = Lattice([3.0 0 0; 0 3.0 0; 0 0 3.0])           # columns are the cell vectors aᵢ
cr  = Crystal(lat,
              [0.0 0.5; 0.0 0.5; 0.0 0.5],            # 3 × n_atoms fractional positions
              [1, 1],                                 # species index per atom
              ["Fe"])                                 # label per species
(n_atoms(cr), cr.species_labels)
```

Periodic directions default to all three axes; pass `pbc = (true, true, false)` to a
`Lattice` for a slab. Fractional positions are wrapped into `[0, 1)` — a precondition the
neighbor list relies on.

## The interaction specification

A [`BasisSpec`](@ref) fixes the body order, the cutoff radii, the angular truncation
(per-species `lmax` and per-body-order `lsum`), and whether to keep only the isotropic
channel:

```@example basis
interaction = BasisSpec(; nbody = 2, cutoff = 2.7, lmax = [1], soc = false)
```

- `nbody` — maximum cluster size. `2` is pairwise; `nbody = 3` adds triplets, and so on
  (see [Body order beyond pairs](#Body-order-beyond-pairs)).
- `cutoff` — the radius (Å) within which two atoms form an edge, per body order and
  species pair (a scalar broadcasts to all). It may be `Inf`,
  meaning *every resolvable pair* — the whole Wigner–Seitz cell of the supercell.
- `lmax` — the maximum harmonic degree per species, at every body order. `[1]` is dipolar
  (the only `l` that builds the Heisenberg/DM/Γ bilinears); `[2]` adds quadrupolar
  single-ion and biquadratic channels. `0` removes a species from the basis entirely.
- `lsum` — an optional per-body-order budget on `Σl` over a cluster's sites (omit for
  no cap). This is how a Magesty-style `lsum` model is expressed: `lsum = [2 => 4]` keeps
  the pair channels `(1,1), (1,3), (2,2)` while `lmax = [3]` alone would also admit
  `(3,3)`.
- `soc` — `false` keeps only the rotation-invariant scalar channel (`L_S = 0`, e.g.
  Heisenberg `eᵢ·eⱼ`) — the exact SOC-free selection; `true` (the default) keeps the
  anisotropic channels too. (The former `isotropy = true` keyword is this `soc = false`.)

With several species the ergonomic forms keep a spec readable — species by **label**
(with a `"*"` fallback), cutoffs per body order and species pair (unordered keys,
resolved by specificity: concrete beats `"A-*"` beats `"*-*"`):

```julia
spec = BasisSpec(crystal; nbody = 3, soc = false,
    lmax   = ["*" => 3, "B" => 0],
    lsum   = [1 => 0, 2 => 4, 3 => 4],
    cutoff = [2 => Inf, 3 => ["Fe-*" => 6.0, "*-*" => 8.0]])
```

Label-keyed forms need the species labels, so pass the `Crystal` (or a label vector) as
the first argument. Every species/pair must be covered (add a `"*"` entry as the
fallback), a body order outside `nbody` is an error rather than silently ignored, and an
`N`-body cluster is kept iff **every** internal edge lies within its own species-pair
radius for that order. `display(spec)` prints the resolved truncation table.

## Sector tables: the joint spin–lattice truncation

The dense form above describes a pure-spin (clamped-ion) basis. The joint
spin–lattice expansion is specified as a **sector table** — a union of
[`Sector`](@ref) rows, each declaring one family of decorated clusters: its
spin content (an explicit rank multiset like `[1, 1]`, or a
`(sites, lmax, lsum)` truncation — `sites` counts **spin-decorated** sites, and the
NamedTuple accepts no other keys), its displacement content (a total-degree
budget; the `(k, l)` harmonic labels realizing each degree are derived), a
per-sector SOC switch, and a per-sector cutoff:

```julia
spec = BasisSpec(crystal; lmax = 2, pmax = ["*" => 0, "Fe" => 3], sectors = [
    Sector(disp = (degree = 2:3,), cutoff = 6.0),          # force constants
    Sector(spin = (sites = 2:4, lmax = 2), cutoff = 8.0),  # pure-spin SLCE
    Sector(spin = [1, 1], disp = (degree = 1,), soc = false, cutoff = 5.0)])  # dJ/dr
basis = SLCEBasis(crystal, spec)
```

A row's own `sites` keyword caps the **total** decorated-site count (spin and
displacement sites together, and one site may carry both factors). It defaults to every
body order the row's content can realize.

!!! warning "`Sector(; sites = …)` is a count, not [`BasisSpec`](@ref)'s `nbody` cap"
    An `Int` here means **exactly** that many decorated sites: `sites = 3` is a 3-body
    sector, not "up to 3-body". `BasisSpec(; nbody = 3)` means the opposite, `1:3`. Write
    a range (`sites = 1:3`) when you want the inclusive reading. Carrying the `BasisSpec`
    habit into a sector row silently drops every lower-body term, and the fit still runs
    and still reports a good ``R^2`` on a basis that cannot express exchange.

The admitted labels are the union of the rows, intersected with the global
per-species `lmax`/`pmax` caps (a ligand species is `lmax = 0, pmax > 0`) and
the per-body `lsum`; `nbody` and the per-body cutoff envelope are derived from
the table. `soc = false` keeps only the total-spin-scalar (`L_S = 0`) blocks of
that sector's labels — the exact SOC-free selection rule (for a pure-spin
sector this is the old `Lf = 0` "isotropy" filter, which is why the dense form's
keyword is now spelled `soc`). Overlapping sectors are safe: each label enters
the basis exactly once (key-union invariant), with SOC granted if *any*
admitting sector grants it. A sector-expressed pure-spin spec builds the same
basis, bit for bit, as the dense form.

Spin configurations alone cannot train a displacement-decorated basis — the
spin-configuration `SLCEDataset` path refuses such a basis. Build the dataset from
[`TrainingDatum`](@ref) vectors instead, which carry the displacements and forces;
see [Persistence and I/O](@ref).

### Which sector feeds which deliverable

A sector's displacement `degree` is the *total polynomial degree* in `u`, and
[`force_constants`](@ref) at `order = n` collects exactly the terms of degree `n`.
So the `dJ/dr` row above (`degree = 1`) contributes to the energy and the **forces**,
and contributes nothing to the harmonic constants: a basis whose only spin-carrying
displacement sector is `degree = 1` produces force constants that are bit-identical
for every magnetic state. For magnetic-state-dependent phonons, give a spin-carrying
sector `disp = (degree = 2,)` (`force_constants` warns when the basis is shaped the
other way).

The same care applies to the *symmetry* the basis imposes. A basis with spin content
is projected with the paramagnetic grey group, and evaluating it at a fixed magnetic
state reduces that to the magnetic space group of the state — antiunitary elements
included, which is what makes the force constants right without the magnetic group
ever being declared. A lattice-only basis has no such reduction available: it imposes
the paramagnetic group unconditionally, which for an ordered state is too large. This
is documented with measured parameter counts under [`force_constants`](@ref).

## Periodic resolvability: which pairs are physical

A plain-PBC supercell can only resolve interactions whose displacement lies in its
**Wigner–Seitz cell** — a farther periodic image of an atom carries the *same spin*, so its
interaction is an alias of the minimum-image one and cannot be fit independently. The
[`MinimumImage`](@ref) selection (the default) enumerates exactly this set, with
boundary ties kept as distinct members; [`AllImages`](@ref) keeps every image within the
cutoff. That selection serves the (future) spin-spiral / generalized-Bloch path and, since
a self-image cluster is the only way to *write* a monatomic cell's nearest-neighbor bond,
it is also how a compact reference model is handed to a tiling consumer, which expands the
images into distinct supercell sites. Such a basis cannot be **fitted** on its own cell —
every image of the atom carries the same spin, so the function collapses to a single-site
one — and [`SLCEDataset`](@ref) refuses it; its coefficients are set by hand or transferred
from a supercell fit.

```@example basis
# `build_neighbor_list` is public but unexported — call it qualified.
nl = SLCE.build_neighbor_list(cr, Inf, MinimumImage())   # full Wigner–Seitz cell
length([p for p in nl.pairs if (p.i, p.j) == (1, 2)])  # the 8-fold body-diagonal corner tie
```

This is a load-bearing distinction with subtleties at the cell boundary (and for `N ≥ 3`
clusters); it has its own chapter, [Periodic resolvability](../theory/resolvability.md).
You rarely call `build_neighbor_list` directly — `SLCEBasis` threads the selection through
for you — but `cutoff = Inf` in the `BasisSpec` is how you ask for the whole cell.

## Symmetry backends

Symmetry collapses the raw cluster instances into orbits and projects the harmonic
products onto the space-group-invariant SALCs. The backend is pluggable:

- [`NoSymmetry`](@ref) — the identity only (in-tree, no dependency). Every directed cluster
  is its own orbit, so the basis is larger but still correct.
- [`SpglibBackend`](@ref) — real space-group symmetry via [Spglib](https://github.com/singularitti/Spglib.jl).
  Provided by an extension: `import Spglib` activates it.

```@example basis
basis = SLCEBasis(cr, interaction; backend = SpglibBackend())
(basis.spacegroup.symbol, basis.spacegroup.number, n_salcs(basis))
```

!!! tip "Loading Spglib"
    Use `import Spglib`, not `using Spglib` — Spglib exports `Lattice` / `Crystal`
    names that would clash with this package's. Importing only activates the extension.

## Body order beyond pairs

`nbody = K` enumerates clusters of up to `K` atoms as *pairwise-within-cutoff cliques* of
distinct atoms. At `N ≥ 3` something genuinely new happens: a space-group operation can
permute equivalent sites, mixing the coupling paths and — for unequal `l` — the
`l`-orderings. The SALC projection runs over the combined (ordering × coupling-path ×
`Lf`) space, so a single SALC can carry several `l`-orderings as *terms*. The
[kagome three-body tutorial](../tutorials/kagome_threebody.md) shows such a multi-term
channel; the mechanism is described in [Architecture](../theory/architecture.md).

The projection's internal ordered-image bookkeeping never reaches the user: the
construction's last step folds the members into a **canonical, duplicate-free form**
— one member per physical cluster instance (sites sorted, `shifts[1] = 0` anchored),
one term per site→`l` assignment — so evaluation, model files, and downstream
consumers pay for each instance exactly once.

## What `SLCEBasis` produces

[`SLCEBasis`](@ref) bundles the crystal, the space group, the SALC basis, and the
spec it was built from (the `spec` field). Each basis function is a [`SALC`](@ref)
addressed by a canonical
[`SALCKey`](@ref) — a stable identity `(body, orbit_id, decors, L_S, Lf, block)` that
names a fixed interaction independent of construction order. `decors` is the sorted
multiset of per-site decorations (a pure-spin key's is exactly its old sorted
`l`-multiset, readable back with `spin_ls`) and `L_S` is the total coupled spin rank —
the label the SOC switch and the staged-fit selectors read. The design-matrix
columns, the persisted coefficients, and [`coeftable`](@ref) all key off it.

You can also build a basis from a human-authored `input.toml` instead of constructing the
objects in Julia — see [Persistence and I/O](io.md).

Next: [Data and fitting](fitting.md).
