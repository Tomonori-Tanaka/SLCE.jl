# Periodic resolvability

```@meta
CurrentModule = SLCE
```

A spherical cutoff is the wrong primitive for *what a finite supercell can resolve*. This
chapter explains why the default selection enumerates the **Wigner–Seitz cell**, what the
boundary ties are, and how the same logic extends to three- and four-body clusters.

## Only minimum-image interactions are resolvable

Under plain periodic boundary conditions (ordinary DFT, excluding the generalized-Bloch
spin-spiral case), every periodic image of an atom carries the **same spin**. So if atom
``A`` reaches both the minimum image of ``B`` (at distance ``d``) and a farther image of
``B`` (at ``d' > d``), the two "interactions" are the *same* ``\hat{\boldsymbol e}_A \cdot
\hat{\boldsymbol e}_B`` in every training configuration — perfectly collinear design-matrix
columns. A supercell can only independently resolve the **minimum-image** set: the
displacements lying inside the Wigner–Seitz cell of the (super)lattice.

## The Wigner–Seitz cell is a polyhedron, not a ball

For a cubic cell of side ``L`` the WS cell is the cube ``[-L/2, L/2]^3``. Its inscribed
sphere has radius ``L/2`` (the face centers), but the farthest resolvable displacement is
the body-diagonal corner ``(L/2, L/2, L/2)`` at ``\sqrt 3\,L/2 \approx 0.866\,L``. A
spherical cutoff therefore *cannot* express "all resolvable pairs": to reach the corners it
must exceed ``L/2`` along the faces — exactly where it starts sweeping in the aliased
(non-resolvable) images. This face-vs-corner mismatch is the classic source of
``L/2``-plane double-counting errors.

So `cutoff` is not a sphere radius applied blindly. The default [`MinimumImage`](@ref)
selection keeps, per atom pair, only the minimum-image displacement(s), trimmed by the
radial cutoff; `cutoff = Inf` keeps the whole WS cell (Magesty spells this
`cutoff = -1`).

## Boundary ties

On the WS boundary several images are exactly equidistant, and they are kept as **distinct
members** — each is a genuinely different, independently resolvable geometry:

| WS boundary feature | Multiplicity |
|---------------------|-------------:|
| face (``L/2``) | 2 |
| edge | 4 |
| corner ``(L/2,L/2,L/2)`` | 8 |

For a body-centered cubic cell, the cross pair sits at all eight corners at once:

```@example res
using SLCE
cr = Crystal(Lattice([3.0 0 0; 0 3.0 0; 0 0 3.0]), [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
nl = SLCE.build_neighbor_list(cr, Inf, MinimumImage())   # public-unexported: qualify
length([p for p in nl.pairs if (p.i, p.j) == (1, 2)])     # 8 equidistant corner images
```

Self-pairs (``i = i + R``) are dropped: both ends share ``\hat{\boldsymbol e}_i``, so the
term is a constant (``L_f = 0``) or a one-body alias (``L_f > 0``), never an independent
pair. Likewise an ``N``-body cluster must use distinct atoms.

## The third edge: compact clusters at `N ≥ 3`

For pairs the WS boundary only multiplies tied images. For clusters of three or more atoms
it adds a genuinely new constraint. A triangle ``\{i, j, k\}`` is admissible only if **all
three edges sit at their atom-pair minimum image simultaneously** — the *compact-cluster*
criterion. Having each pair individually minimum-image-resolvable is **not** enough: the
images that make ``i\!-\!j`` and ``i\!-\!k`` minimal may force ``j\!-\!k`` onto a longer,
non-minimum image, which must reject the cluster.

A sharp illustration: three atoms equally spaced around a one-dimensional ring have every
pair minimum-image at the same distance, yet admit *zero* compact triangles — you cannot
realize all three minimal edges at once, just as an equilateral triangle does not embed on
a ring. The enumeration checks every pair of a candidate clique on its *actual chosen
images*, so it counts exactly the resolvable compact clusters, with the boundary ties
multiplying them as they do pairs.

This combinatorics is easy to get subtly wrong, so the full count — the candidate set for
``N = 2, 3, 4`` (including the whole WS cell) and the symmetry-orbit partition — is pinned
against an independent brute-force enumeration on cells deliberately seeded with face,
edge, and corner ties.

## The spin-spiral seam

[`AllImages`](@ref) keeps *every* image within the cutoff, each tagged with its lattice
translation ``R``. For plain-PBC fitting this over-counts (it admits aliases of shorter
bonds), so it is not the default — but it is the representation a **generalized-Bloch /
spin-spiral** extension needs, where the phase ``e^{i\boldsymbol q\cdot\boldsymbol R}``
distinguishes images that a single supercell cannot. This is why [`NeighborPair`](@ref) and
the cluster members retain ``R``. Below half the smallest perpendicular cell width,
``\text{cutoff} < \min_d d_i/2``, the two selections coincide (each in-cutoff image is
already the minimum).
