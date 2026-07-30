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
members** — in the infinite crystal they are genuinely different bonds:

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

### What a tie costs on a finite cell

Distinct members are *not* independently resolvable. Every image of a tie joins the **same
two atoms of the reference cell**, and a [`TrainingDatum`](@ref) carries one spin and one
displacement per reference-cell atom — so all tied members see the identical arguments, and
the orbit sum runs over them with whatever signs the invariant carries. Any content that is
**odd** under the operations permuting the tied images therefore cancels identically.

A tie is *necessary* for this to happen — with a unique minimum image every member has its
own atom content and nothing can cancel — but which content dies depends on how the group
relates the tied members, not on the tie alone, so the reliable statement is the measured
one. [`unresolvable_columns`](@ref) reports it exactly for a given basis. Measured examples —
the counts depend on the group, so each row names the cell and the symmetry it was built with:

| basis | tie | columns | identically zero |
|-------|----:|--------:|-----------------:|
| bcc cross pair, harmonic (degree 2) | 8 | 2 | 1 (the ``L_f = 2`` one) |
| bcc cross pair, degree 3 | 8 | 8 | 8 |
| bcc cross pair, pure spin (`soc`, ``l \le 2``) | 8 | 7 | 4 |
| bcc in a cell doubled along ``z`` only, harmonic | 4 | 10 | 6 |
| bcc as ``2\times2\times2``, harmonic | 1 | 14 | 0 |

The bcc rows are the two-atom cubic cell under ``m\bar 3m``, the doubled one under
``4/mmm``; all five are the fixtures `test/unit/test_resolvability.jl` gates, so they are
reproducible verbatim. Two further contrasts, measured on the Fe–O–Fe chain of the
[joint guide](../guide/joint.md) (Fe at ``z = 0, c/2`` and O at ``c/4, 3c/4``, so both
like-atom pairs sit at exactly ``c/2``; space group from Spglib): its spin ``\times``
degree-1 channel loses **all 4** columns, every one of them odd ``L_f``, while the harmonic
channel of the *same* pairs loses **none** of its 6, every one of them even ``L_f``.

The pattern in the two-fold rows is the one to expect from bond reversal
``\hat{\boldsymbol r} \to -\hat{\boldsymbol r}``, which an invariant of bond rank ``L_f``
carries as ``(-1)^{L_f}`` while the site factors — indexed by the same reference-cell atom
for either image — are unchanged: the odd-``L_f`` content of such a channel goes, the
antisymmetric DMI-like ``L_f = 1`` one included, and the even-``L_f`` content stays. Treat
that as a guide to *where to look*, not as a rule: the projection is taken over the
stabilizer of one member, and whether the transported invariants then cancel is a property of
the whole group, so a larger tie can remove even-``L_f`` content too (the eight-fold rows
above) and a smaller group can leave odd-``L_f`` content standing.

It also has a precondition that is easy to miss: bond reversal has to map a member to
*itself*, which needs the two ends to carry the same decoration. A **two-fold** tie is
therefore no guarantee either. Measured on the stripe-AFM ``P4/mmm`` fixture of
`test/unit/test_forceconstants.jl` — a ``6\times6\times3`` cell whose like-atom pairs sit at
exactly half a cell edge, so every tie is two-fold — the spin ``\times`` degree-1 channel
carries a spin factor on one end and spin ``\times`` displacement on the other, so reversal
does not preserve the term at all, and **all 7** of its columns are identically zero, with
``L_f = 1, 1, 2, 1, 2, 3, 3`` — even ranks included. The same cell's lattice-only degree-2
channel and its spin ``\times`` degree-2 channel lose **none**.

Such a coefficient is **unidentifiable, not absent**, and the reason is the supercell. Tiling
this cell maps the tied images onto *distinct* atoms of the larger cell, so nothing cancels
there and the same coefficient multiplies a function that is nonzero throughout a Monte-Carlo
run. That argument holds for every frozen column, whatever channel it sits in — which is why
the columns are kept in the basis rather than dropped.

A uniform strain *sometimes* reveals it too, and it is worth knowing when it does not. A
strain field is not cell-periodic, so it displaces the tied images by different amounts and
the cancellation can be incomplete: measured, [`affine_energy`](@ref) is nonzero for 6 of the
8 frozen columns of the bcc degree-3 channel and 4 of 10 on a spin ``\times`` degree-1 one.
But it is exactly zero for the bcc harmonic row above, and it is *structurally* zero for
every **pure-spin** column — such a SALC has no displacement slot for a strain to act on, so
`affine_energy` reduces to [`predict_energy`](@ref), which is the annihilated orbit sum. Where
the strain response does survive it is the response to the relative bond-vector change
``\boldsymbol M\cdot\boldsymbol d``, whose transverse part is a *rotation* of the bond
rather than a stretch, so do not read it as a bond-length derivative.

The remedy is a reference cell in which the pair's minimum image is **unique**: break the tie
in *every* direction whose separation component equals half the cell length. Doubling one
axis is not automatically enough — describing bcc with a cell doubled along ``z`` only
reduces the corner tie from eight images to four, and the dead channels stay dead, whereas a
``2\times2\times2`` cell puts the cross pair strictly inside the WS cell and resolves it.

### What the read-outs do about it

The fit is where a frozen column is invisible; the read-outs are where it matters, because
[`force_constants`](@ref), [`strain_derivatives`](@ref),
[`exchange_strain_derivatives`](@ref), [`magnon_phonon_vertices`](@ref) and
[`decorated_terms`](@ref) differentiate the individual cluster *members*, and there the tie
does not cancel. So each of them says so once, in whichever of the two directions applies:

- **the coefficient is zero** (what a fit on this cell returns) — the deliverable carries no
  contribution from that channel, and, per the supercell argument above, that is not zero
  physics. In particular a Monte-Carlo run on a supercell built from such a model is missing
  a coupling the supercell itself could express.
- **the coefficient is nonzero** — a hand-built or externally fixed value, which no data on
  this cell can have determined, reaching ``\Phi`` and every ``\boldsymbol q \neq
  \boldsymbol 0`` of [`dynamical_matrix`](@ref). Legal, and worth naming.

One consequence deserves stating on its own: **the tie is invisible at** ``\boldsymbol q =
\boldsymbol 0``. The ``\Gamma`` sum ``\sum_{\boldsymbol R}\Phi(\boldsymbol R)`` is the Hessian
of exactly the energy this cell can express, so two models differing only in a frozen
coefficient have identical ``\boldsymbol D(\boldsymbol 0)`` — acoustic modes and all — and
differ off ``\Gamma``. A model can therefore pass every ``\Gamma``-point check and still carry
a dispersion the training data never constrained.

## Which interactions the enumeration can represent

The admissibility rule is a statement about **pairs of atoms**, not a bound on `cutoff`.
Spelled out positively, a pair interaction is representable exactly when

1. the two ends are **distinct atoms of the reference cell** — different columns of
   `crystal.frac_positions`, whatever their species; and
2. their separation is the **minimum-image** one, within the radial cutoff for that
   species pair (`cutoff = Inf` keeps the whole WS cell).

Condition 2 is *not* a sphere of radius ``L/2``. In a cubic cell an atom at the origin and
one at the body centre sit at ``\sqrt3\,L/2 \approx 0.866\,L`` — the WS corner — and are
perfectly representable; the pair is admitted at its own minimum-image distance without
any cutoff cap. Nothing here needs `cutoff` to be small.

What condition 1 excludes is an atom paired with **its own periodic image**,
``(a, \boldsymbol 0)``–``(a, \boldsymbol R)``, at any distance and under any cutoff, and
likewise an ``N``-body cluster reusing one atom. The consequence differs by channel:

- **Spin channel** — nothing is lost. Both ends carry the same
  ``\hat{\boldsymbol e}_a``, so the term is a constant (``L_f = 0``) or a one-body alias
  (``L_f > 0``), never an independent pair.
- **Displacement channel** — nothing is lost *for the fit* either, for the same reason
  one step removed: a [`TrainingDatum`](@ref) stores one displacement per reference-cell
  atom, i.e. a cell-periodic field, so ``\boldsymbol u_{a,\boldsymbol 0} =
  \boldsymbol u_{a,\boldsymbol R}`` and a self-pair's energy contribution is a function
  of ``\boldsymbol u_a`` alone — the design-matrix columns are again dependent.
- **Lattice-dynamics readouts** — here it *is* visible, because
  [`force_constants`](@ref) speaks a wider language than the fit does. Its keys are
  ``\Phi[(a,\boldsymbol 0),(b,\boldsymbol R)]``, and no same-atom off-site block is ever
  emitted: ``\Phi_{aa}(\boldsymbol R \neq \boldsymbol 0)`` is absent by construction, so
  the ``\boldsymbol q``-dependence such a block would carry is absent from
  [`dynamical_matrix`](@ref) too.

So the rule for the lattice channel is: **to resolve force constants between atoms of one
sublattice, describe the crystal with a reference cell in which those atoms are distinct
atoms.** A one-atom cell has no pair content at all, and says so — `build_asr` warns that
the basis admits no translation-invariant displacement content, every displacement
coefficient is constrained to zero, and ``D(\boldsymbol q) \equiv 0``. Describing the same
crystal with a two-atom cell turns the same bonds into distinct-atom pairs and the content
appears. This is the same requirement as the aliasing rule that a force-constant fit wants
a reference cell at least three periods wide along each fitted direction.

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
