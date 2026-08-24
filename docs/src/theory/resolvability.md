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
one. [`unresolvable_columns`](@ref) reports it exactly for a given basis.

A tie can also be *near* rather than exact: DFT-relaxed coordinates satisfy their space
group only to within the symmetry tolerance, so distances that would tie exactly on ideal
coordinates split by a small relative amount. When that split exceeds the same-distance
band, symmetry-partner pairs keep *different* tie images, the candidate set loses group
closure, and the build refuses ("cluster enumeration is not closed under the space
group"). The remedy is `SLCEBasis(...; tie_tol)` — widen the band above the coordinate
symmetry residual (relative to the bond length) so the near-ties re-merge; the frozen
content of the merged shell is then classified here exactly as for an exact tie. Keep the
band well below genuine shell spacings (hard cap `1e-2`).

Everything in this subsection assumes the tied images end up in **one orbit**, which needs
the point group to permute them. When it does not — low symmetry — the same tie takes a
different algebraic form and costs more; that is the [next
subsection](#When-symmetry-does-not-fuse-the-tie), and it is the more dangerous of the two.
Measured examples of the fused case —
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

### When symmetry does not fuse the tie

Everything above needed the point group to permute the tied images, which is what puts them
in **one** orbit whose sum weights them equally. That equal weighting is symmetry, so the
surviving even-``L_f`` content is a genuine coupling and stays.

In low symmetry — ``P1``, monoclinic — no operation relates the two equidistant images, so
they sit in **different orbits** carrying **independent** couplings. Nothing cancels and no
column is zero — and the two orbits' columns are not equal to each other either, since a
member's tensors carry its own bond geometry. What collapses is the **span**: every member
of either orbit reads its sites' displacements off the same reference-cell atoms, so the
two orbits span the *same space of functions* of anything this cell can express. The data
therefore fix only how much of that space is used in **total**, never how it divides
between the two couplings.

The undetermined fraction is not "half" in general: it is ``\text{(frozen columns)} -
\operatorname{rank} S[:, \text{frozen}]``, which is ``1 - 1/k`` of the block for a
``k``-fold tie the point group separates completely, and something in between when the
group fuses the images only partially — in which case the two faces coexist in one basis.

A minimal example: a ``P1`` cell whose atoms 1 and 2 differ by exactly half the ``a`` axis,
so their pair sits on the WS face with two equidistant images and no symmetry between them.

```@example resb
using SLCE
cr = Crystal(Lattice([3.0 0 0; 0 3.0 0; 0 0 3.0]),
             [0.0 0.5 0.25; 0.0 0.2 0.3; 0.0 0.1 0.42], [1, 1, 1], ["Fe"])
b = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2,
                            sectors = [Sector(disp = (degree = 1:2,), sites = 1:2,
                                              cutoff = 2.6)]))
frozen = unresolvable_columns(b)
(n_salcs(b), length(frozen))
```

Both images are at the same distance, and each is its own orbit — that is the whole
condition:

```@example resb
using LinearAlgebra
A, fr = cr.lattice.vectors, cr.frac_positions
orbits = unique([(s.key.body, s.key.orbit_id, round(norm(A * (fr[:, m.atoms[2]] .+
                     m.shifts[2] .- fr[:, m.atoms[1]] .- m.shifts[1])); digits = 4))
                 for s in SLCE.salcs(b) for m in s.members if sort(m.atoms) == [1, 2]])
```

**There is no justified way to split that sum.** Equal division is what aliasing-aware
force-constant codes do, but for two bonds no symmetry relates it is an interpolation
ansatz, not a measurement: the two images differ by a lattice vector ``\boldsymbol R_1 -
\boldsymbol R_2`` of *this* cell, so their phases ``e^{2\pi i \boldsymbol q\cdot\boldsymbol
R}`` agree only where ``\boldsymbol q = \boldsymbol 0``. Off ``\Gamma`` the split is pure
choice, and it is a choice that shows up in a published number.

So the package **drops the whole interaction**: every column of every orbit that shares an
atom group with another orbit is held at exactly zero, and [`fit`](@ref) names the atom group
and the count. This deliberately discards the determined sum as well — with the interaction
gone, data that contains it can no longer be fitted exactly, and that nonzero residual is
the intended signal:

```@example resb
using Random
rng = MersenneTwister(3)
truth = SLCEModel(b, 0.0, randn(rng, n_salcs(b)))          # touches the dropped orbits
data = [(u = 0.03 .* randn(rng, 3, 3);
         lattice_datum(predict_energy(truth, nothing, u); displacements = u,
                       forces = predict_force(truth, nothing, u), reference = cr))
        for _ = 1:200]
f = fit(SLCEFit, SLCEDataset(b, data), OLS(); force_weight = 0.4)
round(r2_energy(f); digits = 3)     # < 1: the dropped shell is real in this data
```

**What exactly is lost, and what is only unknown.** The 18 dropped columns split evenly:
the structural expansion has rank 9 over them, so **9 dimensions are the sums — determined by
the data — and the other 9 are the differences, which nothing on this cell can see.** Dropping
the interaction discards both halves. Keeping the determined half would mean writing the sum
down on one image or spreading it over both, i.e. choosing the split, which is the thing
without a basis. (Note that arbitrary values on the frozen columns *are* visible in the
energy — that is the determined half. Only a pure difference is invisible.)

Why the alternative is worse is worth seeing, since it is the failure this treatment exists
to prevent. Two models differing by a pure difference direction are indistinguishable in
everything the cell can express and identical at ``\Gamma``, yet far apart off it — measured
on this fixture and gated in `test/unit/test_resolvability.jl` gate (G):

| quantity | difference between the two models |
|----------|----------------------------------:|
| ``E(\boldsymbol u)`` | ``5\times10^{-19}`` |
| forces | ``4\times10^{-17}`` |
| ``\boldsymbol D(\boldsymbol 0)`` | ``2\times10^{-17}`` |
| ``\boldsymbol D(0.3, 0.1, 0.2)`` | ``4\times10^{-2}`` |

Before the freeze covered this case a *fit* on exactly this cell reached ``\text{rmse}_E =
4\times10^{-16}`` with a clean [`asr_residual`](@ref) and an exact ``\boldsymbol
D(\boldsymbol 0)``, while ``\boldsymbol D(\boldsymbol q)`` came out **52 % wrong**, every
coefficient of the tied shell was arbitrary, and nothing warned — because each column on its
own was perfectly nonzero.

One more practical note: the check is cheap where it does not apply. The structural pre-check
asks only whether some atom group is reached twice, so a cell with unique minimum images pays
nothing for the question.

Real crystals in their standard settings are mostly *not* affected, and the reason is
symmetry — measured with space groups from Spglib, on a spin + displacement basis:

| crystal (standard cell) | ops | columns | fused tie (zero) | unfused tie (dropped) |
|-------------------------|----:|--------:|-----------------:|----------------------:|
| bcc Fe, 1NN and 1+2NN | 96 | 5 | 2 | 0 |
| B2 FeRh | 48 | 6 | 2 | 0 |
| hcp Co | 8 | 12 | 2 | 0 |
| wurtzite GaN, 2.1 Å / 3.3 Å | 4 | 40 / 76 | 8 / 28 | 0 |
| rocksalt MnO (cubic 8-atom) | 192 | 18 | 4 | 0 |

The unfused case is what to expect from monoclinic and triclinic settings, and from any cell
built by hand without symmetry.

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
the cluster members retain ``R``. The same enumeration carries a second, non-fitting
consumer: it is the only way to *write* a monatomic cell's own-image bond, which is how a
compact reference model reaches a tiling consumer that expands the images onto a supercell
where they become distinct sites. On the reference cell those columns are redundant — every
image carries the same spin — so [`SLCEDataset`](@ref) refuses such a basis at the fitting
door with an [`UnclassifiableBasis`](@ref); their coefficients are set by hand or
transferred from a supercell fit. Below half the smallest perpendicular cell width,
``\text{cutoff} < \min_d d_i/2``, the two selections coincide (each in-cutoff image is
already the minimum).
