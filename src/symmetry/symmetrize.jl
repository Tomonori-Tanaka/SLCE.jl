# `symmetrize` and the distance of a crystal from its space group. Ported from
# SphericalTensorFC `4bb2929` (`src/symmetry/symmetrize.jl`), adapted to SLCE's names.

"""
    symmetrize(crystal; backend = SpglibBackend(), tol = 1e-5) -> Crystal

The crystal that has the space group of `crystal` exactly: the same atoms in the same
order, with the lattice and the positions moved onto the symmetric ones.

A structure taken from a relaxation, or written to a file with a few decimals, has its
symmetry only to some residue. The symmetry search accepts it (`tol` is the distance in
Å below which an atom and the image of another count as the same), but what is built
afterwards compares distances far more tightly: two bonds that the group maps onto
each other must have the same length to a relative `1e-8` (`tie_tol` of
[`SLCEBasis`](@ref)), or the build stops with "not closed under the space group".
`symmetrize` removes the residue. Call it once, on the crystal, and give the result to
[`SLCEBasis`](@ref); for a structure with a large residue pass the `tol` at which the
search finds the group you expect, and build with the default afterwards.

What it does, with the operations `(W|t)` that `backend` finds at `tol`:

- **the lattice** becomes the one whose metric (the lengths of the lattice vectors and
  the angles between them) is the average of the given metric over the operations,
  oriented like the given lattice — the lattice [`symmetrized_lattice`](@ref) reports;
- **each atom** goes to the average of the places the operations assign to it: for
  every operation, the image of the atom that the operation maps onto this one. In
  fractional coordinates the operations act on the positions linearly, and averaging
  over a group is the projection onto what the group leaves invariant;
- **the centroid of the atoms stays where it was**, in fractional coordinates: each
  operation is taken with the translation that fits the positions best, not the one
  the search returned, which is known to `tol` only and would shift the crystal as a
  whole.

The part of the structure that the symmetry leaves free is kept: the lattice
parameters, and the free coordinates of the occupied positions, come out as the
average over the equivalent ones. An exactly symmetric crystal is returned unchanged up
to round-off, and so is any crystal with `NoSymmetry()`. A non-periodic direction
(`pbc`) stays non-periodic.

The group is the one the search finds at `tol`, and a search that is given too small
a `tol` for the residue of the structure returns a smaller group without complaint
(the Spglib backend warns when ten times `tol` finds more operations). The result is
then exactly symmetric under that smaller group. The group and the size of the residue
that was removed are therefore logged: check that the group is the one you expect.

The package does not call this itself: the crystal a basis is built on is the one the
caller gave. **Do not pass the result as the `reference` of [`read_extxyz`](@ref)** for
data computed on the original structure: the reader refuses a lattice that differs from
the file's by more than `1e-6` Å, but it reads a difference in the positions as a
DISPLACEMENT — once the removed residue exceeds that same `1e-6` Å, every frame carries
it as `u`, and a spin-only file reads as a joint one. Read the data against the crystal it was computed on, and build
the basis on whichever of the two the data set's own geometry matches.
"""
function symmetrize(crystal::Crystal; backend::AbstractSymmetryBackend = SpglibBackend(),
                    tol::Real = 1e-5)::Crystal
    spacegroup = analyze_symmetry(backend, crystal; tol = tol)
    rotations = [op.rotation_frac for op in spacegroup.ops]
    lattice = Lattice(_symmetric_lattice(crystal.lattice.vectors, rotations);
                      pbc = Tuple(crystal.lattice.pbc))
    correction = _symmetrizing_correction(crystal, spacegroup)
    residual = _symmetry_residual(crystal, spacegroup)
    @info("symmetrized onto $(spacegroup.symbol) (no. $(spacegroup.number), " *
          "$(n_ops(spacegroup)) operations)",
          position_residual = residual.position, lattice_residual = residual.lattice)
    return Crystal(lattice, crystal.frac_positions + correction, crystal.species,
                   crystal.species_labels)
end

# What `symmetrize` adds to the fractional positions (3 × n_atoms, not reduced to
# [0, 1)): the mean over the operations of the offsets of `_image_offsets`, the
# projection onto what the group leaves invariant.
function _symmetrizing_correction(crystal::Crystal,
                                  spacegroup::SpaceGroup)::Matrix{Float64}
    offsets = _image_offsets(crystal, spacegroup)
    return dropdims(sum(offsets; dims = 3); dims = 3) / n_ops(spacegroup)
end

# offsets[:, j, g]: where operation `g` puts the atom that it maps onto atom `j`, minus
# the position of `j` — fractional, reduced to the nearest lattice translation, and
# with the mean over the atoms taken off for each operation. All zero for a crystal
# that has the operations exactly.
#
# The reduction rounds every component, also along a direction that is not periodic,
# where an offset is far below 1/2 anyway.
#
# The mean is taken off because the translation of an operation is only known to the
# tolerance of the search: the mean is the error of that translation, not a property of
# the crystal. What is left is measured against the best translation for the operation.
function _image_offsets(crystal::Crystal, spacegroup::SpaceGroup)::Array{Float64,3}
    positions = crystal.frac_positions
    offsets = zeros(Float64, 3, n_atoms(crystal), n_ops(spacegroup))
    for (g, op) in enumerate(spacegroup.ops)
        for i = 1:n_atoms(crystal)
            j = spacegroup.map_sym[i, g]
            offset = op.rotation_frac * positions[:, i] + op.translation_frac -
                     positions[:, j]
            offsets[:, j, g] = offset - round.(offset)
        end
        offsets[:, :, g] .-= sum(offsets[:, :, g]; dims = 2) / n_atoms(crystal)
    end
    return offsets
end

# How far `crystal` is from having the operations of `spacegroup` exactly, as
# `(; position, lattice)`:
#
# - `position` (Å): the largest distance between the image of an atom under an
#   operation and the atom the operation maps it onto, with the translation of the
#   operation refitted to the positions (`_image_offsets`). Zero for one atom.
# - `lattice` (a pure number): the largest change of the metric `G = AᵀA` under an
#   operation, `max |Wᵀ G W − G|`, relative to the largest entry of `G`. A relative
#   change of the longest lattice vector by `ε` gives about `2ε`; of a shorter one,
#   less by the ratio of the squared lengths.
#
# Both are zero up to round-off for the result of `symmetrize`.
function _symmetry_residual(crystal::Crystal, spacegroup::SpaceGroup)
    A = crystal.lattice.vectors
    offsets = _image_offsets(crystal, spacegroup)
    position = maximum(norm(A * offsets[:, j, g])
                       for j in axes(offsets, 2), g in axes(offsets, 3))
    metric = A' * A
    lattice = maximum(maximum(abs, op.rotation_frac' * metric * op.rotation_frac - metric)
                      for op in spacegroup.ops) / maximum(abs, metric)
    return (; position, lattice)
end
