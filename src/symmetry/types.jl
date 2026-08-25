"""
    SymOp

A single space-group operation `(W | t)`.

# Fields
- `rotation_frac::SMatrix{3,3,Float64,9}`: rotation in fractional coordinates `W`.
- `rotation_cart::SMatrix{3,3,Float64,9}`: rotation in Cartesian coordinates,
  `A·W·A⁻¹` with `A` the lattice (used to rotate spin directions / harmonics).
- `translation_frac::SVector{3,Float64}`: translation `t` (fractional).
- `is_proper::Bool`: `det(rotation_cart) > 0`.
- `is_translation::Bool`: `W == I` (a pure translation).
"""
struct SymOp
    rotation_frac::SMatrix{3,3,Float64,9}
    rotation_cart::SMatrix{3,3,Float64,9}
    translation_frac::SVector{3,Float64}
    is_proper::Bool
    is_translation::Bool
end

"""
    SpaceGroup

Space-group data for a `Crystal`, produced by [`analyze_symmetry`](@ref).

# Fields
- `symbol::String`: international (Hermann–Mauguin) symbol. When the crystal declares
  an aperiodic axis and [`analyze_symmetry`](@ref) had to drop operations that close
  only through the periodicity along it, the symbol carries a `" (pbc subgroup)"`
  suffix — the group held is then a proper subgroup of the named one. The suffix marks
  DROPPED operations only: on an aperiodic axis the translations are re-seated on the
  representative the finite structure has whether or not anything was dropped, so an
  unsuffixed symbol does not promise a backend's `t` verbatim.
- `number::Int`: space-group number (`1` for P1). Deliberately left at the number of
  the group the backend named, even when `symbol` carries the suffix: there is no
  international number for the subgroup, and the pair records what was reported and
  what was kept.
- `ops::Vector{SymOp}`: the operations.
- `map_sym::Matrix{Int}`: `map_sym[atom, op]` is the atom that `atom` maps to
  under operation `op` (a permutation of atoms for each op). Derived in-tree, so
  every backend shares the same convention.
- `translation_ops::Vector{Int}`: indices of the pure-translation operations.
- `tol::Float64`: tolerance used for symmetry matching.
"""
struct SpaceGroup
    symbol::String
    number::Int
    ops::Vector{SymOp}
    map_sym::Matrix{Int}
    translation_ops::Vector{Int}
    tol::Float64
end

"Number of symmetry operations."
n_ops(sg::SpaceGroup)::Int = length(sg.ops)
