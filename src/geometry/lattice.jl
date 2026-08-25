"""
    Lattice(vectors; pbc = (true, true, true)) -> Lattice

A (possibly partially) periodic Bravais lattice.

# Arguments
- `vectors::AbstractMatrix{<:Real}`: 3×3 matrix whose **columns** are the lattice
  vectors `a₁ a₂ a₃` (Å).

# Keyword arguments
- `pbc = (true, true, true)`: periodicity along each lattice direction. An aperiodic
  axis suppresses the neighbor list's images along it, leaves fractional positions
  unwrapped there (there is no period to wrap with), and restricts the space group to
  the operations that do not close through that axis, re-seating the translations of
  the survivors on the representative the finite structure has — see
  [`analyze_symmetry`](@ref), which also names the `symbol` suffix it uses to say so.

# Notes
The reciprocal matrix is `inv(vectors)`; its **rows** `bᵢ` satisfy `aᵢ·bⱼ = δᵢⱼ`
(crystallographic convention, no 2π factor). The interplanar spacing along
direction `i` is `dᵢ = 1/‖bᵢ‖`; see [`interplanar_spacing`](@ref).
"""
struct Lattice
    vectors::SMatrix{3,3,Float64,9}
    reciprocal::SMatrix{3,3,Float64,9}
    pbc::SVector{3,Bool}

    # Inner constructor only: `reciprocal` is *derived* (`inv(vectors)`) and load-bearing
    # for `interplanar_spacing` / the neighbor-list image range, so it must never be
    # supplied independently. A near-singular cell makes `inv` — and everything downstream
    # — garbage, so reject by a relative-volume threshold, not just exact zero.
    function Lattice(vectors::AbstractMatrix{<:Real}; pbc = (true, true, true))
        size(vectors) == (3, 3) || throw(ArgumentError("lattice `vectors` must be 3×3"))
        A = SMatrix{3,3,Float64}(vectors)
        abs(det(A)) > eps(Float64) * norm(A)^3 ||
            throw(ArgumentError("lattice vectors are singular or numerically degenerate " *
                                "(|det| = $(abs(det(A))), ‖A‖ = $(norm(A)))"))
        return new(A, inv(A), SVector{3,Bool}(pbc...))
    end
end

"""
    interplanar_spacing(lattice, i) -> Float64

Interplanar spacing `dᵢ = 1/‖row_i(reciprocal)‖` (Å) along lattice direction `i`.
This sets the cutoff-driven image range in [`build_neighbor_list`](@ref).
"""
interplanar_spacing(lattice::Lattice, i::Integer)::Float64 =
    1.0 / norm(lattice.reciprocal[i, :])
