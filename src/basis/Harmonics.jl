"""
    Harmonics

Real (tesseral) spherical harmonics `Zₗₘ` and their analytic, tangent-projected
Cartesian gradients, following Drautz, *Phys. Rev. B* **102**, 024104 (2020).

Per-site normalization carries `(4π)^(−1/2)`. The polar part uses the Drautz
normalized associated Legendre `P̄ₗₘ = (−1)^m √((2l+1)/(4π)·(l−m)!/(l+m)!)
dᵐPₗ/dzᵐ` (the `(1−z²)^{m/2}` factor is absorbed into the azimuthal `(x+iy)^m`),
so on the unit sphere these reduce to the standard real solid harmonics
(`Z₁₁ ∝ x`, `Z₂₂ ∝ x²−y²`, …).

The gradient is the genuine on-sphere (tangent) gradient
`∇Z = ∂Z − r̂ (r̂·∂Z)`; it is what the torque design matrix needs.
"""
module Harmonics

using LegendrePolynomials: dnPl
using LinearAlgebra: norm
using StaticArrays

# Not exported: callers reach these via `Harmonics.Zlm` etc. The `_unsafe`
# variants skip validation for hot paths.

# Tesseral normalization constants for the l ≤ 2 Cartesian conversions:
# `Z_{1,m} = N1·(y,z,x)_m` and the `Z_{2,m}` prefactors A2 (m = ±1, ±2) / B2 (m = 0).
# The single definition shared by the core's bilinear extraction (`slce/bilinear.jl`)
# and downstream consumers (the SLCETools.jl exchange mapping), so the forward and
# inverse tesseral ↔ Cartesian conversions cannot drift apart.
const N1 = sqrt(3 / (4π))
const A2 = sqrt(15 / (16π))
const B2 = sqrt(5 / (16π))

"""
    lm_index(l, m) -> Int

1-based linear index of `(l, m)` in the contiguous `Zₗₘ` ordering
`(0,0), (1,−1), (1,0), (1,1), (2,−2), …`. Inverse-friendly: `l² + l + m + 1`.
"""
@inline lm_index(l::Integer, m::Integer)::Int = l * l + l + m + 1

"""
    num_lm(lmax) -> Int

Number of `(l, m)` pairs with `l ≤ lmax`, i.e. `(lmax + 1)²`.
"""
@inline num_lm(lmax::Integer)::Int = (lmax + 1)^2

@inline _parity(n::Integer)::Int = isodd(n) ? -1 : 1

# √((2l+1)/(4π) · (l−m)!/(l+m)!) for m ≥ 0, formed without large factorials.
#
# UNDEFINED FOR m > l, and loudly so: the loop range then straddles zero, the accumulator is
# divided by 0, and the `Inf` meets `dnPl`'s exact 0 to give **NaN** rather than 0. That is
# deliberate — the `_unsafe` kernels are the hot path and carry no branch — but it means a
# caller that loops `m` over a table wider than `2l + 1` poisons a whole accumulation
# instead of adding nothing. Every in-tree and downstream caller uses `m = -l:l`; the
# checked entries (`Zlm` / `grad_Zlm`) throw. Do not "fix" this by returning 0.0: a silent
# zero would turn an indexing bug into a plausible number, which is the trade this package
# refuses everywhere else.
@inline function _plm_norm(l::Int, m::Int)::Float64
    acc = (2 * l + 1) / (4π)
    @inbounds for i = (l - m + 1):(l + m)
        acc /= i
    end
    return sqrt(acc)
end

# Drautz normalized associated Legendre P̄ₗₘ(z) and its z-derivative (m via |m|).
# The cache-threaded variants hand `cache` (length ≥ l + 1, contents irrelevant —
# the recursion overwrites every entry it reads) to LegendrePolynomials' `dnPl`,
# whose default argument otherwise allocates a fresh work vector on every call —
# the only allocation on these paths. Results are identical either way.
@inline _barP(l::Integer, m::Integer, z::Real)::Float64 =
    (n = abs(m); _parity(n) * _plm_norm(Int(l), n) * dnPl(z, l, n))
@inline _dbarP(l::Integer, m::Integer, z::Real)::Float64 =
    (n = abs(m); _parity(n) * _plm_norm(Int(l), n) * dnPl(z, l, n + 1))
@inline _barP(l::Integer, m::Integer, z::Real, cache::Vector{Float64})::Float64 =
    (n = abs(m); _parity(n) * _plm_norm(Int(l), n) * dnPl(z, l, n, cache))
@inline _dbarP(l::Integer, m::Integer, z::Real, cache::Vector{Float64})::Float64 =
    (n = abs(m); _parity(n) * _plm_norm(Int(l), n) * dnPl(z, l, n + 1, cache))

@inline function _validate_lm(l::Integer, m::Integer)
    l >= 0 || throw(ArgumentError("need l ≥ 0 (got l=$l)"))
    -l <= m <= l || throw(ArgumentError("need -l ≤ m ≤ l (got l=$l, m=$m)"))
    return nothing
end

@inline function _validate_unit(u::AbstractVector{<:Real}; atol::Real = 1e-8)
    length(u) == 3 || throw(ArgumentError("direction must have length 3 (got $(length(u)))"))
    abs(norm(u) - 1) <= atol ||
        throw(ArgumentError("direction must be a unit vector (‖u‖ = $(norm(u)))"))
    return nothing
end

"""
    Zlm_unsafe(l, m, u[, cache]) -> Float64

Tesseral harmonic `Zₗₘ(u)` for a unit vector `u`, without input validation.
Passing `cache` (a `Vector{Float64}` of length ≥ `l + 1`, contents irrelevant)
reuses it as the associated-Legendre recursion workspace, making the call
allocation-free; the returned value is identical with or without it.

!!! warning "Preconditions: `l ≥ 0`, `|m| ≤ l`, `‖u‖ = 1` — outside them the value is junk"
    "Without input validation" means the preconditions are the caller's, not that the
    result is defined without them. In particular `|m| > l` returns **`NaN`** (the
    normalization divides by zero), so a caller looping `m` over a table wider than
    `2l + 1` poisons its whole accumulation rather than adding nothing. Use the checked
    [`Zlm`](@ref) when the indices are not yours to guarantee.
"""
@inline function Zlm_unsafe(l::Integer, m::Integer, u::AbstractVector{<:Real},
                            cache::Vector{Float64})::Float64
    n = abs(m)
    plm = _barP(l, n, u[3], cache)
    n == 0 && return plm
    c = _parity(n) * sqrt(2.0) * plm
    zpow = ComplexF64(u[1], u[2])^n
    return m > 0 ? c * real(zpow) : c * imag(zpow)
end

@inline function Zlm_unsafe(l::Integer, m::Integer, u::AbstractVector{<:Real})::Float64
    n = abs(m)
    plm = _barP(l, n, u[3])
    n == 0 && return plm
    c = _parity(n) * sqrt(2.0) * plm
    zpow = ComplexF64(u[1], u[2])^n
    return m > 0 ? c * real(zpow) : c * imag(zpow)
end

"""
    Zlm(l, m, u) -> Float64

Tesseral harmonic `Zₗₘ` at the unit direction `u` (3-vector).

# Arguments
- `l::Integer`: angular momentum (`l ≥ 0`).
- `m::Integer`: order (`-l ≤ m ≤ l`).
- `u::AbstractVector{<:Real}`: unit direction `[x, y, z]`.

# Returns
- `Float64`: `Zₗₘ(u)`.
"""
function Zlm(l::Integer, m::Integer, u::AbstractVector{<:Real})::Float64
    _validate_lm(l, m)
    _validate_unit(u)
    return Zlm_unsafe(l, m, u)
end

"""
    grad_Zlm_unsafe(l, m, u[, cache]) -> SVector{3,Float64}

Tangent-projected Cartesian gradient `∇Zₗₘ(u) = ∂Z − u (u·∂Z)`, without
validation. The optional `cache` is the same allocation-free recursion workspace
as in [`Zlm_unsafe`](@ref) (length ≥ `l + 1`; the value is identical either way).
"""
@inline function grad_Zlm_unsafe(l::Integer, m::Integer, u::AbstractVector{<:Real},
                                 cache::Vector{Float64})::SVector{3,Float64}
    x, y, z = u[1], u[2], u[3]
    n = abs(m)
    plm = _barP(l, n, z, cache)
    dplm = _dbarP(l, n, z, cache)
    return _grad_zlm_assemble(l, m, n, x, y, z, plm, dplm)
end

@inline function grad_Zlm_unsafe(l::Integer, m::Integer,
                                 u::AbstractVector{<:Real})::SVector{3,Float64}
    x, y, z = u[1], u[2], u[3]
    n = abs(m)
    plm = _barP(l, n, z)
    dplm = _dbarP(l, n, z)
    return _grad_zlm_assemble(l, m, n, x, y, z, plm, dplm)
end

# Shared tail of the two grad_Zlm_unsafe methods (identical arithmetic by
# construction — the cache only affects where the recursion scratch lives).
@inline function _grad_zlm_assemble(l::Integer, m::Integer, n::Integer, x::Real,
                                    y::Real, z::Real, plm::Float64,
                                    dplm::Float64)::SVector{3,Float64}
    if m == 0
        zz = z * dplm
        return SVector{3,Float64}(-x * zz, -y * zz, dplm - z * zz)
    end
    c = _parity(n) * sqrt(2.0)
    zxy = ComplexF64(x, y)
    zpn = zxy^n
    zpn1 = zxy^(n - 1)
    rn, iN = real(zpn), imag(zpn)
    rn1, in1 = real(zpn1), imag(zpn1)
    dZx, dZy, dZz = if m > 0
        (c * n * plm * rn1, -c * n * plm * in1, c * dplm * rn)
    else
        (c * n * plm * in1, c * n * plm * rn1, c * dplm * iN)
    end
    zz = x * dZx + y * dZy + z * dZz   # u · ∂Z, the radial part to remove
    return SVector{3,Float64}(dZx - x * zz, dZy - y * zz, dZz - z * zz)
end

"""
    grad_Zlm(l, m, u) -> SVector{3,Float64}

On-sphere (tangent-projected) Cartesian gradient `∇Zₗₘ(u)`.

# Arguments
- `l::Integer`: angular momentum (`l ≥ 0`).
- `m::Integer`: order (`-l ≤ m ≤ l`).
- `u::AbstractVector{<:Real}`: unit direction `[x, y, z]`.

# Returns
- `SVector{3,Float64}`: `(∂ₓ, ∂_y, ∂_z) Zₗₘ` with the radial component removed, so
  `u·∇Zₗₘ = 0`.
"""
function grad_Zlm(l::Integer, m::Integer, u::AbstractVector{<:Real})::SVector{3,Float64}
    _validate_lm(l, m)
    _validate_unit(u)
    return grad_Zlm_unsafe(l, m, u)
end

end # module Harmonics
