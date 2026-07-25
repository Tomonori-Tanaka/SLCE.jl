"""
    SolidHarmonics

Real solid harmonics `Rₗₘ(u)` for the displacement channel, evaluated as
homogeneous polynomials in the Cartesian components of `u` — never as the
`|u|^l · Zₗₘ(û)` composition, whose direction `û` is undefined at `u = 0`. In an
expansion about a reference structure `u ≈ 0` is the densest sampling region, so
the evaluator (and its Euclidean gradient, which force rows need) must be a plain
polynomial there: smooth, and exactly zero for `l ≥ 1`.

# Normalization (4π-free, Racah-type)

The displacement kernel deliberately does **not** carry the per-site `(4π)^(−1/2)`
of the spin-side tesseral kernel (`Harmonics.Zlm`). The convention is Racah-type:

    Rₗₘ(u) = √(4π/(2l+1)) · |u|^l · Zₗₘ(û)

so on the unit sphere `Rₗₘ(û) = √(4π/(2l+1)) · Zₗₘ(û)`, and the rank-1 factors
are literally the Cartesian components, `R₁₋₁, R₁₀, R₁₁ = y, z, x`. With this
choice only spin sites contribute to the per-term `(4π)^(n_spin/2)` design-matrix
scale; displacement factors are scale-free.

# Gradient

`solid_harmonics_grad` returns the plain Euclidean gradient `∂Rₗₘ/∂(x, y, z)` —
no tangent projection, in contrast to the on-sphere `Harmonics.grad_Zlm`. The two
evaluation kernels are distinct objects and neither substitutes for the other.

# Supported degree range

The `(2n − 1)!!` seed and the shrinking Racah norm are separate factors, so
their product degrades in floating point at extreme degrees: `R_{l,±l}`
silently underflows to `0.0` from `l ≈ 88` and hits `NaN` from `l ≈ 165` (the
same envelope as `Harmonics.Zlm`). Physical expansions live at `l ≲ 20`; treat
`l ≤ 80` as the validated range.
"""
module SolidHarmonics

using StaticArrays

# Not exported: callers reach these via `SolidHarmonics.Rlm` etc.

"""
    solid_harmonic_index(l, m) -> Int

1-based linear index of `(l, m)` in the contiguous ordering
`(0,0), (1,−1), (1,0), (1,1), (2,−2), …`: `l² + l + m + 1` (identical to the
spin-side `Harmonics.lm_index`).
"""
@inline solid_harmonic_index(l::Integer, m::Integer)::Int = l * l + l + m + 1

"""
    num_solid_harmonics(lmax) -> Int

Number of `(l, m)` pairs with `l ≤ lmax`, i.e. `(lmax + 1)²`.
"""
@inline num_solid_harmonics(lmax::Integer)::Int = (lmax + 1)^2

# (2n − 1)!! with the empty-product convention (−1)!! = 1.
@inline function _double_factorial_odd(n::Integer)::Float64
    acc = 1.0
    for k = 1:2:(2 * n - 1)
        acc *= k
    end
    return acc
end

# Racah-type tesseral normalization: 1 for m = 0, √(2·(l−m)!/(l+m)!) for m ≥ 1,
# formed without large factorials. Equals √(4π/(2l+1)) times the spin-side
# `Harmonics._plm_norm`-based factor.
@inline function _racah_norm(l::Int, m::Int)::Float64
    m == 0 && return 1.0
    acc = 2.0
    @inbounds for i = (l - m + 1):(l + m)
        acc /= i
    end
    return sqrt(acc)
end

@inline function _validate_lm(l::Integer, m::Integer)
    l >= 0 || throw(ArgumentError("need l ≥ 0 (got l=$l)"))
    -l <= m <= l || throw(ArgumentError("need -l ≤ m ≤ l (got l=$l, m=$m)"))
    return nothing
end

@inline function _validate_u(u::AbstractVector{<:Real})
    length(u) == 3 ||
        throw(ArgumentError("displacement must have length 3 (got $(length(u)))"))
    return nothing
end

# ---------------------------------------------------------------------------
# Core evaluator (values and Euclidean gradients)
# ---------------------------------------------------------------------------
#
# Factorized form (the sphere-constrained tesseral assembly with the constraint
# removed): for n = |m|,
#
#   R_{l,0}(u)  = K_{l0} A_l^0(z, r²)
#   R_{l,m}(u)  = K_{ln} A_l^n(z, r²) c_n(x, y)   (m > 0)
#   R_{l,-n}(u) = K_{ln} A_l^n(z, r²) s_n(x, y)   (m < 0)
#
# where c_n + i s_n = (x + i y)^n, K = _racah_norm, and
# A_l^n(z, r²) = r^{l−n} P_l^{(n)}(z/r) is a homogeneous polynomial of degree
# l − n in (x, y, z) (a polynomial in z and r² = x² + y² + z²). The
# associated-Legendre three-term recurrence homogenizes to
#
#   (l − n) A_l^n = (2l − 1) z A_{l−1}^n − (l + n − 1) r² A_{l−2}^n
#
# with seeds A_n^n = (2n−1)!! and A_{n+1}^n = (2n+1)!! z. Everything is a
# polynomial identity in (x, y, z): no division by |u| appears anywhere, so the
# evaluation is regular (and exact) at u = 0.

function _solid_harmonics_impl!(
    vals::AbstractVector{Float64},
    grads::Union{Nothing, AbstractMatrix{Float64}},
    lmax::Int,
    x::Float64,
    y::Float64,
    z::Float64,
)::Nothing
    r2 = x * x + y * y + z * z

    # c_n, s_n recurrence state: current (c, s) and previous (cm1, sm1).
    c = 1.0
    s = 0.0
    cm1 = 0.0
    sm1 = 0.0

    @inbounds for n = 0:lmax
        # A-recurrence state along l = n .. lmax at fixed n.
        # (A, Az, Ar) = (A_l^n, ∂A/∂z at fixed r², ∂A/∂(r²)).
        Aprev = 0.0
        Azprev = 0.0
        Arprev = 0.0
        A = _double_factorial_odd(n)
        Az = 0.0
        Ar = 0.0
        for l = n:lmax
            K = _racah_norm(l, n)
            if n == 0
                idx = solid_harmonic_index(l, 0)
                vals[idx] = K * A
                if grads !== nothing
                    grads[1, idx] = K * 2.0 * x * Ar
                    grads[2, idx] = K * 2.0 * y * Ar
                    grads[3, idx] = K * (Az + 2.0 * z * Ar)
                end
            else
                idxp = solid_harmonic_index(l, n)
                idxm = solid_harmonic_index(l, -n)
                vals[idxp] = K * A * c
                vals[idxm] = K * A * s
                if grads !== nothing
                    dAx = 2.0 * x * Ar
                    dAy = 2.0 * y * Ar
                    dAz = Az + 2.0 * z * Ar
                    grads[1, idxp] = K * (dAx * c + A * n * cm1)
                    grads[2, idxp] = K * (dAy * c - A * n * sm1)
                    grads[3, idxp] = K * dAz * c
                    grads[1, idxm] = K * (dAx * s + A * n * sm1)
                    grads[2, idxm] = K * (dAy * s + A * n * cm1)
                    grads[3, idxm] = K * dAz * s
                end
            end

            # Advance the A-recurrence to l + 1 (skip past lmax).
            l == lmax && break
            lnew = l + 1
            if lnew == n + 1
                Anew = (2.0 * n + 1.0) * z * A
                Aznew = (2.0 * n + 1.0) * A
                Arnew = 0.0
            else
                denom = lnew - n
                a1 = 2.0 * lnew - 1.0
                a2 = lnew + n - 1.0
                Anew = (a1 * z * A - a2 * r2 * Aprev) / denom
                Aznew = (a1 * (A + z * Az) - a2 * r2 * Azprev) / denom
                Arnew = (a1 * z * Ar - a2 * (Aprev + r2 * Arprev)) / denom
            end
            Aprev, Azprev, Arprev = A, Az, Ar
            A, Az, Ar = Anew, Aznew, Arnew
        end

        # Advance (c_n, s_n) -> (c_{n+1}, s_{n+1}).
        cm1, sm1 = c, s
        c, s = x * cm1 - y * sm1, x * sm1 + y * cm1
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

"""
    solid_harmonics!(vals, lmax, u) -> vals

In-place [`solid_harmonics`](@ref): fill `vals` (length ≥ `(lmax + 1)²`) with all
`Rₗₘ(u)` for `l = 0..lmax`, indexed by [`solid_harmonic_index`](@ref).
Allocation-free.
"""
function solid_harmonics!(
    vals::AbstractVector{Float64},
    lmax::Integer,
    u::AbstractVector{<:Real},
)::AbstractVector{Float64}
    lmax >= 0 || throw(ArgumentError("need lmax ≥ 0 (got $lmax)"))
    _validate_u(u)
    length(vals) >= num_solid_harmonics(lmax) || throw(ArgumentError(
        "vals too short: need $(num_solid_harmonics(lmax)), got $(length(vals))"))
    _solid_harmonics_impl!(vals, nothing, Int(lmax), Float64(u[1]), Float64(u[2]),
        Float64(u[3]))
    return vals
end

"""
    solid_harmonics(lmax, u) -> Vector{Float64}

Evaluate all real solid harmonics `Rₗₘ(u)` for `l = 0..lmax`, `m = −l..l`, as
homogeneous polynomials in the Cartesian components of `u` (regular at `u = 0`;
no normalization of `u` is required or performed). 4π-free Racah-type
normalization: on the unit sphere `Rₗₘ(û) = √(4π/(2l+1)) · Harmonics.Zlm(l, m, û)`.

# Arguments
- `lmax::Integer`: maximum degree (`lmax ≥ 0`).
- `u::AbstractVector{<:Real}`: displacement vector (length 3, arbitrary norm).

# Returns
- `Vector{Float64}` of length `(lmax + 1)²`, indexed by
  [`solid_harmonic_index`](@ref).
"""
function solid_harmonics(lmax::Integer, u::AbstractVector{<:Real})::Vector{Float64}
    lmax >= 0 || throw(ArgumentError("need lmax ≥ 0 (got $lmax)"))
    _validate_u(u)
    vals = Vector{Float64}(undef, num_solid_harmonics(lmax))
    _solid_harmonics_impl!(vals, nothing, Int(lmax), Float64(u[1]), Float64(u[2]),
        Float64(u[3]))
    return vals
end

"""
    solid_harmonics_grad!(vals, grads, lmax, u) -> (vals, grads)

In-place [`solid_harmonics_grad`](@ref): fill `vals` (length ≥ `(lmax + 1)²`)
and `grads` (size ≥ `3 × (lmax + 1)²`, rows = x, y, z). Allocation-free.
"""
function solid_harmonics_grad!(
    vals::AbstractVector{Float64},
    grads::AbstractMatrix{Float64},
    lmax::Integer,
    u::AbstractVector{<:Real},
)::Tuple{AbstractVector{Float64}, AbstractMatrix{Float64}}
    lmax >= 0 || throw(ArgumentError("need lmax ≥ 0 (got $lmax)"))
    _validate_u(u)
    n = num_solid_harmonics(lmax)
    length(vals) >= n ||
        throw(ArgumentError("vals too short: need $n, got $(length(vals))"))
    (size(grads, 1) == 3 && size(grads, 2) >= n) || throw(ArgumentError(
        "grads must be 3 × ≥$n (got $(size(grads)))"))
    _solid_harmonics_impl!(vals, grads, Int(lmax), Float64(u[1]), Float64(u[2]),
        Float64(u[3]))
    return (vals, grads)
end

"""
    solid_harmonics_grad(lmax, u) -> (values, gradients)

Evaluate all real solid harmonics and their Euclidean gradients
`∂Rₗₘ/∂(x, y, z)` (plain Cartesian derivatives, no tangent projection — this is
what force rows need). Both are polynomials, hence smooth at `u = 0`.

# Arguments
- `lmax::Integer`: maximum degree (`lmax ≥ 0`).
- `u::AbstractVector{<:Real}`: displacement vector (length 3, arbitrary norm).

# Returns
- `Tuple{Vector{Float64}, Matrix{Float64}}`: values (length `(lmax + 1)²`) and
  gradients (size `3 × (lmax + 1)²`, rows = x, y, z), both indexed by
  [`solid_harmonic_index`](@ref).
"""
function solid_harmonics_grad(
    lmax::Integer,
    u::AbstractVector{<:Real},
)::Tuple{Vector{Float64}, Matrix{Float64}}
    lmax >= 0 || throw(ArgumentError("need lmax ≥ 0 (got $lmax)"))
    _validate_u(u)
    n = num_solid_harmonics(lmax)
    vals = Vector{Float64}(undef, n)
    grads = Matrix{Float64}(undef, 3, n)
    _solid_harmonics_impl!(vals, grads, Int(lmax), Float64(u[1]), Float64(u[2]),
        Float64(u[3]))
    return (vals, grads)
end

"""
    Rlm(l, m, u) -> Float64

Real solid harmonic `Rₗₘ(u)` for a single `(l, m)`, evaluated as a homogeneous
polynomial in the Cartesian components of `u` (regular at `u = 0`, exactly zero
there for `l ≥ 1`). 4π-free Racah-type normalization: `R₁₋₁, R₁₀, R₁₁ = y, z, x`
exactly, and on the unit sphere `Rₗₘ(û) = √(4π/(2l+1)) · Harmonics.Zlm(l, m, û)`.

Convenience accessor: internally evaluates the full batch up to degree `l`
(`O(l²)` work and an allocation). Hot paths should use
[`solid_harmonics!`](@ref) and index with [`solid_harmonic_index`](@ref).

# Arguments
- `l::Integer`: degree (`l ≥ 0`).
- `m::Integer`: order (`−l ≤ m ≤ l`).
- `u::AbstractVector{<:Real}`: displacement vector (length 3, arbitrary norm).

# Returns
- `Float64`: the value `Rₗₘ(u)`.
"""
function Rlm(l::Integer, m::Integer, u::AbstractVector{<:Real})::Float64
    _validate_lm(l, m)
    return solid_harmonics(l, u)[solid_harmonic_index(l, m)]
end

"""
    grad_Rlm(l, m, u) -> SVector{3,Float64}

Euclidean gradient `∂Rₗₘ/∂(x, y, z)` of a single real solid harmonic (no tangent
projection — contrast `Harmonics.grad_Zlm`).

Convenience accessor: internally evaluates the full batch up to degree `l`
(`O(l²)` work and allocations). Hot paths should use
[`solid_harmonics_grad!`](@ref) and index with [`solid_harmonic_index`](@ref).

# Arguments
- `l::Integer`: degree (`l ≥ 0`).
- `m::Integer`: order (`−l ≤ m ≤ l`).
- `u::AbstractVector{<:Real}`: displacement vector (length 3, arbitrary norm).

# Returns
- `SVector{3,Float64}`: `∇Rₗₘ(u)`.
"""
function grad_Rlm(l::Integer, m::Integer, u::AbstractVector{<:Real})::SVector{3, Float64}
    _validate_lm(l, m)
    _, grads = solid_harmonics_grad(l, u)
    idx = solid_harmonic_index(l, m)
    return SVector{3, Float64}(grads[1, idx], grads[2, idx], grads[3, idx])
end

end # module SolidHarmonics
