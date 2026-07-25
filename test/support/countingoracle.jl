"""
    CountingOracle

Independent test oracle for decorated-cluster invariant counting and Reynolds
projection, mixing axial spin slots and polar displacement slots. Ported from the
exploratory DisplacementBases prototype; the production basis engine must agree
with this module on invariant counts and projector ranks (the "count ≡ projector
rank" gate), which is the primary internal consistency check of the joint
construction.

Slot representation matrices are derived by explicit polynomial composition, so
axial/polar parity and improper operations are handled with no hand-written
determinant factors — an implementation path deliberately different from the
production projector.

# Scope restrictions (read before comparing against production)

- **No time reversal.** On Σl_spin-odd inputs this oracle happily returns T-odd
  invariants (e.g. `(ê × u)_z`). Oracle-vs-production count comparisons are valid
  ONLY on Σl_spin-even inputs; production never enumerates the odd sector.
- **No supercell translation folding** — clusters are finite point sets.
- **No strain (global, site-less) slot** and **no spin-side symmetric powers**.
- Normalization: spin slots use orthonormal tesseral polynomials, displacement
  slots the 4π-free Racah solid harmonics. Counting and ranks are normalization
  independent; do not compare raw projector matrix entries across kernels.
"""
module CountingOracle

using LinearAlgebra
using StaticArrays

export SpinSlot, DispSlot, DispSymSlot, SpinPairSlot, ClusterOp
export cluster_op, spin_only_op, identity_op, product_op, product_ops, stabilizer_ops
export count_invariants, representation_matrix, invariant_projector, invariant_basis
export assert_group_closure
export HomogeneousPolynomial, solid_harmonic_polynomial, harmonic_blocks, sym_dimension

# ---------------------------------------------------------------------------
# Homogeneous polynomials on R^3 (explicit coefficient representation)
# ---------------------------------------------------------------------------

"""
    HomogeneousPolynomial

A homogeneous polynomial of fixed degree on ℝ³, stored as a dense coefficient
vector over the monomials `x^a y^b z^c` with `a + b + c = degree`, in the
deterministic order of `_monomial_exponents(degree)`. Callable: `q(u)`.
"""
struct HomogeneousPolynomial
    degree::Int
    coeffs::Vector{Float64}

    function HomogeneousPolynomial(degree::Int, coeffs::Vector{Float64})
        degree >= 0 || throw(ArgumentError("degree must be non-negative"))
        nmono = div((degree + 1) * (degree + 2), 2)
        length(coeffs) == nmono || throw(ArgumentError(
            "expected $nmono coefficients for degree $degree, got $(length(coeffs))"))
        return new(degree, coeffs)
    end
end

# Deterministic monomial order for degree p: a from p down to 0, then b from
# p - a down to 0, with c = p - a - b.
function _monomial_exponents(p::Int)::Vector{NTuple{3, Int}}
    exps = NTuple{3, Int}[]
    for a = p:-1:0, b = (p - a):-1:0
        push!(exps, (a, b, p - a - b))
    end
    return exps
end

_monomial_index_map(p::Int)::Dict{NTuple{3, Int}, Int} =
    Dict(e => i for (i, e) in enumerate(_monomial_exponents(p)))

function (q::HomogeneousPolynomial)(u::AbstractVector{<:Real})::Float64
    x, y, z = Float64(u[1]), Float64(u[2]), Float64(u[3])
    acc = 0.0
    for (i, (a, b, c)) in enumerate(_monomial_exponents(q.degree))
        coef = q.coeffs[i]
        coef == 0.0 && continue
        acc += coef * x^a * y^b * z^c
    end
    return acc
end

function Base.:+(q1::HomogeneousPolynomial, q2::HomogeneousPolynomial)
    q1.degree == q2.degree ||
        throw(ArgumentError("cannot add homogeneous polynomials of different degree"))
    return HomogeneousPolynomial(q1.degree, q1.coeffs .+ q2.coeffs)
end

function Base.:-(q1::HomogeneousPolynomial, q2::HomogeneousPolynomial)
    q1.degree == q2.degree ||
        throw(ArgumentError("cannot subtract homogeneous polynomials of different degree"))
    return HomogeneousPolynomial(q1.degree, q1.coeffs .- q2.coeffs)
end

Base.:*(a::Real, q::HomogeneousPolynomial) =
    HomogeneousPolynomial(q.degree, Float64(a) .* q.coeffs)
Base.:*(q::HomogeneousPolynomial, a::Real) = a * q

function Base.:*(q1::HomogeneousPolynomial, q2::HomogeneousPolynomial)
    p = q1.degree + q2.degree
    idx = _monomial_index_map(p)
    coeffs = zeros(Float64, div((p + 1) * (p + 2), 2))
    e1 = _monomial_exponents(q1.degree)
    e2 = _monomial_exponents(q2.degree)
    for (i, (a1, b1, c1)) in enumerate(e1)
        v1 = q1.coeffs[i]
        v1 == 0.0 && continue
        for (j, (a2, b2, c2)) in enumerate(e2)
            v2 = q2.coeffs[j]
            v2 == 0.0 && continue
            coeffs[idx[(a1 + a2, b1 + b2, c1 + c2)]] += v1 * v2
        end
    end
    return HomogeneousPolynomial(p, coeffs)
end

function Base.:^(q::HomogeneousPolynomial, k::Integer)
    k >= 0 || throw(ArgumentError("negative power of a polynomial"))
    acc = _one_poly()
    for _ = 1:k
        acc = acc * q
    end
    return acc
end

_one_poly() = HomogeneousPolynomial(0, [1.0])
_x_poly() = HomogeneousPolynomial(1, [1.0, 0.0, 0.0])
_y_poly() = HomogeneousPolynomial(1, [0.0, 1.0, 0.0])
_z_poly() = HomogeneousPolynomial(1, [0.0, 0.0, 1.0])
# x^2 + y^2 + z^2 in the degree-2 monomial order
# (2,0,0),(1,1,0),(1,0,1),(0,2,0),(0,1,1),(0,0,2).
_r2_poly() = HomogeneousPolynomial(2, [1.0, 0.0, 0.0, 1.0, 0.0, 1.0])

# q(A u): substitute the linear map, returning a polynomial of the same degree.
function _compose_linear(
    q::HomogeneousPolynomial,
    A::AbstractMatrix{<:Real},
)::HomogeneousPolynomial
    size(A) == (3, 3) || throw(ArgumentError("A must be 3x3"))
    lin = ntuple(i -> HomogeneousPolynomial(1, [Float64(A[i, 1]), Float64(A[i, 2]),
        Float64(A[i, 3])]), 3)
    p = q.degree
    nmono = div((p + 1) * (p + 2), 2)
    acc = HomogeneousPolynomial(p, zeros(Float64, nmono))
    for (i, (a, b, c)) in enumerate(_monomial_exponents(p))
        v = q.coeffs[i]
        v == 0.0 && continue
        acc = acc + v * (lin[1]^a * lin[2]^b * lin[3]^c)
    end
    return acc
end

# ---------------------------------------------------------------------------
# Solid harmonics as explicit polynomials; Sym^p harmonic-block decomposition
# ---------------------------------------------------------------------------

# (2n - 1)!! with the empty-product convention (-1)!! = 1.
@inline function _double_factorial_odd(n::Integer)::Float64
    acc = 1.0
    for k = 1:2:(2 * n - 1)
        acc *= k
    end
    return acc
end

# 4π-free Racah-type tesseral normalization (matches SLCE.SolidHarmonics).
@inline function _racah_norm(l::Int, m::Int)::Float64
    m == 0 && return 1.0
    acc = 2.0
    for i = (l - m + 1):(l + m)
        acc /= i
    end
    return sqrt(acc)
end

"""
    solid_harmonic_polynomial(l, m) -> HomogeneousPolynomial

The real solid harmonic `Rₗₘ` as an explicit homogeneous polynomial of degree
`l`, in the 4π-free Racah normalization of `SLCE.SolidHarmonics` (so
`R₁₋₁, R₁₀, R₁₁ = y, z, x` exactly).
"""
function solid_harmonic_polynomial(l::Integer, m::Integer)::HomogeneousPolynomial
    abs(m) <= l || throw(ArgumentError("require -l <= m <= l, got (l, m) = ($l, $m)"))
    l = Int(l)
    n = abs(Int(m))
    zp = _z_poly()
    r2 = _r2_poly()

    # A_l^n(z, r²) via the homogenized associated-Legendre recurrence.
    Aprev = _one_poly()  # placeholder; unused until two seeds exist
    A = _double_factorial_odd(n) * _one_poly()
    if l > n
        Aprev = A
        A = (2.0 * n + 1.0) * (zp * Aprev)
        for li = (n + 2):l
            Anew = (1.0 / (li - n)) *
                   ((2.0 * li - 1.0) * (zp * A) - Float64(li + n - 1) * (r2 * Aprev))
            Aprev = A
            A = Anew
        end
    end

    # Angular factor c_n / s_n = Re / Im of (x + i y)^n.
    n == 0 && return _racah_norm(l, 0) * A
    xp, yp = _x_poly(), _y_poly()
    cpoly = _one_poly()
    spoly = HomogeneousPolynomial(0, [0.0])
    for _ = 1:n
        cnew = xp * cpoly - yp * spoly
        snew = xp * spoly + yp * cpoly
        cpoly, spoly = cnew, snew
    end
    return _racah_norm(l, n) * (A * (m > 0 ? cpoly : spoly))
end

"""
    sym_dimension(p) -> Int

Dimension of `Sym^p(ℝ³)`: `(p + 1)(p + 2) / 2`.
"""
sym_dimension(p::Integer)::Int = div((p + 1) * (p + 2), 2)

"""
    harmonic_blocks(p)
        -> Vector{@NamedTuple{k::Int, L::Int, polys::Vector{HomogeneousPolynomial}}}

Harmonic-block decomposition `Sym^p(ℝ³) ≅ ⊕_k |u|^{2k} · {rank-(p−2k) solid
harmonics}`. Block `k` carries `L = p − 2k` and the `2L + 1` basis polynomials
`|u|^{2k} R_{L,M}(u)`. The `L = 0` trace channels are the radial factors and are
never pruned. Dimension identity: `Σ_k (2(p−2k) + 1) = (p+1)(p+2)/2`.
"""
function harmonic_blocks(
    p::Integer,
)::Vector{@NamedTuple{k::Int, L::Int, polys::Vector{HomogeneousPolynomial}}}
    p >= 0 || throw(ArgumentError("p must be non-negative, got $p"))
    p = Int(p)
    r2 = _r2_poly()
    blocks = @NamedTuple{k::Int, L::Int, polys::Vector{HomogeneousPolynomial}}[]
    for k = 0:div(p, 2)
        L = p - 2 * k
        radial = r2^k
        polys = [radial * solid_harmonic_polynomial(L, M) for M = -L:L]
        push!(blocks, (; k, L, polys))
    end
    return blocks
end

# ---------------------------------------------------------------------------
# Decorated-cluster slots (constructor-validated) and symmetry operations
# ---------------------------------------------------------------------------

"""
    AbstractSlot

Supertype of the tensor-factor ("slot") specifications of a decorated cluster.
A decorated-cluster basis function is a product of one factor per slot.
"""
abstract type AbstractSlot end

"""
    SpinSlot(site::Int, l::Int)

A spin factor `Z_{l m}(ê_site)` of rank `l ≥ 1` at cluster site `site`. Spin
directions are axial: symmetry operations act through the proper part
`det(R) R` of the lattice rotation.
"""
struct SpinSlot <: AbstractSlot
    site::Int
    l::Int

    function SpinSlot(site::Int, l::Int)
        site >= 1 || throw(ArgumentError("site must be ≥ 1, got $site"))
        l >= 1 || throw(ArgumentError("spin rank must be ≥ 1, got l = $l"))
        return new(site, l)
    end
end

"""
    DispSlot(site::Int, k::Int, L::Int)

A displacement factor `|u_site|^{2k} R_{L M}(u_site)` — one harmonic block of
`Sym^{2k + L}` at site `site`, with `2k + L ≥ 1`. Displacements are polar:
improper operations act with the full (negative-determinant) matrix.
`k ≥ 1, L = 0` blocks are the radial trace channels.
"""
struct DispSlot <: AbstractSlot
    site::Int
    k::Int
    L::Int

    function DispSlot(site::Int, k::Int, L::Int)
        site >= 1 || throw(ArgumentError("site must be ≥ 1, got $site"))
        k >= 0 || throw(ArgumentError("radial label must be ≥ 0, got k = $k"))
        L >= 0 || throw(ArgumentError("rank must be ≥ 0, got L = $L"))
        2 * k + L >= 1 ||
            throw(ArgumentError("degree 2k + L must be ≥ 1, got k = $k, L = $L"))
        return new(site, k, L)
    end
end

"""
    DispSymSlot(site::Int, p::Int)

The full symmetrized displacement power `Sym^p(u_site)` at site `site` (all
harmonic blocks of degree `p ≥ 1` together, dimension `(p+1)(p+2)/2`). Character
contributions use symmetrized-power (plethysm) characters, not `χ(R)^p`.
"""
struct DispSymSlot <: AbstractSlot
    site::Int
    p::Int

    function DispSymSlot(site::Int, p::Int)
        site >= 1 || throw(ArgumentError("site must be ≥ 1, got $site"))
        p >= 1 || throw(ArgumentError("degree must be ≥ 1, got p = $p"))
        return new(site, p)
    end
end

"""
    SpinPairSlot(site_a::Int, site_b::Int, l_a::Int, l_b::Int, L_S::Int)

Two spin factors of ranks `l_a` (at `site_a`) and `l_b` (at `site_b`) coupled to
total spin rank `L_S` (triangle inequality enforced; `site_a ≠ site_b`).
Transforms with the rank-`L_S` rotation of the (proper) spin rotation; mapping
onto a pair slot with the opposite site orientation multiplies by the
Clebsch--Gordan exchange sign `(−1)^{l_a + l_b − L_S}`. Use `L_S = 0` to enforce
the no-SOC scalar restriction on a pair cluster.
"""
struct SpinPairSlot <: AbstractSlot
    site_a::Int
    site_b::Int
    l_a::Int
    l_b::Int
    L_S::Int

    function SpinPairSlot(site_a::Int, site_b::Int, l_a::Int, l_b::Int, L_S::Int)
        site_a >= 1 && site_b >= 1 ||
            throw(ArgumentError("sites must be ≥ 1, got ($site_a, $site_b)"))
        site_a != site_b || throw(ArgumentError(
            "SpinPairSlot requires two distinct sites, got site $site_a twice"))
        l_a >= 1 && l_b >= 1 ||
            throw(ArgumentError("spin ranks must be ≥ 1, got ($l_a, $l_b)"))
        abs(l_a - l_b) <= L_S <= l_a + l_b || throw(ArgumentError(
            "L_S = $L_S violates the triangle rule for (l_a, l_b) = ($l_a, $l_b)"))
        return new(site_a, site_b, l_a, l_b, L_S)
    end
end

_slot_dim(s::SpinSlot)::Int = 2 * s.l + 1
_slot_dim(s::DispSlot)::Int = 2 * s.L + 1
_slot_dim(s::DispSymSlot)::Int = sym_dimension(s.p)
_slot_dim(s::SpinPairSlot)::Int = 2 * s.L_S + 1

_exchange_sign(s::SpinPairSlot)::Float64 = Float64((-1)^(s.l_a + s.l_b - s.L_S))

# (site, channel) occupancy used by the repeated-slot rejection. Slots ≠ sites:
# a site may carry one spin factor AND one displacement factor, but never two
# factors of the same channel.
_occupied(s::SpinSlot) = [(s.site, :spin)]
_occupied(s::DispSlot) = [(s.site, :disp)]
_occupied(s::DispSymSlot) = [(s.site, :disp)]
_occupied(s::SpinPairSlot) = [(s.site_a, :spin), (s.site_b, :spin)]

function _validate_slots(slots::Vector{<:AbstractSlot})
    isempty(slots) && throw(ArgumentError("empty slot list"))
    seen = Set{Tuple{Int, Symbol}}()
    for s in slots, occ in _occupied(s)
        occ in seen && throw(ArgumentError(
            "repeated $(occ[2]) factor at site $(occ[1]): at most one factor per " *
            "(site, channel) pair"))
        push!(seen, occ)
    end
    return nothing
end

"""
    ClusterOp

A symmetry operation acting on a decorated cluster: a lattice (polar) rotation
`rotation` (possibly improper), a proper spin rotation `spin_rotation`, and a
site permutation `perm` (site `i` maps to `perm[i]`). For crystal operations
`spin_rotation = det(R) R` (axial action); pure spin rotations (no-SOC
averaging) carry `rotation = I` and `perm = identity`.
"""
struct ClusterOp
    rotation::SMatrix{3, 3, Float64, 9}
    spin_rotation::SMatrix{3, 3, Float64, 9}
    perm::Vector{Int}

    function ClusterOp(
        rotation::AbstractMatrix{<:Real},
        spin_rotation::AbstractMatrix{<:Real},
        perm::AbstractVector{<:Integer},
    )
        R = SMatrix{3, 3, Float64, 9}(rotation)
        S = SMatrix{3, 3, Float64, 9}(spin_rotation)
        norm(R' * R - I) < 1e-8 || throw(ArgumentError("rotation is not orthogonal"))
        norm(S' * S - I) < 1e-8 ||
            throw(ArgumentError("spin_rotation is not orthogonal"))
        det(S) > 0 || throw(ArgumentError("spin_rotation must be proper"))
        isperm(perm) || throw(ArgumentError("perm is not a permutation"))
        return new(R, S, collect(Int, perm))
    end
end

"""
    cluster_op(R, sites; tol = 1e-8) -> ClusterOp

Build a [`ClusterOp`](@ref) from a (possibly improper) orthogonal matrix `R`
acting on the cluster-local Cartesian site positions `sites`. The site
permutation is found by matching `R * sites[i]` to a site position; the spin
rotation is the proper part `det(R) R`.
"""
function cluster_op(
    R::AbstractMatrix{<:Real},
    sites::AbstractVector{<:AbstractVector{<:Real}};
    tol::Real = 1e-8,
)::ClusterOp
    nsites = length(sites)
    perm = zeros(Int, nsites)
    for i = 1:nsites
        target = R * sites[i]
        j = findfirst(s -> norm(target - s) < tol, sites)
        j === nothing &&
            throw(ArgumentError("operation does not map site $i onto the site list"))
        perm[i] = j
    end
    return ClusterOp(R, det(SMatrix{3, 3, Float64, 9}(R)) * R, perm)
end

"""
    spin_only_op(R_spin, nsites) -> ClusterOp

A pure spin rotation (identity on the lattice and on site labels). Averaging over
a suitable finite subgroup of SO(3) in the spin sector enforces the no-SOC
restriction (kills all spin channels with `1 ≤ L_S ≤ 3` for the octahedral
rotation group).
"""
spin_only_op(R_spin::AbstractMatrix{<:Real}, nsites::Integer)::ClusterOp =
    ClusterOp(SMatrix{3, 3, Float64, 9}(I), R_spin, collect(1:Int(nsites)))

"""
    identity_op(nsites) -> ClusterOp

The identity operation on a cluster with `nsites` sites.
"""
function identity_op(nsites::Integer)::ClusterOp
    eye = SMatrix{3, 3, Float64, 9}(I)
    return ClusterOp(eye, eye, collect(1:Int(nsites)))
end

"""
    product_op(g1, g2) -> ClusterOp

Group product `g1 ∘ g2` (apply `g2` first).
"""
function product_op(g1::ClusterOp, g2::ClusterOp)::ClusterOp
    perm = [g1.perm[g2.perm[i]] for i = 1:length(g2.perm)]
    return ClusterOp(g1.rotation * g2.rotation, g1.spin_rotation * g2.spin_rotation, perm)
end

"""
    product_ops(ops1, ops2) -> Vector{ClusterOp}

All pairwise products `g1 ∘ g2`. The result is a group only when both input
lists are groups and one normalizes the other — e.g. lattice operations × pure
spin rotations, where the spin subgroup must be normalized by the proper parts
of the lattice rotations (a spin subgroup that is merely a group on its own does
NOT suffice). Verify with [`assert_group_closure`](@ref) before feeding the
result to [`count_invariants`](@ref).
"""
product_ops(ops1::Vector{ClusterOp}, ops2::Vector{ClusterOp})::Vector{ClusterOp} =
    [product_op(g1, g2) for g1 in ops1 for g2 in ops2]

# Two ops equal within tolerance (rotations and permutation).
function _op_isapprox(a::ClusterOp, b::ClusterOp; tol::Real = 1e-8)::Bool
    return a.perm == b.perm && norm(a.rotation - b.rotation) < tol &&
           norm(a.spin_rotation - b.spin_rotation) < tol
end

"""
    assert_group_closure(ops; tol = 1e-8)

Verify that `ops` contains the identity, is closed under composition, and has no
duplicate elements; throw an `ArgumentError` otherwise. O(|G|³) matching — meant
for oracle-sized groups only.
"""
function assert_group_closure(ops::Vector{ClusterOp}; tol::Real = 1e-8)
    isempty(ops) && throw(ArgumentError("empty operation list"))
    n = length(ops)
    for i = 1:n, j = (i + 1):n
        _op_isapprox(ops[i], ops[j]; tol) &&
            throw(ArgumentError("duplicate operations at positions $i and $j"))
    end
    nsites = length(ops[1].perm)
    any(op -> _op_isapprox(op, identity_op(nsites); tol), ops) ||
        throw(ArgumentError("operation list does not contain the identity"))
    for g1 in ops, g2 in ops
        prod = product_op(g1, g2)
        any(op -> _op_isapprox(prod, op; tol), ops) ||
            throw(ArgumentError("operation list is not closed under composition"))
    end
    return nothing
end

# ---------------------------------------------------------------------------
# Slot representation matrices (derived from the polynomial realization)
# ---------------------------------------------------------------------------

_slot_polys(s::SpinSlot)::Vector{HomogeneousPolynomial} =
    [solid_harmonic_polynomial(s.l, m) for m = (-s.l):(s.l)]
function _slot_polys(s::DispSlot)::Vector{HomogeneousPolynomial}
    radial = _r2_poly()^s.k
    return [radial * solid_harmonic_polynomial(s.L, M) for M = (-s.L):(s.L)]
end
_slot_polys(s::DispSymSlot)::Vector{HomogeneousPolynomial} =
    reduce(vcat, [b.polys for b in harmonic_blocks(s.p)])
_slot_polys(s::SpinPairSlot)::Vector{HomogeneousPolynomial} =
    [solid_harmonic_polynomial(s.L_S, m) for m = (-s.L_S):(s.L_S)]

_slot_rotation(::SpinSlot, op::ClusterOp) = op.spin_rotation
_slot_rotation(::DispSlot, op::ClusterOp) = op.rotation
_slot_rotation(::DispSymSlot, op::ClusterOp) = op.rotation
_slot_rotation(::SpinPairSlot, op::ClusterOp) = op.spin_rotation

# Representation matrix of one slot under `op`, in the slot's polynomial basis
# {q_m}: q_m(R⁻¹ u) = Σ_{m'} M[m', m] q_{m'}(u). Derived by explicit polynomial
# composition, so axial/polar parity and improper operations are handled
# automatically (no manual det factors). Orientation-dependent exchange signs of
# SpinPairSlot mappings are NOT applied here — they belong to the slot mapping,
# not the slot content, and are attached where the slot permutation is known
# (representation_matrix / the cycle characters).
function _slot_matrix(slot::AbstractSlot, op::ClusterOp)::Matrix{Float64}
    polys = _slot_polys(slot)
    R = _slot_rotation(slot, op)
    Rinv = transpose(R)  # orthogonal
    B = reduce(hcat, [q.coeffs for q in polys])
    C = reduce(hcat, [_compose_linear(q, Rinv).coeffs for q in polys])
    M = B \ C
    resid = norm(B * M - C)
    resid < 1e-8 * max(1.0, norm(C)) ||
        error("slot space is not closed under the operation (residual $resid)")
    return M
end

# Slot spec after acting with `op` on the site labels (non-site fields kept).
_slot_image(s::SpinSlot, op::ClusterOp) = SpinSlot(op.perm[s.site], s.l)
_slot_image(s::DispSlot, op::ClusterOp) = DispSlot(op.perm[s.site], s.k, s.L)
_slot_image(s::DispSymSlot, op::ClusterOp) = DispSymSlot(op.perm[s.site], s.p)
_slot_image(s::SpinPairSlot, op::ClusterOp) =
    SpinPairSlot(op.perm[s.site_a], op.perm[s.site_b], s.l_a, s.l_b, s.L_S)

# Match result: 0 = no match, 1 = direct, 2 = matched with swapped orientation.
_slot_match_kind(target::AbstractSlot, cand::AbstractSlot)::Int =
    target == cand ? 1 : 0
function _slot_match_kind(target::SpinPairSlot, cand::SpinPairSlot)::Int
    target == cand && return 1
    if target.site_a == cand.site_b && target.site_b == cand.site_a &&
       target.l_a == cand.l_b && target.l_b == cand.l_a && target.L_S == cand.L_S
        return 2
    end
    return 0
end

# Permutation τ on slot indices induced by `op` (τ[j] = index of the slot that
# slot j maps onto) together with the per-slot swap flags (true when the image
# matched its target with swapped pair orientation), or `nothing` when the
# decoration is not preserved.
function _slot_permutation(
    slots::Vector{<:AbstractSlot},
    op::ClusterOp,
)::Union{Nothing, Tuple{Vector{Int}, Vector{Bool}}}
    n = length(slots)
    τ = zeros(Int, n)
    swap = falses(n)
    used = falses(n)
    for j = 1:n
        target = _slot_image(slots[j], op)
        found = 0
        kind = 0
        for jp = 1:n
            used[jp] && continue
            kind = _slot_match_kind(target, slots[jp])
            if kind != 0
                found = jp
                break
            end
        end
        found == 0 && return nothing
        τ[j] = found
        swap[j] = kind == 2
        used[found] = true
    end
    return (τ, swap)
end

"""
    stabilizer_ops(slots, ops) -> Vector{ClusterOp}

Filter `ops` down to the operations that map the decorated cluster onto itself
(possibly permuting its slots). By Frobenius reciprocity, counting invariants
over this stabilizer equals counting over the full orbit space of the
decoration.
"""
function stabilizer_ops(
    slots::Vector{<:AbstractSlot},
    ops::Vector{ClusterOp},
)::Vector{ClusterOp}
    _validate_slots(slots)
    return [op for op in ops if _slot_permutation(slots, op) !== nothing]
end

"""
    representation_matrix(slots, op) -> Matrix{Float64}

The full representation matrix `D(op)` on the tensor product of the slot spaces,
including the slot permutation induced by the site permutation of `op` and the
Clebsch--Gordan exchange sign of every pair-slot mapping with swapped
orientation. Basis index order: slot 1 fastest (column-major `LinearIndices`
over the slot dimensions).
"""
function representation_matrix(
    slots::Vector{<:AbstractSlot},
    op::ClusterOp,
)::Matrix{Float64}
    _validate_slots(slots)
    στ = _slot_permutation(slots, op)
    στ === nothing && throw(ArgumentError("operation does not preserve the decoration"))
    τ, swap = στ
    nslots = length(slots)
    mats = Matrix{Float64}[_slot_matrix(slots[j], op) for j = 1:nslots]
    for j = 1:nslots
        swap[j] || continue
        mats[j] = _exchange_sign(slots[j]::SpinPairSlot) .* mats[j]
    end
    dims = Tuple(_slot_dim(s) for s in slots)
    n = prod(dims)
    D = zeros(Float64, n, n)
    ci = CartesianIndices(dims)
    li = LinearIndices(dims)
    for src in ci
        col = li[src]
        for dst in ci
            val = 1.0
            for j = 1:nslots
                val *= mats[j][dst[τ[j]], src[j]]
                val == 0.0 && break
            end
            val == 0.0 && continue
            D[li[dst], col] = val
        end
    end
    return D
end

"""
    invariant_projector(slots, ops) -> Matrix{Float64}

Reynolds average `P = (1/|G|) Σ_g D(g)` over `ops` (which must form a group
containing the identity; verify with [`assert_group_closure`](@ref)). `P`
projects onto the invariant subspace; its rank must agree with
[`count_invariants`](@ref).
"""
function invariant_projector(
    slots::Vector{<:AbstractSlot},
    ops::Vector{ClusterOp},
)::Matrix{Float64}
    isempty(ops) && throw(ArgumentError("empty operation list"))
    P = representation_matrix(slots, ops[1])
    for op in ops[2:end]
        P .+= representation_matrix(slots, op)
    end
    P ./= length(ops)
    return P
end

"""
    invariant_basis(slots, ops; tol = 1e-8) -> Matrix{Float64}

Orthonormal basis of the symmetry-invariant subspace, from the eigenvectors of
the Reynolds projector with eigenvalue 1. Eigenvalues must be 0 or 1 within
`tol`; anything else means the operation list is not a group.
"""
function invariant_basis(
    slots::Vector{<:AbstractSlot},
    ops::Vector{ClusterOp};
    tol::Real = 1e-8,
)::Matrix{Float64}
    P = invariant_projector(slots, ops)
    asym = norm(P - P')
    asym < 1e-8 * max(1.0, norm(P)) ||
        error("Reynolds projector is not symmetric (asymmetry $asym)")
    vals, vecs = eigen(Symmetric((P + P') / 2))
    for v in vals
        (abs(v) < tol || abs(v - 1.0) < tol) ||
            error("projector eigenvalue $v is neither 0 nor 1; " *
                  "the operation list is probably not a group")
    end
    keep = findall(v -> abs(v - 1.0) < tol, vals)
    return vecs[:, keep]
end

# ---------------------------------------------------------------------------
# Invariant counting (cycle-wise character formula)
# ---------------------------------------------------------------------------

# Character of the rank-l irrep of SO(3) at a proper rotation P.
function _chi_rotation(l::Int, P::AbstractMatrix{<:Real})::Float64
    θ = acos(clamp((tr(P) - 1.0) / 2.0, -1.0, 1.0))
    acc = 1.0
    for k = 1:l
        acc += 2.0 * cos(k * θ)
    end
    return acc
end

# Character of Sym^p of the (possibly improper) 3x3 orthogonal matrix M: the
# complete homogeneous symmetric polynomial h_p of the eigenvalues, via the
# Newton-type recurrence h_p = (1/p) Σ_{i=1}^p tr(M^i) h_{p-i}. This is the
# plethysm character; using χ(M)^p instead is wrong.
function _chi_sym_power(p::Int, M::AbstractMatrix{<:Real})::Float64
    h = zeros(Float64, p + 1)
    h[1] = 1.0
    psums = zeros(Float64, p)
    Mp = SMatrix{3, 3, Float64, 9}(I)
    for i = 1:p
        Mp = Mp * M
        psums[i] = tr(Mp)
    end
    for k = 1:p
        acc = 0.0
        for i = 1:k
            acc += psums[i] * h[k - i + 1]
        end
        h[k + 1] = acc / k
    end
    return h[p + 1]
end

# Contribution of one τ-cycle of length `cyclen` whose slots share spec `slot`;
# `sign` is the product of the pair-orientation swap signs along the cycle
# (1.0 for non-pair slots).
_cycle_character(slot::SpinSlot, op::ClusterOp, cyclen::Int, sign::Float64)::Float64 =
    sign * _chi_rotation(slot.l, op.spin_rotation^cyclen)
function _cycle_character(slot::DispSlot, op::ClusterOp, cyclen::Int,
                          sign::Float64)::Float64
    M = op.rotation^cyclen
    d = det(M)
    # Polar rank-L block times the invariant radial factor |u|^{2k}.
    return sign * d^slot.L * _chi_rotation(slot.L, d * M)
end
_cycle_character(slot::DispSymSlot, op::ClusterOp, cyclen::Int, sign::Float64)::Float64 =
    sign * _chi_sym_power(slot.p, op.rotation^cyclen)
_cycle_character(slot::SpinPairSlot, op::ClusterOp, cyclen::Int, sign::Float64)::Float64 =
    sign * _chi_rotation(slot.L_S, op.spin_rotation^cyclen)

# tr D(op) evaluated cycle-wise (never materializes D). The swap flags enter as
# the product of exchange signs along each cycle — this is where the pre-port
# prototype was wrong (it read a single-slot flag and rejected pair-slot cycles
# of length > 1 outright).
function _op_character(
    slots::Vector{<:AbstractSlot},
    op::ClusterOp,
    τ::Vector{Int},
    swap::Vector{Bool},
)::Float64
    n = length(slots)
    visited = falses(n)
    chi = 1.0
    for j = 1:n
        visited[j] && continue
        # Walk the τ-cycle through j, accumulating the swap-sign product.
        cyclen = 0
        sign = 1.0
        jp = j
        while true
            visited[jp] = true
            cyclen += 1
            if swap[jp]
                sign *= _exchange_sign(slots[jp]::SpinPairSlot)
            end
            jp = τ[jp]
            jp == j && break
            # All slots on a cycle share their non-site spec fields by
            # construction of τ (images preserve them); no further check needed.
        end
        chi *= _cycle_character(slots[j], op, cyclen, sign)
    end
    return chi
end

"""
    count_invariants(slots, ops) -> Int

Number of linearly independent symmetry invariants of the decorated cluster,
`n_ω = (1/|G_ω|) Σ_g tr D(g)`, evaluated cycle-wise without building any
representation matrix:

- each cycle of length `c` of the induced slot permutation contributes the
  single-slot character at `R_g^c` (axial `χ^{(l)}` for spin slots, polar
  `(det R)^L χ^{(L)}` for displacement blocks) — the naive per-slot character
  product `Π χ(R_g)` is wrong whenever `g` permutes sites;
- pair-slot cycles carry the product of the Clebsch--Gordan exchange signs of
  their orientation-swapped mappings;
- repeated same-site displacement powers (`DispSymSlot`) use the
  symmetrized-power (plethysm) character, not `χ(R)^p`.

All `ops` must preserve the decoration; use [`stabilizer_ops`](@ref) first. The
result must be an integer for a complete group — a non-integer character sum
raises an error (the usual cause is an operation list that is not closed; check
with [`assert_group_closure`](@ref)).
"""
function count_invariants(
    slots::Vector{<:AbstractSlot},
    ops::Vector{ClusterOp},
)::Int
    _validate_slots(slots)
    isempty(ops) && throw(ArgumentError("empty operation list"))
    total = 0.0
    for op in ops
        στ = _slot_permutation(slots, op)
        στ === nothing && throw(ArgumentError(
            "an operation does not preserve the decoration; filter with stabilizer_ops"))
        total += _op_character(slots, op, στ[1], στ[2])
    end
    count = total / length(ops)
    rounded = round(count)
    abs(count - rounded) < 1e-6 || throw(ArgumentError(
        "character sum $count is not an integer; the operation list is probably " *
        "not a closed group (verify with assert_group_closure)"))
    return Int(rounded)
end

end # module CountingOracle
