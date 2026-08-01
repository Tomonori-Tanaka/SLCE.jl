"""
    AngularMomentum

Angular-momentum coupling primitives, from scratch:

- `clebsch_gordan(j1, m1, j2, m2, J, M)` — Clebsch–Gordan coefficient via the
  Racah single-sum formula (integer momenta).
- `wignerD_real(l, R)` — the real Wigner-D matrix of a 3×3 rotation `R` in the
  tesseral-harmonic basis. It is built **directly from this package's own**
  `Harmonics.Zₗₘ` by solving the (exact, oversampled) linear system
  `Zₗ,m(R·r̂) = Σ_{m'} Δ[m,m'] Zₗ,m'(r̂)`. This guarantees the rotation matrix is
  consistent with the harmonic convention, uses real arithmetic only, and works
  for improper rotations (reflections, inversion) without special cases.
"""
module AngularMomentum

using LinearAlgebra: norm, I
using StaticArrays

using ..Harmonics: Zlm_unsafe

export clebsch_gordan, wignerD_real
export c2r_matrix, coupling_paths, coeff_tensor_complex, complex_to_real_tensor,
    build_real_bases

# n! as Float64 (no overflow for the l-range used here); negative → 0.
@inline function _fact(n::Integer)::Float64
    n < 0 && return 0.0
    f = 1.0
    @inbounds for i = 2:n
        f *= i
    end
    return f
end

"""
    clebsch_gordan(j1, m1, j2, m2, J, M) -> Float64

Clebsch–Gordan coefficient `⟨j1 m1 j2 m2 | J M⟩` (integer angular momenta), via the
Racah single-sum formula. Returns `0.0` outside the selection rules. The `Float64`
factorials cap the admissible momenta at `j1 + j2 + J + 1 ≤ 170` (beyond, `n!`
overflows to `Inf`); larger inputs throw instead of silently returning `NaN` —
far outside the small-`l` regime this package targets.

# Arguments
- `j1, m1, j2, m2, J, M::Integer`: the coupling quantum numbers.

# Returns
- `Float64`: the coefficient.
"""
function clebsch_gordan(j1::Integer, m1::Integer, j2::Integer, m2::Integer,
                        J::Integer, M::Integer)::Float64
    (m1 + m2 == M) || return 0.0
    (j1 >= 0 && j2 >= 0 && J >= 0) || return 0.0
    (abs(m1) <= j1 && abs(m2) <= j2 && abs(M) <= J) || return 0.0
    (abs(j1 - j2) <= J <= j1 + j2) || return 0.0
    # largest factorial argument below; 171! overflows Float64 to Inf — fail loudly
    j1 + j2 + J + 1 <= 170 || throw(ArgumentError(
        "clebsch_gordan: j1 + j2 + J = $(j1 + j2 + J) exceeds the Float64 " *
        "factorial range (n! overflows at n = 171)"))

    tri = _fact(j1 + j2 - J) * _fact(j1 - j2 + J) * _fact(-j1 + j2 + J) /
          _fact(j1 + j2 + J + 1)
    pref = sqrt((2J + 1) * tri) *
           sqrt(_fact(j1 + m1) * _fact(j1 - m1) * _fact(j2 + m2) *
                _fact(j2 - m2) * _fact(J + M) * _fact(J - M))

    kmin = max(0, j2 - m1 - J, j1 + m2 - J)
    kmax = min(j1 + j2 - J, j1 - m1, j2 + m2)
    s = 0.0
    @inbounds for k = kmin:kmax
        denom = _fact(k) * _fact(j1 + j2 - J - k) * _fact(j1 - m1 - k) *
                _fact(j2 + m2 - k) * _fact(J - j2 + m1 + k) * _fact(J - j1 - m2 + k)
        s += (isodd(k) ? -1.0 : 1.0) / denom
    end
    c = pref * s
    # factorial *products* can still overflow inside the admitted range; never
    # return a silent Inf/NaN
    isfinite(c) || throw(ArgumentError(
        "clebsch_gordan: Float64 overflow in the Racah sum for (j1, j2, J) = " *
        "($j1, $j2, $J); momenta this large are outside the supported range"))
    return c
end

# Deterministic, well-spread sample directions on the unit sphere (Fibonacci
# sphere), used to fit the rotation matrix. No RNG so results are reproducible.
@inline function _fib_dir(k::Int, K::Int)::SVector{3,Float64}
    golden = π * (3.0 - sqrt(5.0))
    z = 1.0 - 2.0 * (k - 0.5) / K
    r = sqrt(max(0.0, 1.0 - z * z))
    θ = golden * k
    return SVector{3,Float64}(r * cos(θ), r * sin(θ), z)
end

"""
    wignerD_real(l, R) -> Matrix{Float64}

Real Wigner-D matrix `Δ` (size `(2l+1)×(2l+1)`) of the rotation `R` (3×3, proper or
improper) in the tesseral-harmonic basis, indexed by `m = -l:l`
(`row/col = m + l + 1`), satisfying

```
Zₗ,m(R·r̂) = Σ_{m'} Δ[m, m'] · Zₗ,m'(r̂)   for all r̂.
```

Built from the package's own `Zₗₘ` by an exact oversampled least-squares fit, so it
is consistent with the harmonic convention by construction.

# Arguments
- `l::Integer`: angular momentum (`l ≥ 0`).
- `R::AbstractMatrix{<:Real}`: a 3×3 orthogonal matrix.

# Returns
- `Matrix{Float64}`: the `(2l+1)×(2l+1)` real Wigner-D matrix.
"""
function wignerD_real(l::Integer, R::AbstractMatrix{<:Real})::Matrix{Float64}
    l >= 0 || throw(ArgumentError("need l ≥ 0 (got $l)"))
    size(R) == (3, 3) || throw(ArgumentError("R must be 3×3"))
    l == 0 && return fill(1.0, 1, 1)
    d = 2l + 1
    K = 3d                      # oversample for conditioning; fit is exact
    Rs = SMatrix{3,3,Float64}(R)
    A = Matrix{Float64}(undef, K, d)
    B = Matrix{Float64}(undef, K, d)
    @inbounds for k = 1:K
        u = _fib_dir(k, K)
        ru = Rs * u
        ru = ru / norm(ru)
        for (col, mp) in enumerate(-l:l)
            A[k, col] = Zlm_unsafe(l, mp, u)
            B[k, col] = Zlm_unsafe(l, mp, ru)
        end
    end
    # Z(R·) = A·Δᵀ  ⇒  Δᵀ = A \ B  (least squares; residual ≈ 0).
    return permutedims(A \ B)
end

# ---------------------------------------------------------------------------
# Coupling: complex CG-chained tensors → real (tesseral) coupled bases.
# ---------------------------------------------------------------------------

"""
    c2r_matrix(l) -> Matrix{ComplexF64}

Complex→real (tesseral) transform `U` for degree `l`, `Z = U·Y`, indexed by
`m = -l:l` (`row/col = m+l+1`). Unitary. Consistent with `Harmonics.Zₗₘ`.
"""
function c2r_matrix(l::Integer)::Matrix{ComplexF64}
    l >= 0 || throw(ArgumentError("need l ≥ 0 (got $l)"))
    U = zeros(ComplexF64, 2l + 1, 2l + 1)
    index(m) = m + l + 1
    U[index(0), index(0)] = 1
    for m = 1:l
        s = 1 / sqrt(2.0)
        sgn = iseven(m) ? 1.0 : -1.0
        U[index(m), index(m)] = sgn * s
        U[index(m), index(-m)] = s
        U[index(-m), index(m)] = -im * sgn * s
        U[index(-m), index(-m)] = im * s
    end
    return U
end

"""
    coupling_paths(ls) -> Vector{Tuple{Vector{Int},Int}}

All admissible left-coupling sequences for `ls = [l1,…,lN]`, as `(Lseq, Lf)` where
`Lseq = [L₁₂, L₁₂₃, …]` (length `N-2`, empty for `N ≤ 2`) and `Lf` is the final
total angular momentum.
"""
function coupling_paths(ls::AbstractVector{<:Integer})::Vector{Tuple{Vector{Int},Int}}
    N = length(ls)
    N >= 1 || throw(ArgumentError("ls must be non-empty"))
    paths = Tuple{Vector{Int},Int}[]
    if N == 1
        push!(paths, (Int[], Int(ls[1])))
        return paths
    end
    if N == 2
        for Lf = abs(ls[1] - ls[2]):(ls[1]+ls[2])
            push!(paths, (Int[], Lf))
        end
        return paths
    end
    function rec(k::Int, Lprev::Int, seq::Vector{Int})
        if k == N - 1
            for Lf = abs(Lprev - ls[end]):(Lprev+ls[end])
                push!(paths, (copy(seq), Lf))
            end
            return
        end
        for L = abs(Lprev - ls[k+1]):(Lprev+ls[k+1])
            rec(k + 1, L, vcat(seq, L))
        end
    end
    for L12 = abs(ls[1] - ls[2]):(ls[1]+ls[2])
        rec(2, L12, [L12])
    end
    return paths
end

"""
    coeff_tensor_complex(ls, Lseq, Lf) -> Array{Float64, N+1}

Complex-basis coupled coefficient tensor `C[m₁,…,m_N, Mf] = ⟨(l₁m₁)…(l_Nm_N)|Lf Mf⟩`
along the left-coupling tree. The running intermediate `M`'s are fixed by the CG
selection rule (`M_{1..k} = Σ_{i≤k} m_i`), so this is a plain product of CGs.
"""
function coeff_tensor_complex(ls::AbstractVector{<:Integer}, Lseq::AbstractVector{<:Integer},
                              Lf::Integer)::Array{Float64}
    N = length(ls)
    dims = ntuple(i -> i <= N ? 2ls[i] + 1 : 2Lf + 1, N + 1)
    C = zeros(Float64, dims...)
    if N == 1
        l1 = ls[1]
        Lf == l1 || return C
        for (i, m) = enumerate(-l1:l1)
            C[i, m + Lf + 1] = 1.0
        end
        return C
    end
    # running coupled momenta P[1]=l1, P[2..N-1]=Lseq, P[N]=Lf
    P = Vector{Int}(undef, N)
    P[1] = ls[1]
    for k = 2:(N-1)
        P[k] = Lseq[k-1]
    end
    P[N] = Lf
    site_dims = ntuple(i -> 2ls[i] + 1, N)
    for index in CartesianIndices(site_dims)
        runM = index[1] - ls[1] - 1            # m₁
        amp = 1.0
        ok = true
        for k = 2:N
            mk = index[k] - ls[k] - 1
            Mk = runM + mk
            amp *= clebsch_gordan(P[k-1], runM, ls[k], mk, P[k], Mk)
            runM = Mk
            if amp == 0.0
                ok = false
                break
            end
        end
        ok || continue
        abs(runM) <= Lf || continue
        C[Tuple(index)..., runM + Lf + 1] = amp
    end
    return C
end

# Mode-n product: apply matrix `M` on mode `n` of tensor `A`.
function nmode_mul(A::AbstractArray, M::AbstractMatrix, n::Int)
    sz = size(A)
    d = length(sz)
    perm = (n, setdiff(1:d, (n,))...)
    invp = invperm(perm)
    B = permutedims(A, perm)
    Bmat = reshape(B, sz[n], :)
    Cmat = M * Bmat
    C = reshape(Cmat, (size(M, 1), sz[setdiff(1:d, (n,))]...))
    return permutedims(C, invp)
end

"""
    complex_to_real_tensor(Ccx, ls, Lf; tol = 1e-10) -> Array{Float64, N+1}

Transform a complex coupled tensor to the real (tesseral) basis: apply `conj(U_l)`
on each site axis, `U_Lf` on the final multiplet axis, then a global phase
`exp(−iπ/2·(Σl − Lf))` that renders the result real. Errors if the imaginary
residue exceeds `tol`.
"""
function complex_to_real_tensor(Ccx::AbstractArray, ls::AbstractVector{<:Integer},
                                Lf::Integer; tol::Real = 1e-10)::Array{Float64}
    N = length(ls)
    C = ComplexF64.(Ccx)
    for i = 1:N
        C = nmode_mul(C, conj(c2r_matrix(ls[i])), i)
    end
    C = nmode_mul(C, c2r_matrix(Lf), N + 1)
    C .*= exp(-1im * (π / 2) * (sum(ls) - Lf))
    maxim = maximum(abs ∘ imag, C)
    maxim <= tol || error("complex_to_real_tensor: imaginary residue $maxim > $tol")
    return Array{Float64}(real.(C))
end

"""
    build_real_bases(ls; scalar_only = false) -> Vector{Tuple{Vector{Int},Int,Array{Float64}}}

All real coupled tensors for `ls`, as `(Lseq, Lf, tensor)`. With
`scalar_only = true` only the scalar `Lf == 0` sector is kept — the spec-level
`soc = false` restriction, so `scalar_only ≡ !soc`.
"""
function build_real_bases(ls::AbstractVector{<:Integer};
                          scalar_only::Bool = false)::Vector{Tuple{Vector{Int},Int,Array{Float64}}}
    out = Tuple{Vector{Int},Int,Array{Float64}}[]
    for (Lseq, Lf) in coupling_paths(ls)
        (scalar_only && Lf != 0) && continue
        Ccx = coeff_tensor_complex(ls, Lseq, Lf)
        push!(out, (Lseq, Lf, complex_to_real_tensor(Ccx, ls, Lf)))
    end
    return out
end

end # module AngularMomentum
