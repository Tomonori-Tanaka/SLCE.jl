"""
    CoupledBasis{R}

One symmetry-channel coupled basis function over an `N`-site cluster
(`R = N + 1`): angular momenta `ls = [l₁,…,l_N]` coupled along the left tree
`Lseq` to a final total `Lf`, with the real (tesseral) coefficient `tensor` of
rank `R` indexed `[m₁,…,m_N, Mf]`.

The coupled field components `f_Mf(e₁,…,e_N) = Σ_{m₁…m_N} tensor[m₁…m_N, Mf] ·
∏ᵢ Zₗᵢ,mᵢ(eᵢ)` transform among themselves as the `Lf` irrep under a common
rotation of all `eᵢ`.
"""
struct CoupledBasis{R}
    ls::Vector{Int}
    Lseq::Vector{Int}
    Lf::Int
    tensor::Array{Float64,R}
end

"""
    body_order(cb::CoupledBasis) -> Int

Number of sites `N` in the cluster (`= R - 1`).
"""
body_order(cb::CoupledBasis{R}) where {R} = R - 1

"""
    coupled_bases(ls; scalar_only = false) -> Vector{CoupledBasis{length(ls)+1}}

All coupled bases for the per-site angular momenta `ls`, one per `(Lseq, Lf)`
coupling path. With `scalar_only = true` only the scalar `Lf == 0` sector is kept.
On a pure-spin label (every slot a spin slot) `L_S ≡ Lf`, so this is the spec's
`soc = false` restriction for the pure-spin builder; the decor engine screens on
`L_S` instead and on a mixed label the two screens differ — see
[`AngularMomentum.build_real_bases`](@ref).
"""
function coupled_bases(ls::AbstractVector{<:Integer}; scalar_only::Bool = false)
    lsv = collect(Int, ls)
    R = length(lsv) + 1
    out = CoupledBasis{R}[]
    for (Lseq, Lf, tensor) in AngularMomentum.build_real_bases(lsv;
                                                              scalar_only = scalar_only)
        push!(out, CoupledBasis{R}(lsv, Lseq, Lf, tensor))
    end
    return out
end
