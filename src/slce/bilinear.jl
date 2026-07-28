"""
Bilinear / single-ion extraction — tesseral → Cartesian, on the training supercell.

A fitted [`SLCEModel`](@ref) is a polynomial in unit spin directions of arbitrary
body order and `l`. Exactly two SLCE channels map onto the classical bilinear form
`Σ eₐ'·M·e_b + Σ eₐ'·A·eₐ`:

- `ls = [1, 1]` (2-body, the `Lf = 0,1,2` parts fold into one 3×3 matrix that
  carries Heisenberg, Dzyaloshinskii–Moriya, and symmetric-anisotropic exchange),
- `ls = [2]` (1-body, a traceless symmetric single-ion tensor).

`ls = [0…]` channels are constants (folded into `j0`, dropped); every other SALC
(3-body and up, higher `l`) is **skipped** and reported.

The numerically delicate part — turning a fitted coefficient and its folded
tesseral tensor into a Cartesian matrix with the right normalization, and summing
the directed cluster members into one matrix per undirected bond — lives here and
is validated by reconstructing the SLCE energy
(`_reconstruct_energy ≈ predict_energy − j0`). Consumers: the public
[`bilinear_terms`](@ref) introspection (`slce/introspect.jl`) and the Sunny export
(`interop/sunny.jl` + `SLCESunnyExt`).
"""

# Cartesian axis (x,y,z) = (1,2,3) → the m-index of the folded tensor, with index
# (1,2,3) ↔ m = (−1,0,+1): real `Z_{1,m}` is `√(3/4π)·(y,z,x)` for m = (−1,0,+1),
# so x↔m=+1 (index 3), y↔m=−1 (index 1), z↔m=0 (index 2).
const _L1_AXIS_TO_MIDX = (3, 1, 2)

"""
    _l1_pair_matrix(folded) -> SMatrix{3,3}

The Cartesian bilinear matrix `M` with `eₐ'·M·e_b = Σ_{m₁,m₂} folded[m₁,m₂]
Z_{1,m₁}(eₐ) Z_{1,m₂}(e_b)`, i.e. `(3/4π)` (= `Harmonics.N1²`) times the folded
`(m₁,m₂)` tensor reindexed to Cartesian axes. `folded` is the rank-2 term tensor of
an `ls = [1,1]` SALC (already `Lf`-contracted), so its symmetric part is
Heisenberg + Γ and its antisymmetric part is the DM vector.
"""
function _l1_pair_matrix(folded::AbstractArray)::SMatrix{3,3,Float64,9}
    pref = 3 / (4π)
    p = _L1_AXIS_TO_MIDX
    return SMatrix{3,3,Float64}(ntuple(9) do k
        r = (k - 1) % 3 + 1
        c = (k - 1) ÷ 3 + 1
        pref * folded[p[r], p[c]]
    end...)
end

"""
    _l2_onsite_matrix(folded) -> SMatrix{3,3}

The traceless symmetric single-ion matrix `A` with `e'·A·e = Σ_m folded[m]
Z_{2,m}(e)`, from the rank-1 term tensor of an `ls = [2]` SALC (`folded[1..5]` ↔
`m = −2..+2`). The `Z_{2,m}` normalization enters through `Harmonics.A2` / `Harmonics.B2`.
"""
function _l2_onsite_matrix(folded::AbstractArray)::SMatrix{3,3,Float64,9}
    a = Harmonics.A2
    b = Harmonics.B2
    Bxy = a * folded[1]      # m = −2
    Byz = a * folded[2]      # m = −1
    d0 = folded[3]           # m =  0
    Bxz = a * folded[4]      # m = +1
    d2 = folded[5]           # m = +2
    Bxx = -b * d0 + a * d2
    Byy = -b * d0 - a * d2
    Bzz = 2b * d0
    return SMatrix{3,3,Float64}(Bxx, Bxy, Bxz,
                                Bxy, Byy, Byz,
                                Bxz, Byz, Bzz)
end

# Which bilinear channel an SALC maps to, from its sorted decoration label. Any
# displacement-decorated SALC is unsupported here by construction (the bilinear
# spin form has no lattice factor).
function _classify_salc(decors::AbstractVector{SiteDecor})::Symbol
    all(is_pure_spin, decors) || return :unsupported
    ls = Int[d.spin_l for d in decors]
    ls == [1, 1] && return :pair            # bilinear exchange
    ls == [2] && return :onsite             # single-ion anisotropy
    return :unsupported                      # 3-body+, higher-l pairs
end

_unsupported_salc_string(k::SALCKey) =
    "SALC(body=$(k.body), decors=$(k.decors), Lf=$(k.Lf)): not representable as a " *
    "bilinear pair or single-ion term (only pure-spin ls=[1,1] pairs and ls=[2] " *
    "single-ion extract)"

"""
    BilinearTerms

The supercell-route intermediate of the bilinear extraction: one bilinear matrix per
directed supercell bond `(a, b, R)` (canonicalized to `a ≤ b`), one single-ion
matrix per atom, and the human-readable list of skipped SALCs. Built by
[`_bilinear_terms`](@ref); consumed by the public [`bilinear_terms`](@ref), the
`Sunny` extension, and the Sunny-free energy reconstruction
[`_reconstruct_energy`](@ref).
"""
struct BilinearTerms
    pairs::Dict{Tuple{Int,Int,SVector{3,Int}},SMatrix{3,3,Float64,9}}
    onsites::Dict{Int,SMatrix{3,3,Float64,9}}
    skipped::Vector{String}
end

# Accumulate `m` into `d[key]` (zero-initialized).
_accum!(d, key, m::SMatrix{3,3,Float64,9}) =
    (d[key] = get(d, key, zero(SMatrix{3,3,Float64,9})) + m)

"""
    _bilinear_terms(model) -> BilinearTerms

Walk the fitted SALCs and fold each bilinear-representable channel onto the training
supercell: an `ls=[1,1]` member `(a, b, R)` contributes `jϕ·(4π)^(N/2)·_l1_pair_matrix`
to the bond `(a, b, R)`, an `ls=[2]` member contributes
`jϕ·(4π)^(N/2)·_l2_onsite_matrix` to its atom. Canonical (v4) members present each
physical bond as **one** member with sites sorted by `(atom, shift)` — the two
directed contributions are already summed into its `folded` tensor by
`_canonicalize_members`, so `a ≤ b` holds by construction and the fold is direct
(a non-canonical member is an internal error, not merged). The energy is then
`Σ eₐ'·M·e_b + Σ eₐ'·A·eₐ`, matching `predict_energy − j0` for a model with only
these channels.
"""
function _bilinear_terms(model::SLCEModel)::BilinearTerms
    pairs = Dict{Tuple{Int,Int,SVector{3,Int}},SMatrix{3,3,Float64,9}}()
    onsites = Dict{Int,SMatrix{3,3,Float64,9}}()
    skipped = String[]
    salcs = model.basis.salc_basis.salcs
    @inbounds for k in eachindex(model.jphi)
        salc = salcs[k]
        j = model.jphi[k]
        cls = _classify_salc(salc.decors)
        if cls === :unsupported
            push!(skipped, _unsupported_salc_string(salc.key))
            continue
        end
        scale = j * (4π)^(salc.body / 2)
        scale == 0.0 && continue
        for m in salc.members
            for t in m.terms
                if cls === :pair
                    a, b = m.atoms[1], m.atoms[2]
                    # canonical (v4) members sort sites by (atom, shift), so the
                    # a > b reverse-transpose merge of the pre-v4 layout is gone
                    a <= b || error("internal: non-canonical SALC member atoms " *
                                    "$(m.atoms); canonical members sort sites by " *
                                    "(atom, shift)")
                    R = m.shifts[2] - m.shifts[1]
                    _accum!(pairs, (a, b, R), scale * _l1_pair_matrix(t.folded))
                else # :onsite
                    _accum!(onsites, m.atoms[1], scale * _l2_onsite_matrix(t.folded))
                end
            end
        end
    end
    return BilinearTerms(pairs, onsites, skipped)
end

"""
    _reconstruct_energy(terms, e) -> Float64

The classical energy `Σ_{(a,b,R)} eₐ'·M·e_b + Σ_a eₐ'·A·eₐ` of a [`BilinearTerms`](@ref)
on a spin configuration `e` (`3 × n_atoms`, unit columns). For a model whose only
channels are `ls=[1,1]`/`ls=[2]` this equals `predict_energy(model, e) − j0` — the
Sunny-free gate on the conversion math.
"""
function _reconstruct_energy(terms::BilinearTerms, e::AbstractMatrix{<:Real})::Float64
    tot = 0.0
    @inbounds for ((a, b, _), M) in terms.pairs
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        eb = SVector{3,Float64}(e[1, b], e[2, b], e[3, b])
        tot += dot(ea, M * eb)
    end
    @inbounds for (a, A) in terms.onsites
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        tot += dot(ea, A * ea)
    end
    return tot
end
