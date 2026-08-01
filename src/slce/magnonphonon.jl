# Magnon–phonon vertices — the mixed second derivative of the energy surface (design
# record §7's third named strain-tier deliverable).
#
# WHAT THE OBJECT IS. Force constants differentiate twice in `u`; the bilinear couplings
# differentiate twice in the spin directions. The thing that couples the two subsystems is
# the MIXED derivative
#
#     V_{aα, bβ}(R) = ∂²E_cell / ∂u_{aα}(0) ∂e_{bβ}(R)
#
# taken at `u = 0` and at a given magnetic state. Feed it a phonon eigenvector on the left
# and a magnon eigenvector on the right and you have the one-magnon–one-phonon vertex; it
# is the quantity behind magnon–phonon hybridization gaps, phonon-mediated magnon damping
# and the Einstein–de Haas class of couplings.
#
# WHY THE SPIN INDEX IS CARTESIAN AND STILL UNAMBIGUOUS. A magnon amplitude is a TANGENTIAL
# deviation — the spin length is fixed, so only the two directions orthogonal to `ê_b` are
# physical, and any local-frame `(f₁, f₂)` is a convention. This returns the full Cartesian
# `β` of the TANGENTIAL derivative (`Harmonics.grad_Zlm`, which projects out the radial
# part, is the same object the torque design matrix is built from), so `V[α, :] · ê_b = 0`
# identically and the caller projects onto whichever frame their magnon basis uses. No
# frame convention is invented here, and none has to be pinned.
#
# WHY IT IS ADIABATIC. It is a derivative of the STATIC energy surface: the electronic and
# magnetic subsystems are assumed to follow the lattice instantaneously. Retardation, the
# Berry-phase (kinetic) term that gives phonons angular momentum, and any spin-lattice
# relaxation channel are outside a static cluster expansion by construction — the same
# scope line §7 draws for the multipole readout.

"""
    MagnonPhononVertices

The adiabatic magnon–phonon vertices of a [`SLCEModel`](@ref) at one spin configuration,
as produced by [`magnon_phonon_vertices`](@ref).

`vertices[(a, b, R)][α, β]` is

```
∂²E_cell / ∂u_{aα}(0) ∂e_{bβ}(R)
```

— atom `a` in the home cell is displaced, atom `b` in the cell at lattice shift `R` has
its spin direction varied. The `β` derivative is **tangential**: the radial part is
projected out, so `V·ê_b = 0` and only the physical (magnon) directions survive. The pair
is *ordered* — `a` is the displaced site and `b` the magnetic one — so `(a, b, R)` and
`(b, a, −R)` are different entries with different meanings, unlike the bond keys of
[`bilinear_terms`](@ref).

Entries are per training cell, in the model's energy unit per length unit (the spin
directions are dimensionless unit vectors).
"""
struct MagnonPhononVertices
    crystal::Crystal
    spins::SpinConfiguration
    vertices::Dict{Tuple{Int,Int,SVector{3,Int}},Matrix{Float64}}
end

Base.length(v::MagnonPhononVertices) = length(v.vertices)

function Base.show(io::IO, v::MagnonPhononVertices)
    print(io, "MagnonPhononVertices(", length(v.vertices), " (displaced, magnetic) pairs, ",
          n_atoms(v.crystal), " atoms)")
end

"""
    magnon_phonon_vertices(model::SLCEModel; spins) -> MagnonPhononVertices

The **adiabatic magnon–phonon vertices** of `model` at the spin configuration `spins`
(`3 × n_atoms`, unit columns): the exact mixed second derivatives

```
V_{aα, bβ}(R) = ∂²E_cell / ∂u_{aα}(0) ∂e_{bβ}(R)
```

at `u = 0`, with the `e` derivative taken **tangentially** (the spin length is fixed, so
the radial direction is not a degree of freedom). Contract the `α` index with a phonon
eigenvector and the `β` index with a magnon polarization to get the one-magnon–one-phonon
vertex; `V[α, :] · ê_b = 0` holds identically, which is what makes any local-frame
projection safe.

Only terms carrying **exactly one displacement factor of degree 1** together with at least
one spin factor contribute — a first derivative in `u` sees nothing else, and a spin
factor is what makes the derivative in `e` nonzero. So a model whose magnetoelastic sector
is declared at `disp = (degree = 2,)` has no vertices at all (that content feeds the
*spin-dependent force constants* instead), and a pure-spin or lattice-only model yields an
empty set rather than an error.

!!! note "\"Adiabatic\" is a scope statement, not a caveat about accuracy"
    These are derivatives of the static energy surface: the magnetic subsystem is assumed
    to follow the lattice instantaneously. Retardation, the Berry-phase term that carries
    phonon angular momentum, and spin-lattice relaxation are outside a static cluster
    expansion by construction — no choice of basis or truncation brings them in.

!!! note "The translation sum rule applies here too"
    Summing over the displaced atom and its lattice shifts gives zero for a model
    satisfying the acoustic sum rule: translating the crystal rigidly cannot change the
    energy, and therefore cannot change its derivative with respect to any spin either.
    Unlike [`strain_derivatives`](@ref) this function does not *require* the sum rule — no
    absolute positions enter — but a violating model gives vertices that do not vanish for
    a uniform displacement, which is unphysical in the same way nonzero acoustic
    frequencies are.

See also [`force_constants`](@ref), [`bilinear_terms`](@ref),
[`exchange_strain_derivatives`](@ref).
"""
function magnon_phonon_vertices(model::SLCEModel;
                                spins::Union{AbstractMatrix{<:Real},Nothing} = nothing)
    # Refuse BEFORE resolving: this deliverable is undefined without spin content, so
    # the spin-free branch of `_resolve_spins` (which returns `nothing`) is
    # unreachable here and `e` is a `SpinConfiguration` for the rest of the function.
    _basis_has_spin(model.basis) || throw(ArgumentError(
        "magnon_phonon_vertices needs a model with spin content: this basis has none, so " *
        "every derivative with respect to a spin direction is zero. Use " *
        "`force_constants` for the lattice-only response."))
    e = _resolve_spins(model, spins, "the magnon–phonon vertices")::SpinConfiguration
    espin = Matrix(e)
    _warn_unresolvable(model, "magnon_phonon_vertices")
    out = Dict{Tuple{Int,Int,SVector{3,Int}},Matrix{Float64}}()
    polycache = Dict{NTuple{3,Int},SolidHarmonics._Poly}()
    salcs = model.basis.salc_basis.salcs
    for k in eachindex(model.jphi)
        jphi = model.jphi[k]
        jphi == 0.0 && continue
        salc = salcs[k]
        weight = jphi * (4π)^(count(has_spin, salc.decors) / 2)
        for mem in salc.members
            for t in mem.terms
                _accumulate_mpv!(out, weight, t, mem, espin, polycache)
            end
        end
    end
    for key in [k for (k, T) in out if all(iszero, T)]
        delete!(out, key)
    end
    return MagnonPhononVertices(model.basis.crystal, e, out)
end

# One term's contribution. The structure is `_fill_fcs_tensor!`'s with one axis moved from
# the "evaluate" column to the "differentiate" column: the single degree-1 displacement
# factor gives the `α` index as before, and each spin slot in turn is differentiated
# (product rule — a term with two spin factors on the same atom contributes twice, through
# different `midx` axes) while the rest are evaluated.
function _accumulate_mpv!(out, weight::Float64, t::SALCTerm, mem::SALCMember,
                          e::Matrix{Float64}, polycache)
    dslots = Int[]
    sslots = Int[]
    for i in eachindex(t.slots)
        if t.slots[i].factor.channel == DISP
            push!(dslots, i)
        else
            push!(sslots, i)
        end
    end
    isempty(sslots) && return nothing
    # A first derivative in `u` sees a term only if its displacement degrees sum to exactly
    # 1. Every displacement factor has degree `2k + l ≥ 1`, so that single test also forces
    # "exactly one factor, with (k, l) = (0, 1)" — stated as the degree sum because that is
    # the same test the force constants and the strain path apply.
    sum(2 * t.slots[i].factor.k + t.slots[i].factor.l for i in dslots; init = 0) == 1 ||
        return nothing
    id = dslots[1]
    sd = t.slots[id].site
    for is in sslots
        sb = t.slots[is].site
        key = (mem.atoms[sd], mem.atoms[sb], mem.shifts[sb] - mem.shifts[sd])
        T = get!(() -> zeros(Float64, 3, 3), out, key)
        _fill_mpv_tensor!(T, weight, t, id, is, sslots, mem, e, polycache)
    end
    return nothing
end

function _fill_mpv_tensor!(T::Matrix{Float64}, weight::Float64, t::SALCTerm, id::Int,
                           is::Int, sslots::Vector{Int}, mem::SALCMember,
                           e::Matrix{Float64}, polycache)
    bl = t.slots[is].factor.l
    eb = _mpv_spin(e, mem.atoms[t.slots[is].site])
    for midx in CartesianIndices(t.folded)
        w = t.folded[midx]
        w == 0.0 && continue
        for i in sslots
            i == is && continue                        # this one is differentiated below
            sl = t.slots[i]
            l = sl.factor.l
            # Unchecked: `_resolve_spins` is this path's door (see forceconstants.jl).
            w *= Harmonics.Zlm_unsafe(l, midx[i] - l - 1,
                                      _mpv_spin(e, mem.atoms[sl.site]))
            w == 0.0 && break
        end
        w == 0.0 && continue
        # the displacement axis: the α-derivative of `R_{1m}(u)` at u = 0, i.e. the same
        # monomial coefficients the force constants read
        pk = (0, 1, midx[id] - 2)
        poly = get!(() -> SolidHarmonics.solid_harmonic_poly(pk[1], pk[2], pk[3]),
                    polycache, pk)
        # the spin axis: the TANGENTIAL gradient, radial part projected out
        g = Harmonics.grad_Zlm_unsafe(bl, midx[is] - bl - 1, eb)
        for α = 1:3
            dα = _site_derivative(poly, (α,))
            dα == 0.0 && continue
            for β = 1:3
                T[α, β] += weight * w * dα * g[β]
            end
        end
    end
    return nothing
end

@inline _mpv_spin(e::Matrix{Float64}, a::Int) =
    SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
