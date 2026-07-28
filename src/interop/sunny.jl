"""
Sunny export — the code-agnostic core.

Sunny.jl represents only **bilinear pair exchange** (a 3×3 matrix per bond) and
**single-ion anisotropy** (a quadratic form per site), so the export covers exactly
the two SCE channels the bilinear extraction (`slce/bilinear.jl`,
[`_bilinear_terms`](@ref)) produces; every other SALC is skipped and reported.
This file owns the Sunny-specific half: unfolding the supercell matrices onto the
chemical primitive cell ([`_sunny_primitive`](@ref)) and the [`to_sunny`](@ref)
entry point. The actual `Sunny.System` assembly is the thin `SLCESunnyExt`
extension (loaded by `using Sunny`), which consumes these intermediates.
"""

"""
    SunnyPrimitive

The primitive-cell (unfolded) intermediate of a Sunny export: the chemical
primitive lattice and sublattice basis recovered from the supercell's pure
translations, the integer `reshape` matrix (`supercell = primitive · reshape`), the
per-supercell-atom `sublattice` map, one bilinear matrix per **primitive** bond
`(i, j, n)` (sublattices `i,j`, primitive-cell offset `n`) placed **without
multiplicity** (Sunny's periodic replication restores it), one single-ion matrix per
sublattice, and a `clean` flag. `clean = false` means the fitted model does not fold
onto the primitive cell (e.g. the interaction range reaches the supercell boundary),
and callers fall back to the exact supercell route.
"""
struct SunnyPrimitive
    latvecs::SMatrix{3,3,Float64,9}
    positions::Vector{SVector{3,Float64}}
    types::Vector{String}
    reshape::SMatrix{3,3,Int,9}
    sublattice::Vector{Int}
    bonds::Dict{Tuple{Int,Int,SVector{3,Int}},SMatrix{3,3,Float64,9}}
    onsites::Dict{Int,SMatrix{3,3,Float64,9}}
    clean::Bool
    skipped::Vector{String}
end

# Lexicographic n ≥ 0 — the canonical direction for a same-sublattice bond.
_ge0(n::SVector{3,Int}) = (n[1], n[2], n[3]) >= (0, 0, 0)

# Three shortest linearly independent vectors among `cands` (sorted by length):
# the primitive lattice vectors.
function _shortest_independent(cands::Vector{SVector{3,Float64}}, tol::Float64)
    v = SVector{3,Float64}[]
    for c in sort(cands; by = norm)
        norm(c) <= tol && continue
        if isempty(v)
            push!(v, c)
        elseif length(v) == 1
            norm(cross(v[1], c)) > tol * norm(c) && push!(v, c)
        else
            abs(dot(cross(v[1], v[2]), c)) > tol * norm(c)^2 && push!(v, c)
        end
        length(v) == 3 && break
    end
    length(v) == 3 || error("could not recover three primitive lattice vectors")
    M = SMatrix{3,3,Float64}(v[1]..., v[2]..., v[3]...)
    # Sunny requires right-handed lattice vectors; swapping two columns flips the sign
    # of the determinant while spanning the same lattice.
    return det(M) < 0 ? SMatrix{3,3,Float64}(v[1]..., v[3]..., v[2]...) : M
end

"""
    _sunny_primitive(model) -> SunnyPrimitive

Unfold the supercell [`BilinearTerms`](@ref) onto the chemical primitive cell: recover
the primitive lattice from the space group's pure translations, group supercell atoms
into sublattices by their translation orbits, and map each supercell bond `(a,b,R)`
to a primitive bond `(i,j,n)`. Translation-equivalent supercell bonds collapse to one
primitive bond; `clean` records whether every offset is integral and every such group
agrees (i.e. the model genuinely lives on the primitive cell).
"""
function _sunny_primitive(model::SLCEModel)::SunnyPrimitive
    crystal = model.basis.crystal
    sg = model.basis.spacegroup
    A = crystal.lattice.vectors
    nat = n_atoms(crystal)
    cart = cartesian_positions(crystal)
    tol = max(sg.tol, 1e-8)

    # primitive lattice from the pure translations (+ the supercell vectors)
    cands = SVector{3,Float64}[]
    for t in sg.translation_ops
        push!(cands, A * sg.ops[t].translation_frac)
    end
    for j = 1:3
        push!(cands, SVector{3,Float64}(A[1, j], A[2, j], A[3, j]))
    end
    Lp = _shortest_independent(cands, tol)
    reshape_m = SMatrix{3,3,Int}(round.(Int, Lp \ A))

    # sublattices = translation orbits of the supercell atoms
    sublattice = zeros(Int, nat)
    reps = Int[]
    @inbounds for a = 1:nat
        sublattice[a] == 0 || continue
        push!(reps, a)
        s = length(reps)
        for t in sg.translation_ops
            sublattice[sg.map_sym[a, t]] = s
        end
    end
    nsubl = length(reps)
    # `_wrap01`, not a bare `mod`: a sublattice origin landing at exactly 1.0
    # instead of 0.0 is handed to Sunny as a position outside the unit cell, and
    # two sublattices at 0.0 and 1.0 are the same site described twice.
    positions = [SVector{3,Float64}(_wrap01.(Lp \ SVector{3,Float64}(cart[1, reps[i]],
                                                                     cart[2, reps[i]],
                                                                     cart[3, reps[i]])))
                 for i = 1:nsubl]
    types = [crystal.species_labels[crystal.species[reps[i]]] for i = 1:nsubl]

    # primitive-cell index of supercell atom `a` (relative to its sublattice origin)
    cellof(a::Int) = round.(Int, Lp \ SVector{3,Float64}(cart[1, a], cart[2, a], cart[3, a]) -
                            positions[sublattice[a]])

    terms = _bilinear_terms(model)
    bonds = Dict{Tuple{Int,Int,SVector{3,Int}},SMatrix{3,3,Float64,9}}()
    # The recovered cell must tile the supercell exactly: `supercell = primitive·reshape`
    # with `|det(reshape)|` primitive cells. A single-ion-only model has no bonds to
    # catch a mis-recovered lattice, so check the geometry directly.
    clean = abs(det(SMatrix{3,3,Float64}(reshape_m))) ≈ length(sg.translation_ops) &&
            isapprox(Lp * SMatrix{3,3,Float64}(reshape_m), A; atol = tol * (1 + norm(A)))
    for ((a, b, R), M) in terms.pairs
        i, j = sublattice[a], sublattice[b]
        fb = Lp \ (SVector{3,Float64}(cart[1, b], cart[2, b], cart[3, b]) +
                   A * SVector{3,Float64}(R[1], R[2], R[3]))
        cb = round.(Int, fb - positions[j])
        clean &= maximum(abs.(fb - positions[j] - cb)) < tol
        n = SVector{3,Int}(cb - cellof(a))
        key, val = (i < j || (i == j && _ge0(n))) ? ((i, j, n), M) :
                   ((j, i, -n), SMatrix{3,3,Float64}(transpose(M)))
        if haskey(bonds, key)
            clean &= isapprox(bonds[key], val; atol = 1e-9 * (1 + norm(val)))
        else
            bonds[key] = val
        end
    end

    onsites = Dict{Int,SMatrix{3,3,Float64,9}}()
    for (a, Aos) in terms.onsites
        i = sublattice[a]
        if haskey(onsites, i)
            clean &= isapprox(onsites[i], Aos; atol = 1e-9 * (1 + norm(Aos)))
        else
            onsites[i] = Aos
        end
    end

    return SunnyPrimitive(Lp, positions, types, reshape_m, sublattice, bonds, onsites,
                          clean, terms.skipped)
end

"""
    to_sunny(model; spins, g = 2, mode = :auto, scaling = :auto, placement = :auto) -> Sunny.System

Export a fitted [`SLCEModel`](@ref) to a Sunny.jl `System`. **Requires `using Sunny`**
(the export is a package extension). Only the Sunny-representable channels are
exported — bilinear pair (`ls=[1,1]`) exchange and single-ion (`ls=[2]`) anisotropy;
higher-order / higher-`l` SALCs are skipped and reported via `@warn`.

`spins` gives the physical effective spin length `S_eff = m/(g μ_B)` (a number, or a
per-species `label => S` mapping). The SCE couplings are fit from *unit* spin
directions, so they absorb the moment magnitude (`J_SCE = J_phys S²`); the magnon
frequency, however, scales as `1/S_eff`, so the export must carry `S_eff` explicitly.

`scaling` chooses how:

- `:moment` — put `S_eff` directly into Sunny's `Moment` and rescale each bilinear bond
  by `1/(SₐS_b)`. Both the static energy *and* the dispersion are exact, but Sunny's
  `Moment` accepts only half-integer spins, so `S_eff` must be a half-integer.
- `:coupling` — keep `Moment` at a placeholder `s₀ = 1` and fold `S_eff` into the
  couplings (`J = M/(s₀·√(SᵢSⱼ))`, single-ion `1/(s₀ Sᵢ)`). Works for **any** positive
  `S_eff` (itinerant / non-half-integer moments). Only the magnon *dispersion* is
  physical; the represented static energy is rescaled (the dispersion is invariant under
  an overall spin scale `sᵢ → c sᵢ, J → J/c`). Exact for a uniform `S_eff`; the on-site
  (Larmor) term is approximate for a non-uniform one.
- `:auto` (default) — `:moment` when every `S_eff` is a half-integer, else `:coupling`.

`mode` is the Sunny spin mode: `:dipole` applies the quantum rank-2 renormalization to
the single-ion term (exchange is mode-independent), `:dipole_uncorrected` keeps it purely
classical, `:auto` (default) picks `:dipole` for half-integer spins and
`:dipole_uncorrected` otherwise. A `:coupling` placeholder `Moment` cannot carry the
quantum quadrupole, so single-ion + `:dipole` + `:coupling` is rejected.

`placement = :primitive` unfolds onto the chemical primitive cell (physical, unfolded
dispersion) when the model maps cleanly, else falls back to the training supercell
(`:explicit`, exact but folded); `:auto` picks primitive when clean.
"""
function to_sunny(model::SLCEModel, args...; kwargs...)
    error("to_sunny requires Sunny.jl; run `using Sunny` first " *
          "(it activates the SLCE Sunny export extension).")
end
