# Export harmonic force constants to phonopy.
#
# This is the joint expansion's exit: `force_constants` produces Φ, and everything a
# phonon study wants downstream of Φ — band structures, DOS, thermal properties,
# group velocities, thermal conductivity — already exists in phonopy. Writing its
# `FORCE_CONSTANTS` is a far better use of effort than reimplementing any of it.
#
# The one hazard is that the format is positional: `FORCE_CONSTANTS` is a matrix over
# SUPERCELL atom indices, and phonopy builds that supercell itself from the unit cell
# and `--dim`. Disagree with its ordering and the result is a permuted force-constant
# matrix — which still diagonalizes, still gives three acoustic modes at Γ, and is
# simply wrong. So the ordering is not inferred from the documentation here; it was
# read off phonopy itself and is pinned by a test that runs phonopy and compares its
# frequencies against `dynamical_matrix` (`test/phonopy/`).
#
# The convention, for the record: for `dim = (n1, n2, n3)` the supercell atom index is
#
#     s(a, l) = (a - 1) * n1*n2*n3 + (l1 + n1 * (l2 + n2 * l3)) + 1
#
# with `a` the unit-cell atom (1-based, in the POSCAR's order) and `l` the 0-based
# lattice point. The unit-cell atom is the SLOWEST axis and `l1` the fastest.
#
# The POSCAR is written by the same call, deliberately: it is what fixes `a`. SLCE's
# `Crystal` may interleave species, VASP's format may not, so the atoms are permuted
# into species-grouped order — and the force constants are permuted with them. Writing
# only the force constants would leave that permutation for the user to rediscover.

"""
    write_phonopy(dir, fcs::ForceConstantSet; dim = nothing, comment = "") -> NamedTuple

Write `fcs` as a phonopy calculation: `POSCAR` (the unit cell) and `FORCE_CONSTANTS`
(the harmonic force constants over the `dim` supercell phonopy builds from it). Only
`fcs.order == 2` is representable. Returns `(; poscar, force_constants, dim, n_super)`.

```julia
fcs = force_constants(model; spins = afm, order = 2)
write_phonopy("phonons/", fcs)
```
```console
\$ cd phonons && phonopy --readfc --dim="3 3 3" -c POSCAR --band="0 0 0  1/2 0 0"
```

The printed `--dim` must be the one in the returned `dim` — the file is a matrix over
supercell atom indices, and a mismatched supercell silently permutes it. `dim`
defaults to the smallest odd box holding every lattice shift in `fcs` without
wraparound (`2·max|R_d| + 1`); a smaller one is allowed and folds the shifts that
collide, which is the same aliasing a smaller supercell would have had, with a
warning naming how many pairs folded.

Units are `fcs`'s own: eV/Å² if the model was fitted to eV and Å, which is what
phonopy expects from a VASP-convention calculation.

!!! note "The magnetic state is already baked in"
    `fcs` was evaluated at one spin configuration, and phonopy has no notion of one.
    Export two magnetic states to two directories and the band structures differ by
    exactly the magnetoelastic content of the model — see [`force_constants`](@ref)
    for which sector supplies it, and for why the joint path is what makes those
    constants carry the right magnetic symmetry in the first place.

!!! warning "Atoms are reordered"
    VASP's format groups atoms by species; a `Crystal` need not. The `POSCAR` is
    written species-grouped and `FORCE_CONSTANTS` is permuted to match, so the pair is
    consistent — but the indices are the POSCAR's, not `fcs.crystal`'s. Use the
    written `POSCAR` as phonopy's `-c` cell; substituting another file with the same
    atoms in a different order reintroduces exactly the permutation this pairing
    exists to prevent.
"""
function write_phonopy(dir::AbstractString, fcs::ForceConstantSet;
                       dim::Union{Nothing,NTuple{3,Integer},AbstractVector{<:Integer}} =
                           nothing,
                       comment::AbstractString = "")
    fcs.order == 2 || throw(ArgumentError(
        "phonopy's FORCE_CONSTANTS is the harmonic matrix; got order = $(fcs.order). " *
        "Export order 2 (higher orders would need the ALAMODE or ShengBTE formats)."))
    # An EMPTY set writes a valid, all-zero FORCE_CONSTANTS that phonopy reads happily
    # and turns into an all-zero band structure. That is the package's own
    # "plausible-looking output from an empty computation" failure — `force_constants`
    # documents that a pure-spin model yields an empty set, and `_warn_spin_blind`
    # deliberately stays silent there (no displacement content at this order at all),
    # so nothing else in the chain says a word.
    isempty(fcs.constants) && @warn(
        "write_phonopy: this force-constant set is EMPTY, so the file will be an " *
        "all-zero matrix — phonopy will read it and produce an all-zero band " *
        "structure. A pure-spin model, or a basis with no displacement term at " *
        "order $(fcs.order), yields an empty set.")
    crystal = fcs.crystal
    nat = n_atoms(crystal)
    d = dim === nothing ? _phonopy_auto_dim(fcs) : NTuple{3,Int}(Tuple(dim))
    all(>(0), d) || throw(ArgumentError("dim must be positive; got $d"))
    ncell = prod(d)
    nsup = nat * ncell
    perm = _species_grouped_perm(crystal)          # POSCAR order → Crystal atom
    inv_perm = invperm(perm)                       # Crystal atom → POSCAR order

    Φ = zeros(Float64, 3 * nsup, 3 * nsup)
    folded = 0
    seen = Set{Tuple{Int,Int,NTuple{3,Int}}}()
    for ((atoms, shifts), T) in fcs.constants
        a, b = atoms[1], atoms[2]
        R = Tuple(Int.(shifts[2] - shifts[1]))     # anchored, so shifts[1] == 0
        key = (a, b, map(mod, R, d))
        key in seen ? (folded += 1) : push!(seen, key)
        pa, pb = inv_perm[a], inv_perm[b]
        for l3 = 0:(d[3] - 1), l2 = 0:(d[2] - 1), l1 = 0:(d[1] - 1)
            i = _phonopy_index(pa, (l1, l2, l3), d, ncell)
            j = _phonopy_index(pb, map(mod, (l1, l2, l3) .+ R, d), d, ncell)
            for α = 1:3, β = 1:3
                Φ[3 * (i - 1) + α, 3 * (j - 1) + β] += T[α, β]
            end
        end
    end
    folded == 0 || @warn "write_phonopy: dim = $d is too small for $(folded) " *
                         "force-constant entries — their lattice shifts wrap onto " *
                         "shifts already present and were summed. This is the " *
                         "aliasing a supercell of this size genuinely has; pass a " *
                         "larger dim (the default $(_phonopy_auto_dim(fcs)) avoids it)."

    isdir(dir) || mkpath(dir)
    pos = joinpath(dir, "POSCAR")
    fcp = joinpath(dir, "FORCE_CONSTANTS")
    open(io -> _write_poscar(io, crystal, perm, comment), pos, "w")
    open(io -> _write_fc(io, Φ, nsup), fcp, "w")
    return (; poscar = pos, force_constants = fcp, dim = d, n_super = nsup)
end

# The smallest odd box that holds every shift without wraparound: shifts run over
# [-m, m] per axis, and 2m + 1 distinct residues are exactly what that needs.
function _phonopy_auto_dim(fcs::ForceConstantSet)::NTuple{3,Int}
    m = [0, 0, 0]
    for (atoms_shifts, _) in fcs.constants
        s = atoms_shifts[2]
        R = s[2] - s[1]
        for k = 1:3
            m[k] = max(m[k], abs(R[k]))
        end
    end
    return (2m[1] + 1, 2m[2] + 1, 2m[3] + 1)
end

@inline _phonopy_index(a::Int, l::NTuple{3,Int}, d::NTuple{3,Int}, ncell::Int)::Int =
    (a - 1) * ncell + (l[1] + d[1] * (l[2] + d[2] * l[3])) + 1

# POSCAR order: species blocks in order of first appearance, atoms within a block in
# their `Crystal` order. `sortperm` with `by = first appearance` is stable, so the
# within-block order is deterministic and matches what a reader would expect.
function _species_grouped_perm(crystal::Crystal)::Vector{Int}
    order = Dict{Int,Int}()
    for s in crystal.species
        haskey(order, s) || (order[s] = length(order) + 1)
    end
    return sortperm(crystal.species; by = s -> order[s])
end

function _write_poscar(io::IO, crystal::Crystal, perm::Vector{Int},
                       comment::AbstractString)
    println(io, isempty(comment) ? "written by SLCE.write_phonopy" : comment)
    println(io, "1.0")
    A = crystal.lattice.vectors                  # columns are aᵢ; POSCAR wants rows
    for i = 1:3
        @printf(io, "  %22.16f %22.16f %22.16f\n", A[1, i], A[2, i], A[3, i])
    end
    labels = String[]
    counts = Int[]
    for a in perm
        lab = crystal.species_labels[crystal.species[a]]
        if isempty(labels) || labels[end] != lab
            push!(labels, lab)
            push!(counts, 0)
        end
        counts[end] += 1
    end
    println(io, "  " * join(labels, " "))
    println(io, "  " * join(counts, " "))
    println(io, "Direct")
    for a in perm
        f = @view crystal.frac_positions[:, a]
        @printf(io, "  %22.16f %22.16f %22.16f\n", f[1], f[2], f[3])
    end
    return nothing
end

# phonopy's FORCE_CONSTANTS, full form: the two atom counts, then one 3 × 3 block per
# ordered pair, each preceded by its 1-based indices.
function _write_fc(io::IO, Φ::Matrix{Float64}, nsup::Int)
    @printf(io, "%4d %4d\n", nsup, nsup)
    for i = 1:nsup, j = 1:nsup
        @printf(io, "%4d %4d\n", i, j)
        for α = 1:3
            @printf(io, "  %22.15f %22.15f %22.15f\n", Φ[3 * (i - 1) + α, 3 * (j - 1) + 1],
                    Φ[3 * (i - 1) + α, 3 * (j - 1) + 2],
                    Φ[3 * (i - 1) + α, 3 * (j - 1) + 3])
        end
    end
    return nothing
end
