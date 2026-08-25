"""
    AbstractSymmetryBackend

Strategy for obtaining a crystal's space-group operations. Implement a backend by
subtyping this and defining

    analyze_symmetry(::MyBackend, crystal::Crystal; tol) -> SpaceGroup

Backends only need to supply raw `(rotations, translations, symbol, number)`; a
shared in-tree assembler (`_assemble_spacegroup`) derives Cartesian rotations,
properness, and the atom permutation table `map_sym`, so every backend shares one
convention.
"""
abstract type AbstractSymmetryBackend end

"""
    NoSymmetry()

In-tree fallback backend: returns the trivial P1 group (identity only). Lets the
whole pipeline run with no external dependency; the SALC projector still averages
over `{E} × {time reversal}`.
"""
struct NoSymmetry <: AbstractSymmetryBackend end

"""
    SpglibBackend()

Space-group analysis via Spglib. The type lives in the core package (so it can be
named without Spglib loaded), but `analyze_symmetry(::SpglibBackend, …)` is
provided only when `Spglib` is loaded (`using Spglib`).
"""
struct SpglibBackend <: AbstractSymmetryBackend end

"""
    analyze_symmetry(backend, crystal; tol = 1e-5) -> SpaceGroup

Compute the space group of `crystal` using `backend`.

A backend analyzes the cell as a fully periodic 3D crystal — Spglib is not told about
`Lattice(...; pbc)` and has no way to be. When the crystal declares an aperiodic axis,
the assembled group is intersected with what that declaration allows: operations that
close only through the periodicity along the aperiodic axis are dropped (with a warning
naming how many, and a `" (pbc subgroup)"` suffix on [`SpaceGroup`](@ref)`.symbol` so
the group cannot be mistaken for the one the backend named), and the translations of
the survivors are re-seated on the representative the finite structure actually has, so
no cell shift is ever produced along an axis that has no cells. The result is a
subgroup, so the basis built from it is larger than the fully periodic one, never
short. A fully periodic crystal is untouched.

This happens in the shared assembler (`_assemble_spacegroup`), which every in-tree
backend routes through — a custom backend that builds a [`SpaceGroup`](@ref) itself
gets none of it and is responsible for its own `pbc` handling.
"""
function analyze_symmetry(backend::AbstractSymmetryBackend, crystal::Crystal;
                          tol::Real = 1e-5)::SpaceGroup
    error("analyze_symmetry has no method for $(typeof(backend)); load the backend " *
          "package first (e.g. `using Spglib` for SpglibBackend).")
end

function analyze_symmetry(::NoSymmetry, crystal::Crystal; tol::Real = 1e-5)::SpaceGroup
    return _assemble_spacegroup(crystal, [SMatrix{3,3,Float64}(I)],
                                [SVector{3,Float64}(0, 0, 0)], "P1", 1; tol = tol)
end

"""
    _assemble_spacegroup(crystal, rotations, translations, symbol, number; tol) -> SpaceGroup

Shared assembler: turn raw fractional `rotations`/`translations` into `SymOp`s
(deriving Cartesian rotations, properness, pure-translation flags) and derive
`map_sym` in-tree. Backends call this so they all share one convention.

The assembled set is validated as a group (see `_validate_ops`) before it is
returned.
"""
function _assemble_spacegroup(crystal::Crystal,
                              rotations::AbstractVector,
                              translations::AbstractVector,
                              symbol::AbstractString, number::Integer;
                              tol::Real)::SpaceGroup
    length(rotations) == length(translations) ||
        throw(ArgumentError("rotations and translations must have equal length"))
    A = crystal.lattice.vectors
    Ainv = crystal.lattice.reciprocal
    ops = Vector{SymOp}(undef, length(rotations))
    @inbounds for i in eachindex(rotations)
        W = SMatrix{3,3,Float64}(rotations[i])
        Rcart = A * W * Ainv
        t = translations[i]
        tfrac = SVector{3,Float64}(ntuple(k -> abs(t[k]) >= tol ? Float64(t[k]) : 0.0, 3))
        ops[i] = SymOp(W, Rcart, tfrac, det(Rcart) > 0, isapprox(W, I; atol = tol))
    end
    _validate_ops(ops, tol)
    # `_restrict_to_pbc` has to match every atom under every operation to decide what
    # to keep, so it hands the resulting permutations back rather than making
    # `_build_map_sym` repeat the same `n_ops × n_atoms²` pass. It returns `nothing`
    # for a fully periodic crystal, which it leaves untouched.
    ops, symbol, cols = _restrict_to_pbc(crystal, ops, String(symbol), tol)
    map_sym = cols === nothing ? _build_map_sym(crystal, ops, tol) :
              reduce(hcat, cols; init = Matrix{Int}(undef, n_atoms(crystal), 0))
    translation_ops = [i for i in eachindex(ops) if ops[i].is_translation]
    return SpaceGroup(symbol, Int(number), ops, map_sym, translation_ops, Float64(tol))
end

"""
    _restrict_to_pbc(crystal, ops, symbol, tol) -> (ops, symbol, permutations)

Keep only the operations that are symmetries of the crystal under its DECLARED
periodicity, returning them with a symbol that says so.

Backends analyse the cell as a fully periodic 3D crystal — Spglib is not told about
`Lattice(...; pbc)` and has no way to be. For a slab (`pbc = (true, true, false)`) that
list can therefore contain operations that only close through the artificial
periodicity along the aperiodic axis. They are not harmless: `build_clusters` demands
that the candidate set be closed under the group, while the neighbour list refuses to
emit an image along an aperiodic axis, so such an operation shows up as a closure
failure blamed on image selection and the tie tolerance.

The kept set is a **subgroup**: after re-seating, the aperiodic axes match exactly, so
the condition composes (`g` maps `x_i` to `x_j` exactly, `h` maps `x_j` to `x_l`
exactly, hence `g∘h` maps `x_i` to `x_l` exactly), the identity satisfies it, and an
operation that permutes the atoms bijectively has an inverse that does too. Using a
subgroup never over-reduces: the basis it builds is larger than necessary, never short.

The load-bearing property is checked directly, by
`_check_zero_aperiodic_shift`, and NOT by `_validate_ops` — that one compares
translations through `_tclose`, which folds mod 1 on all three axes and so cannot tell
a re-seated operation from the representative it replaced.

A fully periodic crystal keeps every operation and never reaches the filter.
"""
function _restrict_to_pbc(crystal::Crystal, ops::Vector{SymOp}, symbol::String,
                          tol::Real)
    pbc = crystal.lattice.pbc
    all(pbc) && return ops, symbol, nothing
    kept = SymOp[]
    cols = Vector{Int}[]
    ndrop = 0
    for (i, op) in enumerate(ops)
        r = _op_permutation(crystal, op, tol, i, false)
        if r === nothing
            ndrop += 1
            continue
        end
        # Re-seat the translation on the representative that is the physical
        # operation, so `_site_image` — whose cell shift is `round(W·x_a + t − x_b)` —
        # sees an exact match and a zero shift along the aperiodic axes.
        seated = SymOp(op.rotation_frac, op.rotation_cart,
                       op.translation_frac - r[2], op.is_proper, op.is_translation)
        _check_zero_aperiodic_shift(crystal, seated, r[1], length(kept) + 1)
        push!(kept, seated)
        push!(cols, r[1])
    end
    # `_validate_ops` runs on the kept set whether or not anything was dropped: the
    # re-seated translations are a change on their own. It does NOT see the re-seating
    # (`_tclose` folds mod 1 on all three axes, so `t_z = 0` and `t_z = 1` are the same
    # operation to it) — that is what `_check_zero_aperiodic_shift` above is for. What
    # it does check is that dropping operations left a group.
    _validate_ops(kept, tol)
    # `kept` is returned even when nothing was dropped: the re-seated translations are
    # the point on their own, and leaving them at the backend's mod-1 representative
    # would keep handing `_site_image` a cell shift along an axis that has no cells.
    ndrop == 0 && return kept, symbol, cols
    axes = join((("a", "b", "c")[k] for k = 1:3 if !pbc[k]), ", ")
    @warn "symmetry: $ndrop of $(length(ops)) operations " *
          "reported for the fully periodic cell close only through the periodicity " *
          "along $axes, which this lattice declares aperiodic — dropping them. The " *
          "basis is built from the remaining subgroup, so it is larger than the " *
          "fully periodic one, never short. If the cell is a vacuum-padded slab and " *
          "you meant the 3D group of that padded cell, pass the default " *
          "`pbc = (true, true, true)` (`[structure].pbc = [true, true, true]` in a " *
          "TOML input): the vacuum and the cutoff already keep the images apart. " *
          "`symbol` carries a \" (pbc subgroup)\" suffix while this is in " *
          "effect." group = symbol kept = length(kept)
    return kept, symbol * " (pbc subgroup)", cols
end

# The largest operation count any conventional crystallographic setting can reach:
# 48 point operations × 4 centering translations (F). Anything above it is a
# supercell, whose extra operations are pure translations.
const _MAX_CLOSURE_OPS = 192

# Are two fractional translations equal modulo a lattice vector?
_translations_equal(a::SVector{3,Float64}, b::SVector{3,Float64}, tol::Real) =
    all(k -> (d = abs(a[k] - b[k]) % 1.0; min(d, 1.0 - d) <= tol), 1:3)

# Index of the operation `(W|t)` in `ops` (0 if absent). `bykey` groups operation
# indices by their integer rotation, so only the same-rotation coset is scanned.
function _find_op(ops::Vector{SymOp}, bykey::Dict{SMatrix{3,3,Int,9},Vector{Int}},
                  W::SMatrix{3,3,Int,9}, t::SVector{3,Float64}, tol::Real)::Int
    index = get(bykey, W, nothing)
    index === nothing && return 0
    for i in index
        _translations_equal(ops[i].translation_frac, t, tol) && return i
    end
    return 0
end

"""
    _validate_ops(ops, tol)

Check that an assembled operation list really is a space group of the lattice.

Backends are trusted for *which* group a crystal has, not for handing back a
well-formed one — and downstream code consumes a `SpaceGroup` as a group, not as
a list: `build_clusters` reduces orbits by stabilizer counting and
`build_salc_basis` projects with `(1/|G|) Σ_g`. A set that is merely a plausible
list of matrices therefore does not fail; it silently yields wrong orbit
multiplicities and a projector that is not idempotent. (Measured on a
deliberately non-closed set: `sum(multiplicity) = 8` over `6` candidate
clusters.)

Checked, in increasing cost: each `W` integral with `|det W| = 1` and a Cartesian
image that is orthogonal (i.e. a symmetry of *this* lattice's metric, which
integrality alone does not imply — a shear is integral too); `(I|0)` present; no
two operations equal modulo a lattice translation; the rotation parts closed
under multiplication (always cheap — at most 48 distinct rotations); every
operation's inverse present. The full pairwise `(W|t)` closure runs for up to
`_MAX_CLOSURE_OPS` operations, which covers every conventional setting; beyond
that (supercells) the preceding checks stand in for it.
"""
function _validate_ops(ops::Vector{SymOp}, tol::Real)
    n = length(ops)
    n == 0 && throw(ArgumentError("the operation list is empty; a space group " *
                                  "contains at least the identity"))
    otol = max(Float64(tol), 1e-8)
    Wint = Vector{SMatrix{3,3,Int,9}}(undef, n)
    for i = 1:n
        W = ops[i].rotation_frac
        isapprox(W, round.(W); atol = tol) || throw(ArgumentError(
            "symmetry op $i: the fractional rotation is not an integer matrix " *
            "($(W)) — a space-group operation maps the lattice onto itself"))
        Wi = SMatrix{3,3,Int,9}(round.(Int, W))
        dW = round(Int, det(Wi))             # exact: Wi is integral
        abs(dW) == 1 || throw(ArgumentError(
            "symmetry op $i: |det W| = $(abs(dW)) ≠ 1 — the operation is not a " *
            "bijection of the lattice"))
        R = ops[i].rotation_cart
        maximum(abs, R' * R - I) <= otol || throw(ArgumentError(
            "symmetry op $i: the Cartesian rotation is not orthogonal to $otol — " *
            "the fractional matrix is integral but does not preserve this " *
            "lattice's metric, so it is not a symmetry of the crystal"))
        Wint[i] = Wi
    end

    bykey = Dict{SMatrix{3,3,Int,9},Vector{Int}}()
    for i = 1:n
        push!(get!(Vector{Int}, bykey, Wint[i]), i)
    end
    for (_, index) in bykey, a in eachindex(index), b in (a + 1):length(index)
        _translations_equal(ops[index[a]].translation_frac, ops[index[b]].translation_frac, tol) &&
            throw(ArgumentError(
                "symmetry ops $(index[a]) and $(index[b]) are the same operation modulo " *
                "a lattice translation — duplicates inflate every orbit multiplicity"))
    end

    id = SMatrix{3,3,Int,9}(I)
    _find_op(ops, bykey, id, zero(SVector{3,Float64}), tol) == 0 &&
        throw(ArgumentError("the identity (I|0) is missing from the operation list"))

    rots = Set(Wint)
    for a in rots, b in rots
        a * b in rots || throw(ArgumentError(
            "the rotation parts are not closed under multiplication ($(a) ∘ $(b) is " *
            "absent) — the operation list is not a group"))
    end

    for i = 1:n
        Wi = SMatrix{3,3,Int,9}(round.(Int, inv(ops[i].rotation_frac)))
        ti = -(SMatrix{3,3,Float64}(Wi) * ops[i].translation_frac)
        _find_op(ops, bykey, Wi, ti, tol) == 0 && throw(ArgumentError(
            "symmetry op $i has no inverse in the operation list — the list is not " *
            "a group"))
    end

    n <= _MAX_CLOSURE_OPS || return nothing
    for i = 1:n, j = 1:n
        W = Wint[i] * Wint[j]
        t = ops[i].rotation_frac * ops[j].translation_frac + ops[i].translation_frac
        _find_op(ops, bykey, W, t, tol) == 0 && throw(ArgumentError(
            "the composition of symmetry ops $i ∘ $j is not in the operation list — " *
            "the list is not closed, so orbit multiplicities and the SALC projector " *
            "are both wrong"))
    end
    return nothing
end

# Does `W` keep the periodic and aperiodic axes in separate blocks? An operation that
# sends an in-plane lattice translation into the aperiodic direction (or the reverse)
# does not map the *declared* translation lattice onto itself — there are no lattice
# vectors along an aperiodic axis to receive it — so it cannot be a symmetry however
# well it happens to permute the atoms.
@inline function _axis_blocks_ok(W::SMatrix{3,3,Float64,9}, pbc::SVector{3,Bool},
                                 tol::Real)::Bool
    @inbounds for k = 1:3, m = 1:3
        pbc[k] == pbc[m] && continue
        abs(W[k, m]) <= tol || return false
    end
    return true
end

# The permutation `op` induces when its APERIODIC axes are shifted by the integer
# vector `dt`, or `nothing` when some atom has no image under that shift.
#
# Matching folds mod 1 on every axis to find a candidate, then requires the candidate's
# integer offset along each aperiodic axis to be exactly `dt[k]`. Folding alone is not
# enough there: an aperiodic axis is never wrapped (there is no period to wrap with), so
# a crystal may legitimately list positions spanning more than one cell along it, and
# two atoms an exact cell apart are then indistinguishable to a folded comparison. With
# `z = 0` and `z = 1` that used to make even the IDENTITY fail as "not a permutation".
#
# `strict` is a fully periodic crystal: there are no aperiodic axes, `dt` is vacuous,
# and "no image" / "not a permutation" stay the errors they have always been — nothing
# may be dropped there, so a backend handing back an operation the crystal does not
# have is a fault, not a filter.
function _perm_with_shift(crystal::Crystal, op::SymOp, tol::Real, o::Integer,
                          dt::SVector{3,Float64}, strict::Bool)::Union{Nothing,Vector{Int}}
    nat = n_atoms(crystal)
    x = crystal.frac_positions
    species = crystal.species
    pbc = crystal.lattice.pbc
    tol2 = tol * tol
    col = zeros(Int, nat)
    hit = falses(nat)                      # each column must be a *permutation*
    W = op.rotation_frac
    t = op.translation_frac
    @inbounds for iat = 1:nat
        xi = SVector{3,Float64}(x[1, iat], x[2, iat], x[3, iat])
        xn = W * xi + t
        found = 0
        for jat = 1:nat
            species[jat] == species[iat] || continue
            d2 = 0.0
            ok = true
            for k = 1:3
                d = xn[k] - x[k, jat]
                if !pbc[k] && round(d) != dt[k]
                    ok = false
                    break
                end
                v = abs(d) % 1.0
                d2 += min(v, 1.0 - v)^2
            end
            if ok && d2 < tol2
                found = jat
                break
            end
        end
        if found == 0
            strict || return nothing
            error("map_sym: atom $iat has no image under operation $o (tol=$tol)")
        end
        # Two atoms sharing an image means the operation does not map the crystal onto
        # itself at this tolerance. `SpaceGroup` documents these columns as permutations
        # and the orbit code inverts them, so catch it here.
        if hit[found]
            strict || return nothing
            prev = findfirst(==(found), view(col, 1:(iat - 1)))
            error("map_sym: atoms $prev and $iat share the image $found under " *
                  "operation $o (tol=$tol) — the column is not a permutation")
        end
        hit[found] = true
        col[iat] = found
    end
    return col
end

# The property this whole restriction exists to establish: `_site_image`
# (`src/clusters/orbits.jl`) reads a cluster's cell shift off
# `round.(Int, W·x_a + t − x_b)`, and along an aperiodic axis there are no cells for a
# nonzero one to name — the neighbour list emits none (`nrange[d] = 0` there), so a
# nonzero shift reappears downstream as `build_clusters`' closure assertion, blamed on
# image selection and the tie tolerance. `_validate_ops` cannot see this (its `_tclose`
# folds mod 1 on every axis, so it reads a re-seated operation and the representative it
# replaced as the same thing), so check it here, on the data already in hand.
function _check_zero_aperiodic_shift(crystal::Crystal, op::SymOp, col::Vector{Int},
                                     o::Integer)
    pbc = crystal.lattice.pbc
    all(pbc) && return nothing
    x = crystal.frac_positions
    @inbounds for a in eachindex(col)
        xa = SVector{3,Float64}(x[1, a], x[2, a], x[3, a])
        b = col[a]
        xb = SVector{3,Float64}(x[1, b], x[2, b], x[3, b])
        d = op.rotation_frac * xa + op.translation_frac - xb
        for k = 1:3
            pbc[k] && continue
            round(Int, d[k]) == 0 || error(
                "symmetry op $o maps atom $a to atom $b with a cell shift of " *
                "$(round(Int, d[k])) along aperiodic axis $k — the translation was not " *
                "re-seated on the representative the finite structure has, and " *
                "`_site_image` would hand that shift to a cluster the neighbour list " *
                "cannot build")
        end
    end
    return nothing
end

# The whole-operation shifts along the aperiodic axes worth trying, read off the images
# atom 1 could map to, NEAREST ZERO FIRST. Zero leads whenever it is a candidate, so the
# identity is never re-seated away from `t` by an accident of atom ordering. More than
# one seed exists only when two same-species atoms sit an exact number of cells apart
# along an aperiodic axis with their remaining coordinates equal; every seed is then
# tried, and if none yields a permutation the operation is dropped — the conservative
# direction (a larger basis, never a short one).
function _shift_seeds(crystal::Crystal, op::SymOp,
                      tol::Real)::Vector{SVector{3,Float64}}
    nat = n_atoms(crystal)
    x = crystal.frac_positions
    species = crystal.species
    pbc = crystal.lattice.pbc
    tol2 = tol * tol
    xi = SVector{3,Float64}(x[1, 1], x[2, 1], x[3, 1])
    xn = op.rotation_frac * xi + op.translation_frac
    seeds = SVector{3,Float64}[]
    @inbounds for jat = 1:nat
        species[jat] == species[1] || continue
        d2 = 0.0
        for k = 1:3
            v = abs(xn[k] - x[k, jat]) % 1.0
            d2 += min(v, 1.0 - v)^2
        end
        d2 < tol2 || continue
        dt = SVector{3,Float64}(ntuple(k -> pbc[k] ? 0.0 : round(xn[k] - x[k, jat]), 3))
        dt in seeds || push!(seeds, dt)
    end
    sort!(seeds; by = norm)
    return seeds
end

# The permutation `op` induces under the crystal's DECLARED periodicity, together with
# the integer translation its aperiodic axes must be shifted by, or `nothing` when it is
# not a symmetry there.
#
# Backends report `t` modulo a lattice translation, and along a periodic axis that is
# the whole truth. Along an aperiodic axis it is not: the coset has no lattice to be
# taken modulo, so ONE representative is the physical operation and the others are not.
# A slab centred at `z = 1/2` has a mirror there, and Spglib is entitled to hand it back
# as the mirror at `z = 0` with `t_z = 0` — the same operation modulo `c`, and the only
# one of the two the finite slab does not have. So the test is not "does `t` match
# exactly" but "is there an integer shift of `t` along the aperiodic axes under which
# every atom matches" — ONE shift for the whole operation, which is what makes the
# surviving set a group and forces `_site_image` to produce zero cell shifts there.
function _op_permutation(crystal::Crystal, op::SymOp, tol::Real, o::Integer,
                         strict::Bool)
    z = zero(SVector{3,Float64})
    strict && return _perm_with_shift(crystal, op, tol, o, z, true), z
    _axis_blocks_ok(op.rotation_frac, crystal.lattice.pbc, tol) || return nothing
    for dt in _shift_seeds(crystal, op, tol)
        col = _perm_with_shift(crystal, op, tol, o, dt, false)
        col === nothing || return col, dt
    end
    return nothing
end

# map_sym[iat, op] = the atom that iat maps to under (W|t), for a FULLY PERIODIC
# crystal — the aperiodic path gets its columns straight from `_restrict_to_pbc`, which
# had to build them anyway. `strict = true`: nothing may be dropped here, so a missing
# image or a non-permutation is a fault, not a filter.
function _build_map_sym(crystal::Crystal, ops::Vector{SymOp}, tol::Real)::Matrix{Int}
    map_sym = Matrix{Int}(undef, n_atoms(crystal), length(ops))
    @inbounds for (o, op) in enumerate(ops)
        map_sym[:, o] = first(_op_permutation(crystal, op, tol, o, true))
    end
    return map_sym
end
