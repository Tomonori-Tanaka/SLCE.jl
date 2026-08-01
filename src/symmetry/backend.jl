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
    map_sym = _build_map_sym(crystal, ops, tol)
    translation_ops = [i for i in eachindex(ops) if ops[i].is_translation]
    return SpaceGroup(String(symbol), Int(number), ops, map_sym, translation_ops, Float64(tol))
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

# map_sym[iat, op] = the atom that iat maps to under (W|t), matched by minimum-image
# fractional distance within `tol` and the same species.
function _build_map_sym(crystal::Crystal, ops::Vector{SymOp}, tol::Real)::Matrix{Int}
    nat = n_atoms(crystal)
    x = crystal.frac_positions
    species = crystal.species
    tol2 = tol * tol
    map_sym = zeros(Int, nat, length(ops))
    hit = falses(nat)                      # each column must be a *permutation*
    @inbounds for (o, op) in enumerate(ops)
        fill!(hit, false)
        W = op.rotation_frac
        t = op.translation_frac
        for iat = 1:nat
            xi = SVector{3,Float64}(x[1, iat], x[2, iat], x[3, iat])
            xn = W * xi + t
            found = 0
            for jat = 1:nat
                species[jat] == species[iat] || continue
                d2 = 0.0
                for k = 1:3
                    v = abs(xn[k] - x[k, jat]) % 1.0
                    v = min(v, 1.0 - v)
                    d2 += v * v
                end
                if d2 < tol2
                    found = jat
                    break
                end
            end
            found == 0 &&
                error("map_sym: atom $iat has no image under operation $o (tol=$tol)")
            # Two atoms sharing an image means the operation does not map the crystal
            # onto itself at this tolerance. `SpaceGroup` documents these columns as
            # permutations and the orbit code inverts them, so catch it here.
            if hit[found]
                prev = findfirst(==(found), view(map_sym, 1:(iat - 1), o))
                error("map_sym: atoms $prev and $iat share the image $found under " *
                      "operation $o (tol=$tol) — the column is not a permutation")
            end
            hit[found] = true
            map_sym[iat, o] = found
        end
    end
    return map_sym
end
