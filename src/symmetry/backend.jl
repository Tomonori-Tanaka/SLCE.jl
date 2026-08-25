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

`tol` is the backend's symprec: a **Cartesian distance in Å**, not a fractional one.
Every in-tree comparison against it — matching an atom to its image, deciding whether
two translations name the same operation — is made in Å, so the same `tol` means the
same physical slack in a primitive cell and in a supercell of it, and in a skewed cell
along every direction. Tests that ask whether a *matrix* is integral, orthogonal, or
the identity are dimensionless and do not use `tol` at all.
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

# ── tolerances ───────────────────────────────────────────────────────────────────
#
# `tol` is the backend's symprec: a CARTESIAN distance in Å. That is Spglib's own
# definition of the number. Every comparison of `tol` against a POSITION or a
# TRANSLATION is therefore made in Å. Comparing it against a difference of FRACTIONAL
# coordinates instead makes the effective tolerance scale with the cell — the factor
# is the cell edge measured in Å, so already 3× in a 3 Å primitive cell and 24× in an
# 8×8×8 supercell of it — and direction-dependent in an anisotropic one (4× between
# the two axes of a c/a = 4 hexagonal cell).
#
# A MATRIX tolerance is a different question with a different unit. `W` and `RᵀR − I`
# are dimensionless, so `tol` cannot be handed to them directly either; what symprec
# vouches for is a LENGTH, and each of the two quantities converts that length into
# its own dimensionless band. Both bands sit above a floor set by the round-off of
# `R = A W A⁻¹`, which grows with the cell's condition number and is scale free, so
# unlike a fractional threshold it does not drift with the cell size.
const _MATRIX_RTOL = 1e-8
const _MATRIX_FLOOR_MAX = 1e-4

# The round-off floor, capped. A cell conditioned badly enough to need more than
# `_MATRIX_FLOOR_MAX` is not one whose symmetry any tolerance can rescue, and an
# uncapped floor would silently disable every matrix test at once — at a floor of 1 an
# `isapprox(W, I)` marks EVERY operation a pure translation, which wrecks
# `translation_ops` and the primitive-cell recovery that reads it.
_matrix_floor(lattice::Lattice)::Float64 =
    min(_MATRIX_RTOL * norm(lattice.vectors) * norm(lattice.reciprocal),
        _MATRIX_FLOOR_MAX)

# Band on an entry of the FRACTIONAL matrix `W` — its integrality, the identity test,
# and the periodic/aperiodic axis blocks. A deviation `δ` in one entry moves a point of
# the cell by up to `δ·max_k‖a_k‖` Å, so the length `tol` buys `δ ≤ tol/max_k‖a_k‖`.
_fractional_matrix_atol(lattice::Lattice, tol::Real)::Float64 =
    max(_matrix_floor(lattice), tol / maximum(k -> norm(lattice.vectors[:, k]), 1:3))

# Band on the orthogonality residual `RᵀR − I`. Its size is set by the precision of the
# lattice vectors the caller typed, not by round-off: the residual is a relative
# distortion of SQUARED lengths, so a vector carrying an absolute error of `tol` Å shows
# up in it as `2·tol/‖a_k‖` — worst on the SHORTEST vector, which is therefore the one
# the band is written against. Anything tighter refuses cells the backend legitimately
# accepted at the same symprec. Measured on a hexagonal cell (a = 3 Å, c = 5 Å) whose
# `a√3/2` entry is written to a fixed number of decimals: six decimals put the residual
# at 1.2e-7 and five at 2.2e-6, while the lattice errors they imply are 2.1e-7 Å and
# 3.8e-6 Å — both inside a symprec of 1e-5 Å, and both accepted here. Four decimals
# (2.4e-5 Å, outside it) is refused, and comes back as soon as `tol` covers it.
_orthogonality_atol(lattice::Lattice, tol::Real)::Float64 =
    max(_matrix_floor(lattice), 2 * tol / minimum(k -> norm(lattice.vectors[:, k]), 1:3))

# Fold a fractional difference to the half-cell: the minimum image along a periodic
# axis, and the residual after the operation's own integer shift along an aperiodic one
# (callers require that shift to be `round(d)` before they get here). The interval is
# [−1/2, 1/2]; which end a half-integer lands on follows `round`'s ties-to-even rule,
# which no caller here can reach (their bands are ≪ 1/2). Not `_wrap01`
# (`geometry/crystal.jl`), whose convention is the opposite one, [0, 1).
@inline _wrap_half(d::Real)::Float64 = d - round(d)

# Squared Cartesian length (Å²) of a fractional offset.
@inline _cartesian_distance2(A::SMatrix{3,3,Float64,9},
                             df::SVector{3,Float64})::Float64 = sum(abs2, A * df)

# Per-axis fractional bound implied by `tol`: ‖A df‖ ≤ tol forces |df[k]| ≤ tol/d_k
# with `d_k` the interplanar spacing, because df = B r with B = A⁻¹ whose rows are the
# reciprocal vectors. Necessary, not sufficient — it rejects almost every candidate pair
# before the matrix–vector product is paid for. The 1-ulp slack keeps it a pure filter:
# a candidate that would pass the Å test can never be dropped here by the rounding of
# the division.
_frac_bound(lattice::Lattice, tol::Real)::SVector{3,Float64} =
    SVector{3,Float64}(ntuple(k -> tol / interplanar_spacing(lattice, k) * (1 + 1e-12), 3))

# Minimum-image length (Å) of a Cartesian displacement, over an image box proved
# sufficient for it (`_sufficient_range`, `geometry/neighborlist.jl`). Diagnostic paths
# only: an axis-by-axis fold is NOT the minimum image in a skewed cell, and these
# callers run precisely where the deviation is large enough for that to show.
function _min_image_length(lattice::Lattice, delta::SVector{3,Float64})::Float64
    A = lattice.vectors
    brow = SVector{3,Float64}(norm(lattice.reciprocal[1, :]),
                              norm(lattice.reciprocal[2, :]),
                              norm(lattice.reciprocal[3, :]))
    s0 = ntuple(k -> lattice.pbc[k] ? 2 : 0, 3)
    d = _min_image_dist(delta, A, s0)
    s = _sufficient_range(brow, lattice.pbc, d, s0)
    s == s0 || (d = _min_image_dist(delta, A, s))
    return d
end

# The same-species atom nearest to the fractional point `xn`, and that distance in Å.
function _nearest_same_species(crystal::Crystal, xn::SVector{3,Float64},
                               iat::Integer)::Tuple{Int,Float64}
    lattice = crystal.lattice
    x = crystal.frac_positions
    best = Inf
    at = 0
    @inbounds for jat = 1:n_atoms(crystal)
        crystal.species[jat] == crystal.species[iat] || continue
        d = _min_image_length(lattice, lattice.vectors * SVector{3,Float64}(
            xn[1] - x[1, jat], xn[2] - x[2, jat], xn[3] - x[3, jat]))
        d < best && (best = d; at = jat)
    end
    return at, best
end

# Minimum-image distance (Å) between two atoms of the crystal.
_atom_separation(crystal::Crystal, a::Integer, b::Integer)::Float64 =
    (x = crystal.frac_positions;
     _min_image_length(crystal.lattice, crystal.lattice.vectors * SVector{3,Float64}(
         x[1, a] - x[1, b], x[2, a] - x[2, b], x[3, a] - x[3, b])))

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
    _check_symprec(crystal.lattice, tol)
    A = crystal.lattice.vectors
    Ainv = crystal.lattice.reciprocal
    matol = _fractional_matrix_atol(crystal.lattice, tol)
    ops = Vector{SymOp}(undef, length(rotations))
    @inbounds for i in eachindex(rotations)
        W = SMatrix{3,3,Float64}(rotations[i])
        Rcart = A * W * Ainv
        t = translations[i]
        # Snap a component to zero by the DISTANCE it moves an atom, which for a
        # single component is exactly |t_k|·‖a_k‖ Å (`_frac_bound`'s `d_k` answers a
        # different question — the largest fractional component a bounded Cartesian
        # length can have — and is not the scale here). `tol` is a length, so the bare
        # fractional component is not comparable to it.
        tfrac = SVector{3,Float64}(ntuple(
            k -> abs(t[k]) * norm(A[:, k]) >= tol ? Float64(t[k]) : 0.0, 3))
        ops[i] = SymOp(W, Rcart, tfrac, det(Rcart) > 0, isapprox(W, I; atol = matol))
    end
    _validate_ops(ops, crystal.lattice, tol)
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

# `tol` is a length in Å, and two things bound it. It must be positive and finite to
# be a distance at all; and it must stay below half the smallest interplanar spacing,
# because every image search here folds a fractional offset axis by axis (`round`), and
# that fold is the minimum image only while the true offset is well inside the half
# cell. Above that bound `tol` is not a symmetry tolerance any more, whatever it is.
function _check_symprec(lattice::Lattice, tol::Real)
    (isfinite(tol) && tol > 0) || throw(ArgumentError(
        "`tol` must be a finite positive distance in Å (the backend's symprec); " *
        "got $tol"))
    half = minimum(k -> interplanar_spacing(lattice, k), 1:3) / 2
    tol < half || throw(ArgumentError(
        "`tol` = $(Float64(tol)) Å is not below half the smallest interplanar " *
        "spacing ($half Å). At that size the per-axis fold that matches an atom to " *
        "its image is no longer the minimum image, so `tol` has stopped being a " *
        "symmetry tolerance"))
    return nothing
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
translations through `_translations_equal`, which folds mod 1 on all three axes
and so cannot tell a re-seated operation from the representative it replaced.

A fully periodic crystal keeps every operation and never reaches the filter.
"""
function _restrict_to_pbc(crystal::Crystal, ops::Vector{SymOp}, symbol::String,
                          tol::Real)
    pbc = crystal.lattice.pbc
    all(pbc) && return ops, symbol, nothing
    matol = _fractional_matrix_atol(crystal.lattice, tol)
    kept = SymOp[]
    cols = Vector{Int}[]
    ndrop = 0
    for (i, op) in enumerate(ops)
        r = _op_permutation(crystal, op, tol, i, false, matol)
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
    # (`_translations_equal` folds mod 1 on all three axes, so `t_z = 0` and `t_z = 1`
    # are the same operation to it) — that is what `_check_zero_aperiodic_shift` above
    # is for. What it does check is that dropping operations left a group.
    _validate_ops(kept, crystal.lattice, tol)
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

# Are two fractional translations equal modulo a lattice vector? Folded on all three
# axes and compared IN ÅNGSTRÖM — see the tolerance note above.
_translations_equal(A::SMatrix{3,3,Float64,9}, a::SVector{3,Float64},
                    b::SVector{3,Float64}, tol::Real) =
    _cartesian_distance2(A, SVector{3,Float64}(ntuple(k -> _wrap_half(a[k] - b[k]),
                                                    3))) <= tol * tol

# Index of the operation `(W|t)` in `ops` (0 if absent). `bykey` groups operation
# indices by their integer rotation, so only the same-rotation coset is scanned.
function _find_op(ops::Vector{SymOp}, bykey::Dict{SMatrix{3,3,Int,9},Vector{Int}},
                  W::SMatrix{3,3,Int,9}, t::SVector{3,Float64},
                  A::SMatrix{3,3,Float64,9}, tol::Real)::Int
    index = get(bykey, W, nothing)
    index === nothing && return 0
    for i in index
        _translations_equal(A, ops[i].translation_frac, t, tol) && return i
    end
    return 0
end

"""
    _validate_ops(ops, lattice, tol)

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
function _validate_ops(ops::Vector{SymOp}, lattice::Lattice, tol::Real)
    n = length(ops)
    n == 0 && throw(ArgumentError("the operation list is empty; a space group " *
                                  "contains at least the identity"))
    A = lattice.vectors
    # Two DIMENSIONLESS bands, both derived from the length `tol` — see the tolerance
    # note above. They differ because the quantities do: `matol` bounds an entry of the
    # fractional `W`, `otol` bounds a relative distortion of squared lengths.
    matol = _fractional_matrix_atol(lattice, tol)
    otol = _orthogonality_atol(lattice, tol)
    Wint = Vector{SMatrix{3,3,Int,9}}(undef, n)
    for i = 1:n
        W = ops[i].rotation_frac
        maximum(abs, W - round.(W)) <= matol || throw(ArgumentError(
            "symmetry op $i: the fractional rotation is not an integer matrix to " *
            "$matol ($(W)) — a space-group operation maps the lattice onto itself"))
        Wi = SMatrix{3,3,Int,9}(round.(Int, W))
        dW = round(Int, det(Wi))             # exact: Wi is integral
        abs(dW) == 1 || throw(ArgumentError(
            "symmetry op $i: |det W| = $(abs(dW)) ≠ 1 — the operation is not a " *
            "bijection of the lattice"))
        R = ops[i].rotation_cart
        dev = maximum(abs, R' * R - I)
        dev <= otol || throw(ArgumentError(
            "symmetry op $i: the Cartesian rotation is not orthogonal to $otol " *
            "(measured $dev) — the fractional matrix is integral but does not " *
            "preserve this lattice's metric, so it is not a symmetry of the crystal. " *
            "The band is what `tol` = $(Float64(tol)) Å buys on the shortest lattice " *
            "vector, so if the lattice is genuinely this one, write its vectors to " *
            "more decimals or raise `tol` to about " *
            "$(round(dev * minimum(k -> norm(A[:, k]), 1:3) / 2; sigdigits = 2)) Å"))
        Wint[i] = Wi
    end

    bykey = Dict{SMatrix{3,3,Int,9},Vector{Int}}()
    for i = 1:n
        push!(get!(Vector{Int}, bykey, Wint[i]), i)
    end
    for (_, index) in bykey, a in eachindex(index), b in (a + 1):length(index)
        _translations_equal(A, ops[index[a]].translation_frac,
                            ops[index[b]].translation_frac, tol) &&
            throw(ArgumentError(
                "symmetry ops $(index[a]) and $(index[b]) are the same operation modulo " *
                "a lattice translation — duplicates inflate every orbit multiplicity"))
    end

    id = SMatrix{3,3,Int,9}(I)
    _find_op(ops, bykey, id, zero(SVector{3,Float64}), A, tol) == 0 &&
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
        _find_op(ops, bykey, Wi, ti, A, tol) == 0 && throw(ArgumentError(
            "symmetry op $i has no inverse in the operation list — the list is not " *
            "a group"))
    end

    n <= _MAX_CLOSURE_OPS || return nothing
    for i = 1:n, j = 1:n
        W = Wint[i] * Wint[j]
        t = ops[i].rotation_frac * ops[j].translation_frac + ops[i].translation_frac
        _find_op(ops, bykey, W, t, A, tol) == 0 && throw(ArgumentError(
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
# `atol` is the dimensionless matrix band (`_fractional_matrix_atol`), not symprec
# itself: `W` is a fractional rotation, and a length cannot be compared to its entries
# until it has been divided by one.
@inline function _axis_blocks_ok(W::SMatrix{3,3,Float64,9}, pbc::SVector{3,Bool},
                                 atol::Real)::Bool
    @inbounds for k = 1:3, m = 1:3
        pbc[k] == pbc[m] && continue
        abs(W[k, m]) <= atol || return false
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
    A = crystal.lattice.vectors
    tol2 = tol * tol
    frac_bound = _frac_bound(crystal.lattice, tol)
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
            d1 = xn[1] - x[1, jat]
            d2 = xn[2] - x[2, jat]
            d3 = xn[3] - x[3, jat]
            r1 = round(d1)
            r2 = round(d2)
            r3 = round(d3)
            # An aperiodic axis is never wrapped, so the offset there must be exactly
            # this operation's own integer shift, not merely an integer.
            (pbc[1] || r1 == dt[1]) && (pbc[2] || r2 == dt[2]) &&
                (pbc[3] || r3 == dt[3]) || continue
            df = SVector{3,Float64}(d1 - r1, d2 - r2, d3 - r3)
            abs(df[1]) <= frac_bound[1] && abs(df[2]) <= frac_bound[2] &&
                abs(df[3]) <= frac_bound[3] || continue
            if _cartesian_distance2(A, df) <= tol2
                found = jat
                break
            end
        end
        if found == 0
            strict || return nothing
            near, dist = _nearest_same_species(crystal, xn, iat)
            error("map_sym: atom $iat has no image under operation $o: the nearest " *
                  "same-species atom ($near) is $(round(dist; sigdigits = 3)) Å away " *
                  "(minimum image), while `tol` = $(Float64(tol)) Å — the backend's " *
                  "symprec, a Cartesian distance. Symmetrize the coordinates, or " *
                  "raise `tol` above $(round(dist; sigdigits = 2)) Å")
        end
        # Two atoms sharing an image means the operation does not map the crystal onto
        # itself at this tolerance. `SpaceGroup` documents these columns as permutations
        # and the orbit code inverts them, so catch it here.
        if hit[found]
            strict || return nothing
            # `hit[found]` guarantees an earlier atom mapped there, so the search
            # cannot come back empty — `something` is what says so in the type.
            prev = something(findfirst(==(found), view(col, 1:(iat - 1))))
            sep = _atom_separation(crystal, iat, prev)
            error("map_sym: atoms $prev and $iat share the image $found under " *
                  "operation $o: they sit $(round(sep; sigdigits = 3)) Å apart, " *
                  "inside `tol` = $(Float64(tol)) Å, so the column is not a " *
                  "permutation — lower `tol` below $(round(sep; sigdigits = 2)) Å")
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
# image selection and the tie tolerance. `_validate_ops` cannot see this (its
# `_translations_equal` folds mod 1 on every axis, so it reads a re-seated operation and
# the representative it replaced as the same thing), so check it here, on the data
# already in hand.
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
    A = crystal.lattice.vectors
    tol2 = tol * tol
    frac_bound = _frac_bound(crystal.lattice, tol)
    xi = SVector{3,Float64}(x[1, 1], x[2, 1], x[3, 1])
    xn = op.rotation_frac * xi + op.translation_frac
    seeds = SVector{3,Float64}[]
    @inbounds for jat = 1:nat
        species[jat] == species[1] || continue
        r = SVector{3,Float64}(ntuple(k -> round(xn[k] - x[k, jat]), 3))
        df = SVector{3,Float64}(ntuple(k -> xn[k] - x[k, jat] - r[k], 3))
        all(k -> abs(df[k]) <= frac_bound[k], 1:3) || continue
        _cartesian_distance2(A, df) <= tol2 || continue
        dt = SVector{3,Float64}(ntuple(k -> pbc[k] ? 0.0 : r[k], 3))
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
                         strict::Bool, matol::Real)
    z = zero(SVector{3,Float64})
    strict && return _perm_with_shift(crystal, op, tol, o, z, true), z
    _axis_blocks_ok(op.rotation_frac, crystal.lattice.pbc, matol) || return nothing
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
        # The matrix band is unread on the strict path — `_op_permutation` returns
        # before the axis-block test, which is the only thing that reads it — so the
        # value passed here never enters a comparison.
        map_sym[:, o] = first(_op_permutation(crystal, op, tol, o, true, 0.0))
    end
    return map_sym
end
