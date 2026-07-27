"""
    NeighborPair(i, j, shift, offset, distance)

A directed neighbor pair: atom `i` in the home cell and atom `j` in the periodic
image translated by integer lattice `shift` (cartesian `offset =
lattice.vectors * shift`), separated by `distance` (Å).

The integer `shift` (the inter-site lattice translation `R`) is retained so that
reciprocal-space / spin-spiral design rows — which need `R` to form the phase
factor `e^{i q·R}` — remain constructible later without re-deriving the geometry.
"""
struct NeighborPair
    i::Int
    j::Int
    shift::SVector{3,Int}
    offset::SVector{3,Float64}
    distance::Float64
end

"""
    NeighborList(cutoff, pairs, tol)

All directed neighbor pairs within `cutoff` (Å), built by
[`build_neighbor_list`](@ref), together with the relative same-distance tolerance
`tol` the list was built with.

`tol` travels *with* the list because it is not only a build-time detail: every
consumer that re-decides "is this edge inside a radius" — `candidate_clusters`'s
edge admissibility and the per-sector re-admission in `build_salc_basis` — must
use the same band, or a degenerate shell is admitted by one and split by the
other. Reading the field is the only way those stay in step when a caller passes
a non-default `tol`.
"""
struct NeighborList
    cutoff::Float64
    pairs::Vector{NeighborPair}
    tol::Float64

    function NeighborList(cutoff::Real, pairs::Vector{NeighborPair}, tol::Real)
        # `cutoff` is the largest radius the list was BUILT with, so 0 is legitimate
        # (an all-zero per-species matrix admits nothing) — only NaN is not.
        cutoff >= 0 || throw(ArgumentError("`cutoff` must be ≥ 0; got $cutoff"))
        0 <= tol < 1 ||
            throw(ArgumentError("`tol` is a *relative* band and must lie in [0, 1); " *
                                "got $tol"))
        return new(Float64(cutoff), pairs, Float64(tol))
    end
end

"""
    AbstractImageSelection

How periodic images of an atom pair are admitted into the neighbor list — the
choice that fixes what a finite supercell can resolve. Concrete subtypes:
[`MinimumImage`](@ref) (plain periodic boundary conditions) and [`AllImages`](@ref)
(the generalized-Bloch / spin-spiral seam).
"""
abstract type AbstractImageSelection end

"""
    MinimumImage()

Plain periodic boundary conditions: for each atom pair, keep **only the
minimum-image displacement(s)** — the representative inside the Wigner–Seitz cell
of the (super)lattice — with ties on the WS boundary (the `L/2` faces, edges, and
`(L/2, L/2, L/2)`-type corners) kept as distinct members. This is the only set a
finite supercell can resolve: a *farther* periodic image of the same atom carries
the *same spin*, so the interaction with it is not independent from the
minimum-image one (the design-matrix columns would be collinear). The radial cutoff
trims the minimum-image set and may be `Inf` to keep the whole WS cell (every
resolvable pair). This is the default for SCE fitting.
"""
struct MinimumImage <: AbstractImageSelection end

"""
    AllImages()

Keep **every** periodic image within the cutoff, each with its lattice translation
`R` distinguished (the directed enumeration of [`build_neighbor_list`](@ref)). For
plain-PBC supercell fitting this over-counts beyond `L/2` (it admits aliases of
shorter bonds), so it is *not* the fitting default; it is the representation a
**generalized-Bloch / spin-spiral** extension needs, where the phase `e^{i q·R}`
resolves images a single supercell cannot. Requires a finite cutoff.
"""
struct AllImages <: AbstractImageSelection end

# Relative tolerance for "same distance" decisions (minimum-image ties at the WS
# boundary and the radial-cutoff edge). Far above Float64 round-off (~1e-16) and far
# below any physical shell spacing, so a degenerate shell is never split by a ULP.
const _SAME_DIST_RTOL = 1e-8

"""
    build_neighbor_list(crystal, cutoff, selection; tol = $(_SAME_DIST_RTOL), search = 2)
        -> NeighborList

Neighbor list under an [`AbstractImageSelection`](@ref). [`AllImages`](@ref)
reproduces the two-argument [`build_neighbor_list`](@ref) (every directed image
within `cutoff`); [`MinimumImage`](@ref) keeps, per atom pair, the minimum-image
displacement(s) with WS-boundary ties within relative `tol`, trimmed to `cutoff`
(which may be `Inf`). `search` is the per-periodic-axis image half-range scanned to
locate the minimum image (with positions wrapped to `[0,1)`, `2` covers the WS cell
and its boundary ties for non-pathological cells).

`cutoff` is either one radius for every pair or a symmetric per-**species**-pair
matrix (Å, indexed like `crystal.species`); a pair `(i, j)` is then admitted
against `cutoff[species[i], species[j]]`, with `Inf` = no cutoff for that pair
(`MinimumImage` only) and `0` excluding it. The stored `NeighborList.cutoff` is
the largest radius the list was built with.
"""
build_neighbor_list(crystal::Crystal, cutoff::Real, ::AllImages;
                    tol::Real = _SAME_DIST_RTOL, search::Integer = 2)::NeighborList =
    (isfinite(cutoff) ||
         throw(ArgumentError("AllImages needs a finite cutoff; got $cutoff " *
                             "(use MinimumImage for the full Wigner–Seitz cell)"));
     build_neighbor_list(crystal, cutoff; tol = tol))

# Shared validation for the per-species-pair matrix methods.
function _check_cutoff_matrix(crystal::Crystal, M::AbstractMatrix{<:Real})
    nkd = length(crystal.species_labels)
    size(M) == (nkd, nkd) ||
        throw(ArgumentError("cutoff matrix is $(size(M)) for $nkd species"))
    all(v -> !isnan(v) && v >= 0, M) ||
        throw(ArgumentError("cutoff matrix entries must be ≥ 0 Å or Inf"))
    M == M' || throw(ArgumentError("cutoff matrix must be symmetric"))
    return nothing
end

function build_neighbor_list(crystal::Crystal, cutoff::AbstractMatrix{<:Real},
                             ::AllImages; tol::Real = _SAME_DIST_RTOL,
                             search::Integer = 2)::NeighborList
    _check_cutoff_matrix(crystal, cutoff)
    all(isfinite, cutoff) ||
        throw(ArgumentError("AllImages needs finite cutoffs; the matrix has Inf " *
                            "entries (use MinimumImage for the full Wigner–Seitz cell)"))
    return _build_nl_allimages(crystal, Float64.(cutoff), Float64(tol))
end

build_neighbor_list(crystal::Crystal, cutoff::AbstractMatrix{<:Real}, ::MinimumImage;
                    tol::Real = _SAME_DIST_RTOL, search::Integer = 2)::NeighborList =
    (_check_cutoff_matrix(crystal, cutoff);
     _build_nl_minimage(crystal, Float64.(cutoff), Float64(tol), Int(search)))

# Minimum distance of a displacement `δ` over the image box `±s` (per axis).
@inline function _min_image_dist(δ::SVector{3,Float64}, A::SMatrix{3,3,Float64,9},
                                 s::NTuple{3,Int})::Float64
    best = Inf
    @inbounds for n1 = -s[1]:s[1], n2 = -s[2]:s[2], n3 = -s[3]:s[3]
        d = norm(δ + A * SVector{3,Float64}(n1, n2, n3))
        d < best && (best = d)
    end
    return best
end

# Per-axis image half-range provably sufficient to contain the minimum image and
# its ties: the minimum-image vector `g = δ + A·n` with `|g| = best` obeys
# `|nᵈ| ≤ ‖bᵈ‖·best + |δᵈ_frac| ≤ ‖bᵈ‖·best + 1`. Any upper bound on `best` (a
# too-small box can only overshoot `best`) merely enlarges the range, so one
# refinement off the first scan is always enough — guards heavily skewed /
# non-reduced cells where a fixed `±2` box would miss the true minimum.
@inline _sufficient_range(brow::SVector{3,Float64}, pbc::SVector{3,Bool}, best::Float64,
                          s::NTuple{3,Int})::NTuple{3,Int} =
    ntuple(d -> pbc[d] ? max(s[d], ceil(Int, brow[d] * best) + 1) : 0, 3)

function build_neighbor_list(crystal::Crystal, cutoff::Real, ::MinimumImage;
                             tol::Real = _SAME_DIST_RTOL, search::Integer = 2)::NeighborList
    (cutoff > 0 && !isnan(cutoff)) || throw(ArgumentError("`cutoff` must be positive"))
    nkd = length(crystal.species_labels)
    return _build_nl_minimage(crystal, fill(Float64(cutoff), nkd, nkd), Float64(tol),
                              Int(search))
end

function _build_nl_minimage(crystal::Crystal, cutmat::Matrix{Float64}, rtol::Float64,
                            search::Int)::NeighborList
    lat = crystal.lattice
    A = lat.vectors
    nat = n_atoms(crystal)
    sp = crystal.species
    cart = cartesian_positions(crystal)
    brow = SVector{3,Float64}(norm(lat.reciprocal[1, :]), norm(lat.reciprocal[2, :]),
                              norm(lat.reciprocal[3, :]))
    s0 = ntuple(d -> lat.pbc[d] ? search : 0, 3)
    pairs = NeighborPair[]
    @inbounds for i = 1:nat
        ri = SVector{3,Float64}(cart[1, i], cart[2, i], cart[3, i])
        for j = 1:nat
            # Plain-PBC self-pairs carry the same spin on both ends (`eᵢ`), so an
            # `(i, i+R)` term is a single-site function (a constant for `Lf = 0`, a
            # 1-body alias otherwise), never an independent pair — drop it. (For the
            # generalized-Bloch `AllImages` seam the spiral makes it independent.)
            i == j && continue
            cut = cutmat[sp[i], sp[j]]           # this pair's species-pair radius
            cut > 0 || continue
            δ = SVector{3,Float64}(cart[1, j], cart[2, j], cart[3, j]) - ri
            # pass 1: minimum image distance, with the image box grown to a range
            # that is provably sufficient for this (skewed-cell-safe) pair
            best = _min_image_dist(δ, A, s0)
            s = _sufficient_range(brow, lat.pbc, best, s0)
            s == s0 || (best = _min_image_dist(δ, A, s))
            # minimum image beyond the radial cutoff ⇒ pair contributes nothing
            # (the tie band is relative to this pair's own radius, so a degenerate
            # shell at a species-pair-specific cutoff is never split)
            (isfinite(best) && best <= cut * (1 + rtol)) || continue
            # pass 2: emit every image tied with the minimum (the WS-boundary
            # multiplicity), each as its own directed member
            thr = best * (1 + rtol)
            for n1 = -s[1]:s[1], n2 = -s[2]:s[2], n3 = -s[3]:s[3]
                offset = A * SVector{3,Float64}(n1, n2, n3)
                d = norm(δ + offset)
                d <= thr && push!(pairs, NeighborPair(i, j, SVector{3,Int}(n1, n2, n3),
                                                      offset, d))
            end
        end
    end
    return NeighborList(maximum(cutmat), pairs, rtol)
end

"""
    build_neighbor_list(crystal, cutoff; tol = $(_SAME_DIST_RTOL)) -> NeighborList

Enumerate every directed atom pair `(i, j, shift)` with interatomic distance
`≤ cutoff` (Å), generalizing a fixed image grid to a cutoff-driven one. This is the
[`AllImages`](@ref) enumeration; for plain-PBC SCE fitting use the three-argument
form with [`MinimumImage`](@ref).

Pair admission here is exact (`d ≤ cutoff`, no band). `tol` is recorded on the
returned list for the *downstream* same-distance decisions — cluster-edge
admissibility and per-sector re-admission — which need one shared band; see
[`NeighborList`](@ref).

# Notes
Along each periodic direction `d` the image range is `N_d = ceil(cutoff / spacingᵈ)`
with `spacingᵈ = 1/‖row_d(reciprocal)‖` the interplanar spacing; non-periodic
directions use no images. With fractional coordinates wrapped to `[0, 1)`, the
projection bound `|n_d + Δf_d| ≤ cutoff/spacingᵈ` guarantees this box encloses
every in-cutoff image (valid for triclinic cells too). Self-pairs (`i == j` at zero
shift) are excluded; both `(i,j,R)` and `(j,i,−R)` appear (directed list).
"""
function build_neighbor_list(crystal::Crystal, cutoff::Real;
                             tol::Real = _SAME_DIST_RTOL)::NeighborList
    cutoff > 0 || throw(ArgumentError("`cutoff` must be positive"))
    isfinite(cutoff) || throw(ArgumentError("`cutoff` must be finite"))
    nkd = length(crystal.species_labels)
    return _build_nl_allimages(crystal, fill(Float64(cutoff), nkd, nkd), Float64(tol))
end

function _build_nl_allimages(crystal::Crystal, cutmat::Matrix{Float64},
                             rtol::Float64)::NeighborList
    lat = crystal.lattice
    A = lat.vectors
    nat = n_atoms(crystal)
    sp = crystal.species
    # The image box must enclose every in-cutoff image of every pair, so it is
    # sized from the largest species-pair radius; each pair is then admitted
    # against its own radius inside. N_d = ceil(cut/spacingᵈ) = ceil(cut·‖b_d‖);
    # zero on non-periodic axes.
    cmax = maximum(cutmat)
    nrange = ntuple(d -> lat.pbc[d] ? ceil(Int, cmax * norm(lat.reciprocal[d, :])) : 0, 3)
    cart = cartesian_positions(crystal)
    pairs = NeighborPair[]
    @inbounds for i = 1:nat
        ri = SVector{3,Float64}(cart[1, i], cart[2, i], cart[3, i])
        for j = 1:nat
            cut2 = cutmat[sp[i], sp[j]]^2
            cut2 > 0 || continue
            rj0 = SVector{3,Float64}(cart[1, j], cart[2, j], cart[3, j])
            for n1 = -nrange[1]:nrange[1], n2 = -nrange[2]:nrange[2], n3 = -nrange[3]:nrange[3]
                (i == j && n1 == 0 && n2 == 0 && n3 == 0) && continue
                offset = A * SVector{3,Float64}(n1, n2, n3)
                d2 = sum(abs2, rj0 + offset - ri)
                if d2 <= cut2
                    push!(pairs, NeighborPair(i, j, SVector{3,Int}(n1, n2, n3),
                                              offset, sqrt(d2)))
                end
            end
        end
    end
    return NeighborList(cmax, pairs, rtol)
end
