# Volume grids — `StrainedModels`, the K(ε) container (design record §9a, §14 M5-3).
#
# WHAT THE GRID IS, AND WHY IT IS VOLUME-ONLY. A strained reference generically has a
# lower point group, so per-grid-point bases would have different SALC keys and nothing
# could be interpolated. Isotropic volume strain `ε = ηI` is the one family that preserves
# the full point group, and v0 restricts the grid to it (§9a). The deformation is
# `F = (1 + η)I` in the pinned Biot measure (§9e), so one LINEAR SCALE `s = 1 + η` labels a
# grid point and the cell is `A_i = s_i·A_0` with the fractional coordinates untouched.
#
# THE INVARIANT IS A SIMILARITY STATEMENT, NOT A LABEL. `A_i = s_i·A_0` with unchanged
# fractional positions means: identical space group (a Cartesian rotation `A R_frac A⁻¹` is
# invariant under `A → sA`), identical neighbour topology PROVIDED the cutoffs scale with
# `s`, identical cluster orbits, and therefore an identical `SALCKey` set with identical
# folded tensors — the SALC content is angular, and similarity does not move angles. That
# is why this file can build the basis at an interpolated scale by SURGERY (scale the
# crystal and the cutoffs, keep the SALC basis and the space group) instead of rebuilding
# it, and why the surgery is verified against every user-supplied grid point at
# construction time rather than trusted.
#
# THE FOUR THINGS THE CONSTRUCTOR REFUSES.
#   (1) Cutoffs that do not scale. With an ABSOLUTE cutoff a neighbour shell crosses it as
#       η varies and the key sets diverge silently — §9a's key-stability pin. Both cutoff
#       surfaces must scale: `BasisSpec.cutoff` AND every `SectorRule.cutoff`.
#   (2) A `disp_scale` that moves. Coefficients are interpolated in ONE normalization, so
#       the displacement scale is frozen at its η = 0 value. Asserted now even though
#       `disp_scale ≠ 1` is still refused elsewhere, so it stays true the day that lifts.
#   (3) A different `SALCKey` set. Interpolating coefficient `k` across grid points is
#       meaningful only if `k` labels the same invariant at every point.
#   (4) A different MEMBER structure — the same atoms at the same integer shifts. This is
#       the **home-image condition**, and it is here because of the 2026-07-29 finding that
#       the ε² strain response depends on which image the cluster list calls "home": a grid
#       assembled from externally standardized cells can land on the wrong side of it while
#       satisfying (3), since the key set is blind to the images. (Along a similarity grid
#       the images cannot drift on their own — this catches a grid whose points were built
#       independently rather than scaled.)
#
# THE TWO MEANINGS OF "THE STRAIN DERIVATIVE", WHICH IS THE POINT OF THE GRID. At a grid
# point there are now two of them and they are not the same object:
#   • the INTRA-MODEL INCREMENTAL one — `strain_derivatives(model; order = 1)`, the model's
#     own response to the affine displacement field, with its coefficients held fixed;
#   • the GRID FINITE DIFFERENCE — `grid_strain_derivative`, `dE/dη` taken ALONG the grid,
#     which also carries the drift of the coefficients themselves.
# Their agreement is the acceptance gate of the whole K(ε) design (§14 M5-2): it says the
# expansion captures the strain response through its displacement channel rather than
# through coefficient drift, which is the validity condition for using one grid point at
# finite strain. It catches a wrong shear factor, a mislabelled η and basis truncation in
# one number.

const _SM_GEOM_ATOL = 1e-10
const _SM_SCALE_RTOL = 1e-10

"""
    StrainedModels

A volume grid of fitted models: the same crystal at a set of isotropic linear scales
`s = 1 + η`, each with its own [`SLCEModel`](@ref), plus the interpolation that turns them
into a model at any scale in between.

Construction asserts the invariants that make interpolation meaningful — similarity of the
cells, cutoffs scaled with `s`, a frozen `disp_scale`, an identical `SALCKey` set and an
identical member (atom, shift) structure. See [`model_at`](@ref) and
[`grid_strain_derivative`](@ref).
"""
struct StrainedModels
    scales::Vector{Float64}
    models::Vector{SLCEModel}
    abscissa::Symbol
    degree::Int

    # Inner, so a field-wise call (persistence, a future reader) cannot assemble a grid
    # whose scales are unsorted or whose degree exceeds what the points can support —
    # either would make `_grid_interpolate` return a number rather than fail. The PHYSICAL
    # invariants (similarity, key equality, the gauge) live in the outer constructor
    # because they need the models compared pairwise, which is not a shape check.
    function StrainedModels(scales::Vector{Float64}, models::Vector{SLCEModel},
                            abscissa::Symbol, degree::Int)
        length(scales) == length(models) || throw(DimensionMismatch(
            "StrainedModels: $(length(scales)) scales but $(length(models)) models"))
        length(scales) >= 2 || throw(ArgumentError(
            "a volume grid needs at least 2 points; got $(length(scales))"))
        issorted(scales) && allunique(scales) || throw(ArgumentError(
            "StrainedModels: scales must be sorted and distinct; got $scales"))
        all(>(0), scales) || throw(ArgumentError("scales must be positive; got $scales"))
        abscissa in (:volume, :linear, :logvolume) || throw(ArgumentError(
            "abscissa must be :volume, :linear or :logvolume; got $abscissa"))
        0 <= degree <= length(scales) - 1 || throw(ArgumentError(
            "degree must be between 0 and $(length(scales) - 1) for a " *
            "$(length(scales))-point grid; got $degree"))
        return new(scales, models, abscissa, degree)
    end
end

Base.length(sm::StrainedModels) = length(sm.scales)

function Base.show(io::IO, sm::StrainedModels)
    print(io, "StrainedModels(", length(sm), " points, s ∈ [", round(first(sm.scales);
          sigdigits = 5), ", ", round(last(sm.scales); sigdigits = 5),
          "], abscissa = ", sm.abscissa, ", degree = ", sm.degree, ")")
end

"""
    scales(sm::StrainedModels) -> Vector{Float64}

The grid's linear scales `s = 1 + η`, increasing. The cell at point `i` is `s_i·A₀`.
"""
scales(sm::StrainedModels) = copy(sm.scales)

"""
    volumes(sm::StrainedModels) -> Vector{Float64}

The grid's cell volumes.
"""
volumes(sm::StrainedModels) =
    [abs(det(m.basis.crystal.lattice.vectors)) for m in sm.models]

# The interpolation abscissa. A modelling choice, not a coordinate change (design record
# §9e part 2 / gate (w)) — which is why it is a field and not a hard-coded convention.
function _grid_abscissa(sm::StrainedModels, s::Real)::Float64
    V0 = abs(det(sm.models[1].basis.crystal.lattice.vectors)) / sm.scales[1]^3
    sm.abscissa === :linear && return Float64(s)
    sm.abscissa === :volume && return V0 * s^3
    sm.abscissa === :logvolume && return log(V0 * s^3)
    throw(ArgumentError("unknown abscissa $(sm.abscissa)"))
end

# d(abscissa)/ds — the chain-rule factor the grid derivative needs.
function _grid_dabscissa(sm::StrainedModels, s::Real)::Float64
    V0 = abs(det(sm.models[1].basis.crystal.lattice.vectors)) / sm.scales[1]^3
    sm.abscissa === :linear && return 1.0
    sm.abscissa === :volume && return 3 * V0 * s^2
    sm.abscissa === :logvolume && return 3 / s
    throw(ArgumentError("unknown abscissa $(sm.abscissa)"))
end

"""
    StrainedModels(models, scales; abscissa = :linear, degree = nothing, tol = 1e-8)

Assemble a volume grid from models fitted at isotropically scaled cells.

`scales` is required and not inferred: `s = 1 + η` is measured against the reference the
grid was built around, and nothing in a set of cells says which one that is (the volumes
give only ratios). The ratios ARE checked against the geometry, so a scale labels a real
cell and is not free metadata.

`abscissa` selects what the coefficients are interpolated *in* (`:linear`, `:volume`,
`:logvolume`) — a modelling choice, not a coordinate change, which is why it is recorded on
the object. The default is the **linear scale**, and that is not arbitrary: the map relating
one grid point to another is the re-expansion of §9d, which is *exactly polynomial* in the
affine displacement and hence in `s`, of the same degree as the displacement content. So on
a surface the expansion can represent, coefficients are polynomials in `s` and the
interpolation is exact, while in the volume they are polynomials in `s³` — measured on a
controlled fixture, leave-one-out at the omitted point gives **6e-14 in `:linear` against
2.5e-8 in `:volume`** and 3e-10 in `:logvolume`. Use `:volume` when the quantity you care
about is an equation of state rather than a coupling.

`degree` is the interpolating polynomial's degree, defaulting to `length(models) - 1`
(exact interpolation through every point); a smaller value fits by least squares, which is
the sane choice on a grid large enough for Runge oscillation to matter.

The constructor refuses a grid that cannot be interpolated:

- cells that are not similar (`A_i = s_i·A₀`, fractional coordinates unchanged, same
  species);
- cutoffs that do not scale with `s` — **both** `BasisSpec.cutoff` and every
  `SectorRule.cutoff`, since an absolute cutoff lets a neighbour shell cross it as the
  volume changes and the key sets then diverge silently;
- a `disp_scale` that differs between points (coefficients are interpolated in one
  normalization);
- a different `SALCKey` set, or a different member (atom, shift) structure — the latter is
  the **home-image condition**, which the key set alone cannot see.

`tol` is the relative tolerance for the geometric comparisons.
"""
function StrainedModels(models::AbstractVector{SLCEModel},
                        scales::AbstractVector{<:Real};
                        abscissa::Symbol = :linear,
                        degree::Union{Integer,Nothing} = nothing,
                        tol::Real = 1e-8)
    n = length(models)
    n >= 2 || throw(ArgumentError(
        "a volume grid needs at least 2 points; got $n. A single model is just a model — " *
        "use it directly."))
    abscissa in (:volume, :linear, :logvolume) || throw(ArgumentError(
        "abscissa must be :volume, :linear or :logvolume; got $abscissa"))
    ms = collect(SLCEModel, models)
    ss = _grid_scales(ms, scales, tol)
    p = sortperm(ss)
    ss, ms = ss[p], ms[p]
    allunique(ss) || throw(ArgumentError(
        "grid scales must be distinct; got $ss (two models at the same volume have " *
        "nothing to interpolate between)"))
    deg = degree === nothing ? n - 1 : Int(degree)
    0 <= deg <= n - 1 || throw(ArgumentError(
        "degree must be between 0 and $(n - 1) for a $n-point grid; got $deg"))
    ref = ms[1]
    for i = 2:n
        _grid_check_similar(ref, ms[i], ss[i] / ss[1], tol, i)
    end
    return StrainedModels(ss, ms, abscissa, deg)
end

# Validate the supplied scales against the cells. The volume ratio is a single number and
# cannot itself detect a shear, so this checks the RATIOS here and `_grid_check_similar`
# checks the full `A_i = s_i·A₀` afterwards.
function _grid_scales(ms::Vector{SLCEModel}, scales::AbstractVector{<:Real}, tol::Real)
    V = [abs(det(m.basis.crystal.lattice.vectors)) for m in ms]
    derived = [(V[i] / V[1])^(1 / 3) for i in eachindex(ms)]
    length(scales) == length(ms) || throw(DimensionMismatch(
        "scales has $(length(scales)) entries for $(length(ms)) models"))
    all(>(0), scales) || throw(ArgumentError("scales must be positive; got $scales"))
    rel = [s / first(scales) for s in scales]
    for i in eachindex(ms)
        isapprox(rel[i], derived[i]; rtol = max(tol, _SM_SCALE_RTOL)) ||
            throw(ArgumentError(
            "scales[$i] / scales[1] = $(rel[i]) disagrees with the cell volumes, which " *
            "give $(derived[i]). The scale labels a GEOMETRY; it is not free metadata."))
    end
    return collect(Float64, scales)
end

function _grid_check_similar(a::SLCEModel, b::SLCEModel, s::Real, tol::Real, i::Int)
    ca, cb = a.basis.crystal, b.basis.crystal
    ca.species == cb.species && ca.species_labels == cb.species_labels ||
        throw(ArgumentError(
        "grid point $i has a different atomic basis than point 1 — a volume grid is one " *
        "crystal at several volumes"))
    isapprox(ca.frac_positions, cb.frac_positions; atol = _SM_GEOM_ATOL) ||
        throw(ArgumentError(
        "grid point $i has different fractional coordinates than point 1. The grid " *
        "invariant is a SIMILARITY (A_i = s_i·A₀ with the fractional basis unchanged), " *
        "which is what keeps the space group, the orbits and the SALC keys identical; " *
        "internally relaxed grid points break it, and §9a pins the reference geometry to " *
        "the affinely scaled, internally UNRELAXED one for exactly this reason."))
    A0, Ai = ca.lattice.vectors, cb.lattice.vectors
    isapprox(Ai, s * A0; rtol = max(tol, _SM_SCALE_RTOL)) || throw(ArgumentError(
        "grid point $i is not an isotropic scaling of point 1: A_i ≠ $s·A₀. v0 grids are " *
        "volume-only (design record §9a) — a symmetry-breaking strain needs the explicit " *
        "global strain channel, and a c/a path needs its own path-specific grid."))
    _grid_check_spec(a.basis.spec, b.basis.spec, s, tol, i)
    _grid_check_basis(a.basis, b.basis, i)
    return nothing
end

function _grid_check_spec(sa::BasisSpec, sb::BasisSpec, s::Real, tol::Real, i::Int)
    sa.nbody == sb.nbody && sa.lmax == sb.lmax && sa.pmax == sb.pmax &&
        sa.lsum == sb.lsum && sa.soc == sb.soc || throw(ArgumentError(
        "grid point $i has a different truncation than point 1 (nbody/lmax/pmax/lsum/soc)"))
    sa.disp_scale == sb.disp_scale || throw(ArgumentError(
        "grid point $i has disp_scale = $(sb.disp_scale) against point 1's " *
        "$(sa.disp_scale). The displacement normalization is FROZEN across a grid: " *
        "coefficients can only be interpolated in one normalization, so it keeps " *
        "its η = 0 value at every point."))
    length(sa.cutoff) == length(sb.cutoff) || throw(ArgumentError(
        "grid point $i has $(length(sb.cutoff)) cutoff matrices against point 1's " *
        "$(length(sa.cutoff))"))
    for b in eachindex(sa.cutoff)
        _grid_cutoff_scaled(sa.cutoff[b], sb.cutoff[b], s, tol, i,
                          "BasisSpec.cutoff[$b]")
    end
    length(sa.sector_rules) == length(sb.sector_rules) || throw(ArgumentError(
        "grid point $i has $(length(sb.sector_rules)) sectors against point 1's " *
        "$(length(sa.sector_rules))"))
    for r in eachindex(sa.sector_rules)
        ra, rb = sa.sector_rules[r], sb.sector_rules[r]
        ra.spin_mode == rb.spin_mode && ra.spin_ls == rb.spin_ls &&
            ra.spin_nsites == rb.spin_nsites && ra.spin_lmax == rb.spin_lmax &&
            ra.spin_lsum == rb.spin_lsum && ra.disp_degree == rb.disp_degree &&
            ra.sites == rb.sites && ra.soc == rb.soc || throw(ArgumentError(
            "grid point $i's sector $r differs from point 1's in something other " *
            "than its cutoff"))
        _grid_cutoff_scaled(ra.cutoff, rb.cutoff, s, tol, i, "sector $r's cutoff")
    end
    return nothing
end

function _grid_cutoff_scaled(ca::AbstractMatrix, cb::AbstractMatrix, s::Real, tol::Real,
                           i::Int, what::AbstractString)
    size(ca) == size(cb) || throw(ArgumentError(
        "grid point $i: $what has size $(size(cb)) against point 1's $(size(ca))"))
    isapprox(cb, s .* ca; rtol = max(tol, _SM_SCALE_RTOL), atol = _SM_GEOM_ATOL) ||
        throw(ArgumentError(
            "grid point $i: $what is not scaled by $s. Cutoffs on a volume grid must be " *
            "expressed in units of the (strained) d_NN — with an ABSOLUTE cutoff a " *
            "neighbour shell crosses it as the volume changes, the lists differ, " *
            "and the SALC key sets diverge with nothing to say so (design record §9a's " *
            "key-stability pin). Both surfaces scale: `BasisSpec.cutoff` and every " *
            "`SectorRule.cutoff`."))
    return nothing
end

function _grid_check_basis(ba::SLCEBasis, bb::SLCEBasis, i::Int)
    ka, kb = ba.salc_basis.keys, bb.salc_basis.keys
    ka == kb || throw(ArgumentError(
        "grid point $i has a different SALC key set than point 1 ($(length(kb)) vs " *
        "$(length(ka)) keys" * (length(ka) == length(kb) ? ", same count" : "") *
        "). Interpolating coefficient k across the grid is meaningful only if it " *
        "same invariant at every point; the usual cause is a cutoff that did not scale."))
    for (sa, sb) in zip(ba.salc_basis.salcs, bb.salc_basis.salcs)
        length(sa.members) == length(sb.members) || throw(ArgumentError(
            "grid point $i: SALC $(sa.key) has $(length(sb.members)) members against " *
            "point 1's $(length(sa.members))"))
        for (ma, mb) in zip(sa.members, sb.members)
            _grid_check_gauge(sa, ma, mb, i)
            ma.atoms == mb.atoms && ma.shifts == mb.shifts || throw(ArgumentError(
                "grid point $i: SALC $(sa.key) has a member at different images than " *
                "point 1 (atoms $(mb.atoms), shifts $(mb.shifts) vs $(ma.atoms), " *
                "$(ma.shifts)). This is the HOME-IMAGE condition, separate from key " *
                "equality: a term's content is anchored at whichever image the cluster " *
                "list calls home, and a grid assembled from independently standardized " *
                "cells can move that while keeping every key. Build the grid by SCALING " *
                "one reference cell."))
        end
    end
    return nothing
end

# The SALC GAUGE must also agree, and this is not pedantry: the projector fixes each
# invariant only up to a sign/normalization, and if one grid point's basis came out with a
# flipped sign for one SALC, its coefficient would be interpolated against the others'
# with the wrong sign — silently, since every key still matches and every fit is still
# perfect at its own point. Along a similarity grid the construction is deterministic and
# the angular content is identical, so the tensors agree bitwise; the tolerance is here for
# the ordering of a distance comparison, not for a different gauge.
function _grid_check_gauge(salc, ma, mb, i::Int)
    length(ma.terms) == length(mb.terms) || throw(ArgumentError(
        "grid point $i: SALC $(salc.key) has a member with $(length(mb.terms)) terms " *
        "against point 1's $(length(ma.terms))"))
    for (ta, tb) in zip(ma.terms, mb.terms)
        isapprox(ta.folded, tb.folded; rtol = 1e-12, atol = 1e-14) || throw(ArgumentError(
            "grid point $i: SALC $(salc.key) has a member whose folded tensor differs " *
            "from point 1's. Keys and images match, so this is the SALC GAUGE: the " *
            "projector " *
            "fixes each invariant only up to a sign and normalization, and interpolating " *
            "coefficient k across points whose basis functions differ by a sign is " *
            "silently wrong — every key matches and every fit is perfect at its own " *
            "point. Build the " *
            "grid by SCALING one reference cell rather than constructing each point."))
    end
    return nothing
end

"""
    model_at(sm::StrainedModels, s::Real) -> SLCEModel

The model at linear scale `s = 1 + η`, with the coefficients interpolated across the grid
in `sm.abscissa` and the basis built by scaling the reference one.

At a grid point this returns that point's own coefficients to within the interpolation's
own accuracy (exactly, for the default `degree = length(sm) - 1`). Between points it is an
interpolation and outside them an extrapolation — which is not refused, because a small
step past the end of a grid is a legitimate thing to do, but is nobody's idea of a
validated model.

The basis is produced by **surgery**: the cell and both cutoff surfaces are scaled, and the
space group and the SALC basis are carried over unchanged, which is exactly what the
similarity invariant licenses (a Cartesian rotation is invariant under `A → sA`, and the
SALC content is angular). That the surgery reproduces the user's own grid bases is checked
at construction time, not assumed here.
"""
function model_at(sm::StrainedModels, s::Real)::SLCEModel
    s > 0 || throw(ArgumentError("scale must be positive; got $s"))
    j0, jphi = _grid_interpolate(sm, s, 0)
    return SLCEModel(_scaled_basis(sm.models[1].basis, s / sm.scales[1]), j0, jphi)
end

"""
    grid_strain_derivative(sm::StrainedModels, s::Real; spins = nothing) -> Float64

`dE_cell/dη` at scale `s = 1 + η`, taken **along the grid**: the derivative of the
interpolated coefficients, evaluated at `u = 0` and the spin configuration `spins`.

This is the second of the two meanings of "the strain derivative" on a grid. The first is
the **intra-model incremental** one — `strain_derivatives(model_at(sm, s); spins,
order = 1)`, whose trace is the same quantity computed inside a single model with its
coefficients held fixed. Their agreement is the acceptance gate of the whole K(ε) design
(design record §14, M5-2): it says the expansion captures the strain response through its
displacement channel rather than through coefficient drift, which is the condition under
which one grid point may be used at finite strain. A disagreement is not noise — it is
basis truncation, a mislabelled `η`, or a wrong shear factor.

`spins` follows the same rule as everywhere else: required for a basis with spin content,
omitted only for a lattice-only one.
"""
function grid_strain_derivative(
        sm::StrainedModels, s::Real;
        spins::Union{AbstractMatrix{<:Real},Nothing} = nothing)::Float64
    s > 0 || throw(ArgumentError("scale must be positive; got $s"))
    e = _resolve_spins(sm.models[1], spins, "the grid strain derivative")
    dj0, djphi = _grid_interpolate(sm, s, 1)
    # d/dη, NOT d/ds. `η` is always measured from the reference the derivative is taken at
    # — the same convention `strain_derivatives` uses, since a model knows only its own
    # reference — and the total scale is `s(1 + η)`, so `dE/dη = s·dE/ds`. Dropping this
    # factor is precisely the "mislabelled η" this gate exists to catch: it is 1 at the
    # unstrained point and off by the strain everywhere else, which is exactly the shape of
    # error that looks like agreement on a first look.
    dj0 *= s
    djphi = djphi .* s
    # `predict_energy` on the DERIVATIVE coefficients: the energy at u = 0 is linear in
    # (j0, jphi) with the spin factors as the (scale-independent) coefficients, so
    # differentiating the model is differentiating its coefficients.
    dmodel = SLCEModel(_scaled_basis(sm.models[1].basis, s / sm.scales[1]), dj0, djphi)
    nat = n_atoms(dmodel.basis.crystal)
    return predict_energy(dmodel, e, zeros(Float64, 3, nat))
end

# Polynomial interpolation of (j0, jphi) in the chosen abscissa, differentiated `order`
# times with respect to the LINEAR SCALE s (the chain rule through the abscissa is applied
# here, so callers never see it).
function _grid_interpolate(sm::StrainedModels, s::Real, order::Int)
    n = length(sm)
    d = sm.degree
    x = [_grid_abscissa(sm, si) for si in sm.scales]
    # Center and scale the abscissa: a Vandermonde in raw volumes of a few hundred Å³ is
    # hopelessly conditioned by degree 3, and the answer would be silently wrong rather
    # than obviously so.
    x0 = sum(x) / n
    w = maximum(abs, x .- x0)
    w = w == 0 ? 1.0 : w
    z = (x .- x0) ./ w
    V = [z[i]^(j - 1) for i = 1:n, j = 1:(d + 1)]
    B = Matrix{Float64}(undef, n, length(sm.models[1].jphi) + 1)
    for i = 1:n
        B[i, 1] = sm.models[i].j0
        B[i, 2:end] = sm.models[i].jphi
    end
    C = V \ B                                        # (d+1) × (1 + n_salcs)
    zs = (_grid_abscissa(sm, s) - x0) / w
    if order == 0
        v = [zs^(j - 1) for j = 1:(d + 1)]
        r = transpose(C) * v
        return r[1], r[2:end]
    end
    order == 1 || throw(ArgumentError("only orders 0 and 1 are implemented; got $order"))
    dv = [j == 1 ? 0.0 : (j - 1) * zs^(j - 2) for j = 1:(d + 1)]
    chain = _grid_dabscissa(sm, s) / w
    r = (transpose(C) * dv) .* chain
    return r[1], r[2:end]
end

# The scaled basis, by surgery. Legal exactly because the grid invariant is a similarity:
# `R_cart = A R_frac A⁻¹` is invariant under `A → sA`, so the space group is the same
# object, and the SALC members and folded tensors are angular/combinatorial.
function _scaled_basis(basis::SLCEBasis, s::Real)::SLCEBasis
    # EXACTLY one, not `≈ 1`. The honest test for "no surgery needed" is equality: `isapprox`
    # at its default `rtol = √eps ≈ 1.5e-8` silently returned the REFERENCE cell — and the
    # reference cutoff surfaces — for a requested scale up to ~1e-8 away, which then reached
    # `strain_derivatives`' absolute site positions and `magnetoelastic_constants`' volume.
    # The grid's own bookkeeping is exact (`s / sm.scales[1]`), so there is nothing to
    # tolerate here.
    s == 1 && return basis
    cr = basis.crystal
    cr2 = Crystal(Lattice(s .* cr.lattice.vectors; pbc = Tuple(cr.lattice.pbc)),
                  cr.frac_positions, cr.species, cr.species_labels)
    return SLCEBasis(cr2, basis.spacegroup, basis.salc_basis, _scaled_spec(basis.spec, s))
end

function _scaled_spec(spec::BasisSpec, s::Real)::BasisSpec
    rules = [SectorRule(r.spin_mode, copy(r.spin_ls), r.spin_nsites, r.spin_lmax,
                        r.spin_lsum, r.disp_degree, r.sites, r.soc, s .* r.cutoff)
             for r in spec.sector_rules]
    # `disp_scale` is deliberately NOT scaled: it is the frozen displacement normalization
    # the whole grid's coefficients live in.
    return BasisSpec(spec.nbody, copy(spec.lmax), copy(spec.pmax), copy(spec.lsum),
                     [s .* c for c in spec.cutoff], spec.soc, rules, spec.disp_scale,
                     copy(spec.species_labels))
end
