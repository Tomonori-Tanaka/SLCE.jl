# The ε-LINEAR magnetoelastic deliverables (design record §7, §12 gate (u)).
#
# WHY ε-LINEAR IS ITS OWN TIER. Two independent arguments put the v0 clamped-ion
# deliverables at first order in ε and nowhere else. (i) §13 risk 2: ε-linear content is
# Seth–Hill measure-independent unconditionally, while second-order elastic constants
# coincide across measures only at a stress-free reference — and in a spin–lattice model
# the reference stress is a function of the magnetic state, so a cell can be stress-free
# for at most one of them. (ii) The 2026-07-29 strain finding: the acoustic sum rule buys
# origin independence only where the affine field is periodic, which is `ε = 0` and hence
# order 1. Everything in this file is therefore order 1, and everything in it is safe.
#
# WHAT IS PINNED HERE. The B₁/B₂ ambiguities are NOT the strain measure — they are tensor
# vs engineering shear (a factor 2 on B₂), `α²` vs `α² − 1/3` (which shifts B₁ against the
# volume-magnetostriction constant), `Σ_{i<j}` vs `Σ_{i≠j}` (another factor 2), the overall
# sign of `E_me`, and clamped vs relaxed ion. None of them was pinned anywhere in the
# package before this file; gate (e2) fixes the *span* of the magnetoelastic block and says
# in its own comment that it fixes no normalization. The pin, stated once and gated in
# `test/unit/test_magnetoelastic.jl` against a closed form written out by hand:
#
#     E_me / V = B₁ Σ_i ε_ii (α_i² − 1/3) + 2 B₂ Σ_{i<j} ε_ij α_i α_j
#
# with ε the TENSOR strain (engineering γ = 2ε is an I/O view only, §9e), α the unit
# magnetization direction, and E_me an energy DENSITY — so B₁ and B₂ carry energy/volume.
#
# HOW THE CONSTANTS ARE EXTRACTED, AND WHY BY PROJECTION. The cell's ε-linear response
# `D⁽¹⁾(α) = ∂E_cell/∂ε` is exact (`strain_derivatives`), and the pinned form says what it
# must look like as a function of α:
#
#     D⁽¹⁾_ij(α) = D⁰_ij + V·B₁·δ_ij (α_i² − 1/3) + V·B₂·(1 − δ_ij) α_i α_j
#
# where `D⁰` is everything the magnetization direction does not touch (the reference
# stress, any `l = 0` spin content). Reading B₁ and B₂ off two or three well-chosen
# directions would work for a model that is exactly of this form and would silently return
# a number for one that is not. Projecting a whole direction set onto it instead costs one
# small least-squares solve and yields the thing that matters: `residual`, the fraction of
# the α-dependent ε-linear response the two-constant cubic form does NOT explain. A
# non-cubic crystal, `l = 4` spin content, or a magnetoelastic sector that does not exist
# all show up there rather than in a plausible-looking pair of constants.

# Directions the projection samples. Deterministic and shaped by what it must separate:
# the three axes and the six face diagonals carry B₁ and B₂ respectively, the four body
# diagonals tie them together, and the six generic directions are what a model with
# content beyond the two-constant form fails on — a high-symmetry-only set can be blind to
# an `l = 4` piece that vanishes along every axis and diagonal.
const _ME_DIRECTIONS = let
    axes = [SVector(1.0, 0.0, 0.0), SVector(0.0, 1.0, 0.0), SVector(0.0, 0.0, 1.0)]
    faces = [SVector(1.0, 1.0, 0.0), SVector(1.0, -1.0, 0.0), SVector(1.0, 0.0, 1.0),
             SVector(1.0, 0.0, -1.0), SVector(0.0, 1.0, 1.0), SVector(0.0, 1.0, -1.0)]
    bodies = [SVector(1.0, 1.0, 1.0), SVector(1.0, 1.0, -1.0),
              SVector(1.0, -1.0, 1.0), SVector(-1.0, 1.0, 1.0)]
    generic = [SVector(1.0, 2.0, 3.0), SVector(3.0, -1.0, 2.0), SVector(2.0, 3.0, -1.0),
               SVector(1.0, -2.0, 4.0), SVector(4.0, 1.0, -2.0), SVector(-2.0, 4.0, 1.0)]
    SVector{3,Float64}[v / norm(v) for v in vcat(axes, faces, bodies, generic)]
end

# The (i, j) components of the symmetric response, in the order the design matrix rows use.
const _ME_COMPONENTS = ((1, 1), (2, 2), (3, 3), (1, 2), (1, 3), (2, 3))

# Above this the two-constant cubic form is not what the model contains, and B₁/B₂ are a
# projection rather than a decomposition. Loose on purpose: the residual is normalized by
# the α-DEPENDENT part of the response, so 1e-6 is already ~13 significant digits below the
# magnetoelastic signal itself, and anything larger is structure, not roundoff.
const _ME_RESIDUAL_TOL = 1e-6

function _me_signs(model::SLCEModel,
                   signs::Union{AbstractVector{<:Real},Nothing})::Vector{Float64}
    nat = n_atoms(model.basis.crystal)
    signs === nothing && return ones(Float64, nat)
    length(signs) == nat || throw(DimensionMismatch(
        "signs has $(length(signs)) entries; expected one per atom ($nat)"))
    all(s -> abs(abs(s) - 1) <= 1e-12, signs) || throw(ArgumentError(
        "signs must be ±1 — they select each atom's sublattice orientation along the " *
        "common axis α, and the spin factors are evaluated on UNIT directions. A " *
        "non-collinear state has no single α and therefore no B₁/B₂ in this sense."))
    return Float64[s < 0 ? -1.0 : 1.0 for s in signs]
end

"""
    magnetoelastic_constants(model::SLCEModel; signs = nothing, tol = 1e-6)
        -> (; B1, B2, ion, residual, volume)

The **cubic magnetoelastic constants** of `model`, in the pinned convention

```
E_me / V = B₁ Σ_i ε_ii (α_i² − 1/3) + 2 B₂ Σ_{i<j} ε_ij α_i α_j
```

where `ε` is the **tensor** strain (engineering `γ = 2ε` is an I/O view, never the
internal object), `α` the unit magnetization direction, and `E_me` an energy **density**
— so `B₁` and `B₂` carry energy per volume (eV/Å³ for a model fitted in eV and Å;
1 eV/Å³ = 160.21766 GPa).

Every part of that sentence is a convention that differs between papers — the factor 2,
the `Σ_{i<j}` range, the `−1/3` trace subtraction, the overall sign — so it is pinned
here, gated against an independently hand-derived closed form
(`test/unit/test_magnetoelastic.jl`, design record §12 gate (u)), and it is what the
returned numbers mean. Nothing else in the package pins it.

!!! warning "These are CLAMPED-ION constants — `ion = :clamped` rides in the return value"
    The internal-strain relaxation `Λ = −Φ⁻¹Ξ` is not applied: the ions sit where the
    homogeneous strain puts them, rather than relaxing to zero force at fixed `ε`. The
    clamped-vs-relaxed difference is routinely a factor ~2 and can change a sign, so a
    clamped-ion `B₂` compared against experiment is a wrong number, not an approximate
    one. The qualifier is a field of the result and not only prose, so it cannot be
    dropped by quoting `result.B2` alone.

# How they are obtained, and what `residual` means

The cell's exact ε-linear response `D⁽¹⁾(α) = ∂E_cell/∂ε` ([`strain_derivatives`](@ref))
is computed at 19 magnetization directions and projected onto

```
D⁽¹⁾_ij(α) = D⁰_ij + V B₁ δ_ij (α_i² − 1/3) + V B₂ (1 − δ_ij) α_i α_j
```

by least squares, with `D⁰` (the α-independent part: the reference stress and any `l = 0`
spin content) free. `residual` is the norm of what the fit does not explain divided by
the norm of the α-**dependent** part of the response — i.e. the fraction of the model's
magnetoelastic content that is *not* of the two-constant cubic form. It is small only for
a cubic crystal whose spin content stops at `l = 2`; a warning fires above `tol`, and
`B₁`/`B₂` are then the projection of something else onto this form, which is a summary
and not a decomposition.

# Arguments

- `signs`: per-atom `±1` sublattice orientations along the common axis, so
  `spins[:, a] = signs[a] · α`. The default is a ferromagnet. A non-collinear state has
  no single `α` and therefore no `B₁`/`B₂` in this sense — use
  [`strain_derivatives`](@ref) directly.
- `tol`: the `residual` above which the warning fires.

The model must satisfy the acoustic sum rule (inherited from
[`strain_derivatives`](@ref), which throws otherwise) and must carry spin content.

See also [`strain_derivatives`](@ref), [`exchange_strain_derivatives`](@ref).
"""
function magnetoelastic_constants(model::SLCEModel;
                                  signs::Union{AbstractVector{<:Real},Nothing} = nothing,
                                  tol::Real = _ME_RESIDUAL_TOL)
    _basis_has_spin(model.basis) || throw(ArgumentError(
        "magnetoelastic_constants needs a model with spin content: this basis has none, " *
        "so its strain response is the same for every magnetic state and B₁ = B₂ = 0 " *
        "trivially. Use `strain_derivatives` for the lattice-only response."))
    tol >= 0 || throw(ArgumentError("tol must be ≥ 0; got $tol"))
    sg = _me_signs(model, signs)
    nat = n_atoms(model.basis.crystal)
    V = abs(det(model.basis.crystal.lattice.vectors))
    nd = length(_ME_DIRECTIONS)
    nc = length(_ME_COMPONENTS)
    A = zeros(Float64, nd * nc, 8)
    y = zeros(Float64, nd * nc)
    e = zeros(Float64, 3, nat)
    for (d, α) in enumerate(_ME_DIRECTIONS)
        for a = 1:nat
            e[1, a] = sg[a] * α[1]
            e[2, a] = sg[a] * α[2]
            e[3, a] = sg[a] * α[3]
        end
        D = strain_derivatives(model; spins = e, order = 1)
        for (c, (i, j)) in enumerate(_ME_COMPONENTS)
            r = (d - 1) * nc + c
            y[r] = D[i, j]
            A[r, c] = 1.0                                   # the α-independent part D⁰
            if i == j
                A[r, 7] = α[i]^2 - 1 / 3                    # V·B₁
            else
                A[r, 8] = α[i] * α[j]                       # V·B₂
            end
        end
    end
    x = A \ y
    resid = norm(A * x - y)
    # Normalize by the α-DEPENDENT content, not by ‖y‖: the reference stress can be orders
    # of magnitude larger than the magnetoelastic signal and would hide it completely.
    dev = copy(y)
    for c = 1:nc
        rows = c:nc:(nd * nc)
        dev[rows] .-= sum(@view dev[rows]) / nd
    end
    residual = resid / max(norm(dev), eps())
    if residual > tol
        # Deliberately NOT `maxlog`-limited: this reports a property of ONE model, and
        # `maxlog` silences a call site for the whole process — the second model in a
        # sweep would then look cubic because the first one warned (the `fit.jl:180`
        # reasoning).
        @warn "magnetoelastic_constants: the model's ε-linear response is not of the " *
              "two-constant cubic form — a fraction $(residual) of its " *
              "magnetization-dependent part is unexplained by (B₁, B₂). The returned " *
              "constants are the least-squares PROJECTION onto that form, not a " *
              "decomposition of it. Expected causes: a non-cubic crystal (the " *
              "two-constant " *
              "form is cubic-specific), spin content beyond l = 2, or a magnetic state " *
              "whose sublattices do not share one axis."
    end
    return (; B1 = x[7] / V, B2 = x[8] / V, ion = :clamped, residual, volume = V)
end

# ---------------------------------------------------------------------------------------
# The strain derivatives of the bilinear couplings — `dJ/dε`, the other ε-linear
# deliverable (design record §7).
# ---------------------------------------------------------------------------------------
#
# WHAT THIS IS. `bilinear_terms` reads a model's pure-spin `ls = [1,1]` / `ls = [2]`
# content out as one Cartesian matrix per bond and per site. This reads out the ε-LINEAR
# part of the same picture: how each of those matrices moves under a homogeneous strain,
#
#     ∂M_ab^{αβ} / ∂ε_{γδ}      and      ∂A_a^{αβ} / ∂ε_{γδ}
#
# from the terms that carry two `l = 1` spin factors (or one `l = 2`) AND exactly one
# `degree = 1` displacement factor. It is the Bethe–Slater `dJ/dr` of the model, resolved
# per bond and per strain component instead of collapsed onto a bond length, and it is
# what a magnetoelastic Hamiltonian for a spin-dynamics or MC engine wants.
#
# WHY THE PER-BOND SPLIT NEEDS A CHECK AND THE TOTAL DOES NOT. A homogeneous strain moves
# site `s` by `ε·(R_s − origin)`, so a single term's contribution carries that site's
# ABSOLUTE position. The origin cancels out of the total by the acoustic sum rule (which
# `strain_derivatives` enforces), but the sum rule is a statement about the whole model,
# not about one bond: a bond whose own displacement content is not purely relative has an
# `∂M/∂ε` that depends on where the origin was put, and is therefore not a property of the
# bond at all. Symmetry usually rules this out — a bond orbit with a site-swap operation
# admits only the relative combination `u_b − u_a`, which is why the check normally passes
# silently — but "usually" is not "always", so it is MEASURED here, at a probe-shifted
# origin, in the same idiom `strain_derivatives` uses at order ≥ 2.

const _EX_ORIGIN_RTOL = 1e-8

"""
    ExchangeStrainDerivatives

The ε-linear strain derivatives of a model's bilinear couplings, as produced by
[`exchange_strain_derivatives`](@ref).

- `pairs[(a, b, R)][α, β, γ, δ]` is `∂M_ab^{αβ} / ∂ε_{γδ}`, with `M_ab` the bilinear
  matrix of [`bilinear_terms`](@ref) (`E ⊃ e_a' M_ab e_b`, one entry per undirected bond,
  sites canonicalized to `a ≤ b`, `R` the lattice shift from `a` to `b`).
- `onsites[a][α, β, γ, δ]` is `∂A_a^{αβ} / ∂ε_{γδ}` for the single-ion (`ls = [2]`)
  matrices.
- `skipped` names the SALCs whose spin content is not bilinear-representable but that DO
  carry ε-linear displacement content — magnetoelastic coupling this view cannot show.
- `origin` is the strain origin the derivatives were taken about (see
  [`exchange_strain_derivatives`](@ref) on why it is recorded).

`γ, δ` are the strain indices in the **unsymmetrized** convention — the derivative with
respect to a general affine map `M`, so that `Σ_{γδ} (∂M_ab/∂ε_{γδ}) ε_{γδ}` is the change
under `ε`. Symmetrize in `(γ, δ)` if you want the symmetric-strain form.
"""
struct ExchangeStrainDerivatives
    pairs::Dict{Tuple{Int,Int,SVector{3,Int}},Array{Float64,4}}
    onsites::Dict{Int,Array{Float64,4}}
    skipped::Vector{String}
    origin::SVector{3,Float64}
end

function Base.show(io::IO, x::ExchangeStrainDerivatives)
    print(io, "ExchangeStrainDerivatives(", length(x.pairs), " bonds, ",
          length(x.onsites), " sites", isempty(x.skipped) ? "" :
          ", $(length(x.skipped)) skipped", ")")
end

# `R_{1m}(u)` is linear, so its ε-derivative is one number per (m, Cartesian component):
# `∂R_{1m}/∂ε_{γδ} = g[m, γ]·d_δ`. Same monomial coefficients the force constants and the
# ASR builder read, so the three cannot drift apart.
const _EX_L1_GRAD = let g = zeros(3, 3)
    for mi = 1:3
        poly = SolidHarmonics.solid_harmonic_poly(0, 1, mi - 2)
        for γ = 1:3
            g[mi, γ] = _site_derivative(poly, (γ,))
        end
    end
    SMatrix{3,3,Float64,9}(g)
end

# Which bilinear channel a term maps to, and where its slots are. Returns `nothing` for
# anything this view cannot express. Read off the TERM (slots), not the SALC key, because
# the displacement decoration is per slot.
function _ex_classify(t::SALCTerm)
    spin = Int[]
    disp = Int[]
    for i in eachindex(t.slots)
        if t.slots[i].factor.channel == SPIN
            push!(spin, i)
        else
            push!(disp, i)
        end
    end
    # Displacement degrees summing to exactly 1: that is what "ε-linear" means. Every
    # displacement factor has degree `2k + l ≥ 1`, so this also forces a single factor with
    # (k, l) = (0, 1); a degree-2 factor contributes at O(ε²) rather than being
    # unrepresentable.
    isempty(disp) && return nothing
    sum(2 * t.slots[i].factor.k + t.slots[i].factor.l for i in disp) == 1 || return nothing
    ls = sort([t.slots[i].factor.l for i in spin])
    ls == [1, 1] && return (:pair, spin, disp[1])
    ls == [2] && return (:onsite, spin, disp[1])
    return nothing
end

"""
    exchange_strain_derivatives(model::SLCEModel; origin = zeros(3), check_origin = true)
        -> ExchangeStrainDerivatives

The **strain derivatives of the bilinear couplings**: `∂M_ab/∂ε` for every bond and
`∂A_a/∂ε` for every single-ion matrix of [`bilinear_terms`](@ref), from the terms that
carry bilinear (`ls = [1,1]`) or single-ion (`ls = [2]`) spin content together with
exactly one `degree = 1` displacement factor.

This is the model's `dJ/dr` resolved per bond and per strain component — the ε-linear
magnetoelastic coupling in the form a spin Hamiltonian consumes, as opposed to the two
cubic numbers [`magnetoelastic_constants`](@ref) compresses it into.

Contract, with `d_s = R_s − origin` the displaced site's own position:

```
∂M_ab^{αβ}/∂ε_{γδ} = Σ_terms (∂M_ab^{αβ}/∂u_{s γ}) d_{s δ}
```

so `γ, δ` are **unsymmetrized** (the derivative with respect to a general affine map).
Only `degree = 1` displacement content contributes; `degree = 2` factors are `O(ε²)` and
are not "skipped" but simply absent from a first derivative. Pure-spin SALCs are the
couplings themselves, not their derivatives, and are likewise not reported. `skipped`
lists what a user would actually miss: SALCs with ε-linear displacement content whose
spin part is not bilinear-representable (three-body, higher `l`).

!!! warning "The per-bond split needs the bond's own content to be relative"
    A homogeneous strain moves a site by `ε·(R_s − origin)`, so each term carries an
    absolute position. The origin cancels from the TOTAL by the acoustic sum rule, but
    that is a statement about the whole model — a single bond whose displacement content
    is not purely relative (`u_b − u_a`) has an `∂M/∂ε` that depends on where the origin
    was put, and is then not a property of the bond. Bond orbits with a site-swap
    operation admit only the relative combination, so the check normally passes silently;
    when it does not, this function **throws** rather than returning a
    convention-dependent coupling. `check_origin = false` returns it anyway, for
    diagnosis. The total is unaffected either way — use [`strain_derivatives`](@ref).

The model must satisfy the acoustic sum rule (checked, and a throw if not — the same
precondition and the same tolerance as [`strain_derivatives`](@ref)).

See also [`bilinear_terms`](@ref), [`magnetoelastic_constants`](@ref),
[`strain_derivatives`](@ref).
"""
function exchange_strain_derivatives(
        model::SLCEModel;
        origin::AbstractVector{<:Real} = SVector(0.0, 0.0, 0.0),
        check_origin::Bool = true)::ExchangeStrainDerivatives
    length(origin) == 3 ||
        throw(DimensionMismatch("origin has length $(length(origin)); expected 3"))
    all(isfinite, origin) || throw(ArgumentError("origin has non-finite entries"))
    _require_asr(model, "exchange_strain_derivatives")
    o = SVector{3,Float64}(origin[1], origin[2], origin[3])
    out = _ex_derivatives(model, o)
    check_origin && _check_exchange_origin(model, out, o)
    return out
end

function _ex_derivatives(model::SLCEModel, o::SVector{3,Float64})
    pairs = Dict{Tuple{Int,Int,SVector{3,Int}},Array{Float64,4}}()
    onsites = Dict{Int,Array{Float64,4}}()
    skipped = String[]
    A = model.basis.crystal.lattice.vectors
    frac = model.basis.crystal.frac_positions
    ss = model.basis.salc_basis.salcs
    F2 = zeros(Float64, 3, 3, 3)                       # [m₁, m₂, γ] for the pair channel
    F1 = zeros(Float64, 5, 3)                          # [m,  γ]     for the single-ion one
    for k in eachindex(model.jphi)
        jphi = model.jphi[k]
        jphi == 0.0 && continue
        salc = ss[k]
        weight = jphi * (4π)^(count(has_spin, salc.decors) / 2)
        reported = false
        for mem in salc.members
            for t in mem.terms
                cls = _ex_classify(t)
                if cls === nothing
                    # report only content that IS ε-linear: a pure-spin or degree-2 term
                    # is not missing from a first derivative, it does not belong in one
                    if !reported && _ex_is_linear_disp(t) &&
                       any(has_spin, salc.decors)
                        push!(skipped, _unsupported_salc_string(salc.key))
                        reported = true
                    end
                    continue
                end
                kind, spin, id = cls
                sd = t.slots[id].site
                d = A * SVector{3,Float64}(frac[1, mem.atoms[sd]] + mem.shifts[sd][1],
                                           frac[2, mem.atoms[sd]] + mem.shifts[sd][2],
                                           frac[3, mem.atoms[sd]] + mem.shifts[sd][3]) - o
                if kind === :pair
                    i1, i2 = spin[1], spin[2]
                    # order the pair by SITE index; canonical (v4) members sort sites by
                    # (atom, shift), so this is also the `a ≤ b` order `bilinear_terms`
                    # canonicalizes to
                    s1, s2 = t.slots[i1].site, t.slots[i2].site
                    if s1 > s2
                        i1, i2 = i2, i1
                        s1, s2 = s2, s1
                    end
                    fill!(F2, 0.0)
                    for midx in CartesianIndices(t.folded)
                        w = t.folded[midx]
                        w == 0.0 && continue
                        md = midx[id]
                        for γ = 1:3
                            g = _EX_L1_GRAD[md, γ]
                            g == 0.0 && continue
                            F2[midx[i1], midx[i2], γ] += w * g
                        end
                    end
                    key = (mem.atoms[s1], mem.atoms[s2], mem.shifts[s2] - mem.shifts[s1])
                    T = get!(() -> zeros(Float64, 3, 3, 3, 3), pairs, key)
                    for γ = 1:3
                        Mγ = _l1_pair_matrix(@view F2[:, :, γ])
                        for α = 1:3, β = 1:3, δ = 1:3
                            T[α, β, γ, δ] += weight * Mγ[α, β] * d[δ]
                        end
                    end
                else
                    i1 = spin[1]
                    fill!(F1, 0.0)
                    for midx in CartesianIndices(t.folded)
                        w = t.folded[midx]
                        w == 0.0 && continue
                        md = midx[id]
                        for γ = 1:3
                            g = _EX_L1_GRAD[md, γ]
                            g == 0.0 && continue
                            F1[midx[i1], γ] += w * g
                        end
                    end
                    T = get!(() -> zeros(Float64, 3, 3, 3, 3), onsites,
                             mem.atoms[t.slots[i1].site])
                    for γ = 1:3
                        Aγ = _l2_onsite_matrix(@view F1[:, γ])
                        for α = 1:3, β = 1:3, δ = 1:3
                            T[α, β, γ, δ] += weight * Aγ[α, β] * d[δ]
                        end
                    end
                end
            end
        end
    end
    for key in [k for (k, T) in pairs if all(iszero, T)]
        delete!(pairs, key)
    end
    for key in [k for (k, T) in onsites if all(iszero, T)]
        delete!(onsites, key)
    end
    return ExchangeStrainDerivatives(pairs, onsites, skipped, o)
end

# Does this term carry exactly one displacement factor of degree 1? (The "would have
# contributed if its spin part were representable" test behind `skipped`.)
function _ex_is_linear_disp(t::SALCTerm)::Bool
    deg = 0
    nd = 0
    for s in t.slots
        s.factor.channel == DISP || continue
        nd += 1
        deg += 2 * s.factor.k + s.factor.l
    end
    return nd == 1 && deg == 1
end

function _check_exchange_origin(model::SLCEModel, out::ExchangeStrainDerivatives,
                                o::SVector{3,Float64})
    probe = o + model.basis.crystal.lattice.vectors * _STRAIN_ORIGIN_PROBE
    other = _ex_derivatives(model, probe)
    scale = max(_ex_norm(out), eps())
    # Compare in BOTH directions: the zero-pruning can drop a key from one side only, and
    # a coupling that exists at one origin and not at the other is the loudest possible
    # form of the failure this checks for.
    dev = max(_ex_dev(out.pairs, other.pairs), _ex_dev(other.pairs, out.pairs),
              _ex_dev(out.onsites, other.onsites), _ex_dev(other.onsites, out.onsites))
    dev <= _EX_ORIGIN_RTOL * scale && return nothing
    throw(ArgumentError(
        "the per-bond strain derivatives of this model are not origin-independent " *
        "(‖ΔdM/dε‖/‖dM/dε‖ = $(dev / scale) between two origins), so they are not " *
        "properties of the bonds. A homogeneous strain moves each site by ε·(R_s − " *
        "origin), and the origin cancels only from the TOTAL, by the acoustic sum rule — " *
        "which holds for this model, since it got this far. What fails is the per-bond " *
        "split: a bond whose displacement content is not purely relative (u_b − u_a) " *
        "carries an absolute position of its own. The total response is unaffected — use " *
        "`strain_derivatives(model; order = 1)`, or `magnetoelastic_constants` for the " *
        "cubic constants. Pass `check_origin = false` to see the origin-dependent " *
        "couplings anyway, for diagnosis only."))
end

_ex_dev(a::Dict, b::Dict)::Float64 =
    isempty(a) ? 0.0 : maximum(norm(T .- get(() -> zero(T), b, key)) for (key, T) in a)

function _ex_norm(x::ExchangeStrainDerivatives)::Float64
    acc = 0.0
    for (_, T) in x.pairs
        acc += sum(abs2, T)
    end
    for (_, T) in x.onsites
        acc += sum(abs2, T)
    end
    return sqrt(acc)
end
