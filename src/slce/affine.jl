# Affine displacement fields, and the finite-ω rotational-invariance diagnostic
# built on them (design record §12 gate (q); docs/design-notes.md "Layer 2 — affine
# invariance").
#
# WHY THIS NEEDS ITS OWN EVALUATION PATH. `predict_energy(model, e, u)` resolves the
# displacement of a cluster site as `u[:, atom]`, so the only fields it can express
# are CELL-PERIODIC. A rigid rotation is not one: the atom at `R + L` moves by
# `(O − I)(R + L)`, which grows with the lattice vector `L`, and the same is true of
# any homogeneous strain. Translation — the ASR's operation — is the one affine field
# that IS periodic, which is exactly why the ASR could be tested through the ordinary
# predictors and rotation cannot.
#
# What is well defined per unit cell is the energy of the cluster instances anchored
# in the home cell — precisely the `SALC.members` list the periodic evaluator already
# sums over — with the affine field resolved at each member site's own equilibrium
# position, image shift included. That is `affine_energy`. At `M = 0` it reduces to
# the periodic path member by member and term by term, which is the bit-identity gate
# the test suite pins: the machinery below is a re-indexing of the evaluator, not a
# second implementation of it.

"""
    affine_energy(model, e, M; origin = zeros(3), base = nothing) -> Float64

Energy (eV per unit cell) of `model` under the **affine displacement field**

    u(R) = M · (R − origin) + base[:, atom(R)],

where `R` is the equilibrium Cartesian position of a cluster site *including its
periodic image shift* and `base` is an ordinary cell-periodic `3 × n_atoms` field
(`nothing` means zero). `e` is the `3 × n_atoms` spin configuration, or `nothing`
for a lattice-only model.

This is the evaluation path [`predict_energy`](@ref) cannot express. The periodic
predictors read a site's displacement as `u[:, atom]`, so a field that varies from
image to image — a rigid rotation `M = O − I`, a homogeneous strain `M = ε` — is
outside their domain. The energy returned here is that of the cluster instances
anchored in the home cell, the same `members` sum the periodic evaluator performs;
with `M = 0` the two agree exactly:

```julia
affine_energy(model, e, zeros(3, 3); base = u) == predict_energy(model, e, u)
```

The `origin` shifts the field by the constant `−M·origin`, i.e. by a rigid
translation, so on a model that satisfies the acoustic sum rule
(`asr_residual(model) ≈ 0`) the result does not depend on it. That independence is
a *test* of the ASR, not an assumption: on an unconstrained model the origin moves
the answer.

Intended uses are diagnostics — [`rotational_residual`](@ref) is built on this — and
strain response. It is not a fitting path: the strain channel is not a model
coordinate (design record §9a/§9b).

See also [`rotational_residual`](@ref), [`predict_energy`](@ref),
[`asr_residual`](@ref).
"""
function affine_energy(model::SLCEModel, e::AbstractMatrix{<:Real},
                       M::AbstractMatrix{<:Real};
                       origin::AbstractVector{<:Real} = SVector(0.0, 0.0, 0.0),
                       base::Union{Nothing,AbstractMatrix{<:Real}} = nothing)::Float64
    nat = n_atoms(model.basis.crystal)
    u0 = base === nothing ? zeros(Float64, 3, nat) : base
    _validate_config_pair(model, e, u0)
    size(M) == (3, 3) ||
        throw(DimensionMismatch("affine map M is $(size(M, 1)) × $(size(M, 2)); " *
                                "expected 3 × 3"))
    all(isfinite, M) || throw(ArgumentError("affine map M has non-finite entries"))
    length(origin) == 3 ||
        throw(DimensionMismatch("origin has length $(length(origin)); expected 3"))
    all(isfinite, origin) || throw(ArgumentError("origin has non-finite entries"))
    Mm = SMatrix{3,3,Float64}(M)
    o = SVector{3,Float64}(origin)
    A = model.basis.crystal.lattice.vectors
    frac = model.basis.crystal.frac_positions
    ss = model.basis.salc_basis.salcs
    scratch = SALCScratch()
    # Per-member local views: site `i` of the member occupies column `i`, so the
    # evaluator kernel's `atoms[slots[i].site]` indirection resolves to `i` through
    # the identity map. Same kernel, same loop order, per-site displacements.
    eloc = Matrix{Float64}(undef, 3, 4)
    uloc = Matrix{Float64}(undef, 3, 4)
    idmap = collect(1:4)
    val = model.j0
    @inbounds for k in eachindex(model.jphi)
        s = ss[k]
        scale = (4π)^(count(has_spin, s.decors) / 2)
        total = 0.0
        for m in s.members
            nsite = length(m.atoms)
            if size(uloc, 2) < nsite
                eloc = Matrix{Float64}(undef, 3, nsite)
                uloc = Matrix{Float64}(undef, 3, nsite)
                idmap = collect(1:nsite)
            end
            for i = 1:nsite
                a = m.atoms[i]
                sh = m.shifts[i]
                R = A * SVector{3,Float64}(frac[1, a] + sh[1], frac[2, a] + sh[2],
                                           frac[3, a] + sh[3])
                ui = Mm * (R - o)
                uloc[1, i] = ui[1] + u0[1, a]
                uloc[2, i] = ui[2] + u0[2, a]
                uloc[3, i] = ui[3] + u0[3, a]
                eloc[1, i] = e[1, a]
                eloc[2, i] = e[2, a]
                eloc[3, i] = e[3, a]
            end
            for t in m.terms
                total += _eval_term_mixed(t.folded, t.slots, idmap, eloc, uloc, scratch)
            end
        end
        val += model.jphi[k] * (scale * total)
    end
    return val
end

affine_energy(model::SLCEModel, ::Nothing, M::AbstractMatrix{<:Real}; kwargs...) =
    affine_energy(model, _no_spins(model), M; kwargs...)

# The generator of a right-handed rotation about the unit axis `n`: `W·v = n × v`,
# and `exp(ω W) = I + sin(ω) W + (1 − cos(ω)) W²` (Rodrigues, exact because
# `W³ = −W`). Written out rather than deferred to the matrix `exp` so the axis
# convention is readable at the one site that fixes it.
function _rotation_generator(axis)::SMatrix{3,3,Float64,9}
    length(axis) == 3 ||
        throw(DimensionMismatch("axis has length $(length(axis)); expected 3"))
    n = SVector{3,Float64}(axis[1], axis[2], axis[3])
    all(isfinite, n) || throw(ArgumentError("rotation axis has non-finite entries"))
    nn = norm(n)
    nn > 0 || throw(ArgumentError("rotation axis is the zero vector"))
    n = n / nn
    return @SMatrix [0.0 -n[3] n[2]; n[3] 0.0 -n[1]; -n[2] n[1] 0.0]
end

_rotation_matrix(W::SMatrix{3,3,Float64,9}, omega::Float64) =
    SMatrix{3,3,Float64,9}(I) + sin(omega) * W + (1 - cos(omega)) * (W * W)

# The six symmetric strain directions, Frobenius-normalized. Used only to set the
# scale of `rotational_residual`; the choice is the isotropic-in-Voigt-space one
# (three normal, three shear, equal weight), so the reference is a property of the
# model and the magnitude, never of a preferred axis.
const _STRAIN_DIRECTIONS = let s = 1 / sqrt(2)
    (@SMatrix([1.0 0 0; 0 0 0; 0 0 0]), @SMatrix([0.0 0 0; 0 1 0; 0 0 0]),
     @SMatrix([0.0 0 0; 0 0 0; 0 0 1]), @SMatrix([0.0 s 0; s 0 0; 0 0 0]),
     @SMatrix([0.0 0 0; 0 0 s; 0 s 0]), @SMatrix([0.0 0 s; 0 0 0; s 0 0]))
end

# The three energy changes a rigid rotation by `omega` about `axis` produces: the
# lattice half (displacements only), the spin half (directions only), and the joint
# operation. `E0` and the reference scale come along because every caller needs them.
# One place resolves the geometry so the two public diagnostics cannot drift apart.
function _rotation_pieces(model::SLCEModel, e::AbstractMatrix{<:Real}, omega::Real,
                          axis, origin::AbstractVector{<:Real},
                          u0::Union{Nothing,AbstractMatrix{<:Real}})
    isfinite(omega) || throw(ArgumentError("omega must be finite; got $omega"))
    W = _rotation_generator(axis)
    O = _rotation_matrix(W, Float64(omega))
    Mrot = O - SMatrix{3,3,Float64,9}(I)
    nat = n_atoms(model.basis.crystal)
    base = u0 === nothing ? zeros(Float64, 3, nat) : Matrix{Float64}(u0)
    E0 = affine_energy(model, e, zero(SMatrix{3,3,Float64,9}); origin, base)
    dU = affine_energy(model, e, Mrot; origin, base = O * base) - E0
    dS = affine_energy(model, O * e, zero(SMatrix{3,3,Float64,9}); origin, base) - E0
    dJ = affine_energy(model, O * e, Mrot; origin, base = O * base) - E0
    # Reference: affine fields of the SAME Frobenius size that are real deformations.
    # RMS over the six strain directions rather than one of them, so a model that
    # happens to be soft along the compared direction cannot flatter or inflate the
    # ratio.
    nrm = norm(Mrot)
    acc = 0.0
    for S in _STRAIN_DIRECTIONS
        d = affine_energy(model, e, nrm * S; origin, base) - E0
        acc += d * d
    end
    return (dU = dU, dS = dS, dJ = dJ, ref = sqrt(acc / length(_STRAIN_DIRECTIONS)))
end

"""
    rotational_residual(model, e; omega = 0.05, axis = (0, 0, 1), origin = zeros(3),
                        u0 = nothing) -> Float64

How much energy `model` charges for **rotating the lattice rigidly by the finite
angle `omega`** (radians) about `axis`, as a fraction of what a genuine deformation
of the same size costs. Zero for a rotationally invariant model; `≈ 1` means the
model cannot tell a rotation from a strain. This is the package's rotational-
invariance diagnostic (design record §12 gate (q)).

Site displacements become `u(R) = (O − I)(R − origin) + O·u0`, evaluated through
[`affine_energy`](@ref) — spin directions are held fixed, so this measures the
lattice half `𝓡_U E` alone. The reference scale is the RMS response to the six
Frobenius-normalized symmetric (strain) directions of the same magnitude
`‖O − I‖_F`, which makes the result dimensionless and — whenever the two responses
share their leading power of `ω`, the generic case — independent of `omega` to
leading order. Where they do not, the ratio still carries the answer through its
`ω`-scaling; see the traps below.

**A diagnostic, not a constraint.** Rotational invariance is *not* imposed anywhere
in the package: the SALC projection is invariant under the crystal's point group,
which says nothing about a continuous rigid rotation, and the only affine condition
realized as a constraint is the translational one ([`asr_residual`](@ref),
`build_asr`). The Born–Huang rotational conditions are independent of both the ASR
and the pair-exchange (Huang) conditions — see `docs/design-notes.md` and Gazis &
Wallis, *Phys. Rev.* **151**, 578 (1966).

**Why the angle must be finite.** With `F = ∂E/∂u|₀` and `Φ` the harmonic force
constants,

    ΔE(ω) = ω·F·(Wd) + ω²·[½·F·(W²d) + ½·(Wd)ᵀΦ(Wd)] + O(ω³),

while linearizing the rotation to `u = ωWd` reproduces every term except
`½ω²·F·(W²d)`. On a model whose forces follow the bond directions the linear test
returns *exactly* zero — `d·(Wd) = 0` for antisymmetric `W` — and stays zero however
badly the model violates invariance. The finite-`ω` contraction is the content.

**Reading the number in a SOC sector.** Zero is the right answer only where the
lattice and spin rotations are separate invariances, i.e. on a SOC-free model. Where
the sector couples them, the lattice's rotational response *transfers* to the spin
channel (`𝓡_U E = −𝓡_S E`) instead of vanishing, and a nonzero value here is that
transfer rather than an error. [`rotation_transfer_residual`](@ref) is the statement
that holds in both cases.

**Two traps when reading the number.** A truncated model is never *exactly*
invariant — the rotational condition ties order `n` to order `n + 1` — so a fit to a
genuinely invariant potential lands on a residual that **vanishes with `ω`** rather
than on zero (measured `~10⁻³` and falling, against `~1.7` and flat for a random
ASR-feasible model of the same basis). The decay, not the value at one `ω`, is what
separates truncation from violation. And a model's on-site (1-body) force content is
attached to the *home-cell representative* of each atom: two descriptions of the same
crystal differing only in which periodic image is "home" fit identical periodic data
equally well and have residuals ~10³ apart at a stressed reference (measured 1349×). Periodic training
data cannot choose between them — this can.

A model with no displacement content has no affine response at all; the residual is
then `0.0` by convention (and `Inf` in the degenerate case of a rotation response
with a vanishing reference scale).

See also [`rotation_transfer_residual`](@ref), [`affine_energy`](@ref),
[`asr_residual`](@ref), [`force_constants`](@ref).
"""
function rotational_residual(model::SLCEModel, e::AbstractMatrix{<:Real};
                             omega::Real = 0.05, axis = (0, 0, 1),
                             origin::AbstractVector{<:Real} = SVector(0.0, 0.0, 0.0),
                             u0::Union{Nothing,AbstractMatrix{<:Real}} = nothing)::Float64
    p = _rotation_pieces(model, e, omega, axis, origin, u0)
    p.ref == 0.0 && return p.dU == 0.0 ? 0.0 : Inf
    return abs(p.dU) / p.ref
end

rotational_residual(model::SLCEModel, ::Nothing; kwargs...) =
    rotational_residual(model, _no_spins(model); kwargs...)

"""
    rotation_transfer_residual(model, e; omega = 0.05, axis = (0, 0, 1),
                               origin = zeros(3), u0 = nothing) -> Float64

How much of the lattice and spin rotational responses **fail to cancel** when the
crystal is rotated as one rigid body — lattice *and* spins — by the finite angle
`omega` about `axis`:

    |ΔE(lattice + spins)| / (|ΔE(lattice)| + |ΔE(spins)|).

`0` means the two halves cancel exactly, which is the physical statement in every
sector: a rigid rotation of the whole crystal costs nothing. `1` means they do not
cancel at all. Unlike [`rotational_residual`](@ref) this needs no external energy
scale — the two halves scale it themselves — so it stays meaningful where the
lattice half is *supposed* to be nonzero.

**Use it where the sector couples the channels.** On a SOC-free model — and on a
lattice-only one — the spin half is identically zero, so this ratio degenerates to
`1` whenever the lattice half is nonzero and to `0` when it is not: it reports
*whether* the model is invariant but says nothing about how far off it is.
[`rotational_residual`](@ref) is the graded measure there.

This is the implementation-level form of the SOC rotation law `𝓡_U E = −𝓡_S E`
(design record §12 gate (r)): in a SOC sector the rotation rules are transferred
between channels, not absent. Note the two halves must be able to meet at the same
order in `ω` — the lattice half starts at `ω·F·(Wd)`, so a basis with no
degree-1 displacement content has `F ≡ 0` and *cannot* cancel a spin response at
that order whatever its coefficients are. A residual pinned near `1` with a small
`|ΔE(lattice)|` is that truncation artifact, not a fit failure.

Returns `0.0` only when both halves vanish exactly. A model with no displacement
content but nonzero single-ion anisotropy returns `1.0` — its spin response has
nothing to cancel against, which is the honest answer rather than a degenerate one.

See also [`rotational_residual`](@ref), [`affine_energy`](@ref).
"""
function rotation_transfer_residual(model::SLCEModel, e::AbstractMatrix{<:Real};
                                    omega::Real = 0.05, axis = (0, 0, 1),
                                    origin::AbstractVector{<:Real} = SVector(0.0, 0.0,
                                                                             0.0),
                                    u0::Union{Nothing,AbstractMatrix{<:Real}} = nothing)::Float64
    p = _rotation_pieces(model, e, omega, axis, origin, u0)
    scale = abs(p.dU) + abs(p.dS)
    scale == 0.0 && return p.dJ == 0.0 ? 0.0 : Inf
    return abs(p.dJ) / scale
end

rotation_transfer_residual(model::SLCEModel, ::Nothing; kwargs...) =
    rotation_transfer_residual(model, _no_spins(model); kwargs...)
