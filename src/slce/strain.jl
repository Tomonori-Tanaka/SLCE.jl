# Homogeneous-strain response — the exact ε-derivatives of the cell energy (design
# record §7's strain deliverable tier, §9d's origin-independence discipline, §9e's
# measure pin, §12 gate (t)).
#
# WHY THIS IS EXACT, AND WHY IT IS NOT A FINITE DIFFERENCE OF A FIT. A homogeneous
# strain acts on the cluster sites as the affine field `u_i = ε·(R_i − origin)`, which
# is NOT cell-periodic (`slce/affine.jl` says why at length), so it cannot be reached
# through `predict_energy`. But every displacement site factor `|u|^{2k} R_{lm}(u)` is
# a homogeneous polynomial of degree `2k + l`, so substituting the affine field turns a
# term of total displacement degree `n` into a homogeneous polynomial of degree `n` in
# the entries of `ε`. Hence: the order-`n` strain derivative gets contributions from
# terms of displacement degree EXACTLY `n`, and each contribution is read off the same
# monomial coefficients `force_constants` differentiates —
#
#     ∂ⁿE/∂ε_{α₁β₁}⋯∂ε_{α_nβ_n} = Σ_{s₁…s_n} [∂ⁿE/∂u_{s₁α₁}⋯∂u_{s_nα_n}] d_{s₁β₁}⋯d_{s_nβ_n}
#
# where `s` runs over MEMBER SITES (not atoms) and `d_s = R_s − origin` is that site's
# own equilibrium position, image shift included. The inner bracket is exactly what
# `_fill_fcs_tensor!` computes, which is why this file calls it rather than reimplementing
# it: a change to the displacement kernel moves the evaluator, the ASR matrix, the force
# constants and the strain response together, by construction.
#
# WHY THE ASR IS A HARD PRECONDITION, NOT A CAVEAT. An origin shift `t` maps the affine
# field to `u_i + ε·t`, i.e. adds a uniform translation, so `∂E/∂t = ε : (Σ_i ∇_i E)`.
# The strain response is origin-independent **iff** `Σ_i ∇_i E ≡ 0` — exactly `A·β = 0`.
# Without it the answer is not inaccurate but undefined (the origin is unbounded), so
# every entry point here throws rather than warns, at a tolerance tighter than the 1e-10
# used elsewhere because the strain path weights the residual by |R_i|.
#
# AND WHY THAT IS ENOUGH ONLY AT ORDER 1 (measured 2026-07-29). The ASR is the identity
# `Σ_a ∂E/∂u_a ≡ 0` in the ATOM variables — i.e. on cell-periodic fields. At `ε = 0` the
# affine field IS periodic (it is zero), so the order-1 derivative is origin-independent
# unconditionally, and it measures exactly so on every fixture. At order ≥ 2 the identity
# is needed one order out in `u`, at a field that is NOT periodic: what it would take is
# the stronger SITE-level statement `Σ_slots ∂E/∂u_slot ≡ 0`, and the atom-level
# constraint does not imply it. The mechanism is the home-image gauge already recorded for
# gate (q): a term's content is anchored at whichever image the member list calls "home",
# while its translation partners sit on the images the clusters actually reach, so the
# cancellation is split across cells and the per-cell energy of a home-anchored expansion
# acquires a piece linear in the cell's position. Measured: a 1-D chain whose home image
# IS the bonded partner is origin-independent to 1e-13 at order 2, the SAME crystal
# described with the bond crossing the cell edge is off by a factor 8, and a bcc-like
# two-atom cell by a factor ~25. Since the answer is either exact or wrong by O(1), the
# deliverable MEASURES it: at order ≥ 2 the tensor is recomputed at a probe-shifted origin
# and a disagreement is refused, in the package's recompute-never-trust idiom.

# The ASR tolerance the strain path demands. Tighter than the 1e-10 the rest of the
# package quotes: `Σ_i ∇_i E` enters here multiplied by site positions, so a residual is
# amplified by ~R_cut/d_NN before it reaches the answer (design record §9d).
const _STRAIN_ASR_RTOL = 1e-12

# Origin-consistency at order ≥ 2. The discrepancy is either roundoff or O(1) — there is
# no interesting middle — so the cut is loose on purpose and the probe shift is one fixed
# generic vector in lattice coordinates: the dependence is affine in the shift, so a
# single non-degenerate probe detects it.
const _STRAIN_ORIGIN_RTOL = 1e-8
const _STRAIN_ORIGIN_PROBE = SVector(0.37, -0.61, 0.83)

function _require_asr(model::SLCEModel, what::AbstractString)
    r = asr_residual(model)
    r <= _STRAIN_ASR_RTOL || throw(ArgumentError(
        "$what requires a model that satisfies the acoustic sum rule: " *
        "asr_residual(model) = $r exceeds $(_STRAIN_ASR_RTOL). A homogeneous strain " *
        "displaces site i by ε·(R_i − origin), so shifting the origin by t changes the " *
        "energy by ε : (Σ_i ∇_i E). With Σ_i ∇_i E ≠ 0 there is no origin-independent " *
        "strain response at all — the quantity is undefined, not merely inaccurate, and " *
        "the error grows with the cutoff radius. Fit under the constraint " *
        "(`fit(...; asr = true)`, the default). The tolerance is deliberately tighter " *
        "than the 1e-10 used elsewhere, because the strain path weights the residual " *
        "by |R_i|."))
    return nothing
end

# Interleaved linear index of `out[α₁, β₁, α₂, β₂, …]` from the two order-`n`
# CartesianIndices. Written as index arithmetic rather than tuple splatting because the
# rank `2n` is a runtime value here (`force_constants` has the same shape).
function _strain_linear(c::CartesianIndex, b::CartesianIndex)::Int
    lin = 1
    stride = 1
    @inbounds for p = 1:length(c.I)
        lin += (c[p] - 1) * stride
        stride *= 3
        lin += (b[p] - 1) * stride
        stride *= 3
    end
    return lin
end

# `out[…α_p, β_p…] ← ½(out[…α_p, β_p…] + out[…β_p, α_p…])` for every pair p at once —
# the average over all 2ⁿ subsets of pairs to swap, since the per-pair symmetrizations
# commute. Cross-pair symmetry is automatic (mixed partials) and is not imposed.
function _symmetrize_strain_pairs!(out::Array{Float64}, n::Int)
    src = copy(out)
    nsub = 1 << n
    dig = Vector{Int}(undef, 2n)
    @inbounds for lin in eachindex(out)
        r = lin - 1
        for p = 1:(2n)
            dig[p] = r % 3
            r ÷= 3
        end
        acc = 0.0
        for s = 0:(nsub - 1)
            index = 0
            stride = 1
            for p = 1:n
                a, b = dig[2p - 1], dig[2p]
                if (s >> (p - 1)) & 1 == 1
                    a, b = b, a
                end
                index += a * stride
                stride *= 3
                index += b * stride
                stride *= 3
            end
            acc += src[index + 1]
        end
        out[lin] = acc / nsub
    end
    return out
end

"""
    strain_derivatives(model::SLCEModel; spins = nothing, order = 1, origin = zeros(3),
                       symmetrize = true, check_origin = true) -> Array{Float64}

The exact order-`order` derivative of the **cell energy with respect to homogeneous
strain**, at the model's own reference geometry `ε = 0` and at the spin configuration
`spins`:

```
D^{(n)}_{α₁β₁ … α_nβ_n} = ∂ⁿE_cell / ∂ε_{α₁β₁} ⋯ ∂ε_{α_nβ_n} |_{ε = 0}
```

returned as a rank-`2·order` array indexed `[α₁, β₁, α₂, β₂, …]` — Cartesian pairs
adjacent, so `order = 1` is a 3×3 matrix and `order = 2` a 3×3×3×3 one. The Taylor
series it belongs to carries the usual factorials:

```
E(ε) = E(0) + Σ D⁽¹⁾:ε + (1/2!) Σ D⁽²⁾:ε:ε + …
```

**Strain measure.** Biot / Seth–Hill `m = 1`: the deformation is *defined* as
`F = I + ε` with `ε` symmetric, so the affine substitution `u_i − u_j = ε·d_ij` is
exact rather than a linearization (design record §9e). `order = 1` is
measure-independent across the Seth–Hill family; `order = 2` and beyond are not.

**What the numbers are.** `order = 1` is the reference stress times the cell volume,
`D⁽¹⁾_{αβ} = V σ_{αβ}` — nonzero at a geometry that is not internally relaxed, which is
real physics and not an error. `order = 2` is the **clamped-ion** elastic tensor times
the volume: the internal-strain relaxation `Λ = −Φ⁻¹Ξ` is not applied, and the
clamped-vs-relaxed difference is routinely a factor ~2. Both depend on `spins`, which is
what a spin–lattice expansion is for.

**Which terms contribute.** Only those whose displacement degrees sum to exactly
`order` — each site factor `|u|^{2k} R_{lm}(u)` is homogeneous of degree `2k + l`. So a
model truncated below `order` returns zeros rather than an error, and the ε-linear
magnetoelastic content lives in the degree-1 (relative-displacement) sectors, never in
single-site `p = 2` monomials, which enter only at `O(ε²)`.

**The acoustic sum rule is a precondition, not a caveat.** An origin shift `t` adds the
uniform translation `ε·t` to the field, changing the energy by `ε : (Σ_i ∇_i E)`. This
function therefore **throws** — it does not warn — when `asr_residual(model)` exceeds
`$(_STRAIN_ASR_RTOL)`, a tolerance deliberately tighter than the 1e-10 used elsewhere
because the strain path weights the residual by `|R_i|`.

!!! warning "At `order ≥ 2` the sum rule is necessary but not sufficient"
    The ASR is the identity `Σ_a ∂E/∂u_a ≡ 0` in the **atom** variables, i.e. on
    cell-periodic fields. At `ε = 0` the affine field *is* periodic, so `order = 1` is
    origin-independent unconditionally. One order further out the field is not periodic,
    and what would be needed is the stronger per-**site** identity — which the atom-level
    constraint does not imply. The gap is the home-**image gauge**: a term's content is
    anchored at whichever image the cluster list calls "home", its translation partners
    sit on the images the clusters actually reach, and when those differ the cancellation
    is split across cells, leaving the per-cell energy with a piece linear in the cell's
    position. It is not a small error — measured exact on a chain whose home image *is*
    the bonded partner, and off by a factor 8 for the same crystal described with the
    bond crossing the cell edge.

    So `order ≥ 2` **measures** it rather than assuming it: the tensor is recomputed at a
    shifted origin and a disagreement is refused, naming the crystal description as the
    thing to change. `check_origin = false` skips the check and returns the
    origin-dependent number — for diagnosing the failure, not for using the result.

Either way the answer returned is independent of `origin` — guaranteed by the sum rule at
`order = 1`, measured and enforced above it. Both halves are gated
(`test/unit/test_strain.jl`), neither is assumed.

**`symmetrize`.** With `symmetrize = true` (the default) each `(α_p, β_p)` pair is
symmetrized, because strain *is* the symmetric part of the affine map. The discarded
antisymmetric part is not lost information: under SOC the rotation law
`𝓡_U E = −𝓡_S E` fixes it to minus the axial response of the spin channel, which is
what [`rotation_transfer_residual`](@ref) measures. `symmetrize = false` returns the raw
derivative with respect to a general affine map `M`, whose contraction with an arbitrary
`M` reproduces [`affine_energy`](@ref) — the finite-difference gate.

!!! note "Two `ε` derivatives, and their agreement is a gate"
    At `ε = 0` in a single model there is one strain derivative. On a
    [`StrainedModels`](@ref) grid there are two, and they must agree: this
    **intra-model incremental** one, taken inside the model fitted at a grid point with
    its coefficients held fixed, and the **grid finite difference**
    ([`grid_strain_derivative`](@ref)) between neighbouring points, which also carries
    the drift of those coefficients. Their agreement is the grid's acceptance gate
    (design record §14) and says the expansion captures the strain response through its
    displacement channel rather than through coefficient drift. Mind the measure:
    `dE/dη = s·dE/ds`.

See also [`affine_energy`](@ref), [`force_constants`](@ref), [`asr_residual`](@ref).
"""
function strain_derivatives(model::SLCEModel;
                            spins::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                            order::Integer = 1,
                            origin::AbstractVector{<:Real} = SVector(0.0, 0.0, 0.0),
                            symmetrize::Bool = true,
                            check_origin::Bool = true)::Array{Float64}
    order >= 1 || throw(ArgumentError("order must be ≥ 1; got $order"))
    length(origin) == 3 ||
        throw(DimensionMismatch("origin has length $(length(origin)); expected 3"))
    all(isfinite, origin) || throw(ArgumentError("origin has non-finite entries"))
    e = _resolve_spins(model, spins, "the strain derivatives")
    espin = _spin_kernel_matrix(e, n_atoms(model.basis.crystal))
    _require_asr(model, "strain_derivatives")
    n = Int(order)
    _warn_strain_spin_blind(model.basis, n)
    _warn_unresolvable(model, "strain_derivatives")
    out = zeros(Float64, ntuple(_ -> 3, 2n))
    tbuf = zeros(Float64, ntuple(_ -> 3, n))
    polycache = Dict{NTuple{3,Int},SolidHarmonics._Poly}()
    A = model.basis.crystal.lattice.vectors
    frac = model.basis.crystal.frac_positions
    o = SVector{3,Float64}(origin[1], origin[2], origin[3])
    salcs = model.basis.salc_basis.salcs
    dvecs = SVector{3,Float64}[]
    for k in eachindex(model.jphi)
        jphi = model.jphi[k]
        jphi == 0.0 && continue
        salc = salcs[k]
        weight = jphi * (4π)^(count(has_spin, salc.decors) / 2)
        for mem in salc.members
            resize!(dvecs, length(mem.atoms))
            for i in eachindex(mem.atoms)
                a = mem.atoms[i]
                sh = mem.shifts[i]
                dvecs[i] = A * SVector{3,Float64}(frac[1, a] + sh[1],
                                                  frac[2, a] + sh[2],
                                                  frac[3, a] + sh[3]) - o
            end
            for t in mem.terms
                _accumulate_strain!(out, tbuf, weight, t, mem, dvecs, espin, n, polycache)
            end
        end
    end
    symmetrize && _symmetrize_strain_pairs!(out, n)
    check_origin && n >= 2 && _check_strain_origin(model, e, n, o, symmetrize, out, A)
    return out
end

# Recompute at one probe-shifted origin and refuse a disagreement. Only at order ≥ 2:
# order 1 is origin-independent under the ASR alone (the affine field is zero there, hence
# periodic, hence covered by the constraint), and paying 2× for a guaranteed identity
# would be a tax on the deliverable that actually gets used.
function _check_strain_origin(model::SLCEModel,
                              e::Union{SpinConfiguration,Nothing}, n::Int,
                              o::SVector{3,Float64}, symmetrize::Bool,
                              out::Array{Float64}, A)
    probe = o + A * _STRAIN_ORIGIN_PROBE
    other = strain_derivatives(model; spins = e, order = n, origin = probe,
                               symmetrize, check_origin = false)
    scale = norm(out)
    dev = norm(other .- out)
    dev <= _STRAIN_ORIGIN_RTOL * max(scale, eps()) && return nothing
    throw(ArgumentError(
        "the order-$n strain response of this model is not origin-independent " *
        "(‖ΔD‖/‖D‖ = $(dev / max(scale, eps())) between two origins), so it is not a " *
        "property of the model and no number can be returned. The acoustic sum rule is " *
        "satisfied — it is the identity Σ_a ∂E/∂u_a ≡ 0 in the ATOM variables, i.e. on " *
        "cell-periodic fields, and at order ≥ 2 the affine field is not periodic; what " *
        "would be needed is the per-SITE identity, which the constraint does not imply. " *
        "The gap is the home-image gauge: content anchored at an atom's home " *
        "representative cannot cancel against partners the clusters reach at a different " *
        "image, so the per-cell energy picks up a piece linear in the cell's position. " *
        "Fix the crystal DESCRIPTION, not the fit — choose fractional coordinates whose " *
        "home images are the ones the clusters use (for a dimer chain, the bonded " *
        "partner rather than its far image), OR describe the crystal with a larger cell. " *
        "The second option is not a fallback: in a small primitive cell there may be NO " *
        "such choice — in wurtzite each cation bonds to three distinct images of the same " *
        "anion, so no assignment of fractional coordinates makes every bonded partner a " *
        "home image, and the ε² tier is out of reach until the cell is enlarged. " *
        "`order = 1` is unaffected and needs no such choice. Pass " *
        "`check_origin = false` to see the origin-dependent number anyway, for diagnosis " *
        "only."))
end

# The same trap `_warn_spin_blind` catches for the force constants, one order lower and
# with the opposite advice: here `degree = 1` is what a magnetoelastic sector NEEDS, and
# the failure is that no term pairs it with a spin factor.
function _warn_strain_spin_blind(basis::SLCEBasis, order::Int)
    if _spin_blind_at_order(basis, order)
        @warn "strain_derivatives: the order-$order strain response does not depend " *
              "on `spins` — the basis has spin content and displacement terms of " *
              "degree $order, but no term carries both, so ∂E/∂ε is identical for " *
              "every magnetic state and the model has no magnetoelastic coupling at " *
              "this order. Magnetoelasticity needs a sector whose `disp = " *
              "(degree = $order,)` sits under a spin-carrying sector." maxlog = 1
    end
    return nothing
end

# One term's contribution. The displacement-degree filter and the ordered site tuples
# are `_accumulate_fcs!`'s, and the per-tuple tensor is `_fill_fcs_tensor!`'s — what is
# different is only the last step: instead of storing the tensor under a lattice-dynamics
# key, contract each derivative axis with its site's own position vector.
function _accumulate_strain!(out::Array{Float64}, tbuf::Array{Float64}, weight::Float64,
                             t::SALCTerm, mem::SALCMember,
                             dvecs::Vector{SVector{3,Float64}}, e::Matrix{Float64},
                             order::Int, polycache)
    dslots = [i for i in eachindex(t.slots) if t.slots[i].factor.channel == DISP]
    isempty(dslots) && return nothing
    deg = sum(2 * t.slots[i].factor.k + t.slots[i].factor.l for i in dslots)
    deg == order || return nothing
    dsites = [t.slots[i].site for i in dslots]
    degof = Dict(t.slots[i].site => 2 * t.slots[i].factor.k + t.slots[i].factor.l
                 for i in dslots)
    for choice in Iterators.product(ntuple(_ -> dsites, order)...)
        counts = Dict{Int,Int}()
        for s in choice
            counts[s] = get(counts, s, 0) + 1
        end
        all(degof[s] == get(counts, s, 0) for s in dsites) || continue
        fill!(tbuf, 0.0)
        _fill_fcs_tensor!(tbuf, weight, t, dslots, choice, mem, e, polycache)
        _contract_strain!(out, tbuf, dvecs, choice)
    end
    return nothing
end

function _contract_strain!(out::Array{Float64}, tbuf::Array{Float64},
                           dvecs::Vector{SVector{3,Float64}}, choice)
    index = CartesianIndices(tbuf)
    @inbounds for c in index
        v = tbuf[c]
        v == 0.0 && continue
        for b in index
            w = v
            for p in eachindex(choice)
                w *= dvecs[choice[p]][b[p]]
            end
            w == 0.0 && continue
            out[_strain_linear(c, b)] += w
        end
    end
    return nothing
end
