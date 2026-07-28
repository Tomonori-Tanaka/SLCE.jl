module SLCESunnyExt

# Assembles a Sunny.jl `System` from a fitted SLCEModel. All the conversion math
# (folded tesseral tensor → Cartesian matrix, directed-member folding, the supercell
# → primitive unfold, the energy gate) lives in the core (`src/slce/sunny.jl`); this
# extension only places the core-computed matrices into Sunny constructs, so it stays
# thin and the numerics are validated without Sunny.

using SLCE: SLCEModel, BilinearTerms, SunnyPrimitive, _bilinear_terms,
    _sunny_primitive, n_atoms
import SLCE: to_sunny
using Sunny
using StaticArrays
using LinearAlgebra: dot

# A `label -> S` effective-spin lookup from the user's `spins` (scalar or mapping).
function _spin_lookup(spin_length)
    _checked(s) = (s > 0 ? Float64(s) :
                   error("effective spin S must be positive; got $s"))
    spin_length isa Real && (s = _checked(spin_length); return _ -> s)
    if spin_length isa AbstractDict
        return label -> (haskey(spin_length, label) ?
                         _checked(spin_length[label]) :
                         error("`spin_length` has no entry for species \"$label\""))
    end
    error("`spin_length` must be a number or a `species_label => S` Dict")
end

# A spin length is "half-integer" when 2s is an integer (Sunny's `Moment` accepts
# only such values directly). Positivity is checked separately by `_spin_lookup`.
_is_half_integer(s::Real)::Bool = isapprox(2s, round(2s); atol = 1e-9)

# Placeholder Moment spin for the `:coupling` route. The magnon dispersion is invariant
# under an overall spin scale (sᵢ → c sᵢ, J → J/c), so any half-integer works; s₀ = 1
# is the simplest. Picking the placeholder keeps the `Moment` half-integer while the
# physical S_eff is folded into the couplings — the only way to export a non-half-integer
# (itinerant) moment, since Sunny rejects such `Moment` spins.
const _COUPLING_S0 = 1.0

# Classical (`:dipole_uncorrected`) vs quantum (`:dipole`) single-ion rescaling. The
# Moment spin `smom` may differ from the physical `sphys` (the `:coupling` route), so
# both are passed; for the `:moment` route they coincide.
#   :dipole_uncorrected — classical limit; the operator's classical value scales as
#       `smom²`, so `1/(smom·sphys)` makes the landscape scale by `smom/sphys`,
#       consistent with the bilinear rescale (= `1/S²` when `smom = sphys = S`).
#   :dipole — Sunny renormalizes the rank-2 operator by `smom(2smom−1)/2` (the quantum
#       quadrupole, vanishing at `smom = 1/2`); the inverse reproduces the classical
#       form, but only yields the physical single-ion dispersion when `smom = sphys`
#       (the `:moment` route). A placeholder Moment cannot carry it, so it is rejected.
function _onsite_factor(smom::Float64, sphys::Float64, mode::Symbol)::Float64
    mode === :dipole_uncorrected && return 1 / (smom * sphys)
    if mode === :dipole
        smom ≈ sphys ||
            error("single-ion anisotropy in mode = :dipole needs the Moment spin to be " *
                  "the physical spin (s = $smom, S_eff = $sphys); a placeholder Moment " *
                  "(scaling = :coupling) cannot carry the quantum quadrupole. Use " *
                  "mode = :dipole_uncorrected or scaling = :moment.")
        smom > 0.5 ||
            error("mode = :dipole has no rank-2 single-ion anisotropy for s ≤ 1/2 " *
                  "(got s = $smom); use :dipole_uncorrected or drop the ls=[2] channel.")
        return 2 / (smom * (2smom - 1))
    end
    error("unsupported mode $mode for single-ion anisotropy (use :dipole or :dipole_uncorrected)")
end

# A resolved spin plan for one `to_sunny` call. `sphys` maps a species label to its
# physical S_eff; `route` is `:moment` (Moment spin = S_eff; static energy and dispersion
# both exact; needs half-integer S_eff) or `:coupling` (placeholder Moment, couplings
# carry S_eff; dispersion-only, any positive S_eff). `mode` is the resolved single-ion
# mode. The per-site Moment spin is `_smom(plan, type)` — used by both placements.
struct _SpinPlan
    sphys::Function    # species label -> S_eff
    mode::Symbol
    route::Symbol
end
_smom(plan::_SpinPlan, type::AbstractString)::Float64 =
    plan.route === :moment ? plan.sphys(type) : _COUPLING_S0

# Resolve `spins` over the crystal's species and pick the scaling route and single-ion
# mode. `:auto` keys off whether every S_eff is a half-integer: `:moment` + `:dipole`
# for genuine quantum spins, `:coupling` + `:dipole_uncorrected` for itinerant ones.
function _resolve_plan(model::SLCEModel, spin_length, mode::Symbol,
                       scaling::Symbol)::_SpinPlan
    cr = model.basis.crystal
    lookup = _spin_lookup(spin_length)
    sphys = Float64[lookup(t) for t in cr.species_labels]   # one per species
    half = all(_is_half_integer, sphys)

    route = if scaling === :auto
        half ? :moment : :coupling
    elseif scaling === :moment
        half || error("scaling = :moment requires every effective spin to be a " *
                      "half-integer (Sunny's `Moment` rejects other values), got " *
                      "$(unique(sphys)). Use scaling = :coupling (or :auto) to carry " *
                      "the physical S_eff in the couplings with a placeholder Moment.")
        :moment
    elseif scaling === :coupling
        :coupling
    else
        error("scaling must be :auto, :moment, or :coupling; got :$scaling")
    end

    smode = if mode === :auto
        half ? :dipole : :dipole_uncorrected
    elseif mode in (:dipole, :dipole_uncorrected)
        mode
    else
        error("mode must be :auto, :dipole, or :dipole_uncorrected; got :$mode")
    end

    if route === :coupling && length(unique(round.(sphys; digits = 9))) > 1
        @warn "to_sunny: scaling = :coupling with a non-uniform S_eff " *
              "($(sort(unique(sphys)))); the dispersion is exact only for a uniform " *
              "S_eff. The off-diagonal exchange stays exact, but the on-site (Larmor) " *
              "term is approximate. For an exact multi-sublattice dispersion use " *
              "scaling = :moment with half-integer spins."
    end
    return _SpinPlan(lookup, smode, route)
end

# Per-bond denominator D so the emitted coupling is `M/D` (pair energy `êᵢ'(M/D)êⱼ`
# under the Moment spins). `:moment` → `Sᵢ Sⱼ` (energy-preserving). `:coupling` →
# `√(smomᵢ smomⱼ)·√(Sᵢ Sⱼ)`, the physical off-diagonal exchange `√(SᵢSⱼ)·J_phys` for
# any spins, reducing to `M/(s₀ S)` for a uniform S.
function _bond_denom(plan::_SpinPlan, ta::AbstractString, tb::AbstractString)::Float64
    plan.route === :moment && return plan.sphys(ta) * plan.sphys(tb)
    return sqrt(_smom(plan, ta) * _smom(plan, tb)) * sqrt(plan.sphys(ta) * plan.sphys(tb))
end

# The Sunny anisotropy operator `factor · Σ_ij A[i,j] (Sᵢ Sⱼ + Sⱼ Sᵢ)/2` from the
# traceless symmetric Cartesian matrix `A` and the site spin operators `S`.
_onsite_operator(A::SMatrix{3,3,Float64,9}, S, factor::Float64) =
    factor * sum(A[i, j] * (S[i] * S[j] + S[j] * S[i]) / 2 for i = 1:3, j = 1:3)

# The training supercell as a P1 Sunny crystal (every atom its own sublattice).
function _build_supercell(model::SLCEModel, plan::_SpinPlan, g::Real)
    cr = model.basis.crystal
    nat = n_atoms(cr)
    latvecs = Matrix{Float64}(cr.lattice.vectors)
    positions = [Vector{Float64}(cr.frac_positions[:, i]) for i = 1:nat]
    types = [cr.species_labels[cr.species[i]] for i = 1:nat]
    cryst = Sunny.Crystal(latvecs, positions, 1; types = types)
    moments = [i => Sunny.Moment(s = _smom(plan, types[i]), g = Float64(g)) for i = 1:nat]
    sys = Sunny.to_inhomogeneous(Sunny.System(cryst, moments, plan.mode))

    terms = _bilinear_terms(model)
    for ((a, b, R), M) in terms.pairs
        J = M / _bond_denom(plan, types[a], types[b])
        Sunny.set_exchange_at!(sys, Matrix(J), (1, 1, 1, a), (1, 1, 1, b);
                               offset = (R[1], R[2], R[3]))
    end
    for (a, A) in terms.onsites
        f = _onsite_factor(_smom(plan, types[a]), plan.sphys(types[a]), plan.mode)
        Sunny.set_onsite_coupling_at!(sys, S -> _onsite_operator(A, S, f), (1, 1, 1, a))
    end
    return sys, terms.skipped
end

# The chemical primitive cell, with one Sunny bond per primitive bond (Sunny's
# periodic replication restores the supercell multiplicity).
function _build_primitive(prim::SunnyPrimitive, plan::_SpinPlan, g::Real)
    latvecs = Matrix{Float64}(prim.latvecs)
    positions = [Vector{Float64}(p) for p in prim.positions]
    types = prim.types
    # symbol = 1 forces P1: bonds are placed explicitly from the SLCE model, so Sunny
    # must not re-derive (and symmetrize over) the crystal symmetry.
    cryst = Sunny.Crystal(latvecs, positions, 1; types = types)
    moments = [i => Sunny.Moment(s = _smom(plan, types[i]), g = Float64(g))
               for i in eachindex(types)]
    sys = Sunny.System(cryst, moments, plan.mode)
    for ((i, j, n), M) in prim.bonds
        J = M / _bond_denom(plan, types[i], types[j])
        Sunny.set_exchange!(sys, Matrix(J), Sunny.Bond(i, j, [n[1], n[2], n[3]]))
    end
    for (i, A) in prim.onsites
        f = _onsite_factor(_smom(plan, types[i]), plan.sphys(types[i]), plan.mode)
        Sunny.set_onsite_coupling!(sys, S -> _onsite_operator(A, S, f), i)
    end
    return sys
end

function to_sunny(model::SLCEModel; spin_length, g::Real = 2, mode::Symbol = :auto,
                  scaling::Symbol = :auto, placement::Symbol = :auto)
    placement in (:auto, :primitive, :explicit) ||
        error("placement must be :auto, :primitive, or :explicit; got $placement")
    plan = _resolve_plan(model, spin_length, mode, scaling)
    local sys, skipped
    if placement === :explicit
        sys, skipped = _build_supercell(model, plan, g)
    else
        prim = _sunny_primitive(model)
        if prim.clean
            sys = _build_primitive(prim, plan, g)
            skipped = prim.skipped
        else
            msg = "to_sunny: model does not unfold cleanly onto the primitive cell; " *
                  "using the exact supercell route instead (dispersion will be folded)."
            placement === :primitive ? @warn(msg) : @info(msg)
            sys, skipped = _build_supercell(model, plan, g)
        end
    end
    isempty(skipped) ||
        @warn "to_sunny: $(length(skipped)) SALC channel(s) are not Sunny-representable " *
              "and were skipped:\n  " * join(skipped, "\n  ")
    return sys
end

end # module SLCESunnyExt
