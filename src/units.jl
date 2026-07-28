# The kelvin ↔ model-energy-unit boundary, shared by every sampler in the family.
#
# The fitting core itself has no temperature: a fitted model is a zero-temperature
# energy surface. This file lives here anyway because the *convention* is the core's
# to own — SLCEMonteCarlo (`run_mc` / `run_pt`), SLCEDynamics (the stochastic
# thermostat) and SLCETools (`MetropolisSampler`) all convert kelvin into the energy
# units the model was fitted in, and two independent copies of that conversion are two
# things that can drift apart while every test stays green.
#
# Every public entry point downstream takes exactly one of two keywords — `temperature`
# in kelvin or `kT` in the model's energy units. The two live under distinct names
# deliberately: a single keyword serving both units would let `temperature = 300` (meant
# as kelvin) be read as 300 eV — a silent infinite-temperature run.

"""
    KB_EV

Boltzmann's constant in eV/K (the exact CODATA ratio `1.380649e-23 J/K` /
`1.602176634e-19 J/eV`). Converts a kelvin control to the energy scale of an
**eV-fitted** model: `kT = KB_EV * temperature`.

The models this package fits carry no unit metadata — their energy unit is whatever
the training data used. Every kelvin control in the family therefore *assumes* eV;
`kT` is the escape hatch for a model fitted in anything else.
"""
const KB_EV = 1.380649e-23 / 1.602176634e-19

"""
    resolve_kt(temperature, kT) -> Vector{Float64}

Resolve exactly one of `temperature` (kelvin) / `kT` (`k_B·T`, model energy units) —
scalar or collection — into a validated `k_B·T` vector in the model's energy units.
The two controls live under distinct names so a kelvin value can never be silently
read as an energy (or vice versa); kelvin input is validated in kelvin first, so the
error echoes the unit the caller used, then converted with [`KB_EV`](@ref).
"""
function resolve_kt(temperature, kT)::Vector{Float64}
    (temperature === nothing) == (kT === nothing) && throw(ArgumentError(
        "provide exactly one of `temperature` (kelvin) or `kT` " *
        "(k_B·T, model energy units)"))
    vals = if kT !== nothing
        kT isa Real ? [Float64(kT)] : Float64[Float64(x) for x in kT]
    else
        # validate in kelvin first, so the error echoes the unit the caller used
        ts = temperature isa Real ? [Float64(temperature)] :
            Float64[Float64(x) for x in temperature]
        for T in ts
            (isfinite(T) && T > 0) || throw(ArgumentError(
                "temperature must be finite and > 0 kelvin; got $T"))
        end
        KB_EV .* ts
    end
    isempty(vals) && throw(ArgumentError("the temperature/kT collection is empty"))
    for x in vals
        (isfinite(x) && x > 0) || throw(ArgumentError(
            "kT must be finite and > 0 (k_B·T, model energy units); got $x"))
    end
    return vals
end
