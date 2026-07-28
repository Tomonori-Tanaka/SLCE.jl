# Provenance metadata for training data. Split out of io/dftsource.jl because
# `SLCEDataset` (slce/model.jl, included earlier) stores a dataset-level
# DatumProvenance identity summary.

"""
    DatumProvenance(; constrained = false, torque_qualified = false,
                    reference_id = nothing, reference_fingerprint = nothing,
                    setup_id = nothing, soc = nothing) -> DatumProvenance

Load-bearing metadata about **how** a [`TrainingDatum`](@ref)'s observables were
computed — the part of the training-data contract the raw numbers cannot carry.
Stamped by the DFT adapter that builds the datum; consumed as invariants by the
`SLCEDataset` constructor.

- `constrained::Bool` — the configuration came from a constrained-noncollinear
  calculation (a nonzero constraining field was applied).
- `torque_qualified::Bool` — the torque targets may enter the torque design block
  `X_T`. The [`SpinDatum`](@ref) convenience constructor derives this as
  `any(!iszero, field)`: a zero field means the run recorded no constraint
  information, and claiming `τ = 0` for it is only valid if the moments were
  *relaxed to self-consistency* — which the numbers alone cannot certify. To assert
  exactly that (a converged **unconstrained** noncollinear run, whose `τ = 0` is a
  genuine stationarity observation), pass an explicit
  `provenance = DatumProvenance(; torque_qualified = true)`.
  The datum constructors **upgrade** this flag, never revoke it: a nonzero `field`,
  or `torques` passed explicitly, sets it even on a hand-built provenance. That
  matters because a joint datum has no choice but to build one by hand — the
  displacement channel requires `reference_id`/`reference_fingerprint` — and before
  the upgrade, stamping the reference silently dropped the qualification. To keep
  torques out of a fit, use `use_torque = false` at the [`SLCEDataset`](@ref) level,
  which is where that decision belongs.
- `reference_id::Union{String,Nothing}` — human-readable label of the clamped-ion
  reference crystal the displacements `u = r − r_ref` were measured against
  (`nothing` = unset; allowed only against a pure-spin basis).
- `reference_fingerprint::Union{String,Nothing}` — [`crystal_fingerprint`](@ref) of
  that reference crystal. `SLCEDataset` requires it to match the fingerprint of
  `basis.crystal` whenever the basis carries displacement-decorated (`p ≥ 1`) SALCs —
  the double-counting protocol (all data from ONE clamped-ion reference) enforced as
  an invariant rather than a docs warning.
- `setup_id::Union{String,Nothing}` — label of the computational setup (code,
  XC functional, `ENCUT`/k-mesh family, collinear vs noncollinear, SOC on/off).
  `SLCEDataset` rejects a mixture of distinct setups: total energies from different
  setups sit on different scales (reference-energy zero points, SOC content), and the
  offset is family-correlated, so it biases coefficients instead of averaging into
  `j0`. Fit heterogeneous families in stages (per-family datasets +
  `frozen`/`sector_mask`) instead of mixing them in one regression.
- `soc::Union{Bool,Nothing}` — whether spin–orbit coupling was included
  (`nothing` = unknown). Checked for uniformity together with `setup_id`.
"""
struct DatumProvenance
    constrained::Bool
    torque_qualified::Bool
    reference_id::Union{String,Nothing}
    reference_fingerprint::Union{String,Nothing}
    setup_id::Union{String,Nothing}
    soc::Union{Bool,Nothing}
end

DatumProvenance(; constrained::Bool = false, torque_qualified::Bool = false,
                reference_id::Union{AbstractString,Nothing} = nothing,
                reference_fingerprint::Union{AbstractString,Nothing} = nothing,
                setup_id::Union{AbstractString,Nothing} = nothing,
                soc::Union{Bool,Nothing} = nothing)::DatumProvenance =
    DatumProvenance(constrained, torque_qualified,
                    reference_id === nothing ? nothing : String(reference_id),
                    reference_fingerprint === nothing ? nothing :
                        String(reference_fingerprint),
                    setup_id === nothing ? nothing : String(setup_id), soc)
