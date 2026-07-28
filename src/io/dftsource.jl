"""
    AbstractDFTSource

A source of DFT training data: `read_configs(src::AbstractDFTSource) ->
Vector{TrainingDatum}` turns some DFT output into fit-ready configurations, and
`SLCEDataset(basis, src)` goes straight from a source to a dataset. Concrete sources
(e.g. `SLCETools.VASP.Oszicar`, in the SLCETools.jl package) live alongside their format
reader, so the SCE pipeline consumes only [`TrainingDatum`](@ref) / [`SLCEDataset`](@ref)
and never depends on the originating DFT code.
"""
abstract type AbstractDFTSource end

"""
    TrainingDatum(; energy, directions, magmoms, displacements = nothing,
                  forces = nothing, field = nothing, torques = nothing,
                  provenance = DatumProvenance()) -> TrainingDatum

One training configuration's observables, as consumed by the SCE pipeline — the
code-agnostic, in-memory form every DFT adapter produces. The spin channel is
required; every other channel is optional (`nothing` = **not observed**, which is
distinct from an observed zero):

- `energy::Float64` — total energy (eV). Required: every electronic-structure code
  emits one. (For VASP constrained-noncollinear runs TOTEN contains exactly one
  penalty energy `E_p` — negligible when converged, and subtractable by the adapter.)
- `directions::Matrix{Float64}` — `3 × n_atoms` unit spin directions `e_a`
  (collinear/Ising configurations enter as `±ẑ` columns).
- `magmoms::Vector{Float64}` — per-atom moment **magnitudes** `‖m_a‖ ≥ 0` (μ_B).
  The sign convention is load-bearing: the direction carries the sign, the magnitude
  is nonnegative — a collinear adapter must lift a signed scalar `m` to
  `directions[:, a] = (0, 0, sign(m))`, `magmoms[a] = |m|` (enforced here).
- `displacements` — `3 × n_atoms` Cartesian displacements `u_a = r_a − r_a^ref` (Å)
  against the clamped-ion reference crystal named by the provenance;
  `nothing` means `u = 0` **exactly**. Displacements are NOT rotated by any spin
  frame (a VASP adapter rotates moments/fields out of the SAXIS frame; positions
  live in the lattice Cartesian frame — two different frames in one datum).
- `forces` — `3 × n_atoms` forces (eV/Å), sign pinned as `f_a = −∂E/∂u_a` (the plain
  Euclidean gradient, no projection). Independent of `displacements`: forces at
  `u = 0` (clamped-ion reference forces) are real, fittable physics, so
  `forces !== nothing` with `displacements === nothing` is a legitimate datum.
  Forces are assumed to be the gradient of the **same** energy surface `energy`
  belongs to — for a constrained run, quarantining any penalty-term contribution
  is the adapter's responsibility (the `E_p` precedent).
- `field` — `3 × n_atoms` constraining field `B_a` (eV/μ_B); `nothing` means the
  transverse field was **not computed** (collinear runs, codes without constrained
  noncollinear magnetism). A present all-zero field is different: it asserts the
  field was computed and vanishes.
- `torques` — `3 × n_atoms` torque targets `τ_a = m_a × B_a = ‖m_a‖ (e_a × B_a)`
  (eV), the physical / Landau–Lifshitz torque that the SCE torque
  `τ_a = −e_a × ∂E/∂e_a` is fit to. Derived automatically from `field` when present;
  a caller passing `torques` directly (a code that emits torques without exposing
  `B`) must match this exact convention — `τ = m × B`, the same sign as the model's
  `predict_torque = −e × ∇E`, NOT the energy-rotation gradient `+e × ∇E`.
- `provenance::DatumProvenance` — see [`DatumProvenance`](@ref). When omitted it is
  derived exactly as in [`spin_datum`](@ref): `constrained = torque_qualified =
  any(!iszero, field)`, and explicitly passed `torques` set
  `torque_qualified = true` (you supplied them, so they are meant to be fit);
  pass an explicit provenance to override.

All present channels must agree on `n_atoms` and be finite; `directions` columns must
be unit vectors. The displacement-radius guard (amplitudes vs half the shortest
reference interatomic distance — the classic un-minimum-imaged-adapter symptom)
lives at the design boundary: [`SLCEDataset`](@ref) warns when it is exceeded.

Three convenience constructors build one of these and hand back an ordinary
`TrainingDatum` — one per corner of the expansion, so the raw keyword form above is
only needed for something none of them covers:

| constructor | what it is for |
|:--|:--|
| [`spin_datum`](@ref) | spin only, from raw DFT moments (+ constraining field) |
| [`lattice_datum`](@ref) | lattice only — no magnetic state to give |
| [`joint_datum`](@ref) | both channels: moments/field **and** a displaced structure |
"""
struct TrainingDatum
    energy::Float64
    directions::Matrix{Float64}
    magmoms::Vector{Float64}
    displacements::Union{Matrix{Float64},Nothing}
    forces::Union{Matrix{Float64},Nothing}
    field::Union{Matrix{Float64},Nothing}
    torques::Union{Matrix{Float64},Nothing}
    provenance::DatumProvenance

    function TrainingDatum(energy::Float64, directions::Matrix{Float64},
                           magmoms::Vector{Float64},
                           displacements::Union{Matrix{Float64},Nothing},
                           forces::Union{Matrix{Float64},Nothing},
                           field::Union{Matrix{Float64},Nothing},
                           torques::Union{Matrix{Float64},Nothing},
                           provenance::DatumProvenance)
        isfinite(energy) || throw(ArgumentError("`energy` is not finite ($energy)"))
        size(directions, 1) == 3 ||
            throw(ArgumentError("`directions` must be 3 × n_atoms"))
        nat = size(directions, 2)
        length(magmoms) == nat ||
            throw(ArgumentError("`magmoms` length $(length(magmoms)) ≠ n_atoms $nat"))
        @inbounds for a = 1:nat
            e = SVector{3,Float64}(directions[1, a], directions[2, a], directions[3, a])
            all(isfinite, e) ||
                throw(ArgumentError("`directions` column $a is not finite"))
            abs(norm(e) - 1) <= 1e-6 ||
                throw(ArgumentError("`directions` column $a is not a unit vector " *
                                    "(‖e‖ = $(norm(e))); the direction carries the " *
                                    "spin sign, `magmoms` the magnitude"))
            (isfinite(magmoms[a]) && magmoms[a] >= 0) ||
                throw(ArgumentError("`magmoms[$a]` = $(magmoms[a]) must be a finite " *
                                    "magnitude ≥ 0; a collinear adapter must lift a " *
                                    "signed scalar m to direction (0,0,sign(m)) and " *
                                    "magnitude |m|"))
        end
        for (name, ch) in (("displacements", displacements), ("forces", forces),
                           ("field", field), ("torques", torques))
            ch === nothing && continue
            size(ch) == (3, nat) ||
                throw(ArgumentError("`$name` must be 3 × $nat (got $(size(ch)))"))
            all(isfinite, ch) || throw(ArgumentError("`$name` contains non-finite entries"))
        end
        return new(energy, directions, magmoms, displacements, forces, field,
                   torques, provenance)
    end
end

function TrainingDatum(; energy::Real, directions::AbstractMatrix{<:Real},
                       magmoms::AbstractVector{<:Real},
                       displacements::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                       forces::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                       field::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                       torques::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                       provenance::Union{DatumProvenance,Nothing} = nothing)::TrainingDatum
    _mat(x) = x === nothing ? nothing : Matrix{Float64}(x)
    dirs = Matrix{Float64}(directions)
    mags = collect(Float64, magmoms)
    fld = _mat(field)
    trq = _mat(torques)
    # Same derivation as the spin_datum constructor — the two construction paths of one
    # type must not disagree on the load-bearing torque_qualified gate (a mismatch
    # silently drops hand-built configs' torque rows in a mixed batch). A nonzero
    # field qualifies; explicitly passed torques qualify (the caller owns the
    # convention, per the docstring); an all-zero field does not (assert a converged
    # unconstrained run via an explicit provenance).
    c = fld !== nothing && any(!iszero, fld)
    qual = c || trq !== nothing
    if provenance === nothing
        provenance = DatumProvenance(; constrained = c, torque_qualified = qual)
    elseif qual && !provenance.torque_qualified
        # The evidence for qualification is the same whoever built the provenance —
        # and a JOINT datum has no choice but to build one by hand, because the
        # displacement channel requires `reference_id`/`reference_fingerprint`. Before
        # this upgrade the two requirements collided: stamping the reference silently
        # dropped the auto-qualification, and a datum carrying displacements AND
        # torques then failed the dataset build with "pass use_torque = false" — the
        # exact opposite of what the caller was trying to do. Upgrade only; an
        # explicit `torque_qualified = true` is never revoked, and withholding
        # torques from the fit is `use_torque = false`'s job, at the dataset level.
        provenance = DatumProvenance(; constrained = provenance.constrained || c,
                                     torque_qualified = true,
                                     reference_id = provenance.reference_id,
                                     reference_fingerprint =
                                         provenance.reference_fingerprint,
                                     setup_id = provenance.setup_id,
                                     soc = provenance.soc)
    end
    if fld !== nothing && trq === nothing
        # Derive the torque target τ_a = m_a × B_a (the physical / Landau–Lifshitz
        # torque) from the stored decomposition m_a = magmoms[a] · directions[:, a].
        nat = size(dirs, 2)
        size(fld) == (3, nat) ||
            throw(ArgumentError("`field` must be 3 × $nat (got $(size(fld)))"))
        trq = Matrix{Float64}(undef, 3, nat)
        @inbounds for a = 1:nat
            m = mags[a] * SVector{3,Float64}(dirs[1, a], dirs[2, a], dirs[3, a])
            B = SVector{3,Float64}(fld[1, a], fld[2, a], fld[3, a])
            t = cross(m, B)
            trq[1, a] = t[1]; trq[2, a] = t[2]; trq[3, a] = t[3]
        end
    end
    return TrainingDatum(Float64(energy), dirs, mags, _mat(displacements),
                         _mat(forces), fld, trq, provenance)
end

function Base.show(io::IO, d::TrainingDatum)
    chans = String[]
    d.displacements === nothing || push!(chans, "u")
    d.forces === nothing || push!(chans, "f")
    d.torques === nothing || push!(chans, "τ")
    print(io, "TrainingDatum(E = ", d.energy, ", n_atoms = ", size(d.directions, 2),
          isempty(chans) ? "" : ", +" * join(chans, "+"), ")")
end

# ----------------------------------------------------------------------------
# Stable crystal fingerprint (reference-identity provenance)
# ----------------------------------------------------------------------------

# FNV-1a, 64-bit. Hand-rolled (NOT `Base.hash`, which is Julia-version dependent —
# the same reason `io/persist.jl` recomputes the SALC fingerprint on load): this one
# is *trusted* across sessions and Julia versions, so it must be pinned here.
const _FP_OFFSET = 0xcbf29ce484222325
const _FP_PRIME = 0x00000100000001b3

_fp_byte(h::UInt64, b::UInt8)::UInt64 = (h ⊻ b) * _FP_PRIME
function _fp_u64(h::UInt64, x::UInt64)::UInt64
    for s = 0:8:56                       # explicit little-endian byte order
        h = _fp_byte(h, UInt8((x >> s) & 0xff))
    end
    return h
end
_fp_int(h::UInt64, x::Integer)::UInt64 = _fp_u64(h, reinterpret(UInt64, Int64(x)))
_fp_float(h::UInt64, x::Float64)::UInt64 = _fp_u64(h, reinterpret(UInt64, _jnum(x)))
function _fp_str(h::UInt64, s::String)::UInt64
    cu = codeunits(s)
    h = _fp_int(h, length(cu))           # length prefix: ["Fe","eO"] ≠ ["Fe","e","O"]
    for b in cu
        h = _fp_byte(h, b)
    end
    return h
end

# Quantize a coordinate to a 1e-10 grid before hashing, so values that differ only
# by sub-1e-10 construction noise (e.g. `mod(-1e-17, 1.0) == 1.0` vs `0.0` at the
# fractional wrap boundary) fingerprint identically. `wrap` re-folds the [0, 1)
# boundary after quantization; `_jnum` (via `_fp_float`) folds a -0.0. The single
# `-= 1.0` fold is sufficient only because `Crystal`'s inner constructor already
# wraps periodic fractional coordinates with `mod(·, 1.0)` into [0, 1] — if that
# wrap ever moves, this becomes a partial mod.
function _fp_quant(x::Float64; wrap::Bool = false)::Float64
    q = round(x * 1e10) / 1e10
    wrap && q >= 1.0 && (q -= 1.0)
    return q
end

"""
    crystal_fingerprint(crystal::Crystal) -> String

A stable 16-hex-digit content fingerprint of a [`Crystal`](@ref) — lattice vectors,
periodicity flags, fractional positions (in storage order), species indices, and
species labels. Used as the reference-identity check of the data layer: a DFT
adapter stamps `crystal_fingerprint(reference)` into
[`DatumProvenance`](@ref)`.reference_fingerprint`, and `SLCEDataset` requires it to
equal `crystal_fingerprint(basis.crystal)` for a displacement-decorated basis.

Properties and scope:
- **Stable**: hand-rolled FNV-1a over a canonical byte serialization — independent
  of the Julia version and session (unlike `Base.hash`). Coordinates are quantized
  to a `1e-10` grid and `-0.0` is folded to `0.0` first, so a save/load round trip
  through `SLCE.save`/`SLCE.load` (lossless Float64 printing) cannot flip the
  fingerprint, and neither can construction noise that stays on one side of a grid
  line (the fractional wrap boundary `mod(-ε, 1) == 1.0` included). Two values
  straddling a grid line still differ — the failure mode is a loud false mismatch,
  never a silent false match.
- **Same-lineage identity, not symmetry equivalence**: two `Crystal`s built from the
  same file agree; a re-relaxed, re-generated, symmetry-transformed, or reordered
  (atom permutation) description of the *physically same* crystal is deliberately a
  mismatch — the double-counting protocol wants exactly one reference object, so
  "physically the same but rebuilt" must fail loudly, not pass silently.
- **Not cryptographic**: 64 bits is ample for non-adversarial provenance
  bookkeeping (a handful of reference crystals), and avoids an SHA dependency.
"""
function crystal_fingerprint(crystal::Crystal)::String
    h = _fp_str(_FP_OFFSET, "SLCE-crystal-fp-v1")
    A = crystal.lattice.vectors
    for j = 1:3, i = 1:3
        h = _fp_float(h, _fp_quant(A[i, j]))
    end
    for d = 1:3
        h = _fp_int(h, Int(crystal.lattice.pbc[d]))
    end
    nat = n_atoms(crystal)
    h = _fp_int(h, nat)
    fr = crystal.frac_positions
    for a = 1:nat, i = 1:3
        h = _fp_float(h, _fp_quant(fr[i, a]; wrap = crystal.lattice.pbc[i]))
    end
    h = _fp_int(h, length(crystal.species_labels))
    for lab in crystal.species_labels
        h = _fp_str(h, lab)
    end
    for s in crystal.species
        h = _fp_int(h, s)
    end
    return string(h; base = 16, pad = 16)
end

# ----------------------------------------------------------------------------
# spin_datum convenience constructors (spin-only TrainingDatum)
# ----------------------------------------------------------------------------

# Shared derivation: raw per-atom moment vectors → (unit directions, magnitudes).
# A near-zero moment (below `zero_moment_atol`) gets the ẑ placeholder direction —
# a divide-by-zero guard; `SLCEDataset` rejects a placeholder on any basis-referenced
# atom (see `_check_referenced_moments`).
function _moments_to_dirs(moments::AbstractMatrix{<:Real};
                          zero_moment_atol::Real)::Tuple{Matrix{Float64},Vector{Float64}}
    size(moments, 1) == 3 || throw(ArgumentError("`moments` must be 3 × n_atoms"))
    n = size(moments, 2)
    dirs = Matrix{Float64}(undef, 3, n)
    mags = Vector{Float64}(undef, n)
    @inbounds for i = 1:n
        mi = SVector{3,Float64}(moments[1, i], moments[2, i], moments[3, i])
        mag = norm(mi)
        mags[i] = mag
        ei = mag <= zero_moment_atol ? SVector{3,Float64}(0, 0, 1) : mi / mag
        dirs[1, i] = ei[1]; dirs[2, i] = ei[2]; dirs[3, i] = ei[3]
    end
    return dirs, mags
end

"""
    spin_datum(energy, moments, field; zero_moment_atol = 1e-10,
              provenance = nothing) -> TrainingDatum
    spin_datum(energy, moments; zero_moment_atol = 1e-10,
              provenance = nothing) -> TrainingDatum

Build a spin-only [`TrainingDatum`](@ref) from raw per-atom magnetic moment vectors
`moments` (`3 × n_atoms`, μ_B) — the convenience entry point for DFT spin data
(`spin_datum` names the spin-only special case; the returned value is an ordinary
`TrainingDatum` with no displacement/force channels).

The three-argument form takes the per-atom constraining field `field`
(`3 × n_atoms`, eV/μ_B) from a constrained-noncollinear calculation. The spin
direction is `e_a = m_a / ‖m_a‖` (a near-zero moment, below `zero_moment_atol`, gets
the placeholder `ẑ` and a zero torque), the magnitude is `‖m_a‖`, and the torque
target is `τ_a = m_a × B_a` (eV) — the physical / Landau–Lifshitz torque, matching
the SCE model torque `−e_a × ∂E/∂e_a`. When `provenance` is not given, it is derived
as `DatumProvenance(; constrained = c, torque_qualified = c)` with
`c = any(!iszero, field)`: an all-zero field carries no constraint information, so
its `τ = 0` rows are **not** admitted by default — if the run really was a converged
*unconstrained* noncollinear calculation (whose `τ = 0` is a genuine stationarity
observation), say so explicitly with
`provenance = DatumProvenance(; torque_qualified = true)`.

The two-argument form is for sources with no field/torque output at all (collinear
runs, codes without constrained noncollinear magnetism): `field` and `torques` are
absent (`nothing`), so the datum can only contribute energy rows.

The torque target carries the per-config moment magnitude `‖m_a‖`, while the SCE
model torque depends on directions only — so a co-fit assumes the moment magnitudes
are roughly constant across configurations (large longitudinal variation would bias
it). A *magnetic* site that quenches to `‖m_a‖ ≈ 0` enters with the fictitious `ẑ`
direction — prefer dropping such configurations. (`SLCEDataset` rejects a
placeholder on a basis-referenced atom; if you change this tolerance, pass the same
value to its `zero_moment_atol` so the guard stays aligned.)
"""
function spin_datum(energy::Real, moments::AbstractMatrix{<:Real},
                   field::AbstractMatrix{<:Real}; zero_moment_atol::Real = 1e-10,
                   provenance::Union{DatumProvenance,Nothing} = nothing)::TrainingDatum
    size(field) == size(moments) ||
        throw(ArgumentError("`field` $(size(field)) must match `moments` $(size(moments))"))
    dirs, mags = _moments_to_dirs(moments; zero_moment_atol = zero_moment_atol)
    n = size(moments, 2)
    torq = Matrix{Float64}(undef, 3, n)
    @inbounds for i = 1:n
        mi = SVector{3,Float64}(moments[1, i], moments[2, i], moments[3, i])
        Bi = SVector{3,Float64}(field[1, i], field[2, i], field[3, i])
        ti = cross(mi, Bi)                  # τ = m × B  (physical / LL torque)  [eV]
        torq[1, i] = ti[1]; torq[2, i] = ti[2]; torq[3, i] = ti[3]
    end
    c = any(!iszero, field)
    # Upgrade-only, matching the keyword constructor: the two construction paths of
    # one type must not disagree on this gate, so a hand-built provenance carrying
    # setup metadata cannot suppress the qualification a nonzero field earns.
    provenance = provenance === nothing ?
                 DatumProvenance(; constrained = c, torque_qualified = c) :
                 (c && !provenance.torque_qualified ?
                  DatumProvenance(; constrained = true, torque_qualified = true,
                                  reference_id = provenance.reference_id,
                                  reference_fingerprint =
                                      provenance.reference_fingerprint,
                                  setup_id = provenance.setup_id,
                                  soc = provenance.soc) : provenance)
    return TrainingDatum(Float64(energy), dirs, mags, nothing, nothing,
                         Matrix{Float64}(field), torq, provenance)
end

function spin_datum(energy::Real, moments::AbstractMatrix{<:Real};
                   zero_moment_atol::Real = 1e-10,
                   provenance::Union{DatumProvenance,Nothing} = nothing)::TrainingDatum
    dirs, mags = _moments_to_dirs(moments; zero_moment_atol = zero_moment_atol)
    provenance === nothing && (provenance = DatumProvenance())
    return TrainingDatum(Float64(energy), dirs, mags, nothing, nothing, nothing,
                         nothing, provenance)
end

"""
    lattice_datum(energy; displacements = nothing, forces = nothing,
                 reference = nothing, reference_id = "reference",
                 n_atoms = nothing, provenance = nothing) -> TrainingDatum

A [`TrainingDatum`](@ref) for **spin-free** training data: one energy, a displacement
field, and the forces that came with it. The counterpart of [`spin_datum`](@ref) at the
other end of the joint expansion.

`TrainingDatum` requires a spin channel, because the package's centre of gravity is a
spin–lattice expansion and a datum that silently omits its magnetic state is the
harder failure to diagnose. A lattice-only user has no magnetic state to give, so this
constructor supplies the inert one: the `ẑ` placeholder direction (the same one
[`spin_datum`](@ref) uses for a quenched moment) with **exactly zero** moment
magnitudes. Zero is the load-bearing part. A lattice-only basis references no spin
site, so the placeholder is never read; feed the same datum to a basis that *does*
carry spin content and [`SLCEDataset`](@ref)'s zero-moment invariant rejects it by
name, rather than fitting it as a fabricated ferromagnet.

Pass `reference` (the clamped-ion reference `Crystal`) and the provenance stamp the
displacement channel requires is built for you — `reference_id` plus
[`crystal_fingerprint`](@ref). `n_atoms` is otherwise inferred from whichever of
`displacements` / `forces` is present, and is required only when neither is (an
energy-only datum). An explicit `provenance` overrides all of it.

```julia
data = [lattice_datum(E[i]; displacements = u[i], forces = F[i], reference = crystal)
        for i in eachindex(E)]
ds = SLCEDataset(basis, data)              # use_torque resolves to false by itself
f  = fit(SLCEFit, ds, OLS(); force_weight = 1.0)
```
"""
function lattice_datum(energy::Real;
                       displacements::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                       forces::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                       reference::Union{Crystal,Nothing} = nothing,
                       reference_id::AbstractString = "reference",
                       n_atoms::Union{Integer,Nothing} = nothing,
                       provenance::Union{DatumProvenance,Nothing} = nothing)::TrainingDatum
    nat = n_atoms !== nothing ? Int(n_atoms) :
          displacements !== nothing ? size(displacements, 2) :
          forces !== nothing ? size(forces, 2) :
          reference !== nothing ? size(reference.frac_positions, 2) :
          throw(ArgumentError("lattice_datum: cannot infer the atom count — pass " *
                              "`displacements`, `forces`, `reference`, or `n_atoms`"))
    nat > 0 || throw(ArgumentError("lattice_datum: n_atoms must be ≥ 1; got $nat"))
    if provenance === nothing && reference !== nothing
        provenance = DatumProvenance(; reference_id = reference_id,
                                     reference_fingerprint =
                                         crystal_fingerprint(reference))
    end
    dirs = repeat(Float64[0.0, 0.0, 1.0], 1, nat)
    return TrainingDatum(; energy = energy, directions = dirs, magmoms = zeros(nat),
                         displacements = displacements, forces = forces,
                         provenance = provenance)
end

"""
    joint_datum(energy; moments, field = nothing, displacements = nothing,
                forces = nothing, reference = nothing, reference_id = "reference",
                zero_moment_atol = 1e-10, provenance = nothing) -> TrainingDatum

A [`TrainingDatum`](@ref) carrying **both** channels: a magnetic state from raw DFT
moments and a displaced structure. The third sibling of [`spin_datum`](@ref) and
[`lattice_datum`](@ref), and the one the joint spin–lattice expansion is actually
about — a `TrainingDatum(; ...)` written out by hand does the same thing, but leaves
each caller to redo the two derivations below.

It performs both: the spin side exactly as [`spin_datum`](@ref) (direction
`e_a = m_a/‖m_a‖`, magnitude `‖m_a‖`, torque `τ_a = m_a × B_a` when `field` is given,
`ẑ` placeholder below `zero_moment_atol`), and the reference stamp exactly as
[`lattice_datum`](@ref) (`reference_id` plus [`crystal_fingerprint`](@ref), which the
displacement channel requires).

Passing both at once is why this exists. The two requirements used to collide: a
joint datum needs a hand-built [`DatumProvenance`](@ref) for the reference stamp, and
a hand-built provenance used to drop the torque qualification a nonzero `field`
earns — so a datum carrying displacements *and* torques failed the dataset build. The
qualification is derived here from the same evidence as everywhere else and upgraded,
never revoked.

```julia
data = [joint_datum(E[i]; moments = m[i], field = B[i],
                    displacements = u[i], forces = F[i], reference = crystal)
        for i in eachindex(E)]
ds = SLCEDataset(basis, data)                       # both channels resolve by themselves
f  = fit(SLCEFit, ds, OLS(); torque_weight = 0.3, force_weight = 0.3)
```
"""
function joint_datum(energy::Real;
                     moments::AbstractMatrix{<:Real},
                     field::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                     displacements::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                     forces::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                     reference::Union{Crystal,Nothing} = nothing,
                     reference_id::AbstractString = "reference",
                     zero_moment_atol::Real = 1e-10,
                     provenance::Union{DatumProvenance,Nothing} = nothing)::TrainingDatum
    field === nothing || size(field) == size(moments) ||
        throw(ArgumentError("`field` $(size(field)) must match `moments` " *
                            "$(size(moments))"))
    dirs, mags = _moments_to_dirs(moments; zero_moment_atol = zero_moment_atol)
    if provenance === nothing && reference !== nothing
        provenance = DatumProvenance(; reference_id = reference_id,
                                     reference_fingerprint =
                                         crystal_fingerprint(reference))
    end
    # The keyword constructor owns the torque derivation and the upgrade-only
    # `torque_qualified` gate; both are shared with `spin_datum` so a mixed batch
    # cannot disagree about which rows are admissible.
    return TrainingDatum(; energy = energy, directions = dirs, magmoms = mags,
                         displacements = displacements, forces = forces,
                         field = field, provenance = provenance)
end

"""
    read_configs(src::AbstractDFTSource) -> Vector{TrainingDatum}

Read all training configurations from a DFT source. Implemented per source type
(e.g. `SLCETools.VASP.Oszicar` in the SLCETools.jl package).
"""
read_configs(src::AbstractDFTSource) =
    throw(ArgumentError("read_configs is not implemented for $(typeof(src))"))

# ----------------------------------------------------------------------------
# Dataset-boundary invariants
# ----------------------------------------------------------------------------

# Every atom the SALC basis references must carry a nonzero magnetic moment in every
# configuration: a quenched (‖m‖ ≈ 0) moment on a referenced atom enters the design
# matrix through the ẑ placeholder direction of `spin_datum` and silently biases the
# fit. Unreferenced atoms (species removed with `lmax = 0`, or sites outside every
# admitted cluster) may be non-magnetic — their moments are never consulted.
function _check_referenced_moments(basis::SLCEBasis,
                                   data::AbstractVector{TrainingDatum}; atol::Real)
    ref = _referenced_atoms(basis)
    nat = length(ref)
    labels = basis.crystal.species_labels
    for (i, d) in enumerate(data)
        length(d.magmoms) == nat ||
            throw(DimensionMismatch("config $i has $(length(d.magmoms)) atoms, " *
                                    "basis expects $nat"))
        for a = 1:nat
            (ref[a] && d.magmoms[a] <= atol) || continue
            lab = labels[basis.crystal.species[a]]
            throw(ArgumentError(
                "config $i: atom $a ($lab) has a zero magnetic moment " *
                "(‖m‖ = $(d.magmoms[a]) ≤ $atol) but is referenced by the SALC basis " *
                "— its placeholder ẑ direction would silently bias the fit. Drop the " *
                "configuration, or, if the species is non-magnetic, remove it from " *
                "the basis with lmax = 0"))
        end
    end
    return nothing
end

# One dataset = one computational setup: total energies from different setups
# (collinear vs noncollinear, SOC on/off, different XC/cutoff families) sit on
# different scales, and the offset is family-correlated — it biases the coefficients
# instead of averaging into j0. SOC-sector basis functions do not vanish on Ising
# configurations, so an unmasked mixed fit actively corrupts the SOC coefficients.
# Multi-fidelity use goes through staged fits (per-family datasets + frozen /
# sector_mask), not through one mixed regression.
function _check_setup_uniformity(data::AbstractVector{TrainingDatum})
    p1 = data[1].provenance
    for (i, d) in enumerate(data)
        p = d.provenance
        (p.setup_id == p1.setup_id && p.soc == p1.soc) || throw(ArgumentError(
            "config $i has setup_id = $(repr(p.setup_id)), soc = $(repr(p.soc)) but " *
            "config 1 has setup_id = $(repr(p1.setup_id)), soc = $(repr(p1.soc)) — " *
            "one SLCEDataset must come from one computational setup (energies from " *
            "different setups are not on a common scale). Build one dataset per " *
            "family and fit in stages (frozen / sector_mask) instead"))
    end
    return nothing
end

# (`_basis_has_disp` — the spec-keyed displacement trigger — lives in slce/model.jl
# next to the other basis predicates; it is shared with the predict-layer guards.)

# The double-counting protocol as an invariant (theory paper, protocol rule 1): all
# data must come from ONE clamped-ion reference structure. The trigger is the BASIS,
# not the datum's channels — against a p ≥ 1 basis, even a spin-only datum asserts
# "u = 0 exactly", which is a claim about a geometry and meaningless without naming
# the reference (feeding relaxed-geometry legacy data into a p ≥ 1 basis is exactly
# the −½FᵀΦ⁻¹F double count). Pure-spin bases keep the unpinned path so existing
# spin-only workflows are untouched.
function _check_reference(basis::SLCEBasis, data::AbstractVector{TrainingDatum})
    if _basis_has_disp(basis)
        fp = crystal_fingerprint(basis.crystal)
        ref1 = data[1].provenance.reference_id
        for (i, d) in enumerate(data)
            p = d.provenance
            p.reference_id === nothing && throw(ArgumentError(
                "config $i carries no reference_id, but the basis has " *
                "displacement-decorated (p ≥ 1) SALCs — every datum (spin-only " *
                "included: it asserts u = 0 at the reference) must be pinned to the " *
                "clamped-ion reference crystal via DatumProvenance(reference_id, " *
                "reference_fingerprint)"))
            p.reference_id == ref1 || throw(ArgumentError(
                "config $i is referenced to $(repr(p.reference_id)) but config 1 to " *
                "$(repr(ref1)) — all data in one SLCEDataset must share ONE " *
                "clamped-ion reference (the double-counting protocol)"))
            p.reference_fingerprint == fp || throw(ArgumentError(
                "config $i: reference_fingerprint $(repr(p.reference_fingerprint)) " *
                "does not match crystal_fingerprint(basis.crystal) = \"$fp\" — the " *
                "displacements were measured against a different crystal than the " *
                "basis was built on (same-lineage identity is required; rebuild the " *
                "data or the basis from the same reference)"))
        end
    else
        for (i, d) in enumerate(data)
            d.displacements === nothing || throw(ArgumentError(
                "config $i carries displacements but the basis is pure-spin — its " *
                "energies belong to displaced geometries and would silently corrupt " *
                "a clamped-ion (u = 0) fit. Use a displacement-decorated basis, or " *
                "drop the displaced configurations"))
        end
        if any(d -> d.forces !== nothing, data)
            @warn "force data present but the basis is pure-spin — forces are " *
                  "ignored (no force design block exists for a p = 0 basis)"
        end
    end
    return nothing
end

"""
    SLCEDataset(basis, data::AbstractVector{TrainingDatum}; use_torque = nothing,
               use_force = true, zero_moment_atol = 1e-10) -> SLCEDataset
    SLCEDataset(basis, src::AbstractDFTSource; use_torque = nothing) -> SLCEDataset

Build a fit-ready [`SLCEDataset`](@ref) from training data (or directly from a DFT
source, which is read first). The spin directions become the configurations and the
energies the energy targets; with `use_torque = true` the per-atom torque targets of
every **qualified** configuration — `torques !== nothing` and
`provenance.torque_qualified` — are included for an energy+torque co-fit (see
[`fit`](@ref)). Configurations without qualified torque data contribute energy rows
only, so a mixed dataset (e.g. constrained-noncollinear configs plus
unconstrained/collinear energy-only configs **from the same computational setup**)
is a first-class object: its torque design has rows exactly for the qualified
configurations. Pass `use_torque = false` to force a dataset without torque targets.

`use_torque = nothing` (the default) resolves from the **basis**: `true` when it
carries spin content, `false` for a lattice-only expansion, which has no torque
design block to build and for which demanding torque data is a requirement no
correct call can satisfy. It is deliberately *not* resolved from whether the data
happen to carry torques — that would turn "the adapter dropped the constraining
fields", today a loud error, into a silent energy-only fit.

Against a **displacement-decorated basis** the designs are evaluated jointly at each
configuration's `(e, u)`: a datum's `displacements` become its `u` field (a datum
without displacements contributes `u = 0` exactly — atoms at the pinned clamped-ion
reference), and with `use_force = true` (the default) the per-atom forces
(`f_a = −∂E/∂u_a`, the sign convention pinned in [`TrainingDatum`](@ref)) of every
force-carrying configuration enter the compact force design block `X_F` for a
three-block co-fit. Force rows follow the same ragged bookkeeping as torque rows
(rows only for force-bearing configs, per-row `force_config`), and only atoms some
SALC displacement slot actually reads get rows — forces the model is structurally
blind to are excluded (with a warning when nonzero), never zero-padded. Torque rows
on this path are likewise restricted to spin-referenced atoms (a
displacement-only ligand contributes exactly-zero rows on both sides). The
displacement-radius guard warns when an amplitude exceeds half the shortest
reference interatomic distance (outside the fixed-topology regime, or
un-minimum-imaged input). Pass `use_force = false` for a dataset without force
targets.

Boundary invariants checked here (all fail loudly rather than bias silently):

- **One computational setup**: all data must agree on `provenance.setup_id`/`soc`
  (see [`DatumProvenance`](@ref) — cross-setup energies are not on a common scale).
- **One clamped-ion reference** (displacement-decorated bases): every datum must
  carry the same `reference_id` and a `reference_fingerprint` equal to
  `crystal_fingerprint(basis.crystal)`; against a pure-spin basis, displaced data
  are rejected outright.
- **Referenced atoms stay magnetic**: every atom the SALC basis references must
  carry a nonzero moment (`> zero_moment_atol`, μ_B) in every configuration — a
  quenched moment would enter the fit through the `ẑ` placeholder direction of
  [`spin_datum`](@ref). Atoms the basis never reads are exempt. The guard re-derives
  quenched atoms from the stored magnitudes, so if the data were built with a custom
  `zero_moment_atol`, pass the same value here (both default to `1e-10`).
"""
function SLCEDataset(basis::SLCEBasis, data::AbstractVector{TrainingDatum};
                    use_torque::Union{Bool,Nothing} = nothing, use_force::Bool = true,
                    zero_moment_atol::Real = 1e-10)::SLCEDataset
    isempty(data) && throw(ArgumentError("no training data"))
    # A spin-free basis has no torque design block to build, so demanding torque data
    # for it is a requirement no correct call can satisfy. Resolve from the BASIS,
    # never from whether the data happen to carry torques: reading the data would
    # turn "the adapter dropped the constraining fields" — today a loud error — into
    # a silent energy-only fit.
    ut = use_torque === nothing ? _basis_has_spin(basis) : use_torque
    _check_setup_uniformity(data)
    _check_reference(basis, data)
    _check_referenced_moments(basis, data; atol = zero_moment_atol)
    configs = [d.directions for d in data]
    energies = Float64[d.energy for d in data]
    # Dataset-level identity summary (uniformity was just asserted, so datum 1 is
    # representative); `vcat` re-asserts it when datasets are concatenated.
    p1 = data[1].provenance
    ident = DatumProvenance(; reference_id = p1.reference_id,
                            reference_fingerprint = p1.reference_fingerprint,
                            setup_id = p1.setup_id, soc = p1.soc)
    if !_basis_has_disp(basis)
        # Pure-spin basis: the established spin-configuration path (`use_force` is
        # moot — `_check_reference` already warned if forces are present).
        ut ||
            return SLCEDataset(basis, configs, energies; provenance = ident)
        sel = _qualified_torque_sel(data)
        torques = Matrix{Float64}[data[i].torques::Matrix{Float64} for i in sel]
        return SLCEDataset(basis, configs, energies, torques, sel; provenance = ident)
    end
    return _joint_dataset(basis, data, configs, energies, ident;
                          use_torque = ut, use_force = use_force)
end

# Torque-qualified configuration indices (presence AND provenance qualification),
# with the loud no-torque error and the all-zero-target warning shared by the
# pure-spin and joint branches.
function _qualified_torque_sel(data::AbstractVector{TrainingDatum})::Vector{Int}
    qual = [d.torques !== nothing && d.provenance.torque_qualified for d in data]
    if !any(qual)
        throw(ArgumentError("use_torque = true but no configuration carries " *
                            "qualified torque data (torques present and " *
                            "provenance.torque_qualified) — pass use_torque = false " *
                            "for a dataset without torque targets"))
    end
    sel = findall(qual)
    # `qual` guarantees presence; the assertion narrows the Union{Matrix,Nothing}
    # field type for downstream consumers (and JET).
    torques = Matrix{Float64}[data[i].torques::Matrix{Float64} for i in sel]
    all(t -> all(iszero, t), torques) &&
        @warn "every qualified torque target is exactly zero — the torque block " *
              "carries no signal beyond stationarity; check that the constraining " *
              "fields were read correctly" configs = length(sel)
    return sel
end

# Joint (displacement-decorated) branch: energy / torque designs evaluated at each
# configuration's (e, u), plus the compact force design block. A datum without
# displacements contributes u = 0 exactly (atoms at the pinned reference — data,
# not absence; `_check_reference` has already pinned the reference identity).
function _joint_dataset(basis::SLCEBasis, data::AbstractVector{TrainingDatum},
                        configs::Vector{Matrix{Float64}}, energies::Vector{Float64},
                        ident::DatumProvenance;
                        use_torque::Bool, use_force::Bool)::SLCEDataset
    nat = n_atoms(basis.crystal)
    configs = Matrix{Float64}[copy(c) for c in configs]   # never alias datum fields
    _validate_configs(basis, configs)
    disps = Matrix{Float64}[d.displacements === nothing ? zeros(3, nat) :
                            copy(d.displacements) for d in data]
    for (i, u) in enumerate(disps)
        size(u) == (3, nat) ||
            throw(DimensionMismatch("config $i displacement field is $(size(u, 1)) × " *
                                    "$(size(u, 2)), basis expects 3 × $nat"))
    end
    _check_displacement_radius(basis.crystal, disps)
    X_E = _design_energy(basis, configs, disps)
    m = size(X_E, 2)
    X_T = Matrix{Float64}(undef, 0, m)
    y_T = Float64[]
    tc = Int[]
    if use_torque
        sel = _qualified_torque_sel(data)
        torques = Matrix{Float64}[data[i].torques::Matrix{Float64} for i in sel]
        # Torque rows restricted to spin-referenced atoms — a displacement-only
        # site has no SPIN slot, so its rows are exactly zero in X_T and (m = 0)
        # in y_T; keeping them would only dilute the √(w_T/n_T) whitening. The
        # pure-spin path keeps its historical all-atom layout (oracle parity).
        tref = _referenced_atoms(basis)
        tatoms = findall(tref)
        texcl = findall(.!tref)
        if !isempty(texcl) &&
           any(T -> any(!iszero, @view T[:, texcl]), torques)
            @warn "nonzero torque targets on atoms no SALC spin slot reads — " *
                  "those rows are structurally zero in the model and are excluded " *
                  "from the torque block (check the per-species lmax if these " *
                  "atoms should respond)" atoms = texcl maxlog = 1
        end
        X_T = _design_torque(basis, configs[sel], disps[sel], tatoms)
        y_T = _flatten_atom_rows(torques, tatoms, nat, "torque")
        tc = repeat(sel; inner = 3 * length(tatoms))
    end
    X_F = Matrix{Float64}(undef, 0, 0)
    y_F = Float64[]
    fc = Int[]
    fcols = Int[]
    if use_force
        fsel = findall(d -> d.forces !== nothing, data)
        isempty(fsel) &&
            throw(ArgumentError("use_force = true but no configuration carries " *
                                "force data — pass use_force = false for a dataset " *
                                "without force targets"))
        fcols = _disp_active_cols(basis)
        if isempty(fcols)
            @warn "force data present but the displacement sectors admitted no " *
                  "SALCs — forces are ignored (there are no force design columns)"
            fcols = Int[]
        else
            forces = Matrix{Float64}[data[i].forces::Matrix{Float64} for i in fsel]
            fref = _disp_referenced_atoms(basis)
            fatoms = findall(fref)
            excl = findall(.!fref)
            if !isempty(excl) &&
               any(F -> any(!iszero, @view F[:, excl]), forces)
                @warn "nonzero forces on atoms no SALC displacement slot reads — " *
                      "those rows are structurally zero in the model and are " *
                      "excluded from the force block (check the per-species pmax " *
                      "if these atoms should respond)" atoms = excl maxlog = 1
            end
            all(F -> all(iszero, F), forces) &&
                @warn "every force target is exactly zero — the force block " *
                      "carries no signal; check that the forces were read " *
                      "correctly" configs = length(fsel)
            X_F = _design_force(basis, configs[fsel], disps[fsel], fcols, fatoms)
            # All-zero design ≠ all-zero targets: e.g. every displacement factor of
            # degree ≥ 2 evaluated at u = 0 has zero gradient. The block would then
            # spend its whole weight w_F on 0 = y rows — no information, silently
            # rescaling the energy block.
            all(iszero, X_F) &&
                @warn "the force design block is identically zero at these " *
                      "displacements — no displacement-active SALC responds " *
                      "(e.g. all displacement factors of degree ≥ 2 at u = 0); " *
                      "the force block carries no information at force_weight > 0"
            y_F = _flatten_forces(forces, fatoms, nat)
            fc = repeat(fsel; inner = 3 * length(fatoms))
        end
    end
    # Translation-invariance machinery: built once per basis here (the
    # fit-boundary), applied by `_assemble_problem`. The one recoverable refusal
    # is the AllImages self-image case (same-site products need a Gaunt
    # expansion): the dataset is still constructible, but `fit`'s default
    # `asr = true` will then error rather than silently skip — the user must
    # write `asr = false` deliberately. Any other builder error propagates.
    asrrep = try
        build_asr(basis)
    catch err
        (err isa ArgumentError && occursin("self-image", err.msg)) || rethrow()
        @warn "ASR is unavailable for this basis (AllImages self-image " *
              "clusters); fits must pass asr = false explicitly" maxlog = 1
        nothing
    end
    return SLCEDataset(basis, configs, X_E, energies, X_T, y_T, tc, ident;
                      disps = disps, X_F = X_F, y_F = y_F, force_config = fc,
                      force_cols = fcols, asr = asrrep)
end

# Shortest interatomic distance of the reference crystal under periodic images
# (shifts in [-1, 1]³ — adequate for the mildly skewed cells the guard serves;
# includes the self-image case, i.e. the shortest lattice translation).
function _min_reference_distance(crystal::Crystal)::Float64
    A = crystal.lattice.vectors
    nat = n_atoms(crystal)
    dmin = Inf
    for a = 1:nat, b = a:nat
        df = crystal.frac_positions[:, b] .- crystal.frac_positions[:, a]
        for s1 = -1:1, s2 = -1:1, s3 = -1:1
            (a == b && s1 == 0 && s2 == 0 && s3 == 0) && continue
            d = norm(A * (df .+ SVector{3,Float64}(s1, s2, s3)))
            d < dmin && (dmin = d)
        end
    end
    return dmin
end

# Displacement-radius guard (design record §8): the expansion's cluster topology is
# frozen at the reference, so displacements approaching half the shortest reference
# interatomic distance leave the regime where the fixed-topology polynomial model is
# meaningful (and a displacement of order a lattice vector is the classic
# un-minimum-imaged adapter bug). A warning, not an error — the user may knowingly
# probe large amplitudes.
function _check_displacement_radius(crystal::Crystal,
                                    disps::Vector{Matrix{Float64}})
    thr = 0.5 * _min_reference_distance(crystal)       # finite: self-images always counted
    umax = 0.0
    iworst = 0
    for (i, u) in enumerate(disps)
        for a in axes(u, 2)
            na = norm(@view u[:, a])
            if na > umax
                umax = na
                iworst = i
            end
        end
    end
    if umax > thr
        @warn "displacement amplitude exceeds half the shortest reference " *
              "interatomic distance — the fixed-topology expansion is outside its " *
              "regime (or the displacements were not minimum-imaged against the " *
              "reference)" max_norm = round(umax; sigdigits = 4) threshold =
            round(thr; sigdigits = 4) config = iworst maxlog = 1
    end
    return nothing
end

SLCEDataset(basis::SLCEBasis, src::AbstractDFTSource;
           use_torque::Union{Bool,Nothing} = nothing,
           use_force::Bool = true, zero_moment_atol::Real = 1e-10)::SLCEDataset =
    SLCEDataset(basis, read_configs(src); use_torque = use_torque,
               use_force = use_force, zero_moment_atol = zero_moment_atol)
