# Reader for Magesty's EMBSET training-set format — the legacy plain-text exchange
# format for constrained-noncollinear DFT spin configurations. Kept in the core (not
# an SLCETools adapter) because the format is DFT-code-agnostic: it carries exactly
# the observables a spin-only `TrainingDatum` stores (energy, per-atom moment and
# constraining-field vectors) with no code-specific metadata.

"""
    EmbsetFile(path; n_atoms = nothing, zero_moment_atol = 1e-10, setup_id = nothing)

An [`AbstractDFTSource`](@ref) for a Magesty EMBSET training-set file, so a legacy
training set drops straight into the pipeline:

```julia
dataset = SLCEDataset(basis, EmbsetFile("EMBSET"))
```

`read_configs` on it is [`read_embset`](@ref)`(path; ...)` — see there for the format
and the keyword arguments.
"""
struct EmbsetFile <: AbstractDFTSource
    path::String
    n_atoms::Union{Int,Nothing}
    zero_moment_atol::Float64
    setup_id::Union{String,Nothing}
end

function EmbsetFile(path::AbstractString; n_atoms::Union{Integer,Nothing} = nothing,
                    zero_moment_atol::Real = 1e-10,
                    setup_id::Union{AbstractString,Nothing} = nothing)::EmbsetFile
    return EmbsetFile(String(path), n_atoms === nothing ? nothing : Int(n_atoms),
                      Float64(zero_moment_atol),
                      setup_id === nothing ? nothing : String(setup_id))
end

read_configs(src::EmbsetFile)::Vector{TrainingDatum} =
    read_embset(src.path; n_atoms = src.n_atoms,
                zero_moment_atol = src.zero_moment_atol, setup_id = src.setup_id)

"""
    read_embset(path; n_atoms = nothing, zero_moment_atol = 1e-10,
                setup_id = nothing) -> Vector{TrainingDatum}

Read a Magesty EMBSET training-set file into spin-only [`TrainingDatum`](@ref)s
(via the [`spin_datum`](@ref) constructor). `setup_id`, when given, is stamped into
every datum's [`DatumProvenance`](@ref) (one EMBSET file = one computational setup).

The format (as written by Magesty's `oszicar_to_embset`): `#` comment lines and blank
lines are ignored; what remains is a sequence of per-configuration blocks, each one
**energy line** (a single number, eV) followed by **one line per atom** with at least
7 whitespace-separated columns

    index   m_x m_y m_z   B_x B_y B_z

(the magnetic moment in μ_B and the constraining field in eV/μ_B; extra columns are
ignored). The atom count is auto-detected from the first block — the run of
multi-column lines after the first energy line — and every block must then have the
same shape; pass `n_atoms` to override the detection. Every numeric field must be
finite (a `NaN`/`Inf` — e.g. from a failed SCF — is an error, not a silent training
row). Two deliberate deviations from Magesty's reader: detection is token-based, so a
file *without* `#` block separators also parses (Magesty requires them), and the
1-based `index` column must match the atom's position within its block (Magesty
ignores the column; requiring it makes a permuted or corrupted file fail loudly
instead of silently reassigning moments).

Torques are derived as `τ_a = m_a × B_a` by the [`spin_datum`](@ref) constructor, which
also receives `zero_moment_atol` (the ẑ-placeholder threshold for non-magnetic atoms —
`SLCEDataset` then rejects a placeholder on any basis-referenced atom). Provenance
follows the `spin_datum` derivation: a configuration whose field block is entirely
zero gets `torque_qualified = false` (its torque rows are not admitted into a
co-fit), the rest `true`.
"""
function read_embset(path::AbstractString; n_atoms::Union{Integer,Nothing} = nothing,
                     zero_moment_atol::Real = 1e-10,
                     setup_id::Union{AbstractString,Nothing} = nothing,
                     )::Vector{TrainingDatum}
    isfile(path) || throw(ArgumentError("no such EMBSET file: $path"))
    lines = String[]
    for raw in eachline(path)
        s = strip(raw)
        (isempty(s) || startswith(s, "#")) && continue
        push!(lines, String(s))
    end
    isempty(lines) && throw(ArgumentError("EMBSET file $path has no data lines"))

    nat = n_atoms === nothing ? _embset_detect_natoms(lines, path) : Int(n_atoms)
    nat >= 1 || throw(ArgumentError("n_atoms must be ≥ 1 (got $nat)"))
    blk = nat + 1
    length(lines) % blk == 0 ||
        throw(ArgumentError("EMBSET file $path: $(length(lines)) data lines are not " *
                            "a multiple of n_atoms + 1 = $blk"))

    nconf = length(lines) ÷ blk
    data = Vector{TrainingDatum}(undef, nconf)
    moments = Matrix{Float64}(undef, 3, nat)
    field = Matrix{Float64}(undef, 3, nat)
    for c = 1:nconf
        base = blk * (c - 1)
        energy = _embset_number(lines[base + 1], "config $c: energy line", path)
        for i = 1:nat
            tok = split(lines[base + 1 + i])
            length(tok) >= 7 ||
                throw(ArgumentError("EMBSET file $path, config $c, atom line $i: " *
                                    "expected ≥ 7 columns, got $(length(tok))"))
            idx = tryparse(Int, tok[1])
            idx == i ||
                throw(ArgumentError("EMBSET file $path, config $c, atom line $i: " *
                                    "index column is \"$(tok[1])\", expected $i"))
            for k = 1:3
                moments[k, i] = _embset_number(tok[1 + k],
                                               "config $c, atom $i, moment", path)
                field[k, i] = _embset_number(tok[4 + k],
                                             "config $c, atom $i, field", path)
            end
        end
        c_flag = any(!iszero, field)   # same derivation as spin_datum's default
        prov = DatumProvenance(; constrained = c_flag, torque_qualified = c_flag,
                               setup_id = setup_id)
        data[c] = spin_datum(energy, moments, field;
                            zero_moment_atol = zero_moment_atol, provenance = prov)
    end
    return data
end

function _embset_number(s::AbstractString, what::String, path::AbstractString)::Float64
    v = tryparse(Float64, s)
    (v === nothing || !isfinite(v)) &&
        throw(ArgumentError("EMBSET file $path: cannot parse $what from \"$s\"" *
                            (v === nothing ? "" : " (non-finite — a failed SCF?)")))
    return v
end

# The atom count of the first block: line 1 must be an energy line (a single number),
# followed by the run of atom lines — anything that itself parses as a single number
# starts the next block. Robust to files with or without `#` block separators.
function _embset_detect_natoms(lines::Vector{String}, path::AbstractString)::Int
    tryparse(Float64, lines[1]) !== nothing ||
        throw(ArgumentError("EMBSET file $path: first data line is not a single " *
                            "energy value: \"$(lines[1])\""))
    nat = 0
    for s in Iterators.drop(lines, 1)
        tryparse(Float64, s) === nothing || break
        nat += 1
    end
    nat >= 1 ||
        throw(ArgumentError("EMBSET file $path: no atom lines after the first " *
                            "energy line"))
    return nat
end
