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
    energies, moments, fields = _read_embset_blocks(path; n_atoms = n_atoms)
    data = Vector{TrainingDatum}(undef, length(energies))
    for c in eachindex(energies)
        c_flag = any(!iszero, fields[c])   # same derivation as spin_datum's default
        prov = DatumProvenance(; constrained = c_flag, torque_qualified = c_flag,
                               setup_id = setup_id)
        data[c] = spin_datum(energies[c], moments[c], fields[c];
                            zero_moment_atol = zero_moment_atol, provenance = prov)
    end
    return data
end

# The parsing core, shared with `read_embset_pair`: raw per-configuration blocks —
# `(energies, moments, fields)` with `moments[c]`/`fields[c]` the exact 3 × n_atoms
# parsed values (no direction/magnitude decomposition, so a bitwise cross-file
# comparison compares what the file said).
function _read_embset_blocks(path::AbstractString;
                             n_atoms::Union{Integer,Nothing} = nothing)
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
    energies = Vector{Float64}(undef, nconf)
    moments = Vector{Matrix{Float64}}(undef, nconf)
    fields = Vector{Matrix{Float64}}(undef, nconf)
    for c = 1:nconf
        base = blk * (c - 1)
        energies[c] = _embset_number(lines[base + 1], "config $c: energy line", path)
        m = Matrix{Float64}(undef, 3, nat)
        b = Matrix{Float64}(undef, 3, nat)
        for i = 1:nat
            tok = split(lines[base + 1 + i])
            length(tok) >= 7 ||
                throw(ArgumentError("EMBSET file $path, config $c, atom line $i: " *
                                    "expected ≥ 7 columns, got $(length(tok))"))
            index = tryparse(Int, tok[1])
            index == i ||
                throw(ArgumentError("EMBSET file $path, config $c, atom line $i: " *
                                    "index column is \"$(tok[1])\", expected $i"))
            for k = 1:3
                m[k, i] = _embset_number(tok[1 + k],
                                         "config $c, atom $i, moment", path)
                b[k, i] = _embset_number(tok[4 + k],
                                         "config $c, atom $i, field", path)
            end
        end
        moments[c] = m
        fields[c] = b
    end
    return energies, moments, fields
end

"""
    read_embset_pair(mw_path, mint_path; n_atoms = nothing, zero_moment_atol = 1e-10,
                     setup_id = nothing, constraint_mode = nothing,
                     constraint_axes = nothing, sign_gate_min = 5e-3,
                     axis_angle_p99_max = 5.0) -> Vector{TrainingDatum}

The **legacy archive reader**: an `EMBSET` file carrying the smoothed moments (VASP
`MW_int` — the quantity the constraint acts on) paired with its `EMBSET_mint`
sibling carrying the bare moments (`M_int`). New data should be generated as
extended-XYZ ([`write_extxyz`](@ref)); this exists so archived pairs remain
ingestible without regeneration.

Both files are parsed with the full [`read_embset`](@ref) validation (so the index
columns of both files are independently pinned to 1…n_atoms), then the pairing is
verified loudly: same configuration count, same block shape, and the **field blocks
bitwise identical** (both files were written from the same runs, so their
constraining fields must agree to the last bit — a mismatch means the files are not
siblings). The **energy lines are deliberately NOT compared**: the two writers'
energy conventions differ (measured ΔE = 0.148 eV on the FeRh archive), and the
`mw_path` energy is the one used.

The result is [`spin_datum`](@ref)s built from the `mw_path` observables with
`moments_bare` from `mint_path`. `constraint_mode` / `constraint_axes` (a single
`3 × n_atoms` matrix applied to every configuration, or one matrix per
configuration) attach the axis information the EMBSET format cannot carry — for a
mode-1 archive the axes come from the archived per-sample inputs, and the axis
gates ([`check_moment_gates`](@ref)) re-verify them against the converged moments,
exactly as the extxyz path does.
"""
function read_embset_pair(mw_path::AbstractString, mint_path::AbstractString;
                          n_atoms::Union{Integer,Nothing} = nothing,
                          zero_moment_atol::Real = 1e-10,
                          setup_id::Union{AbstractString,Nothing} = nothing,
                          constraint_mode::Union{Integer,Nothing} = nothing,
                          constraint_axes::Union{Nothing,AbstractMatrix{<:Real},
                                                 AbstractVector{<:AbstractMatrix{<:Real}}} = nothing,
                          sign_gate_min::Real = 5e-3,
                          axis_angle_p99_max::Real = 5.0)::Vector{TrainingDatum}
    e_mw, m_mw, b_mw = _read_embset_blocks(mw_path; n_atoms = n_atoms)
    e_mint, m_mint, b_mint = _read_embset_blocks(mint_path; n_atoms = n_atoms)
    # the same file twice (or a byte copy) would pass every sibling check and make
    # moments_bare ≡ the smoothed moments — a pair whose moment blocks are all
    # bitwise equal is not an MW / M_int pair
    (length(e_mw) == length(e_mint) &&
     all(m_mw[c] == m_mint[c] for c in eachindex(e_mw))) &&
        throw(ArgumentError("EMBSET pair: every moment block of $mint_path equals " *
                            "$mw_path's — the two files carry the same moments, not " *
                            "the smoothed MW and the bare M_int of one run"))
    length(e_mw) == length(e_mint) ||
        throw(ArgumentError("EMBSET pair: $(length(e_mw)) configurations in " *
                            "$mw_path vs $(length(e_mint)) in $mint_path — not " *
                            "siblings (the FeRh EMBSET_mint_100 count-mismatch " *
                            "class of provenance bug)"))
    for c in eachindex(e_mw)
        size(m_mw[c]) == size(m_mint[c]) ||
            throw(ArgumentError("EMBSET pair: config $c block shape " *
                                "$(size(m_mw[c])) in $mw_path vs " *
                                "$(size(m_mint[c])) in $mint_path"))
        b_mw[c] == b_mint[c] ||
            throw(ArgumentError("EMBSET pair: config $c constraining-field block " *
                                "differs between $mw_path and $mint_path — the " *
                                "files are not from the same runs (the field is " *
                                "run-identical; only the moment columns differ " *
                                "between the MW and M_int writers)"))
    end
    axes_for = if constraint_axes === nothing
        _ -> nothing
    elseif constraint_axes isa AbstractMatrix
        ax = Matrix{Float64}(constraint_axes)
        _ -> ax
    else
        length(constraint_axes) == length(e_mw) ||
            throw(ArgumentError("EMBSET pair: $(length(constraint_axes)) " *
                                "constraint_axes matrices for $(length(e_mw)) " *
                                "configurations"))
        axv = [Matrix{Float64}(a) for a in constraint_axes]
        c -> axv[c]
    end
    data = Vector{TrainingDatum}(undef, length(e_mw))
    for c in eachindex(e_mw)
        c_flag = any(!iszero, b_mw[c])
        prov = DatumProvenance(; constrained = c_flag, torque_qualified = c_flag,
                               setup_id = setup_id)
        data[c] = spin_datum(e_mw[c], m_mw[c], b_mw[c];
                            zero_moment_atol = zero_moment_atol,
                            moments_bare = m_mint[c],
                            constraint_axes = axes_for(c),
                            constraint_mode = constraint_mode,
                            provenance = prov)
    end
    check_moment_gates(data; sign_gate_min = sign_gate_min,
                       axis_angle_p99_max = axis_angle_p99_max,
                       label = "read_embset_pair($mw_path, $mint_path)")
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
