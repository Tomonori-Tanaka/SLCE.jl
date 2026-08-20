# Extended-XYZ (ASE dialect) training-data container — the canonical on-disk format
# for constrained-noncollinear spin (and joint spin–lattice) training sets. One frame
# per configuration: a `Lattice`/`Properties` info line plus one line per atom. The
# design decisions (recorded in the adiabatic-moment design record, D8):
#
#   * The structure (lattice + positions) is ALWAYS stored, even for spin-only data —
#     self-containment structurally removes the "which POSCAR pairs with which
#     EMBSET" provenance-bug class. The redundancy costs a few MB.
#   * spin-only vs joint is decided FROM THE DATA (positions bitwise identical across
#     frames → spin-only), never from a flag; a `config_type` claim is allowed but a
#     mismatch with the measured answer is a loud error.
#   * Per-atom columns exist only when observed: `species:S:1:pos:R:3:mw:R:3`
#     [`:bcon:R:3`][`:mint:R:3`][`:mconstr:R:3`][`:forces:R:3`], 1:1 with the
#     `Union{…,Nothing}` channels of `TrainingDatum` (`mw` = smoothed moment vectors
#     `magmoms .* directions`; `bcon` = constraining field; `mint` = bare moments;
#     `mconstr` = constraint axes).
#   * A generated file is self-contained INCLUDING the constraint axes: the axis
#     gates ([`check_moment_gates`](@ref)) run at generation and again at every load,
#     so archived axes are re-verified, never believed.
#   * DFT-code vocabulary stays in the generator (SLCETools' `oszicar_to_extxyz`);
#     this reader/writer is code-neutral.

"""
    ExtxyzFile(path; reference = nothing, zero_moment_atol = 1e-10,
               sign_gate_min = 5e-3, axis_angle_p99_max = 5.0)

An [`AbstractDFTSource`](@ref) for an extended-XYZ training-set file, so it drops
straight into the pipeline:

```julia
dataset = SLCEDataset(basis, ExtxyzFile("train.extxyz"))
```

`read_configs` on it is [`read_extxyz`](@ref)`(path; ...)` — see there for the format
and the keyword arguments.
"""
struct ExtxyzFile <: AbstractDFTSource
    path::String
    reference::Union{Crystal,Nothing}
    zero_moment_atol::Float64
    sign_gate_min::Float64
    axis_angle_p99_max::Float64
end

function ExtxyzFile(path::AbstractString; reference::Union{Crystal,Nothing} = nothing,
                    zero_moment_atol::Real = 1e-10, sign_gate_min::Real = 5e-3,
                    axis_angle_p99_max::Real = 5.0)::ExtxyzFile
    return ExtxyzFile(String(path), reference, Float64(zero_moment_atol),
                      Float64(sign_gate_min), Float64(axis_angle_p99_max))
end

read_configs(src::ExtxyzFile)::Vector{TrainingDatum} =
    read_extxyz(src.path; reference = src.reference,
                zero_moment_atol = src.zero_moment_atol,
                sign_gate_min = src.sign_gate_min,
                axis_angle_p99_max = src.axis_angle_p99_max)

# ── info-line lexing ───────────────────────────────────────────────────────────────

# Split an extxyz info line into `key = value` pairs; values may be double-quoted
# (quotes stripped, spaces preserved). Insertion order is irrelevant to the reader.
function _xyz_info(line::AbstractString, path::AbstractString,
                   frame::Int)::Dict{String,String}
    out = Dict{String,String}()
    i = firstindex(line)
    n = lastindex(line)
    while i <= n
        c = line[i]
        if isspace(c)
            i = nextind(line, i)
            continue
        end
        j = i                      # key runs to '='
        while j <= n && line[j] != '='
            isspace(line[j]) &&
                throw(ArgumentError("extxyz $path frame $frame: bare token " *
                                    "\"$(line[i:prevind(line, j)])\" in the info " *
                                    "line (expected key=value pairs)"))
            j = nextind(line, j)
        end
        j <= n || throw(ArgumentError("extxyz $path frame $frame: key without " *
                                      "'=' at the end of the info line"))
        key = line[i:prevind(line, j)]
        isempty(key) && throw(ArgumentError("extxyz $path frame $frame: empty key " *
                                            "in the info line"))
        i = nextind(line, j)       # past '='
        if i <= n && line[i] == '"'
            i = nextind(line, i)
            k = i
            while k <= n && line[k] != '"'
                k = nextind(line, k)
            end
            k <= n || throw(ArgumentError("extxyz $path frame $frame: unterminated " *
                                          "quote in the info line (key \"$key\")"))
            out[key] = line[i:prevind(line, k)]
            i = nextind(line, k)
        else
            k = i
            while k <= n && !isspace(line[k])
                k = nextind(line, k)
            end
            out[key] = line[i:prevind(line, k)]
            i = k
        end
    end
    return out
end

# Parse `Properties=species:S:1:pos:R:3:…` into an ordered `(name, type, ncols)` list.
function _xyz_properties(spec::AbstractString, path::AbstractString,
                         frame::Int)::Vector{Tuple{String,Char,Int}}
    tok = split(spec, ':')
    length(tok) % 3 == 0 && !isempty(tok) ||
        throw(ArgumentError("extxyz $path frame $frame: malformed Properties " *
                            "\"$spec\" (need name:type:count triples)"))
    props = Tuple{String,Char,Int}[]
    for t = 1:3:length(tok)
        name = String(tok[t])
        ty = tok[t + 1]
        length(ty) == 1 && ty[1] in ('S', 'R', 'I', 'L') ||
            throw(ArgumentError("extxyz $path frame $frame: unsupported column " *
                                "type \"$ty\" for \"$name\""))
        nc = tryparse(Int, tok[t + 2])
        (nc === nothing || nc < 1) &&
            throw(ArgumentError("extxyz $path frame $frame: bad column count " *
                                "\"$(tok[t + 2])\" for \"$name\""))
        push!(props, (name, ty[1], nc))
    end
    return props
end

_xyz_number(s::AbstractString, what::String, path::AbstractString)::Float64 = begin
    v = tryparse(Float64, s)
    (v === nothing || !isfinite(v)) &&
        throw(ArgumentError("extxyz $path: cannot parse $what from \"$s\"" *
                            (v === nothing ? "" : " (non-finite)")))
    v
end

# One parsed frame: species labels, per-column 3 × nat blocks, and the info dict.
struct _XyzFrame
    species::Vector{String}
    cols::Dict{String,Matrix{Float64}}
    info::Dict{String,String}
end

function _read_xyz_frames(path::AbstractString)::Vector{_XyzFrame}
    isfile(path) || throw(ArgumentError("no such extxyz file: $path"))
    lines = readlines(path)
    frames = _XyzFrame[]
    i = 1
    while i <= length(lines)
        if isempty(strip(lines[i]))
            i += 1
            continue
        end
        frame = length(frames) + 1
        nat = tryparse(Int, strip(lines[i]))
        (nat === nothing || nat < 1) &&
            throw(ArgumentError("extxyz $path frame $frame: expected an atom count " *
                                "line, got \"$(lines[i])\""))
        i + 1 <= length(lines) ||
            throw(ArgumentError("extxyz $path frame $frame: missing info line"))
        info = _xyz_info(lines[i + 1], path, frame)
        haskey(info, "Properties") ||
            throw(ArgumentError("extxyz $path frame $frame: no Properties key"))
        props = _xyz_properties(info["Properties"], path, frame)
        props[1][1] == "species" && props[1][2] == 'S' && props[1][3] == 1 ||
            throw(ArgumentError("extxyz $path frame $frame: the first property " *
                                "must be species:S:1"))
        ntok = sum(p[3] for p in props)
        i + 1 + nat <= length(lines) ||
            throw(ArgumentError("extxyz $path frame $frame: truncated — $nat atom " *
                                "lines declared, file ends early"))
        species = Vector{String}(undef, nat)
        cols = Dict{String,Matrix{Float64}}(p[1] => Matrix{Float64}(undef, p[3], nat)
                                            for p in props if p[2] != 'S')
        for a = 1:nat
            tok = split(lines[i + 1 + a])
            length(tok) == ntok ||
                throw(ArgumentError("extxyz $path frame $frame atom $a: expected " *
                                    "$ntok columns, got $(length(tok))"))
            off = 0
            for (name, ty, nc) in props
                if ty == 'S'
                    species[a] = String(tok[off + 1])
                else
                    m = cols[name]
                    for k = 1:nc
                        m[k, a] = _xyz_number(tok[off + k],
                                              "frame $frame atom $a column $name",
                                              path)
                    end
                end
                off += nc
            end
        end
        push!(frames, _XyzFrame(species, cols, info))
        i += 2 + nat
    end
    isempty(frames) && throw(ArgumentError("extxyz $path: no frames"))
    return frames
end

_xyz_lattice(info::Dict{String,String}, path::AbstractString, frame::Int) = begin
    haskey(info, "Lattice") ||
        throw(ArgumentError("extxyz $path frame $frame: no Lattice key — the " *
                            "structure is always stored (self-containment rule)"))
    v = split(info["Lattice"])
    length(v) == 9 ||
        throw(ArgumentError("extxyz $path frame $frame: Lattice needs 9 numbers"))
    A = Matrix{Float64}(undef, 3, 3)
    for c = 1:3, r = 1:3
        A[r, c] = _xyz_number(v[3 * (c - 1) + r], "frame $frame Lattice", path)
    end
    A                                       # columns = lattice vectors
end

# ── reader ─────────────────────────────────────────────────────────────────────────

"""
    read_extxyz(path; reference = nothing, zero_moment_atol = 1e-10,
                sign_gate_min = 5e-3, axis_angle_p99_max = 5.0)
        -> Vector{TrainingDatum}

Read an extended-XYZ training-set file (the format [`write_extxyz`](@ref) emits; ASE
dialect) into [`TrainingDatum`](@ref)s. Per-atom columns map 1:1 onto the datum's
channels: `mw` → `magmoms .* directions` (the smoothed moment decomposition), `bcon`
→ `field`, `mint` → `moments_bare`, `mconstr` → `constraint_axes`, `forces` →
`forces`. Info keys consumed: `energy` (required, eV), `constraint_mode` (`1`/`4`,
required when `mconstr` columns are present), `units_field` (must be `eV/muB` when
present), `setup_id` / `soc` (stamped into the provenance), `config_type`
(`spin-only`/`joint`, an optional claim), `pbc` (must be fully periodic).

**spin-only vs joint is measured, not read**: with `reference = nothing`, every
frame's positions must be bitwise identical (spin-only; `displacements = nothing`
exactly) — differing positions are an error asking for the reference. With a
`reference::Crystal`, displacements `u = pos − ref` are computed per frame; if every
`u` is exactly zero the data are spin-only, else joint (each datum stamped with the
reference id + [`crystal_fingerprint`](@ref)). A `config_type` claim that contradicts
the measured answer is a loud error.

Every load re-runs the axis-consistency gates ([`check_moment_gates`](@ref)) — the
archived constraint axes are re-verified against the converged moment directions,
never believed.
"""
function read_extxyz(path::AbstractString;
                     reference::Union{Crystal,Nothing} = nothing,
                     zero_moment_atol::Real = 1e-10,
                     sign_gate_min::Real = 5e-3,
                     axis_angle_p99_max::Real = 5.0)::Vector{TrainingDatum}
    frames = _read_xyz_frames(path)
    nat = length(frames[1].species)

    # cross-frame consistency: one file = one structure family
    A1 = _xyz_lattice(frames[1].info, path, 1)
    for (f, fr) in enumerate(frames)
        length(fr.species) == nat ||
            throw(ArgumentError("extxyz $path frame $f: $(length(fr.species)) " *
                                "atoms, frame 1 has $nat"))
        fr.species == frames[1].species ||
            throw(ArgumentError("extxyz $path frame $f: species differ from frame 1"))
        _xyz_lattice(fr.info, path, f) == A1 ||
            throw(ArgumentError("extxyz $path frame $f: Lattice differs from " *
                                "frame 1 (varying cells are not supported here)"))
        haskey(fr.cols, "pos") && size(fr.cols["pos"], 1) == 3 ||
            throw(ArgumentError("extxyz $path frame $f: no pos:R:3 columns"))
        haskey(fr.cols, "mw") ||
            throw(ArgumentError("extxyz $path frame $f: no mw:R:3 columns (the " *
                                "smoothed moment channel is required)"))
        for key in ("mw", "bcon", "mint", "mconstr", "forces")
            haskey(fr.cols, key) == haskey(frames[1].cols, key) ||
                throw(ArgumentError("extxyz $path frame $f: column \"$key\" " *
                                    "presence differs from frame 1 (one file = " *
                                    "one observation set)"))
            haskey(fr.cols, key) && size(fr.cols[key], 1) != 3 &&
                throw(ArgumentError("extxyz $path frame $f: column \"$key\" must " *
                                    "be :R:3"))
        end
        pbc = get(fr.info, "pbc", "T T T")
        pbc == "T T T" ||
            throw(ArgumentError("extxyz $path frame $f: pbc = \"$pbc\" — only " *
                                "fully periodic cells are supported"))
        uf = get(fr.info, "units_field", "eV/muB")
        uf == "eV/muB" ||
            throw(ArgumentError("extxyz $path frame $f: units_field = \"$uf\" — " *
                                "the constraining field must be in eV/muB (the " *
                                "generator converts; a \"T\" here is the header " *
                                "mislabel this key exists to correct)"))
    end

    # constraint_mode: uniform across frames, required iff mconstr columns exist
    modes = [get(fr.info, "constraint_mode", nothing) for fr in frames]
    allequal(modes) ||
        throw(ArgumentError("extxyz $path: constraint_mode differs across frames " *
                            "(one file = one constraint scheme)"))
    cmode = nothing
    m1 = modes[1]              # bound local: the !== nothing narrowing must be
    if m1 !== nothing          # inference-visible (JET: tryparse(Int, ::Nothing))
        cmode = tryparse(Int, m1)
        cmode === nothing &&
            throw(ArgumentError("extxyz $path: constraint_mode = \"$m1\" " *
                                "is not an integer"))
    end
    haskey(frames[1].cols, "mconstr") && cmode === nothing &&
        throw(ArgumentError("extxyz $path: mconstr columns without a " *
                            "constraint_mode info key — the axis rule is keyed by " *
                            "the mode, declare it (1 = transverse-penalty type, " *
                            "4 = direction-pinning type)"))

    # spin-only vs joint: measured from positions
    ref_pos = frames[1].cols["pos"]
    identical = all(fr.cols["pos"] == ref_pos for fr in frames)
    joint = false
    disps = Vector{Union{Matrix{Float64},Nothing}}(undef, length(frames))
    if reference === nothing
        identical ||
            throw(ArgumentError("extxyz $path: positions differ across frames — " *
                                "displaced (joint) data need the clamped-ion " *
                                "reference: pass `reference::Crystal`"))
        fill!(disps, nothing)
    else
        size(reference.frac_positions, 2) == nat ||
            throw(ArgumentError("extxyz $path: $nat atoms per frame, reference " *
                                "crystal has $(size(reference.frac_positions, 2))"))
        reflab = [reference.species_labels[s] for s in reference.species]
        reflab == frames[1].species ||
            throw(ArgumentError("extxyz $path: species differ from the reference " *
                                "crystal ($(frames[1].species[1]) … vs " *
                                "$(reflab[1]) …)"))
        refc = Matrix(cartesian_positions(reference))
        maximum(abs, Matrix(reference.lattice.vectors) - A1) <= 1e-8 ||
            throw(ArgumentError("extxyz $path: Lattice differs from the reference " *
                                "crystal's lattice"))
        for (f, fr) in enumerate(frames)
            u = fr.cols["pos"] - refc
            disps[f] = all(iszero, u) ? nothing : u
        end
        joint = any(u -> u !== nothing, disps)
        if joint                                 # a joint set materializes u = 0
            for f in eachindex(disps)
                disps[f] === nothing && (disps[f] = zeros(3, nat))
            end
        end
    end

    # config_type: an optional claim, checked against the measurement
    claims = unique(get(fr.info, "config_type", nothing) for fr in frames)
    claim = length(claims) == 1 ? claims[1] :
            throw(ArgumentError("extxyz $path: config_type differs across frames"))
    if claim !== nothing
        claim in ("spin-only", "joint") ||
            throw(ArgumentError("extxyz $path: config_type = \"$claim\" (expected " *
                                "spin-only or joint)"))
        measured = joint ? "joint" : "spin-only"
        claim == measured ||
            throw(ArgumentError("extxyz $path: config_type claims \"$claim\" but " *
                                "the positions say \"$measured\" — the flag never " *
                                "overrides the measurement"))
    end

    setups = unique(get(fr.info, "setup_id", nothing) for fr in frames)
    length(setups) == 1 ||
        throw(ArgumentError("extxyz $path: setup_id differs across frames (one " *
                            "file = one computational setup)"))
    socs = unique(get(fr.info, "soc", nothing) for fr in frames)
    length(socs) == 1 ||
        throw(ArgumentError("extxyz $path: soc differs across frames"))
    soc = socs[1] === nothing ? nothing :
          socs[1] in ("true", "T") ? true :
          socs[1] in ("false", "F") ? false :
          throw(ArgumentError("extxyz $path: soc = \"$(socs[1])\" is not a boolean"))

    # `joint ⇒ reference isa Crystal` (joint is only set in the reference branch),
    # but that guard is a runtime fact inference cannot carry into the loop's
    # ternary — hoist the fingerprint so the Nothing arm never reaches the call
    # (JET: crystal_fingerprint(::Nothing)).
    ref_fp = reference === nothing ? "" : crystal_fingerprint(reference)
    data = Vector{TrainingDatum}(undef, length(frames))
    for (f, fr) in enumerate(frames)
        haskey(fr.info, "energy") ||
            throw(ArgumentError("extxyz $path frame $f: no energy key"))
        energy = _xyz_number(fr.info["energy"], "frame $f energy", path)
        dirs, mags = _moments_to_dirs(fr.cols["mw"];
                                      zero_moment_atol = zero_moment_atol)
        prov = joint ?
               DatumProvenance(; reference_id = get(fr.info, "reference_id",
                                                    "reference"),
                               reference_fingerprint = ref_fp,
                               setup_id = setups[1], soc = soc) :
               DatumProvenance(; setup_id = setups[1], soc = soc,
                               constrained = haskey(fr.cols, "bcon") &&
                                             any(!iszero, fr.cols["bcon"]),
                               torque_qualified = haskey(fr.cols, "bcon") &&
                                                  any(!iszero, fr.cols["bcon"]))
        data[f] = TrainingDatum(; energy = energy, directions = dirs,
                                magmoms = mags,
                                displacements = disps[f],
                                forces = get(fr.cols, "forces", nothing),
                                field = get(fr.cols, "bcon", nothing),
                                moments_bare = get(fr.cols, "mint", nothing),
                                constraint_axes = get(fr.cols, "mconstr", nothing),
                                constraint_mode = cmode,
                                provenance = prov)
    end
    check_moment_gates(data; sign_gate_min = sign_gate_min,
                       axis_angle_p99_max = axis_angle_p99_max,
                       label = "read_extxyz($path)")
    return data
end

# ── writer ─────────────────────────────────────────────────────────────────────────

"""
    write_extxyz(path, data, crystal; field_sign = nothing, source = nothing,
                 comment = nothing) -> Nothing

Write [`TrainingDatum`](@ref)s as an extended-XYZ training-set file (the format
[`read_extxyz`](@ref) reads back; numbers are printed shortest-round-trip, so every
stored value survives the file bit-exactly — the one non-bitwise step in a datum
round-trip is re-deriving `directions`/`magmoms` from the written moment vectors,
exact up to the unit normalization). The structure comes from `crystal` (the clamped-ion
reference): positions are `reference + displacements` per frame, and spin-only data
(`displacements === nothing`) get the reference positions verbatim — the structure
is **always** stored, self-containment being the point of the format.

Channel presence must be uniform across `data` (one file = one observation set), the
`constraint_mode` must be uniform, and the setup identity must be uniform (same rule
as `SLCEDataset`). The axis-consistency gates ([`check_moment_gates`](@ref)) run
before anything is written — a file that would fail its own load gate is never
produced. `field_sign` / `source` are provenance strings recorded verbatim in the
info line (the generator documents there which sign convention it normalized from,
and what it read).
"""
function write_extxyz(path::AbstractString, data::AbstractVector{TrainingDatum},
                      crystal::Crystal;
                      field_sign::Union{Nothing,AbstractString} = nothing,
                      source::Union{Nothing,AbstractString} = nothing,
                      comment::Union{Nothing,AbstractString} = nothing,
                      sign_gate_min::Real = 5e-3,
                      axis_angle_p99_max::Real = 5.0)::Nothing
    isempty(data) && throw(ArgumentError("write_extxyz: no data"))
    nat = size(crystal.frac_positions, 2)
    for (c, d) in enumerate(data)
        size(d.directions, 2) == nat ||
            throw(ArgumentError("write_extxyz: config $c has " *
                                "$(size(d.directions, 2)) atoms, crystal has $nat"))
    end
    for (name, get_ch) in (("bcon", d -> d.field), ("mint", d -> d.moments_bare),
                           ("mconstr", d -> d.constraint_axes),
                           ("forces", d -> d.forces),
                           ("displacements", d -> d.displacements))
        present = get_ch(data[1]) !== nothing
        all((get_ch(d) !== nothing) == present for d in data) ||
            throw(ArgumentError("write_extxyz: channel \"$name\" is present on " *
                                "some configs and absent on others — one file = " *
                                "one observation set"))
    end
    allequal(d.constraint_mode for d in data) ||
        throw(ArgumentError("write_extxyz: constraint_mode differs across configs " *
                            "(one file = one constraint scheme)"))
    _check_setup_uniformity(data)
    check_moment_gates(data; sign_gate_min = sign_gate_min,
                       axis_angle_p99_max = axis_angle_p99_max,
                       label = "write_extxyz($path)")

    cmode = data[1].constraint_mode
    joint = data[1].displacements !== nothing
    has_bcon = data[1].field !== nothing
    has_mint = data[1].moments_bare !== nothing
    has_mconstr = data[1].constraint_axes !== nothing
    has_forces = data[1].forces !== nothing

    A = Matrix(crystal.lattice.vectors)
    latstr = join(string.(vec(A)), " ")         # columns = lattice vectors
    refc = Matrix(cartesian_positions(crystal))
    labels = [crystal.species_labels[s] for s in crystal.species]
    props = "species:S:1:pos:R:3:mw:R:3" * (has_bcon ? ":bcon:R:3" : "") *
            (has_mint ? ":mint:R:3" : "") * (has_mconstr ? ":mconstr:R:3" : "") *
            (has_forces ? ":forces:R:3" : "")
    prov = data[1].provenance

    open(path, "w") do io
        for d in data
            println(io, nat)
            print(io, "Lattice=\"", latstr, "\" Properties=", props,
                  " energy=", string(d.energy), " pbc=\"T T T\" config_type=",
                  joint ? "joint" : "spin-only")
            cmode === nothing || print(io, " constraint_mode=", cmode)
            has_bcon && print(io, " units_field=eV/muB")
            field_sign === nothing || print(io, " field_sign=", field_sign)
            prov.setup_id === nothing || print(io, " setup_id=", prov.setup_id)
            prov.soc === nothing || print(io, " soc=", prov.soc ? "true" : "false")
            joint && print(io, " reference_id=",
                           prov.reference_id === nothing ? "reference" :
                           prov.reference_id)
            source === nothing || print(io, " source=", source)
            comment === nothing || print(io, " comment=\"", comment, "\"")
            println(io)
            u = d.displacements
            for a = 1:nat
                print(io, labels[a])
                for k = 1:3
                    print(io, " ", string(refc[k, a] + (u === nothing ? 0.0 :
                                                        u[k, a])))
                end
                for k = 1:3
                    print(io, " ", string(d.magmoms[a] * d.directions[k, a]))
                end
                for (flag, m) in ((has_bcon, d.field), (has_mint, d.moments_bare),
                                  (has_mconstr, d.constraint_axes),
                                  (has_forces, d.forces))
                    flag || continue
                    for k = 1:3
                        print(io, " ", string(m[k, a]))
                    end
                end
                println(io)
            end
        end
    end
    return nothing
end
