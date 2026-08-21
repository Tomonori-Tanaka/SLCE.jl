"""
Persistence (basis / model serialization).

The on-disk format is a **self-contained, human-readable** document: the crystal,
the symmetry operations, the interaction spec, and the *full* SALC basis (every
member, term, and folded coefficient tensor) — so a reload reconstructs the basis
verbatim, without re-running the (expensive) symmetry projection and without
depending on the construction code's gauge convention staying fixed. A model
document adds the reference energy `j0` and the SALC coefficients `jϕ`, each tagged
by its [`SALCKey`](@ref) so they re-pair to the basis **by key**, not by position.

The document is a plain `Dict{String,Any}` tree (`_to_doc` / `*_from_doc`), kept
deliberately format-agnostic so it can be unit-tested without any serializer and
re-backed by another format later. It is serialized as **TOML** (Julia's standard
library — no external dependency; Float64 round-trips exactly) by [`save`](@ref) /
[`load`](@ref).
"""

const _SCHEMA_BASIS = "slce/basis"
const _SCHEMA_MODEL = "slce/model"
# Documents written before the package was renamed carry the old tags. Reading
# them costs two dictionary entries; refusing them would strand every model
# already saved to disk, which is a different thing from an API rename.
const _LEGACY_SCHEMA_TAGS = Dict("scefitting/sce-basis" => _SCHEMA_BASIS,
                                 "scefitting/sce-model" => _SCHEMA_MODEL)
# v2: the basis-spec section key renamed "interaction" → "spec" (with the SLCEBasis field).
# v3: BasisSpec canonical truncation — "pair_cutoff" replaced by per-body-order
#     "cutoff" matrices (species × species, Inf = no cutoff), plus "lsum" (per body
#     order, typemax(Int64) = uncapped) and "species_labels". v2 docs are still
#     readable (`_spec_from` expands the legacy scalar).
# v4: SALC members stored in the canonical duplicate-free form
#     (`_canonicalize_members`): one member per physical cluster instance instead of
#     one per ordered image (up to N! smaller). Older docs are still readable —
#     `_basis_from_doc` folds v2/v3 members on load (exact regrouping; SALC keys,
#     coefficients, and the fingerprint are unaffected).
# v5: joint spin–lattice key layout — SALC keys store the decoration multiset
#     ("decors": per-site [spin_l, disp_k, disp_l] triples, 0 = channel absent)
#     and the total spin rank "L_S" instead of "ls". v4 keys are mapped on read
#     (per-site l → a pure-spin decor, L_S := Lf — total and value-preserving),
#     so v4 files load with identical predictions and no migration tool.
#     The spec section stores "soc" (= !isotropy), "pmax", "disp_scale", and the
#     resolved "sectors" table; legacy "isotropy"-keyed spec docs are still
#     readable (`_spec_from` branches on the "soc" key, not the version).
# v6: the sector-table key "nbody" renamed "sites" (a per-sector decorated-site
#     COUNT, never the BasisSpec-style body-order cap), and the schema tags moved
#     from "scefitting/sce-*" to "slce/*". Early-v5 documents are read through the
#     legacy key and the legacy tags.
const PERSIST_SCHEMA_VERSION = 6
const _PERSIST_READABLE_VERSIONS = (2, 3, 4, 5, 6)

# Normalize -0.0 → +0.0 so two builds of the same object serialize byte-identically
# (eigensolvers on different BLAS can flip a sign of zero); -0.0 == 0.0 anyway.
_toml_float(x::Real)::Float64 = (y = Float64(x); y == 0.0 ? 0.0 : y)

# ----------------------------------------------------------------------------
# struct → Dict
# ----------------------------------------------------------------------------

_key_doc(k::SALCKey) = Dict{String,Any}(
    "body" => k.body, "orbit_id" => k.orbit_id,
    "decors" => [Int[d.spin_l, d.disp_k, d.disp_l] for d in k.decors],
    "L_S" => k.L_S, "Lf" => k.Lf, "block" => k.block)

_term_doc(t::SALCTerm) = Dict{String,Any}(
    "slots" => [Int[s.site, Int(s.factor.channel), s.factor.k, s.factor.l]
                for s in t.slots],
    "shape" => collect(Int, size(t.folded)),
    "folded" => Float64[_toml_float(v) for v in vec(t.folded)])   # column-major flat

_member_doc(m::SALCMember) = Dict{String,Any}(
    "atoms" => collect(Int, m.atoms),
    "shifts" => [collect(Int, s) for s in m.shifts],
    "terms" => [_term_doc(t) for t in m.terms])

_salc_doc(s::SALC) = Dict{String,Any}(
    "key" => _key_doc(s.key),
    "members" => [_member_doc(m) for m in s.members])

function _crystal_doc(c::Crystal)
    A = c.lattice.vectors
    cols = [[_toml_float(A[1, j]), _toml_float(A[2, j]), _toml_float(A[3, j])] for j = 1:3]   # one per lattice vector
    pos = [[_toml_float(c.frac_positions[1, a]), _toml_float(c.frac_positions[2, a]),
            _toml_float(c.frac_positions[3, a])] for a = 1:n_atoms(c)]          # one per atom
    return Dict{String,Any}(
        "lattice_vectors" => cols,
        "pbc" => [c.lattice.pbc[1], c.lattice.pbc[2], c.lattice.pbc[3]],
        "frac_positions" => pos,
        "species" => collect(Int, c.species),
        "species_labels" => collect(String, c.species_labels))
end

function _symmetry_doc(sg::SpaceGroup)
    # Store only the fractional ops (+ symbol/number/tol); the Cartesian rotations,
    # properness flags, and the atom permutation table are re-derived deterministically
    # by `_assemble_spacegroup` on load.
    rots = [[[_toml_float(op.rotation_frac[i, j]) for j = 1:3] for i = 1:3] for op in sg.ops]
    trans = [[_toml_float(op.translation_frac[k]) for k = 1:3] for op in sg.ops]
    return Dict{String,Any}(
        "symbol" => sg.symbol, "number" => sg.number, "tol" => _toml_float(sg.tol),
        "rotations_frac" => rots, "translations_frac" => trans)
end

_cutoff_rows(M::AbstractMatrix) =
    [[_toml_float(M[i, j]) for j in axes(M, 2)] for i in axes(M, 1)]      # row-major

_sector_doc(r::SectorRule) = Dict{String,Any}(
    "spin_mode" => String(r.spin_mode),
    "spin_ls" => collect(Int, r.spin_ls),
    "spin_nsites" => Int[r.spin_nsites...],
    "spin_lmax" => r.spin_lmax,                             # typemax(Int64) = uncapped
    "spin_lsum" => r.spin_lsum,                             # typemax(Int64) = uncapped
    "disp_degree" => Int[r.disp_degree...],
    "sites" => Int[r.sites...],
    "soc" => r.soc,
    "cutoff" => _cutoff_rows(r.cutoff))

_spec_doc(sp::BasisSpec) = Dict{String,Any}(
    "nbody" => sp.nbody, "lmax" => collect(Int, sp.lmax),
    "pmax" => collect(Int, sp.pmax),
    "lsum" => collect(Int, sp.lsum),                       # typemax(Int64) = uncapped
    "cutoff" => [_cutoff_rows(M) for M in sp.cutoff],       # per body order
    "soc" => sp.soc,
    "sectors" => [_sector_doc(r) for r in sp.sector_rules],
    "disp_scale" => _toml_float(sp.disp_scale),
    "species_labels" => collect(String, sp.species_labels))

function _basis_doc(b::SLCEBasis)
    return Dict{String,Any}(
        "crystal" => _crystal_doc(b.crystal),
        "symmetry" => _symmetry_doc(b.spacegroup),
        "spec" => _spec_doc(b.spec),
        "fingerprint" => string(b.salc_basis.fingerprint),   # UInt64 as a string (JSON numbers lose >2^53)
        "salcs" => [_salc_doc(s) for s in b.salc_basis.salcs])
end

"""
    _to_doc(x) -> Dict{String,Any}

Build the (backend-agnostic) serialization document for an [`SLCEBasis`](@ref),
[`SLCEModel`](@ref), or [`SLCEFit`](@ref). The inverse is `_basis_from_doc` /
`_model_from_doc`.
"""
function _to_doc(b::SLCEBasis)
    d = Dict{String,Any}("schema" => _SCHEMA_BASIS, "schema_version" => PERSIST_SCHEMA_VERSION)
    merge!(d, _basis_doc(b))
    return d
end

function _to_doc(m::SLCEModel)
    d = Dict{String,Any}("schema" => _SCHEMA_MODEL, "schema_version" => PERSIST_SCHEMA_VERSION)
    merge!(d, _basis_doc(m.basis))
    d["j0"] = _toml_float(m.j0)
    d["couplings"] = [Dict{String,Any}("key" => _key_doc(m.keys[k]), "jphi" => _toml_float(m.jphi[k]))
                      for k in eachindex(m.jphi)]
    return d
end

_to_doc(f::SLCEFit) = _to_doc(SLCEModel(f))

# ----------------------------------------------------------------------------
# Dict → struct   (accessors work on both Dict{String,Any} and a parsed JSON object)
# ----------------------------------------------------------------------------

_intvec(x)::Vector{Int} = Int[Int(v) for v in x]
_floatvec(x)::Vector{Float64} = Float64[Float64(v) for v in x]

function _matrix3(rows)::SMatrix{3,3,Float64,9}
    M = MMatrix{3,3,Float64}(undef)
    @inbounds for i = 1:3, j = 1:3
        M[i, j] = Float64(rows[i][j])
    end
    return SMatrix(M)
end

# v5 keys carry "decors" + "L_S"; v2–v4 keys carry "ls" and are mapped through
# the total, value-preserving v4→v5 relabel (pure-spin decors, L_S := Lf).
function _key_from(d)::SALCKey
    Lf = Int(d["Lf"])
    if haskey(d, "decors")
        decors = SiteDecor[
            SiteDecor(; spin = Int(t[1]),
                      disp = (Int(t[2]), Int(t[3])) == (0, 0) ? nothing :
                             (Int(t[2]), Int(t[3])))
            for t in d["decors"]]
        return SALCKey(Int(d["body"]), Int(d["orbit_id"]), decors,
                       Int(d["L_S"]), Lf, Int(d["block"]))
    end
    return SALCKey(Int(d["body"]), Int(d["orbit_id"]), spin_decors(_intvec(d["ls"])),
                   Lf, Lf, Int(d["block"]))
end

function _term_from(d, natoms::Int)::SALCTerm
    # v5 terms carry explicit slots; v2-v4 terms carry the per-site "ls", which
    # maps to the identity pure-spin slot list.
    slots = haskey(d, "slots") ?
        Slot[Slot(Int(t[1]), SiteFactor(Channel(UInt8(t[2])), Int(t[3]), Int(t[4])))
             for t in d["slots"]] :
        spin_slots(_intvec(d["ls"]))
    shape = _intvec(d["shape"])
    flat = _floatvec(d["folded"])
    # The slot list and the tensor must describe the same object: one slot per
    # axis, every slot addressing a site of the member, every axis of extent
    # 2l + 1. `SALCTerm` has no inner constructor, so a term that violates any
    # of these would otherwise be built here and fail later, inside a kernel,
    # as a BoundsError or a silently truncated contraction.
    length(slots) == length(shape) ||
        throw(ArgumentError("SALC term: $(length(slots)) slots for a rank-" *
                            "$(length(shape)) tensor"))
    for (i, sl) in enumerate(slots)
        1 <= sl.site <= natoms ||
            throw(ArgumentError("SALC term: slot $i addresses site $(sl.site) of a " *
                                "$natoms-site member"))
        shape[i] == 2 * sl.factor.l + 1 ||
            throw(ArgumentError("SALC term: axis $i has extent $(shape[i]) for " *
                                "l = $(sl.factor.l) (expected $(2 * sl.factor.l + 1))"))
    end
    n = prod(shape; init = 1)
    length(flat) == n ||
        throw(ArgumentError("SALC term: $(length(flat)) coefficients for shape $shape (expected $n)"))
    folded = copy(reshape(flat, Tuple(shape)))   # owned dense Array{Float64,N}
    return SALCTerm(slots, folded)
end

function _member_from(d)::SALCMember
    atoms = _intvec(d["atoms"])
    shifts = [SVector{3,Int}(Int(s[1]), Int(s[2]), Int(s[3])) for s in d["shifts"]]
    terms = SALCTerm[_term_from(t, length(atoms)) for t in d["terms"]]
    return SALCMember(atoms, shifts, terms)
end

function _salc_from(d)::SALC
    key = _key_from(d["key"])
    members = SALCMember[_member_from(m) for m in d["members"]]
    # body / decors / L_S / Lf are fully determined by the key — so the members
    # must agree with it. A term whose slots do not reconstruct the key's label
    # (say a DISP slot under a pure-spin key) passes every per-term check and
    # then defeats the spin-only kernels' refusal, which reads `decors`: the
    # DISP axis would be evaluated as a spin harmonic under the wrong (4π)
    # scale, silently. `SALCKey` and `SALC` have no inner constructors; the
    # reader is the gate.
    key.body == length(key.decors) || throw(ArgumentError(
        "SALC key: body $(key.body) with $(length(key.decors)) decors"))
    for m in members
        length(m.atoms) == key.body || throw(ArgumentError(
            "SALC member with $(length(m.atoms)) atoms under a $(key.body)-body key"))
        for t in m.terms
            _term_decors(t.slots, key.body) == key.decors || throw(ArgumentError(
                "SALC term whose slots $(t.slots) do not reconstruct the key's " *
                "decoration label $(key.decors)"))
        end
    end
    return SALC(key, key.body, copy(key.decors), key.L_S, key.Lf, members)
end

# The sorted decoration label a slot list implies: one SPIN and/or one DISP factor
# per site. Two factors of one channel on one site, or a site with none, is not a
# label at all and is refused (the latter by `SiteDecor` itself).
function _term_decors(slots::Vector{Slot}, natoms::Int)::Vector{SiteDecor}
    spin = zeros(Int, natoms)
    disp = Vector{Union{Nothing,Tuple{Int,Int}}}(nothing, natoms)
    for sl in slots
        if sl.factor.channel == SPIN
            spin[sl.site] == 0 ||
                throw(ArgumentError("SALC term: two SPIN factors on site $(sl.site)"))
            spin[sl.site] = sl.factor.l
        elseif sl.factor.channel == DISP
            disp[sl.site] === nothing ||
                throw(ArgumentError("SALC term: two DISP factors on site $(sl.site)"))
            disp[sl.site] = (sl.factor.k, sl.factor.l)
        else
            throw(ArgumentError("SALC term: unsupported channel $(sl.factor.channel)"))
        end
    end
    return sort!(SiteDecor[SiteDecor(; spin = spin[s], disp = disp[s]) for s = 1:natoms])
end

function _crystal_from(d)::Crystal
    cols = d["lattice_vectors"]
    A = MMatrix{3,3,Float64}(undef)
    @inbounds for j = 1:3, i = 1:3
        A[i, j] = Float64(cols[j][i])
    end
    pbc = (Bool(d["pbc"][1]), Bool(d["pbc"][2]), Bool(d["pbc"][3]))
    lat = Lattice(SMatrix(A); pbc = pbc)
    poslist = d["frac_positions"]
    nat = length(poslist)
    fr = Matrix{Float64}(undef, 3, nat)
    @inbounds for a = 1:nat, i = 1:3
        fr[i, a] = Float64(poslist[a][i])
    end
    labels = String[String(s) for s in d["species_labels"]]
    return Crystal(lat, fr, _intvec(d["species"]), labels)
end

function _symmetry_from(crystal::Crystal, d)::SpaceGroup
    rots = d["rotations_frac"]
    trans = d["translations_frac"]
    length(rots) == length(trans) ||
        throw(ArgumentError("symmetry: $(length(rots)) rotations but $(length(trans)) translations"))
    rotations = SMatrix{3,3,Float64,9}[_matrix3(r) for r in rots]
    translations = SVector{3,Float64}[SVector{3,Float64}(Float64(t[1]), Float64(t[2]), Float64(t[3]))
                                      for t in trans]
    return _assemble_spacegroup(crystal, rotations, translations,
                                String(d["symbol"]), Int(d["number"]); tol = Float64(d["tol"]))
end

_cutoff_matrix(M) =
    Matrix{Float64}(reduce(hcat, (_floatvec(row) for row in M))')  # stored row-major

_tuple2(v) = (Int(v[1]), Int(v[2]))

_sector_from(d)::SectorRule =
    SectorRule(Symbol(String(d["spin_mode"])), _intvec(d["spin_ls"]),
               _tuple2(d["spin_nsites"]), Int(d["spin_lmax"]),
               Int(d["spin_lsum"]), _tuple2(d["disp_degree"]),
               _tuple2(haskey(d, "sites") ? d["sites"] : d["nbody"]),   # v5 key
               Bool(d["soc"]), _cutoff_matrix(d["cutoff"]))

function _spec_from(d)::BasisSpec
    nbody = Int(d["nbody"])
    lmax = _intvec(d["lmax"])
    if haskey(d, "pair_cutoff")     # legacy v2: one scalar radius, no lsum, no labels
        return BasisSpec(; nbody = nbody, lmax = lmax, soc = !Bool(d["isotropy"]),
                         cutoff = Float64(d["pair_cutoff"]))
    end
    labels = String[String(s) for s in d["species_labels"]]
    lsum = _intvec(d["lsum"])
    cutoff = Matrix{Float64}[_cutoff_matrix(M) for M in d["cutoff"]]
    if !haskey(d, "soc")            # legacy v3/v4 layout: "isotropy", no channels
        return BasisSpec(labels; nbody = nbody, lmax = lmax,
                         lsum = [n => v for (n, v) in enumerate(lsum)],
                         cutoff = cutoff, soc = !Bool(d["isotropy"]))
    end
    # v5 sector-capable layout — dense already, so use the canonical constructor.
    return BasisSpec(nbody, lmax, _intvec(d["pmax"]), lsum, cutoff, Bool(d["soc"]),
                     SectorRule[_sector_from(s) for s in d["sectors"]],
                     Float64(d["disp_scale"]), labels)
end

function _check_schema(d, allowed::Tuple)
    s = get(d, "schema", nothing)
    s = get(_LEGACY_SCHEMA_TAGS, s, s)          # pre-rename documents
    s in allowed ||
        throw(ArgumentError("unexpected schema $(repr(s)); expected one of $(allowed)"))
    v = get(d, "schema_version", nothing)
    (v isa Integer && Int(v) in _PERSIST_READABLE_VERSIONS) ||
        throw(ArgumentError("unsupported schema_version $(repr(v)); this build reads " *
                            "$(_PERSIST_READABLE_VERSIONS)"))
    return nothing
end

"""
    _basis_from_doc(d) -> SLCEBasis

Reconstruct an [`SLCEBasis`](@ref) from a serialization document (a basis or a model
document — the basis part is read either way). The SALCs are rebuilt verbatim from
their stored tensors (no re-projection); the space group is re-derived from the
stored fractional ops. The structural `fingerprint` is recomputed locally (the
stored one is provenance only — `hash` is Julia-version dependent).
"""
function _basis_from_doc(d)::SLCEBasis
    _check_schema(d, (_SCHEMA_BASIS, _SCHEMA_MODEL))
    crystal = _crystal_from(d["crystal"])
    sg = _symmetry_from(crystal, d["symmetry"])
    spec = _spec_from(d["spec"])
    salcs = SALC[_salc_from(s) for s in d["salcs"]]
    if Int(d["schema_version"]) < 4
        # Pre-v4 docs store one member per ordered image; fold them to the canonical
        # form (exact regrouping — see `_canonicalize_members`). A SALC is kept even
        # if folding empties it, so the stored coefficients still pair by key.
        salcs = SALC[SALC(s.key, s.body, s.decors, s.L_S, s.Lf,
                          _canonicalize_members(s.members))
                     for s in salcs]
    end
    keyvec = SALCKey[s.key for s in salcs]
    if !issorted(keyvec)
        p = sortperm(keyvec)
        salcs = salcs[p]
        keyvec = keyvec[p]
    end
    allunique(keyvec) ||
        throw(ArgumentError("loaded SALC keys are not injective (duplicate design-matrix columns)"))
    sb = SALCBasis(salcs, keyvec)
    return SLCEBasis(crystal, sg, sb, spec)
end

"""
    _model_from_doc(d) -> SLCEModel

Reconstruct an [`SLCEModel`](@ref) from a model document, re-pairing each stored
coefficient to the rebuilt basis **by [`SALCKey`](@ref)** (not by position). Errors
if any basis column lacks a coefficient or the counts disagree.
"""
function _model_from_doc(d)::SLCEModel
    _check_schema(d, (_SCHEMA_MODEL,))
    basis = _basis_from_doc(d)
    j0 = Float64(d["j0"])
    coup = Dict{SALCKey,Float64}()
    for c in d["couplings"]
        k = _key_from(c["key"])
        haskey(coup, k) && throw(ArgumentError("duplicate coefficient for SALC key $k"))
        coup[k] = Float64(c["jphi"])
    end
    keys = basis.salc_basis.keys
    length(coup) == length(keys) ||
        throw(ArgumentError("model has $(length(coup)) coefficients for $(length(keys)) basis SALCs"))
    jphi = Vector{Float64}(undef, length(keys))
    @inbounds for (i, k) in enumerate(keys)
        haskey(coup, k) || throw(ArgumentError("no coefficient for SALC key $k in the file"))
        jphi[i] = coup[k]
    end
    return SLCEModel(basis, j0, jphi, copy(keys))
end

# ----------------------------------------------------------------------------
# save / load entry points
# ----------------------------------------------------------------------------

"""
    save(path_or_io, x)

Serialize an [`SLCEBasis`](@ref), [`SLCEModel`](@ref), or [`SLCEFit`](@ref) (a fit is
saved as its model) to `path_or_io` as a self-contained, human-readable TOML
document. Not exported (the name clashes with FileIO / JLD2 / CSV); call as
`SLCE.save("model.toml", model)`. Inverse: [`load`](@ref SLCE.load).
"""
function save(io::IO, x::Union{SLCEBasis,SLCEModel,SLCEFit})
    TOML.print(io, _to_doc(x))
    return nothing
end
save(path::AbstractString, x::Union{SLCEBasis,SLCEModel,SLCEFit}) =
    (open(io -> save(io, x), path, "w"); nothing)

"""
    load(SLCEBasis, path_or_io) -> SLCEBasis
    load(SLCEModel, path_or_io) -> SLCEModel

Inverse of [`save`](@ref SLCE.save): rebuild a basis or model from a TOML
document. The SALC basis is reconstructed verbatim (no re-projection); a basis can be
loaded from a model document too (the coefficients are ignored), and a model load
re-pairs coefficients to the basis **by key** (not by position). Not exported; call as
`SLCE.load(SLCEModel, "model.toml")`.
"""
load(::Type{SLCEBasis}, io::IO)::SLCEBasis = _basis_from_doc(TOML.parse(io))
load(::Type{SLCEModel}, io::IO)::SLCEModel = _model_from_doc(TOML.parse(io))
load(::Type{SLCEBasis}, path::AbstractString)::SLCEBasis = _basis_from_doc(TOML.parsefile(path))
load(::Type{SLCEModel}, path::AbstractString)::SLCEModel = _model_from_doc(TOML.parsefile(path))
