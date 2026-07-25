"""
    BasisSpec([labels_or_crystal]; nbody, lmax, cutoff, lsum = nothing, soc = true)
    BasisSpec([labels_or_crystal]; sectors, lmax, pmax = nothing, ...)

An SLCE basis **specification** — the knobs that define the basis, not an
interaction term itself. Stored in resolved, dense canonical form:

- `nbody::Int` — maximum body order (decorated-site count).
- `lmax::Vector{Int}` — per-species maximum spin rank `l` (per site, every body
  order; `0` makes a species spin-inactive).
- `pmax::Vector{Int}` — per-species per-site maximum displacement degree
  `2k + l` (`0` clamps the species; a ligand is `lmax = 0, pmax > 0`).
- `lsum::Vector{Int}` — per-body-order cap on `Σl` over the **spin** factors
  (index = body order; `LSUM_UNCAPPED` where uncapped).
- `cutoff::Vector{Matrix{Float64}}` — per-body-order symmetric species-pair
  cutoff radii in Å (`cutoff[N - 1][a, b]` for body order `N ≥ 2`); an
  `N`-body cluster is kept iff **every** internal edge is within its own
  species-pair radius. `Inf` = no cutoff (every resolvable pair — the whole
  Wigner–Seitz cell under [`MinimumImage`](@ref)); `0` excludes the pair. In
  sector mode this is the derived per-body **envelope** (elementwise max over
  the sectors admitting each body order); each sector re-admits within its own
  radii.
- `soc::Bool` — dense pure-spin form only: `false` keeps only the total-spin
  scalar (`L_S = 0`, here `≡ Lf = 0`) channel. In sector mode SOC selection is
  per sector and this field is `true`.
- `sectors::Vector{SLCE.SectorRule}` — the resolved sector table (empty = the
  dense pure-spin specification above; see [`Sector`](@ref)).
- `disp_scale::Float64` — the fixed displacement scale (Å) the displacement
  kernels are evaluated in units of; persisted with the basis (like the `(4π)`
  spin normalization, it is part of the model definition, not a fit knob).
  Until the joint data layer (M3) wires it into the kernels, only the default
  `1.0` is accepted — a declared-but-inapplied unit convention is refused.
- `species_labels::Vector{String}` — the labels the spec was resolved against
  (empty when constructed index-keyed).

# Dense pure-spin form

```julia
BasisSpec(crystal; nbody = 3, soc = true,               # or BasisSpec(labels; ...)
    lmax   = ["*" => 3, "B" => 0],                       # label-keyed, "*" fallback
    lsum   = [1 => 0, 2 => 4, 3 => 4],                   # body-keyed (or one Int)
    cutoff = [2 => Inf, 3 => ["Fe-*" => 6.0, "*-*" => 8.0]])
```

`lmax` may be an `Int` (all species), a per-species `Vector{Int}`, or
label-keyed pairs; `lsum` an `Int` (all body orders), body-keyed pairs, or
`nothing` (no cap); `cutoff` a scalar, a pair table (all body orders),
body-keyed pairs of either, or the canonical `Vector{Matrix}`. Pair keys are
unordered (`"Fe-Nd" ≡ "Nd-Fe"`) and resolve by specificity (concrete beats
`"A-*"` beats `"*-*"`); equal-specificity conflicts, unknown labels, uncovered
species/pairs, and body orders outside `nbody` are errors. Label-keyed forms
need the labels: pass them (or a `Crystal`) as the first argument.

The former `isotropy` keyword is a deprecation error: `isotropy = true` kept
only the scalar channel, which is exactly `soc = false` (for a pure-spin basis
`L_S ≡ Lf`).

# Sector-table (joint spin–lattice) form

```julia
BasisSpec(crystal; lmax = 2, pmax = ["*" => 0, "Fe" => 3], sectors = [
    Sector(disp = (degree = 2:3,), cutoff = 6.0),                # force constants
    Sector(spin = (nbody = 2:4, lmax = 2), cutoff = 8.0),        # pure-spin SCE
    Sector(spin = [1, 1], disp = (degree = 1,), soc = false, cutoff = 5.0)])
```

The admitted decoration labels are the **union over the sector rows**,
intersected with the global per-species `lmax`/`pmax` caps and the per-body
`lsum`. In sector mode `nbody` is derived from the sectors (an explicit value
caps it), the per-body `cutoff` envelope is derived (the keyword is rejected —
radii are per sector), `soc` is per sector, and `pmax` is required as soon as
any sector carries displacement content.
"""
struct BasisSpec
    nbody::Int
    lmax::Vector{Int}
    pmax::Vector{Int}
    lsum::Vector{Int}
    cutoff::Vector{Matrix{Float64}}
    soc::Bool
    sectors::Vector{SectorRule}
    disp_scale::Float64
    species_labels::Vector{String}

    # Positional canonical-form constructor; all construction paths validate here.
    function BasisSpec(nbody::Int, lmax::Vector{Int}, pmax::Vector{Int},
                       lsum::Vector{Int}, cutoff::Vector{Matrix{Float64}},
                       soc::Bool, sectors::Vector{SectorRule}, disp_scale::Float64,
                       species_labels::Vector{String})
        nbody >= 1 || throw(ArgumentError("nbody must be ≥ 1; got $nbody"))
        isempty(lmax) && throw(ArgumentError("lmax must have one entry per species"))
        all(>=(0), lmax) ||
            throw(ArgumentError("lmax entries must be ≥ 0; got $lmax"))
        length(pmax) == length(lmax) ||
            throw(ArgumentError("pmax has $(length(pmax)) entries for " *
                                "$(length(lmax)) species"))
        all(>=(0), pmax) ||
            throw(ArgumentError("pmax entries must be ≥ 0; got $pmax"))
        length(lsum) == nbody ||
            throw(ArgumentError("lsum has $(length(lsum)) entries for nbody = $nbody"))
        all(>=(0), lsum) || throw(ArgumentError("lsum entries must be ≥ 0; got $lsum"))
        length(cutoff) == nbody - 1 ||
            throw(ArgumentError("cutoff has $(length(cutoff)) matrices for body " *
                                "orders 2:$nbody"))
        nkd = length(lmax)
        for (k, M) in enumerate(cutoff)
            size(M) == (nkd, nkd) ||
                throw(ArgumentError("cutoff body$(k + 1) matrix is $(size(M)), " *
                                    "expected ($nkd, $nkd)"))
            M == M' ||
                throw(ArgumentError("cutoff body$(k + 1) matrix is not symmetric"))
            all(v -> !isnan(v) && v >= 0, M) ||
                throw(ArgumentError("cutoff body$(k + 1) entries must be ≥ 0 or Inf"))
        end
        if isempty(sectors)
            all(==(0), pmax) ||
                throw(ArgumentError("pmax > 0 needs a sector table with " *
                                    "displacement content (`sectors = ...`)"))
        else
            soc || throw(ArgumentError("with a sector table, SOC selection is " *
                                       "per sector (`Sector(; soc = false)`), " *
                                       "not a global flag"))
            for (i, r) in enumerate(sectors)
                size(r.cutoff) == (nkd, nkd) ||
                    throw(ArgumentError("sectors[$i] cutoff matrix is " *
                                        "$(size(r.cutoff)), expected ($nkd, $nkd)"))
                r.nbody[1] <= nbody ||
                    throw(ArgumentError("sectors[$i] admits only body orders " *
                                        "$(r.nbody[1]):$(r.nbody[2]) but nbody = " *
                                        "$nbody — the sector would contribute " *
                                        "nothing"))
                r.spin_mode == :explicit &&
                    maximum(r.spin_ls) > maximum(lmax) &&
                    throw(ArgumentError("sectors[$i]: explicit spin rank " *
                                        "$(maximum(r.spin_ls)) exceeds every " *
                                        "species' lmax ($lmax) — the sector " *
                                        "would contribute nothing"))
            end
        end
        isfinite(disp_scale) && disp_scale > 0 ||
            throw(ArgumentError("disp_scale must be a finite positive length (Å); " *
                                "got $disp_scale"))
        # Refusing beats a silent unit lie: the field is part of the persisted
        # format, but no displacement kernel applies it yet — accepting ≠ 1
        # would persist a unit convention the numbers do not honour. The joint
        # data layer (M3) wires it in and lifts this guard.
        disp_scale == 1.0 ||
            throw(ArgumentError("disp_scale ≠ 1 is declared but not yet applied " *
                                "by the displacement kernels (the joint data " *
                                "layer, M3, wires it in) — refusing a unit " *
                                "convention the stored tensors would not honour"))
        isempty(species_labels) || length(species_labels) == nkd ||
            throw(ArgumentError("$(length(species_labels)) species labels for " *
                                "$nkd lmax entries"))
        allunique(species_labels) ||
            throw(ArgumentError("species labels must be unique; got $species_labels"))
        return new(nbody, copy(lmax), copy(pmax), copy(lsum),
                   Matrix{Float64}[copy(M) for M in cutoff], soc, copy(sectors),
                   disp_scale, copy(species_labels))
    end
end

function BasisSpec(labels::AbstractVector{<:AbstractString} = String[];
                   lmax, nbody = nothing, cutoff = nothing, lsum = nothing,
                   soc::Bool = true, sectors = nothing, pmax = nothing,
                   disp_scale::Real = 1.0, isotropy = nothing, pair_cutoff = nothing)
    isotropy === nothing ||
        throw(ArgumentError("`isotropy` was replaced by the SOC selection rule: " *
                            "use `soc = false` for the scalar channel (note the " *
                            "inversion — `isotropy = true` ⇔ `soc = false`; in a " *
                            "sector table SOC is per sector, `Sector(; soc)`)"))
    pair_cutoff === nothing ||
        throw(ArgumentError("`pair_cutoff` was replaced by `cutoff` (a scalar is " *
                            "equivalent: cutoff = $pair_cutoff; see the BasisSpec " *
                            "docstring for per-body / per-pair forms)"))
    lv = collect(String, labels)
    nkd = !isempty(lv) ? length(lv) :
          lmax isa AbstractVector{<:Integer} ? length(lmax) :
          throw(ArgumentError("the species count is unknown: pass the labels " *
                              "(`BasisSpec(labels; ...)` / `BasisSpec(crystal; ...)`) " *
                              "or give `lmax` as a per-species Vector{Int}"))
    if sectors === nothing
        # dense pure-spin form
        nbody === nothing &&
            throw(ArgumentError("the dense (sector-less) form needs `nbody`"))
        cutoff === nothing &&
            throw(ArgumentError("the dense (sector-less) form needs `cutoff`"))
        pmax === nothing ||
            throw(ArgumentError("`pmax` needs a sector table with displacement " *
                                "content (`sectors = ...`)"))
        nb = Int(nbody)
        nb >= 1 || throw(ArgumentError("nbody must be ≥ 1; got $nbody"))
        return BasisSpec(nb,
                         _resolve_species_table(lmax, nkd, lv, "lmax"),
                         zeros(Int, nkd),
                         _resolve_lsum(lsum, nb),
                         _resolve_cutoff(cutoff, nb, nkd, lv),
                         soc, SectorRule[], Float64(disp_scale), lv)
    end
    # sector-table form
    cutoff === nothing ||
        throw(ArgumentError("with a sector table, cutoffs are per sector " *
                            "(`Sector(; cutoff = ...)`); the per-body envelope " *
                            "is derived"))
    soc || throw(ArgumentError("with a sector table, SOC selection is per sector " *
                               "(`Sector(; soc = false)`), not a global flag"))
    rules, nb, envelope = _resolve_sectors(sectors, nkd, lv, nbody)
    has_disp_content = any(r -> r.disp_degree[2] > 0, rules)
    pmax_v = pmax === nothing ?
        (has_disp_content ?
             throw(ArgumentError("the sector table has displacement content — " *
                                 "give `pmax` (per-species per-site displacement " *
                                 "degree cap; 0 clamps a species)")) :
             zeros(Int, nkd)) :
        _resolve_species_table(pmax, nkd, lv, "pmax")
    maxdeg = maximum(r.disp_degree[2] for r in rules; init = 0)
    if maxdeg > 0 && isodd(maxdeg)
        @warn "the maximum total displacement degree $maxdeg is odd: the leading " *
              "displacement form is unbounded below (a necessary boundedness " *
              "condition only — see the design record §5)"
    end
    return BasisSpec(nb,
                     _resolve_species_table(lmax, nkd, lv, "lmax"),
                     pmax_v,
                     _resolve_lsum(lsum, nb),
                     envelope,
                     true, rules, Float64(disp_scale), lv)
end

BasisSpec(crystal::Crystal; kwargs...) = BasisSpec(crystal.species_labels; kwargs...)

Base.:(==)(a::BasisSpec, b::BasisSpec) =
    a.nbody == b.nbody && a.lmax == b.lmax && a.pmax == b.pmax &&
    a.lsum == b.lsum && a.cutoff == b.cutoff && a.soc == b.soc &&
    a.sectors == b.sectors && a.disp_scale == b.disp_scale &&
    a.species_labels == b.species_labels

function Base.show(io::IO, ::MIME"text/plain", sp::BasisSpec)
    nkd = length(sp.lmax)
    labels = isempty(sp.species_labels) ? ["#$k" for k = 1:nkd] : sp.species_labels
    println(io, "BasisSpec: nbody = ", sp.nbody, ", soc = ", sp.soc)
    println(io, "  lmax:  ", join(("$(labels[k]) = $(sp.lmax[k])" for k = 1:nkd), ", "))
    any(>(0), sp.pmax) &&
        println(io, "  pmax:  ",
                join(("$(labels[k]) = $(sp.pmax[k])" for k = 1:nkd), ", "))
    lsumstr(v) = v == LSUM_UNCAPPED ? "—" : string(v)
    println(io, "  lsum:  ",
            join(("body$N = $(lsumstr(sp.lsum[N]))" for N = 1:sp.nbody), ", "))
    for N = 2:sp.nbody
        M = sp.cutoff[N - 1]
        if all(==(M[1, 1]), M)
            println(io, "  cutoff body$N: ", M[1, 1], " Å (all pairs)")
        else
            println(io, "  cutoff body$N (Å):")
            for i = 1:nkd
                println(io, "    ", rpad(labels[i], 6), " ",
                        join((string(M[i, j]) for j = 1:nkd), "  "))
            end
        end
    end
    sp.disp_scale == 1.0 ||
        println(io, "  disp_scale: ", sp.disp_scale, " Å")
    for (i, r) in enumerate(sp.sectors)
        println(io, "  sector $i: ", sprint(show, r))
    end
end

"""
    SLCEBasis(crystal, spec; backend = NoSymmetry(), tol = 1e-5,
             images = MinimumImage())

Build the SCE basis for `crystal`: analyze symmetry, enumerate cluster orbits, and
construct the symmetry-adapted SALC basis. Pass `backend = SpglibBackend()` (with
`using Spglib`) for real space-group symmetry.

`images` selects how periodic images are admitted (see [`AbstractImageSelection`](@ref)):
[`MinimumImage`](@ref) (the default) keeps only the minimum-image, resolvable pairs of
a plain-PBC supercell; [`AllImages`](@ref) keeps every image within the cutoff (the
generalized-Bloch / spin-spiral seam, finite cutoff only). The `images` value is also
applied to the cluster-edge admissibility, so the neighbor list and the clusters stay
consistent.
"""
struct SLCEBasis
    crystal::Crystal
    spacegroup::SpaceGroup
    salc_basis::SALCBasis
    spec::BasisSpec
end

function SLCEBasis(crystal::Crystal, spec::BasisSpec;
                 backend::AbstractSymmetryBackend = NoSymmetry(), tol::Real = 1e-5,
                 images::AbstractImageSelection = MinimumImage())::SLCEBasis
    length(spec.lmax) == length(crystal.species_labels) ||
        throw(ArgumentError("spec covers $(length(spec.lmax)) species, crystal has " *
                            "$(length(crystal.species_labels))"))
    isempty(spec.species_labels) || spec.species_labels == crystal.species_labels ||
        throw(ArgumentError("spec species labels $(spec.species_labels) do not match " *
                            "the crystal's $(crystal.species_labels)"))
    sg = analyze_symmetry(backend, crystal; tol = tol)
    # The neighbor list is built at the per-pair superset radius (element-wise max
    # over body orders); each body order's own radii then trim edges per cluster in
    # `candidate_clusters`. In sector mode `spec.cutoff` is the derived envelope;
    # each sector then re-admits orbits within its own radii in the builder.
    nl = build_neighbor_list(crystal, _superset_cutoff(spec), images)
    clusters = build_clusters(crystal, nl, sg; nbody = spec.nbody, selection = images,
                              cutoff = spec.cutoff)
    salcs = isempty(spec.sectors) ?
        build_salc_basis(crystal, sg, clusters;
                         lmax_by_species = spec.lmax, lsum_by_body = spec.lsum,
                         isotropy = !spec.soc) :
        build_salc_basis(crystal, sg, clusters, spec; neighbors = nl,
                         selection = images)
    return SLCEBasis(crystal, sg, salcs, spec)
end

"""
    n_salcs(basis::SLCEBasis) -> Int

The number of SALC basis functions in `basis` — equivalently, the number of
design-matrix columns / fitted coefficients.
"""
n_salcs(b::SLCEBasis) = length(b.salc_basis)

"""
    salcs(basis::SLCEBasis) -> Vector{SALC}

The ordered SALC basis functions of `basis` (column order of the design matrix).
"""
salcs(b::SLCEBasis) = b.salc_basis.salcs

"""
    SLCEDataset(basis, configs, energies)
    SLCEDataset(basis, configs, energies, torques)

Pair an [`SLCEBasis`](@ref) with training data: spin configurations `configs`
(each `3 × n_atoms`, unit columns) and their `energies`. Materializes the energy
design matrix `X_E[config, salc] = evaluate_salc(salc, config)`.

The four-argument form additionally takes per-configuration torques (each
`3 × n_atoms`, the DFT torque `τ_a = m_a × B_a = −e_a × ∂E/∂e_a` on every atom) and builds the
torque design matrix `X_T` for an energy+torque co-fit (see [`fit`](@ref)). Its
rows are flattened config-major, then atom-major, then `xyz`:
`row = 3·n_atoms·(config−1) + 3·(atom−1) + component`. Torque-free datasets leave
`X_T`/`y_T` empty.

Datasets support `length` (number of configs), configuration slicing
`dataset[idx]` (integer vector/range, `Bool` mask, or `:`), and `vcat` of parts
built on the same basis — see those methods for train/test splitting and
incremental data addition.
"""
struct SLCEDataset
    basis::SLCEBasis
    configs::Vector{Matrix{Float64}}
    X_E::Matrix{Float64}
    y_E::Vector{Float64}
    X_T::Matrix{Float64}
    y_T::Vector{Float64}
end

"""
    has_torque(dataset) -> Bool

Whether `dataset` carries torque training data (so an energy+torque co-fit is
possible).
"""
has_torque(d::SLCEDataset) = !isempty(d.y_T)

# Atoms that appear in some SALC member of `basis` — the sites whose spin the model
# actually reads. A species removed by `lmax = 0` (or a site that never enters an
# admitted cluster) is unreferenced, and its training moments are never consulted.
function _referenced_atoms(basis::SLCEBasis)::BitVector
    ref = falses(n_atoms(basis.crystal))
    for s in salcs(basis), mem in s.members, a in mem.atoms
        ref[a] = true
    end
    return ref
end

# --- dataset views: length / slicing / concatenation ---------------------------

"""
    length(dataset::SLCEDataset) -> Int

The number of training configurations in `dataset`.
"""
Base.length(d::SLCEDataset) = length(d.configs)

Base.firstindex(d::SLCEDataset) = 1
Base.lastindex(d::SLCEDataset) = length(d)

"""
    dataset[idx] -> SLCEDataset

Select the configurations `idx` — an integer vector or range, a `Bool` mask, or `:`
— into a new [`SLCEDataset`](@ref) on the same basis. Design-matrix rows are sliced,
never recomputed, so train/test splits and filters are cheap:
`train, test = dataset[1:80], dataset[81:end]`. Duplicate indices are allowed
(bootstrap-style resampling). See also `vcat` for the reverse operation.
"""
function Base.getindex(d::SLCEDataset, idx::AbstractVector{<:Integer})::SLCEDataset
    isempty(idx) && throw(ArgumentError("empty configuration selection"))
    cfgs = d.configs[idx]                       # BoundsError on an out-of-range index
    X_E = d.X_E[idx, :]
    y_E = d.y_E[idx]
    has_torque(d) || return SLCEDataset(d.basis, cfgs, X_E, y_E, d.X_T, d.y_T)
    # torque rows come in per-config blocks of 3·n_atoms (config-major layout)
    nrow = 3 * n_atoms(d.basis.crystal)
    rows = reduce(vcat, [(nrow * (i - 1) + 1):(nrow * i) for i in idx])
    return SLCEDataset(d.basis, cfgs, X_E, y_E, d.X_T[rows, :], d.y_T[rows])
end

function Base.getindex(d::SLCEDataset, mask::AbstractVector{Bool})::SLCEDataset
    length(mask) == length(d) ||
        throw(DimensionMismatch("mask has $(length(mask)) entries, dataset has " *
                                "$(length(d)) configs"))
    return d[findall(mask)]
end

Base.getindex(d::SLCEDataset, ::Colon)::SLCEDataset = d[1:length(d)]

"""
    vcat(a::SLCEDataset, b::SLCEDataset...) -> SLCEDataset

Concatenate datasets built on the **same basis** — checked by the SALC-basis
fingerprint, so a dataset built from a persisted-and-reloaded basis concatenates
with one built in-session. All parts must agree on carrying torque data or not.
Together with `dataset[idx]` this supports incremental data addition and
resampling without rebuilding design matrices.
"""
function Base.vcat(a::SLCEDataset, rest::SLCEDataset...)::SLCEDataset
    isempty(rest) && return a
    parts = (a, rest...)
    for (k, b) in enumerate(parts)
        b.basis.salc_basis.fingerprint == a.basis.salc_basis.fingerprint ||
            throw(ArgumentError("dataset $k was built on a different basis " *
                                "(SALC fingerprint mismatch)"))
        n_atoms(b.basis.crystal) == n_atoms(a.basis.crystal) ||
            throw(ArgumentError("dataset $k has $(n_atoms(b.basis.crystal)) atoms " *
                                "per config, dataset 1 has $(n_atoms(a.basis.crystal))"))
        has_torque(b) == has_torque(a) ||
            throw(ArgumentError("cannot vcat torque-bearing and energy-only datasets " *
                                "(dataset $k differs from dataset 1)"))
    end
    return SLCEDataset(a.basis,
                      reduce(vcat, [p.configs for p in parts]),
                      reduce(vcat, [p.X_E for p in parts]),
                      reduce(vcat, [p.y_E for p in parts]),
                      reduce(vcat, [p.X_T for p in parts]),
                      reduce(vcat, [p.y_T for p in parts]))
end

# Spin configurations must be `3 × n_atoms` with finite, unit-norm columns — the
# contract the harmonic kernels assume (they call `Zlm_unsafe`, skipping per-call
# checks). Enforce it once at the data boundary so malformed DFT input fails loudly
# here instead of silently biasing the design matrix / prediction. `label` carries the
# config index into the message; `atol` is the unit-norm tolerance.
function _validate_config(c::AbstractMatrix{<:Real}, nat::Int; atol::Real = 1e-6,
                          label::AbstractString = "spin config")
    size(c, 1) == 3 ||
        throw(ArgumentError("$label must have 3 rows (got $(size(c, 1)))"))
    size(c, 2) == nat ||
        throw(DimensionMismatch("$label has $(size(c, 2)) atoms, basis expects $nat"))
    @inbounds for a in axes(c, 2)
        u = SVector{3,Float64}(c[1, a], c[2, a], c[3, a])
        all(isfinite, u) ||
            throw(ArgumentError("$label column $a is not finite ($(Tuple(u)))"))
        abs(norm(u) - 1) <= atol ||
            throw(ArgumentError("$label column $a is not a unit vector (‖e‖ = $(norm(u)))"))
    end
    return nothing
end

function _validate_configs(basis::SLCEBasis, cfgs::Vector{Matrix{Float64}}; atol::Real = 1e-6)
    nat = n_atoms(basis.crystal)
    for (i, c) in enumerate(cfgs)
        _validate_config(c, nat; atol = atol, label = "config $i")
    end
    return nothing
end

# The spin-configuration dataset path evaluates spin-only SALCs; refuse a
# displacement-decorated basis at the boundary (the joint data layer — energies
# with displacements, forces — is M3) instead of erroring inside the threaded
# design assembly.
function _require_pure_spin_basis(basis::SLCEBasis)
    all(s -> all(is_pure_spin, s.key.decors), salcs(basis)) ||
        throw(ArgumentError("the basis carries displacement-decorated SALCs — " *
                            "the spin-configuration SLCEDataset path is pure-spin " *
                            "only (the joint data layer lands in M3)"))
    return nothing
end

function SLCEDataset(basis::SLCEBasis, configs::AbstractVector, energies::AbstractVector;
                   atol::Real = 1e-6)::SLCEDataset
    length(configs) == length(energies) ||
        throw(DimensionMismatch("got $(length(configs)) configs but $(length(energies)) energies"))
    _require_pure_spin_basis(basis)
    cfgs = [Matrix{Float64}(c) for c in configs]
    _validate_configs(basis, cfgs; atol = atol)
    X = _design_energy(basis, cfgs)
    empty_T = Matrix{Float64}(undef, 0, size(X, 2))
    return SLCEDataset(basis, cfgs, X, collect(Float64, energies), empty_T, Float64[])
end

function SLCEDataset(basis::SLCEBasis, configs::AbstractVector, energies::AbstractVector,
                   torques::AbstractVector; atol::Real = 1e-6)::SLCEDataset
    length(configs) == length(energies) ||
        throw(DimensionMismatch("got $(length(configs)) configs but $(length(energies)) energies"))
    _require_pure_spin_basis(basis)
    cfgs = [Matrix{Float64}(c) for c in configs]
    length(torques) == length(cfgs) ||
        throw(ArgumentError("got $(length(torques)) torque blocks for $(length(cfgs)) configs"))
    _validate_configs(basis, cfgs; atol = atol)
    X_E = _design_energy(basis, cfgs)
    X_T = _design_torque(basis, cfgs)
    y_T = _flatten_torques(torques, cfgs)
    return SLCEDataset(basis, cfgs, X_E, collect(Float64, energies), X_T, y_T)
end
"""
    SLCEModel

The **lightweight, persistable predictor**: the [`SLCEBasis`](@ref) plus the fitted
reference energy `j0` and SALC coefficients `jphi` (one per basis function), with
`keys` recording each coefficient's [`SALCKey`](@ref). Unlike [`SLCEFit`](@ref) it
carries no training data, design matrices, or diagnostics — just enough to evaluate
[`predict_energy`](@ref) / [`predict_torque`](@ref) and to round-trip through
`SLCE.save` / `SLCE.load`.

Obtain one from a fit with `SLCEModel(fit)`. Within a session `jphi[k]` pairs with
`salcs(basis)[k]` positionally; on reload the coefficients are re-paired to a freshly
built basis **by `SALCKey`** (`keys`), so a persisted model stays valid across sessions.
"""
struct SLCEModel
    basis::SLCEBasis
    j0::Float64
    jphi::Vector{Float64}
    keys::Vector{SALCKey}
end

"""
    SLCEModel(basis, j0, jphi) -> SLCEModel

Assemble a predictor directly from a basis and coefficients — `jphi[k]` pairs with
`salcs(basis)[k]` positionally and the `keys` are filled in from the basis. The public
way to build a synthetic model (hand-set couplings for tests, demos, or model studies)
without touching the SALC-basis internals; a fitted model comes from `SLCEModel(fit)`.
"""
function SLCEModel(basis::SLCEBasis, j0::Real, jphi::AbstractVector{<:Real})::SLCEModel
    length(jphi) == n_salcs(basis) ||
        throw(DimensionMismatch("jphi has $(length(jphi)) coefficients for " *
                                "$(n_salcs(basis)) SALC basis functions"))
    return SLCEModel(basis, Float64(j0), collect(Float64, jphi), basis.salc_basis.keys)
end

"""
    SLCEFit

The **full result of [`fit`](@ref)**: the [`SLCEDataset`](@ref) (with its design
matrices), the fitted `j0`/`jphi`, the `estimator`, the `torque_weight` used, and the
(energy) residuals. This is the heavyweight, data-bearing object you query for
diagnostics ([`r2_energy`](@ref), [`rmse_energy`](@ref), [`residuals_energy`](@ref), …).
For prediction and storage, convert it to the lightweight [`SLCEModel`](@ref) with
`SLCEModel(fit)`.
"""
struct SLCEFit
    dataset::SLCEDataset
    j0::Float64
    jphi::Vector{Float64}
    estimator::AbstractEstimator
    residuals::Vector{Float64}
    torque_weight::Float64
end
