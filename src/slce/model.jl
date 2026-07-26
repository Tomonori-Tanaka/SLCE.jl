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
    ASRReparam

The acoustic-sum-rule (translation-invariance) reparameterization of a joint
basis: the row-normalized constraint matrix `A` (`A·β = 0` ⇔ the model energy is
invariant under rigid translations `u_a → u_a + t`), an **orthonormal** null-space
basis `Z` (`β = beta_p + Z·γ`; identity blocks on pure-spin columns), the
particular solution `beta_p` (all-zero in the homogeneous case), and the number of
constraints binding the reparameterized columns. Built once per basis by
[`build_asr`](@ref) at `SLCEDataset` construction and stored on `dataset.asr`;
`_assemble_problem` only applies it. Never persisted — `Z`'s gauge is
factorization-dependent, while `β = Z·γ` is gauge-invariant, so `jphi` remains the
only stored coefficient object and [`asr_residual`](@ref) re-verifies models from
the basis.

The same type carries a **staged** fit's reparameterization (`fit`'s
`frozen` / `sector_mask`, `fitting/staged.jl`): `Z` then has a zero row on every
frozen column and spans the ASR null space of the free ones only, `beta_p` carries
the frozen values plus — when the frozen part violates the ASR by itself — the
particular solution of `A_free·β_free = −A_frozen·β_frozen`, and `rank` counts the
constraints binding the free columns. So `size(Z, 2) == p − rank` holds for a
basis-level reparameterization but NOT for a stage (which fits fewer columns);
the invariant checked here is the dimensional one.
"""
struct ASRReparam
    A::Matrix{Float64}
    Z::Matrix{Float64}
    beta_p::Vector{Float64}
    rank::Int
    # The columns this reparameterization FITS (all of them for a basis-level one;
    # a stage's free set). Not derivable from the other fields — `beta_p` mixes the
    # frozen values with the particular solution, and a free column that the
    # constraint structurally zeroes has the same all-zero `Z` row as a frozen one.
    # `refit` needs the distinction to rebuild a sub-stage over the SAME frozen part.
    free::Vector{Int}

    function ASRReparam(A::Matrix{Float64}, Z::Matrix{Float64},
                        beta_p::Vector{Float64}, rank::Int,
                        free::Vector{Int} = collect(axes(A, 2)))
        p = size(A, 2)
        (issorted(free) && allunique(free) &&
         (isempty(free) || (first(free) >= 1 && last(free) <= p))) ||
            throw(ArgumentError("free columns must be sorted, unique and within " *
                                "1:$p"))
        size(Z, 1) == p ||
            throw(DimensionMismatch("Z has $(size(Z, 1)) rows for $p constraint " *
                                    "columns"))
        length(beta_p) == p ||
            throw(DimensionMismatch("beta_p has length $(length(beta_p)) for $p " *
                                    "columns"))
        # `size(Z, 2) == p − rank` only for a basis-level reparameterization; a
        # stage fits a subset of the columns, so the free-parameter count is
        # `|free| − rank(A_free)` — still bounded by `p − rank`, which keeps the
        # shape check meaningful for both.
        size(Z, 2) <= p - rank ||
            throw(DimensionMismatch("Z has $(size(Z, 2)) columns, more than the " *
                                    "$(p - rank) = p − rank a feasible space can " *
                                    "have"))
        0 <= rank <= p || throw(ArgumentError("rank must be in 0:$p; got $rank"))
        return new(A, Z, beta_p, rank, free)
    end
end

"""
    SLCEDataset(basis, configs, energies)
    SLCEDataset(basis, configs, energies, torques)
    SLCEDataset(basis, configs, energies, torques, torque_sel)

Pair an [`SLCEBasis`](@ref) with training data: spin configurations `configs`
(each `3 × n_atoms`, unit columns) and their `energies`. Materializes the energy
design matrix `X_E[config, salc] = evaluate_salc(salc, config)`.

The four-argument form additionally takes per-configuration torques (each
`3 × n_atoms`, the DFT torque `τ_a = m_a × B_a = −e_a × ∂E/∂e_a` on every atom) and
builds the torque design matrix `X_T` for an energy+torque co-fit (see
[`fit`](@ref)). The five-argument form is the **mixed** variant: `torque_sel` names
the (strictly increasing) configuration indices that carry torque data, and
`torques` has one `3 × n_atoms` block per entry of `torque_sel` — configurations
outside `torque_sel` contribute energy rows only. Torque rows are flattened
config-major, then atom-major, then `xyz`, and rows of torque-free configurations
are **excluded entirely** (never zero-padded: padded rows would silently inflate the
`√(w/n_T)` whitening and dilute the observed torques). The per-row provenance is
stored in `torque_config` — `torque_config[r]` is the configuration index of torque
row `r` — which every consumer (slicing, `vcat`, `_assemble_problem`'s grouped-CV
labels) reads instead of re-deriving row offsets from a uniform-block assumption.
Torque-free datasets leave `X_T`/`y_T`/`torque_config` empty.

Against a **displacement-decorated basis** a dataset additionally carries the
per-config displacement fields `disps` (u = 0 exactly for a spin-only datum — atoms
at the pinned clamped-ion reference) and, when force data enter, the force design
block: `X_F` stored **compact** over the displacement-active SALC columns
`force_cols` (a pure-spin SALC has `∂Φ/∂u ≡ 0`; the zero columns exist only in the
assembled fit, never here), with `y_F`/`force_config` following the same
ragged-row contract as the torque block, restricted to atoms some SALC displacement
slot actually reads. Torque rows on this path are likewise restricted to
spin-referenced atoms (a displacement-only ligand site contributes exactly-zero
`X_T` and `y_T` rows, which would only dilute the `√(w_T/n_T)` whitening); the
pure-spin constructors below keep their historical all-atom torque layout. These
joint blocks enter only through the `TrainingDatum` construction path — the
spin-configuration constructors below are pure-spin-only.

Datasets support `length` (number of configs), configuration slicing
`dataset[idx]` (integer vector/range, `Bool` mask, or `:`), and `vcat` of parts
built on the same basis — see those methods for train/test splitting and
incremental data addition. The recommended construction path from DFT data is
`SLCEDataset(basis, data::AbstractVector{TrainingDatum})` (see the io layer), which
derives the torque/force row sets from each datum's channels and provenance.
"""
struct SLCEDataset
    basis::SLCEBasis
    configs::Vector{Matrix{Float64}}
    # Per-config displacement fields (3 × n_atoms, Å, measured from the pinned
    # clamped-ion reference). Empty for pure-spin datasets; length == length(configs)
    # whenever displacement data enter (a spin-only datum against a joint basis
    # stores an explicit zero field — "u = 0 at the reference" is data, not absence).
    disps::Vector{Matrix{Float64}}
    X_E::Matrix{Float64}
    y_E::Vector{Float64}
    X_T::Matrix{Float64}
    y_T::Vector{Float64}
    torque_config::Vector{Int}   # config index of each torque row (nondecreasing)
    # Force block (ragged, same stored-bookkeeping contract as the torque block).
    # `X_F` is COMPACT: its columns are only the displacement-active SALCs listed in
    # `force_cols` (a pure-spin SALC has an identically zero force derivative — the
    # zero columns are never materialized; `_assemble_problem` scatters the block
    # into the full design width). Rows exist only for force-bearing configs
    # (`force_config[r]` is row r's config index) and only for atoms some SALC's
    # displacement slot actually reads — structurally zero rows are excluded, never
    # padded, or the `√(w_F/n_F)` whitening silently dilutes the real forces.
    X_F::Matrix{Float64}
    y_F::Vector{Float64}
    force_config::Vector{Int}    # config index of each force row (nondecreasing)
    force_cols::Vector{Int}      # SALC columns with displacement content (increasing)
    # ASR reparameterization (basis property, `force_cols` discipline): `nothing`
    # on pure-spin bases — the structural fast path that keeps those fits bitwise
    # identical. Built once at construction, carried by slicing/`vcat`.
    asr::Union{Nothing,ASRReparam}
    # Dataset-level identity summary (setup_id / soc / reference_id /
    # reference_fingerprint; the per-datum booleans stay false here). The
    # one-setup / one-reference invariants are checked at construction from
    # TrainingDatum vectors — storing the identity lets `vcat` (the documented
    # incremental-addition path) re-assert them instead of silently bypassing.
    provenance::DatumProvenance

    function SLCEDataset(basis::SLCEBasis, configs::Vector{Matrix{Float64}},
                        X_E::Matrix{Float64}, y_E::Vector{Float64},
                        X_T::Matrix{Float64}, y_T::Vector{Float64},
                        torque_config::Vector{Int},
                        provenance::DatumProvenance = DatumProvenance();
                        disps::Vector{Matrix{Float64}} = Matrix{Float64}[],
                        X_F::Matrix{Float64} = Matrix{Float64}(undef, 0, 0),
                        y_F::Vector{Float64} = Float64[],
                        force_config::Vector{Int} = Int[],
                        force_cols::Vector{Int} = Int[],
                        asr::Union{Nothing,ASRReparam} = nothing)
        nc = length(configs)
        (size(X_E, 1) == nc && length(y_E) == nc) ||
            throw(DimensionMismatch("X_E/y_E rows ($(size(X_E, 1))/$(length(y_E))) " *
                                    "≠ number of configs $nc"))
        (size(X_T, 1) == length(y_T) == length(torque_config)) ||
            throw(DimensionMismatch("X_T rows $(size(X_T, 1)), y_T length " *
                                    "$(length(y_T)), torque_config length " *
                                    "$(length(torque_config)) must agree"))
        issorted(torque_config) ||
            throw(ArgumentError("torque_config must be nondecreasing (config-major " *
                                "torque row order)"))
        if !isempty(torque_config)
            (torque_config[1] >= 1 && torque_config[end] <= nc) ||
                throw(ArgumentError("torque_config entries must index configs 1:$nc"))
        end
        isempty(disps) || length(disps) == nc ||
            throw(DimensionMismatch("disps has $(length(disps)) displacement fields " *
                                    "for $nc configs (empty = no displacement data)"))
        (size(X_F, 1) == length(y_F) == length(force_config)) ||
            throw(DimensionMismatch("X_F rows $(size(X_F, 1)), y_F length " *
                                    "$(length(y_F)), force_config length " *
                                    "$(length(force_config)) must agree"))
        isempty(y_F) || size(X_F, 2) == length(force_cols) ||
            throw(DimensionMismatch("X_F has $(size(X_F, 2)) columns but force_cols " *
                                    "lists $(length(force_cols)) displacement-active " *
                                    "SALCs (X_F is stored compact)"))
        issorted(force_cols; lt = <=) ||               # strictly increasing
            throw(ArgumentError("force_cols must be strictly increasing SALC column " *
                                "indices"))
        if !isempty(force_cols)
            (force_cols[1] >= 1 && force_cols[end] <= n_salcs(basis)) ||
                throw(ArgumentError("force_cols entries must index SALC columns " *
                                    "1:$(n_salcs(basis))"))
        end
        issorted(force_config) ||
            throw(ArgumentError("force_config must be nondecreasing (config-major " *
                                "force row order)"))
        if !isempty(force_config)
            (force_config[1] >= 1 && force_config[end] <= nc) ||
                throw(ArgumentError("force_config entries must index configs 1:$nc"))
            isempty(disps) &&
                throw(ArgumentError("force rows present but disps is empty — force " *
                                    "targets are meaningless without the displacement " *
                                    "fields they were computed at"))
        end
        asr === nothing || size(asr.A, 2) == n_salcs(basis) ||
            throw(DimensionMismatch("asr constraint matrix has $(size(asr.A, 2)) " *
                                    "columns for $(n_salcs(basis)) SALC columns"))
        return new(basis, configs, disps, X_E, y_E, X_T, y_T, torque_config,
                   X_F, y_F, force_config, force_cols, asr, provenance)
    end
end

"""
    has_torque(dataset) -> Bool

Whether `dataset` carries torque training data (so an energy+torque co-fit is
possible).
"""
has_torque(d::SLCEDataset) = !isempty(d.y_T)

"""
    has_force(dataset) -> Bool

Whether `dataset` carries force training data (so an energy+force co-fit is
possible).
"""
has_force(d::SLCEDataset) = !isempty(d.y_F)

# Atoms whose spin some SALC actually reads (a SPIN slot site). A species removed by
# `lmax = 0` (or a site that never enters an admitted cluster) is unreferenced, and
# its training moments are never consulted. Channel-aware on purpose: in a mixed SALC
# a displacement-only site (e.g. a spin-inactive ligand) appears among the member
# atoms but must NOT count as spin-referenced — the zero-moment guard would otherwise
# reject legitimate ligand configurations. On a pure-spin basis every member atom is
# a spin site, so this reduces to the plain member-atom scan.
function _referenced_atoms(basis::SLCEBasis)::BitVector
    ref = falses(n_atoms(basis.crystal))
    for s in salcs(basis), mem in s.members, t in mem.terms, sl in t.slots
        if sl.factor.channel == SPIN
            ref[mem.atoms[sl.site]] = true
        end
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

# Locate the derivative-channel rows of the selected configs through the stored
# per-row config index (ragged layout: a config without data in the channel has no
# rows), never through a uniform 3·n_atoms block assumption. The index is
# nondecreasing, so each config's rows are one `searchsorted` range; duplicate
# indices (bootstrap resampling) duplicate the rows. Returns the sliced
# `(X, y, config_index)` triple.
function _slice_channel_rows(X::Matrix{Float64}, y::Vector{Float64},
                             cfg::Vector{Int}, idx::AbstractVector{<:Integer})
    isempty(cfg) && return X[1:0, :], Float64[], Int[]
    rows = Int[]
    rc = Int[]
    for (k, i) in enumerate(idx)
        r = searchsorted(cfg, i)
        isempty(r) && continue
        append!(rows, r)
        append!(rc, fill(k, length(r)))
    end
    return X[rows, :], y[rows], rc
end

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
    dsp = isempty(d.disps) ? d.disps : d.disps[idx]
    X_E = d.X_E[idx, :]
    y_E = d.y_E[idx]
    X_T, y_T, tc = _slice_channel_rows(d.X_T, d.y_T, d.torque_config, idx)
    X_F, y_F, fc = _slice_channel_rows(d.X_F, d.y_F, d.force_config, idx)
    return SLCEDataset(d.basis, cfgs, X_E, y_E, X_T, y_T, tc, d.provenance;
                      disps = dsp, X_F = X_F, y_F = y_F, force_config = fc,
                      force_cols = d.force_cols, asr = d.asr)
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
with one built in-session. Parts may differ in torque presence (a torque-bearing
part and an energy-only part concatenate into a mixed dataset; each torque row
keeps its configuration through the re-offset `torque_config`). Together with
`dataset[idx]` this supports incremental data addition and resampling without
rebuilding design matrices.
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
        # The one-setup / one-reference invariants are checked when a dataset is
        # built from TrainingDatum vectors; vcat (the incremental-addition path)
        # must re-assert them or it silently reintroduces exactly the
        # family-correlated energy-offset bias they exist to prevent.
        b.provenance == a.provenance ||
            throw(ArgumentError("dataset $k has a different setup/reference " *
                                "identity (setup_id = $(repr(b.provenance.setup_id)), " *
                                "soc = $(repr(b.provenance.soc)), reference_id = " *
                                "$(repr(b.provenance.reference_id))) than dataset 1 " *
                                "(setup_id = $(repr(a.provenance.setup_id)), soc = " *
                                "$(repr(a.provenance.soc)), reference_id = " *
                                "$(repr(a.provenance.reference_id))) — one dataset " *
                                "must come from one computational setup and one " *
                                "clamped-ion reference"))
    end
    # Displacement fields: parts must agree on whether they carry them — a part
    # without disps has no recorded geometry, and concatenating it into a
    # displacement-bearing dataset would silently fabricate one.
    anydisp = any(p -> !isempty(p.disps), parts)
    if anydisp && !all(p -> !isempty(p.disps), parts)
        throw(ArgumentError("cannot vcat displacement-bearing and displacement-free " *
                            "datasets: a part without stored displacement fields has " *
                            "no recorded geometry (a spin-only part built against a " *
                            "joint basis stores explicit zeros)"))
    end
    # Force columns are a property of the shared (fingerprint-checked) basis, not of
    # the rows: take them from ANY part that carries them — including a part whose
    # force rows were sliced away — so slicing and re-concatenating never silently
    # drops the column set. Force-bearing parts must agree on them.
    fcols = Int[]
    for p in parts
        isempty(p.force_cols) && continue
        if isempty(fcols)
            fcols = p.force_cols
        elseif p.force_cols != fcols
            throw(ArgumentError("parts disagree on force_cols — the compact X_F " *
                                "blocks are not column-compatible"))
        end
    end
    # The ASR reparameterization is a basis property like `force_cols`: take it
    # from any part that carries one, but refuse silent disagreement — a
    # hand-built part missing its `asr` must not inherit another part's Z/A
    # unnoticed when other parts carry a DIFFERENT reparameterization object.
    asr = nothing
    for p in parts
        p.asr === nothing && continue
        if asr === nothing
            asr = p.asr
        elseif !(p.asr === asr || (p.asr.A == asr.A && p.asr.Z == asr.Z))
            throw(ArgumentError("parts disagree on the ASR reparameterization — " *
                                "rebuild the hand-built part with " *
                                "asr = SLCE.build_asr(basis)"))
        end
    end
    tc = Int[]
    fc = Int[]
    off = 0
    for p in parts
        append!(tc, p.torque_config .+ off)
        append!(fc, p.force_config .+ off)
        off += length(p)
    end
    fparts = [p.X_F for p in parts if !isempty(p.y_F)]
    X_F = isempty(fparts) ? Matrix{Float64}(undef, 0, 0) : reduce(vcat, fparts)
    return SLCEDataset(a.basis,
                      reduce(vcat, [p.configs for p in parts]),
                      reduce(vcat, [p.X_E for p in parts]),
                      reduce(vcat, [p.y_E for p in parts]),
                      reduce(vcat, [p.X_T for p in parts]),
                      reduce(vcat, [p.y_T for p in parts]),
                      tc, a.provenance;
                      disps = anydisp ? reduce(vcat, [p.disps for p in parts]) :
                              Matrix{Float64}[],
                      X_F = X_F,
                      y_F = reduce(vcat, [p.y_F for p in parts]),
                      force_config = fc,
                      force_cols = fcols,
                      asr = asr)
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
# displacement-decorated basis at the boundary instead of erroring inside the
# threaded design assembly.
function _require_pure_spin_basis(basis::SLCEBasis)
    all(s -> all(is_pure_spin, s.key.decors), salcs(basis)) ||
        throw(ArgumentError("the basis carries displacement-decorated SALCs — " *
                            "the spin-configuration SLCEDataset path is pure-spin " *
                            "only; build the joint dataset from TrainingDatum " *
                            "vectors (SLCEDataset(basis, data))"))
    return nothing
end

# Keyed off the SPEC as well as the surviving SALCs: a displacement-decorated spec
# whose SALCs all happen to be annihilated by symmetry must still pin the reference
# (the data were generated in a p ≥ 1 setting) — surviving-SALC inspection alone
# would fail OPEN on an invariant that exists to fail loud.
_basis_has_disp(basis::SLCEBasis)::Bool =
    any(>(0), basis.spec.pmax) || any(s -> any(has_disp, s.key.decors), salcs(basis))

# SALC columns with displacement content — the only columns with a nonzero force
# derivative (`∂Φ/∂u ≡ 0` for a pure-spin SALC), i.e. the compact column set of the
# stored force design block `X_F`.
_disp_active_cols(basis::SLCEBasis)::Vector{Int} =
    findall(s -> any(has_disp, s.key.decors), salcs(basis))

# Atoms some SALC displacement slot actually reads — the sites with a structurally
# nonzero force prediction. Read off the evaluator's own slot→site maps (member
# atom order and key decor order need not align after canonicalization), so this is
# exactly the set of atoms `accumulate_grad!` can write a nonzero `Gu` column for.
function _disp_referenced_atoms(basis::SLCEBasis)::BitVector
    ref = falses(n_atoms(basis.crystal))
    for s in salcs(basis), m in s.members, t in m.terms, sl in t.slots
        if sl.factor.channel == DISP
            ref[m.atoms[sl.site]] = true
        end
    end
    return ref
end

function SLCEDataset(basis::SLCEBasis, configs::AbstractVector, energies::AbstractVector;
                   atol::Real = 1e-6,
                   provenance::DatumProvenance = DatumProvenance())::SLCEDataset
    length(configs) == length(energies) ||
        throw(DimensionMismatch("got $(length(configs)) configs but $(length(energies)) energies"))
    _require_pure_spin_basis(basis)
    cfgs = [Matrix{Float64}(c) for c in configs]
    _validate_configs(basis, cfgs; atol = atol)
    X = _design_energy(basis, cfgs)
    empty_T = Matrix{Float64}(undef, 0, size(X, 2))
    return SLCEDataset(basis, cfgs, X, collect(Float64, energies), empty_T, Float64[],
                      Int[], provenance)
end

SLCEDataset(basis::SLCEBasis, configs::AbstractVector, energies::AbstractVector,
           torques::AbstractVector; kwargs...)::SLCEDataset =
    SLCEDataset(basis, configs, energies, torques, 1:length(configs); kwargs...)

function SLCEDataset(basis::SLCEBasis, configs::AbstractVector, energies::AbstractVector,
                   torques::AbstractVector, torque_sel::AbstractVector{<:Integer};
                   atol::Real = 1e-6,
                   provenance::DatumProvenance = DatumProvenance())::SLCEDataset
    isempty(configs) && throw(ArgumentError("no training configurations"))
    length(configs) == length(energies) ||
        throw(DimensionMismatch("got $(length(configs)) configs but $(length(energies)) energies"))
    _require_pure_spin_basis(basis)
    cfgs = [Matrix{Float64}(c) for c in configs]
    sel = collect(Int, torque_sel)
    issorted(sel; lt = <=) ||                          # strictly increasing
        throw(ArgumentError("torque_sel must be strictly increasing"))
    isempty(sel) && throw(ArgumentError("torque_sel is empty — use the three-argument " *
                                        "energy-only form instead"))
    (sel[1] >= 1 && sel[end] <= length(cfgs)) ||
        throw(ArgumentError("torque_sel entries must index configs 1:$(length(cfgs))"))
    length(torques) == length(sel) ||
        throw(ArgumentError("got $(length(torques)) torque blocks for " *
                            "$(length(sel)) torque-bearing configs"))
    _validate_configs(basis, cfgs; atol = atol)
    X_E = _design_energy(basis, cfgs)
    tcfgs = cfgs[sel]
    X_T = _design_torque(basis, tcfgs)
    y_T = _flatten_torques(torques, tcfgs)
    tc = repeat(sel; inner = 3 * n_atoms(basis.crystal))
    return SLCEDataset(basis, cfgs, X_E, collect(Float64, energies), X_T, y_T, tc,
                      provenance)
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
matrices), the fitted `j0`/`jphi`, the `estimator`, the `torque_weight` /
`force_weight` used, whether the ASR constraint was applied (`asr`, with the
achieved relative residual `asr_residual` — see [`asr_residual`](@ref)), and the
(energy) residuals. This is the heavyweight, data-bearing object you query for
diagnostics ([`r2_energy`](@ref), [`rmse_energy`](@ref),
[`residuals_energy`](@ref), …). For prediction and storage, convert it to the
lightweight [`SLCEModel`](@ref) with `SLCEModel(fit)`.
"""
struct SLCEFit
    dataset::SLCEDataset
    j0::Float64
    jphi::Vector{Float64}
    estimator::AbstractEstimator
    residuals::Vector{Float64}
    torque_weight::Float64
    force_weight::Float64
    # Whether the ASR reparameterization was APPLIED (false for pure-spin bases
    # even under `asr = true` — there is nothing to constrain) and the achieved
    # relative residual ‖A·β‖/(‖A‖·‖β‖). Part of the re-assembly contract:
    # `refit`/`gcv`/`effective_dof` reconstruct the same problem from these
    # fields, exactly like `torque_weight`/`force_weight`.
    asr::Bool
    asr_residual::Float64
    # The reparameterization the estimator actually solved under, or `nothing` for
    # a plain unconstrained fit: `dataset.asr` for an ordinary constrained fit, and
    # the STAGE's affine reparameterization for a staged one (`frozen` /
    # `sector_mask` — its `Z` pins the frozen columns and its `beta_p` carries their
    # values). Every consumer that re-derives the solve — `refit`, `gcv`,
    # `effective_dof`, `identifiability`, `dof` — must read this field rather than
    # `dataset.asr`, or a staged fit is silently re-assembled as an unstaged one.
    reparam::Union{Nothing,ASRReparam}
end
