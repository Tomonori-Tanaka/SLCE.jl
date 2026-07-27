# Sector-driven SALC basis construction (M2b-3b): expand a resolved sector
# table (`BasisSpec.sectors`, see `slce/truncation.jl`) into per-orbit
# decoration labels and project them through the mixed-channel decor engine
# (`basis/salcbasis.jl`). Lives in its own file because it consumes the spec
# types, which are include-ordered after the basis layer.

# The decoration labels one sector rule contributes to an `N`-body orbit whose
# sites have the given species. `lsumN` is the spec-global per-body Σl_spin cap.
function _sector_orbit_labels(rule::SectorRule, N::Int, species_present::Vector{Int},
                              lmax::Vector{Int}, pmax::Vector{Int},
                              lsumN::Int)::Vector{Vector{SiteDecor}}
    rule.nbody[1] <= N <= rule.nbody[2] || return Vector{SiteDecor}[]
    lcap_present = maximum(lmax[sp] for sp in species_present)
    pcap_present = maximum(pmax[sp] for sp in species_present)
    # spin multiset candidates
    spins = Vector{Int}[]
    if rule.spin_mode == :none
        push!(spins, Int[])
    elseif rule.spin_mode == :explicit
        S = rule.spin_ls
        if all(l -> l <= lcap_present, S) &&
           (lsumN == LSUM_UNCAPPED || sum(S) <= lsumN)
            push!(spins, S)
        end
    else # :any
        lcap = min(rule.spin_lmax, lcap_present)
        lsum_cap = min(rule.spin_lsum, lsumN)
        for ns = rule.spin_nsites[1]:min(rule.spin_nsites[2], N)
            append!(spins, _spin_multisets(ns, lcap, lsum_cap))
        end
    end
    isempty(spins) && return Vector{SiteDecor}[]
    # displacement factor multiset candidates (degree 0 = the no-disp option,
    # admitted when the sector's budget starts at 0)
    disps = Vector{Tuple{Int,Int}}[]
    plo, phi = rule.disp_degree
    for p = plo:phi
        append!(disps, _disp_multisets(p, pcap_present))
    end
    isempty(disps) && return Vector{SiteDecor}[]
    out = Vector{SiteDecor}[]
    for S in spins, D in disps
        append!(out, _marry_multisets(S, D, N))
    end
    return out
end

# Sector admission of an orbit: body order in range, every edge within the
# sector's species-pair radius (same relative band as `candidate_clusters`).
function _sector_admits(rule::SectorRule, N::Int,
                        gates::Vector{Tuple{Int,Int,Float64}}, tol::Float64)::Bool
    rule.nbody[1] <= N <= rule.nbody[2] || return false
    fac = (1 + tol)^2
    for (spa, spb, gate2) in gates
        gate2 <= rule.cutoff[spa, spb]^2 * fac || return false
    end
    return true
end

# All SALCs of one cluster orbit under a sector table: union the admitting
# sectors' labels (per-label effective `soc` = OR over sectors — a label under
# both a soc=false and a soc=true sector gets all its L_S blocks exactly once:
# the key-union invariant), then project via the decor engine with the
# per-species caps.
function _orbit_salcs_sectors(crystal::Crystal, spacegroup::SpaceGroup, N::Int,
                              orbit_id::Int, O::ClusterOrbit, spec::BasisSpec,
                              gates::Vector{Tuple{Int,Int,Float64}}, tol::Float64,
                              wcache::_WigCache)::Vector{SALC}
    lsumN = spec.lsum[N]
    soc_by_label = Dict{Vector{SiteDecor},Bool}()
    for rule in spec.sectors
        _sector_admits(rule, N, gates, tol) || continue
        for label in _sector_orbit_labels(rule, N, O.species, spec.lmax, spec.pmax,
                                          lsumN)
            soc_by_label[label] = get(soc_by_label, label, false) | rule.soc
        end
    end
    isempty(soc_by_label) && return SALC[]
    l_soc = sort!([l for (l, s) in soc_by_label if s])
    l_nosoc = sort!([l for (l, s) in soc_by_label if !s])
    out = SALC[]
    for (labels, soc) in ((l_soc, true), (l_nosoc, false))
        isempty(labels) && continue
        append!(out, _orbit_salcs_decors(crystal, spacegroup, N, orbit_id, O,
                                         labels, soc, wcache;
                                         lmax_by_species = spec.lmax,
                                         pmax_by_species = spec.pmax))
    end
    return out
end


"""
    build_salc_basis(crystal, spacegroup, clusters, spec::BasisSpec;
                     neighbors, selection) -> SALCBasis

Sector-driven construction: for every cluster orbit, union the labels of the
sectors admitting it (per-sector cutoff re-admission against `neighbors`'s
distances under `selection`'s semantics; per-label effective `soc` = OR over
the admitting sectors) and project them through the mixed-channel decor
engine. The spec's per-species `lmax`/`pmax` and per-body `lsum` caps apply
throughout. The result is sorted by [`SALCKey`](@ref) and key uniqueness is
asserted (the key-union invariant: overlapping sectors must never produce
duplicate — hence exactly collinear — design columns).
"""
function build_salc_basis(crystal::Crystal, spacegroup::SpaceGroup,
                          clusters::ClusterSet, spec::BasisSpec;
                          neighbors::NeighborList,
                          selection::AbstractImageSelection = MinimumImage())::SALCBasis
    isempty(spec.sectors) &&
        throw(ArgumentError("the spec has no sectors — use the dense keyword " *
                            "form of build_salc_basis"))
    # Wigner cache over BOTH channels' ranks: spin l ≤ max(lmax), displacement
    # l ≤ the maximum total degree (k = 0 realizes l = 2k + l).
    maxl = max(maximum(spec.lmax; init = 0),
               maximum(r.disp_degree[2] for r in spec.sectors; init = 0))
    wcache = _build_wig_cache(spacegroup, maxl)
    use_minimage = selection isa MinimumImage
    dmin2 = use_minimage ? _dmin2_matrix(neighbors, n_atoms(crystal)) :
            Matrix{Float64}(undef, 0, 0)
    cart = cartesian_positions(crystal)      # read-only, shared across orbits
    work = Tuple{Int,Int,ClusterOrbit}[]
    for N in sort(collect(keys(clusters.by_body)))
        for (orbit_id, O) in enumerate(clusters.by_body[N])
            push!(work, (N, orbit_id, O))
        end
    end
    parts = Vector{Vector{SALC}}(undef, length(work))
    Threads.@threads for w in eachindex(work)
        (N, orbit_id, O) = work[w]
        gates = _orbit_edge_gates(crystal, O, cart, dmin2, use_minimage)
        parts[w] = _orbit_salcs_sectors(crystal, spacegroup, N, orbit_id, O, spec,
                                        gates, neighbors.tol, wcache)
    end
    salcs = isempty(parts) ? SALC[] : reduce(vcat, parts)
    sort!(salcs; by = s -> s.key)
    keyvec = SALCKey[s.key for s in salcs]
    allunique(keyvec) ||
        error("duplicate SALC keys in the sector union — key-union invariant " *
              "violated (this is a bug; please report the spec)")
    return SALCBasis(salcs, keyvec)
end
