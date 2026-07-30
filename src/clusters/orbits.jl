"""
    ClusterOrbit

A symmetry-equivalence class of cluster instances: `members` are all candidate
clusters related by the space group, `representative` is the canonical one,
`multiplicity = length(members)`, and `species` is the per-site species of the
representative.
"""
struct ClusterOrbit
    body::Int
    representative::ClusterMember
    members::Vector{ClusterMember}
    multiplicity::Int
    species::Vector{Int}
end

"""
    ClusterSet

Cluster orbits grouped by body order. See [`build_clusters`](@ref).
"""
struct ClusterSet
    by_body::Dict{Int,Vector{ClusterOrbit}}
end

# Image of site (atom, R) under operation `g`: the image atom `b = map_sym[atom,g]`
# carries lattice translation `τ_{g,atom} + W·R`, where τ closes the home-cell map.
@inline function _site_image(crystal::Crystal, sg::SpaceGroup, g::Int,
                             atom::Int, R::SVector{3,Int})
    op = sg.ops[g]
    b = sg.map_sym[atom, g]
    xa = SVector{3,Float64}(crystal.frac_positions[1, atom],
                            crystal.frac_positions[2, atom],
                            crystal.frac_positions[3, atom])
    xb = SVector{3,Float64}(crystal.frac_positions[1, b],
                            crystal.frac_positions[2, b],
                            crystal.frac_positions[3, b])
    τ = round.(Int, op.rotation_frac * xa + op.translation_frac - xb)
    Wi = round.(Int, op.rotation_frac)
    return b, SVector{3,Int}(τ + Wi * R)
end

# Sort a small tuple of site tuples into ascending order without heap allocation
# (the `_canonical_key` inner loop runs n_candidates × n_ops × N times).
@inline _sorted_sites(t::NTuple{1}) = t
@inline _sorted_sites(t::NTuple{2}) = t[1] <= t[2] ? t : (t[2], t[1])
@inline function _sorted_sites(t::NTuple{N}) where {N}
    v = MVector(t)                         # stack-allocated for isbits eltype
    @inbounds for i = 2:N                  # insertion sort (N ≤ nbody, small)
        x = v[i]
        j = i - 1
        while j >= 1 && v[j] > x
            v[j+1] = v[j]
            j -= 1
        end
        v[j+1] = x
    end
    return Tuple(v)
end

# Canonical key of a cluster: the lexicographic minimum, over all symmetry images
# and all choices of anchor site, of the site list sorted as (atom, Rx, Ry, Rz).
# Two clusters share a key iff they lie in the same orbit. A `Val(N)` barrier keeps
# the per-image site tuples statically sized so the whole hot loop stays on the stack.
_canonical_key(crystal::Crystal, sg::SpaceGroup, m::ClusterMember) =
    _canonical_key(crystal, sg, m, Val(length(m.atoms)))

function _canonical_key(crystal::Crystal, sg::SpaceGroup, m::ClusterMember,
                        ::Val{N}) where {N}
    atoms = m.atoms
    shifts = m.shifts
    best = ntuple(_ -> (0, 0, 0, 0), Val(N))
    found = false
    for g = 1:n_ops(sg)
        imgs = ntuple(k -> _site_image(crystal, sg, g, atoms[k], shifts[k]), Val(N))
        for s = 1:N
            R0 = imgs[s][2]
            sites = ntuple(k -> (imgs[k][1], imgs[k][2][1] - R0[1], imgs[k][2][2] - R0[2],
                                 imgs[k][2][3] - R0[3]), Val(N))
            key = _sorted_sites(sites)
            if !found || key < best
                best = key
                found = true
            end
        end
    end
    return best
end

# Translation-normalized signature of a cluster: its site list sorted and shifted so
# the first sorted site sits at R = 0. Invariant under an overall lattice translation
# and under the anchor choice, so the several anchor-variants that `candidate_clusters`
# emits for one physical cluster (it anchors on each home-cell atom in turn) collapse to
# a single signature. Used to grow orbits without an O(n_ops) canonical key per candidate.
@inline function _normalize_sites(sites::NTuple{N,NTuple{4,Int}}) where {N}
    s = _sorted_sites(sites)
    R0 = s[1]
    return ntuple(k -> (s[k][1], s[k][2] - R0[2], s[k][3] - R0[3], s[k][4] - R0[4]), Val(N))
end

_member_sig(m::ClusterMember) = _member_sig(m, Val(length(m.atoms)))
@inline function _member_sig(m::ClusterMember, ::Val{N}) where {N}
    sites = ntuple(k -> (m.atoms[k], m.shifts[k][1], m.shifts[k][2], m.shifts[k][3]), Val(N))
    return _normalize_sites(sites)
end

# Signature of the image of `m` under operation `g` — the same normalization applied to
# the symmetry-mapped sites, so it matches the signature of whichever candidate equals it.
_image_sig(crystal::Crystal, sg::SpaceGroup, g::Int, m::ClusterMember) =
    _image_sig(crystal, sg, g, m, Val(length(m.atoms)))
@inline function _image_sig(crystal::Crystal, sg::SpaceGroup, g::Int, m::ClusterMember,
                            ::Val{N}) where {N}
    imgs = ntuple(k -> _site_image(crystal, sg, g, m.atoms[k], m.shifts[k]), Val(N))
    sites = ntuple(k -> (imgs[k][1], imgs[k][2][1], imgs[k][2][2], imgs[k][2][3]), Val(N))
    return _normalize_sites(sites)
end

"""
    build_clusters(crystal, neighbors, spacegroup; nbody = 2, selection = MinimumImage())
        -> ClusterSet

Enumerate candidate clusters and reduce them to symmetry orbits under
`spacegroup`. Orbits are ordered deterministically by their canonical key.

# Arguments
- `crystal::Crystal`, `neighbors::NeighborList`, `spacegroup::SpaceGroup`.

# Keyword arguments
- `nbody::Integer = 2`: maximum body order.
- `selection::AbstractImageSelection = MinimumImage()`: the image-admissibility rule
  for the cluster edges; must match the one that built `neighbors` (the same rule the
  internal `candidate_clusters` enumeration applies).
- `cutoff = nothing`: optional per-body-order, per-species-pair edge radii (the
  [`BasisSpec`](@ref) canonical `Vector{Matrix}` form) — see
  `candidate_clusters`.

# Returns
- `ClusterSet`: orbits grouped by body order.
"""
function build_clusters(crystal::Crystal, neighbors::NeighborList, spacegroup::SpaceGroup;
                        nbody::Integer = 2,
                        selection::AbstractImageSelection = MinimumImage(),
                        cutoff::Union{Nothing,AbstractVector{<:AbstractMatrix{<:Real}}} =
                            nothing)::ClusterSet
    cand = candidate_clusters(crystal, neighbors, nbody; selection = selection,
                              cutoff = cutoff)
    by_body = Dict{Int,Vector{ClusterOrbit}}()
    for body in sort(collect(keys(cand)))
        members = cand[body]
        # Group candidates into translation classes (anchor-variants of one cluster share
        # a signature). `order` keeps the first-seen (enumeration) order of the classes.
        sig2members = Dict{Any,Vector{Int}}()
        order = Any[]
        for i in eachindex(members)
            s = _member_sig(members[i])
            v = get(sig2members, s, nothing)
            if v === nothing
                sig2members[s] = Int[i]
                push!(order, s)
            else
                push!(v, i)
            end
        end
        # Grow each orbit from one representative class by applying every operation: the
        # reachable signatures are that class's symmetry orbit, so the canonical key (the
        # O(n_ops) part) is computed once per orbit, not once per candidate.
        assigned = Set{Any}()
        keyed = Tuple{Any,Vector{ClusterMember}}[]
        for s0 in order
            s0 in assigned && continue
            rep_member = members[sig2members[s0][1]]
            orbit_sigs = Set{Any}()
            for g = 1:n_ops(spacegroup)
                push!(orbit_sigs, _image_sig(crystal, spacegroup, g, rep_member))
            end
            idxs = Int[]
            for s in orbit_sigs
                push!(assigned, s)
                # The candidate list is closed under the space group, so every image of a
                # candidate IS a candidate. Assert it instead of skipping: a missing image
                # would leave the orbit short — a biased design column whose orbit sum omits
                # cluster instances — and `assigned` has already swallowed the signature, so
                # it would never reappear as an orbit of its own. Either the invariant holds
                # (no cost) or it is violated (must be loud).
                haskey(sig2members, s) || error(
                    "cluster enumeration is not closed under the space group: the image " *
                    "$s of a candidate cluster is not itself a candidate. The orbit would " *
                    "be built short, silently biasing its design column — check the image " *
                    "selection and the tie tolerance rather than proceeding")
                append!(idxs, sig2members[s])
            end
            sort!(idxs)
            ms = ClusterMember[members[j] for j in idxs]
            push!(keyed, (_canonical_key(crystal, spacegroup, ms[1]), ms))
        end
        sort!(keyed; by = first)
        orbits = ClusterOrbit[ClusterOrbit(body, ms[1], ms, length(ms),
                              Int[crystal.species[a] for a in ms[1].atoms])
                              for (_, ms) in keyed]
        by_body[body] = orbits
    end
    return ClusterSet(by_body)
end
