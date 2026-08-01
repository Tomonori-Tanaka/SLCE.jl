# SALC construction (arbitrary body order). For each cluster orbit, project the
# representative's coupled coefficient tensor onto the trivial irrep of its site
# stabilizer (orbit–stabilizer theorem), gauge-fix a deterministic orthonormal
# basis, then transport to all orbit members.
#
# At N ≥ 3 a stabilizer operation can permute symmetry-equivalent sites. That
# permutation (a) mixes coupling paths sharing the same Lf and (b) — when the
# permuted sites carry unequal l — mixes l-orderings. The projection therefore
# runs over the COMBINED space of (ordering o, coupling path p, Mf), with each
# operation acting by rotating every site axis (wignerD_real(o_i, R)) and then
# permuting the site axes by π; the action matrix is read off by contracting
# against the package's own orthonormal coupled tensors (no 6j/9j). A surviving
# invariant can span several orderings, so a SALC carries one `SALCTerm` per
# ordering (`basis/salc.jl`). For 1/2-body and equal-l channels this reduces to a
# single term and a single path, matching the earlier construction.
#
# The absolute drop thresholds below (1e-10/1e-12) and the idempotency guard (1e-6)
# are sized for the validated regime (per-site l ≤ 2, small body order), where the
# CG/recoupling products stay O(1e-3)–O(1). The ground-truth invariance test and the
# exact-0/1 eigenvalue assertion are the in-band gates; at much higher l these
# thresholds would want to scale with the coupled-tensor magnitude.

"""
Sentinel meaning "no `Σl` cap" for a body order (in `BasisSpec.lsum` and
`build_salc_basis`'s `lsum_by_body`). Only ever compared against (the per-site
enumeration cap subtracts at most `nbody − 1` from it, which cannot overflow).
"""
const LSUM_UNCAPPED = typemax(Int)

# Per-site l-tuples to enumerate for an orbit, one canonical representative per
# orbit of the site-permutation group `perms` (Σl even = time-reversal even,
# Σl ≤ lsumN = the body order's angular budget). Each per-site range is also
# tightened to `lsumN − (N − 1)` (every other site carries l ≥ 1), so an
# uncapped species lmax does not blow up the product before the filter.
function _enumerate_ls(N::Int, species::Vector{Int}, lmax::Vector{Int}, lsumN::Int,
                       perms::Vector{Vector{Int}})
    cap = lsumN - (N - 1)
    ranges = ntuple(i -> 1:min(lmax[species[i]], cap), N)
    seen = Set{Vector{Int}}()
    out = Vector{Int}[]
    for tt in Iterators.product(ranges...)
        t = collect(Int, tt)
        s = sum(t)
        (iseven(s) && s <= lsumN) || continue
        t in seen && continue
        orbit = unique([t[p] for p in perms])
        for o in orbit
            push!(seen, o)
        end
        push!(out, minimum(orbit))   # lex-min canonical of the permutation orbit
    end
    return out
end

# Site permutation aligning `src` (= g·rep images) to `dst` up to a common lattice
# translation: `perm[i] = j` iff `g·(rep siteᵢ)` lands on `dst[j]`. `nothing` if no
# alignment exists.
function _align(src::Vector{Tuple{Int,SVector{3,Int}}},
                dst::Vector{Tuple{Int,SVector{3,Int}}})
    N = length(src)
    @inbounds for j = 1:N
        src[1][1] == dst[j][1] || continue
        t = src[1][2] - dst[j][2]               # translation aligning src[1] → dst[j]
        perm = zeros(Int, N)
        used = falses(N)
        ok = true
        for i = 1:N
            atom = src[i][1]
            shift = src[i][2] - t
            found = 0
            for k = 1:N
                used[k] && continue
                if dst[k][1] == atom && dst[k][2] == shift
                    found = k
                    break
                end
            end
            found == 0 && (ok = false; break)
            perm[i] = found
            used[found] = true
        end
        ok && return perm
    end
    return nothing
end

_sites(m::ClusterMember) = Tuple{Int,SVector{3,Int}}[(m.atoms[i], m.shifts[i]) for i in eachindex(m.atoms)]

_op_images(crystal::Crystal, sg::SpaceGroup, g::Int, rep::ClusterMember) =
    Tuple{Int,SVector{3,Int}}[_site_image(crystal, sg, g, rep.atoms[i], rep.shifts[i])
                             for i in eachindex(rep.atoms)]

# Site stabilizer: ops mapping the (unordered) representative to itself, each with
# its induced site permutation.
function _stabilizer(crystal::Crystal, sg::SpaceGroup, rep::ClusterMember)
    dst = _sites(rep)
    out = Tuple{Int,Vector{Int}}[]
    for g = 1:n_ops(sg)
        perm = _align(_op_images(crystal, sg, g, rep), dst)
        perm === nothing || push!(out, (g, perm))
    end
    return out
end

# Translation signature of a raw site list (same normalization as `_member_sig`).
_sig_of_sites(sites::Vector{Tuple{Int,SVector{3,Int}}}) =
    _sig_of_sites(sites, Val(length(sites)))
@inline function _sig_of_sites(sites, ::Val{N}) where {N}
    t = ntuple(k -> (sites[k][1], sites[k][2][1], sites[k][2][2], sites[k][2][3]), Val(N))
    return _normalize_sites(t)
end

# Op + induced permutation connecting the representative to every orbit member
# (`g·rep` aligns to `member`), in a single O(n_ops) sweep: apply each op to `rep`
# once, match its image (by translation signature) to the member(s) of that class
# — the orbit lists every anchor-variant, so one class can hold several — and record
# the first-ascending connecting op for each. The transported term is
# stabilizer-invariant, so which connecting op is chosen is moot.
function _connect_all(crystal::Crystal, sg::SpaceGroup, rep::ClusterMember,
                      members::Vector{ClusterMember})
    M = length(members)
    sig2members = Dict{Any,Vector{Int}}()
    for j = 1:M
        push!(get!(sig2members, _member_sig(members[j]), Int[]), j)
    end
    conns = Vector{Tuple{Int,Vector{Int}}}(undef, M)
    filled = falses(M)
    remaining = M
    for g = 1:n_ops(sg)
        remaining == 0 && break
        imgs = _op_images(crystal, sg, g, rep)
        js = get(sig2members, _sig_of_sites(imgs), nothing)
        js === nothing && continue
        for j in js
            filled[j] && continue
            perm = _align(imgs, _sites(members[j]))
            perm === nothing && continue
            conns[j] = (g, perm)
            filled[j] = true
            remaining -= 1
        end
    end
    remaining == 0 || error("could not connect all orbit members to the representative")
    return conns
end

# The Mf-th multiplet slice of a coupled tensor (rank N+1 → rank N), as a view: the
# last axis is column-major-contiguous, so this is a strided view with no copy. Every
# consumer (`nmode_mul`, `_frobenius_inner`, the fold broadcast) reads it without mutating.
_mf_slice(T::AbstractArray, Mf::Int) = selectdim(T, ndims(T), Mf)

_frobenius_inner(A::AbstractArray, B::AbstractArray) = sum(A .* B)

# Real Wigner-D matrix `D^l(R_g)`, keyed by `(l, g)` for the whole basis build (it
# depends only on the operation and `l`, but is needed once per stabilizer column and
# once per transported member/term — recomputing it dominated the cost). The full
# `(l, g)` grid is bounded (`l ≤ lmax`, `g ≤ n_ops`) and cheap, so it is precomputed
# **serially** up front; the cache is then **read-only** during the threaded orbit
# loop (a concurrent `get!` on a shared `Dict` would race and corrupt it).
const _WigCache = Dict{Tuple{Int,Int},Matrix{Float64}}
function _build_wig_cache(sg::SpaceGroup, maxl::Int)::_WigCache
    cache = _WigCache()
    for l = 1:maxl, g = 1:n_ops(sg)
        cache[(l, g)] = AngularMomentum.wignerD_real(l, sg.ops[g].rotation_cart)
    end
    return cache
end
_wigner_d(cache::_WigCache, l::Int, g::Int) = cache[(l, g)]   # read-only lookup

# Project the combined (ordering, path, Mf) space onto stabilizer invariants for a
# fixed final `Lf`, gauge-fix, and fold each invariant into per-ordering tensors.
# Returns a list of blocks; each block is `Vector{(ls, folded)}` (the rep terms).
function _project_and_fold(stab::Vector{Tuple{Int,Vector{Int}}},
                           orderings::Vector{Vector{Int}}, cbs::Vector, Lf::Int,
                           wcache::_WigCache)
    N = length(orderings[1])
    # coupled tensors of this Lf, per ordering — selected from the prebuilt `cbs`
    # (built once per ordering in `_orbit_salcs`, not recomputed for each Lf).
    tens = [Array{Float64}[] for _ in orderings]
    for oi in eachindex(orderings)
        for cb in cbs[oi]
            cb.Lf == Lf && push!(tens[oi], cb.tensor)
        end
    end
    cols = Tuple{Int,Int,Int}[]            # (ordering index, path index, Mf index)
    for oi in eachindex(orderings), p in eachindex(tens[oi]), Mf = 1:(2Lf + 1)
        push!(cols, (oi, p, Mf))
    end
    D = length(cols)
    D == 0 && return Vector{Tuple{Vector{Int},Array{Float64}}}[]
    colidx = Dict(cols[k] => k for k = 1:D)

    P = zeros(Float64, D, D)
    for (g, perm) in stab
        for k = 1:D
            (oi, p, Mf) = cols[k]
            o = orderings[oi]
            v = _mf_slice(tens[oi][p], Mf)
            for i = 1:N
                v = AngularMomentum.nmode_mul(v, _wigner_d(wcache, o[i], g), i)
            end
            # U_g sends rep axis i → position π(i)=perm[i], i.e. permutedims by π⁻¹.
            q = invperm(perm)
            v = permutedims(v, q)
            oprime = o[q]
            oi2 = findfirst(==(oprime), orderings)
            oi2 === nothing && error("ordering set not closed under the stabilizer")
            for p2 in eachindex(tens[oi2]), Mf2 = 1:(2Lf + 1)
                c = _frobenius_inner(_mf_slice(tens[oi2][p2], Mf2), v)
                abs(c) < 1e-12 && continue
                P[colidx[(oi2, p2, Mf2)], k] += c
            end
        end
    end
    P ./= length(stab)
    P .= (P .+ P') ./ 2
    # `D` is tiny (a few dozen), so OpenBLAS runs this `eigen` serially even though
    # we are inside `Threads.@threads`; no Julia↔BLAS thread oversubscription in practice.
    F = eigen(Symmetric(P))
    all(λ -> abs(λ) < 1e-6 || abs(λ - 1) < 1e-6, F.values) ||
        error("SALC projector is not idempotent (eigenvalues $(F.values)); convention bug")
    V1 = F.vectors[:, findall(>(0.5), F.values)]
    W = _canonical_basis(V1)

    blocks = Vector{Tuple{Vector{Int},Array{Float64}}}[]
    for b = 1:size(W, 2)
        c = view(W, :, b)
        terms = Tuple{Vector{Int},Array{Float64}}[]
        for (oi, o) in enumerate(orderings)
            Fo = zeros(Float64, ntuple(i -> 2o[i] + 1, N)...)
            for p in eachindex(tens[oi]), Mf = 1:(2Lf + 1)
                coef = c[colidx[(oi, p, Mf)]]
                coef == 0.0 && continue
                Fo .+= coef .* _mf_slice(tens[oi][p], Mf)
            end
            @inbounds for index in eachindex(Fo)
                abs(Fo[index]) < 1e-10 && (Fo[index] = 0.0)
            end
            norm(Fo) > 1e-10 && push!(terms, (collect(o), Fo))
        end
        push!(blocks, terms)
    end
    return blocks
end

# Deterministic, BLAS-independent gauge: axis-pivoted modified Gram–Schmidt on the
# block projector, with a first-significant-component sign fix.
function _canonical_basis(V1::AbstractMatrix{Float64})
    n, d = size(V1)
    d == 0 && return zeros(Float64, n, 0)
    Q = V1 * V1'
    W = zeros(Float64, n, d)
    k = 0
    for j = 1:n
        u = Q[:, j]
        for i = 1:k
            u .-= dot(view(W, :, i), u) .* view(W, :, i)
        end
        nu = norm(u)
        if nu > 1e-8
            k += 1
            W[:, k] = u ./ nu
            _sign_canon!(view(W, :, k))
        end
        k == d && break
    end
    k == d || error("canonical gauge failed to span the invariant subspace")
    return W
end

function _sign_canon!(v)
    for x in v
        if abs(x) > 1e-8
            x < 0 && (v .*= -1)
            return
        end
    end
end

# Transport a rep term `(o, F)` to a member connected by `(g, perm)`: rotate each
# site axis by `wignerD_real(o_i, R_g)`, then relabel rep site i → member site
# perm[i]. Returns the member-ordered `(ls, folded)`.
function _transport_term(o::Vector{Int}, F::Array{Float64}, g::Int,
                         perm::Vector{Int}, wcache::_WigCache)
    N = length(o)
    T = F
    for i = 1:N
        T = AngularMomentum.nmode_mul(T, _wigner_d(wcache, o[i], g), i)
    end
    q = invperm(perm)                       # member axis j ← rep axis q[j]
    G = Array{Float64}(permutedims(T, q))
    @inbounds for index in eachindex(G)
        abs(G[index]) < 1e-10 && (G[index] = 0.0)
    end
    return o[q], G                          # member ls = o[invperm(perm)]
end

# ---------------------------------------------------------------------------
# Mixed-channel (decor) projection engine — joint M2b-2.
#
# The pure-spin `_project_and_fold`/`_transport_term` above stay the production
# path for spin-only bases (oracle-pinned bitwise); the `_decors` engine below
# generalizes the same construction to per-site `SiteDecor` assignments. The
# "engines agree on pure spin" testset in `test/unit/test_mixedsalc.jl` is the
# anti-drift gate coupling the two — change one and re-check the other.
#
# Slot conventions (see `basis/salc.jl`): an assignment `a::Vector{SiteDecor}`
# (per site) realizes the canonical slot list SPIN axes before DISP axes, each
# by site order. Coupling runs over the slot `l`s in slot order (spin first),
# so the total spin rank `L_S` is the running coupled momentum after the last
# spin slot — a good quantum number (site permutations act within channels and
# commute with the diagonal rotation), read off each `coupling_paths` entry by
# `_path_LS`. Both channels rotate through the SAME polar Wigner cache: the
# Σl_spin-even screen makes det(R)^{Σl_spin} ≡ +1, so the axial spin action
# equals the polar one (design record §4); displacement radial factors |u|^{2k}
# are rotation-invariant passive labels.
# ---------------------------------------------------------------------------

# Canonical slot list of a per-site decor assignment.
function _assignment_slots(a::Vector{SiteDecor})::Vector{Slot}
    slots = Slot[]
    for s in eachindex(a)
        has_spin(a[s]) && push!(slots, Slot(s, SiteFactor(SPIN, 0, a[s].spin_l)))
    end
    for s in eachindex(a)
        has_disp(a[s]) &&
            push!(slots, Slot(s, SiteFactor(DISP, a[s].disp_k, a[s].disp_l)))
    end
    return slots
end

_slot_ls(slots::Vector{Slot})::Vector{Int} = Int[sl.factor.l for sl in slots]
_n_spin_slots(slots::Vector{Slot})::Int =
    count(sl -> sl.factor.channel == SPIN, slots)

# Total spin rank of a left-coupling path over a spin-first slot list: the
# running coupled momentum after the last spin slot (0 / l₁ / Lseq / Lf edges).
function _path_LS(ls::Vector{Int}, Lseq::Vector{Int}, Lf::Int, n_spin::Int)::Int
    n_spin == 0 && return 0
    n_spin == 1 && return ls[1]
    n_spin == length(ls) && return Lf
    return Lseq[n_spin - 1]
end

# All distinct arrangements of a decor multiset over the cluster sites (swap
# recursion + dedup; orbit body orders are tiny). Sorted for determinism.
function _multiset_arrangements(label::Vector{SiteDecor})::Vector{Vector{SiteDecor}}
    n = length(label)
    out = Set{Vector{SiteDecor}}()
    perm = collect(1:n)
    function rec(k::Int)
        if k >= n
            push!(out, label[perm])
            return
        end
        for i = k:n
            perm[k], perm[i] = perm[i], perm[k]
            rec(k + 1)
            perm[k], perm[i] = perm[i], perm[k]
        end
    end
    n == 0 && return [SiteDecor[]]
    rec(1)
    return sort!(collect(out))
end

# Image of an assignment under a stabilizer site permutation (site s → perm[s]).
function _assignment_image(a::Vector{SiteDecor}, perm::Vector{Int})::Vector{SiteDecor}
    a2 = Vector{SiteDecor}(undef, length(a))
    for s in eachindex(a)
        a2[perm[s]] = a[s]
    end
    return a2
end

# Axis permutation induced on the canonical slot lists by a site permutation:
# σ[j] = position of the image of slot j (site perm[s], same factor) in the
# image assignment's canonical slot list. Unique by the (site, channel) slot
# invariant.
function _slot_sigma(slots::Vector{Slot}, slots2::Vector{Slot},
                     perm::Vector{Int})::Vector{Int}
    σ = Vector{Int}(undef, length(slots))
    for j in eachindex(slots)
        target = Slot(perm[slots[j].site], slots[j].factor)
        jp = findfirst(==(target), slots2)
        jp === nothing && error("slot lists not closed under the site permutation")
        σ[j] = jp
    end
    return σ
end

# Coupled bases of one assignment, tagged with L_S: (L_S, Lf, tensor) triples
# over the slot-order `l`s (spin first ⇒ L_S well-defined per path).
function _decor_coupled_bases(slots::Vector{Slot})
    ls = _slot_ls(slots)
    n_spin = _n_spin_slots(slots)
    out = Tuple{Int,Int,Array{Float64}}[]
    for (Lseq, Lf, tensor) in AngularMomentum.build_real_bases(ls)
        push!(out, (_path_LS(ls, Lseq, Lf, n_spin), Lf, tensor))
    end
    return out
end

# Decor-general `_project_and_fold`: project the combined (assignment, path, Mf)
# space onto stabilizer invariants for a fixed (L_S, Lf) block. Returns a list
# of invariant blocks; each is `Vector{(assignment index, folded)}`.
function _project_and_fold_decors(stab::Vector{Tuple{Int,Vector{Int}}},
                                  assignments::Vector{Vector{SiteDecor}},
                                  slotlists::Vector{Vector{Slot}},
                                  cbs::Vector, L_S::Int, Lf::Int,
                                  wcache::_WigCache)
    tens = [Array{Float64}[] for _ in assignments]
    for ai in eachindex(assignments)
        for (ls_path, lf_path, tensor) in cbs[ai]
            (ls_path == L_S && lf_path == Lf) && push!(tens[ai], tensor)
        end
    end
    cols = Tuple{Int,Int,Int}[]            # (assignment index, path index, Mf)
    for ai in eachindex(assignments), p in eachindex(tens[ai]), Mf = 1:(2Lf + 1)
        push!(cols, (ai, p, Mf))
    end
    D = length(cols)
    D == 0 && return Vector{Tuple{Int,Array{Float64}}}[]
    colidx = Dict(cols[k] => k for k = 1:D)

    P = zeros(Float64, D, D)
    for (g, perm) in stab
        for k = 1:D
            (ai, p, Mf) = cols[k]
            slots = slotlists[ai]
            v = _mf_slice(tens[ai][p], Mf)
            for j in eachindex(slots)
                slots[j].factor.l == 0 && continue          # D⁰ = 1 (trace axes)
                v = AngularMomentum.nmode_mul(v, _wigner_d(wcache, slots[j].factor.l, g), j)
            end
            a2 = _assignment_image(assignments[ai], perm)
            ai2 = findfirst(==(a2), assignments)
            ai2 === nothing && error("assignment set not closed under the stabilizer")
            σ = _slot_sigma(slots, slotlists[ai2], perm)
            v = permutedims(v, invperm(σ))
            for p2 in eachindex(tens[ai2]), Mf2 = 1:(2Lf + 1)
                c = _frobenius_inner(_mf_slice(tens[ai2][p2], Mf2), v)
                abs(c) < 1e-12 && continue
                P[colidx[(ai2, p2, Mf2)], k] += c
            end
        end
    end
    P ./= length(stab)
    P .= (P .+ P') ./ 2
    F = eigen(Symmetric(P))
    all(λ -> abs(λ) < 1e-6 || abs(λ - 1) < 1e-6, F.values) ||
        error("decor SALC projector is not idempotent (eigenvalues $(F.values))")
    V1 = F.vectors[:, findall(>(0.5), F.values)]
    W = _canonical_basis(V1)

    blocks = Vector{Tuple{Int,Array{Float64}}}[]
    for b = 1:size(W, 2)
        c = view(W, :, b)
        terms = Tuple{Int,Array{Float64}}[]
        for ai in eachindex(assignments)
            isempty(tens[ai]) && continue
            Fo = zeros(Float64, size(tens[ai][1])[1:(end - 1)]...)
            for p in eachindex(tens[ai]), Mf = 1:(2Lf + 1)
                coef = c[colidx[(ai, p, Mf)]]
                coef == 0.0 && continue
                Fo .+= coef .* _mf_slice(tens[ai][p], Mf)
            end
            @inbounds for index in eachindex(Fo)
                abs(Fo[index]) < 1e-10 && (Fo[index] = 0.0)
            end
            norm(Fo) > 1e-10 && push!(terms, (ai, Fo))
        end
        push!(blocks, terms)
    end
    return blocks
end

# Decor-general `_transport_term`: rotate every slot axis, relabel sites through
# the connecting permutation, and return the member term in its canonical slot
# order.
function _transport_term_decors(a::Vector{SiteDecor}, slots::Vector{Slot},
                                F::Array{Float64}, g::Int, perm::Vector{Int},
                                wcache::_WigCache)::SALCTerm
    T = F
    for j in eachindex(slots)
        slots[j].factor.l == 0 && continue                  # D⁰ = 1 (trace axes)
        T = AngularMomentum.nmode_mul(T, _wigner_d(wcache, slots[j].factor.l, g), j)
    end
    a2 = _assignment_image(a, perm)
    slots2 = _assignment_slots(a2)
    σ = _slot_sigma(slots, slots2, perm)
    G = Array{Float64}(permutedims(T, invperm(σ)))
    @inbounds for index in eachindex(G)
        abs(G[index]) < 1e-10 && (G[index] = 0.0)
    end
    return SALCTerm(slots2, G)
end

# Per-site species caps on a decor assignment (spin rank ≤ lmax[species], disp
# degree ≤ pmax[species]). Stabilizer site permutations preserve species, so the
# outcome is a permutation-orbit invariant — checking the canonical
# representative decides the whole orbit, mirroring `_enumerate_ls`'s
# enumeration-with-caps (a capped-out orbit never emits, so `block` indices
# stay aligned between the two engines).
function _admit_assignment(t::Vector{SiteDecor}, species::Vector{Int},
                           lmax::Vector{Int}, pmax::Vector{Int})::Bool
    for s in eachindex(t)
        t[s].spin_l <= lmax[species[s]] || return false
        disp_degree(t[s]) <= pmax[species[s]] || return false
    end
    return true
end

# All SALCs of one cluster orbit for explicitly-given decoration labels
# (sorted `SiteDecor` multisets). The M2b-3 sector spec drives this through the
# public builder; tests drive it directly. `soc = false` keeps only L_S = 0.
# With `lmax_by_species`/`pmax_by_species` given, permutation orbits whose sites
# violate the per-species caps are skipped (both or neither — pass the pair).
function _orbit_salcs_decors(crystal::Crystal, spacegroup::SpaceGroup, N::Int,
                             orbit_id::Int, O::ClusterOrbit,
                             labels::Vector{Vector{SiteDecor}}, soc::Bool,
                             wcache::_WigCache;
                             lmax_by_species::Union{Nothing,Vector{Int}} = nothing,
                             pmax_by_species::Union{Nothing,Vector{Int}} = nothing)::Vector{SALC}
    caps = lmax_by_species === nothing ?
        (pmax_by_species === nothing ? nothing :
         throw(ArgumentError("give lmax_by_species and pmax_by_species together"))) :
        (pmax_by_species === nothing ?
         throw(ArgumentError("give lmax_by_species and pmax_by_species together")) :
         (lmax_by_species, pmax_by_species))
    out = SALC[]
    rep = O.representative
    stab = _stabilizer(crystal, spacegroup, rep)
    perms = unique([perm for (_, perm) in stab])
    conns = _connect_all(crystal, spacegroup, rep, O.members)
    allunique(labels) || throw(ArgumentError(
        "duplicate decoration labels: each label projects to the same SALCs " *
        "twice (exactly collinear design columns)"))
    blockcount = Dict{Tuple{Vector{SiteDecor},Int,Int},Int}()
    for label in labels
        length(label) == N ||
            throw(ArgumentError("decor label has $(length(label)) sites for an " *
                                "$N-body orbit"))
        issorted(label) ||
            throw(ArgumentError("decor label must be sorted (canonical multiset)"))
        iseven(sum(d.spin_l for d in label)) ||
            throw(ArgumentError("Σl_spin must be even (time-reversal screen); " *
                                "got label $label"))
        # Distinct site assignments of the multiset, one canonical representative
        # per orbit of the site-permutation group — in EXACTLY `_enumerate_ls`'s
        # order (review blocker): orbits are discovered in colex (site-1-fastest,
        # i.e. `Iterators.product`) order, the emitted representative is the
        # lex-min of the orbit, and the assignment list is `unique(rep[p])`.
        # Any other order relabels `block` indices / permutes the gauge columns
        # on pure-spin labels, silently breaking key-addressed coefficient
        # re-pairing the moment this engine backs `build_salc_basis`.
        arrangements = sort(_multiset_arrangements(label);
                            by = a -> reverse!([_decortuple(d) for d in a]))
        seen = Set{Vector{SiteDecor}}()
        for tarr in arrangements
            tarr in seen && continue
            orbit = unique([tarr[p] for p in perms])
            for o in orbit
                push!(seen, o)
            end
            t = minimum(orbit)                   # lex-min canonical representative
            caps === nothing ||
                _admit_assignment(t, O.species, caps[1], caps[2]) ||
                continue
            assignments = unique([t[p] for p in perms])
            slotlists = [_assignment_slots(a) for a in assignments]
            cbs = [_decor_coupled_bases(sl) for sl in slotlists]
            blockset = sort(unique((ls, lf) for cbo in cbs for (ls, lf, _) in cbo))
            for (L_S, Lf) in blockset
                soc || is_soc_free(L_S) || continue   # shared with the :soc_free mask
                blocks = _project_and_fold_decors(stab, assignments, slotlists,
                                                  cbs, L_S, Lf, wcache)
                for terms_rep in blocks
                    isempty(terms_rep) && continue
                    members = SALCMember[]
                    for (m, (g, perm)) in zip(O.members, conns)
                        mterms = SALCTerm[]
                        for (ai, F) in terms_rep
                            push!(mterms,
                                  _transport_term_decors(assignments[ai],
                                                         slotlists[ai], F, g,
                                                         perm, wcache))
                        end
                        push!(members, SALCMember(m.atoms, m.shifts, mterms))
                    end
                    members = _canonicalize_members(members)
                    isempty(members) && continue
                    block = get(blockcount, (label, L_S, Lf), 0) + 1
                    blockcount[(label, L_S, Lf)] = block
                    key = SALCKey(N, orbit_id, copy(label), L_S, Lf, block)
                    push!(out, SALC(key, N, key.decors, L_S, Lf, members))
                end
            end
        end
    end
    return out
end

# ---------------------------------------------------------------------------
# sector-driven label generation (M2b-3b)
#
# A resolved `SectorRule` (slce/truncation.jl) describes decoration *content*;
# these generators expand it, per cluster orbit, into the sorted `SiteDecor`
# multiset labels `_orbit_salcs_decors` consumes. Species-resolved per-site
# caps are NOT applied here (a multiset does not know which site gets which
# decor) — generation caps by the loosest species present, and the engine's
# `_admit_assignment` filter decides per permutation orbit.
# ---------------------------------------------------------------------------

# All nondecreasing spin multisets of length `n` with entries in `1:lcap`,
# Σl even, and Σl ≤ lsum_cap.
function _spin_multisets(n::Int, lcap::Int, lsum_cap::Int)::Vector{Vector{Int}}
    n == 0 && return [Int[]]
    budget = lsum_cap == LSUM_UNCAPPED ? n * lcap : lsum_cap
    out = Vector{Int}[]
    ms = Int[]
    function rec(k::Int, lo::Int, rem::Int)
        if k > n
            iseven(sum(ms)) && push!(out, copy(ms))
            return
        end
        for l = lo:min(lcap, rem - (n - k))      # each later site needs l ≥ 1
            push!(ms, l)
            rec(k + 1, l, rem - l)
            pop!(ms)
        end
    end
    rec(1, 1, budget)
    return out
end

# All (k, l) displacement labels of homogeneous degree `2k + l = d` — the exact
# harmonic decomposition of the degree-`d` polynomials (no plethysm needed:
# enumerating the labels IS the Sym^d restriction).
_disp_labels_of_degree(d::Int)::Vector{Tuple{Int,Int}} =
    Tuple{Int,Int}[(k, d - 2k) for k = 0:(d ÷ 2)]

# All nondecreasing multisets of (k, l) displacement factors with total degree
# Σ(2k + l) = p and every factor degree ≤ pcap (factors ordered by (degree, k)).
function _disp_multisets(p::Int, pcap::Int)::Vector{Vector{Tuple{Int,Int}}}
    p == 0 && return [Tuple{Int,Int}[]]
    out = Vector{Tuple{Int,Int}}[]
    ms = Tuple{Int,Int}[]
    degkey(f::Tuple{Int,Int}) = (2 * f[1] + f[2], f[1])
    function rec(rem::Int, lokey::Tuple{Int,Int})
        if rem == 0
            push!(out, copy(ms))
            return
        end
        for d = 1:min(rem, pcap), f in _disp_labels_of_degree(d)
            degkey(f) >= lokey || continue
            push!(ms, f)
            rec(rem - d, degkey(f))
            pop!(ms)
        end
    end
    rec(p, (0, 0))
    return out
end

# All sorted `SiteDecor` multisets realizing spin multiset `S` + displacement
# factor multiset `D` on exactly `N` sites: `length(S) + length(D) - N` sites
# carry one factor of each channel (every site carries ≥ 1 factor). Tiny sizes;
# dedup by a Set over the sorted labels.
function _marry_multisets(S::Vector{Int}, D::Vector{Tuple{Int,Int}},
                          N::Int)::Vector{Vector{SiteDecor}}
    ns, nd = length(S), length(D)
    shared = ns + nd - N
    (0 <= shared <= min(ns, nd)) || return Vector{SiteDecor}[]
    out = Set{Vector{SiteDecor}}()
    for sidx in _index_subsets(ns, shared), didx in _index_subsets(nd, shared)
        drest = setdiff(1:nd, didx)
        for pperm in _permutations_of(didx)
            label = SiteDecor[]
            for (a, b) in zip(sidx, pperm)
                push!(label, SiteDecor(; spin = S[a], disp = D[b]))
            end
            for a in setdiff(1:ns, sidx)
                push!(label, SiteDecor(; spin = S[a]))
            end
            for b in drest
                push!(label, SiteDecor(; disp = D[b]))
            end
            push!(out, sort!(label))
        end
    end
    return sort!(collect(out))
end

# Ordered index subsets (combinations) of 1:n of size k, lexicographic.
function _index_subsets(n::Int, k::Int)::Vector{Vector{Int}}
    k == 0 && return [Int[]]
    out = Vector{Int}[]
    function rec(start::Int, acc::Vector{Int})
        if length(acc) == k
            push!(out, copy(acc))
            return
        end
        for i = start:(n - (k - length(acc)) + 1)
            push!(acc, i)
            rec(i + 1, acc)
            pop!(acc)
        end
    end
    rec(1, Int[])
    return out
end

# All permutations of a small index vector (swap recursion, deduplicated).
function _permutations_of(v::Vector{Int})::Vector{Vector{Int}}
    out = Set{Vector{Int}}()
    p = copy(v)
    function rec(k::Int)
        if k >= length(p)
            push!(out, copy(p))
            return
        end
        for i = k:length(p)
            p[k], p[i] = p[i], p[k]
            rec(k + 1)
            p[k], p[i] = p[i], p[k]
        end
    end
    isempty(p) ? push!(out, Int[]) : rec(1)
    return sort!(collect(out))
end

# Squared edge "gate" distances of one orbit representative, with the species
# pair of each edge — the quantity a sector cutoff is compared against, using
# EXACTLY `candidate_clusters`'s admission semantics: under `MinimumImage` the
# pair is gated by its minimum-image distance (`dmin2`), under `AllImages` by
# each edge's own distance. Symmetry images share edge-length multisets, so the
# representative decides the orbit.
function _orbit_edge_gates(crystal::Crystal, O::ClusterOrbit,
                           cart::Matrix{Float64}, dmin2::Matrix{Float64},
                           use_minimage::Bool)::Vector{Tuple{Int,Int,Float64}}
    rep = O.representative
    N = length(rep.atoms)
    A = crystal.lattice.vectors
    pos(k) = SVector{3,Float64}(cart[1, rep.atoms[k]], cart[2, rep.atoms[k]],
                                cart[3, rep.atoms[k]]) +
             A * SVector{3,Float64}(rep.shifts[k])
    sp = crystal.species
    out = Tuple{Int,Int,Float64}[]
    for x = 1:N, y = (x + 1):N
        ax, ay = rep.atoms[x], rep.atoms[y]
        gate2 = use_minimage ? dmin2[ax, ay] : sum(abs2, pos(y) - pos(x))
        push!(out, (sp[ax], sp[ay], gate2))
    end
    return out
end

# All SALCs of one cluster orbit. Self-contained (its `blockcount` and output are
# orbit-local; `wcache` is read-only), so orbits are processed independently — the
# unit of parallelism in `build_salc_basis`.
function _orbit_salcs(crystal::Crystal, spacegroup::SpaceGroup, N::Int, orbit_id::Int,
                      O::ClusterOrbit, lmax::Vector{Int}, lsumN::Int, scalar_only::Bool,
                      wcache::_WigCache)::Vector{SALC}
    out = SALC[]
    rep = O.representative
    stab = _stabilizer(crystal, spacegroup, rep)
    perms = unique([perm for (_, perm) in stab])
    conns = _connect_all(crystal, spacegroup, rep, O.members)
    # `block` runs across ALL canonical l-tuples sharing one sorted label so the key
    # stays injective: when the site permutations are a proper subgroup of Sₙ they
    # split a degenerate multiset's arrangements into several orbits (e.g. (1,1,2) and
    # (2,1,1) on a mirror-only triangle put l=2 on inequivalent sites), all sharing
    # `ls = sort(t)` but giving distinct SALCs.
    blockcount = Dict{Tuple{Vector{Int},Int},Int}()
    for t in _enumerate_ls(N, O.species, lmax, lsumN, perms)
        orderings = unique([t[p] for p in perms])
        # Build the coupled bases for each ordering once. They are reused across every
        # `Lf`; rebuilding them inside `_project_and_fold` per `Lf` recomputed the whole
        # chained-CG construction `|Lfset|` times. The `Lf` set is the CG decomposition
        # of the `l`-multiset (permutation-invariant), so the union over orderings equals
        # the previous `coupled_bases(t)` set.
        cbs = [coupled_bases(o; scalar_only = scalar_only) for o in orderings]
        Lfset = sort(unique(cb.Lf for cbo in cbs for cb in cbo))
        lab = sort(collect(t))
        for Lf in Lfset
            blocks = _project_and_fold(stab, orderings, cbs, Lf, wcache)
            for terms_rep in blocks
                isempty(terms_rep) && continue
                members = SALCMember[]
                for (m, (g, perm)) in zip(O.members, conns)
                    mterms = SALCTerm[]
                    for (o, F) in terms_rep
                        mls, G = _transport_term(o, F, g, perm, wcache)
                        push!(mterms, SALCTerm(spin_slots(mls), G))
                    end
                    push!(members, SALCMember(m.atoms, m.shifts, mterms))
                end
                # Fold the ordered, anchored images into the canonical duplicate-free
                # form (exact regrouping; empty ⇒ the channel vanished — skip it, as
                # the previous per-term `norm > 1e-10` check did).
                members = _canonicalize_members(members)
                isempty(members) && continue
                block = get(blockcount, (lab, Lf), 0) + 1
                blockcount[(lab, Lf)] = block
                # Pure-spin construction: decors = spin_decors(lab), L_S = Lf
                # (the p = 0 edge of the joint key layout).
                key = SALCKey(N, orbit_id, spin_decors(lab), Lf, Lf, block)
                push!(out, SALC(key, N, key.decors, Lf, Lf, members))
            end
        end
    end
    return out
end

"""
    build_salc_basis(crystal, spacegroup, clusters; lmax_by_species,
                     lsum_by_body = nothing, scalar_only = false) -> SALCBasis

Construct the symmetry-adapted (and time-reversal-even) SLCE basis for every cluster
orbit and body order. For each `(orbit, l-multiset, Lf)` the stabilizer-invariant
coefficient subspace (over orderings × coupling paths) is found, gauge-fixed, and
transported to all orbit members.

# Keyword arguments
- `lmax_by_species::AbstractVector{<:Integer}`: per-species maximum `l`.
- `lsum_by_body`: per-body-order cap on `Σl` over the cluster sites, indexed by
  body order (`nothing` = no cap; entries of `LSUM_UNCAPPED` mean no cap for
  that order).
- `scalar_only::Bool = false`: keep only the scalar `Lf == 0` channel if `true`.
  This is the builder-level spelling of the spec's SOC selection rule, with the
  polarity written into the name: `scalar_only ≡ !`[`BasisSpec`](@ref)`.soc`. The
  retired `isotropy` keyword meant the same thing under the opposite-sounding name,
  which is why it is gone rather than aliased — see the `BasisSpec` deprecation.

# Status
Arbitrary body order. Isotropic (`Lf == 0`) and anisotropic (`Lf > 0`) channels —
including those that mix coupling paths (`N ≥ 3`) or `l`-orderings on
symmetry-equivalent sites (e.g. `l=(1,1,2)`, `Lf>0` on an equilateral triangle) —
are validated by the ground-truth invariance test `Φ(g·e)=Φ(e)` (non-collinear
spins, all `Lf`) and time-reversal evenness `Φ(−e)=Φ(e)`.

The orbits are built in parallel over `Threads.nthreads()` (set `julia -t` /
`JULIA_NUM_THREADS`); each orbit is independent and the result is sorted by
`SALCKey`, so it is byte-for-byte identical at any thread count.
"""
function build_salc_basis(crystal::Crystal, spacegroup::SpaceGroup, clusters::ClusterSet;
                          lmax_by_species::AbstractVector{<:Integer},
                          lsum_by_body::Union{Nothing,AbstractVector{<:Integer}} = nothing,
                          scalar_only::Bool = false)::SALCBasis
    lmax = collect(Int, lmax_by_species)
    maxbody = isempty(clusters.by_body) ? 0 : maximum(keys(clusters.by_body))
    lsum_by_body === nothing || length(lsum_by_body) >= maxbody ||
        throw(ArgumentError("lsum_by_body has $(length(lsum_by_body)) entries; " *
                            "the clusters reach body order $maxbody"))
    lsumN(N) = lsum_by_body === nothing ? LSUM_UNCAPPED : Int(lsum_by_body[N])
    # Precompute the Wigner-D cache serially so it is read-only (race-free) below.
    wcache = _build_wig_cache(spacegroup, isempty(lmax) ? 0 : maximum(lmax))
    # Flatten the orbits into one work list, keeping `(N, orbit_id)` for the SALC keys.
    # Orbits are independent; the final `sort!` by key makes the result identical at any
    # thread count, so the parallel order is irrelevant.
    work = Tuple{Int,Int,ClusterOrbit}[]
    for N in sort(collect(keys(clusters.by_body)))
        for (orbit_id, O) in enumerate(clusters.by_body[N])
            push!(work, (N, orbit_id, O))
        end
    end
    parts = Vector{Vector{SALC}}(undef, length(work))
    Threads.@threads for w in eachindex(work)
        (N, orbit_id, O) = work[w]
        parts[w] = _orbit_salcs(crystal, spacegroup, N, orbit_id, O, lmax, lsumN(N),
                                scalar_only, wcache)
    end
    salcs = isempty(parts) ? SALC[] : reduce(vcat, parts)
    sort!(salcs; by = s -> s.key)
    return SALCBasis(salcs, SALCKey[s.key for s in salcs])
end
