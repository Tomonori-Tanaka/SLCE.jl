# The pointed (site-marked) SALC basis for the adiabatic site-moment expansion
# m_i(e) — the moment channel's counterpart of `SLCEBasis`. Design record:
# _brain_storming/adiabatic-moment-sce (D1–D10 + the D8 addendum, M3-1/M3-2).
#
# The MARK is the displacement decor `SiteDecor(disp = (1, 0))` — the polar l = 0,
# k = 1 radial factor |u|² — evaluated on the synthetic indicator field u = x̂ at the
# marked atom: the factor is 1 on the marked site and 0 elsewhere, so the existing
# decor SALC engine, evaluation kernels, and canonical-member machinery carry the
# pointed basis without modification. The marked site's own ê factor is the decor's
# SPIN part (`SiteDecor(spin = l, disp = (1, 0))`); at evaluation the spin matrix's
# marked COLUMN is substituted by the evaluation axis ê (the D8 addendum's
# marked-column substitution — identity for a mode-4 datum, the constraint axis for
# mode 1), which is exact because every pointed label carries exactly one mark, so a
# member not marked at the row's atom contributes zero before it can read the
# substituted column.
#
# Three mark-aware rules the generic engine cannot express, all confined here:
#   * the marked site's ê factor is allowed for ANY species (an E-inactive species'
#     induced moment is exactly what the channel predicts — a literal per-species
#     l-cap would delete it), while ENVIRONMENT spin factors are allowed only for
#     species the consumer samples (M3-1 decision A; slaving makes the mediated
#     physics expressible through sampled-species clusters);
#   * the star cutoff (`N ≥ 3`) is MARK–ENVIRONMENT-SPOKE based, not all-edge: the
#     star is pinned by its `N−1` spokes (each minimum-image within its radius), and
#     the environment–environment edges are free — an all-edge cut at the same radius
#     keeps 3 of the 15 nn star pairs on FeGe and loses 20–32 % in σ (M2-5);
#   * time reversal: y = ê·M is TR-even, so only even-Σl labels exist (the mark's
#     spin rank counts) — enforced by the engine's existing screen.

# ── MomentSpec ─────────────────────────────────────────────────────────────────────

# The largest body order the pointed enumeration will build. The code is general in
# `N`; this is where the test oracles stop, so raising it means extending them first.
const _MOMENT_NBODY_MAX = 4

"""
    MomentSpec(; lmax_env, sampled, lmax_mark = 2, marked = nothing, nbody = 3,
               cutoff_pair, cutoff_star = nothing, lsum = nothing, soc = false)

The truncation spec of a pointed moment basis ([`MomentBasis`](@ref)). Species are
indexed like `Crystal.species`; `lmax_env` fixes the species count.

- `lmax_env::Vector{Int}` — per-species cap on **environment** spin factors
  (`0` = this species' spins never enter the environment).
- `sampled::Vector{Bool}` — which species the downstream consumer (MC / LLG)
  actually samples. **Required, and load-bearing**: every species with
  `lmax_env > 0` must be sampled (a basis reading unsampled spins cannot be
  evaluated at run time) — refused loudly, never fixed up.
- `lmax_mark::Int` — cap on the marked site's own ê factor (any species; the mark
  rank counts toward the even-Σl time-reversal screen, so odd ranks pair with odd
  environment content).
- `marked::Vector{Bool}` — which species' site moments the basis expands
  (default: every species).
- `nbody` (`N` below) — 1 to 4 (1-body = the per-orbit intercepts μ₀ plus even-rank
  single-site ê invariants; 3 and up are pointed stars). The enumeration is general in
  `N`; the cap is where the test oracles stop, not where the code does. Each body
  order starts at total spin rank `Σl = 2⌈(N−1)/2⌉` — every environment slot needs
  `l ≥ 1` and time reversal keeps only even `Σl` — so the 4-body sector begins at
  `Σl = 4`, and its naive lowest member (a rank-0 mark with three `l = 1`
  environments) is absent: the only `L_S = 0` invariant of three vectors is the
  pseudoscalar triple product, which is TR-odd.
- `cutoff_pair` — mark–environment bond radius (Å) for 2-body clusters: a scalar
  or a symmetric per-species-pair matrix.
- `cutoff_star` — mark–environment bond radius for stars (`nbody ≥ 3`; default:
  `cutoff_pair`). One radius (scalar or symmetric per-species-pair matrix) applies to
  every star order; **a vector gives one radius per star order**, entry `i` for body
  order `i + 2`, and its length must be `nbody - 2`. Per-body radii are what make a
  4-body probe affordable — star members grow as `C(z, N−1)·N!`, so keeping the
  3-body shell wide while cutting the 4-body one to the first shell is the difference
  between thousands of columns and tens. (Per-*spoke* radii are a different thing and
  are NOT expressible: the label is a multiset, so permuting the environment sites
  leaves the same label and "the first spoke is short, the second long" has no
  symmetry-invariant meaning. A total-spoke-length or cluster-diameter cap would be
  permutation invariant; neither is implemented.) Only the `N−1` mark–environment
  spokes are constrained; the environment–environment edges are free. That asymmetry
  with the energy side's compact-cluster rule is deliberate: a star has a distinguished centre, so — **as long
  as every spoke has a unique minimum image** — each environment site is fixed by the
  mark's cell plus its own spoke, and two orbits cannot carry the same monomial. Where
  a spoke has several minimum images (a Wigner–Seitz tie) that uniqueness is exactly
  what fails: the tie-induced member multiplicity grows as `(tie)^(N−1)`, two orbits
  can then carry the same monomial, and `moment_resolvability` is what catches the
  degeneracy.
- `lsum` — optional cap on the total spin rank `Σl` of a label (`nothing` = uncapped).
  A scalar caps every body order; **body-keyed pairs or a `Dict` cap one order each**
  (`lsum = [3 => 6, 4 => 4]`), unnamed orders staying uncapped — the spelling
  [`BasisSpec`](@ref)'s `lsum` uses. Per-order caps are not a convenience: a body
  order starts at `Σl = 2⌈(N−1)/2⌉`, so one global cap starves the high orders. At
  `lsum = 4` the 2- and 3-body sectors get two even levels (`Σl = 2, 4`) while the
  4-body sector gets one, and the same number that opens the 4-body sector is what
  cuts the 3-body sector's `Σl = 6` content away. Unlike `cutoff_star`, the entries
  are indexed by the body order itself (`lsum[N]`, `N` from 1), and a positional
  vector is refused: this table is **partial** — an order it does not name stays
  uncapped — which a positional list cannot express.
- `soc` — keep `L_S ≠ 0` blocks (default `false`: the adiabatic map is treated as
  spin-rotation covariant, exactly like a `soc = false` energy basis).
"""
struct MomentSpec
    nbody::Int
    lmax_mark::Int
    lmax_env::Vector{Int}
    sampled::Vector{Bool}
    marked::Vector{Bool}
    cutoff_pair::Matrix{Float64}
    # One entry per STAR order: `cutoff_star[N - 2]` is body order `N`, so the vector
    # is empty for `nbody < 3`. Use `_star_cutoff(spec, N)` rather than indexing here.
    cutoff_star::Vector{Matrix{Float64}}
    # One entry per body order, indexed by the order ITSELF: `lsum[N]`, `N` from 1, so
    # the vector has length `nbody`. `LSUM_UNCAPPED` means no cap. The offset differs
    # from `cutoff_star[N - 2]` on purpose — that one exists only for `N ≥ 3`, this one
    # bites from `N = 1`. Same layout as `BasisSpec.lsum`. Read it through
    # `_label_lsum`, never by raw index.
    lsum::Vector{Int}
    soc::Bool

    # The two per-order vectors carry lengths nothing else can restore, and the
    # exactly-field-typed positional call cannot be routed through the keyword
    # constructor — so the invariant is asserted HERE, where every path arrives.
    function MomentSpec(nbody, lmax_mark, lmax_env, sampled, marked, cutoff_pair,
                        cutoff_star, lsum, soc)
        length(lsum) == nbody || throw(ArgumentError(
            "lsum has $(length(lsum)) entries for nbody = $nbody (one per body order)"))
        length(cutoff_star) == max(nbody - 2, 0) || throw(ArgumentError(
            "cutoff_star has $(length(cutoff_star)) entries for nbody = $nbody " *
            "(one per STAR order, so $(max(nbody - 2, 0)))"))
        return new(nbody, lmax_mark, lmax_env, sampled, marked, cutoff_pair,
                   cutoff_star, lsum, soc)
    end
end

function MomentSpec(; lmax_env::AbstractVector{<:Integer},
                    sampled::AbstractVector{Bool},
                    lmax_mark::Integer = 2,
                    marked::Union{Nothing,AbstractVector{Bool}} = nothing,
                    nbody::Integer = 3,
                    cutoff_pair::Union{Real,AbstractMatrix{<:Real}},
                    cutoff_star::Union{Nothing,Real,AbstractMatrix{<:Real},
                                       AbstractVector,AbstractDict} = nothing,
                    lsum::Union{Nothing,Integer,AbstractVector,AbstractDict} = nothing,
                    soc::Bool = false)::MomentSpec
    nkd = length(lmax_env)
    nkd >= 1 || throw(ArgumentError("lmax_env must name at least one species"))
    all(l -> l >= 0, lmax_env) ||
        throw(ArgumentError("lmax_env entries must be ≥ 0; got $lmax_env"))
    lmax_mark >= 0 || throw(ArgumentError("lmax_mark must be ≥ 0; got $lmax_mark"))
    1 <= nbody <= _MOMENT_NBODY_MAX || throw(ArgumentError(
        "nbody must be in 1:$_MOMENT_NBODY_MAX; got $nbody. The enumeration and the " *
        "SALC projection are written for general N, but only N ≤ $_MOMENT_NBODY_MAX " *
        "is covered by the test oracles — raising the cap without extending them " *
        "would promise an unverified region. The oracles that set it are the " *
        "pointed-star brute force in test/unit/test_ws_nbody.jl and the absolute " *
        "normalization gate in test/unit/test_momentbasis.jl."))
    length(sampled) == nkd ||
        throw(ArgumentError("sampled has $(length(sampled)) entries for $nkd species"))
    for s = 1:nkd
        lmax_env[s] > 0 && !sampled[s] &&
            throw(ArgumentError("species $s carries environment spin factors " *
                                "(lmax_env = $(lmax_env[s])) but is not sampled by " *
                                "the consumer — a basis reading unsampled spins " *
                                "cannot be evaluated at run time. Either sample the " *
                                "species or set its lmax_env to 0 (M3-1: induced " *
                                "physics is expressible through sampled-species " *
                                "clusters)"))
    end
    mk = marked === nothing ? fill(true, nkd) : collect(Bool, marked)
    length(mk) == nkd ||
        throw(ArgumentError("marked has $(length(mk)) entries for $nkd species"))
    any(mk) || throw(ArgumentError("no species is marked — nothing to expand"))
    _cut(c) = c isa Real ? fill(Float64(c), nkd, nkd) : Matrix{Float64}(c)
    cp = _cut(cutoff_pair)
    # `cutoff_star` is stored per star order (`nbody - 2` entries, body `N` at `N - 2`).
    # Four spellings, all resolved here: a scalar or a matrix means "the same radius at
    # every star order"; a body-keyed collection (`[3 => 4.1, 4 => 2.5]`, or a `Dict`)
    # names the order it applies to — the same shape the energy side's `cutoff` takes
    # and the same shape the TOML reader emits; and a plain vector is positional. The
    # keyed form exists because the positional one has no way to catch a transposed
    # pair: `[2.5, 4.1]` is a perfectly valid spec that builds the WIDE sector at the
    # NARROW order, which is the memory blow-up per-order radii were added to avoid.
    # Below body order 3 there is no star, so the vector is empty and an explicit
    # `cutoff_star` is refused rather than silently stored and never read.
    nstar = max(Int(nbody) - 2, 0)
    _star_entry(c, what) = c isa Real || c isa AbstractMatrix{<:Real} ? _cut(c) :
        throw(ArgumentError("$what must be a number or a species-pair matrix; " *
                            "got $(typeof(c))"))
    # An explicit `cutoff_star` below body order 3 is refused whatever its shape — an
    # empty collection included. There is no star to cut, so the value would be stored
    # and never read.
    cutoff_star === nothing || nstar > 0 || throw(ArgumentError(
        "cutoff_star was given but nbody = $nbody has no star order (stars start at " *
        "body order 3), so the radius would never be read — drop it or raise nbody"))
    want = collect(3:Int(nbody))
    cs = if cutoff_star === nothing
        Matrix{Float64}[copy(cp) for _ = 1:nstar]
    elseif cutoff_star isa AbstractDict ||
           (cutoff_star isa AbstractVector && !isempty(cutoff_star) &&
            all(x -> x isa Pair, cutoff_star))
        kv = collect(cutoff_star)
        keys_ = sort([Int(first(e)) for e in kv])
        keys_ == want || throw(ArgumentError(
            "cutoff_star's body-order keys must be exactly $want for nbody = $nbody; " *
            "got $keys_. Stars start at body order 3, and every order needs its own " *
            "entry — a scalar or a species-pair matrix applies to all of them"))
        byN = Dict(Int(first(e)) => last(e) for e in kv)
        Matrix{Float64}[_star_entry(byN[N], "cutoff_star[$N]") for N in want]
    elseif cutoff_star isa AbstractVector
        length(cutoff_star) == nstar || throw(ArgumentError(
            "cutoff_star has $(length(cutoff_star)) entries but nbody = $nbody needs " *
            "$nstar (one per star order, entry i for body order i + 2). The " *
            "alternatives are a scalar, a species-pair matrix (both apply to every " *
            "star order), or the body-keyed form [3 => 4.1, 4 => 2.5]"))
        Matrix{Float64}[_star_entry(c, "cutoff_star[$i]")
                        for (i, c) in enumerate(cutoff_star)]
    else
        Matrix{Float64}[_cut(cutoff_star) for _ = 1:nstar]
    end
    named = Tuple{String,Matrix{Float64}}[("cutoff_pair", cp)]
    for (i, m) in enumerate(cs)
        push!(named, ("cutoff_star[$i] (body order $(i + 2))", m))
    end
    for (name, m) in named
        size(m) == (nkd, nkd) ||
            throw(ArgumentError("$name is $(size(m)) for $nkd species"))
        m == m' || throw(ArgumentError("$name must be symmetric"))
        all(v -> !isnan(v) && v >= 0, m) ||
            throw(ArgumentError("$name entries must be ≥ 0 Å"))
    end
    # `cutoff_star` accepts a POSITIONAL vector and `lsum` does not, in the same
    # constructor call — so say why here rather than let the shared resolver answer
    # with a generic "unsupported form".
    lsum isa AbstractVector && !isempty(lsum) && !all(x -> x isa Pair, lsum) &&
        throw(ArgumentError(
            "lsum: a positional vector is not accepted (unlike cutoff_star, whose " *
            "entries are offset by 2). This table is indexed by the body order " *
            "itself AND is partial — an order it does not name stays uncapped, " *
            "which a positional list cannot express. Write body-keyed pairs: " *
            "lsum = [3 => 6, 4 => 4]"))
    # Same resolver as the energy side, so the two `lsum` keywords accept exactly the
    # same spellings and report the same errors. A scalar broadcasts to every order.
    ls = _resolve_lsum(lsum, Int(nbody))
    return MomentSpec(Int(nbody), Int(lmax_mark), collect(Int, lmax_env),
                      collect(Bool, sampled), mk, cp, cs, ls, soc)
end

# The `Σl` cap of body order `N`. One accessor so the per-order layout of `spec.lsum`
# is named in exactly one place — and so nobody writes `spec.lsum[N - 2]` by analogy
# with `cutoff_star`, which would silently read a DIFFERENT order's cap rather than
# fail. The two vectors in this struct genuinely use different offsets; only these two
# accessors know that.
function _label_lsum(spec::MomentSpec, N::Int)::Int
    1 <= N <= length(spec.lsum) || throw(ArgumentError(
        "body order $N is outside the spec's 1:$(spec.nbody)"))
    return spec.lsum[N]
end

# The star radius of body order `N` (`N ≥ 3`). One accessor so the per-order layout of
# `spec.cutoff_star` is named in exactly one place.
function _star_cutoff(spec::MomentSpec, N::Int)::Matrix{Float64}
    1 <= N - 2 <= length(spec.cutoff_star) || throw(ArgumentError(
        "body order $N is outside the spec's star orders 3:$(spec.nbody)"))
    return spec.cutoff_star[N - 2]
end

# The elementwise envelope of every star radius — the radius the ONE shared neighbor
# list must reach. Sharing it changes nothing, and the reason is not "each order
# re-filters" alone (the `dmin2_star` test lives INSIDE `admit`, so a wider list does
# feed that test a possibly different matrix). It is this: `admit` only ever consults
# `dmin2_star[a, b]` for a pair whose edge already passed `edges ≤ r_N·fac`, and every
# image of that pair is at least the pair's true minimum distance — so the true minimum
# is itself ≤ `r_N·fac` and was therefore already in the NARROW list too. On every entry
# the conjunction reads, the envelope-built matrix and the per-order one agree. Where
# they differ (a pair with no image inside `r_N`) the edge test has already rejected the
# star. The minimum-image search box is cutoff-independent, so the entries themselves
# are exact distances, not radius-truncated ones.
function _star_cutoff_envelope(spec::MomentSpec)::Matrix{Float64}
    isempty(spec.cutoff_star) && return zeros(0, 0)
    out = copy(spec.cutoff_star[1])
    for m in spec.cutoff_star
        out .= max.(out, m)
    end
    return out
end

_mark_decor(l::Int)::SiteDecor =
    l == 0 ? SiteDecor(disp = (1, 0)) : SiteDecor(spin = l, disp = (1, 0))

is_marked(d::SiteDecor)::Bool = has_disp(d)     # in a pointed label the disp factor IS the mark

# All sorted decor multisets for body order N: exactly one marked slot (spin rank
# 0…lmax_mark), every environment slot a pure spin factor (rank ≥ 1), total spin
# rank even (TR) and ≤ THIS body order's `lsum` entry. Which SITE may carry which decor is the per-assignment
# `admit` closure's business, not the label's.
function _moment_labels(spec::MomentSpec, N::Int)::Vector{Vector{SiteDecor}}
    N >= 1 || return Vector{SiteDecor}[]
    lsum_N = _label_lsum(spec, N)   # also the range check: `spec.lsum` is indexed by N
    lem = maximum(spec.lmax_env; init = 0)
    # Non-decreasing environment multisets, so each multiset is enumerated once; which
    # SITE carries which decor is the per-assignment `admit` closure's business.
    envs = Vector{Vector{Int}}()
    function grow!(cur::Vector{Int}, lo::Int)
        if length(cur) == N - 1
            push!(envs, copy(cur))
            return
        end
        for l = lo:lem
            push!(cur, l)
            grow!(cur, l)
            pop!(cur)
        end
    end
    grow!(Int[], 1)
    labs = Vector{Vector{SiteDecor}}()
    for lm = 0:spec.lmax_mark, e in envs
        t = lm + sum(e; init = 0)
        (iseven(t) && t <= lsum_N) || continue
        push!(labs, sort(vcat([_mark_decor(lm)],
                              [SiteDecor(spin = l) for l in e])))
    end
    sort!(labs)
    unique!(labs)
    return labs
end

# ── pointed star candidates (N ≥ 3) ────────────────────────────────────────────────

# Star clusters {mark, env₁, …, env_{N−1}}: every mark–environment spoke is a
# minimum-image neighbor pair within the mark–env star radius for its species pair; the
# environment–environment edges are FREE (the star is pinned by its `N−1` spokes, so
# no periodic alias hides there — M2-5; what that rests on is a spoke's minimum image
# being unique, which a Wigner–Seitz tie breaks). The candidate set is closed under
# the space group by construction (species and minimum-image distances are symmetry
# invariants), which `_orbits_from_members`' closure assertion re-checks.
#
# MULTIPLICITY CONVENTION: `candidate_clusters` lists every physical instance once
# per SITE ORDERING (`N!` anchored variants at `N` distinct sites — 6 for a 3-body,
# 24 for a 4-body) — the space the SALC projection needs, and the multiplicity
# `_canonicalize_members` folds into the member weights. A pointed candidate set with
# fewer orderings per instance yields SALCs scaled down by the missing factor
# (measured: 3 of 6 orderings halved the closed-star column against the prototype's
# 6.0 geometric oracle), silently breaking cross-orbit coefficient comparability. So
# the enumeration here first finds each star once (per translation class), then expands
# EVERY class to all `N!` re-anchored orderings — identical members to what
# `candidate_clusters` would emit for the clusters it also admits.
function _pointed_star_candidates(crystal::Crystal, nl::NeighborList,
                                  spec::MomentSpec, N::Int)::Vector{ClusterMember}
    N >= 3 || throw(ArgumentError("pointed stars start at body order 3; got $N"))
    nat = n_atoms(crystal)
    sp = crystal.species
    fac = 1.0 + nl.tol
    cut = _star_cutoff(spec, N)
    nbrs = [Tuple{Int,SVector{3,Int}}[] for _ = 1:nat]
    for p in nl.pairs
        spec.marked[sp[p.i]] || continue
        p.distance <= cut[sp[p.i], sp[p.j]] * fac || continue
        push!(nbrs[p.i], (p.j, p.shift))
    end
    z = SVector{3,Int}(0, 0, 0)
    classes = Dict{Any,ClusterMember}()
    sites = Vector{Tuple{Int,SVector{3,Int}}}(undef, N - 1)
    for i = 1:nat
        ns = nbrs[i]
        length(ns) >= N - 1 || continue
        for combo in _combinations(length(ns), N - 1)
            for (k, c) in enumerate(combo)
                sites[k] = ns[c]
            end
            # Only an exact (atom, shift) repeat is skipped. Two DIFFERENT minimum
            # images of one neighbor (same atom, different shift — a small cell's
            # tie) are kept: under plain PBC every environment factor then reads the
            # one spin and the member reduces to a lower-body function;
            # `moment_resolvability` refuses such a basis as `UnclassifiableBasis`,
            # and `MomentDataset` runs that gate at its door — a hard refusal, never
            # a silent overcount. (The energy side's `candidate_clusters` drops these
            # members instead; the pointed enumeration does not, deliberately.)
            #
            # The check is defensive: a minimum-image neighbor list emits each
            # `(i, j, R)` once, so `ns` holds no duplicate and no subset of distinct
            # INDICES can repeat a site. Ascending indices do not imply that on their
            # own — with `ns = [A, B, A]` the repeat would be non-adjacent — so the
            # test is over all pairs rather than adjacent ones.
            allunique(sites) || continue
            m = ClusterMember(vcat([i], [site[1] for site in sites]),
                              vcat([z], [site[2] for site in sites]))
            get!(classes, _member_sig(m), m)
        end
    end
    out = ClusterMember[]
    perms = _permutations(N)
    for sig in sort!(collect(keys(classes)))     # deterministic emission order
        m = classes[sig]
        for p in perms
            s1 = m.shifts[p[1]]
            push!(out, ClusterMember([m.atoms[q] for q in p],
                                     [m.shifts[q] - s1 for q in p]))
        end
    end
    return out
end

# ── MomentBasis ────────────────────────────────────────────────────────────────────

"""
    MomentBasis(crystal, spec; backend = NoSymmetry(), tol = 1e-5,
                tie_tol = same-distance default)

The pointed SALC basis of the adiabatic site-moment expansion `m_i(e)`: per marked
reference-cell atom `i`, the design row is the SALC vector evaluated with the mark
on `i` (`X[(config, i), φ] = Φ_φ(i; e, ê_i)`), and one shared coefficient vector
serves every symmetry-equivalent site. Built on the ordinary decor SALC engine —
full space group, orbits split under the site group through the SALC projection —
with the mark-aware admission rules of the design record (see the file header).

Fields: `crystal`, `spacegroup`, `spec`, `salc_basis` (keys sorted, addressable),
`records` (one NamedTuple per SALC: representative geometry for reporting —
`body`, `species`, `edges`, `nmem`), and `marked_atoms` (the design's row atoms).
`tie_tol` records the boundary-tie band the basis was built with (the local-field
diagnostics read it back so their neighbor enumeration matches the basis's).
The trailing `resolvability` field is [`moment_resolvability`](@ref)'s default-`rtol`
cache (a `Ref`, `nothing` until the first call) — derived data, never set by hand.
"""
struct MomentBasis
    crystal::Crystal
    spacegroup::SpaceGroup
    spec::MomentSpec
    salc_basis::SALCBasis
    records::Vector{NamedTuple}
    marked_atoms::Vector{Int}
    tie_tol::Float64
    resolvability::Base.RefValue{Union{Nothing,NamedTuple}}
end

n_salcs(mb::MomentBasis)::Int = length(mb.salc_basis.salcs)
salcs(mb::MomentBasis)::Vector{SALC} = mb.salc_basis.salcs

function Base.show(io::IO, mb::MomentBasis)
    # The REALIZED maximum body order, not the requested cap. A truncation that cannot
    # reach a sector's `Σl` floor drops it silently (`_moment_labels` returns nothing
    # for it), and printing the request would make the object assert columns it does
    # not have. The request is shown alongside when the two differ.
    got = maximum(k.body for k in mb.salc_basis.keys)
    print(io, "MomentBasis(", length(mb.salc_basis.salcs), " SALCs, ",
          length(mb.marked_atoms), " marked atoms, nbody = ", got,
          got == mb.spec.nbody ? "" : " of $(mb.spec.nbody) requested", ")")
end

function MomentBasis(crystal::Crystal, spec::MomentSpec;
                     backend::AbstractSymmetryBackend = NoSymmetry(),
                     tol::Real = 1e-5,
                     tie_tol::Real = _SAME_DIST_RTOL)::MomentBasis
    nkd = length(crystal.species_labels)
    length(spec.lmax_env) == nkd ||
        throw(ArgumentError("spec names $(length(spec.lmax_env)) species, crystal " *
                            "has $nkd"))
    (isfinite(tie_tol) && 0 <= tie_tol < _TIE_TOL_MAX) ||
        throw(ArgumentError("tie_tol must be in [0, $( _TIE_TOL_MAX)); got $tie_tol"))
    sg = analyze_symmetry(backend, crystal; tol = tol)
    sp = crystal.species
    nat = n_atoms(crystal)
    marked_atoms = [a for a = 1:nat if spec.marked[sp[a]]]
    isempty(marked_atoms) &&
        throw(ArgumentError("no atom of a marked species in the reference cell"))

    # clusters: bodies 1–2 from the ordinary enumeration at the pair radii (the
    # single edge IS the mark–env bond), bodies 3 and up from the pointed star
    # enumeration. The two are NOT merged: the candidate sources differ and so do
    # their multiplicity conventions.
    nl2 = build_neighbor_list(crystal, spec.cutoff_pair, MinimumImage();
                              tol = tie_tol)
    cs = build_clusters(crystal, nl2, sg; nbody = min(spec.nbody, 2))
    orbits = Tuple{Int,Int,ClusterOrbit}[]           # (body, per-body id, orbit)
    for body = 1:min(spec.nbody, 2)
        for (k, O) in enumerate(get(cs.by_body, body, ClusterOrbit[]))
            push!(orbits, (body, k, O))
        end
    end
    dmin2_star = Matrix{Float64}(undef, 0, 0)
    if spec.nbody >= 3
        # ONE neighbor list for every star order, built at the elementwise envelope of
        # the per-order radii; each order then re-filters on its own radius inside
        # `_pointed_star_candidates` and `admit`. Only the mark–environment spokes are
        # cut on it whatever N is.
        nl_star = build_neighbor_list(crystal, _star_cutoff_envelope(spec), MinimumImage();
                                      tol = tie_tol)
        for N = 3:spec.nbody
            stars = _pointed_star_candidates(crystal, nl_star, spec, N)
            for (k, O) in enumerate(_orbits_from_members(crystal, sg, stars, N))
                push!(orbits, (N, k, O))
            end
        end
        dmin2_star = _dmin2_matrix(nl_star, nat)
    end

    maxl = max(spec.lmax_mark, maximum(spec.lmax_env; init = 0))
    wcache = _build_wig_cache(sg, maxl)
    cart = cartesian_positions(crystal)
    A = Matrix(crystal.lattice.vectors)
    fac = 1.0 + tie_tol

    # One label list per body order, not per orbit: it is a pure function of the spec
    # and the recursion behind it is no longer the old straight-line branch.
    labels_by_body = Dict(b => _moment_labels(spec, b) for b in 1:spec.nbody)

    # Threaded over orbits, exactly as `build_salc_basis` threads the same work. Each
    # task owns one orbit and writes only its own slot, `wcache` is built above and
    # read-only, and the output is sorted by key below — so the result is identical at
    # any thread count and any schedule.
    #
    # `:greedy` because per-orbit cost spans orders of magnitude AND the expensive
    # orbits are all at the end: `orbits` is built in ascending body order, and the
    # projection's `eigen` grows with the carrier dimension. The default schedule (and
    # `:dynamic`, which differs only in thread affinity) cuts the range into one
    # CONTIGUOUS chunk per thread, so every high-body orbit lands in the last chunk and
    # runs serially while the rest idle. `:greedy` hands out iterations one at a time.
    parts = Vector{Vector{SALC}}(undef, length(orbits))
    rec_parts = Vector{Vector{NamedTuple}}(undef, length(orbits))
    Threads.@threads :greedy for w in eachindex(orbits)
        body, oid, O = orbits[w]
        labels = labels_by_body[body]
        if isempty(labels)
            parts[w] = SALC[]
            rec_parts[w] = NamedTuple[]
            continue
        end
        rep = O.representative
        # representative-site cartesian positions (image shifts included) → edges
        pos = [SVector{3,Float64}(cart[:, rep.atoms[s]]) +
               SVector{3,Float64}(A * Float64.(rep.shifts[s])) for s = 1:body]
        edges = [norm(pos[s] - pos[t]) for s = 1:body, t = 1:body]
        star_cut = body >= 3 ? _star_cutoff(spec, body) : zeros(0, 0)
        admit = function (t::Vector{SiteDecor})
            for s in eachindex(t)
                d = t[s]
                if is_marked(d)
                    # the mark: any marked species, its own ê-rank cap, and — for
                    # stars — every mark–env spoke minimum-image within the radius
                    spec.marked[O.species[s]] || return false
                    d.spin_l <= spec.lmax_mark || return false
                    if body >= 3
                        for u in eachindex(t)
                            u == s && continue
                            r = star_cut[O.species[s], O.species[u]]
                            edges[s, u] <= r * fac || return false
                            # Minimum-image spoke test. Note it is VACUOUS on the
                            # diagonal: `_dmin2_matrix` starts at `Inf` and the list
                            # drops `i == j`, so a spoke whose environment sits on an
                            # image of the mark's own reference-cell atom passes here
                            # on the raw radius alone. The refusal for that case is the
                            # `allunique` guard in the classifier, and it is the only
                            # one — see the comment there before changing either.
                            edges[s, u]^2 <=
                                dmin2_star[rep.atoms[s], rep.atoms[u]] * fac^2 ||
                                return false
                        end
                    end
                else
                    # environment spins: sampled species only, per-species cap
                    d.spin_l >= 1 || return false
                    d.spin_l <= spec.lmax_env[O.species[s]] || return false
                end
            end
            return true
        end
        got = _orbit_salcs_decors(crystal, sg, body, oid, O, labels, spec.soc,
                                  wcache; admit = admit)
        edgetuple = Tuple(sort([round(edges[a, b]; digits = 6)
                                for a = 1:body for b = (a + 1):body]))
        parts[w] = got
        rec_parts[w] = [(; body = body, species = Tuple(O.species),
                           edges = edgetuple, nmem = length(O.members)) for _ in got]
    end
    out = reduce(vcat, parts; init = SALC[])
    recs = reduce(vcat, rec_parts; init = NamedTuple[])

    isempty(out) &&
        throw(ArgumentError("the moment basis is empty: no pointed SALC survives " *
                            "the spec — check the cutoffs, lmax_mark / lmax_env, " *
                            "and that a marked species has admissible neighbors"))
    # A requested body order that contributes NOTHING is a silent truncation: the user
    # asked for a sector and got a basis without it. The usual cause is the label
    # screen rather than the geometry: an N-body sector starts at `Σl = 2⌈(N−1)/2⌉`,
    # so `lsum` or the `lmax` caps can put it out of reach while every cutoff is
    # generous. Say which, and say what would fix it.
    let have = Set(s.key.body for s in out)
        for body = 1:spec.nbody
            body in have && continue
            floor_l = 2 * cld(body - 1, 2)
            @warn "moment basis: body order $body contributes no SALC — the basis " *
                  "does not carry that sector" *
                  (isempty(labels_by_body[body]) ?
                   ". No label survives the screens: this sector starts at " *
                   "Σl = $floor_l (every environment slot needs l ≥ 1 and time " *
                   "reversal keeps only even Σl), so raise " *
                   (_label_lsum(spec, body) == LSUM_UNCAPPED ? "" :
                    "lsum[$body] (the cap of THIS body order, " *
                    "$(_label_lsum(spec, body))) / ") *
                   "lmax_mark / lmax_env to reach it" :
                   ". Labels exist, so no cluster orbit admits them: check " *
                   (body >= 3 ?
                    "cutoff_star for body order $body (currently " *
                    "$(_star_cutoff(spec, body)) Å)" : "cutoff_pair") *
                   " and the marked and sampled species")
        end
    end
    perm = sortperm(out; by = s -> s.key)
    keyvec = [out[j].key for j in perm]
    allunique(keyvec) || error("duplicate pointed SALC keys — enumeration bug")
    return MomentBasis(crystal, sg, spec, SALCBasis(out[perm], keyvec), recs[perm],
                       marked_atoms, Float64(tie_tol),
                       Ref{Union{Nothing,NamedTuple}}(nothing))
end

# ── evaluation (marked-column substitution) ────────────────────────────────────────

# One design entry: Φ_φ(a; e, ê) with the spin matrix's marked column substituted by
# the evaluation axis and the mark realized by the indicator field u = x̂ at `a`
# (the mark factor |u|²R₀₀ is direction-independent). Exact, not approximate: every
# pointed label carries exactly one mark, so a member not marked at `a` is killed by
# its |u_b|² = 0 factor before any substituted value could reach it.
function _eval_moment_entry(salc::SALC, esub::Matrix{Float64}, u::Matrix{Float64},
                            scratch::SALCScratch)::Float64
    return evaluate_salc(salc, esub, u, scratch)
end

"""
    _design_moment(mb, configs, axes; member_index = true) -> Matrix{Float64}

The moment channel's design matrix: rows are (configuration-major, marked-atom
minor) pairs `(c, a)` over `mb.marked_atoms`, columns the pointed SALCs. `axes[c]`
is the per-atom evaluation-axis matrix of configuration `c` (the D8 addendum's
mode rule resolves it at the dataset layer: mode 4 → the datum's `directions`,
mode 1 → its `constraint_axes`); only the marked column of `e` is substituted per
row, environment columns stay configuration coordinates in both modes.
`member_index = true` (default) evaluates each row through the mark→term index —
value-identical to the full per-SALC evaluation (`member_index = false`, the
in-tree oracle path); see `_mark_term_index`. `index` supplies that index instead
of rebuilding it, for a caller that assembles the design in chunks.
"""
function _design_moment(mb::MomentBasis, configs::Vector{Matrix{Float64}},
                        axes::Vector{Matrix{Float64}};
                        member_index::Bool = true,
                        index::Union{Nothing,Vector{Vector{Vector{Tuple{Int,Int}}}}} =
                            nothing)::Matrix{Float64}
    length(configs) == length(axes) ||
        throw(ArgumentError("$(length(configs)) configs, $(length(axes)) axes"))
    sal = salcs(mb)
    atoms = mb.marked_atoms
    nat = n_atoms(mb.crystal)
    nrow = length(configs) * length(atoms)
    X = Matrix{Float64}(undef, nrow, length(sal))
    # `index` lets a caller that builds the design in chunks pay for the symbolic
    # mark→term walk once instead of once per chunk; it is a pure function of the
    # basis, so a supplied index is value-identical to a rebuilt one.
    idx = member_index ? (index === nothing ? _mark_term_index(sal, atoms) : index) :
          nothing
    # Shape checks once, serially: a throw from inside the threaded loop surfaces
    # as a TaskFailedException wrapping the ArgumentError.
    for (ci, e) in enumerate(configs)
        size(e) == (3, nat) ||
            throw(ArgumentError("config $ci is $(size(e)), expected (3, $nat)"))
        size(axes[ci]) == (3, nat) ||
            throw(ArgumentError("axes $ci is $(size(axes[ci])), expected (3, $nat)"))
    end
    # `:greedy`: a 1-body intercept column and a 4-body star column differ by orders of
    # magnitude in cost, and the columns are emitted in orbit order, so the heavy ones
    # sit together at the end — the default (one contiguous chunk per thread) would put
    # them all in one task. Each `j` writes only column `j`, so no value depends on the
    # schedule.
    Threads.@threads :greedy for j = 1:length(sal)
        scratch = SALCScratch()
        esub = Matrix{Float64}(undef, 3, nat)
        u = zeros(3, nat)
        for (ci, e) in enumerate(configs)
            for (ai, a) in enumerate(atoms)
                copyto!(esub, e)
                esub[1, a] = axes[ci][1, a]
                esub[2, a] = axes[ci][2, a]
                esub[3, a] = axes[ci][3, a]
                u[1, a] = 1.0
                X[(ci - 1) * length(atoms) + ai, j] = idx === nothing ?
                    _eval_moment_entry(sal[j], esub, u, scratch) :
                    _eval_moment_members(sal[j], idx[j][ai], esub, u, scratch)
                u[1, a] = 0.0
            end
        end
    end
    return X
end

# ── fast design path: mark → term index ────────────────────────────────────────────

# For each pointed SALC and each marked atom, the (member, term) pairs whose mark
# sits on that atom: the design row (c, a) reads exactly these — every other term
# dies on its |u|² = 0 factor before contributing. The granularity is the TERM,
# not the member: a member's terms enumerate distinct decor→site assignments, so
# one member can carry its mark on different sites in different terms. Skipping
# the dead terms removes only exact-zero additions from each accumulated sum
# (live terms keep their relative order), so the fast path is VALUE-IDENTICAL to
# the full evaluation up to the sign of a zero; the equality gates in
# test_momentbasis.jl/test_momentfit.jl hold both paths to elementwise ==.
function _mark_term_index(sal::Vector{SALC},
                          atoms::Vector{Int})::Vector{Vector{Vector{Tuple{Int,Int}}}}
    pos = Dict(a => i for (i, a) in enumerate(atoms))
    idx = [[Tuple{Int,Int}[] for _ in atoms] for _ in sal]
    for (j, s) in enumerate(sal), (mi, m) in enumerate(s.members),
        (ti, t) in enumerate(m.terms)

        # One mark per pointed label (the marked-column-substitution invariant):
        # the FIRST DISP slot is therefore THE mark, the same slot every other
        # consumer (e.g. moment_resolvability) sees. The exact-zero skip argument
        # additionally needs the mark's radial power k ≥ 1 (0.0^0 == 1.0 would
        # make skipped terms nonzero) — pinned loudly, not assumed.
        site = 0
        for sl in t.slots
            if sl.factor.channel == DISP
                sl.factor.k >= 1 ||
                    error("pointed SALC $j member $mi term $ti: mark radial " *
                          "power k = $(sl.factor.k) < 1 breaks the dead-term skip")
                site = sl.site
                break
            end
        end
        site == 0 && error("pointed SALC $j member $mi term $ti carries no mark")
        a = m.atoms[site]
        # A mark on an atom outside `atoms` contributes to no design row in either
        # path (no row ever raises its |u|²); it is simply absent from the index.
        haskey(pos, a) && push!(idx[j][pos[a]], (mi, ti))
    end
    return idx
end

# Subset-sum evaluation of a pointed SALC over the terms marked at the row's
# atom — same scale and same relative accumulation order as
# `evaluate_salc(salc, e, u, scratch)` restricted to the live terms.
function _eval_moment_members(salc::SALC, tids::Vector{Tuple{Int,Int}},
                              esub::Matrix{Float64}, u::Matrix{Float64},
                              scratch::SALCScratch)::Float64
    n_spin = count(has_spin, salc.decors)
    scale = (4π)^(n_spin / 2)
    total = 0.0
    @inbounds for (mi, ti) in tids
        m = salc.members[mi]
        t = m.terms[ti]
        total += _eval_term_mixed(t.folded, t.slots, m.atoms, esub, u, scratch)
    end
    return scale * total
end

# ── pointed resolvability gate ─────────────────────────────────────────────────────

# Row identity of the pointed signature expansion: the marked reference-cell atom,
# the mark factor's (l, μ) in the INDEPENDENT axis variable ê_a, and the sorted
# environment factors (reference-cell atom, l, μ) in the configuration variable e.
# Periodic images fold onto reference-cell atoms — exactly the folding where a
# Wigner–Seitz tie's aliasing becomes visible.
struct _MomentRowKey
    mark_atom::Int
    mark_lm::Tuple{Int,Int}
    env::Vector{Tuple{Int,Int,Int}}
end
Base.hash(k::_MomentRowKey, h::UInt) =
    hash(k.env, hash(k.mark_lm, hash(k.mark_atom, hash(:mrk, h))))
Base.:(==)(a::_MomentRowKey, b::_MomentRowKey) =
    a.mark_atom == b.mark_atom && a.mark_lm == b.mark_lm && a.env == b.env

"""
    moment_resolvability(mb; rtol = nothing) -> NamedTuple

The pointed periodic-resolvability gate (design record D9′): what THIS reference
cell can and cannot determine about the pointed columns. Returns

- `vanishing::Vector{Int}` — columns whose signature expansion cancels identically
  on cell-periodic data (judged per column against its own gross accumulation, the
  `_CANCELLATION_RTOL` convention);
- `rank::Int` and `null_combinations::Vector{Vector{Tuple{Int,Float64}}}` — the
  numerical rank of the kept signature block and, per flat direction, the column
  indices with their weights (the gate names the dependent columns, not just the
  count);
- `census::Vector` — per cluster orbit, the number of stabilizer-inequivalent
  admissible mark placements (`n_mark_classes ≥ 2` preregisters the face-(b)
  hazard: same unmarked cluster, different marks — the pairs whose columns can
  collapse to a determined sum under a boundary tie).

Purely structural (the symbolic expansion, never sampled data); a full-rank result
certifies resolvability of the basis on this cell, not identifiability from any
particular training set.

The default-`rtol` (`rtol = nothing` → `1e-10`) result is cached on the basis (the
answer is a pure function of the basis, and [`MomentDataset`](@ref) runs this gate
at every construction — a train/held-out pair would otherwise pay the symbolic
expansion twice). Single-threaded, cached calls return the SAME object (`===`);
treat it as read-only. Passing an explicit `rtol` — the default value included —
always recomputes and is never cached. Concurrent first calls may compute twice and
race the store — the value is deterministic, so both compute equal (not `===`)
results and the race is benign. An `UnclassifiableBasis` refusal is deliberately
not cached: the gate re-throws loudly on every call.
"""
function moment_resolvability(mb::MomentBasis; rtol::Union{Nothing,Real} = nothing)
    if rtol === nothing
        mb.resolvability[] === nothing &&
            (mb.resolvability[] = _moment_resolvability(mb, 1e-10))
        return mb.resolvability[]::NamedTuple
    end
    return _moment_resolvability(mb, Float64(rtol))
end

function _moment_resolvability(mb::MomentBasis, rtol::Float64)
    sal = salcs(mb)
    # Same refusal as the energy-side `unresolvable_columns`, same reason: a member
    # carrying two ENVIRONMENT spin slots on one reference-cell atom (two periodic
    # images of one neighbor) multiplies two harmonics of the SAME unit vector, and
    # the signature expansion — which keys factors as independent monomials — then
    # OVERCOUNTS the rank (the product reduces by the sphere's Clebsch–Gordan
    # relations; measured: 108 symbolic vs 98 actual on the FeGe primitive cell).
    # The mark factor is exempt: it reads the independent axis variable ê, never
    # the same sphere as an environment factor.
    for (j, s) in enumerate(sal), mem in s.members, t in mem.terms
        env_atoms = Int[]
        mark_site = 0
        for sl in t.slots
            sl.factor.channel == DISP && (mark_site = sl.site)
        end
        # Before the guard reads `mem.atoms[mark_site]`: a markless term would index
        # position 0 and die on a `BoundsError` instead of saying what is wrong. The
        # label enumeration gives every pointed label exactly one mark, so this is the
        # invariant's loud statement, not a reachable path.
        mark_site == 0 && error("pointed SALC $j (key $(s.key)) has a term with no " *
                                "mark slot — every pointed label carries exactly one")
        for sl in t.slots
            (sl.factor.channel == SPIN && sl.site != mark_site) || continue
            push!(env_atoms, mem.atoms[sl.site])
        end
        # The mark is excluded by SITE index above, so include its ATOM here: a star
        # whose environment landed on a periodic image of the mark itself would read
        # the substituted evaluation axis as if it were a spin, and the site-index
        # exclusion cannot see that. This is the ONLY lock on that door. The minimum-image
        # neighbor list having no self-pairs does not close it: that is a statement about
        # the CENTRE's own neighbors, and the `N!` re-anchoring in
        # `_pointed_star_candidates` walks the mark around the star, so an environment
        # of the original centre becomes the mark and another environment can sit on an
        # image of it. Nor does the spoke test inside `admit`: `_dmin2_matrix` leaves the
        # diagonal at `Inf` (the list drops `i == j`), so the `dmin2_star[a,a]` spoke
        # test is identically true and only the raw radius applies. Do not remove the
        # `mem.atoms[mark_site]` term.
        allunique(vcat(env_atoms, mem.atoms[mark_site])) ||
            throw(UnclassifiableBasis("pointed SALC $j (key $(s.key)) has a member " *
                                      "with two spin factors on one reference-cell " *
                                      "atom (two periodic images of one neighbor, or " *
                                      "an environment on an image of the mark): the " *
                                      "symbolic signature cannot classify harmonic " *
                                      "products on a single sphere, so the gate " *
                                      "refuses rather than overcounting the rank. " *
                                      "Reduce cutoff_star below the tied shell (a " *
                                      "vector cuts one star order without touching " *
                                      "the others), step nbody back (the tie " *
                                      "multiplicity grows as (tie)^(N-1), so a higher " *
                                      "body order refuses where a lower one passed), " *
                                      "or use a reference " *
                                      "cell in which the images are distinct atoms"))
    end
    rows = Dict{_MomentRowKey,Int}()
    entries = Vector{Vector{Tuple{Int,Float64}}}()   # per row: (col, weight)
    gross = zeros(length(sal))
    for (j, s) in enumerate(sal)
        scale = (4π)^(count(has_spin, s.decors) / 2)
        for mem in s.members, t in mem.terms
            slots = t.slots
            mslot = findfirst(sl -> sl.factor.channel == DISP, slots)
            mslot === nothing && error("pointed SALC without a mark slot")
            mark_site = slots[mslot].site
            mark_atom = mem.atoms[mark_site]
            for index in CartesianIndices(t.folded)
                w = t.folded[index] * scale
                w == 0.0 && continue
                mark_lm = (0, 0)
                env = Tuple{Int,Int,Int}[]
                for i in eachindex(slots)
                    sl = slots[i]
                    if sl.factor.channel == DISP
                        continue                     # the mark's |u|²R₀₀ ≡ const
                    elseif sl.site == mark_site
                        mark_lm = (sl.factor.l, index[i] - sl.factor.l - 1)
                    else
                        push!(env, (mem.atoms[sl.site], sl.factor.l,
                                    index[i] - sl.factor.l - 1))
                    end
                end
                sort!(env)
                key = _MomentRowKey(mark_atom, mark_lm, env)
                r = get!(rows, key) do
                    push!(entries, Tuple{Int,Float64}[])
                    length(entries)
                end
                push!(entries[r], (j, w))
                gross[j] += abs(w)
            end
        end
    end
    S = zeros(length(entries), length(sal))
    for (r, es) in enumerate(entries), (j, w) in es
        S[r, j] += w
    end
    colnorm = [norm(@view S[:, j]) for j = 1:length(sal)]
    vanishing = [j for j = 1:length(sal)
                 if gross[j] > 0.0 && colnorm[j] <= _CANCELLATION_RTOL * gross[j]]
    kept = setdiff(1:length(sal), vanishing)
    rank = 0
    null_combinations = Vector{Vector{Tuple{Int,Float64}}}()
    if !isempty(kept)
        F = svd(S[:, kept])
        cut = maximum(F.S; init = 0.0) * max(Float64(rtol),
                                             minimum(size(S)) * eps(Float64))
        rank = count(>(cut), F.S)
        c = length(kept)
        _push_comb!(v) = push!(null_combinations,
                               [(kept[t], v[t]) for t in eachindex(kept)
                                if abs(v[t]) > 1e-8])
        for q = (rank + 1):size(F.V, 2)
            _push_comb!(F.V[:, q])
        end
        # A WIDE S (more kept columns than signature rows) has flat directions
        # the economy SVD cannot list: its V spans only min(r, c) directions, and
        # the orthogonal complement of span(V) is null too. Found by the P1
        # face-(b) control (rank 20 of 74 kept, null_combinations empty — the
        # dataset door's dependency disclosure silently missed all 54). The
        # complement is read off a QR completion of V (never a full SVD: the
        # row side can be huge and its full U is never needed).
        if size(F.V, 2) < c
            Qfull = qr(F.V).Q * Matrix{Float64}(I, c, c)
            for q = (size(F.V, 2) + 1):c
                _push_comb!(Qfull[:, q])
            end
        end
    end
    # census: per cluster orbit (body, orbit_id), the stabilizer-inequivalent
    # admissible mark placements of the representative.
    # Reconstruct each orbit's mark classes from the SALCs sharing the orbit: every
    # member's marked atom, folded under "same reference-cell atom". ONE pass, grouping
    # by orbit key — the obvious nested form is orbits × columns × members, which is
    # quadratic in the column count on exactly the many-orbit bases the pointed channel
    # is used on. `order` keeps first appearance, so the census order is unchanged.
    marks_by_key = Dict{Tuple{Int,Int},Set{Int}}()
    order = Tuple{Int,Int}[]
    for s in sal
        key = (s.key.body, s.key.orbit_id)
        marks = get!(marks_by_key, key) do
            push!(order, key)
            Set{Int}()
        end
        for mem in s.members, t in mem.terms
            for sl in t.slots
                sl.factor.channel == DISP || continue
                push!(marks, mem.atoms[sl.site])
            end
        end
    end
    census = NamedTuple[(; body = k[1], orbit_id = k[2],
                          n_mark_atoms = length(marks_by_key[k])) for k in order]
    return (; vanishing, rank, kept, null_combinations, census)
end
