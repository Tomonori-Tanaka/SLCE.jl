# Truncation-spec resolution. The ergonomic ("sugar") forms accepted by the
# `BasisSpec` keyword constructor — label-keyed species tables with a `"*"`
# wildcard, body-order-keyed tables, unordered hyphenated pair keys ("Fe-Nd",
# "Fe-*", "*-*") — are expanded here into dense, validated canonical arrays.
# Resolution is specificity-based (concrete > one-sided wildcard > "*-*"),
# never entry-order-based, so a spec can list entries in any order; entries of
# equal specificity that disagree on a value are an error, not a silent
# override. Unknown labels, uncovered species/pairs, and body orders outside
# the valid range are errors too (no silently-ignored blocks).

# (`LSUM_UNCAPPED` — the no-cap sentinel this file resolves to — lives in
# `basis/salcbasis.jl`, next to the enumeration it parameterizes.)

# One species-table entry key: a concrete label or the "*" wildcard.
function _species_key_index(key::AbstractString, labels::Vector{String},
                            what::String)::Union{Nothing,Int}
    key == "*" && return nothing
    i = findfirst(==(String(key)), labels)
    i === nothing &&
        throw(ArgumentError("$what: unknown species label $(repr(String(key))) " *
                            "(labels: $(join(labels, ", ")))"))
    return i
end

_table_pairs(x::AbstractDict) = collect(x)
_table_pairs(x::AbstractVector{<:Pair}) = collect(x)

# ---------------------------------------------------------------------------
# species-keyed table (lmax): Integer | dense Vector | label-keyed table
# ---------------------------------------------------------------------------

_resolve_species_table(x::Integer, nkd::Int, ::Vector{String}, what::String) =
    fill(Int(x), nkd)

function _resolve_species_table(x::AbstractVector{<:Integer}, nkd::Int,
                                ::Vector{String}, what::String)::Vector{Int}
    length(x) == nkd ||
        throw(ArgumentError("$what: got $(length(x)) entries for $nkd species"))
    return collect(Int, x)
end

function _resolve_species_table(x::Union{AbstractDict{<:AbstractString},
                                         AbstractVector{<:Pair{<:AbstractString}}},
                                nkd::Int, labels::Vector{String},
                                what::String)::Vector{Int}
    isempty(labels) &&
        throw(ArgumentError("$what: label-keyed entries need the species labels — " *
                            "construct via `BasisSpec(labels; ...)` or " *
                            "`BasisSpec(crystal; ...)`"))
    out = fill(-1, nkd)          # -1 = unset
    fallback = -1                # value of "*", if given
    seen = Set{String}()
    for (key, val) in _table_pairs(x)
        k = String(key)
        k in seen && throw(ArgumentError("$what: duplicate key $(repr(k))"))
        push!(seen, k)
        i = _species_key_index(k, labels, what)
        v = Int(val)
        v >= 0 || throw(ArgumentError("$what: entries must be ≥ 0; got $k = $v"))
        i === nothing ? (fallback = v) : (out[i] = v)
    end
    for i = 1:nkd
        out[i] >= 0 && continue
        fallback >= 0 ||
            throw(ArgumentError("$what: species $(repr(labels[i])) is not covered " *
                                "(add it or a \"*\" fallback entry)"))
        out[i] = fallback
    end
    return out
end

_resolve_species_table(x, ::Int, ::Vector{String}, what::String) =
    throw(ArgumentError("$what: unsupported form $(typeof(x)) — use an Int " *
                        "(broadcast), a per-species Vector{Int}, or label-keyed " *
                        "pairs like [\"*\" => 3, \"B\" => 0]"))

# ---------------------------------------------------------------------------
# body-keyed table (lsum): nothing | Integer | (body => value) table
# ---------------------------------------------------------------------------

_resolve_lsum(::Nothing, nbody::Int) = fill(LSUM_UNCAPPED, nbody)

function _resolve_lsum(x::Integer, nbody::Int)::Vector{Int}
    x >= 0 || throw(ArgumentError("lsum: must be ≥ 0; got $x"))
    return fill(Int(x), nbody)
end

function _resolve_lsum(x::Union{AbstractDict{<:Integer},
                                AbstractVector{<:Pair{<:Integer}}},
                       nbody::Int)::Vector{Int}
    out = fill(LSUM_UNCAPPED, nbody)
    seen = Set{Int}()
    for (n, val) in _table_pairs(x)
        n in seen && throw(ArgumentError("lsum: duplicate body order $n"))
        push!(seen, n)
        1 <= n <= nbody ||
            throw(ArgumentError("lsum: body order $n is outside 1:$nbody " *
                                "(nbody = $nbody gates every higher order)"))
        v = Int(val)
        v >= 0 || throw(ArgumentError("lsum: body$n must be ≥ 0; got $v"))
        out[n] = v
    end
    return out
end

_resolve_lsum(x, ::Int) =
    throw(ArgumentError("lsum: unsupported form $(typeof(x)) — use an Int (all " *
                        "body orders), body-keyed pairs like [1 => 0, 2 => 4], " *
                        "or `nothing` (no cap)"))

# ---------------------------------------------------------------------------
# pair-keyed cutoff table → symmetric nkd×nkd matrix
# ---------------------------------------------------------------------------

_check_cutoff_value(v::Real, ctx::String)::Float64 =
    (isnan(v) || v < 0) ?
        throw(ArgumentError("$ctx: cutoff must be ≥ 0 Å or Inf (no cutoff); got $v" *
                            (v == -1 ? " (Magesty's -1 sentinel — use Inf here)" : ""))) :
        Float64(v)

function _pair_key_parts(key::AbstractString, ctx::String)
    parts = split(String(key), '-')
    length(parts) == 2 && !isempty(parts[1]) && !isempty(parts[2]) ||
        throw(ArgumentError("$ctx: pair key $(repr(String(key))) is not of the " *
                            "form \"A-B\" (wildcards \"A-*\", \"*-*\" allowed)"))
    return String(parts[1]), String(parts[2])
end

function _resolve_pair_table(x::Union{AbstractDict{<:AbstractString},
                                      AbstractVector{<:Pair{<:AbstractString}}},
                             nkd::Int, labels::Vector{String},
                             ctx::String)::Matrix{Float64}
    isempty(labels) &&
        throw(ArgumentError("$ctx: pair-keyed cutoffs need the species labels — " *
                            "construct via `BasisSpec(labels; ...)` or " *
                            "`BasisSpec(crystal; ...)`"))
    any(contains('-'), labels) &&
        throw(ArgumentError("$ctx: pair keys are hyphen-separated, but a species " *
                            "label itself contains '-' " *
                            "($(join(filter(contains('-'), labels), ", ")))"))
    M = fill(NaN, nkd, nkd)
    level = fill(-1, nkd, nkd)        # specificity that set each entry
    seen = Set{Tuple{String,String}}()
    for (key, val) in _table_pairs(x)
        a, b = _pair_key_parts(key, ctx)
        canon = a <= b ? (a, b) : (b, a)
        canon in seen &&
            throw(ArgumentError("$ctx: duplicate pair key $(repr(String(key))) " *
                                "(pairs are unordered: \"$a-$b\" ≡ \"$b-$a\")"))
        push!(seen, canon)
        ia = _species_key_index(a, labels, ctx)
        ib = _species_key_index(b, labels, ctx)
        lv = (ia !== nothing) + (ib !== nothing)
        v = _check_cutoff_value(val, "$ctx $(repr(String(key)))")
        for i in (ia === nothing ? (1:nkd) : (ia:ia)),
            j in (ib === nothing ? (1:nkd) : (ib:ib))

            for (p, q) in ((i, j), (j, i))
                if level[p, q] < lv
                    M[p, q] = v
                    level[p, q] = lv
                elseif level[p, q] == lv && M[p, q] != v
                    throw(ArgumentError("$ctx: pair $(labels[p])-$(labels[q]) is " *
                                        "matched by two entries of equal " *
                                        "specificity with different values " *
                                        "($(M[p, q]) vs $v)"))
                end
            end
        end
    end
    for i = 1:nkd, j = i:nkd
        isnan(M[i, j]) &&
            throw(ArgumentError("$ctx: pair $(labels[i])-$(labels[j]) is not " *
                                "covered (add it or a \"*-*\" fallback entry)"))
    end
    return M
end

# ---------------------------------------------------------------------------
# cutoff: Real | pair table | canonical Vector{Matrix} | (body => ...) table
# ---------------------------------------------------------------------------

_is_bodykeyed(x::AbstractDict{<:Integer}) = true
_is_bodykeyed(x::AbstractVector{<:Pair{<:Integer}}) = true
_is_bodykeyed(x) = false

function _resolve_cutoff(x, nbody::Int, nkd::Int, labels::Vector{String})
    nmat = max(nbody - 1, 0)
    if x isa Real
        v = _check_cutoff_value(x, "cutoff")
        return [fill(v, nkd, nkd) for _ = 1:nmat]
    elseif x isa AbstractVector{<:AbstractMatrix{<:Real}}
        # canonical (resolved) form — used by persistence reload
        length(x) == nmat ||
            throw(ArgumentError("cutoff: got $(length(x)) matrices for body " *
                                "orders 2:$nbody"))
        out = Matrix{Float64}[]
        for (k, Mx) in enumerate(x)
            size(Mx) == (nkd, nkd) ||
                throw(ArgumentError("cutoff: body$(k + 1) matrix is $(size(Mx)), " *
                                    "expected ($nkd, $nkd)"))
            M = [_check_cutoff_value(Mx[i, j], "cutoff body$(k + 1)")
                 for i = 1:nkd, j = 1:nkd]
            M == M' || throw(ArgumentError("cutoff: body$(k + 1) matrix is not " *
                                           "symmetric"))
            push!(out, M)
        end
        return out
    elseif _is_bodykeyed(x)
        vals = Vector{Union{Nothing,Matrix{Float64}}}(nothing, nmat)
        seen = Set{Int}()
        for (n, val) in _table_pairs(x)
            n in seen && throw(ArgumentError("cutoff: duplicate body order $n"))
            push!(seen, n)
            2 <= n <= nbody ||
                throw(ArgumentError("cutoff: body order $n is outside 2:$nbody " *
                                    (n == 1 ? "(1-body terms have no cutoff)" :
                                     "(nbody = $nbody gates every higher order)")))
            vals[n - 1] = val isa Real ?
                fill(_check_cutoff_value(val, "cutoff body$n"), nkd, nkd) :
                _resolve_pair_table(val, nkd, labels, "cutoff body$n")
        end
        for n = 2:nbody
            vals[n - 1] === nothing &&
                throw(ArgumentError("cutoff: body order $n is not covered — give " *
                                    "every order in 2:$nbody (or one scalar/pair " *
                                    "table for all)"))
        end
        return Matrix{Float64}[v for v in vals]
    elseif x isa Union{AbstractDict{<:AbstractString},
                       AbstractVector{<:Pair{<:AbstractString}}}
        M = _resolve_pair_table(x, nkd, labels, "cutoff")
        return [copy(M) for _ = 1:nmat]
    end
    throw(ArgumentError("cutoff: unsupported form $(typeof(x)) — use a scalar " *
                        "(Inf = no cutoff), a pair table like " *
                        "[\"Fe-Fe\" => 4.0, \"*-*\" => 8.0], body-keyed pairs " *
                        "like [2 => Inf, 3 => 4.0], or a nested combination"))
end

# Element-wise max over the per-body cutoff matrices: the superset radius the
# neighbor list is built with (per-body admission then trims edges in
# `candidate_clusters`). `nbody == 1` needs no pairs at all.
_superset_cutoff(sp)::Matrix{Float64} =
    isempty(sp.cutoff) ? zeros(length(sp.lmax), length(sp.lmax)) :
    reduce((a, b) -> max.(a, b), sp.cutoff)

# ---------------------------------------------------------------------------
# sector table: `Sector` sugar → `SectorRule` dense canonical form
# ---------------------------------------------------------------------------

"""
    Sector(; spin = nothing, disp = nothing, soc = true, cutoff = Inf,
           nbody = nothing)

One row of a [`BasisSpec`](@ref) sector table — a family of decorated-cluster
labels admitted into the joint spin–lattice basis. The truncation is the
**union** of the rows (each row maps 1:1 onto a sector of the theory note's
truncation table), intersected with the spec-global per-species caps
(`lmax` / `pmax`) and per-body `lsum`.

- `spin` — the spin-factor content of the sector:
  - `nothing`: no spin factors (a lattice-only sector, e.g. force constants);
  - an `Int` or a `Vector{Int}` multiset (e.g. `[1, 1]`): exactly these
    tesseral ranks, one per spin-decorated site. `Σl` must be even — the
    time-reversal screen; an odd multiset can never enter the basis;
  - a NamedTuple `(nbody = 2:4, lmax = 3, lsum = 6)`: every even-`Σl` rank
    multiset over that many spin-decorated sites (`nbody`, an `Int` or range,
    required), per-site rank ≤ `lmax` (optional; intersected with the
    per-species `lmax`), `Σl ≤ lsum` (optional).
- `disp` — the displacement-factor content: `nothing` (a clamped, `p = 0`
  sector), or a total-degree budget — an `Int`, a range, or a NamedTuple
  `(degree = 1:2,)` — for `Σᵢ (2kᵢ + lᵢ)` over the displacement factors. The
  `(k, l)` harmonic labels realizing each degree are enumerated automatically
  (degree-`p` polynomials decompose exactly as `⊕_{2k+l=p} |u|^{2k} R_l`); a
  budget starting at `0` makes the displacement part optional.
- `soc::Bool = true` — `false` keeps only the `L_S = 0` (total-spin-scalar)
  blocks of this sector's labels: together with the always-on even-`Σl_spin`
  screen this is the exact SOC-free selection rule under the one grey-group
  projection (design record §5). It is a truncation rule (model support), not
  a fit-staging mask.
- `cutoff` — sector-local cluster admission: a radius in Å or a species-pair
  table (`["Fe-*" => 6.0, "*-*" => 8.0]`, resolved like the [`BasisSpec`](@ref)
  cutoff sugar); every cluster edge must be within it. `Inf` = every cluster
  the crystal resolves (see [`MinimumImage`](@ref)).
- `nbody` — optional total decorated-site count (`Int` or range). Default:
  every body order realizable from the `spin`/`disp` content — note a site may
  carry a spin factor AND a displacement factor, so e.g. `spin = [2]`,
  `disp = (degree = 1:2,)` realizes body orders 1 (both factors on one site)
  through 3 (the spin site plus two degree-1 displacement sites); pass
  `nbody = 1` for the on-site SOC sector.

A `Sector` is unresolved sugar; the [`BasisSpec`](@ref) constructor resolves it
against the species labels into the dense canonical [`SLCE.SectorRule`](@ref)
form (label-keyed cutoff tables need the labels, so resolution happens there).
"""
struct Sector
    spin::Any
    disp::Any
    soc::Bool
    cutoff::Any
    nbody::Any
end

Sector(; spin = nothing, disp = nothing, soc::Bool = true, cutoff = Inf,
       nbody = nothing) = Sector(spin, disp, soc, cutoff, nbody)

Base.:(==)(a::Sector, b::Sector) =
    isequal(a.spin, b.spin) && isequal(a.disp, b.disp) && a.soc == b.soc &&
    isequal(a.cutoff, b.cutoff) && isequal(a.nbody, b.nbody)

function Base.show(io::IO, s::Sector)
    parts = String[]
    s.spin === nothing || push!(parts, "spin = $(repr(s.spin))")
    s.disp === nothing || push!(parts, "disp = $(repr(s.disp))")
    s.soc || push!(parts, "soc = false")
    s.cutoff isa Real && isinf(s.cutoff) || push!(parts, "cutoff = $(repr(s.cutoff))")
    s.nbody === nothing || push!(parts, "nbody = $(repr(s.nbody))")
    print(io, "Sector(", join(parts, ", "), ")")
end

"""
    SectorRule

The resolved, dense canonical form of one [`Sector`](@ref) row, stored in
`BasisSpec.sectors` (public, unexported — construct via [`Sector`](@ref)):

- `spin_mode::Symbol` — `:none` (no spin factors) | `:explicit` (the exact
  multiset `spin_ls`) | `:any` (every even-`Σl` multiset within the caps);
- `spin_ls::Vector{Int}` — the sorted rank multiset (`:explicit`; else empty);
- `spin_nsites::Tuple{Int,Int}` — `(lo, hi)` count of spin-decorated sites;
- `spin_lmax::Int` / `spin_lsum::Int` — sector-local per-site rank cap and
  `Σl` cap for `:any` (`LSUM_UNCAPPED` = uncapped);
- `disp_degree::Tuple{Int,Int}` — `(lo, hi)` total displacement degree
  `Σ(2k + l)`; `(0, 0)` = no displacement factors;
- `nbody::Tuple{Int,Int}` — `(lo, hi)` total decorated-site count;
- `soc::Bool` — `false` keeps only the `L_S = 0` blocks;
- `cutoff::Matrix{Float64}` — the resolved species-pair admission radii (Å).

Ranges are stored as `(lo, hi)` tuples so the form is dense, comparable, and
persistable.
"""
struct SectorRule
    spin_mode::Symbol
    spin_ls::Vector{Int}
    spin_nsites::Tuple{Int,Int}
    spin_lmax::Int
    spin_lsum::Int
    disp_degree::Tuple{Int,Int}
    nbody::Tuple{Int,Int}
    soc::Bool
    cutoff::Matrix{Float64}

    function SectorRule(spin_mode::Symbol, spin_ls::Vector{Int},
                        spin_nsites::Tuple{Int,Int}, spin_lmax::Int,
                        spin_lsum::Int, disp_degree::Tuple{Int,Int},
                        nbody::Tuple{Int,Int}, soc::Bool,
                        cutoff::Matrix{Float64})
        spin_mode in (:none, :explicit, :any) ||
            throw(ArgumentError("spin_mode must be :none, :explicit, or :any; " *
                                "got $spin_mode"))
        if spin_mode == :explicit
            isempty(spin_ls) &&
                throw(ArgumentError("spin_mode = :explicit needs a nonempty spin_ls"))
            issorted(spin_ls) ||
                throw(ArgumentError("spin_ls must be sorted; got $spin_ls"))
            all(>=(1), spin_ls) ||
                throw(ArgumentError("spin_ls entries must be ≥ 1; got $spin_ls"))
            iseven(sum(spin_ls)) ||
                throw(ArgumentError("Σ spin_ls must be even (time-reversal " *
                                    "screen); got $spin_ls"))
            spin_nsites == (length(spin_ls), length(spin_ls)) ||
                throw(ArgumentError("spin_nsites $spin_nsites does not match the " *
                                    "explicit multiset length $(length(spin_ls))"))
        else
            isempty(spin_ls) ||
                throw(ArgumentError("spin_ls must be empty for spin_mode = " *
                                    "$spin_mode"))
            lo, hi = spin_nsites
            spin_mode == :none && spin_nsites != (0, 0) &&
                throw(ArgumentError("spin_mode = :none requires spin_nsites = (0, 0)"))
            spin_mode == :any && !(1 <= lo <= hi) &&
                throw(ArgumentError("spin_nsites must satisfy 1 ≤ lo ≤ hi; " *
                                    "got $spin_nsites"))
        end
        spin_lmax >= 1 ||
            throw(ArgumentError("spin_lmax must be ≥ 1 (LSUM_UNCAPPED = uncapped); " *
                                "got $spin_lmax"))
        spin_lsum >= 0 ||
            throw(ArgumentError("spin_lsum must be ≥ 0 (LSUM_UNCAPPED = uncapped); " *
                                "got $spin_lsum"))
        dlo, dhi = disp_degree
        disp_degree == (0, 0) || (0 <= dlo <= dhi && dhi >= 1) ||
            throw(ArgumentError("disp_degree must be (0, 0) or satisfy " *
                                "0 ≤ lo ≤ hi with hi ≥ 1; got $disp_degree"))
        spin_mode == :none && disp_degree == (0, 0) &&
            throw(ArgumentError("a sector needs spin and/or displacement content"))
        1 <= nbody[1] <= nbody[2] ||
            throw(ArgumentError("nbody must satisfy 1 ≤ lo ≤ hi; got $nbody"))
        size(cutoff, 1) == size(cutoff, 2) ||
            throw(ArgumentError("sector cutoff matrix is $(size(cutoff)), " *
                                "expected square"))
        cutoff == cutoff' ||
            throw(ArgumentError("sector cutoff matrix is not symmetric"))
        all(v -> !isnan(v) && v >= 0, cutoff) ||
            throw(ArgumentError("sector cutoff entries must be ≥ 0 or Inf"))
        return new(spin_mode, copy(spin_ls), spin_nsites, spin_lmax, spin_lsum,
                   disp_degree, nbody, soc, copy(cutoff))
    end
end

Base.:(==)(a::SectorRule, b::SectorRule) =
    a.spin_mode == b.spin_mode && a.spin_ls == b.spin_ls &&
    a.spin_nsites == b.spin_nsites && a.spin_lmax == b.spin_lmax &&
    a.spin_lsum == b.spin_lsum && a.disp_degree == b.disp_degree &&
    a.nbody == b.nbody && a.soc == b.soc && a.cutoff == b.cutoff

_rangestr(t::Tuple{Int,Int}) = t[1] == t[2] ? string(t[1]) : "$(t[1]):$(t[2])"

function Base.show(io::IO, r::SectorRule)
    parts = String[]
    r.spin_mode == :explicit && push!(parts, "spin = $(r.spin_ls)")
    if r.spin_mode == :any
        caps = String[]
        r.spin_lmax == LSUM_UNCAPPED || push!(caps, "lmax = $(r.spin_lmax)")
        r.spin_lsum == LSUM_UNCAPPED || push!(caps, "lsum = $(r.spin_lsum)")
        push!(parts, "spin = (nbody = $(_rangestr(r.spin_nsites))" *
                     (isempty(caps) ? "" : ", " * join(caps, ", ")) * ")")
    end
    r.disp_degree == (0, 0) ||
        push!(parts, "disp degree = $(_rangestr(r.disp_degree))")
    r.soc || push!(parts, "soc = false")
    push!(parts, "nbody = $(_rangestr(r.nbody))")
    c = r.cutoff
    push!(parts, all(==(c[1, 1]), c) ? "cutoff = $(c[1, 1]) Å" : "cutoff = per-pair")
    print(io, "SectorRule(", join(parts, ", "), ")")
end

# (lo, hi) resolution of an Int-or-range sector knob.
function _resolve_intrange(x, ctx::String; lo_min::Int)::Tuple{Int,Int}
    if x isa Integer
        x >= lo_min || throw(ArgumentError("$ctx must be ≥ $lo_min; got $x"))
        return (Int(x), Int(x))
    elseif x isa AbstractUnitRange{<:Integer}
        isempty(x) && throw(ArgumentError("$ctx is an empty range"))
        first(x) >= lo_min ||
            throw(ArgumentError("$ctx must start ≥ $lo_min; got $x"))
        return (Int(first(x)), Int(last(x)))
    end
    throw(ArgumentError("$ctx: unsupported form $(typeof(x)) — use an Int or a " *
                        "range like 2:4"))
end

_check_nt_keys(nt::NamedTuple, allowed::Tuple, ctx::String) =
    for k in keys(nt)
        k in allowed ||
            throw(ArgumentError("$ctx: unknown key $(repr(k)) (allowed: " *
                                "$(join(allowed, ", ")))"))
    end

# Resolve one `Sector` row against the species labels into a `SectorRule`.
function _resolve_sector(s::Sector, nkd::Int, labels::Vector{String},
                         i::Int)::SectorRule
    ctx = "sectors[$i]"
    # --- spin content ---
    spin_lmax = LSUM_UNCAPPED
    spin_lsum = LSUM_UNCAPPED
    if s.spin === nothing
        spin_mode, spin_ls, spin_nsites = :none, Int[], (0, 0)
    elseif s.spin isa Integer || s.spin isa AbstractVector{<:Integer}
        spin_ls = sort!(collect(Int, s.spin isa Integer ? (s.spin,) : s.spin))
        isempty(spin_ls) &&
            throw(ArgumentError("$ctx: empty spin multiset — use spin = nothing " *
                                "for a lattice-only sector"))
        all(>=(1), spin_ls) ||
            throw(ArgumentError("$ctx: spin ranks must be ≥ 1; got $spin_ls"))
        iseven(sum(spin_ls)) ||
            throw(ArgumentError("$ctx: Σl of the spin multiset must be even " *
                                "(time-reversal screen); got $spin_ls"))
        spin_mode = :explicit
        spin_nsites = (length(spin_ls), length(spin_ls))
    elseif s.spin isa NamedTuple
        _check_nt_keys(s.spin, (:nbody, :lmax, :lsum), "$ctx spin")
        haskey(s.spin, :nbody) ||
            throw(ArgumentError("$ctx spin: the NamedTuple form needs `nbody` " *
                                "(an Int or range of spin-decorated site counts)"))
        spin_nsites = _resolve_intrange(s.spin.nbody, "$ctx spin nbody"; lo_min = 1)
        if haskey(s.spin, :lmax)
            v = s.spin.lmax
            v isa Integer && v >= 1 ||
                throw(ArgumentError("$ctx spin lmax must be an Int ≥ 1; got $v"))
            spin_lmax = Int(v)
        end
        if haskey(s.spin, :lsum)
            v = s.spin.lsum
            v isa Integer && v >= 0 ||
                throw(ArgumentError("$ctx spin lsum must be an Int ≥ 0; got $v"))
            spin_lsum = Int(v)
        end
        spin_mode, spin_ls = :any, Int[]
    else
        throw(ArgumentError("$ctx spin: unsupported form $(typeof(s.spin)) — use " *
                            "nothing, a rank multiset like [1, 1], or a NamedTuple " *
                            "like (nbody = 2:4, lmax = 3)"))
    end
    # --- displacement content ---
    if s.disp === nothing
        disp_degree = (0, 0)
    else
        x = s.disp
        if x isa NamedTuple
            _check_nt_keys(x, (:degree,), "$ctx disp")
            haskey(x, :degree) ||
                throw(ArgumentError("$ctx disp: the NamedTuple form needs `degree`"))
            x = x.degree
        end
        disp_degree = _resolve_intrange(x, "$ctx disp degree"; lo_min = 0)
        disp_degree[2] >= 1 ||
            throw(ArgumentError("$ctx disp degree must reach ≥ 1 — use " *
                                "disp = nothing for a clamped (p = 0) sector"))
    end
    spin_mode == :none && disp_degree == (0, 0) &&
        throw(ArgumentError("$ctx: sector has neither spin nor displacement " *
                            "content"))
    # --- total body order ---
    # Each displacement factor has degree ≥ 1, so a degree budget of hi occupies
    # at most hi sites; spin and displacement factors may share a site, hence
    # the realizable range below.
    dlo = disp_degree[1] > 0 ? 1 : 0
    dhi = disp_degree[2]
    derived = (max(1, spin_nsites[1], dlo), spin_nsites[2] + dhi)
    if s.nbody === nothing
        nbody = derived
    else
        e = _resolve_intrange(s.nbody, "$ctx nbody"; lo_min = 1)
        nbody = (max(e[1], derived[1]), min(e[2], derived[2]))
        nbody[1] <= nbody[2] ||
            throw(ArgumentError("$ctx: nbody = $(repr(s.nbody)) is incompatible " *
                                "with the spin/disp content (realizable body " *
                                "orders $(derived[1]):$(derived[2]))"))
    end
    # --- cutoff ---
    M = s.cutoff isa Real ?
        fill(_check_cutoff_value(s.cutoff, "$ctx cutoff"), nkd, nkd) :
        _resolve_pair_table(s.cutoff, nkd, labels, "$ctx cutoff")
    return SectorRule(spin_mode, spin_ls, spin_nsites, spin_lmax, spin_lsum,
                      disp_degree, nbody, s.soc, M)
end

# Resolve the whole sector table; returns (rules, nbody, cutoff_envelope) with
# `nbody` derived (or validated against an explicit value) and the per-body
# cutoff envelope = elementwise max over the sectors admitting each body order
# (the cluster set is built once at the envelope; each sector then re-admits
# orbits within its own radii).
function _resolve_sectors(sectors, nkd::Int, labels::Vector{String},
                          nbody_kw)::Tuple{Vector{SectorRule},Int,
                                           Vector{Matrix{Float64}}}
    secs = sectors isa Sector ? [sectors] : sectors
    secs isa AbstractVector{<:Any} && all(s -> s isa Sector, secs) ||
        throw(ArgumentError("sectors must be a Sector or a Vector of Sectors"))
    isempty(secs) && throw(ArgumentError("sectors must be nonempty"))
    rules = SectorRule[_resolve_sector(s, nkd, labels, i)
                       for (i, s) in enumerate(secs)]
    derived = maximum(r.nbody[2] for r in rules)
    nbody = nbody_kw === nothing ? derived : Int(nbody_kw)
    nbody >= 1 || throw(ArgumentError("nbody must be ≥ 1; got $nbody"))
    for (i, r) in enumerate(rules)
        r.nbody[1] <= nbody ||
            throw(ArgumentError("sectors[$i] admits only body orders " *
                                "$(_rangestr(r.nbody)) but nbody = $nbody — the " *
                                "sector would contribute nothing"))
    end
    envelope = Matrix{Float64}[zeros(nkd, nkd) for _ = 2:nbody]
    for N = 2:nbody, r in rules
        r.nbody[1] <= N <= r.nbody[2] || continue
        envelope[N - 1] .= max.(envelope[N - 1], r.cutoff)
    end
    return rules, nbody, envelope
end
