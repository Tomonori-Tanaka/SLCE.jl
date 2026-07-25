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
