"""
Tabular access to fitted SCE coefficients.

`coeftable(fit_or_model)` returns an [`SCECoefficients`](@ref): a labeled, ordered
view of the SALC coefficients `Jϕ`, one row per design-matrix column, keyed by the
structural fields of each [`SALCKey`](@ref). It implements the **Tables.jl**
interface, so it drops straight into `DataFrame`, `CSV.write`, `Arrow.write`, … — the
library supplies the data source (it owns the mapping from internal storage to
meaningful rows); the caller brings the table / IO package. The reference energy `j0`
is the model intercept (not a per-row quantity); read it with [`intercept`](@ref).
"""

const _COEF_NAMES = (:body, :orbit_id, :ls, :Lf, :block, :J)
const _COEF_COLTYPES = (Int, Int, String, Int, Int, Float64)
const _CoefRow = NamedTuple{_COEF_NAMES,Tuple{Int,Int,String,Int,Int,Float64}}

# the sorted l-multiset label as a flat string, e.g. [1,1,2] -> "1,1,2"
_ls_string(ls::AbstractVector{<:Integer})::String = join(ls, ",")

"""
    SCECoefficients

A Tables.jl-compatible table of fitted SCE coefficients: parallel `keys` /
`jphi` (column `J`) plus the intercept `j0`. Build it with [`coeftable`](@ref);
iterate it for `NamedTuple` rows, or hand it to any Tables.jl sink.
"""
struct SCECoefficients
    keys::Vector{SALCKey}
    jphi::Vector{Float64}
    j0::Float64

    function SCECoefficients(keys::Vector{SALCKey}, jphi::Vector{Float64}, j0::Real)
        length(keys) == length(jphi) ||
            throw(ArgumentError("got $(length(keys)) keys for $(length(jphi)) coefficients"))
        return new(keys, jphi, Float64(j0))
    end
end

@inline _coef_row(c::SCECoefficients, i::Int)::_CoefRow =
    (body = c.keys[i].body, orbit_id = c.keys[i].orbit_id, ls = _ls_string(c.keys[i].ls),
     Lf = c.keys[i].Lf, block = c.keys[i].block, J = c.jphi[i])

"""
    coeftable(f::SLCEFit) -> SCECoefficients
    coeftable(m::SLCEModel) -> SCECoefficients

A Tables.jl-compatible table of the fitted coefficients — one row per SALC with
columns `body`, `orbit_id`, `ls` (the sorted `l`-multiset as a string, e.g.
`"1,1,2"`), `Lf`, `block`, and `J` (the coefficient `Jϕ`). The intercept `j0` is
available via [`intercept`](@ref). Example: `using DataFrames; DataFrame(coeftable(f))`.
"""
coeftable(f::SLCEFit)::SCECoefficients =
    SCECoefficients(copy(f.dataset.basis.salc_basis.keys), copy(f.jphi), f.j0)
coeftable(m::SLCEModel)::SCECoefficients =
    SCECoefficients(copy(m.keys), copy(m.jphi), m.j0)
# `copy` duplicates the `jphi` and key vectors so the table is independent of later
# refits; the `SALCKey`s themselves are shared (they are treated as immutable — used
# as `Dict` keys — so their `ls` is never mutated).

# --- collection / accessor interface ---
Base.length(c::SCECoefficients) = length(c.keys)
Base.eltype(::Type{SCECoefficients}) = _CoefRow
Base.getindex(c::SCECoefficients, i::Integer) = _coef_row(c, Int(i))
Base.firstindex(::SCECoefficients) = 1
Base.lastindex(c::SCECoefficients) = length(c)
function Base.iterate(c::SCECoefficients, i::Int = 1)
    i > length(c) && return nothing
    return (_coef_row(c, i), i + 1)
end

coef(c::SCECoefficients)::Vector{Float64} = c.jphi
intercept(c::SCECoefficients)::Float64 = c.j0

# --- Tables.jl row-access source ---
Tables.istable(::Type{SCECoefficients}) = true
Tables.rowaccess(::Type{SCECoefficients}) = true
Tables.rows(c::SCECoefficients) = c
Tables.schema(::SCECoefficients) = Tables.Schema(_COEF_NAMES, _COEF_COLTYPES)

# --- display ---
_fmtj(x::Float64)::String = string(round(x; sigdigits = 6))

Base.show(io::IO, c::SCECoefficients) =
    print(io, "SCECoefficients(", length(c), " terms, j0=", c.j0, ")")

function Base.show(io::IO, ::MIME"text/plain", c::SCECoefficients)
    n = length(c)
    print(io, "SCECoefficients: ", n, " term", n == 1 ? "" : "s", ", j0 = ", c.j0)
    n == 0 && return
    nshow = min(n, 20)
    cols = (["body"; [string(c.keys[i].body) for i = 1:nshow]],
            ["orbit_id"; [string(c.keys[i].orbit_id) for i = 1:nshow]],
            ["ls"; [_ls_string(c.keys[i].ls) for i = 1:nshow]],
            ["Lf"; [string(c.keys[i].Lf) for i = 1:nshow]],
            ["block"; [string(c.keys[i].block) for i = 1:nshow]],
            ["J"; [_fmtj(c.jphi[i]) for i = 1:nshow]])
    w = map(col -> maximum(length, col), cols)
    for r = 1:(nshow + 1)
        println(io)
        for k = 1:length(cols)
            print(io, "  ", lpad(cols[k][r], w[k]))
        end
    end
    n > nshow && print(io, "\n  ⋮ (", n - nshow, " more)")
    return
end
