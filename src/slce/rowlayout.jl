# The per-site basis-row layout — the contract between a fitted model and any
# consumer that tabulates site factors into a flat matrix and gathers from it
# (design record §7/§8; the Monte Carlo sweep kernels are the reason it exists).
#
# A sampler evaluates a term by gathering one number per tensor axis out of a
# per-site row table. Which row an axis reads is fixed at construction, so the
# gather loops never branch on channel. This file defines that row numbering ONCE
# so the producer (the model) and the consumer (the sampler's program builder)
# cannot disagree about it.
#
# The layout stacks blocks in Channel-enum order: the SPIN block first, then DISP,
# then (reserved) OCC. The SPIN block is deliberately `Harmonics.lm_index` at
# offset 0 — verbatim, not merely isomorphic — so a pure-spin model's layout is the
# one the spin-only consumers already use, and adding the displacement channel
# cannot move a single existing row.

"""
    RowLayout

The per-site basis-row numbering of a model, as returned by [`row_layout`](@ref).

A consumer tabulates, for each site, one row per `(channel, k, l, m)` the model can
read, and then gathers term factors by row index. Blocks are stacked in
`Channel`-enum order:

| block | rows | value in row `(…, m)` |
|:--|:--|:--|
| `SPIN` | `(spin_lmax + 1)²` at offset 0 | `Z_{l,m}(ê_a)` |
| `DISP` | `2l + 1` per `(k, l)` in `disp_factors`, at `disp_offset` | `\\|u_a\\|^{2k} R_{l,m}(u_a)` |

The `SPIN` block is exactly `SLCE.Harmonics.lm_index(l, m)` at offset 0, so on a
pure-spin model `nrows == (spin_lmax + 1)²` and every row index is the one the
spin-only consumers already use — the displacement channel only ever appends. That
block includes the `l = 0` row, which no `SiteFactor` addresses (a `SPIN` factor has
`l ≥ 1`; `l = 0` would be a constant): it is kept so the numbering is `lm_index`
itself rather than a shifted copy of it.

`spin_lmax == -1` means the model has no spin content at all (an empty `SPIN`
block); `disp_factors` is empty for a pure-spin model.

Use [`row_index`](@ref) rather than reconstructing the arithmetic.
"""
struct RowLayout
    nrows::Int
    spin_lmax::Int
    disp_offset::Int
    disp_factors::Vector{Tuple{Int,Int}}   # sorted (k, l)
    disp_starts::Vector{Int}               # row offset of each (k, l) block
end

# Value semantics: a layout is a description, and two descriptions of the same
# support are the same layout. (The default struct `==` would compare the vector
# fields by identity, so a freshly derived layout would never equal a stored one —
# and "did the support move?" is exactly the question a consumer asks across a
# coefficient hot-swap or a checkpoint reload.)
Base.:(==)(a::RowLayout, b::RowLayout) =
    a.nrows == b.nrows && a.spin_lmax == b.spin_lmax &&
    a.disp_offset == b.disp_offset && a.disp_factors == b.disp_factors &&
    a.disp_starts == b.disp_starts
Base.hash(L::RowLayout, h::UInt) =
    hash(L.disp_starts, hash(L.disp_factors, hash(L.disp_offset,
        hash(L.spin_lmax, hash(L.nrows, h)))))

function Base.show(io::IO, L::RowLayout)
    print(io, "RowLayout(", L.nrows, " rows: spin lmax = ", L.spin_lmax)
    isempty(L.disp_factors) || print(io, ", disp (k,l) = ", L.disp_factors)
    print(io, ")")
end

"""
    row_layout(model::SLCEModel) -> RowLayout
    row_layout(basis::SLCEBasis) -> RowLayout

The per-site basis-row layout a consumer needs to tabulate this model's site
factors: which `(channel, k, l, m)` rows exist, and in what order. See
[`RowLayout`](@ref) for the block structure and [`row_index`](@ref) to address a row.

The layout covers every site factor the **basis** can produce, not merely the ones
with a nonzero coefficient, so it is a property of the model's support and does not
move when coefficients change (a sampler may therefore keep its row tables across a
`set_coefficients!`-style hot swap).
"""
row_layout(model::SLCEModel)::RowLayout = row_layout(model.basis)

function row_layout(basis::SLCEBasis)::RowLayout
    spin_lmax = -1
    dfac = Set{Tuple{Int,Int}}()
    for k in basis.salc_basis.keys
        for d in k.decors
            if has_spin(d)
                spin_lmax = max(spin_lmax, d.spin_l)
            end
            if has_disp(d)
                push!(dfac, (d.disp_k, d.disp_l))
            end
        end
    end
    nspin = spin_lmax < 0 ? 0 : (spin_lmax + 1)^2
    facs = sort!(collect(dfac))
    starts = Vector{Int}(undef, length(facs))
    off = nspin
    for (i, (_, l)) in pairs(facs)
        starts[i] = off
        off += 2l + 1
    end
    return RowLayout(off, spin_lmax, nspin, facs, starts)
end

"""
    row_index(layout::RowLayout, factor::SiteFactor, m::Integer) -> Int

The row a site factor's `m` component occupies in `layout` (1-based).

For a `SPIN` factor this is `SLCE.Harmonics.lm_index(l, m)`; for a `DISP` factor it
is the `(k, l)` block's start plus `m + l + 1`. Throws if the factor is not part of
the layout — that is a basis/consumer mismatch, never something to paper over with a
fallback row.
"""
function row_index(layout::RowLayout, factor::SiteFactor, m::Integer)::Int
    l = factor.l
    -l <= m <= l || throw(ArgumentError("m = $m is out of range for l = $l"))
    if factor.channel === SPIN
        l <= layout.spin_lmax || throw(ArgumentError(
            "the layout's spin block reaches l = $(layout.spin_lmax), but the " *
            "factor has l = $l — the layout was built from a different basis"))
        return Harmonics.lm_index(l, m)
    elseif factor.channel === DISP
        i = findfirst(==((factor.k, l)), layout.disp_factors)
        i === nothing && throw(ArgumentError(
            "the layout has no displacement block for (k, l) = ($(factor.k), $l); " *
            "it carries $(layout.disp_factors) — the layout was built from a " *
            "different basis"))
        return layout.disp_starts[i] + m + l + 1
    end
    throw(ArgumentError("channel $(factor.channel) has no rows in this layout"))
end

"""
    site_rows!(rows, layout::RowLayout, e, u) -> rows

Fill one site's basis-row column: `rows[row_index(layout, factor, m)]` for every
`(factor, m)` the layout carries, with `e` the site's unit spin direction and `u` its
Cartesian displacement.

This is the reference filler — the definition of what each row *contains*, which a
consumer's own (buffered, threaded, device-side) version must reproduce. `rows` must
have at least `layout.nrows` entries.
"""
function site_rows!(rows::AbstractVector{Float64}, layout::RowLayout,
                    e::AbstractVector{<:Real}, u::AbstractVector{<:Real})
    length(rows) >= layout.nrows || throw(DimensionMismatch(
        "rows has $(length(rows)) entries; the layout needs $(layout.nrows)"))
    ev = SVector{3,Float64}(e[1], e[2], e[3])
    @inbounds for l = 0:layout.spin_lmax, m = -l:l
        rows[Harmonics.lm_index(l, m)] = Harmonics.Zlm(l, m, ev)
    end
    isempty(layout.disp_factors) && return rows
    uv = SVector{3,Float64}(u[1], u[2], u[3])
    r2 = dot(uv, uv)
    @inbounds for (i, (k, l)) in pairs(layout.disp_factors)
        r2k = r2^k
        base = layout.disp_starts[i]
        for m = -l:l
            rows[base + m + l + 1] = r2k * SolidHarmonics.Rlm(l, m, uv)
        end
    end
    return rows
end
