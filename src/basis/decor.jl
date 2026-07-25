"""
    Channel

Decoration channel of a [`SiteFactor`](@ref): a `UInt8`-backed enum with the
fixed total order `SPIN < DISP < OCC`. The order is load-bearing — it fixes the
canonical factor order inside a [`SiteDecor`](@ref), the spin-first coupling
order of the joint SALC construction, and the persisted integer codes — so new
channels may only ever be appended.

`OCC` (chemical occupation) is **reserved**: the code is pinned now so a future
occupation channel is additive (design record §10), but constructing an `OCC`
[`SiteFactor`](@ref) is an error in this version.

Public but unexported (the name would shadow `Base.Channel`); reach it as
`SLCE.Channel`, `SLCE.SPIN`, `SLCE.DISP`, `SLCE.OCC`.
"""
@enum Channel::UInt8 SPIN = 0x01 DISP = 0x02 OCC = 0x03

"""
    SiteFactor(channel::Channel, k::Integer, l::Integer)

One decoration factor at a cluster site: the channel plus the radial/angular
labels `(k, l)`.

- `SPIN`: the axial, T-parity `(−1)^l` tesseral factor `Z_{lm}(ê)`. Requires
  `k == 0` (spin directions are unit vectors — no radial degree) and `l ≥ 1`
  (the constant is not a factor).
- `DISP`: the polar, T-even displacement factor `|u|^{2k} R_{lm}(u)` of
  homogeneous degree `2k + l ≥ 1` (`k ≥ 0`, `l ≥ 0`; `l = 0, k ≥ 1` are the
  radial trace channels).
- `OCC`: reserved, unconstructable (see [`Channel`](@ref)).

Degree-0 constants are unconstructable in every channel.
"""
struct SiteFactor
    channel::Channel
    k::Int
    l::Int

    function SiteFactor(channel::Channel, k::Integer, l::Integer)
        if channel == SPIN
            k == 0 || throw(ArgumentError("SPIN factor requires k = 0, got k = $k"))
            l >= 1 || throw(ArgumentError("SPIN factor requires l ≥ 1, got l = $l"))
        elseif channel == DISP
            (k >= 0 && l >= 0) ||
                throw(ArgumentError("DISP factor requires k ≥ 0 and l ≥ 0, " *
                                    "got (k, l) = ($k, $l)"))
            2 * k + l >= 1 ||
                throw(ArgumentError("DISP factor requires degree 2k + l ≥ 1, " *
                                    "got (k, l) = ($k, $l)"))
        else
            throw(ArgumentError("the OCC channel is reserved and cannot be " *
                                "constructed in this version"))
        end
        return new(channel, Int(k), Int(l))
    end
end

_factortuple(f::SiteFactor) = (f.channel, f.k, f.l)
Base.isless(a::SiteFactor, b::SiteFactor) = _factortuple(a) < _factortuple(b)

"""
    SiteDecor(; spin::Integer = 0, disp = nothing)
    SiteDecor(factors::SiteFactor...)

The combined decoration of one cluster site: **at most one factor per channel**
(the `(site, channel)` slot invariant — a site may carry a spin factor AND a
displacement factor, never two of the same channel), at least one factor in
total. `spin = l` attaches `SiteFactor(SPIN, 0, l)`; `disp = (k, l)` attaches
`SiteFactor(DISP, k, l)`.

`SiteDecor`s are `isbits` value labels: [`SALCKey`](@ref) carries a sorted
`Vector{SiteDecor}` as its decoration multiset label (the joint generalization
of the sorted `ls`). The total order is lexicographic in
`(spin_l, disp_k, disp_l)` with 0 = channel absent, so pure-spin decors sort
exactly like the v4 sorted `ls` label.
"""
struct SiteDecor
    spin_l::Int
    disp_k::Int
    disp_l::Int

    function SiteDecor(; spin::Integer = 0,
                       disp::Union{Nothing,Tuple{<:Integer,<:Integer}} = nothing)
        sl = Int(spin)
        sl == 0 || SiteFactor(SPIN, 0, sl)              # validate via the factor rules
        dk, dl = disp === nothing ? (0, 0) : (Int(disp[1]), Int(disp[2]))
        disp === nothing || SiteFactor(DISP, dk, dl)
        sl >= 1 || disp !== nothing ||
            throw(ArgumentError("a SiteDecor needs at least one factor " *
                                "(spin = l and/or disp = (k, l))"))
        return new(sl, dk, dl)
    end
end

function SiteDecor(factors::SiteFactor...)
    isempty(factors) &&
        throw(ArgumentError("a SiteDecor needs at least one factor"))
    spin = 0
    disp = nothing
    for f in factors
        if f.channel == SPIN
            spin == 0 || throw(ArgumentError(
                "at most one factor per channel: two SPIN factors given"))
            spin = f.l
        else # DISP (OCC is unconstructable)
            disp === nothing || throw(ArgumentError(
                "at most one factor per channel: two DISP factors given"))
            disp = (f.k, f.l)
        end
    end
    return SiteDecor(; spin = spin, disp = disp)
end

_decortuple(d::SiteDecor) = (d.spin_l, d.disp_k, d.disp_l)
Base.isless(a::SiteDecor, b::SiteDecor) = _decortuple(a) < _decortuple(b)

"""
    has_spin(d::SiteDecor) -> Bool

Whether the decor carries a `SPIN` factor.
"""
has_spin(d::SiteDecor)::Bool = d.spin_l >= 1

"""
    has_disp(d::SiteDecor) -> Bool

Whether the decor carries a `DISP` factor.
"""
has_disp(d::SiteDecor)::Bool = 2 * d.disp_k + d.disp_l >= 1

"""
    spin_rank(d::SiteDecor) -> Int

Spin rank `l` of the decor's SPIN factor, or `0` when it has none.
"""
spin_rank(d::SiteDecor)::Int = d.spin_l

"""
    disp_degree(d::SiteDecor) -> Int

Homogeneous displacement degree `2k + l` of the decor's DISP factor
(`0` when it has none).
"""
disp_degree(d::SiteDecor)::Int = 2 * d.disp_k + d.disp_l

"""
    factors(d::SiteDecor) -> Vector{SiteFactor}

The decor's factors in canonical channel order (`SPIN` first).
"""
function factors(d::SiteDecor)::Vector{SiteFactor}
    out = SiteFactor[]
    has_spin(d) && push!(out, SiteFactor(SPIN, 0, d.spin_l))
    has_disp(d) && push!(out, SiteFactor(DISP, d.disp_k, d.disp_l))
    return out
end

"""
    is_pure_spin(d::SiteDecor) -> Bool

Whether the decor is a bare spin factor (no displacement factor) — the v4
(spin-only) decoration shape.
"""
is_pure_spin(d::SiteDecor)::Bool = has_spin(d) && !has_disp(d)

"""
    spin_decors(ls) -> Vector{SiteDecor}

The pure-spin decoration vector of a per-site `l` list (the v4 → v5 label map:
each `l` becomes `SiteFactor(SPIN, 0, l)` alone on its site). Preserves order,
so a sorted `ls` maps to a sorted decor label.
"""
spin_decors(ls::AbstractVector{<:Integer})::Vector{SiteDecor} =
    SiteDecor[SiteDecor(; spin = l) for l in ls]

"""
    rep_scale(channel::Channel, detR::Real, l::Integer) -> Float64

The axial/polar scalar relating a channel's O(3) representation to the polar
real Wigner matrix: `D_channel(l, R) = rep_scale(channel, det R, l) · D_polar(l, R)`.
`SPIN` is axial (`det(R)^l` — spin directions are pseudovectors), `DISP` polar
(`1.0`); `OCC` is reserved (its representation is not declared yet) and throws.

This is the **single declared source of the per-channel group action** (design
record §4). Production projection never applies it: the even-`Σl_spin`
enumeration screen makes the product over spin slots `det(R)^{Σl_spin} ≡ +1`,
so the one polar Wigner cache is exactly correct for both channels. The seat
exists for the verification layer — the independent oracle and the gate (o)
representation pins (e.g. inversion: axial `+I` for every `l`, polar
`(−1)^l I`) consume it, and a mutation reinstating a global `det(R)^{Σl_all}`
rule must fail against it.
"""
function rep_scale(channel::Channel, detR::Real, l::Integer)::Float64
    channel == OCC &&
        throw(ArgumentError("the OCC channel is reserved — its representation " *
                            "is not declared in this version"))
    return channel == SPIN ? Float64(detR)^Int(l) : 1.0
end

function Base.show(io::IO, d::SiteDecor)
    parts = String[]
    has_spin(d) && push!(parts, "spin=$(d.spin_l)")
    has_disp(d) && push!(parts, "disp=($(d.disp_k),$(d.disp_l))")
    print(io, "SiteDecor(", join(parts, ", "), ")")
end
