# Spin directions as a type — the invariant that used to be a rule every door had to
# remember (audit 2026-08-01).
#
# "This is a unit vector" is an invariant with no representation: it lived only as a
# rule that each entry point had to enforce, at independently chosen tolerances. So the
# audited bug was not an oversight, it was structural — `_resolve_spins` was the eighth
# door and dropped both checks the other seven shared — and nothing downstream can
# catch that class of mistake: the acoustic modes of `D(0)` and `asr_residual` are BLIND
# to an off-unit spin state, because the ASR's rows are keyed per spin monomial, so
# `Aβ = 0` is an identity in `e` that a wrong `e` satisfies too.
#
# So the rule lives here, once, and a type carries it. A function taking a
# `UnitVector3` has no policy to forget, and `Harmonics.Zlm_unsafe`'s `‖u‖ = 1`
# precondition is discharged by the argument's type rather than hoped for.
#
# TWO CONSTRUCTORS, and the split is the whole design. Accepting a value from a user is
# not the same operation as reconstructing one that an evolution already produced:
#
#   * `UnitVector3(v)` — validate, then project. For data entering the package.
#   * `UnitVector3(v, Trusted())` — validate, keep the bits. For a value already on the
#     sphere.
#
# The second is not a loophole, it is a correctness requirement found downstream:
# `normalize` is NOT bitwise idempotent (measured, 37.64 % of 200 000 already-unit
# directions move by up to 4.4e-16 under a second application), and SLCEDynamics'
# default integrator advances spins by a rigid Rodrigues rotation that preserves
# `|e| = 1` to rounding with no renormalization step. Re-projecting such a value on
# checkpoint resume applies an operation the uninterrupted run never applied, which
# forks a chaotic trajectory — which is exactly why `SLCEDynamics._config_verbatim`
# refuses to normalize. That refusal is a serialization-identity requirement, not a
# tolerance, and it belongs in the type as a second door rather than as a bypass.
#
# WHAT THIS DOES NOT PROMISE. The guarantee is established at construction, and the
# sampling packages mutate working state in place at a granularity no wrapper reaches
# without entering their hot loops. Working buffers therefore stay raw — they are
# points being retracted onto a manifold, and the moves themselves keep them there —
# and the type sits at the transport boundary, where `to_matrix`/`from_matrix` already
# are. The honest claim: this reduces the places that must know the policy from ~40
# kernel call sites to a handful of named doors, and makes "no magnetic state"
# type-checkable. It does not make the invariant true by construction family-wide.

# The default band. Public through `SLCEDataset(...; atol = ...)` and friends, and with
# the projection in place widening it no longer degrades any number — the harmonics
# error budget it used to buy is zero once the stored value is exactly unit, so `atol`
# is now purely "how far from unit may a thing be and still be called this direction".
# There is real demand for widening it: a MAGMOM written to four decimals
# (`0.5774 0.5774 0.5774`) is ~2e-5 off unit, which is a file format, not a bug.
const _DIRECTION_ATOL = 1.0e-6

# ... but not without limit, or a genuinely wrong vector is silently projected onto the
# sphere and handed back as a direction. This cap is the line between "the trace of a
# rounded decimal" and "a different vector". A caller who means to accept something
# past it should normalize deliberately, in their own code, where it is visible.
const _DIRECTION_ATOL_MAX = 1.0e-2

"""
    Trusted

Marker selecting the [`UnitVector3`](@ref) / [`SpinConfiguration`](@ref) constructors
that validate **without projecting**, preserving the caller's bits exactly.

Use it when the value is already on the sphere because something put it there: a
checkpoint being restored, or a state an integrator advanced by a norm-preserving
rotation. It does not skip validation — both constructors check.

The distinction matters because `v / norm(v)` is not bitwise idempotent, so projecting
a restored state applies an operation the original run never applied — enough to fork a
chaotic trajectory and break a bit-identical resume.
"""
struct Trusted end

function _as_direction(v::AbstractVector{<:Real}, what::AbstractString)
    length(v) == 3 || throw(DimensionMismatch(
        "$what must have 3 components (got $(length(v)))"))
    return SVector{3,Float64}(v[1], v[2], v[3])
end

function _check_atol(atol::Real)
    (isfinite(atol) && atol >= 0) ||
        throw(ArgumentError("atol must be finite and nonnegative (got $atol)"))
    atol <= _DIRECTION_ATOL_MAX || throw(ArgumentError(
        "atol = $atol exceeds the hard cap $(_DIRECTION_ATOL_MAX): past it a vector " *
        "that is not a direction at all — a moment-scaled one, a wrongly rotated " *
        "one — would be accepted and silently projected onto the sphere. Normalize " *
        "deliberately in your own code instead, where the decision is visible"))
    return nothing
end

# "Is this a direction at all?" — finite, and near enough to unit that the caller is
# plausibly describing a direction rather than some other vector. This is the half that
# a projecting door asks BEFORE projecting, because it is the half projection must not
# be allowed to paper over.
function _validate_is_direction(u::SVector{3,Float64}, what::AbstractString;
                                atol::Real = _DIRECTION_ATOL)
    all(isfinite, u) ||
        throw(ArgumentError("$what is not finite ($(Tuple(u)))"))
    abs(norm(u) - 1) <= atol ||
        throw(ArgumentError("$what is not a unit vector (‖e‖ = $(norm(u)))"))
    return nothing
end

# The package's ONE unit-direction rule: the above, plus the kernels' hard
# precondition. Every NON-projecting door states it by calling this; nothing below a
# door re-states it. `what` names the offending thing (an argument, or a column of one)
# and is interpolated as written.
function _validate_direction(u::SVector{3,Float64}, what::AbstractString;
                             atol::Real = _DIRECTION_ATOL)
    _validate_is_direction(u, what; atol = atol)
    # NOT redundant with the band, and the half that matters most: the harmonic kernels
    # reach `LegendrePolynomials.dnPl`, whose domain is `|z| ≤ 1`, and near a pole any
    # norm error can push a component past it — measured, a direction `5e-9` off unit
    # clears even a `1e-8` band and still throws a bare `DomainError` from inside an
    # accumulation. No tolerance establishes this precondition; only this bound does,
    # and it rejects nothing harmless (off norm by `1e-7` with largest component `0.8`
    # passes).
    maximum(abs, u) <= 1 || throw(ArgumentError(
        "$what has a component outside [-1, 1] " *
        "($(Tuple(u)), ‖e‖ = $(norm(u))) — the harmonic kernel's Legendre " *
        "recursion is defined only for |e_z| ≤ 1, so this would throw from inside " *
        "the design assembly. Normalize the column (`u ./= norm(u)`) rather than " *
        "widening `atol`, which cannot fix it"))
    return nothing
end

"""
    UnitVector3(v; atol = 1e-6)             -> UnitVector3
    UnitVector3(v, Trusted(); atol = 1e-6)  -> UnitVector3

A Cartesian direction that is a unit vector by construction.

The plain constructor validates `v` and then projects it onto the sphere, so the result
satisfies `‖u‖ = 1` to rounding whatever float noise the caller's value carried. The
[`Trusted`](@ref) constructor validates and keeps the caller's bits — for a value an
evolution already placed on the sphere.

Validation is the package's one rule: finite components, `|‖v‖ − 1| ≤ atol`, and
`max|component| ≤ 1`. The component bound is not redundant with the band and is the one
that establishes the harmonic kernels' hard precondition; see the note in
[`SpinConfiguration`](@ref). `atol` is capped at `1e-2`.

Indexing and iteration forward to the underlying vector, so a `UnitVector3` is usable
anywhere a 3-element `AbstractVector` is.
"""
struct UnitVector3 <: AbstractVector{Float64}
    v::SVector{3,Float64}

    function UnitVector3(v::AbstractVector{<:Real}; atol::Real = _DIRECTION_ATOL,
                         what::AbstractString = "direction")
        _check_atol(atol)
        u = _as_direction(v, what)
        # Ask "is this a direction?" BEFORE projecting — that is the half projection
        # must not paper over. It rejects a value that is a different physical quantity
        # wearing the wrong units (a moment-scaled vector rather than a direction);
        # ordered moments sit nowhere near any admissible band around 1, so that mixup
        # still throws loudly. What projection then removes is only the float noise the
        # caller has already asserted, by choosing this door, is noise.
        _validate_is_direction(u, what; atol = atol)
        w = u / norm(u)
        # The component bound is a precondition on the value the KERNELS will see, so
        # it is asked of the projected result, not of the input. Order matters: a
        # direction `5e-9` off unit norm at the pole has `|z| > 1` before projection
        # and `|z| == 1.0` exactly after it, so asking first would reject the very case
        # projection exists to repair — while asking after still catches anything that
        # survives. Belt and braces: measured over 5·10⁶ fuzzed directions including
        # near-pole and extreme-aspect cases, projection never left a component
        # outside [-1, 1], so this is expected never to fire and costs one comparison
        # at a door that is not on any hot path.
        _validate_direction(w, what; atol = atol)
        return new(w)
    end

    function UnitVector3(v::AbstractVector{<:Real}, ::Trusted;
                         atol::Real = _DIRECTION_ATOL,
                         what::AbstractString = "direction")
        _check_atol(atol)
        u = _as_direction(v, what)
        _validate_direction(u, what; atol = atol)
        return new(u)
    end
end

Base.size(::UnitVector3) = (3,)
Base.IndexStyle(::Type{UnitVector3}) = IndexLinear()
Base.@propagate_inbounds Base.getindex(u::UnitVector3, i::Int) = u.v[i]
Base.convert(::Type{SVector{3,Float64}}, u::UnitVector3) = u.v
Base.show(io::IO, u::UnitVector3) =
    print(io, "UnitVector3(", u.v[1], ", ", u.v[2], ", ", u.v[3], ")")
Base.show(io::IO, ::MIME"text/plain", u::UnitVector3) = show(io, u)

"""
    SpinConfiguration(m; atol = 1e-6)             -> SpinConfiguration
    SpinConfiguration(m, Trusted(); atol = 1e-6)  -> SpinConfiguration

A magnetic state: `3 × n_atoms`, one unit column per atom, unit **by construction**.

The `3 × n` layout is the package's spin convention and is preserved inside the
wrapper, so serialization, the design-matrix assembly and the device-side flat arrays
see the representation they always did; `Matrix(config)` hands it back. What the type
adds is that there is no way to obtain one whose columns are not unit directions.

The plain constructor validates every column and then projects it; the
[`Trusted`](@ref) one validates and keeps the caller's bits (see [`Trusted`](@ref) for
why that distinction is a correctness requirement and not an optimization).

!!! note "The component bound is the load-bearing check, not the tolerance"
    Validation is finite components, `|‖e‖ − 1| ≤ atol`, and `max|component| ≤ 1`. The
    last is not implied by the first: the harmonic kernels reach
    `LegendrePolynomials.dnPl`, whose domain is `|z| ≤ 1`, and near a pole any norm
    error can push a component past it — measured, a column `5e-9` off unit norm clears
    even a `1e-8` band and still throws a bare `DomainError` from inside the design
    assembly. Tightening `atol` cannot establish that precondition; only the component
    bound can.

Indexing forwards to the matrix, and `n_atoms` gives the column count.
"""
struct SpinConfiguration <: AbstractMatrix{Float64}
    m::Matrix{Float64}

    function SpinConfiguration(m::AbstractMatrix{<:Real};
                               atol::Real = _DIRECTION_ATOL,
                               label::AbstractString = "spin config")
        _check_atol(atol)
        size(m, 1) == 3 ||
            throw(ArgumentError("$label must have 3 rows (got $(size(m, 1)))"))
        out = Matrix{Float64}(undef, 3, size(m, 2))
        @inbounds for a in axes(m, 2)
            u = UnitVector3(SVector{3,Float64}(m[1, a], m[2, a], m[3, a]);
                            atol = atol, what = "$label column $a")
            out[1, a], out[2, a], out[3, a] = u[1], u[2], u[3]
        end
        return new(out)
    end

    function SpinConfiguration(m::AbstractMatrix{<:Real}, ::Trusted;
                               atol::Real = _DIRECTION_ATOL,
                               label::AbstractString = "spin config")
        _check_atol(atol)
        size(m, 1) == 3 ||
            throw(ArgumentError("$label must have 3 rows (got $(size(m, 1)))"))
        out = Matrix{Float64}(m)
        @inbounds for a in axes(out, 2)
            _validate_direction(SVector{3,Float64}(out[1, a], out[2, a], out[3, a]),
                                "$label column $a"; atol = atol)
        end
        return new(out)
    end
end

SpinConfiguration(c::SpinConfiguration; kwargs...) = c
SpinConfiguration(c::SpinConfiguration, ::Trusted; kwargs...) = c

Base.size(c::SpinConfiguration) = size(c.m)
Base.IndexStyle(::Type{SpinConfiguration}) = IndexCartesian()
Base.@propagate_inbounds Base.getindex(c::SpinConfiguration, i::Int, j::Int) = c.m[i, j]
Base.Matrix(c::SpinConfiguration) = c.m
Base.convert(::Type{Matrix{Float64}}, c::SpinConfiguration) = c.m
n_atoms(c::SpinConfiguration)::Int = size(c.m, 2)

function Base.show(io::IO, c::SpinConfiguration)
    print(io, "SpinConfiguration(", n_atoms(c), " atoms)")
end
Base.show(io::IO, ::MIME"text/plain", c::SpinConfiguration) = show(io, c)
