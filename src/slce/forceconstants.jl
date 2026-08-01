# Force constants and the dynamical matrix — the physics deliverables of the
# displacement channel (design record §7).
#
# A spin–lattice model's force constants are FUNCTIONS OF THE SPIN CONFIGURATION:
# that is the entire point of the joint expansion. `force_constants(model; spins,
# order)` differentiates the energy `order` times with respect to displacements at
# `u = 0`, with the spins held at the given configuration.
#
# Why this is exact rather than a finite difference. Every displacement site factor
# `|u|^{2k} R_{lm}(u)` is a homogeneous polynomial of degree `2k + l` (that is the
# `SolidHarmonics` kernel's whole reason for existing — regular at `u = 0`, which is
# the densest sampling point of a reference-structure expansion). So a term
# contributes to the order-`n` constants only if its displacement degrees sum to
# exactly `n`, and its contribution is read off the monomial coefficients — no step
# size, no truncation error. `solid_harmonic_poly` returns those coefficients, and it
# is the same function the ASR builder uses, so the two cannot drift.
#
# Indexing follows the lattice-dynamics convention, NOT the SALC-member one:
# `Φ[(a, 0), (b, R), …]` is `∂ⁿE_cell / ∂u_a(0) ∂u_b(R) …`, one entry per ORDERED
# index tuple, anchored so the first index sits in the home cell. The reverse
# ordering is a separate key related by `Φ[(a,0),(b,R)] = Φ[(b,0),(a,−R)]ᵀ` — a
# property to check, never a storage trick.

"""
    ForceConstantSet

Force constants of a [`SLCEModel`](@ref) at one spin configuration, as produced by
[`force_constants`](@ref).

`constants` maps an anchored index tuple `(atoms, shifts)` — both of length `order`,
with `shifts[1]` the zero vector — to the rank-`order` Cartesian tensor

```
Φ[α₁, …, α_order] = ∂^order E_cell / ∂u_{atoms[1] α₁}(shifts[1]) ⋯
```

evaluated at `u = 0` with the spins fixed at `spins`. Entries are per training cell,
in the model's energy unit per (length unit)^`order`. Every ordered tuple that carries
a nonzero tensor is present, so `Φ[(a,0),(b,R)]` and `Φ[(b,0),(a,−R)]` are separate
keys (transposes of each other).

See [`dynamical_matrix`](@ref) for the order-2 reciprocal-space form.
"""
struct ForceConstantSet
    order::Int
    crystal::Crystal
    spins::Union{SpinConfiguration,Nothing}
    constants::Dict{Tuple{Vector{Int},Vector{SVector{3,Int}}},Array{Float64}}
end

Base.length(fcs::ForceConstantSet) = length(fcs.constants)

function Base.show(io::IO, fcs::ForceConstantSet)
    print(io, "ForceConstantSet(order = ", fcs.order, ", ",
          length(fcs.constants), " index tuples, ",
          n_atoms(fcs.crystal), " atoms)")
end

# Resolve the magnetic state a derivative deliverable is evaluated at. Shared by
# `force_constants` and `strain_derivatives` (`slce/strain.jl`) because the rule is one
# rule: omitting the spin state is legal exactly when no spin factor exists to
# evaluate. The predicate is `_basis_has_spin` (spec ∪ surviving SALCs) and NOT
# `is_soc_free`, which asks whether `L_S == 0` — true for most ordinary `soc = false`
# spin labels, so it would wave a spin-carrying model through and silently evaluate it
# at the all-zero state.
function _resolve_spins(model::SLCEModel,
                        spins::Union{AbstractMatrix{<:Real},Nothing},
                        what::AbstractString)::Union{SpinConfiguration,Nothing}
    nat = n_atoms(model.basis.crystal)
    if spins === nothing
        _basis_has_spin(model.basis) && throw(ArgumentError(
            "`spins` is required: this model's basis carries spin content, so $what " *
            "depend on the magnetic state. Pass the 3 × $nat unit directions to " *
            "evaluate at."))
        # `nothing`, not a fabricated all-zero matrix. The zero marker existed only
        # because the field's type was `Matrix{Float64}` and there was no way to SAY
        # "no magnetic state"; with `Union{SpinConfiguration,Nothing}` there is, so
        # the absence is representable instead of encoded. That is what makes
        # `write_alamode`'s cross-set key honest: two lattice-only sets now agree
        # because both carry `nothing`, not because both fabricated the same zeros.
        return nothing
    end
    size(spins) == (3, nat) || throw(DimensionMismatch(
        "spins is $(size(spins)); expected (3, $nat) — one unit column per atom"))
    e = Matrix{Float64}(spins)
    # Validate through the SAME function the dataset boundary uses, at the SAME
    # `atol`, so a magnetic state that builds a dataset and fits also evaluates the
    # readouts. This used to check the shape and nothing else, which broke in both
    # directions. Too strict: everything behind here calls the CHECKED `Zlm` /
    # `grad_Zlm`, whose band is `1e-8`, so a column off unit norm by `5e-7` passed
    # `_validate_config`, fitted, predicted — and then died inside the readout with
    # "direction must be a unit vector", naming neither the argument nor the atom.
    # Too lax: `‖e‖ = 1.7` succeeded silently whenever the requested order carried
    # no spin-dressed term, and the bogus vector was stored in
    # `ForceConstantSet.spins`, which `write_alamode` uses as its cross-set key.
    # `_validate_config`'s component bound is the half that matters most here: the
    # norm band CANNOT establish the kernel's hard precondition (`dnPl` needs
    # `|e_z| ≤ 1`, and near a pole any `δ > 0` can push past it — measured, a column
    # `5e-9` off norm passes a `1e-8` band and still throws a `DomainError` from
    # inside the accumulation), so tightening a tolerance was never the fix.
    #
    # Unconditional, unlike `_validate_config_pair`'s `_basis_has_spin` guard: the
    # spin-free entry path now returns `nothing` above rather than an all-zero matrix,
    # so there is no longer a legal non-unit configuration to make an exception for.
    # Passing a REAL magnetic state to a lattice-only model stays legal — it is simply
    # not read — and it is a genuine configuration, so it validates.
    return SpinConfiguration(e; label = "`spins` (evaluating $what)")
end

# The kernels take a plain `3 × n_atoms` array. A spin-free model has no spin slot for
# them to read, so this filler is unreachable by construction — it exists to keep the
# accumulation monomorphic, and it is a LOCAL: nothing stores it, and the result
# struct records the absence as `nothing`.
_spin_kernel_matrix(e::SpinConfiguration, ::Int)::Matrix{Float64} = Matrix(e)
_spin_kernel_matrix(::Nothing, nat::Int)::Matrix{Float64} = zeros(Float64, 3, nat)

# Derivative of one site's displacement factor at u = 0: the site factor is the
# homogeneous polynomial `|u|^{2k} R_{lm}(u)` of degree `2k + l`, so differentiating
# it `length(comps)` times at the origin is nonzero only when that count equals the
# degree, and is then the matching monomial's coefficient times ∏ exponent!.
function _site_derivative(poly, comps)::Float64
    ex = MVector(0, 0, 0)
    for c in comps
        ex[c] += 1
    end
    key = (ex[1], ex[2], ex[3])
    c = get(poly, key, 0.0)
    c == 0.0 && return 0.0
    return c * factorial(ex[1]) * factorial(ex[2]) * factorial(ex[3])
end

"""
    force_constants(model::SLCEModel; spins = nothing, order = 2) -> ForceConstantSet

The order-`order` force constants of `model` at the spin configuration `spins`
(`3 × n_atoms`, unit columns): the exact `order`-th derivatives of the energy with
respect to atomic displacements at `u = 0`.

```
Φ_{a₁α₁ … }(0, R₂, …) = ∂^order E_cell / ∂u_{a₁α₁}(0) ∂u_{a₂α₂}(R₂) ⋯ |_{u = 0}
```

`order = 2` gives the harmonic force constants (feed them to
[`dynamical_matrix`](@ref)); `order = 3` the cubic anharmonic ones, and so on.

!!! note "`order = 1` is the energy gradient, i.e. MINUS the force"
    `order = 1` is legal and returns `∂E/∂u` at `u = 0` — the residual
    (Hellmann–Feynman) gradient of the reference structure. It is the negative of
    [`predict_force`](@ref), which returns `f = −∂E/∂u`:
    `Σ_shifts Φ⁽¹⁾ ≈ −predict_force(model, spins, zeros(3, n_atoms))`. Sign
    conventions for this quantity live in exactly two other places in the package, so
    the relation is gated rather than merely asserted here.

They
are exact, not finite differences — every displacement factor is a polynomial whose
coefficients `SLCE.SolidHarmonics.solid_harmonic_poly` returns — and they
**depend on `spins`**, which is what a spin–lattice expansion is for: evaluate at two
magnetic states and the difference is the magnetic contribution to the lattice
dynamics.

A pure-spin model has no displacement content and yields an empty set. Only terms
whose displacement degrees sum to exactly `order` contribute; a model truncated below
`order` therefore returns fewer (or no) constants rather than an error.

!!! note "No same-atom off-site block, at any cutoff"
    Clusters are enumerated between **distinct atoms of the reference cell**, so an atom
    paired with its own periodic image is never admitted and
    `Φ[(a,0),(a,R≠0)]` is absent from the result by construction — as is the
    `q`-dependence it would give [`dynamical_matrix`](@ref). A same-sublattice force
    constant exists only when the two atoms are separate atoms of the cell the basis was
    built on; a one-atom cell carries no pair content at all (`build_asr` warns, and
    `D(q) ≡ 0`). This costs the *fit* nothing — a datum's displacement field is
    cell-periodic, so both ends of such a pair move together — and it is not a cutoff
    question. See the `Periodic resolvability` chapter of the manual.

!!! warning "A frozen channel is missing from the result, and the result says so"
    The other way a reference cell limits this read-out. A Wigner–Seitz boundary tie can
    make a whole SALC identically zero on every configuration the cell can express
    ([`unresolvable_columns`](@ref)), so a fit holds its coefficient at exactly zero —
    while these constants *would* carry it, because they differentiate the individual
    cluster members, where the tie does not cancel. `force_constants` therefore says so
    once: with a zero coefficient the result is **missing that channel** (which is not
    zero physics — the same functions are nonzero in a supercell), and with a nonzero
    one (hand-built, or fixed from elsewhere) the value reaches `Φ` and every `q ≠ 0` of
    [`dynamical_matrix`](@ref) although no data on this cell could have determined it.
    Either way it is invisible at `q = 0`, because `Σ_R Φ(R)` is the Hessian of exactly
    the energy this cell can express. The remedy is a cell in which the offending pair's
    minimum image is unique, not a wider cutoff.

`spins` may be omitted only for a model whose basis carries **no spin content at
all** — a lattice-only expansion, where there is nothing for a spin state to feed
and inventing one would be noise. Omitting it against a spin-carrying basis is an
error, never a default state.

!!! note "Magnetic symmetry: which basis you fit decides which group is imposed"
    The returned constants are invariant under the **magnetic space group** of
    `spins` — *including* its antiunitary elements, whose rotation parts constrain
    `Φ` because force constants are time-reversal even. Nothing has to be declared:
    the SALCs are projected with the paramagnetic grey group `G × {1, T}`, and fixing
    `spins` cuts that down to the stabilizer of the magnetic state automatically. On a
    stripe-AFM fixture the joint path spans exactly the 12 symmetry-allowed Γ-Hessian
    parameters, with the unitary *and* the antiunitary elements satisfied to 1.8e-15
    (`test/unit/test_forceconstants.jl`).

    There are two ways to lose that, and both are silent unless you look:

    - **A lattice-only basis** (no spin content anywhere) is projected with the
      *paramagnetic* group, which for an ordered magnetic state is too large — the
      components the order breaks are set to exactly zero. The same fixture allows
      only 7 of the 12. Relabelling the sublattices as distinct species (`"Fe_up"` /
      `"Fe_dn"`) overshoots the other way: it keeps the unitary subgroup and drops the
      antiunitary elements, admitting 16. Only the joint path lands on 12.
    - **A joint basis with no spin-carrying term at this `order`.** A magnetoelastic
      sector declared `disp = (degree = 1,)` contributes to the *forces*, not to the
      harmonic constants; those need `degree = 2` under a spin-carrying sector. Without
      one, `Φ` comes out bit-identical for every `spins` — this function warns when the
      basis is shaped that way.

!!! note "Translation invariance is not automatic"
    The acoustic sum rule `Σ_{b,R} Φ_{aα,bβ}(R) = 0` holds only if the model
    satisfies it — fit with `asr = true` (the default) or check
    [`asr_residual`](@ref). Constants from a violating model give acoustic modes with
    nonzero frequency at `q = 0`.
"""
function force_constants(model::SLCEModel;
                         spins::Union{AbstractMatrix{<:Real},Nothing} = nothing,
                         order::Integer = 2)::ForceConstantSet
    order >= 1 || throw(ArgumentError("order must be ≥ 1; got $order"))
    e = _resolve_spins(model, spins, "the force constants")
    espin = _spin_kernel_matrix(e, n_atoms(model.basis.crystal))
    _warn_spin_blind(model.basis, Int(order))
    _warn_unresolvable(model, "force_constants")
    out = Dict{Tuple{Vector{Int},Vector{SVector{3,Int}}},Array{Float64}}()
    polycache = Dict{NTuple{3,Int},SolidHarmonics._Poly}()
    salcs = model.basis.salc_basis.salcs
    for k in eachindex(model.jphi)
        jphi = model.jphi[k]
        jphi == 0.0 && continue
        salc = salcs[k]
        scale = (4π)^(count(has_spin, salc.decors) / 2)
        for mem in salc.members
            for t in mem.terms
                _accumulate_fcs!(out, jphi * scale, t, mem, espin, Int(order), polycache)
            end
        end
    end
    # Drop index tuples that cancelled to zero. Collect the keys FIRST: deleting from
    # a Dict while iterating it is undefined behavior in Julia, not merely untidy.
    for key in [k for (k, T) in out if all(iszero, T)]
        delete!(out, key)
    end
    return ForceConstantSet(Int(order), model.basis.crystal, e, out)
end

# The trap this catches: a basis that carries spin content AND displacement terms at
# `order`, but no term carrying BOTH. The constants are then bit-identical for every
# `spins` — a "magnetic" phonon calculation that is not magnetic at all — and the
# usual way to land here is declaring the magnetoelastic sector at `disp =
# (degree = 1,)`, which feeds the forces rather than the harmonic constants.
#
# Read off the BASIS, never off `jphi`: a coefficient that happens to have fitted to
# zero is a property of one fit and `refit` moves it, whereas an empty channel is
# permanent. Same reason `spin_multipole_terms` triggers on the spec.
#
# The predicate is shared with `strain_derivatives` (`slce/strain.jl`), which hits the
# identical trap one order lower; only the advice differs, so only the message is
# written twice.
function _spin_blind_at_order(basis::SLCEBasis, order::Int)::Bool
    any_spin = false
    disp_at_order = false
    for salc in basis.salc_basis.salcs
        any_spin |= any(has_spin, salc.decors)
        for mem in salc.members, t in mem.terms
            deg = 0
            spinful = false
            for s in t.slots
                if s.factor.channel == DISP
                    deg += 2 * s.factor.k + s.factor.l
                else
                    spinful = true
                end
            end
            deg == order || continue
            spinful && return false              # a spin-dressed term at this order
            disp_at_order = true
        end
    end
    return any_spin && disp_at_order
end

function _warn_spin_blind(basis::SLCEBasis, order::Int)
    if _spin_blind_at_order(basis, order)
        @warn "force_constants: order-$order constants do not depend on `spins` — the " *
              "basis has spin content and displacement terms of degree $order, but no " *
              "term carries both, so Φ (and D(q)) is identical for every magnetic " *
              "state. A magnetoelastic sector at `disp = (degree = 1,)` contributes " *
              "to the forces; harmonic constants need `degree = $order` under a " *
              "spin-carrying sector." maxlog = 1
    end
    return nothing
end

# The freeze is silent in the fit and LOUD here, which is the wrong way round unless
# somebody says it: a column this reference cell cannot resolve
# ([`unresolvable_columns`](@ref)) is identically zero in the energy of every
# configuration the cell can express — so a fit holds it at exactly zero — but these
# readouts differentiate the individual cluster MEMBERS, where the tie does not cancel.
# Both directions therefore matter and are reported separately:
#
#   * coefficient zero — the deliverable is missing that channel, and the value is not
#     physically zero (it is nonzero in a supercell). Advice is a different reference
#     cell, never a wider cutoff (the WS corner is representable).
#   * coefficient nonzero — a hand-built or externally fixed value, which no data on
#     this cell can have determined, and which DOES reach Φ and every `q ≠ 0` of `D(q)`.
#     Legal, unvalidated, and worth naming: it is the one way a plausible-looking
#     dispersion can come out of a channel the training data never saw.
#
# Shared by every readout that reads the monomials rather than the cell-periodic
# function (`force_constants`, `strain_derivatives`, `exchange_strain_derivatives`,
# `magnon_phonon_vertices`, `decorated_terms` — the Monte-Carlo hand-off, where the
# supercell resolves the tie and the missing channel becomes real physics;
# `magnetoelastic_constants` inherits it through `strain_derivatives`). Cheap on an untied
# basis: the classifier short-circuits on a structural pre-check.
#
# **`maxlog` is keyed per READOUT, via `_id`, and that is load-bearing.** Julia keys
# `maxlog` by the log STATEMENT, so one `@warn` shared by six readouts at `maxlog = 1`
# says it for whichever ran first and silences the other five — measured: on a B2 FeRh
# joint basis `strain_derivatives` warned and `magnetoelastic_constants`,
# `magnon_phonon_vertices`, `decorated_terms` and `force_constants` all returned their
# frozen-channel results in silence. `magnetoelastic.jl` already records the same trap for
# its residual warning. The `what` string is therefore part of the log id, so each
# deliverable says it once and no deliverable is muted by another.
function _warn_unresolvable(model::SLCEModel, what::AbstractString)
    frozen = try
        unresolvable_columns(model.basis)
    catch err
        err isa UnclassifiableBasis || rethrow()
        return nothing            # said once by `build_asr`; not this path's business
    end
    isempty(frozen) && return nothing
    supplied = [j for j in frozen if model.jphi[j] != 0.0]
    absent = [j for j in frozen if model.jphi[j] == 0.0]
    isempty(absent) ||
        @warn "$what: this reference cell cannot resolve $(length(absent)) of its " *
              "$(n_salcs(model.basis)) channels, so their coefficients are held at " *
              "exactly zero and $what carries NO contribution from them. They are not " *
              "physically zero — the cause is a Wigner-Seitz boundary tie, the same " *
              "basis functions are nonzero in a supercell, and the remedy is a cell in " *
              "which the offending pair's minimum image is unique (NOT a wider cutoff). " *
              "`unresolvable_columns` lists them; the Periodic resolvability chapter " *
              "has the argument" columns = first(absent, 10) maxlog = 1 _id =
            Symbol(:slce_unresolvable_absent_, what)
    isempty(supplied) ||
        @warn "$what: the model carries a NONZERO coefficient on $(length(supplied)) " *
              "column(s) this reference cell cannot resolve, and $what does see them: " *
              "these readouts differentiate the individual cluster members, where the " *
              "tie does not cancel, so the value reaches Φ and every q ≠ 0 of D(q) even " *
              "though it is identically zero in the energy of any configuration this " *
              "cell can express. No data on this cell can have determined it. Legal — a " *
              "hand-built or externally fixed coefficient — but nothing here validates " *
              "it" columns = first(supplied, 10) maxlog = 1 _id =
            Symbol(:slce_unresolvable_supplied_, what)
    return nothing
end

function _accumulate_fcs!(out, weight::Float64, t::SALCTerm, mem::SALCMember,
                          e::Matrix{Float64}, order::Int, polycache)
    dslots = [i for i in eachindex(t.slots) if t.slots[i].factor.channel == DISP]
    isempty(dslots) && return nothing
    # Only a term whose displacement degrees sum to `order` survives differentiation
    # at u = 0: below it the derivative kills the term, above it every surviving
    # monomial still carries a positive power of u.
    deg = sum(2 * t.slots[i].factor.k + t.slots[i].factor.l for i in dslots)
    deg == order || return nothing
    # Ordered derivative tuples over the displacement-carrying member sites. The
    # per-site derivative count must match that site's degree exactly, so tuples that
    # cannot are skipped before any polynomial work.
    dsites = [t.slots[i].site for i in dslots]
    degof = Dict(t.slots[i].site => 2 * t.slots[i].factor.k + t.slots[i].factor.l
                 for i in dslots)
    for choice in Iterators.product(ntuple(_ -> dsites, order)...)
        counts = Dict{Int,Int}()
        for s in choice
            counts[s] = get(counts, s, 0) + 1
        end
        all(degof[s] == get(counts, s, 0) for s in dsites) || continue
        key = _fcs_key(mem, choice)
        T = get!(out, key) do
            zeros(Float64, ntuple(_ -> 3, order))
        end
        _fill_fcs_tensor!(T, weight, t, dslots, choice, mem, e, polycache)
    end
    return nothing
end

# The anchored lattice-dynamics key of one ordered site tuple: atoms in the tuple's
# order, shifts translated so the first index sits in the home cell.
function _fcs_key(mem::SALCMember, choice)
    atoms = [mem.atoms[s] for s in choice]
    s0 = mem.shifts[first(choice)]
    shifts = [mem.shifts[s] - s0 for s in choice]
    return (atoms, shifts)
end

# One term's contribution to one ordered index tuple's tensor. The term's tensor sum
# factorizes: each `folded` index fixes an `m` per axis, the spin axes evaluate to a
# number at the given configuration, and the displacement axes contribute the
# derivative of their own polynomial — the whole product then splits site by site.
function _fill_fcs_tensor!(T::Array{Float64}, weight::Float64, t::SALCTerm,
                           dslots::Vector{Int}, choice, mem::SALCMember,
                           e::Matrix{Float64}, polycache)
    # which derivative components land on which member site, per index-tuple position
    order = length(choice)
    for cidx in CartesianIndices(T)
        acc = 0.0
        for midx in CartesianIndices(t.folded)
            w = t.folded[midx]
            w == 0.0 && continue
            # spin axes: plain values at this configuration
            for i in eachindex(t.slots)
                sl = t.slots[i]
                sl.factor.channel == SPIN || continue
                l = sl.factor.l
                a = mem.atoms[sl.site]
                # Unchecked: `_resolve_spins` established the precondition at the
                # door (unit within the package's `atol`, components in [-1, 1]).
                # The checked kernel here used to be the ONLY validation, one layer
                # too deep — it fired mid-accumulation, at a tolerance no other door
                # uses, naming neither the argument nor the atom.
                w *= Harmonics.Zlm_unsafe(l, midx[i] - l - 1,
                                   SVector{3,Float64}(e[1, a], e[2, a], e[3, a]))
                w == 0.0 && break
            end
            w == 0.0 && continue
            # displacement axes: the derivative of this slot's polynomial, taken in
            # the components this index tuple assigns to the slot's site
            for i in dslots
                sl = t.slots[i]
                l = sl.factor.l
                pk = (sl.factor.k, l, midx[i] - l - 1)
                poly = get!(polycache, pk) do
                    SolidHarmonics.solid_harmonic_poly(pk[1], pk[2], pk[3])
                end
                comps = (cidx[p] for p = 1:order if choice[p] == sl.site)
                w *= _site_derivative(poly, comps)
                w == 0.0 && break
            end
            acc += w
        end
        acc == 0.0 && continue
        T[cidx] += weight * acc
    end
    return nothing
end

"""
    dynamical_matrix(fcs::ForceConstantSet, q; masses = nothing) -> Matrix{ComplexF64}

The `3N × N`-block reciprocal-space force-constant matrix of an order-2
[`ForceConstantSet`](@ref) at wavevector `q`:

```
D_{aα,bβ}(q) = (1 / √(Mₐ M_b)) Σ_R Φ_{aα,bβ}(R) exp(2πi q·R)
```

`q` is in **fractional reciprocal coordinates** (units of the reciprocal lattice
vectors), so `q·R` is the plain dot product with the integer lattice shift and
`q = [0,0,0]` is Γ. Rows and columns run atom-major, Cartesian-minor
(`3(a−1) + α`), matching the package's other derivative layouts.

`masses` (length `n_atoms`, positive, **in amu**) applies the `1/√(MₐM_b)` weighting;
omit it to get the unweighted force-constant matrix. The result is Hermitian up to
roundoff, and `D(−q) = conj(D(q))`.

!!! note "The eigenvalues are `ω²` in eV/Å²/amu — convert before quoting a frequency"
    The package fits energies in eV and displacements in Å, so with `masses` in amu
    the eigenvalues `λ` of the weighted `D(q)` carry units of **eV/(Å²·amu)**. They
    are `ω²` in those units and in no other; `sqrt.(λ)` is not a frequency in any
    standard one. The conversions:

    ```
    ν [THz]  = √λ × 15.633302              # ordinary frequency (the 2π is included)
    ν [cm⁻¹] = √λ × 15.633302 × 33.35641
    ```

    A negative `λ` is an imaginary mode; quote it as `−√(−λ)` in the same units.
    (This is the same constant phonopy calls `VaspToTHz`, and the `test/alamode/`
    suite pins it by comparing against `anphon`'s own cm⁻¹ output.)

For a model satisfying the acoustic sum rule, `D(0)` has three zero eigenvalues (the
acoustic modes); if it does not, check [`asr_residual`](@ref) on the model the
constants came from.
"""
function dynamical_matrix(fcs::ForceConstantSet, q::AbstractVector{<:Real};
                          masses::Union{Nothing,AbstractVector{<:Real}} = nothing)
    fcs.order == 2 || throw(ArgumentError(
        "dynamical_matrix needs order-2 force constants; this set is order " *
        "$(fcs.order) (build it with `force_constants(model; spins, order = 2)`)"))
    length(q) == 3 || throw(DimensionMismatch("q must have 3 entries; got $(length(q))"))
    nat = n_atoms(fcs.crystal)
    if masses !== nothing
        length(masses) == nat || throw(DimensionMismatch(
            "masses has $(length(masses)) entries for $nat atoms"))
        all(>(0), masses) || throw(ArgumentError("masses must be positive"))
    end
    D = zeros(ComplexF64, 3 * nat, 3 * nat)
    for ((atoms, shifts), T) in fcs.constants
        a, b = atoms[1], atoms[2]
        R = shifts[2]                                  # shifts[1] is the anchor, 0
        phase = cis(2π * (q[1] * R[1] + q[2] * R[2] + q[3] * R[3]))
        for α = 1:3, β = 1:3
            D[3 * (a - 1) + α, 3 * (b - 1) + β] += T[α, β] * phase
        end
    end
    if masses !== nothing
        for a = 1:nat, b = 1:nat, α = 1:3, β = 1:3
            D[3 * (a - 1) + α, 3 * (b - 1) + β] /= sqrt(masses[a] * masses[b])
        end
    end
    return D
end
