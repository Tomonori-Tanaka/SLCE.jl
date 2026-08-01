# Export force constants to ALAMODE's `anphon`.
#
# phonopy takes the harmonic channel and gives back bands, DOS and thermodynamics
# (`io/phonopy.jl`). What it does NOT take is the anharmonic channel — and cubic and
# quartic constants are exactly what SLCE can produce that nothing downstream can
# otherwise consume. `anphon` reads them, and with them does relaxation-time
# thermal conductivity, self-consistent phonons, Grüneisen parameters and isotope
# scattering. That is the whole reason this writer exists next to the phonopy one.
#
# Like `FORCE_CONSTANTS`, the format is positional, and here it is positional in three
# ways at once, so none of it is inferred from documentation:
#
#   * `pair1 = "a α"` names a PRIMITIVE-cell atom (`anphon` maps it through
#     `map_p2s[a][0]`), while `pairK = "s β c"` for `K ≥ 2` names a SUPERCELL atom.
#   * The supercell atom order is ours to choose, but `Data.Symmetry.Translations`
#     must agree with it, and translation 1 must be the identity — `anphon` reads the
#     primitive representative as `map_p2s[a][0]`.
#   * `c` (`cell_s`) indexes a fixed 27-entry table of SUPERCELL shifts: entry 1 is
#     `(0,0,0)`, entries 2..27 run `ix, iy, iz ∈ {-1,0,1}` skipping the origin with
#     `iz` fastest (`anphon/anharmonic_core.cpp`). It is what lets a shift that points
#     out of the supercell keep its SIGN: `anphon` forms
#     `xr_s[atm2] + xshift_s[c] - xr_s[p2s[atom(atm2)][0]]`, so writing the folded
#     residue with `c = 1` would silently replace `R` by `R mod dim`.
#
# ALAMODE works in Rydberg atomic units, so the constants are converted from eV/Åⁿ.
# All of it — ordering, sign, and units — is pinned by running `anphon` and comparing
# its frequencies against `dynamical_matrix` (`test/alamode/`).

const _RY_PER_EV = 1 / 13.605693122994      # CODATA 2018
const _BOHR_PER_ANGSTROM = 1.8897261246257702

"""
    write_alamode(path, fcs...; dim = nothing, description = "") -> NamedTuple

Write one or more [`ForceConstantSet`](@ref)s as an ALAMODE `FCSXML` document for
`anphon`. The harmonic set (`order = 2`) is required; cubic and quartic sets are
optional and are what this export is *for* — `anphon` turns them into
relaxation-time thermal conductivity, self-consistent phonons and Grüneisen
parameters, none of which the harmonic-only phonopy route can give you. Returns
`(; path, dim, n_super, orders)`.

```julia
model = SLCEModel(f)
write_alamode("si.xml", force_constants(model; spins = e, order = 2),
                        force_constants(model; spins = e, order = 3))
```
```console
\$ anphon phband.in     # with FCSXML = si.xml, MODE = phonons / RTA
```

All sets must come from the same crystal and the same spin state — they are the
expansion of ONE energy surface, and `anphon` has no notion of a magnetic state. `dim`
defaults to the smallest odd box holding every lattice shift across all of them; it
determines the supercell the file is written over, and each shift keeps its sign
through the `cell_s` mechanism rather than being folded.

Values are converted to Rydberg atomic units (Ry/Bohrⁿ), ALAMODE's convention, from
the eV/Å of a model fitted to eV and Å.

!!! note "This is a force-constant file, not an ALM calculation"
    The document carries the structure, the translation map and the constants —
    everything `anphon` reads. It deliberately does not reproduce ALM's
    `HarmonicUnique` / `CubicUnique` sections, which describe how ALM's own fit was
    parameterized; nothing reads them back.
"""
function write_alamode(path::AbstractString, fcs::ForceConstantSet...;
                       dim::Union{Nothing,NTuple{3,Integer},AbstractVector{<:Integer}} =
                           nothing,
                       description::AbstractString = "")
    isempty(fcs) && throw(ArgumentError("write_alamode: no force constants given"))
    orders = [f.order for f in fcs]
    2 in orders || throw(ArgumentError(
        "write_alamode: the harmonic set (order = 2) is required — `anphon` reads " *
        "Data.ForceConstants.HARMONIC first and exits without it. Got orders $orders."))
    allunique(orders) ||
        throw(ArgumentError("write_alamode: duplicate orders $orders"))
    # Same reason as `write_phonopy`: an empty set is a valid file describing no
    # interaction at all, and `anphon` will read it without complaint.
    for f in fcs
        isempty(f.constants) && @warn(
            "write_alamode: the order-$(f.order) force-constant set is EMPTY, so the " *
            "file will describe no interaction at that order. A pure-spin model, or " *
            "a basis with no displacement term at that order, yields an empty set.")
    end
    for f in fcs[2:end]
        f.crystal === fcs[1].crystal || f.crystal == fcs[1].crystal ||
            throw(ArgumentError("write_alamode: the sets are from different crystals"))
        f.spins == fcs[1].spins || throw(ArgumentError(
            "write_alamode: the sets were evaluated at different spin states. They " *
            "must be one energy surface's expansion — `anphon` has no notion of a " *
            "magnetic state, so mixing two would silently describe neither."))
    end
    crystal = fcs[1].crystal
    nat = n_atoms(crystal)
    d = dim === nothing ? _alamode_auto_dim(fcs) : NTuple{3,Int}(Tuple(dim))
    all(>(0), d) || throw(ArgumentError("dim must be positive; got $d"))
    ncell = prod(d)
    nsup = nat * ncell

    open(path, "w") do io
        println(io, """<?xml version="1.0" encoding="utf-8"?>""")
        println(io, "<Data>")
        println(io, "  <Description>")
        println(io, "    <OriginalXML>", isempty(description) ?
                    "written by SLCE.write_alamode" : description, "</OriginalXML>")
        println(io, "  </Description>")
        _alamode_structure(io, crystal, d, ncell, nsup)
        println(io, "  <ForceConstants>")
        for f in sort(collect(fcs); by = x -> x.order)
            _alamode_fc_block(io, f, crystal, d, ncell)
        end
        println(io, "  </ForceConstants>")
        println(io, "</Data>")
    end
    return (; path = String(path), dim = d, n_super = nsup, orders = sort(orders))
end

function _alamode_auto_dim(fcs)::NTuple{3,Int}
    m = [0, 0, 0]
    for f in fcs, (key, _) in f.constants
        s = key[2]
        for t = 2:length(s), k = 1:3
            m[k] = max(m[k], abs(s[t][k] - s[1][k]))
        end
    end
    return (2m[1] + 1, 2m[2] + 1, 2m[3] + 1)
end

# Supercell atom index. The order is ours; `Data.Symmetry.Translations` below is
# written from the same expression, and translation 1 is the identity because
# `anphon` reads a primitive atom's representative as `map_p2s[a][0]`.
@inline _alamode_index(a::Int, l::NTuple{3,Int}, d::NTuple{3,Int}, ncell::Int)::Int =
    (a - 1) * ncell + (l[1] + d[1] * (l[2] + d[2] * l[3])) + 1

# The 27-entry supercell-shift table `anphon` indexes with `cell_s`: entry 1 is the
# origin, then ix, iy, iz over {-1,0,1} skipping it, iz fastest.
function _alamode_cell_s(s::NTuple{3,Int})::Int
    s == (0, 0, 0) && return 1
    all(x -> -1 <= x <= 1, s) || throw(ArgumentError(
        "write_alamode: a lattice shift reaches $(s) supercells away, but ALAMODE's " *
        "cell table only spans ±1. Use a larger dim."))
    i = 1
    for ix = -1:1, iy = -1:1, iz = -1:1
        (ix, iy, iz) == (0, 0, 0) && continue
        i += 1
        (ix, iy, iz) == s && return i
    end
    return 1                                        # unreachable
end

function _alamode_structure(io::IO, crystal::Crystal, d::NTuple{3,Int}, ncell::Int,
                            nsup::Int)
    A = crystal.lattice.vectors                     # columns aᵢ
    labels = crystal.species_labels
    println(io, "  <Structure>")
    println(io, "    <NumberOfAtoms>", nsup, "</NumberOfAtoms>")
    println(io, "    <NumberOfElements>", length(labels), "</NumberOfElements>")
    println(io, "    <AtomicElements>")
    for (k, lab) in enumerate(labels)
        println(io, "      <element number=\"", k, "\">", lab, "</element>")
    end
    println(io, "    </AtomicElements>")
    println(io, "    <LatticeVector>")
    for i = 1:3
        v = (@view A[:, i]) .* (d[i] * _BOHR_PER_ANGSTROM)   # SUPERCELL, in Bohr
        @printf(io, "      <a%d>%25.15e%25.15e%25.15e</a%d>\n", i, v[1], v[2], v[3], i)
    end
    println(io, "    </LatticeVector>")
    println(io, "    <Periodicity>1 1 1</Periodicity>")
    println(io, "    <Position>")
    nat = n_atoms(crystal)
    for a = 1:nat, l3 = 0:(d[3] - 1), l2 = 0:(d[2] - 1), l1 = 0:(d[1] - 1)
        i = _alamode_index(a, (l1, l2, l3), d, ncell)
        f = ((@view crystal.frac_positions[:, a]) .+ (l1, l2, l3)) ./ d
        @printf(io, "      <pos index=\"%d\" element=\"%s\">%25.15e%25.15e%25.15e</pos>\n",
                i, labels[crystal.species[a]], f[1], f[2], f[3])
    end
    println(io, "    </Position>")
    println(io, "  </Structure>")
    println(io, "  <Symmetry>")
    println(io, "    <NumberOfTranslations>", ncell, "</NumberOfTranslations>")
    println(io, "    <Translations>")
    t = 0
    for l3 = 0:(d[3] - 1), l2 = 0:(d[2] - 1), l1 = 0:(d[1] - 1)
        t += 1
        for a = 1:nat
            println(io, "      <map tran=\"", t, "\" atom=\"", a, "\">",
                    _alamode_index(a, (l1, l2, l3), d, ncell), "</map>")
        end
    end
    println(io, "    </Translations>")
    println(io, "  </Symmetry>")
    return nothing
end

function _alamode_fc_block(io::IO, f::ForceConstantSet, crystal::Crystal,
                           d::NTuple{3,Int}, ncell::Int)
    n = f.order
    tag = n == 2 ? "HARMONIC" : "ANHARM$(n)"
    name = n == 2 ? "FC2" : "FC$(n)"
    # eV/Åⁿ → Ry/Bohrⁿ: one energy conversion, and one length conversion per
    # displacement index (Φ⁽ⁿ⁾ is an n-th derivative).
    scale = _RY_PER_EV / _BOHR_PER_ANGSTROM^n
    println(io, "    <", tag, ">")
    for (key, T) in f.constants
        atoms, shifts = key
        for cidx in CartesianIndices(T)
            v = T[cidx] * scale
            v == 0.0 && continue
            print(io, "      <", name, " pair1=\"", atoms[1], " ", cidx[1], "\"")
            for t = 2:n
                R = ntuple(k -> shifts[t][k] - shifts[1][k], 3)
                Rf = ntuple(k -> mod(R[k], d[k]), 3)
                Rs = ntuple(k -> (R[k] - Rf[k]) ÷ d[k], 3)
                print(io, " pair", t, "=\"",
                      _alamode_index(atoms[t], Rf, d, ncell), " ", cidx[t], " ",
                      _alamode_cell_s(Rs), "\"")
            end
            @printf(io, ">%.15e</%s>\n", v, name)
        end
    end
    println(io, "    </", tag, ">")
    return nothing
end
