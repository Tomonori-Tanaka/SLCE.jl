# The ALAMODE export, gated against `anphon` itself.
#
# Local-only, like `test/oracle/`: `anphon` is a C++ binary needing Boost, FFTW,
# spglib and MPI, which is too much to build in CI for one job. Point `ANPHON` at it:
#
#     ANPHON=~/Packages/alamode/_build/anphon/anphon \
#         julia --project=test/alamode test/alamode/runtests.jl
#
# WHY THE COMPARISON HAS THIS SHAPE. `anphon` reports frequencies in cm⁻¹ and this
# package reports eigenvalues in eV/Å²/amu, so comparing them needs the physical
# constants twice, and the two codes round them differently. The gate therefore fits
# ONE scale over all modes at all q and asserts (a) the ABSOLUTE residual after it,
# 3.7e-5 cm⁻¹ measured — bounded below by anphon printing four decimals, which is why
# it is absolute and not relative, a relative tolerance being meaningless on a soft
# mode — and (b) that the fitted scale is 1 to 5.5e-10, which is the units claim and a
# separate one. A single `≈` with enough slack to absorb the constants would pass a
# sign error on a small mode; these two together do not.
#
# Mutation-tested: forcing `cell_s = 1` — writing each shift's folded residue instead
# of its signed value, the one mechanism this format has and phonopy's does not —
# moves the comparison from 2.1e-7 to 5.2e-1.

using Test
using SLCE
using LinearAlgebra
using Random
using Statistics

const ANPHON = get(ENV, "ANPHON", "")

isempty(ANPHON) && error("""
    Set ANPHON to the `anphon` binary, e.g.
    ANPHON=~/Packages/alamode/_build/anphon/anphon julia --project=test/alamode \\
        test/alamode/runtests.jl""")
isfile(ANPHON) || error("ANPHON = $ANPHON is not a file")

const MASSES = [55.845, 58.6934, 55.845]
# eV/Å²/amu → cm⁻¹: √λ × (VaspToTHz) × (THz → cm⁻¹).
const TO_CM = 15.633302 * 33.35641

# Same low-symmetry, interleaved-species cell as the phonopy suite, for the same
# reasons: a cubic cell makes axis permutations crystal symmetries, and a
# single-species one makes the species grouping the identity.
function _fixture(; degree = 2:2, pmax = 2)
    lat = Lattice([3.3 0.0 0.0; 0.0 3.9 0.0; 0.0 0.0 7.4])
    cr = Crystal(lat, [0.06 0.93 0.10; 0.10 0.14 0.88; 0.20 0.23 0.26],
                 [1, 2, 1], ["Fe", "Ni"])
    b = SLCEBasis(cr, BasisSpec(cr; lmax = 0, pmax = pmax,
                                sectors = [Sector(disp = (degree = degree,),
                                                  cutoff = 3.0)]))
    return SLCEModel(b, 0.0, 0.3 .* randn(MersenneTwister(5), n_salcs(b)))
end

_cell_block(cr) = let A = cr.lattice.vectors * 1.8897261246257702
    """
    &cell
      1.0
      $(A[1,1]) $(A[2,1]) $(A[3,1])
      $(A[1,2]) $(A[2,2]) $(A[3,2])
      $(A[1,3]) $(A[2,3]) $(A[3,3])
    /
    """
end

function _anphon_frequencies(dir, cr, qs; extra = "")
    write(joinpath(dir, "in"), """
    &general
      PREFIX = slce
      MODE = phonons
      FCSXML = slce.xml
      NKD = 2; KD = Fe Ni
      MASS = 55.845 58.6934
    /
    $extra
    $(_cell_block(cr))
    &kpoint
      0
    $(join(["  $(q[1]) $(q[2]) $(q[3])" for q in qs], "\n"))
    /
    """)
    log = read(pipeline(Cmd(`$ANPHON in`; dir = dir); stderr = devnull), String)
    out, cur = Vector{Float64}[], Float64[]
    for line in eachline(IOBuffer(log))
        m = match(r"^\s+\d+\s+(-?\d+\.\d+)\s+cm\^-1", line)
        m === nothing && continue
        push!(cur, parse(Float64, m.captures[1]))
        if length(cur) == 9
            push!(out, sort(cur))
            cur = Float64[]
        end
    end
    return out, log
end

@testset "ALAMODE export" begin
    @testset "harmonic: anphon's frequencies are ours" begin
        model = _fixture()
        fcs = force_constants(model; order = 2)
        qs = [[0.0, 0.0, 0.0], [0.13, 0.29, 0.41], [0.5, 0.17, 0.06],
              [0.31, 0.05, 0.44]]
        mktempdir() do dir
            out = write_alamode(joinpath(dir, "slce.xml"), fcs)
            @test out.orders == [2] && out.dim == (3, 3, 1)
            got, _ = _anphon_frequencies(dir, fcs.crystal, qs)
            @test length(got) == length(qs)
            pairs_ = Tuple{Float64,Float64}[]
            for (k, q) in enumerate(qs)
                λ = sort(real.(eigvals(dynamical_matrix(fcs, q; masses = MASSES))))
                ω = [x < 0 ? -sqrt(-x) : sqrt(x) for x in λ] .* TO_CM
                for (a, b) in zip(got[k], ω)
                    abs(b) > 1.0 && push!(pairs_, (a, b))    # skip the acoustic zeros
                end
            end
            @test length(pairs_) > 20
            # `anphon` prints cm⁻¹ to four decimals, so the comparison cannot be
            # sharper than 1e-4 cm⁻¹ absolute — which is why this is an ABSOLUTE
            # residual after one fitted scale, not a relative tolerance that a soft
            # mode would make meaningless.
            c = sum(a * b for (a, b) in pairs_) / sum(b * b for (_, b) in pairs_)
            resid = maximum(abs(a - c * b) for (a, b) in pairs_)
            @test resid < 1e-3                     # 10× anphon's print resolution
            # ...and the fitted scale is 1, up to the two codes' independent rounding
            # of the physical constants. Measured 1 + 2.1e-7. This is a separate
            # claim from the one above, not a slack for it.
            @test abs(c - 1) < 1e-5
            @info "anphon vs dynamical_matrix" scale = c residual_cm = resid
        end
    end

    # The reason this exporter exists: phonopy takes the harmonic channel, and nothing
    # else downstream consumes the cubic one. `GRUNEISEN = 1` is the cheapest anphon
    # mode that actually READS `ANHARM3` — it refuses without cubic constants — so it
    # proves the anharmonic indices parse, not merely that the file is well-formed.
    @testset "cubic: anphon reads ANHARM3" begin
        model = _fixture(; degree = 2:3, pmax = 3)
        f2 = force_constants(model; order = 2)
        f3 = force_constants(model; order = 3)
        @test !isempty(f3.constants)
        mktempdir() do dir
            out = write_alamode(joinpath(dir, "slce.xml"), f2, f3)
            @test out.orders == [2, 3]
            _, log = _anphon_frequencies(dir, f2.crystal, [[0.13, 0.29, 0.41]];
                                         extra = "&analysis\n  GRUNEISEN = 1\n/")
            @test occursin("Cubic force constants are necessary", log)
            @test occursin("Calculating Gruneisen parameters ... done!", log)
            # What this proves and what it does not: `anphon` PARSED the ANHARM3
            # indices (a malformed one aborts the reader — a stray space in the value
            # field is how this exporter first failed) and consumed them without
            # complaint. It does not check the cubic NUMBERS against an independent
            # computation; that would mean reimplementing Grüneisen parameters here.
            # The harmonic block above is what pins units, sign and cell_s, and the
            # cubic block is written by the same code path.
        end
    end

    @testset "refusals" begin
        model = _fixture()
        f2 = force_constants(model; order = 2)
        f3 = force_constants(_fixture(; degree = 2:3, pmax = 3); order = 3)
        mktempdir() do dir
            p = joinpath(dir, "x.xml")
            @test_throws ArgumentError write_alamode(p)
            @test_throws ArgumentError write_alamode(p, f3)          # no harmonic
            @test_throws ArgumentError write_alamode(p, f2, f2)      # duplicate order
            @test_throws ArgumentError write_alamode(p, f2; dim = (0, 1, 1))
        end
    end
end
