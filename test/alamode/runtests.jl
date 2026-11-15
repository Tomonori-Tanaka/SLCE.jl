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
    # else downstream consumes the cubic one. This gate checks the cubic NUMBERS as
    # `anphon` uses them: its mode Grüneisen parameters along a generic path against
    # the same quantity formed here directly from the in-memory constants,
    #
    #     γ_s(q) = −⟨e_s| δD(q) |e_s⟩ / (6 |ω_s²|),
    #     δD(q)_{aα,bβ} = Σ_{R,k,R'} Φ[(a,0),(b,R),(k,R')]_{αβγ} x_{k,γ}(R')
    #                     · e^{2πi q·R} / √(m_a m_b),
    #
    # (`anphon/gruneisen.cpp`: `prepare_delta_fcs` contracts the last index with the
    # atom's position, `calc_gruneisen` divides by −6 times the SQUARE OF THE SIGNED
    # frequency, i.e. by |ω²| — which is why an unstable mode of this random model
    # keeps the sign written here). The position origin is irrelevant because the
    # fixture obeys the sum rule of the tiled crystal (checked below, and printed by
    # `anphon` itself as "Maximum deviation from the translational invariance"). That
    # needs a cell in which no cutoff sphere reaches an image (cutoff 3.0 < every
    # L/2 of the 2×2×1 cell): with overlapping images the folded sum rule leaves the
    # cubic constants origin dependent (OPEN-WORK S-4) and the comparison is moot.
    #
    # Measured 2026-11-15: 4.9e-7 — `anphon` prints γ to six decimals. The writer that
    # listed every ordering of the trailing pairs (before 2026-11-15) differs by 4.3,
    # i.e. this gate also catches the multiplicity defect the read-back test in
    # `test_latticeonly.jl` is built for, and any error in `cell_s`, units or index
    # order of the cubic block.
    @testset "cubic: anphon's Grüneisen parameters are ours" begin
        lat0 = [3.3 0.0 0.0; 0.0 3.9 0.0; 0.0 0.0 7.4]
        x0 = [0.06 0.93 0.10; 0.10 0.14 0.88; 0.20 0.23 0.26]
        pos, typ = Vector{Float64}[], Int[]
        for i = 0:1, j = 0:1, a = 1:3
            push!(pos, (x0[:, a] .+ [i, j, 0]) ./ [2, 2, 1])
            push!(typ, [1, 2, 1][a])
        end
        cr = Crystal(Lattice(lat0 * Diagonal([2, 2, 1])), reduce(hcat, pos), typ,
                     ["Fe", "Ni"])
        b = SLCEBasis(cr, BasisSpec(cr; lmax = 0, pmax = 3,
                                    sectors = [Sector(disp = (degree = 2,), sites = 1:2,
                                                      cutoff = 3.0),
                                               Sector(disp = (degree = 3,), sites = 1:3,
                                                      cutoff = 3.0)]))
        rep = SLCE.build_asr(b; warn = false)
        beta = SLCE._lift_gamma(rep, randn(MersenneTwister(11), size(rep.Z, 2)))
        model = SLCEModel(b, 0.0, beta)
        f2 = force_constants(model; order = 2)
        f3 = force_constants(model; order = 3)

        # the tiled-crystal sum rule, Σ_{k,R'} Φ[(a,0),(b,R),(k,R')] = 0 per (a,b,R)
        sums, scale = Dict{Any,Float64}(), 0.0
        for ((atoms, shifts), T) in f3.constants, c in CartesianIndices(T)
            scale = max(scale, abs(T[c]))
            key = (atoms[1:2], shifts[2], Tuple(c))
            sums[key] = get(sums, key, 0.0) + T[c]
        end
        @test maximum(abs, values(sums)) < 1e-12 * scale

        nat = n_atoms(cr)
        masses = [[55.845, 58.6934][t] for t in typ]
        A = cr.lattice.vectors
        function gamma_here(q)
            D = zeros(ComplexF64, 3nat, 3nat)
            dD = zeros(ComplexF64, 3nat, 3nat)
            for ((atoms, shifts), T) in f2.constants
                a, c = atoms
                ph = cis(2π * dot(q, shifts[2])) / sqrt(masses[a] * masses[c])
                D[3a-2:3a, 3c-2:3c] .+= T .* ph
            end
            for ((atoms, shifts), T) in f3.constants
                a, c, k = atoms
                x = A * (cr.frac_positions[:, k] .+ shifts[3])
                ph = cis(2π * dot(q, shifts[2])) / sqrt(masses[a] * masses[c])
                for γ = 1:3
                    dD[3a-2:3a, 3c-2:3c] .+= (@view T[:, :, γ]) .* (x[γ] * ph)
                end
            end
            F = eigen(Hermitian((D + D') / 2))
            return [-real(F.vectors[:, s]' * dD * F.vectors[:, s]) / (6 * abs(F.values[s]))
                    for s in eachindex(F.values)]
        end

        qend = [0.37, 0.21, 0.13]
        mktempdir() do dir
            out = write_alamode(joinpath(dir, "slce.xml"), f2, f3)
            @test out.orders == [2, 3]
            Ab = A * 1.8897261246257702
            p0, p1 = join(0.2 .* qend, " "), join(qend, " ")
            write(joinpath(dir, "in"), """
            &general
              PREFIX = slce
              MODE = phonons
              FCSXML = slce.xml
              NKD = 2; KD = Fe Ni
              MASS = 55.845 58.6934
            /
            &analysis
              GRUNEISEN = 1
            /
            $(_cell_block(cr))
            &kpoint
              1
              P $p0 Q $p1 5
            /
            """)
            log = read(pipeline(Cmd(`$ANPHON in`; dir = dir); stderr = devnull), String)
            @test occursin("Calculating Gruneisen parameters ... done!", log)
            rows = filter(l -> !startswith(l, "#"),
                          readlines(joinpath(dir, "slce.gruneisen")))
            @test length(rows) == 5                    # both ends of the path included
            worst = 0.0
            for (k, t) in enumerate(range(0.2, 1.0; length = 5))
                got = parse.(Float64, split(rows[k]))[2:end]
                want = gamma_here(t .* qend)
                @test length(got) == length(want) == 3nat
                worst = max(worst, maximum(abs.(got .- want)))
            end
            # 20× the six-decimal print resolution; the defect it guards is O(1).
            @test worst < 1e-5
            @info "anphon Grüneisen vs here" max_abs_difference = worst
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
