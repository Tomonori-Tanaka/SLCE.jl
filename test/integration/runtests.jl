# The integration tier: whole-pipeline passes on named real crystals.
#
# WHY IT EXISTS, given a unit suite of ~48k assertions. Every unit file gates one
# stage against one hand-built fixture. Nothing gated the COMBINATION on a crystal
# with a name — and the last three real defects in this package were all found that
# way, by taking one real system through the whole chain, not by adding assertions to
# a fixture. This tier is that walk, written down: build → data → ASR → fit →
# recovery → every downstream deliverable → persistence, on six crystals chosen so
# that each one forces a different code path.
#
# WHY A SEPARATE ENVIRONMENT. It needs Spglib, and the core suite deliberately does
# not: `Pkg.test()` must never depend on a symmetry backend, so the unit fixtures
# assemble their space groups by hand. Here that would defeat the purpose — the tier
# is about the path a user actually takes, and the real space group of a real crystal
# is part of it. Spglib is also an INDEPENDENT implementation of the one thing the
# fixtures would otherwise assert about themselves.
#
# THE COVERAGE MATRIX. `roster.jl` declares, per row, which columns run and — with a
# reason, in the file — which do not. The driver refuses a row that leaves a column
# unaccounted for, and refuses to report success if a declared column never
# executed. A test tier that quietly covers less than it claims is the failure this
# structure exists to prevent (the same reason `test/runtests.jl` refuses an
# unrecognized `TEST_MODE` instead of running zero tests).
#
# Run:
#     julia --project=test/integration test/integration/runtests.jl
# Threads are not required here: nothing in this tier compares a threaded result
# against a serial one (the core suite owns those gates).

using Test
using SLCE
using LinearAlgebra
using Random
using Printf
import Spglib                      # loads the SpglibBackend extension

include("roster.jl")
# The unit suite's shared helpers: `check_canonical_members` is the DEFINITION of a
# well-formed SALC member, and `randcfg` / `rand_rotation` seed the same way every
# other fixture in the package does. A second copy here would be a second
# definition of both.
include("../unit/testutils.jl")
include("checks.jl")

@testset "SLCE integration tier" begin
    @testset "the coverage matrix is total" begin
        # Declared before anything runs: every row accounts for every column, no
        # column is both run and skipped, and no skip is an empty excuse.
        for row in ROSTER
            declared = union(Set(row.runs), Set(keys(row.skips)))
            @test declared == Set(COLUMNS)
            @test isempty(intersect(Set(row.runs), Set(keys(row.skips))))
            @test all(!isempty(strip(r)) for r in values(row.skips))
            @test all(haskey(CHECKS, c) for c in row.runs)
        end
        @test length(unique(r.id for r in ROSTER)) == length(ROSTER)
    end

    contexts = Dict{Symbol,Any}()
    for row in ROSTER
        @testset "$(row.title)" begin
            t0 = time()
            ctx = prepare(row)
            contexts[row.id] = ctx
            @printf("  %-14s nat=%2d  m=%3d  free=%3d  frozen=%3d  (%.1f s)\n",
                    row.id, ctx.nat, ctx.m, ctx.q, length(ctx.frozen), time() - t0)
            for col in row.runs
                CHECKS[col](row, ctx)
            end
        end
    end

    @testset "the tie contrast: one crystal, two cells" begin
        # bcc Fe in its conventional cell and in a 2×2×2 tiling of it. The same
        # physical orbits give the same COLUMNS — the tie changes nothing about
        # which couplings exist — but the conventional cell cannot determine most
        # of them: its 1NN shell reaches eight images of one atom, all at the same
        # distance, so only their sum is measurable and the difference content is
        # frozen. Tiling separates the eight images onto eight distinct atoms.
        small = contexts[:bcc_Fe]
        big = contexts[:bcc_Fe_222]
        @test small.m == big.m                               # same columns
        @test small.basis.salc_basis.keys == big.basis.salc_basis.keys
        @test !isempty(small.frozen)                         # tied
        @test isempty(big.frozen)                            # not tied
        @test big.q > small.q                                # strictly more resolved
        # The tie is invisible at Γ and only at Γ: Σ_R Φ(R) is the Hessian of
        # exactly the energy the small cell can express, so its acoustic modes are
        # exact, while the frozen content is precisely what D(q ≠ 0) needs.
        fcs = force_constants(small.fitted; spins = small.spins, order = 2)
        ev = eigvals(Hermitian(dynamical_matrix(fcs, [0.0, 0.0, 0.0])))
        @test count(x -> abs(x) < 1e-8 * maximum(abs, ev), ev) == 3
    end

    @testset "every declared column ran" begin
        # The ledger. A check that throws is caught by its own testset; a check
        # that was never CALLED — a dispatch typo, a row edited out of the loop —
        # would otherwise pass silently as an absence of failures.
        declared = Set((row.id, col) for row in ROSTER for col in row.runs)
        @test RAN == declared
        missing_cols = sort(collect(setdiff(declared, RAN)))
        isempty(missing_cols) || @info "columns declared but never run" missing_cols
        extra = sort(collect(setdiff(RAN, declared)))
        isempty(extra) || @info "columns run but not declared" extra
        @printf("  %d (row, column) pairs ran; %d declared skips carry a reason\n",
                length(RAN), sum(length(r.skips) for r in ROSTER))
    end
end
