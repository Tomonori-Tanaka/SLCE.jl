# Decoration labels (joint spin–lattice M2a): SiteFactor / SiteDecor validation,
# the canonical ordering, and the v4→v5 pure-spin key relabel semantics.

using Test
using SLCE
using SLCE: Channel, SPIN, DISP, OCC, SiteFactor, SiteDecor,
            has_spin, has_disp, spin_rank, disp_degree, factors, is_pure_spin,
            spin_decors, spin_ls, SALCKey, rep_scale

@testset "decoration labels" begin
    @testset "Channel order is SPIN < DISP < OCC" begin
        @test SPIN < DISP < OCC
        @test Integer(SPIN) == 1 && Integer(DISP) == 2 && Integer(OCC) == 3
    end

    @testset "SiteFactor validation" begin
        @test SiteFactor(SPIN, 0, 2).l == 2
        @test_throws ArgumentError SiteFactor(SPIN, 1, 2)   # spin has no radial degree
        @test_throws ArgumentError SiteFactor(SPIN, 0, 0)   # spin l ≥ 1
        @test SiteFactor(DISP, 0, 1).l == 1
        @test SiteFactor(DISP, 1, 0).k == 1                 # |u|² trace channel legal
        @test_throws ArgumentError SiteFactor(DISP, 0, 0)   # degree 2k + l ≥ 1
        @test_throws ArgumentError SiteFactor(DISP, -1, 1)
        @test_throws ArgumentError SiteFactor(OCC, 0, 1)    # reserved channel
        @test SiteFactor(SPIN, 0, 1) < SiteFactor(DISP, 0, 1)   # channel-major order
    end

    @testset "SiteDecor construction and accessors" begin
        ds = SiteDecor(; spin = 2)
        dd = SiteDecor(; disp = (1, 1))
        db = SiteDecor(; spin = 1, disp = (0, 2))
        @test has_spin(ds) && !has_disp(ds) && is_pure_spin(ds)
        @test !has_spin(dd) && has_disp(dd) && !is_pure_spin(dd)
        @test has_spin(db) && has_disp(db) && !is_pure_spin(db)
        @test spin_rank(ds) == 2 && spin_rank(dd) == 0
        @test disp_degree(dd) == 3 && disp_degree(db) == 2 && disp_degree(ds) == 0
        @test factors(ds) == [SiteFactor(SPIN, 0, 2)]
        @test factors(db) == [SiteFactor(SPIN, 0, 1), SiteFactor(DISP, 0, 2)]
        # factor-list constructor agrees with the keyword form
        @test SiteDecor(SiteFactor(SPIN, 0, 1), SiteFactor(DISP, 0, 2)) == db
        @test SiteDecor(SiteFactor(DISP, 1, 1)) == dd
        # validation
        @test_throws ArgumentError SiteDecor()                       # no factor
        @test_throws ArgumentError SiteDecor(; spin = -1)
        @test_throws ArgumentError SiteDecor(; disp = (0, 0))
        @test_throws ArgumentError SiteDecor(SiteFactor(SPIN, 0, 1),
                                             SiteFactor(SPIN, 0, 2))  # channel slot
        @test_throws ArgumentError SiteDecor(SiteFactor(DISP, 0, 1),
                                             SiteFactor(DISP, 1, 0))
        # show is compact and channel-labeled
        @test occursin("spin=1", repr(db)) && occursin("disp=(0,2)", repr(db))
    end

    @testset "ordering: pure-spin decors sort like the v4 ls label" begin
        ls = [2, 1, 1, 3]
        @test sort(spin_decors(ls)) == spin_decors(sort(ls))
        # spin-only sorts before spin+disp at equal spin rank
        @test SiteDecor(; spin = 1) < SiteDecor(; spin = 1, disp = (0, 1))
        # disp-only (spin_l = 0) sorts before any spin-carrying decor
        @test SiteDecor(; disp = (0, 1)) < SiteDecor(; spin = 1)
    end

    @testset "SALCKey v5 layout and the pure-spin relabel" begin
        k = SALCKey(2, 3, spin_decors([1, 1]), 0, 0, 1)
        @test spin_ls(k) == [1, 1]
        @test is_pure_spin(k)
        @test k.L_S == 0
        # a mixed key reports only its spin ranks through spin_ls
        km = SALCKey(2, 3, [SiteDecor(; spin = 2), SiteDecor(; disp = (0, 1))], 2, 1, 1)
        @test spin_ls(km) == [2]
        @test !is_pure_spin(km)
        # key ordering: L_S sorts before Lf within one decoration label
        ka = SALCKey(2, 3, spin_decors([1, 1]), 0, 0, 1)
        kb = SALCKey(2, 3, spin_decors([1, 1]), 1, 1, 1)
        @test ka < kb
        @test SALCKey(2, 3, spin_decors([1, 1]), 0, 0, 1) == ka
        @test hash(ka) == hash(SALCKey(2, 3, spin_decors([1, 1]), 0, 0, 1))
    end

    @testset "rep_scale channel trait" begin
        # SPIN is axial: det(R)^l — the inversion representation is +I for every l
        @test rep_scale(SPIN, -1.0, 1) == -1.0     # axial l=1 under a mirror-free det
        @test rep_scale(SPIN, -1.0, 2) == 1.0
        @test rep_scale(SPIN, 1.0, 3) == 1.0
        # DISP is polar: the Wigner matrix itself carries the parity
        @test rep_scale(DISP, -1.0, 1) == 1.0
        @test rep_scale(DISP, -1.0, 2) == 1.0
        # the reserved channel has no declared representation
        @test_throws ArgumentError rep_scale(OCC, -1.0, 1)
        # the production-correctness identity: over an even-Σl_spin label the
        # spin factors' product is +1 for every op — the polar Wigner cache is
        # exact for both channels (design record §4)
        for ls in ([1, 1], [1, 3], [2, 2], [1, 1, 2])
            @test prod(rep_scale(SPIN, -1.0, l) for l in ls) == 1.0
        end
    end
end
