# Oracle validation: SLCE's from-scratch numerics vs Magesty.jl.
#
# Conventions are pinned independently by the core suite (closed forms, on-sphere
# finite differences). Here we additionally confirm bit-for-bit agreement with
# Magesty, which guards the convention all the way to the design matrix and
# predictions downstream. Gauge-dependent quantities (raw SALC coefficients) are
# NOT compared directly — only convention-fixed kernels and gauge-invariant
# aggregates / predictions (added as later milestones land).

using Test
using Random
using StaticArrays
using LinearAlgebra
using OffsetArrays
import SLCE
import Magesty
import WignerSymbols
import Spglib   # triggers SLCESpglibExt

const MRH = SLCE.Harmonics
const MRA = SLCE.AngularMomentum
const MTH = Magesty.TesseralHarmonics
const MTR = Magesty.RotationMatrix

rand_unit(rng) = (v = SVector{3,Float64}(randn(rng), randn(rng), randn(rng)); v / norm(v))

function rand_rotation(rng)
    a = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
    a = a / norm(a)
    θ = 2π * rand(rng)
    Kc = @SMatrix [0.0 -a[3] a[2]; a[3] 0.0 -a[1]; -a[2] a[1] 0.0]
    return SMatrix{3,3,Float64}(I) + sin(θ) * Kc + (1 - cos(θ)) * (Kc * Kc)
end

@testset "oracle vs Magesty" begin
    @testset "tesseral harmonics Zₗₘ and ∇Zₗₘ (M2)" begin
        rng = MersenneTwister(11)
        lmax = 4
        for _ = 1:200
            u = rand_unit(rng)
            for l = 0:lmax, m = -l:l
                @test MRH.Zlm(l, m, u) ≈ MTH.Zₗₘ(l, m, u) atol = 1e-13 rtol = 1e-12
                g_new = MRH.grad_Zlm(l, m, u)
                g_ref = SVector{3,Float64}(MTH.∂ᵢZlm(l, m, u))
                @test isapprox(g_new, g_ref; atol = 1e-12, rtol = 1e-11)
            end
        end
    end

    @testset "Clebsch-Gordan vs WignerSymbols (M3)" begin
        for j1 = 0:3, j2 = 0:3
            for J = abs(j1 - j2):(j1+j2), m1 = -j1:j1, m2 = -j2:j2
                M = m1 + m2
                abs(M) <= J || continue
                ref = Float64(WignerSymbols.clebschgordan(j1, m1, j2, m2, J, M))
                @test MRA.clebsch_gordan(j1, m1, j2, m2, J, M) ≈ ref atol = 1e-12
            end
        end
    end

    @testset "real Wigner-D vs Magesty Δl (M3)" begin
        rng = MersenneTwister(31)
        # Both are the rotation's representation in the same tesseral basis; they agree
        # up to the active/passive (transpose) convention. Resolve that convention ONCE
        # (on the first generic sample) and hold every (R, l) to the same branch — a
        # per-sample `≈ ref || ≈ refᵀ` would let a convention flip-flop slide through.
        transposed = nothing
        for _ = 1:15
            R = rand_rotation(rng)
            α, β, γ = MTR.rotmat2euler(Matrix(R))
            for l = 1:4
                mine = MRA.wignerD_real(l, R)
                ref = MTR.Δl(l, α, β, γ)
                if transposed === nothing
                    transposed = !isapprox(mine, ref; atol = 1e-9)
                end
                @test isapprox(mine, transposed ? permutedims(ref) : ref; atol = 1e-9)
            end
        end
    end

    @testset "coupled real tensors vs Magesty (M4)" begin
        MAC = Magesty.AngularMomentumCoupling
        for ls in ([1], [2], [1, 1], [1, 2], [2, 2], [1, 1, 1])
            mine = MRA.build_real_bases(ls)
            bases_by_L, paths_by_L = MAC.build_all_real_bases(ls)
            @test length(mine) == sum(length(v) for v in values(bases_by_L))
            for (Lseq, Lf, T) in mine
                ks = findall(==(Lseq), paths_by_L[Lf])
                @test length(ks) == 1
                @test isapprox(T, bases_by_L[Lf][ks[1]]; atol = 1e-10)
            end
        end
    end

    @testset "SpglibBackend space groups (M5)" begin
        Lattice = SLCE.Lattice
        Crystal = SLCE.Crystal
        SpglibBackend = SLCE.SpglibBackend
        analyze = SLCE.analyze_symmetry

        a = 3.0
        sc = Crystal(Lattice(Matrix(a * I(3))), reshape([0.0, 0.0, 0.0], 3, 1), [1], ["X"])
        bcc = Crystal(Lattice(Matrix(a * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["X"])
        fcc = Crystal(Lattice(Matrix(4.0 * I(3))),
                      [0.0 0.5 0.5 0.0; 0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.5],
                      [1, 1, 1, 1], ["X"])

        sg_sc = analyze(SpglibBackend(), sc)
        @test sg_sc.number == 221 && length(sg_sc.ops) == 48          # Pm-3m
        @test analyze(SpglibBackend(), bcc).number == 229             # Im-3m
        @test analyze(SpglibBackend(), fcc).number == 225             # Fm-3m

        for sg in (analyze(SpglibBackend(), bcc), analyze(SpglibBackend(), fcc))
            nat = size(sg.map_sym, 1)
            @test any(o -> sg.ops[o].is_translation && all(sg.map_sym[:, o] .== 1:nat),
                      1:length(sg.ops))                                # identity present
            for o = 1:length(sg.ops)
                @test sort(sg.map_sym[:, o]) == collect(1:nat)         # each op permutes atoms
                @test isapprox(sg.ops[o].rotation_cart * sg.ops[o].rotation_cart',
                               Matrix(I, 3, 3); atol = 1e-9)           # orthogonal
            end
        end
    end

    @testset "cluster orbits with Spglib (M6)" begin
        Lattice = SLCE.Lattice
        Crystal = SLCE.Crystal
        SpglibBackend = SLCE.SpglibBackend
        analyze = SLCE.analyze_symmetry
        a = 3.0

        # simple cubic (1 atom): the nearest-neighbor "bond" is the atom with its own
        # periodic image (i == j). Under AllImages (the generalized-Bloch convention;
        # also Magesty's count) the 6 symmetry-equivalent edge neighbors form one
        # 2-body orbit. Under the default MinimumImage it is dropped: in plain PBC both
        # ends carry the same spin, so it is not independently resolvable from a single
        # atom — a deliberate refinement over Magesty (see CLAUDE.md).
        MinimumImage = SLCE.MinimumImage
        AllImages = SLCE.AllImages
        sc = Crystal(Lattice(Matrix(a * I(3))), reshape([0.0, 0.0, 0.0], 3, 1), [1], ["X"])
        sg = analyze(SpglibBackend(), sc)
        nl_all = SLCE.build_neighbor_list(sc, a + 0.1, AllImages())
        cs_all = SLCE.build_clusters(sc, nl_all, sg; nbody = 2, selection = AllImages())
        @test length(cs_all.by_body[1]) == 1
        @test length(cs_all.by_body[2]) == 1                  # Magesty-consistent (self-pair NN)
        nl_min = SLCE.build_neighbor_list(sc, a + 0.1, MinimumImage())
        cs_min = SLCE.build_clusters(sc, nl_min, sg; nbody = 2, selection = MinimumImage())
        @test length(cs_min.by_body[1]) == 1
        @test length(cs_min.by_body[2]) == 0                  # refined: self-pair not resolvable

        # bcc conventional cell (2 atoms): the 8 nearest neighbors are corner↔center
        # (i ≠ j), so they survive MinimumImage as one resolvable orbit; corner/center
        # sites are equivalent under body-centering.
        bcc = Crystal(Lattice(Matrix(a * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["X"])
        sgb = analyze(SpglibBackend(), bcc)
        nlb = SLCE.build_neighbor_list(bcc, 2.7, MinimumImage())
        csb = SLCE.build_clusters(bcc, nlb, sgb; nbody = 2, selection = MinimumImage())
        @test length(csb.by_body[1]) == 1
        @test length(csb.by_body[2]) == 1
    end

    @testset "SALC basis on a real chain (M7)" begin
        Lattice = SLCE.Lattice
        Crystal = SLCE.Crystal
        SpglibBackend = SLCE.SpglibBackend
        analyze = SLCE.analyze_symmetry
        evaluate_salc = SLCE.evaluate_salc

        # 4-atom chain along z (so neighboring spins differ)
        lat = Lattice(Matrix(Diagonal([8.0, 8.0, 10.0])))
        frac = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 0.0 0.25 0.5 0.75]
        chain = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])
        sg = analyze(SpglibBackend(), chain)
        nl = SLCE.build_neighbor_list(chain, 2.6)   # nearest neighbors only
        cs = SLCE.build_clusters(chain, nl, sg; nbody = 2)
        # full basis including anisotropic (Lf>0) channels
        basis = SLCE.build_salc_basis(chain, sg, cs; lmax_by_species = [2])
        @test any(s -> s.Lf > 0, basis.salcs)   # anisotropic channels present

        rng = MersenneTwister(5)
        randcfg() = reduce(hcat, [SVector{3,Float64}((v = randn(rng, 3); v / norm(v))) for _ = 1:4])

        # invariance under the real space group (all Lf, non-collinear spins)
        for _ = 1:5
            e = randcfg()
            for g = 1:length(sg.ops)
                R = Matrix(sg.ops[g].rotation_cart)
                ge = similar(e)
                for a = 1:4
                    b = findfirst(==(a), @view sg.map_sym[:, g])
                    ge[:, a] = R * e[:, b]
                end
                for s in basis.salcs
                    @test isapprox(evaluate_salc(s, ge), evaluate_salc(s, e); atol = 1e-8, rtol = 1e-7)
                end
            end
        end

        # the isotropic (1,1)/Lf=0 SALC is the Heisenberg invariant. Canonical members
        # are one per undirected bond (the two directed images fold together), so
        # Φ = 2√3 Σ_members e_i·e_j.
        heis = only(filter(s -> SLCE.spin_ls(s.key) == [1, 1] && s.Lf == 0, basis.salcs))
        for _ = 1:10
            e = randcfg()
            ref = 2 * sqrt(3.0) *
                  sum(dot(e[:, m.atoms[1]], e[:, m.atoms[2]]) for m in heis.members)
            @test isapprox(evaluate_salc(heis, e), ref; atol = 1e-9, rtol = 1e-8)
        end
    end

    @testset "Heisenberg chain end-to-end: recover J (M8/M9)" begin
        Lattice = SLCE.Lattice
        Crystal = SLCE.Crystal
        SpglibBackend = SLCE.SpglibBackend

        lat = Lattice(Matrix(Diagonal([8.0, 8.0, 10.0])))
        frac = [zeros(2, 4); [0.0 0.25 0.5 0.75]]
        chain = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])
        interaction = SLCE.BasisSpec(; nbody = 2, cutoff = 2.6,
                                                 lmax = [1], isotropy = true)
        basis = SLCE.SLCEBasis(chain, interaction; backend = SpglibBackend())
        @test length(basis.salc_basis) == 1          # only the nearest-neighbor Heisenberg SALC
        heis = basis.salc_basis.salcs[1]

        rng = MersenneTwister(3)
        cfg() = (M = Matrix{Float64}(undef, 3, 4);
                 for a = 1:4; v = randn(rng, 3); M[:, a] = v / norm(v); end; M)
        J_true = 0.0137
        configs = [cfg() for _ = 1:30]
        # physical Heisenberg energies E = J Σ_{undirected nn} e_i·e_j (canonical
        # members are exactly the undirected nearest-neighbor bonds)
        E = [J_true * sum(dot(c[:, m.atoms[1]], c[:, m.atoms[2]]) for m in heis.members)
             for c in configs]

        ds = SLCE.SLCEDataset(basis, configs, E)
        f = SLCE.fit(SLCE.SLCEFit, ds, SLCE.OLS())
        @test SLCE.r2_energy(f) ≈ 1.0 atol = 1e-10
        @test isapprox(SLCE.predict_energy(f, configs), E; atol = 1e-10)
        # recover J from the SALC coefficient: Φ = 2√3·Σ_{undirected} e_i·e_j
        J_recovered = 2 * sqrt(3.0) * SLCE.coef(f)[1]
        @test isapprox(J_recovered, J_true; rtol = 1e-8)

        # torque from the fitted model vs the Heisenberg closed form. With
        # E = J·Σ_members e_i·e_j, ∂E/∂e_a = J·Σ_members(δ_{a,i} e_j + δ_{a,j} e_i)
        # and the physical / Landau–Lifshitz torque τ_a = −e_a × ∂E/∂e_a = ∂E/∂e_a × e_a.
        function analytic_torque(c, J)
            nat = size(c, 2)
            G = zeros(3, nat)
            for mem in heis.members
                i, j = mem.atoms[1], mem.atoms[2]
                G[:, i] .+= J .* c[:, j]
                G[:, j] .+= J .* c[:, i]
            end
            return reduce(hcat, cross(SVector{3}(G[:, a]), SVector{3}(c[:, a])) for a = 1:nat)
        end
        for c in configs[1:5]
            @test isapprox(SLCE.predict_torque(f, c), analytic_torque(c, J_true);
                           atol = 1e-9)
        end
    end

    @testset "N-body SALC dimensions vs Magesty (kagome, 1/2/3-body)" begin
        # A kagome cell (P6/mmm): three equivalent sites whose triangle has C₃ᵥ site
        # symmetry permuting them — exercises coupling-path mixing and, for unequal-l
        # multisets like (1,1,2), l-ordering mixing. The number of independent
        # invariants per (body, ls, Lf) channel is convention-free, so the two
        # from-scratch constructions must agree exactly. This validates *both*
        # implementations (and would expose a bug in either) at N ≥ 3.
        a, c = 2.0, 8.0
        A = [a -a/2 0.0; 0.0 a*sqrt(3)/2 0.0; 0.0 0.0 c]
        frac = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.0 0.0 0.0]
        kd = [1, 1, 1]
        cutoff, nbody = 1.2, 3

        # Magesty: count SALCs per channel by key-group (= design-matrix columns),
        # filtered to per-site l ≤ 2 to match the rebuild's subset.
        st = Magesty.Structures.Structure(A, SVector{3,Bool}(true, true, true), ["Fe"], kd,
                                          frac; verbosity = false)
        sym = Magesty.Symmetries.Symmetry(st, 1e-5; verbosity = false)
        cutarr = OffsetArray(fill(cutoff, nbody - 1, 1, 1), 2:nbody, 1:1, 1:1)
        clu = Magesty.Clusters.Cluster(st, sym, nbody, cutarr; verbosity = false)
        sb = Magesty.SALCBases.SALCBasis(st, sym, clu, [2], OffsetArray([4, 6], 2:nbody),
                                         nbody; isotropy = false, verbosity = false)
        magcount = Dict{Tuple{Int,Vector{Int},Int},Int}()
        for group in sb.salc_list
            cbc = group[1]
            all(l -> l <= 2, cbc.ls) || continue
            k = (length(cbc.atoms), sort(cbc.ls), cbc.Lf)
            magcount[k] = get(magcount, k, 0) + 1
        end

        # SLCE: count SALCs per channel.
        Lattice = SLCE.Lattice
        Crystal = SLCE.Crystal
        xtal = Crystal(Lattice(A), frac, kd, ["Fe"])
        sg = SLCE.analyze_symmetry(SLCE.SpglibBackend(), xtal)
        cs = SLCE.build_clusters(xtal, SLCE.build_neighbor_list(xtal, cutoff),
                                           sg; nbody = nbody)
        basis = SLCE.build_salc_basis(xtal, sg, cs; lmax_by_species = [2])
        mrcount = Dict{Tuple{Int,Vector{Int},Int},Int}()
        for s in basis.salcs
            k = (s.key.body, sort(SLCE.spin_ls(s.key)), s.key.Lf)
            mrcount[k] = get(mrcount, k, 0) + 1
        end

        @test sym.spacegroup_number == 191
        @test mrcount == magcount                       # every (body, ls, Lf) dimension agrees
        @test sum(values(mrcount)) == 42
        @test mrcount[(3, [1, 1, 2], 2)] == 5           # a multi-ordering 3-body channel

        # Independent ground truth on *Magesty*: its own 3-body SALCs are invariant.
        magsc(e) = Magesty.SpinConfigs.SpinConfig(0.0, ones(size(e, 2)), e, zeros(3, size(e, 2)))
        function act_mag(e, g)
            nat = size(e, 2)
            R = Matrix(sym.symdata[g].rotation_cart)
            out = similar(e)
            for x = 1:nat
                b = findfirst(==(x), @view sym.map_sym[:, g])
                out[:, x] = R * e[:, b]
            end
            return out
        end
        rng = MersenneTwister(42)
        rc() = reduce(hcat, [SVector{3,Float64}((v = randn(rng, 3); v / norm(v))) for _ = 1:3])
        for _ = 1:4
            e = Matrix(rc())
            Xe = Magesty.Fitting.build_design_matrix_energy(sb.salc_list, [magsc(e)], sym;
                                                           verbosity = false)
            for g = 1:sym.nsym
                Xg = Magesty.Fitting.build_design_matrix_energy(sb.salc_list,
                                                               [magsc(act_mag(e, g))], sym;
                                                               verbosity = false)
                @test isapprox(Xg, Xe; atol = 1e-7)
            end
        end
    end

    # The VASP POSCAR / OSZICAR parsers moved to SLCETools.jl; their bit-for-bit cross-check
    # against Magesty lives in `SLCETools.jl/test/oracle/`.

    @testset "EMBSET reader vs Magesty" begin
        # Same file through both readers: energies, unit directions, magnitudes, and the
        # constraining field must agree bit-for-bit (both split m into ‖m‖ · e). The
        # torque convention is NOT compared — each package derives its own from (m, B).
        rng = MersenneTwister(7)
        nat, nconf = 5, 3
        dir = mktempdir()
        path = joinpath(dir, "EMBSET")
        open(path, "w") do io
            for c = 1:nconf
                println(io, "# config $c")
                println(io, " ", -10.0 - c)
                for a = 1:nat
                    m = a == 4 ? zeros(3) : randn(rng, 3)      # one non-magnetic atom
                    B = 0.1 .* randn(rng, 3)
                    println(io, " $a  ", join(m, " "), "  ", join(B, " "))
                end
            end
        end
        ours = SLCE.read_embset(path)
        theirs = Magesty.read_embset(path)
        @test length(ours) == length(theirs) == nconf
        for c = 1:nconf
            @test ours[c].energy == theirs[c].energy
            @test ours[c].magmoms == theirs[c].magmom_size
            @test ours[c].directions == theirs[c].spin_directions
            @test ours[c].field == theirs[c].local_magfield
        end
    end

    @testset "fit parity: same data + same span ⇒ same held-out predictions" begin
        # End-to-end cross-check on the anisotropic chain: both packages build a basis
        # with the SAME channel content, train on one shared EMBSET, fit OLS, and must
        # return the same predictions on held-out configurations. Raw SALC coefficients
        # are gauge-dependent and are NOT compared; but with equal spans and a full-rank
        # design the fitted least-squares *function* is unique, so any prediction gap
        # is a real convention divergence (harmonics, folding, design assembly, torque
        # gradient, or the OLS objective), not gauge.
        #
        # Channel-content mapping (the truncation semantics differ):
        #   Magesty:    body1 lmax = 2 (even l only), body2 lsum = 4 with per-site l
        #               uncapped (partitions of total l ∈ {2, 4} over 2 sites)
        #   SLCE: per-site lmax applies to every body order, so lmax = [3]
        #               (= lsum₂ − 1, leaves body 2 uncapped) + per-body
        #               lsum = [2, 4] (body 1: even l ≤ 2 ⇒ {2})
        # Both give body-1 {2} and body-2 {(1,1), (1,3), (2,2), (3,1)}.
        lat_m = Matrix(Diagonal([8.0, 8.0, 10.0]))
        frac = [zeros(2, 4); [0.0 0.25 0.5 0.75]]
        rcut = 2.6

        chain = SLCE.Crystal(SLCE.Lattice(lat_m), frac, [1, 1, 1, 1], ["Fe"])
        spec = SLCE.BasisSpec(; nbody = 2, cutoff = rcut, lmax = [3],
                                    lsum = [1 => 2, 2 => 4], isotropy = false)
        ours_basis = SLCE.SLCEBasis(chain, spec;
                                         backend = SLCE.SpglibBackend())
        theirs_basis = Magesty.SCEBasis(;
            lattice = lat_m, kd = [:Fe], kd_list = [1, 1, 1, 1],
            positions = [frac[:, a] for a = 1:4],
            interaction = (body1 = (lmax = Dict(:Fe => 2),),
                           body2 = (lsum = 4, cutoff = Dict((:Fe, :Fe) => rcut))),
            isotropy = false, verbosity = false)
        @test SLCE.n_salcs(ours_basis) ==
              length(theirs_basis.salcbasis.salc_list)

        # Shared EMBSET: unit-magnitude moments so the torque target m × B is
        # magnitude-convention-free, random constraining fields, random energies
        # (parity needs no physical consistency — both sides fit the same targets).
        rng = MersenneTwister(23)
        nat, nconf = 4, 80
        runit() = (v = randn(rng, 3); v ./ norm(v))
        dir = mktempdir()
        path = joinpath(dir, "EMBSET")
        open(path, "w") do io
            for c = 1:nconf
                println(io, "# config $c")
                println(io, " ", -25.0 + 0.1 * randn(rng))
                for a = 1:nat
                    println(io, " $a  ", join(runit(), " "), "  ",
                            join(0.2 .* randn(rng, 3), " "))
                end
            end
        end
        ours_ds = SLCE.SLCEDataset(ours_basis, SLCE.EmbsetFile(path))
        theirs_ds = Magesty.SCEDataset(theirs_basis, path)

        heldout = [reduce(hcat, [runit() for _ = 1:nat]) for _ = 1:5]

        # The endpoints w = 0 (energy-only) and w = 1 (torque-only) are single-block
        # least squares, so the two packages' block-whitening conventions drop out of
        # the argmin; intermediate w would additionally pin the whitening convention
        # and is deliberately not asserted here.
        f0_ours = SLCE.fit(SLCE.SLCEFit, ours_ds, SLCE.OLS();
                                 torque_weight = 0.0)
        f0_theirs = Magesty.fit(Magesty.SCEFit, theirs_ds, Magesty.OLS();
                                torque_weight = 0.0)
        for c in heldout
            @test isapprox(SLCE.predict_energy(f0_ours, c),
                           Magesty.predict_energy(f0_theirs, c);
                           rtol = 1e-6, atol = 1e-10)
        end

        f1_ours = SLCE.fit(SLCE.SLCEFit, ours_ds, SLCE.OLS();
                                 torque_weight = 1.0)
        f1_theirs = Magesty.fit(Magesty.SCEFit, theirs_ds, Magesty.OLS();
                                torque_weight = 1.0)
        # Known, deliberate convention difference: Magesty's torque target and
        # prediction use `τ = −m (e × B)` (see its SpinConfig docstring), while
        # SLCE flipped to the physical / Landau–Lifshitz `τ = m × B`
        # (2026-06-28). Both sides flip target AND predictor, so the fitted
        # function is identical — the intercept and the energy surface agree
        # directly, and the reported torque agrees up to the global sign.
        @test isapprox(f1_ours.j0, Magesty.intercept(f1_theirs);
                       rtol = 1e-10, atol = 1e-12)
        for c in heldout
            @test isapprox(SLCE.predict_energy(f1_ours, c),
                           Magesty.predict_energy(f1_theirs, c);
                           rtol = 1e-6, atol = 1e-10)
            @test isapprox(SLCE.predict_torque(f1_ours, c),
                           -Magesty.predict_torque(f1_theirs, c);
                           rtol = 1e-6, atol = 1e-10)
        end
    end
end
