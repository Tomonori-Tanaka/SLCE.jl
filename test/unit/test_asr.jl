# ASR (acoustic sum rule) — gate (k) and the §6 amendments: the symbolic
# constraint builder against the numerical translation image (the independent
# rank count), the null-space machinery (orthonormality, forbidden band, zero-row
# subselection), fit/refit under the exact constraint (translation invariance and
# Σf = 0 at physical t, AFTER refit too), the unconstrained-violation
# demonstration, pure-spin bitwise identity, and the diagnostics (dof/gcv) in the
# reparameterized space.

using Test
using SLCE
using SLCE: build_asr, _asr_nullspace, _assemble_problem, _gcv_score,
    accumulate_grad!, salcs, ASRReparam
using SLCE.SolidHarmonics: solid_harmonic_poly, poly_eval, Rlm
using LinearAlgebra
using Random

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_as_unit(rng) = normalize(randn(rng, 3))
_as_cfg(rng, nat) = reduce(hcat, [_as_unit(rng) for _ = 1:nat])

# Numerical translation-image matrix through the production gradient kernel:
# rows = 3 Cartesian generator components at M random (e, u) points, columns =
# SALCs. Its rank is the algorithm-independent count rank(A) is gated against,
# and its null space must agree with Z as a subspace.
function _numeric_translation_image(basis, M, rng)
    ss = salcs(basis)
    nat = n_atoms(basis.crystal)
    B = zeros(3 * M, length(ss))
    for t = 1:M
        e = _as_cfg(rng, nat)
        u = randn(rng, 3, nat)
        for j in eachindex(ss)
            Ge = zeros(3, nat)
            Gu = zeros(3, nat)
            accumulate_grad!(Ge, Gu, ss[j], e, u, 1.0)
            B[(3 * (t - 1) + 1):(3 * t), j] .= vec(sum(Gu; dims = 2))
        end
    end
    return B
end

function _svd_rank(B)
    s = svdvals(B)
    (isempty(s) || s[1] == 0.0) && return 0
    return count(>(maximum(size(B)) * eps() * s[1]), s)
end

@testset "ASR null-space slice (gate k)" begin
    # D4h Fe–Fe bond cell; pmax = 2 admits the on-site partners the difference
    # invariants need (63 feasible directions), pmax = 1 admits none (the loud
    # zero-nullity case).
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    mkspec(pmax) = BasisSpec(cr; lmax = 1, pmax = pmax, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)])
    basis = SLCEBasis(cr, mkspec(2))
    m = n_salcs(basis)
    nat = n_atoms(cr)
    rng = MersenneTwister(0xA5)

    @testset "solid_harmonic_poly ≡ kernel (builder's symbolic side)" begin
        # The bound is scaled to the evaluation's own conditioning. `poly_eval`
        # sums signed monomials, so its error is bounded by `n·eps·Σ|terms|` — and
        # the cancellation `Σ|terms| / |result|` reaches 4e5 here, which is a
        # property of the polynomial form, not a defect. A flat `rtol = 1e-12`
        # (what this used to be) is therefore two things at once: far too loose
        # where the sum is benign, and WRONG where it is not — at 200 draws per
        # (k, l, m) the plain relative error reaches 4.8e-11, 48× over that rtol.
        # It only ever passed because three fixed draws never hit those points.
        #
        # Against `eps·Σ|terms|` the residual is bounded and small: measured max
        # 20 over three seeds × 2000 draws × |u| spanning 0.05–20, so 128 leaves
        # ~6× headroom while being far tighter than the old rtol wherever the
        # evaluation is well conditioned.
        for l = 0:10, mm = -l:l, k = 0:2
            p = solid_harmonic_poly(k, l, mm)
            @test all(key -> sum(key) == 2 * k + l, keys(p))   # homogeneous
            for _ = 1:20
                u = randn(rng, 3) .* rand(rng, (0.05, 1.0, 20.0))
                r2k = (u[1]^2 + u[2]^2 + u[3]^2)^k
                mag = sum(abs(c) * abs(u[1]^ky[1] * u[2]^ky[2] * u[3]^ky[3])
                          for (ky, c) in p)
                @test abs(poly_eval(p, u) - r2k * Rlm(l, mm, u)) <= 128 * eps() * mag
            end
        end
    end

    rep = build_asr(basis)
    @testset "builder: rank vs the numerical translation image" begin
        @test rep isa ASRReparam
        @test size(rep.Z, 2) == m - rep.rank
        @test maximum(abs, rep.Z' * rep.Z - I) < 1e-12        # orthonormal
        @test maximum(abs, rep.A * rep.Z) < 1e-12             # AZ ≈ 0
        @test all(iszero, rep.beta_p)
        # independent count: the numerical image through accumulate_grad!
        B = _numeric_translation_image(basis, 4 * m, rng)
        @test _svd_rank(B) == rep.rank
        # subspace agreement: every Z column is annihilated by the numerical image
        @test maximum(abs, B * rep.Z) / maximum(abs, B) < 1e-10
        # pure-spin columns are untouched (identity block ⊂ Z's column space)
        pcols = findall(s -> !any(SLCE.has_disp, s.key.decors), salcs(basis))
        for c in pcols
            @test all(iszero, @view rep.A[:, c])
        end
        # hand-counted fixture: pure-disp degree 1:2 on the bond — 15 SALCs, 3
        # translation-invariant combinations survive (the difference-vector forms)
        b1 = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 1, sectors = [
            Sector(disp = (degree = 1:2,), sites = 1:2, cutoff = 1.1)]))
        r1 = build_asr(b1)
        @test n_salcs(b1) == 15 && r1.rank == 12 && size(r1.Z, 2) == 3
        B1 = _numeric_translation_image(b1, 60, rng)
        @test _svd_rank(B1) == 12
        @test maximum(abs, B1 * r1.Z) / maximum(abs, B1) < 1e-10
        # zero-nullity truncation (pmax = 1: pair (1,1) splits without on-site
        # partners) is correct AND loud
        b1x = SLCEBasis(cr, mkspec(1))
        bz = @test_logs (:warn, r"no translation-invariant") build_asr(b1x)
        @test size(bz.Z, 2) == count(s -> !any(SLCE.has_disp, s.key.decors),
                                     salcs(b1x))

        # --- group_freedom and the structural cost discount ---------------------
        # s_g = ‖Z[g,:]‖_F² is the group-resolved share of the q feasible directions.
        # Σ_g s_g ≡ q is exact (Z orthonormal), and it is the invariant that makes the
        # quantity gauge-independent — it is tr of the projector Z·Z', not of Z.
        lab_a = SLCE.salc_groups(basis)
        s_a = SLCE.group_freedom(rep, lab_a)
        @test sum(s_a) ≈ size(rep.Z, 2) rtol = 1e-12
        pg_a = [count(==(g), lab_a) for g in eachindex(s_a)]
        @test all(0 .<= s_a .<= pg_a .+ 1e-12)
        # gauge invariance: an orthogonal remix of Z's columns is an equally valid
        # null-space basis and must give the same s_g
        qq = size(rep.Z, 2)
        Qr = qr(randn(MersenneTwister(7), qq, qq)).Q * Matrix(I, qq, qq)
        repg = ASRReparam(rep.A, rep.Z * Qr, rep.beta_p, rep.rank)
        @test SLCE.group_freedom(repg, lab_a) ≈ s_a rtol = 1e-10
        # one label per column ⇒ s_g is the per-column row norm²
        @test SLCE.group_freedom(rep, collect(1:n_salcs(basis))) ≈
              [sum(abs2, @view rep.Z[j, :]) for j in axes(rep.Z, 1)] rtol = 1e-12
        @test_throws ArgumentError SLCE.group_freedom(rep, ones(Int, 3))

        # On the pmax = 1 truncation the ASR kills the displacement content outright,
        # so its group holds none of the freedom — and must not be charged for a
        # Monte-Carlo sweep it can never trigger.
        lab_z = SLCE.salc_groups(b1x)
        s_z = SLCE.group_freedom(bz, lab_z)
        @test sum(s_z) ≈ size(bz.Z, 2) rtol = 1e-12
        dead_g = findall(<(1e-12), s_z)
        @test !isempty(dead_g)                          # the fixture is the dead case
        c_plain = SLCE.group_costs(b1x, lab_z)
        c_disc = SLCE.group_costs(b1x, lab_z; asr = bz)
        @test all(iszero, c_disc[dead_g]) && all(>(0), c_plain[dead_g])
        live_g = setdiff(eachindex(s_z), dead_g)
        @test c_disc[live_g] == c_plain[live_g]         # live groups untouched
        @test sum(c_disc) < 0.1 * sum(c_plain)          # the dead group dominated
        # no dead group ⇒ the discount is the identity
        @test SLCE.group_costs(basis, lab_a; asr = rep) ==
              SLCE.group_costs(basis, lab_a)
        @test_throws DimensionMismatch SLCE.group_costs(b1x, lab_z; asr = rep)

        # The ASR's own granularity is NOT the group: no displacement-touched group
        # has a feasible subspace on its own, which is why a small s_g is a diagnostic
        # and not a licence to drop the group independently.
        for g in eachindex(s_a)
            cols = findall(==(g), lab_a)
            any(!iszero, @view rep.A[:, cols]) || continue
            @test size(_asr_nullspace(rep.A[:, cols])[1], 2) == 0
        end
        # pure-spin basis: nothing (the structural fast path)
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
        @test build_asr(bspin) === nothing
        # residue pruning: within each row, surviving entries clear the relative
        # cut (cancellation residue became exact zeros — review blocker)
        for r in axes(rep.A, 1)
            nz = [abs(rep.A[r, j]) for j in axes(rep.A, 2) if rep.A[r, j] != 0.0]
            @test minimum(nz) > 1e-12 * maximum(nz)
        end
        # component grading: every constraint row couples only columns sharing
        # one (sorted spin-l multiset, total displacement degree) label
        collabel = map(salcs(basis)) do s
            (sort([d.spin_l for d in s.key.decors if SLCE.has_spin(d)]),
             sum(2 * d.disp_k + d.disp_l for d in s.key.decors))
        end
        for r in axes(rep.A, 1)
            labs = unique(collabel[j] for j in axes(rep.A, 2) if rep.A[r, j] != 0.0)
            @test length(labs) == 1
        end
    end

    @testset "_asr_nullspace unit behavior" begin
        # forbidden band: ambiguous singular value must refuse, never guess.
        # (Both rows must touch the same connected component — the rank decision
        # normalizes σ per component — and rows are 2-norm-normalized internally,
        # so the ambiguity is crafted as two nearly dependent rows.)
        @test_throws ArgumentError _asr_nullspace([1.0 1.0; 1.0 1.0 + 1e-10])
        # clean gap: rank 1, nullity 1
        Z2, r2 = _asr_nullspace([1.0 1.0])
        @test r2 == 1 && size(Z2, 2) == 1
        @test abs(abs(Z2[1, 1]) - 1 / sqrt(2)) < 1e-12        # the difference gauge
        # column subselection can zero whole rows — they impose nothing
        Zs, rs = _asr_nullspace([0.0 0.0; 1.0 -1.0])
        @test rs == 1 && size(Zs, 2) == 1
        # untouched columns become identity blocks
        Zi, ri = _asr_nullspace([1.0 1.0 0.0])
        @test ri == 1 && size(Zi, 2) == 2
        @test any(k -> Zi[:, k] == [0.0, 0.0, 1.0], axes(Zi, 2))
        # column subselection must not promote residue to a constraint (review
        # blocker): empty a row's real columns and compare against the cleaned
        # rank computed independently
        deadfull = Set(j for j = 1:m if norm(@view rep.Z[j, :]) < 1e-12)
        hit = false
        for r in axes(rep.A, 1)
            rmax = maximum(abs, @view rep.A[r, :])
            big = [j for j = 1:m if abs(rep.A[r, j]) > 0.1 * rmax]
            (length(big) >= 2 && !(big[1] in deadfull)) || continue
            S = sort(setdiff(1:m, big[2:end]))
            Asub = rep.A[:, S]
            Z_S, rank_S = _asr_nullspace(Asub)
            # independent cleaned rank: drop near-zero rows, plain SVD
            rn = [norm(@view Asub[rr, :]) for rr in axes(Asub, 1)]
            keeprows = findall(>(1e-12 * maximum(rn)), rn)
            @test rank_S == _svd_rank(Asub[keeprows, :])
            @test size(Z_S, 2) == length(S) - rank_S
            # the lone survivor of row r is support-induced dead (its partners
            # were dropped), while the full constraint kept it alive — exactly
            # the case refit's warning is scoped to
            k = findfirst(==(big[1]), S)
            if norm(@view Z_S[k, :]) < 1e-9
                hit = true
            end
            break
        end
        @test hit
        # ASRReparam invariants
        @test_throws DimensionMismatch ASRReparam(zeros(1, 3), zeros(2, 2),
                                                  zeros(3), 1)
        @test_throws DimensionMismatch ASRReparam(zeros(1, 3), zeros(3, 3),
                                                  zeros(2), 1)
        # a NARROWER Z than p − rank is legal: that is a staged fit (fewer free
        # columns); only a Z wider than the space it maps into is a shape error
        @test ASRReparam(zeros(1, 3), zeros(3, 1), zeros(3), 1) isa ASRReparam
        @test_throws DimensionMismatch ASRReparam(zeros(1, 3), zeros(3, 4),
                                                  zeros(3), 1)
    end

    # --- fit-level gates -----------------------------------------------------
    beta_true = rep.Z * (rep.Z' * randn(rng, m))       # feasible ground truth
    model = SLCEModel(basis, 0.2, beta_true)
    @test asr_residual(model) < 1e-13
    fpc = crystal_fingerprint(cr)
    prov = DatumProvenance(; torque_qualified = true, reference_id = "r",
                           reference_fingerprint = fpc, setup_id = "s")
    function mkdatum(mdl; noise = 0.0)
        e = _as_cfg(rng, nat)
        u = 0.08 * randn(rng, 3, nat)
        TrainingDatum(; energy = predict_energy(mdl, e, u) + noise * randn(rng),
                      directions = e, magmoms = ones(nat), displacements = u,
                      forces = predict_force(mdl, e, u),
                      torques = predict_torque(mdl, e, u), provenance = prov)
    end
    data = [mkdatum(model) for _ = 1:70]
    ds = SLCEDataset(basis, data)
    @test ds.asr isa ASRReparam && ds.asr.rank == rep.rank
    @test ds[3:20].asr === ds.asr                       # slicing carries it
    @test vcat(ds[1:30], ds[31:70]).asr === ds.asr      # vcat carries it

    f = fit(SLCEFit, ds, OLS(); torque_weight = 0.3, force_weight = 0.4)
    mfit = SLCEModel(f)
    e0 = _as_cfg(rng, nat)
    u0 = 0.08 * randn(rng, 3, nat)
    tvec = [0.11, -0.07, 0.05]
    Escale = max(abs(predict_energy(mfit, e0, u0)), 1.0)
    @testset "gate (k): exact recovery, translation invariance, Σf = 0" begin
        @test maximum(abs, f.jphi .- beta_true) < 1e-8
        @test f.asr && f.asr_residual < 1e-13           # the smoke-test residual
        @test abs(predict_energy(mfit, e0, u0 .+ tvec) -
                  predict_energy(mfit, e0, u0)) / Escale < 1e-12
        @test maximum(abs, sum(predict_force(mfit, e0, u0); dims = 2)) < 1e-12
        @test asr_residual(mfit) < 1e-13
        @test dof(f) == (m - rep.rank) + 1              # q + 1 free parameters
        @test isfinite(gcv(f)) && effective_dof(f) <= m - rep.rank + 1 + 1e-9
        # dense-hat-matrix gate for the constrained effective dof: the Cholesky-
        # congruence _edof_ns must reproduce tr[X̃(X̃'X̃ + λ·Z'DZ)⁻¹X̃'] + 1,
        # and reduce to _edof at Z = I
        fRg = fit(SLCEFit, ds, Ridge(1e-6); torque_weight = 0.3)
        Xg, _, _, _, _ = _assemble_problem(ds, 0.3, 0.0, ds.asr)
        lamg, wg = SLCE._penalty_diagonal(fRg.estimator, fRg.jphi)
        P = Symmetric(ds.asr.Z' * (wg .* ds.asr.Z))
        H = Xg * ((Symmetric(Xg' * Xg) + lamg * P) \ Xg')
        @test effective_dof(fRg) ≈ tr(H) + 1 rtol = 1e-8
        q = size(ds.asr.Z, 2)
        @test SLCE._edof_ns(Xg, lamg, ones(q), Matrix{Float64}(I, q, q)) ≈
              SLCE._edof(Xg, lamg, ones(q)) rtol = 1e-10
    end

    @testset "violation demonstration (asr = false) and noisy contrast" begin
        noisy = [mkdatum(model; noise = 1e-3) for _ = 1:70]
        dsn = SLCEDataset(basis, noisy)
        fu = fit(SLCEFit, dsn, OLS(); torque_weight = 0.3, force_weight = 0.4,
                 asr = false)
        fc = fit(SLCEFit, dsn, OLS(); torque_weight = 0.3, force_weight = 0.4)
        @test !fu.asr && fu.asr_residual == 0.0
        mu = SLCEModel(fu)
        mc = SLCEModel(fc)
        @test asr_residual(mu) > 1e-8                   # noise leaks into A-space
        @test asr_residual(mc) < 1e-13
        @test abs(predict_energy(mu, e0, u0 .+ tvec) -
                  predict_energy(mu, e0, u0)) > 1e-8    # translation broken
        @test abs(predict_energy(mc, e0, u0 .+ tvec) -
                  predict_energy(mc, e0, u0)) / Escale < 1e-12
        @test maximum(abs, sum(predict_force(mu, e0, u0); dims = 2)) > 1e-8
        @test maximum(abs, sum(predict_force(mc, e0, u0); dims = 2)) < 1e-12
    end

    @testset "refit re-derives the support null space (post-refit gates)" begin
        # threshold = 0 keeps every column: Z_S ≡ Z, so no support-induced dead
        # columns exist and refit must stay SILENT (basis-level dead columns are
        # build_asr's report, not the support's doing)
        fr = @test_logs refit(f)
        @test maximum(abs, fr.jphi .- beta_true) < 1e-8
        @test fr.asr && fr.asr_residual < 1e-13
        mr = SLCEModel(fr)
        @test abs(predict_energy(mr, e0, u0 .+ tvec) -
                  predict_energy(mr, e0, u0)) / Escale < 1e-12
        @test maximum(abs, sum(predict_force(mr, e0, u0); dims = 2)) < 1e-12
        # a support that keeps only ONE side of a coupled pair structurally
        # zeroes the survivor — force it by thresholding away most columns and
        # checking the refit STILL satisfies the constraint
        fr2 = refit(f; threshold = 0.5 * maximum(abs.(f.jphi)))
        @test fr2.asr_residual < 1e-13
        mr2 = SLCEModel(fr2)
        @test abs(predict_energy(mr2, e0, u0 .+ tvec) -
                  predict_energy(mr2, e0, u0)) / max(abs(fr2.j0), 1.0) < 1e-10
    end

    @testset "constrained estimators (β-space penalties through Z)" begin
        # Ridge at tiny λ recovers; the adaptive members shrink near-zero
        # components harder (w → 1/ε), so they are held to feasibility +
        # prediction accuracy, not coefficient identity.
        for est in (Ridge(1e-9), AdaptiveRidge(; lambda = 1e-9),
                    GroupAdaptiveRidge(collect(1:m), ones(m); lambda = 1e-9))
            fe = fit(SLCEFit, ds, est; torque_weight = 0.3)
            @test fe.asr_residual < 1e-12               # feasible by construction
            @test maximum(abs, fe.jphi .- beta_true) < 1e-2
            @test rmse_energy(fe) < 1e-4
        end
        fR = fit(SLCEFit, ds, Ridge(1e-9); torque_weight = 0.3)
        @test maximum(abs, fR.jphi .- beta_true) < 1e-3
        # FixedCoefficients carries β-space coefficients — undefined under ASR
        @test_throws ArgumentError fit(SLCEFit, ds, FixedCoefficients(beta_true))
        fp = fit(SLCEFit, ds, FixedCoefficients(beta_true); asr = false)
        @test fp.jphi == beta_true
    end

    @testset "pure-spin bitwise identity and fences" begin
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
        ms = SLCEModel(bspin, 0.0, randn(rng, n_salcs(bspin)))
        cfgs = [_as_cfg(rng, nat) for _ = 1:20]
        dss = SLCEDataset(bspin, cfgs, predict_energy(ms, cfgs))
        @test dss.asr === nothing
        fa = fit(SLCEFit, dss, OLS())
        fb = fit(SLCEFit, dss, OLS(); asr = false)
        @test fa.jphi == fb.jphi && fa.j0 == fb.j0      # bitwise
        @test !fa.asr && fa.asr_residual == 0.0
        @test asr_residual(SLCEModel(fa)) == 0.0
        # Selection-layer fences on the joint basis. `select_fit` still refuses under
        # the DEFAULT `asr = true`: its λ path solves on an unconstrained Gram, so the
        # support it reports is not the one the constrained solve chose.
        gar = GroupAdaptiveRidge(collect(1:m), ones(m); lambda = 1e-6)
        @test_throws ArgumentError select_fit(ds, gar; lambdas = [1e-6])
        # `select_support` does NOT: it drives `refit`, which re-derives the null space
        # on each support, and the front's alive/cost columns are read back off the
        # returned refits. Every point must stay translation-invariant.
        sc = select_support(f; npoints = 5)
        @test length(sc.threshold) >= 1
        @test sc.selected in eachindex(sc.threshold)
        lab_c = SLCE.salc_groups(basis)
        cg_c = SLCE.group_costs(basis, lab_c; asr = ds.asr)
        for i in eachindex(sc.threshold)
            # the point's fit is exactly `refit` at that threshold, so re-deriving the
            # row here is the same computation the front reports
            fri = refit(f, OLS(); threshold = sc.threshold[i])
            ag = unique(lab_c[findall(!=(0.0), fri.jphi)])
            @test sc.n_alive[i] == length(ag)
            @test sc.cost[i] == sum(cg_c[ag]; init = 0.0)
            # §6 amendment 4: the invariance gates hold AFTER selection too
            mi = SLCEModel(fri)
            @test asr_residual(mi) < 1e-10
            @test maximum(abs, sum(predict_force(mi, e0, u0); dims = 2)) < 1e-10
        end
        @test sc.fit.jphi == refit(f, OLS();
                                   threshold = sc.threshold[sc.selected]).jphi
        # ...but a DELIBERATELY unconstrained joint fit selects end to end. This is
        # the path the constrained one is measured against later; it also exercises
        # `group_costs` on a displacement-decorated basis, which used to be refused.
        pu = select_fit(ds, gar; lambdas = [1e-4, 1e-6], asr = false)
        @test !pu.fit.asr && pu.fit.reparam === nothing
        @test pu.selected in eachindex(pu.lambda)
        @test all(>(0), pu.cost)
        fu = fit(SLCEFit, ds, OLS(); torque_weight = 0.3, asr = false)
        su = select_support(fu; npoints = 4)
        @test length(su.threshold) >= 1 && all(>=(0), su.cost)
        # A plain joint fit is NOT a staged fit — the predicate that decides which
        # refusal fires. Before `_is_staged` this reported "staged (frozen /
        # sector_mask)" for an ordinary `fit(...)` call.
        @test !SLCE._is_staged(f) && f.reparam === ds.asr
        @test !SLCE._is_staged(fu) && fu.reparam === nothing
        fstg = fit(SLCEFit, ds, OLS(); torque_weight = 0.3, sector_mask = :spin)
        @test SLCE._is_staged(fstg)
        @test occursin("staged", sprint(showerror,
            try select_support(fstg) catch e; e end))
        # A constrained fit with no force weight also selects; the ASR refusal that
        # used to fire here is gone, and only the staged one remains.
        fasr = fit(SLCEFit, ds, OLS(); torque_weight = 0.3)
        @test !SLCE._is_staged(fasr) && fasr.reparam === ds.asr
        @test select_support(fasr; npoints = 3) isa SupportPath
        # cross_validate runs constrained per fold; asr = false threads through
        cv = cross_validate(ds, OLS(); torque_weight = 0.3, nfolds = 3)
        @test cv.pooled_rmse_energy < 1e-10
        cvu = cross_validate(ds, OLS(); torque_weight = 0.3, nfolds = 3,
                             asr = false)
        @test cvu.pooled_rmse_energy < 1e-8       # in-span either way; both run
        # GCV n excludes the zero-weight energy block at w_T = 1 (regression for
        # the _gcv_neff fix, which also applies to pure-spin fits): reproduce
        # gcv(f) from the assembled problem with the corrected n by hand
        fw1 = fit(SLCEFit, ds, Ridge(1e-9); torque_weight = 1.0, asr = false)
        Xw, yw, _, _, _ = _assemble_problem(ds, 1.0, 0.0)
        neff = size(Xw, 1) - length(ds.y_E)
        lam, wv = SLCE._penalty_diagonal(fw1.estimator, fw1.jphi)
        @test gcv(fw1) == first(_gcv_score(Xw, yw, fw1.jphi, lam, wv;
                                           n_eff = neff))
        @test neff < size(Xw, 1)                  # the correction is active
        # the affine slot moves the known contribution to the target side:
        # ỹ = y − X_β·beta_p, applied BEFORE centering (so the centered block is
        # exactly the residual of the offset model). Gated against a hand-built
        # reparameterization here; the staged fits that produce one live in
        # test/unit/test_staged.jl.
        bp = 0.01 .* collect(1.0:m)
        repa = ASRReparam(rep.A, rep.Z, bp, rep.rank)
        Xa, ya, _, _, _ = _assemble_problem(ds, 0.0, 0.0, repa)
        X0, y0, _, _, _ = _assemble_problem(ds, 0.0, 0.0, rep)
        yoff = ds.X_E * bp
        @test Xa == X0                                   # the design is unchanged
        @test maximum(abs, ya .- (y0 .- (yoff .- sum(yoff) / length(yoff)))) < 1e-10
    end

    @testset "AllImages self-image clusters are refused" begin
        lat1 = Lattice(Matrix(2.0 * I(3)))
        xt1 = Crystal(lat1, zeros(3, 1), [1], ["Fe"])
        b1 = SLCEBasis(xt1, BasisSpec(xt1; lmax = 1, pmax = 1, sectors = [
                           Sector(spin = [1, 1], disp = (degree = 2,), sites = 2,
                                  cutoff = 2.1)]);
                       images = AllImages())
        hasself = any(s -> any(mem -> !allunique(mem.atoms), s.members), salcs(b1))
        @test hasself                # the fixture must actually exercise the path
        @test_throws ArgumentError build_asr(b1)
    end
    # Every other fixture here is a 2-body bond. Third-order force constants are
    # what the displacement channel exists for (and what M4's `force_constants` /
    # `dynamical_matrix` will read), and a 3-body cluster is the first case where one
    # constraint row couples THREE site blocks at once — the translation generator
    # has to cancel across all of them, not just a pair.
    @testset "3-body clusters: constraint, invariance and Σf = 0" begin
        lat3 = Lattice(Matrix(3.4 * I(3)))
        cr3 = Crystal(lat3, [0.0 0.3 0.0; 0.0 0.0 0.3; 0.0 0.0 0.0], [1, 1, 1], ["Fe"])
        # degree 1:3 is what admits a genuine 3-body displacement term: three sites
        # need three displacement slots, so a degree ≤ 2 budget silently has none.
        b3 = SLCEBasis(cr3, BasisSpec(cr3; lmax = 1, pmax = 3, sectors = [
            Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = 2.3),
            Sector(disp = (degree = 1:3,), sites = 1:3, cutoff = 2.3)]))
        s3 = salcs(b3)
        p3 = length(s3)
        nat3 = n_atoms(cr3)
        @test count(s -> s.key.body == 3, s3) > 0        # the fixture must have them
        # the disp-active columns span two different spin counts, so the constraint
        # rows are graded by spin content, not merely by displacement degree
        @test length(unique(count(SLCE.has_spin, s.key.decors)
                            for s in s3 if any(SLCE.has_disp, s.key.decors))) == 2

        rep3 = build_asr(b3)
        B3 = _numeric_translation_image(b3, 90, MersenneTwister(31))
        @test _svd_rank(B3) == rep3.rank                 # symbolic ≡ numerical
        @test maximum(abs, B3 * rep3.Z) / maximum(abs, B3) < 1e-10   # Z ⊆ null(image)

        # The gates that actually accept the constraint run through the PRODUCTION
        # evaluator, not through ‖Aβ‖ (which is ~eps by construction): a feasible
        # model's energy is invariant under a rigid translation of every
        # displacement, and its forces sum to zero on each configuration.
        rng3 = MersenneTwister(37)
        beta3 = rep3.Z * randn(rng3, size(rep3.Z, 2))
        m3 = SLCEModel(b3, 0.0, beta3)
        shift = [0.31, -0.72, 0.55]
        function _translation_dev(mdl, e, u)
            E0 = predict_energy(mdl, e, u)
            return maximum(abs(predict_energy(mdl, e, u .+ t .* shift) - E0)
                           for t in (1e-3, 1e-2, 5e-2))
        end
        pts = [(_as_cfg(rng3, nat3), 0.05 .* randn(rng3, 3, nat3)) for _ = 1:6]
        for (e, u) in pts
            @test _translation_dev(m3, e, u) < 1e-12
            @test norm(sum(predict_force(m3, e, u); dims = 2)) < 1e-12
        end
        @test asr_residual(m3) < 1e-13

        # teeth: an unconstrained model of the SAME basis breaks both, so the two
        # checks above are not vacuous at this body order
        mbad = SLCEModel(b3, 0.0, randn(MersenneTwister(41), p3))
        e0, u0 = pts[1]
        @test _translation_dev(mbad, e0, u0) > 1e-3
        @test norm(sum(predict_force(mbad, e0, u0); dims = 2)) > 1e-3

        # and the same gates after a fit AND after a refit (the constrained solve
        # and the support re-derivation each rebuild the null space)
        prov3 = DatumProvenance(; torque_qualified = true, reference_id = "r3",
                                reference_fingerprint = crystal_fingerprint(cr3),
                                setup_id = "s3")
        data3 = map(1:120) do _
            e = _as_cfg(rng3, nat3)
            u = 0.05 .* randn(rng3, 3, nat3)
            TrainingDatum(; energy = predict_energy(m3, e, u), directions = e,
                          magmoms = ones(nat3), displacements = u,
                          forces = predict_force(m3, e, u),
                          torques = predict_torque(m3, e, u), provenance = prov3)
        end
        ds3 = SLCEDataset(b3, data3)
        f3 = fit(SLCEFit, ds3, OLS(); torque_weight = 0.3, force_weight = 0.3)
        for g in (f3, refit(f3; threshold = 0.0))
            mg = SLCEModel(g)
            @test asr_residual(mg) < 1e-12
            @test _translation_dev(mg, e0, u0) < 1e-10
            @test norm(sum(predict_force(mg, e0, u0); dims = 2)) < 1e-10
        end
    end

    @testset "vcat refuses silent ASR disagreement" begin
        # a hand-built part without its `asr` stays combinable (it inherits the
        # carrying part's reparameterization only when no CONFLICTING one exists)
        bare = SLCEDataset(basis, [c for c in ds.configs[1:2]], ds.X_E[1:2, :],
                          ds.y_E[1:2], Matrix{Float64}(undef, 0, m), Float64[],
                          Int[], ds.provenance; disps = ds.disps[1:2])
        both = vcat(ds[1:5], bare)
        @test both.asr === ds.asr
        # two structurally different reparameterizations must refuse
        fake = ASRReparam(zeros(1, m), Matrix{Float64}(I, m, m)[:, 1:(m - 1)],
                          zeros(m), 1)
        odd = SLCEDataset(basis, [c for c in ds.configs[1:2]], ds.X_E[1:2, :],
                         ds.y_E[1:2], Matrix{Float64}(undef, 0, m), Float64[],
                         Int[], ds.provenance; disps = ds.disps[1:2], asr = fake)
        @test_throws ArgumentError vcat(ds[1:5], odd)
    end
end
