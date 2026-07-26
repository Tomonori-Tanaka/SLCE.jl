# Staged (hierarchical) fitting — `fit(...; frozen, sector_mask)`: the sector
# selectors (including the anti-drift gate that `:soc_free` is the SAME predicate
# the `Sector(soc = false)` truncation uses), the stage reparameterization
# (frozen columns pinned by zero rows of Z, free columns carrying the ASR null
# space restricted to them), the affine constraint that appears when the frozen
# part violates the ASR — with its infeasibility refusal — and the consumers that
# must follow the STAGE rather than the dataset (`dof`, `gcv`, `identifiability`,
# `refit`).
#
# The physical claim being gated: a chain of stages, each fitted under its own
# ASR, produces a model that is translation-invariant AS A WHOLE (the next stage's
# constraint stays homogeneous — the staged-fit theorem, design record §6
# amendment 8). Staged ≠ joint in general (the earlier stage absorbs what the
# later columns would have explained); it IS exact when the frozen values are.

using Test
using SLCE
using SLCE: build_asr, sector_columns, _stage_reparam, _frozen_coefficients,
    salcs, has_spin, has_disp, is_soc_free, _assemble_problem
using LinearAlgebra
using Random
using Statistics

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_st_unit(rng) = normalize(randn(rng, 3))
_st_cfg(rng, nat) = reduce(hcat, [_st_unit(rng) for _ = 1:nat])

@testset "staged (hierarchical) fitting" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    # pure spin + coupled + a LATTICE-only sector, so all three channel selectors
    # pick out something (and the affine path has spin-free columns to live on)
    spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (nbody = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1),
        Sector(disp = (degree = 2,), nbody = 1:2, cutoff = 1.1)])
    basis = SLCEBasis(cr, spec)
    m = n_salcs(basis)
    nat = n_atoms(cr)
    rep = build_asr(basis)
    ks = basis.salc_basis.keys

    @testset "sector selectors" begin
        cs = Dict(s => sector_columns(basis, s)
                  for s in (:all, :spin, :lattice, :coupled, :soc_free, :soc))
        @test cs[:all] == collect(1:m)
        # channel partition
        @test sort(vcat(cs[:spin], cs[:lattice], cs[:coupled])) == cs[:all]
        @test all(!isempty, (cs[:spin], cs[:lattice], cs[:coupled]))
        # the L_S partition, crosscutting the channel one
        @test sort(vcat(cs[:soc_free], cs[:soc])) == cs[:all]
        @test cs[:soc_free] == findall(k -> k.L_S == 0, ks)
        @test all(j -> !any(has_disp, ks[j].decors), cs[:spin])
        @test all(j -> !any(has_spin, ks[j].decors), cs[:lattice])
        @test all(j -> any(has_spin, ks[j].decors) && any(has_disp, ks[j].decors),
                  cs[:coupled])
        # ANTI-DRIFT (design record §13 risk 4): the staging selector `:soc_free`
        # and the basis-level `Sector(soc = false)` truncation must name the SAME
        # content — the keys the mask keeps are exactly the keys a SOC-less
        # rebuild produces.
        specf = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (nbody = 1:2,), soc = false, cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, soc = false,
                   cutoff = 1.1),
            Sector(disp = (degree = 2,), nbody = 1:2, soc = false, cutoff = 1.1)])
        bfree = SLCEBasis(cr, specf)
        @test bfree.salc_basis.keys == ks[cs[:soc_free]]
        @test all(is_soc_free, bfree.salc_basis.keys)
        # unions, explicit columns, masks, and the error surface
        @test sector_columns(basis, [:spin, :lattice]) ==
              sort(vcat(cs[:spin], cs[:lattice]))
        @test sector_columns(basis, [3, 1, 3]) == [1, 3]
        msk = falses(m); msk[[2, 5]] .= true
        @test sector_columns(basis, msk) == [2, 5]
        @test_throws ArgumentError sector_columns(basis, :exchange)
        @test_throws ArgumentError sector_columns(basis, [0, 2])
        @test_throws ArgumentError sector_columns(basis, [1, m + 1])
        @test_throws DimensionMismatch sector_columns(basis, falses(m - 1))
    end

    # --- synthetic joint data from a feasible ground truth --------------------
    rng = MersenneTwister(0x57)
    beta_true = rep.Z * (rep.Z' * randn(rng, m))
    model = SLCEModel(basis, 0.3, beta_true)
    fpc = crystal_fingerprint(cr)
    prov = DatumProvenance(; torque_qualified = true, reference_id = "r",
                           reference_fingerprint = fpc, setup_id = "s")
    function mkdatum()
        e = _st_cfg(rng, nat)
        u = 0.08 * randn(rng, 3, nat)
        u = u .- mean(u; dims = 2)
        TrainingDatum(; energy = predict_energy(model, e, u), directions = e,
                      magmoms = ones(nat), displacements = u,
                      forces = predict_force(model, e, u),
                      torques = predict_torque(model, e, u), provenance = prov)
    end
    ds = SLCEDataset(basis, [mkdatum() for _ = 1:90])
    spincols = sector_columns(basis, :spin)
    restcols = setdiff(1:m, spincols)
    wts = (torque_weight = 0.3, force_weight = 0.3)

    @testset "no staging request is the untouched path (bitwise)" begin
        f0 = fit(SLCEFit, ds, OLS(); wts...)
        fa = fit(SLCEFit, ds, OLS(); wts..., sector_mask = :all)
        @test f0.jphi == fa.jphi && f0.j0 == fa.j0      # bitwise
        # `sector_mask = :all` with no frozen model is not a stage at all: it takes
        # the dataset's own reparameterization, unmodified
        @test f0.reparam === ds.asr === fa.reparam
        @test dof(f0) == dof(fa) == size(ds.asr.Z, 2) + 1
    end

    @testset "a stage fits only its mask" begin
        f1 = fit(SLCEFit, ds, OLS(); wts..., sector_mask = :spin)
        @test all(iszero, f1.jphi[restcols])            # untouched columns stay 0
        @test dof(f1) == length(spincols) + 1           # no constraint binds them
        @test f1.asr && f1.asr_residual < 1e-13
        @test identifiability(f1).ncols == length(spincols)
        # ... and it is NOT the joint fit: the spin columns absorb what the
        # coupled ones would have explained (staged ≠ joint, by construction)
        fj = fit(SLCEFit, ds, OLS(); wts...)
        @test maximum(abs, f1.jphi[spincols] .- fj.jphi[spincols]) > 1e-6
        @test rmse_energy(f1) > rmse_energy(fj)
    end

    @testset "freezing the true values makes the next stage exact" begin
        truth_spin = zeros(m)
        truth_spin[spincols] .= beta_true[spincols]
        f2 = fit(SLCEFit, ds, OLS(); wts..., frozen = SLCEModel(basis, 0.0, truth_spin),
                 sector_mask = [:coupled, :lattice])
        @test maximum(abs, f2.jphi .- beta_true) < 1e-8
        @test abs(f2.j0 - 0.3) < 1e-8                   # j0 is never frozen
        @test f2.jphi[spincols] == truth_spin[spincols] # frozen part untouched
        @test f2.asr_residual < 1e-13
        @test asr_residual(SLCEModel(f2)) < 1e-13
        # the staged-fit theorem: the frozen part satisfies the ASR on its own, so
        # the stage's constraint is HOMOGENEOUS — beta_p carries the frozen values
        # and nothing else (no particular-solution component on the free columns)
        @test all(iszero, f2.reparam.beta_p[restcols])
        @test f2.reparam.beta_p[spincols] == truth_spin[spincols]
        @test dof(f2) == size(f2.reparam.Z, 2) + 1 < length(restcols) + 1
        # translation invariance of the WHOLE staged model
        mf = SLCEModel(f2)
        e0 = _st_cfg(rng, nat)
        u0 = 0.07 * randn(rng, 3, nat)
        Escale = max(abs(predict_energy(mf, e0, u0)), 1.0)
        @test abs(predict_energy(mf, e0, u0 .+ [0.1, -0.05, 0.03]) -
                  predict_energy(mf, e0, u0)) / Escale < 1e-12
        @test maximum(abs, sum(predict_force(mf, e0, u0); dims = 2)) < 1e-12
    end

    @testset "a chain of stages stays homogeneous and invariant" begin
        s1 = fit(SLCEFit, ds, OLS(); wts..., sector_mask = :spin)
        s2 = fit(SLCEFit, ds, OLS(); wts..., frozen = SLCEModel(s1),
                 sector_mask = :coupled)
        s3 = fit(SLCEFit, ds, OLS(); wts..., frozen = SLCEModel(s2),
                 sector_mask = :lattice)
        for (s, cols) in ((s2, sector_columns(basis, :coupled)),
                          (s3, sector_columns(basis, :lattice)))
            # every stage's constraint is homogeneous: no particular solution on
            # the columns it fits
            @test all(iszero, s.reparam.beta_p[cols])
            @test s.asr_residual < 1e-13
        end
        @test s2.jphi[spincols] == s1.jphi[spincols]    # stage 1 preserved
        @test s3.jphi[sector_columns(basis, :coupled)] ==
              s2.jphi[sector_columns(basis, :coupled)]
        @test asr_residual(SLCEModel(s3)) < 1e-13
        # Each stage can only improve the objective it is handed — the WEIGHTED
        # one it actually minimizes (the energy block alone may worsen while the
        # derivative blocks improve; `L = 0.4·MSE_E + 0.3·MSE_T + 0.3·MSE_F`).
        obj(f) = 0.4 * rmse_energy(f)^2 + 0.3 * rmse_torque(f)^2 +
                 0.3 * rmse_force(f)^2
        @test obj(s2) <= obj(s1) + 1e-12
        @test obj(s3) <= obj(s2) + 1e-12
    end

    @testset "which masks straddle a constraint row" begin
        # A's rows are graded by (spin content, total displacement degree), so a
        # CHANNEL mask never splits one: 0 of 180 rows on this fixture mix
        # channels. The L_S masks DO split rows (54 of 180 here) — which is why the
        # staging axis needs the affine machinery at all (design record §6
        # amendment 8), and why `:soc_free` staging is not the same operation as a
        # `soc = false` rebuild.
        chan(j) = (any(has_spin, ks[j].decors), any(has_disp, ks[j].decors))
        rowcols = [findall(!=(0.0), @view rep.A[r, :]) for r in axes(rep.A, 1)]
        @test all(cs -> length(unique(chan.(cs))) == 1, rowcols)
        @test (count(cs -> length(unique(is_soc_free.(ks[cs]))) > 1, rowcols),
               size(rep.A, 1)) == (54, 180)              # pinned fixture numbers
    end

    @testset "affine path: freezing an ASR-violating model" begin
        # Freeze ONE coefficient at a hand-chosen value that does not satisfy the
        # ASR, and let the whole rest of the model compensate: the stage's
        # constraint is then genuinely affine, and the fitted TOTAL model comes out
        # exactly translation-invariant anyway.
        j0 = findfirst(j -> any(!=(0.0), @view rep.A[:, j]), 1:m)
        bad = zeros(m)
        bad[j0] = 0.4
        mbad = SLCEModel(basis, 0.0, bad)
        @test asr_residual(mbad) > 1e-6                  # a genuine violation
        freec = setdiff(1:m, j0)
        fa = fit(SLCEFit, ds, OLS(); wts..., frozen = mbad, sector_mask = freec)
        # the particular solution lives on the free columns: beta_p is no longer
        # just the frozen values
        @test any(!iszero, fa.reparam.beta_p[freec])
        @test fa.jphi[j0] == 0.4                         # frozen value untouched
        @test fa.asr_residual < 1e-12
        @test asr_residual(SLCEModel(fa)) < 1e-12
        e0 = _st_cfg(rng, nat)
        u0 = 0.06 * randn(rng, 3, nat)
        @test maximum(abs, sum(predict_force(SLCEModel(fa), e0, u0); dims = 2)) < 1e-12
        # a penalized estimator then shrinks toward the particular solution: warn
        @test_logs (:warn, r"affine") match_mode = :any fit(
            SLCEFit, ds, Ridge(1e-8); wts..., frozen = mbad, sector_mask = freec)
        # infeasible: the spin columns carry no constraint row at all, so they
        # cannot balance a displacement-sector violation
        @test_throws ArgumentError fit(SLCEFit, ds, OLS(); wts..., frozen = mbad,
                                       sector_mask = :spin)

        # REFIT on an affine stage must re-derive the particular solution, not
        # inherit or drop it: dropping it (keeping only the frozen values) leaves
        # A·β = A_frozen·β_frozen ≠ 0 — a model reported as constrained that
        # actually breaks translation invariance and Σ_a f_a = 0.
        Xa, = _assemble_problem(ds, wts.torque_weight, wts.force_weight)
        scaled = [abs(fa.jphi[j]) * norm(@view Xa[:, j]) for j = 1:m]
        for thr in (0.0, 0.5 * maximum(scaled[freec]))
            fr = refit(fa; threshold = thr)
            @test fr.jphi[j0] == 0.4                     # frozen value survives
            @test fr.asr_residual < 1e-12                # ... and so does the ASR
            @test asr_residual(SLCEModel(fr)) < 1e-12
            @test maximum(abs, sum(predict_force(SLCEModel(fr), e0, u0);
                                   dims = 2)) < 1e-12
        end
        # the degenerate branch keeps the particular solution too (γ = 0 is
        # feasible by construction — zeroing it is what broke the ASR)
        fr0 = refit(fa; threshold = 1e9)
        @test fr0.asr_residual < 1e-12
        @test maximum(abs, sum(predict_force(SLCEModel(fr0), e0, u0); dims = 2)) < 1e-12
        # a penalized estimator's gcv/effective_dof must use the weight map at
        # β − beta_p (what the solver iterated on), not at the raw jphi
        for est in (AdaptiveRidge(; lambda = 0.01),
                    GroupAdaptiveRidge(collect(1:m), ones(m); lambda = 0.01))
            fp = fit(SLCEFit, ds, est; wts..., frozen = mbad, sector_mask = freec)
            lam, wv = SLCE._penalty_diagonal(fp.estimator,
                                             fp.jphi .- fp.reparam.beta_p)
            @test SLCE._penalty_beta(fp, fp.reparam) == fp.jphi .- fp.reparam.beta_p
            Xg, yg, _, _, _ = _assemble_problem(ds, wts.torque_weight,
                                                wts.force_weight, fp.reparam)
            @test effective_dof(fp) ≈
                  SLCE._edof_ns(Xg, lam, wv, fp.reparam.Z) + 1 rtol = 1e-8
            @test isfinite(gcv(fp))
        end
    end

    @testset "frozen coefficients are matched by SALCKey" begin
        v = collect(1.0:m)
        @test _frozen_coefficients(basis, SLCEModel(basis, 0.0, v)) == v
        # a model whose keys come from a DIFFERENT basis: absent keys carrying a
        # nonzero coefficient are refused rather than silently dropped
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
        mspin = SLCEModel(bspin, 0.0, ones(n_salcs(bspin)))
        got = _frozen_coefficients(basis, mspin)          # keys are shared here
        @test count(!iszero, got) == n_salcs(bspin)
        @test_throws ArgumentError _frozen_coefficients(bspin, SLCEModel(basis, 0.0,
                                                                        ones(m)))
        # a model from a DIFFERENT crystal is refused outright: SALCKeys carry a
        # per-build `orbit_id`, so key equality across crystals is meaningless
        cr2 = Crystal(Lattice(Matrix(3.2 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0],
                      [1, 1], ["Fe"])
        b2 = SLCEBasis(cr2, BasisSpec(cr2; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.2)]))
        @test_throws ArgumentError _frozen_coefficients(
            basis, SLCEModel(b2, 0.0, ones(n_salcs(b2))))
        # a mask leaving the WHOLE frozen model free freezes nothing — say so
        @test_logs (:warn, r"nothing is actually frozen") match_mode = :any fit(
            SLCEFit, ds, OLS(); wts..., frozen = SLCEModel(basis, 0.0, ones(m)),
            sector_mask = :all)
        # a frozen value on a FREE column is ignored (that column is re-fitted)
        seed = zeros(m)
        seed[spincols] .= beta_true[spincols]
        seed[restcols] .= 123.0
        fseed = fit(SLCEFit, ds, OLS(); wts..., frozen = SLCEModel(basis, 0.0, seed),
                    sector_mask = [:coupled, :lattice])
        @test maximum(abs, fseed.jphi .- beta_true) < 1e-8
    end

    @testset "consumers follow the stage" begin
        truth_spin = zeros(m)
        truth_spin[spincols] .= beta_true[spincols]
        fst = fit(SLCEFit, ds, OLS(); wts...,
                  frozen = SLCEModel(basis, 0.0, truth_spin),
                  sector_mask = [:coupled, :lattice])
        # identifiability / dof / gcv all report the STAGE's design, not the
        # dataset's (the plain fit has more free parameters)
        fj = fit(SLCEFit, ds, OLS(); wts...)
        @test identifiability(fst).ncols == size(fst.reparam.Z, 2)
        @test identifiability(fst).ncols < identifiability(fj).ncols
        @test identifiability(fst).nullity == 0
        @test dof(fst) < dof(fj)
        @test isfinite(gcv(fst)) && effective_dof(fst) <= dof(fst) + 1e-9
        # refit stays inside the stage: frozen columns keep their values and the
        # de-biased model is still exactly translation-invariant
        fr = refit(fst)
        @test fr.jphi[spincols] == truth_spin[spincols]
        @test maximum(abs, fr.jphi .- beta_true) < 1e-8
        @test fr.asr_residual < 1e-13
        @test maximum(abs, sum(predict_force(SLCEModel(fr), _st_cfg(rng, nat),
                                             0.05 * randn(rng, 3, nat)); dims = 2)) <
              1e-12
        # Thresholding away most of the stage keeps the frozen part intact. The
        # threshold must be on the SCALED-magnitude scale `refit` uses
        # (|jϕ_j|·‖X[:,j]‖) — a bare coefficient cut selects the frozen columns
        # instead, empties the support after the movable intersection, and silently
        # tests only the degenerate short-circuit (a real hole this gate had).
        Xa, = _assemble_problem(ds, wts.torque_weight, wts.force_weight)
        scaled = [abs(fst.jphi[j]) * norm(@view Xa[:, j]) for j = 1:m]
        thr = 0.5 * maximum(scaled[restcols])
        movable = [j for j in 1:m if norm(@view fst.reparam.Z[j, :]) >= 1e-12]
        sup = [j for j = 1:m if scaled[j] > thr]
        @test !isempty(intersect(sup, movable))          # the real path, not the
        @test length(intersect(sup, movable)) < length(movable)   # short-circuit
        fr2 = refit(fst; threshold = thr)
        @test fr2.jphi[spincols] == truth_spin[spincols]
        @test count(!iszero, fr2.jphi[restcols]) < count(!iszero, fst.jphi[restcols])
        @test fr2.asr_residual < 1e-12
        @test maximum(abs, sum(predict_force(SLCEModel(fr2), _st_cfg(rng, nat),
                                             0.05 * randn(rng, 3, nat)); dims = 2)) <
              1e-12
        # the selection layer refuses a staged fit rather than mis-costing it
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
        ms = SLCEModel(bspin, 0.0, randn(rng, n_salcs(bspin)))
        cfgs = [_st_cfg(rng, nat) for _ = 1:40]
        dss = SLCEDataset(bspin, cfgs, predict_energy(ms, cfgs))
        fsp = fit(SLCEFit, dss, OLS(); sector_mask = [1, 2, 3])
        @test dof(fsp) == 4
        @test !fsp.asr && fsp.asr_residual == 0.0       # a mask is not a constraint
        @test_throws ArgumentError select_support(fsp)
        # cross_validate threads the staging plan to every fold
        cv = cross_validate(dss, OLS(); nfolds = 3, sector_mask = [1, 2, 3])
        @test isfinite(cv.pooled_rmse_energy)
        @test cv.pooled_rmse_energy > cross_validate(dss, OLS();
                                                     nfolds = 3).pooled_rmse_energy
    end

    @testset "stage reparameterization unit behavior" begin
        free = [1, 3, 5]
        Zs = _stage_reparam(basis, free, zeros(m), nothing)
        @test size(Zs.Z) == (m, 3) && Zs.rank == 0
        @test Zs.Z' * Zs.Z == I(3)                       # orthonormal selection
        @test all(iszero, Zs.Z[setdiff(1:m, free), :])
        st = _stage_reparam(basis, sector_columns(basis, :lattice), zeros(m), rep.A)
        @test maximum(abs, rep.A * st.Z) < 1e-12         # feasible directions only
        @test all(iszero, st.Z[sector_columns(basis, :spin), :])
        @test_throws DimensionMismatch _stage_reparam(basis, free, zeros(m - 1),
                                                      nothing)
        @test_throws ArgumentError fit(SLCEFit, ds, OLS(); sector_mask = Int[])
        @test_throws TypeError fit(SLCEFit, ds, OLS(); sector_mask = nothing)
        # a single-column mask: an unconstrained (pure-spin) column is fitted,
        # while a lone displacement column cannot be translation-invariant by
        # itself — the constraint zeroes it and the stage has no free parameter
        f1c = fit(SLCEFit, ds, OLS(); wts..., sector_mask = [spincols[1]])
        @test dof(f1c) == 2 && count(!iszero, f1c.jphi) == 1
        f1d = fit(SLCEFit, ds, OLS(); wts..., sector_mask = [restcols[1]])
        @test dof(f1d) == 1 && all(iszero, f1d.jphi)
        fnoasr = fit(SLCEFit, ds, OLS(); wts..., asr = false, sector_mask = :spin,
                     frozen = SLCEModel(basis, 0.0, beta_true))
        @test !fnoasr.asr && fnoasr.asr_residual == 0.0
        @test fnoasr.jphi[restcols] == beta_true[restcols]   # frozen, unconstrained
        @test dof(fnoasr) == length(spincols) + 1
    end
end
