# Joint data layer (M3 slice 3) — the force design block X_F and the three-block
# co-fit: joint X_E/X_T evaluated at each config's (e, u), the compact force block
# (displacement-active columns / displacement-referenced atoms only), ragged
# force-row bookkeeping (`force_config`, same stored contract as `torque_config`),
# fit/refit with (torque_weight, force_weight), joint predict forms, and the
# model-level half of gate (j): predict_force ≡ −∂(predict_energy)/∂u.

using Test
using SLCE
using SLCE: _design_energy, _design_torque, _design_force, _disp_active_cols,
    _disp_referenced_atoms, accumulate_grad!
using LinearAlgebra
using Random

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_jd_unit(rng) = normalize(randn(rng, 3))
_jd_cfg(rng, nat) = reduce(hcat, [_jd_unit(rng) for _ = 1:nat])

@testset "joint data layer (X_F + three-block co-fit)" begin
    # D4h Fe–Fe bond cell (test_jointgrad fixture) with BOTH pure-spin and mixed
    # sectors, so the design has pure-spin columns (zero force derivative) alongside
    # displacement-active ones.
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    spec = BasisSpec(cr; lmax = 1, pmax = 1, sectors = [
        Sector(spin = (nbody = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1)])
    basis = SLCEBasis(cr, spec)
    nat = n_atoms(cr)
    m = n_salcs(basis)
    fcols = _disp_active_cols(basis)
    fref = _disp_referenced_atoms(basis)
    @test !isempty(fcols) && length(fcols) < m       # proper subset: both kinds exist
    @test all(fref)                                  # both Fe sites carry disp slots
    fp = crystal_fingerprint(cr)

    rng = MersenneTwister(0x5CE)
    jphi = randn(rng, m)
    model = SLCEModel(basis, 0.25, jphi)

    prov(tq) = DatumProvenance(; torque_qualified = tq, reference_id = "ref",
                               reference_fingerprint = fp, setup_id = "s1")
    # Synthetic ground-truth datum: channels generated from the model itself
    # (the recovery-plan-A construction — sample, fit back, compare).
    # uscale stays well inside the displacement-radius guard (the Fe–Fe bond is
    # 1.0 Å ⇒ warn threshold 0.5 Å)
    function mkdatum(; withu = true, withT = true, withF = true, uscale = 0.08)
        e = _jd_cfg(rng, nat)
        u = withu ? uscale * randn(rng, 3, nat) : zeros(3, nat)
        TrainingDatum(; energy = predict_energy(model, e, u), directions = e,
                      magmoms = ones(nat),
                      displacements = withu ? u : nothing,
                      forces = withF ? predict_force(model, e, u) : nothing,
                      torques = withT ? predict_torque(model, e, u) : nothing,
                      provenance = prov(withT))
    end

    data = vcat([mkdatum() for _ = 1:30],
                [mkdatum(withT = false) for _ = 1:10],           # force-only derivs
                [mkdatum(withu = false, withF = false) for _ = 1:10])  # spin-only
    ds = SLCEDataset(basis, data)

    @testset "joint designs and layout" begin
        @test size(ds.X_E) == (50, m)
        @test length(ds.disps) == 50
        @test ds.disps[41:50] == [zeros(3, nat) for _ = 1:10]   # u = 0 materialized
        # torque rows: qualified configs only (1:30 and 41:50), evaluated jointly
        tsel = vcat(1:30, 41:50)
        @test ds.torque_config == repeat(tsel; inner = 3 * nat)   # all atoms spin-referenced
        @test ds.X_T == _design_torque(basis, ds.configs[tsel], ds.disps[tsel],
                                       findall(SLCE._referenced_atoms(basis)))
        # force rows: force-carrying configs (1:40), all atoms disp-referenced here
        @test ds.force_config == repeat(collect(1:40); inner = 3 * nat)
        @test ds.force_cols == fcols
        @test size(ds.X_F) == (40 * 3 * nat, length(fcols))
        # y_F flattening matches the datum forces (config-major / atom-major / xyz)
        F1 = data[1].forces
        @test ds.y_F[1:3*nat] == vec(F1)
        # X_E at u = 0: displacement-active columns vanish exactly; pure-spin
        # columns are bit-identical to the spin-only design path
        cfg0 = [data[i].directions for i = 41:50]
        Xe0 = ds.X_E[41:50, :]
        @test all(iszero, Xe0[:, fcols])
        pcols = setdiff(1:m, fcols)
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
        @test n_salcs(bspin) == length(pcols)
        @test Xe0[:, pcols] == _design_energy(bspin, cfg0)
        # X_F columns against the raw joint gradient kernel (force sign −∂Φ/∂u)
        salcs_ = SLCE.salcs(basis)
        for (jj, j) in ((1, fcols[1]), (length(fcols), fcols[end]))
            for ci in (1, 40)
                Ge = zeros(3, nat)
                Gu = zeros(3, nat)
                accumulate_grad!(Ge, Gu, salcs_[j], data[ci].directions,
                                 data[ci].displacements, 1.0)
                rows = (3 * nat) * (ci - 1) .+ (1:3*nat)
                @test ds.X_F[rows, jj] == vec(-Gu)
            end
        end
    end

    # Fits in this file run `asr = false`: the synthetic ground truth is a
    # deliberately unconstrained random β (this file tests the data layer and
    # design matrices); the ASR-constrained fits and their gates live in
    # test/unit/test_asr.jl.
    @testset "three-block co-fit recovery (plan A, OLS)" begin
        for (wT, wF) in ((0.3, 0.0), (0.2, 0.5), (0.0, 0.6), (0.35, 0.65))
            f = fit(SLCEFit, ds, OLS(); torque_weight = wT, force_weight = wF,
                    asr = false)
            @test maximum(abs, f.jphi .- jphi) < 1e-8
            @test abs(f.j0 - 0.25) < 1e-10
            @test f.torque_weight == wT && f.force_weight == wF
        end
        # energy-only recovery needs an overdetermined energy block (n > m)
        dense = [mkdatum(withT = false, withF = false, uscale = 0.1)
                 for _ = 1:(m + 30)]
        dsE = SLCEDataset(basis, [d for d in dense]; use_torque = false,
                          use_force = false)
        fE = fit(SLCEFit, dsE, OLS(); asr = false)
        @test maximum(abs, fE.jphi .- jphi) < 1e-6
        # refit reproduces the co-fit design (assembles with both weights)
        f = fit(SLCEFit, ds, OLS(); torque_weight = 0.2, force_weight = 0.5,
                asr = false)
        fr = refit(f)
        @test maximum(abs, fr.jphi .- jphi) < 1e-8
        # exact-fit diagnostics
        @test r2_force(f) ≈ 1.0 atol = 1e-12
        @test rmse_force(f) < 1e-10
        @test residuals_force(f) == f.dataset.y_F .- f.dataset.X_F * f.jphi[fcols]
        @test rss_force(f) == sum(abs2, residuals_force(f))
        @test isfinite(gcv(f)) && effective_dof(f) > 0
    end

    # `cross_validate` has no `force_weight` yet and documents that it scores a
    # force-carrying dataset on its energy(+torque) blocks only. Pin that: the same
    # data with the force block dropped must produce the identical result. Whoever
    # adds the force channel will fail here and has to update the note with it,
    # rather than silently changing what every recorded CV score means.
    @testset "cross_validate ignores the force block (documented, pinned)" begin
        dsF = SLCEDataset(basis, [d for d in data])
        dsNoF = SLCEDataset(basis, [d for d in data]; use_force = false)
        @test has_force(dsF) && !has_force(dsNoF)
        cvF = cross_validate(dsF, OLS(); torque_weight = 0.3, nfolds = 3, asr = false)
        cvN = cross_validate(dsNoF, OLS(); torque_weight = 0.3, nfolds = 3, asr = false)
        @test cvF.pooled_rmse_energy == cvN.pooled_rmse_energy
        @test cvF.pooled_rmse_torque == cvN.pooled_rmse_torque
    end

    @testset "gate (j), model level: predict_force ≡ −FD(predict_energy)" begin
        e = _jd_cfg(rng, nat)
        u = 0.25 * randn(rng, 3, nat)
        F = predict_force(model, e, u)
        h = 1e-6
        for a = 1:nat, μ = 1:3
            up = copy(u); up[μ, a] += h
            um = copy(u); um[μ, a] -= h
            fd = -(predict_energy(model, e, up) - predict_energy(model, e, um)) / (2h)
            @test F[μ, a] ≈ fd atol = 1e-7
        end
        # joint torque: tangent finite difference of the joint energy surface
        T = predict_torque(model, e, u)
        for a = 1:nat
            v = randn(rng, 3)
            t = normalize(v .- dot(v, e[:, a]) .* e[:, a])
            ep = copy(e); ep[:, a] = normalize(e[:, a] .+ h .* t)
            em = copy(e); em[:, a] = normalize(e[:, a] .- h .* t)
            ge_t = (predict_energy(model, ep, u) - predict_energy(model, em, u)) / (2h)
            # τ = ∇E × e ⇒ the tangent component of ∇E is recovered as (e × τ)·t
            ea = e[:, a]
            @test dot(cross(ea, T[:, a]), t) ≈ ge_t atol = 1e-6
        end
        # a pure-spin model accepts the joint forms: u is inert, force ≡ 0
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
        ms = SLCEModel(bspin, 0.0, randn(rng, n_salcs(bspin)))
        @test predict_energy(ms, e, u) == predict_energy(ms, e)
        @test predict_force(ms, e, u) == zeros(3, nat)
        @test predict_torque(ms, e, u) == predict_torque(ms, e)
    end

    @testset "ragged force bookkeeping: slicing and vcat" begin
        idx = [2, 31, 35, 45, 41]          # mixed torque/force presence, unsorted
        sub = ds[idx]
        @test length(sub) == 5
        @test sub.disps == ds.disps[idx]
        @test sub.force_cols == ds.force_cols
        # per-row bit-identity against the stored parent rows
        krows = Int[]
        kfc = Int[]
        for (k, i) in enumerate(idx)
            r = searchsorted(ds.force_config, i)
            append!(krows, r)
            append!(kfc, fill(k, length(r)))
        end
        @test sub.X_F == ds.X_F[krows, :]
        @test sub.y_F == ds.y_F[krows]
        @test sub.force_config == kfc
        # a slice with no force-bearing configs drops the rows but stays joint
        noF = ds[[41, 43, 46]]
        @test isempty(noF.y_F) && !isempty(noF.disps)
        @test_throws ArgumentError fit(SLCEFit, noF, OLS(); force_weight = 0.5,
                                       asr = false)
        # partition + vcat rebuilds the dataset bit-identically (both orders)
        a, b = ds[1:35], ds[36:50]
        dd = vcat(a, b)
        @test dd.X_F == ds.X_F && dd.y_F == ds.y_F
        @test dd.force_config == ds.force_config
        @test dd.torque_config == ds.torque_config
        @test dd.disps == ds.disps
        rev = vcat(b, a)
        @test length(rev.y_F) == length(ds.y_F)
        @test rev.force_config[1:length(b.force_config)] == b.force_config
        # concatenating a force-free joint part keeps the force channel intact
        dd2 = vcat(a, noF)
        @test dd2.X_F == a.X_F && dd2.force_cols == a.force_cols
        # a displacement-free (pure-spin-built) part must not concatenate into a
        # displacement-bearing dataset
        bare = SLCEDataset(basis, [c for c in ds.configs[41:50]],
                          ds.X_E[41:50, :], ds.y_E[41:50],
                          Matrix{Float64}(undef, 0, m), Float64[], Int[],
                          ds.provenance)
        @test_throws ArgumentError vcat(ds, bare)
        # force-bearing parts must agree on force_cols (compact X_F blocks are
        # column-compatible only under one column set)
        k = length(fcols)
        odd = SLCEDataset(basis, [c for c in ds.configs[1:1]], ds.X_E[1:1, :],
                         ds.y_E[1:1], Matrix{Float64}(undef, 0, m), Float64[],
                         Int[], ds.provenance; disps = ds.disps[1:1],
                         X_F = zeros(3 * nat, k - 1), y_F = zeros(3 * nat),
                         force_config = fill(1, 3 * nat),
                         force_cols = fcols[1:k-1])
        @test_throws ArgumentError vcat(ds, odd)
    end

    @testset "constructor invariants" begin
        cfgs = ds.configs[1:2]
        Xe = ds.X_E[1:2, :]
        ye = ds.y_E[1:2]
        eT = Matrix{Float64}(undef, 0, m)
        mk(; kw...) = SLCEDataset(basis, cfgs, Xe, ye, eT, Float64[], Int[],
                                 ds.provenance; disps = ds.disps[1:2], kw...)
        k = length(fcols)
        Xf = zeros(2 * 3 * nat, k)
        yf = zeros(2 * 3 * nat)
        fc = repeat([1, 2]; inner = 3 * nat)
        @test mk(; X_F = Xf, y_F = yf, force_config = fc, force_cols = fcols) isa SLCEDataset
        @test_throws ArgumentError mk(; X_F = Xf, y_F = yf,
                                      force_config = reverse(fc), force_cols = fcols)
        @test_throws DimensionMismatch mk(; X_F = Xf[:, 1:k-1], y_F = yf,
                                          force_config = fc, force_cols = fcols)
        @test_throws ArgumentError mk(; X_F = Xf, y_F = yf, force_config = fc,
                                      force_cols = reverse(fcols))
        @test_throws ArgumentError mk(; X_F = Xf, y_F = yf, force_config = fc,
                                      force_cols = vcat(fcols[1:k-1], [m + 1]))
        @test_throws DimensionMismatch mk(; X_F = Xf, y_F = yf[1:3],
                                          force_config = fc, force_cols = fcols)
        # force rows without stored displacement fields are meaningless
        @test_throws ArgumentError SLCEDataset(basis, cfgs, Xe, ye, eT, Float64[],
                                               Int[], ds.provenance; X_F = Xf,
                                               y_F = yf, force_config = fc,
                                               force_cols = fcols)
        # disps length mismatch
        @test_throws DimensionMismatch SLCEDataset(basis, cfgs, Xe, ye, eT,
                                                   Float64[], Int[], ds.provenance;
                                                   disps = ds.disps[1:1])
    end

    @testset "fit / dataset error surface" begin
        @test_throws ArgumentError fit(SLCEFit, ds, OLS(); force_weight = 1.5)
        @test_throws ArgumentError fit(SLCEFit, ds, OLS(); force_weight = -0.1)
        @test_throws ArgumentError fit(SLCEFit, ds, OLS(); torque_weight = 0.6,
                                       force_weight = 0.5)
        dsE = SLCEDataset(basis, [mkdatum(withT = false, withF = false) for _ = 1:8];
                          use_torque = false, use_force = false)
        @test_throws ArgumentError fit(SLCEFit, dsE, OLS(); force_weight = 0.5)
        @test !has_force(dsE) && !has_torque(dsE)
        @test_throws ArgumentError residuals_force(fit(SLCEFit, dsE, OLS()))
        # use_force = true demands force data (mirror of the torque behavior)
        @test_throws ArgumentError SLCEDataset(basis,
            [mkdatum(withT = true, withF = false) for _ = 1:6])
        # 2-argument predict forms refuse a joint model
        e = _jd_cfg(rng, nat)
        @test_throws ArgumentError predict_energy(model, e)
        @test_throws ArgumentError predict_torque(model, e)
        # joint-form validation
        @test_throws DimensionMismatch predict_force(model, e, zeros(3, nat + 1))
        @test_throws ArgumentError predict_force(model, e, fill(NaN, 3, nat))
        # force co-fit support selection is explicitly not implemented yet
        f = fit(SLCEFit, ds, OLS(); torque_weight = 0.2, force_weight = 0.5,
                asr = false)
        @test_throws ArgumentError select_support(f)
    end

    # Octahedral Fe(O)₆: Fe is displacement-clamped (pmax = 0), O spin-inactive —
    # force rows exist only for the O shell; nonzero forces on Fe (which the model
    # is structurally blind to) are excluded with a warning.
    @testset "displacement-unreferenced atoms are excluded from force rows" begin
        # Fe(O)₆ fixture (test_jointgrad geometry) under the default NoSymmetry
        # backend — the structural-zero argument needs no symmetry reduction.
        L = 10.0
        cc = [0.5, 0.5, 0.5]
        shell = [[1.0, 0, 0], [-1.0, 0, 0], [0, 1.0, 0], [0, -1.0, 0],
                 [0, 0, 1.0], [0, 0, -1.0]]
        frac = reduce(hcat, vcat([cc], [cc .+ 2.0 .* v ./ L for v in shell]))
        crO = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 2, 2, 2, 2, 2, 2],
                      ["Fe", "O"])
        specO = @test_logs (:warn, r"odd") BasisSpec(crO;
            lmax = ["Fe" => 2, "O" => 0], pmax = ["Fe" => 0, "O" => 1],
            sectors = [Sector(spin = [2], disp = (degree = 1,), nbody = 2,
                              cutoff = 2.1)])
        bO = SLCEBasis(crO, specO)
        natO = n_atoms(crO)
        refO = _disp_referenced_atoms(bO)
        @test findall(refO) == collect(2:7)            # O shell only
        mO = SLCEModel(bO, 0.0, randn(rng, n_salcs(bO)))
        fpO = crystal_fingerprint(crO)
        provO = DatumProvenance(; reference_id = "refO", reference_fingerprint = fpO)
        function mkdatumO(fe_force)
            e = _jd_cfg(rng, natO)
            u = 0.1 * randn(rng, 3, natO)
            F = predict_force(mO, e, u)
            @test all(iszero, F[:, 1])                  # structurally zero on Fe
            F[1, 1] += fe_force
            TrainingDatum(; energy = predict_energy(mO, e, u), directions = e,
                          magmoms = vcat([1.0], zeros(6)), displacements = u,
                          forces = F, provenance = provO)
        end
        dataO = [mkdatumO(0.0) for _ = 1:6]
        dsO = SLCEDataset(bO, dataO; use_torque = false)
        @test length(dsO.y_F) == 6 * 3 * 6              # O rows only
        @test dsO.force_config == repeat(collect(1:6); inner = 18)
        @test dsO.y_F[1:18] == vec(dataO[1].forces[:, 2:7])
        fO = fit(SLCEFit, dsO, OLS(); force_weight = 0.5, asr = false)
        @test maximum(abs, fO.jphi .- mO.jphi) < 1e-8
        # nonzero Fe force triggers the exclusion warning
        dataW = [mkdatumO(i == 1 ? 0.5 : 0.0) for i = 1:6]
        @test_logs (:warn, r"structurally zero") match_mode = :any SLCEDataset(
            bO, dataW; use_torque = false)
        # torque rows are restricted to spin-referenced atoms the same way: only
        # Fe (atom 1) has a SPIN slot, so each qualified config contributes 3 rows
        dataT = [TrainingDatum(; energy = d.energy, directions = d.directions,
                               magmoms = d.magmoms, displacements = d.displacements,
                               forces = d.forces,
                               torques = predict_torque(mO, d.directions,
                                                        d.displacements),
                               provenance = DatumProvenance(;
                                   torque_qualified = true, reference_id = "refO",
                                   reference_fingerprint = fpO))
                 for d in dataO]
        dsT = SLCEDataset(bO, dataT)
        @test length(dsT.y_T) == 6 * 3                  # Fe rows only
        @test dsT.torque_config == repeat(collect(1:6); inner = 3)
        @test dsT.y_T[1:3] == dataT[1].torques[:, 1]
        fT = fit(SLCEFit, dsT, OLS(); torque_weight = 0.3, force_weight = 0.3,
                 asr = false)
        @test maximum(abs, fT.jphi .- mO.jphi) < 1e-8
        # nonzero torque targets on a spin-unreferenced atom warn (excluded rows)
        dW = dataT[1]
        tq = copy(dW.torques)
        tq[1, 3] = 0.1
        dataTW = vcat([TrainingDatum(; energy = dW.energy,
                                     directions = dW.directions,
                                     magmoms = dW.magmoms,
                                     displacements = dW.displacements,
                                     forces = dW.forces, torques = tq,
                                     provenance = dW.provenance)], dataT[2:6])
        @test_logs (:warn, r"no SALC spin slot") match_mode = :any SLCEDataset(
            bO, dataTW)
    end

    @testset "degenerate-block and radius warnings" begin
        # all-zero force design: every disp factor of the D4h mixed sector has
        # degree ≥ 1 on BOTH bond sites, so Gu ≡ 0 at u = 0 — the force block is
        # identically zero even though it has rows
        z0 = [TrainingDatum(; energy = predict_energy(model, c, zeros(3, nat)),
                            directions = c, magmoms = ones(nat),
                            forces = zeros(3, nat), provenance = prov(false))
              for c in [_jd_cfg(rng, nat) for _ = 1:5]]
        @test_logs (:warn, r"identically zero") (:warn, r"exactly zero") match_mode = :any SLCEDataset(
            basis, z0; use_torque = false)
        # displacement amplitude beyond half the shortest reference distance (the
        # Fe–Fe bond is 1.0 Å): the un-minimum-imaged-adapter symptom warns
        ubig = zeros(3, nat)
        ubig[1, 1] = 3.0                                # a whole lattice vector
        e = _jd_cfg(rng, nat)
        big = TrainingDatum(; energy = predict_energy(model, e, ubig),
                            directions = e, magmoms = ones(nat),
                            displacements = ubig,
                            forces = predict_force(model, e, ubig),
                            provenance = prov(false))
        @test_logs (:warn, r"half the shortest") match_mode = :any SLCEDataset(
            basis, [big]; use_torque = false)
        # in-regime data do not warn (the main fixture dataset was built above
        # without a radius warning — pin the threshold direction explicitly)
        @test SLCE._min_reference_distance(cr) ≈ 1.0 atol = 1e-12
    end
end
