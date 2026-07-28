# Synthetic recovery plan B (derivative-only) and the identifiability accounting
# behind it: with CENTER-OF-MASS-FREE displacement sampling — the physically
# natural DFT protocol — forces and torques ALONE (zero energy weight) determine
# the model exactly *under the ASR*, and not at all without it. The rank ledger
# says why, per channel:
#   * torque measures ∂E/∂e, so it is blind to every SPIN-FREE direction plus the
#     translation-violating ones; on a basis whose displacement content is all
#     spin-decorated those coincide (deficiency = rank(A) exactly, cured by the
#     ASR), and a basis with a lattice-only sector keeps a residual deficiency
#     the constraint cannot cure (second fixture below);
#   * force measures −∂E/∂u, whose atom sum IS −D·E at the sample, so it sees the
#     violating content except what vanishes to second order on the slice;
#   * together, under the ASR, they determine everything (nullity 0) in both
#     fixtures — which is the plan-B result.
# Also gates the two production diagnostics this slice adds: `identifiability`
# (exact, O(n·q²)) and `fit`'s standing dead-column warning (per column, O(n·q),
# a RELATIVE cut — the `Σ_a u_a` columns sit at ~1e-19, not at 0).

using Test
using SLCE
using SLCE: build_asr, _assemble_problem, salcs, has_disp, has_spin
using LinearAlgebra
using Random
using Statistics

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_id_unit(rng) = normalize(randn(rng, 3))
_id_cfg(rng, nat) = reduce(hcat, [_id_unit(rng) for _ = 1:nat])

# Numerical rank at `identifiability`'s tolerance (`LinearAlgebra.rank`'s).
function _id_rank(X)
    s = svdvals(X)
    (isempty(s) || s[1] == 0.0) && return 0
    return count(>(minimum(size(X)) * eps() * s[1]), s)
end

# Dimension of the ASR-feasible directions supported ENTIRELY on the given column
# set: v = Z·g with v vanishing off `cols` ⇔ g ∈ null(Z[other, :]).
function _feasible_within(Z, cols)
    other = setdiff(axes(Z, 1), cols)
    isempty(other) && return size(Z, 2)
    return size(Z, 2) - _id_rank(Z[other, :])
end

@testset "identifiability and derivative-only recovery (plan B)" begin
    # The test_asr D4h fixture: pure-spin sector + spin[1,1] × displacement
    # degree 2, pmax = 2 (the on-site partners the difference invariants need).
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)])
    basis = SLCEBasis(cr, spec)
    nat = n_atoms(cr)
    m = n_salcs(basis)
    rep = build_asr(basis)
    q = size(rep.Z, 2)
    npure = count(s -> !any(has_disp, s.key.decors), salcs(basis))
    @test (m, rep.rank, q, npure) == (198, 135, 63, 9)     # pinned fixture shape

    rng = MersenneTwister(0xB0)
    beta_true = rep.Z * (rep.Z' * randn(rng, m))           # ASR-feasible truth
    model = SLCEModel(basis, 0.2, beta_true)
    fpc = crystal_fingerprint(cr)
    prov = DatumProvenance(; torque_qualified = true, reference_id = "r",
                           reference_fingerprint = fpc, setup_id = "s")

    # `comfree`: Σ_a u_a = 0, the drift-free displacement protocol. The contrast
    # sample lets the whole rank story be attributed to the SAMPLING, not to the
    # basis or the design code.
    function mkdatum(; comfree::Bool = true, scale = 0.08)
        e = _id_cfg(rng, nat)
        u = scale * randn(rng, 3, nat)
        comfree && (u = u .- mean(u; dims = 2))
        TrainingDatum(; energy = predict_energy(model, e, u), directions = e,
                      magmoms = ones(nat), displacements = u,
                      forces = predict_force(model, e, u),
                      torques = predict_torque(model, e, u), provenance = prov)
    end

    ds = SLCEDataset(basis, [mkdatum() for _ = 1:60])          # COM-free
    dsg = SLCEDataset(basis, [mkdatum(comfree = false) for _ = 1:60])
    @test ds.asr isa SLCE.ASRReparam && ds.asr.rank == rep.rank

    @testset "rank ledger under center-of-mass-free sampling" begin
        # (1) torque only: the channel sees ∂E/∂e on the COM-free slice. In THIS
        # basis every displacement-decorated SALC also carries spin factors, so
        # "blind to spin-free content" and "blind to the translation-violating
        # subspace" coincide and the deficiency is rank(A) exactly — the general
        # rule, and a basis where the two differ, are gated below.
        idT = identifiability(ds; torque_weight = 1.0, asr = false)
        @test idT.ncols == m
        @test idT.nullity == rep.rank                       # 135 flat directions
        idTz = identifiability(ds; torque_weight = 1.0)
        @test idTz.ncols == q && idTz.nullity == 0          # ASR restores it fully
        @test idTz.sigma_min > 1e3 * idTz.sigma_cut
        # the rank decisions are not knife-edge: kept/dropped singular values are
        # separated by many orders (or nothing was dropped at all)
        @test idT.gap > 1e6 && idTz.gap == Inf

        # (2) force only: Σ_a f_a is −D·E at the sample, so the channel DOES see
        # violating content — except what vanishes to second order on the slice.
        # The deficiency is therefore smaller than rank(A). Separately, no
        # spin-only SALC has a displacement derivative, so the pure-spin columns
        # are structurally zero and stay flat even under the constraint.
        idF = identifiability(ds; force_weight = 1.0, asr = false)
        @test 0 < idF.nullity < rep.rank
        idFz = identifiability(ds; force_weight = 1.0)
        @test idFz.nullity == npure                         # exactly the spin columns
        XF, = _assemble_problem(ds, 0.0, 1.0)               # bookkeeping cross-check
        @test count(j -> all(iszero, @view XF[:, j]), axes(XF, 2)) == npure

        # (3) the plan-B fit: torques + forces, zero energy weight. Still
        # rank-deficient unconstrained; exactly determined under the ASR.
        id0 = identifiability(ds; torque_weight = 0.4, force_weight = 0.6,
                              asr = false)
        idz = identifiability(ds; torque_weight = 0.4, force_weight = 0.6)
        @test id0.nullity > 0
        @test idz.ncols == q && idz.nullity == 0 && idz.gap == Inf
        # the constraint rows supply exactly the information the data lack:
        # stacking A onto the unconstrained design restores full column rank.
        # (Equivalent to `idz.nullity == 0` given that Z spans null(A) — kept as a
        # direct reading of the statement, not as independent corroboration.)
        X0, = _assemble_problem(ds, 0.4, 0.6)
        @test _id_rank(vcat(X0 ./ maximum(abs, X0), rep.A ./ maximum(abs, rep.A))) == m

        # the ledger is a property of the protocol, not of the sample size: twice
        # the data (well past the row-starved regime) reproduces it exactly
        ds2 = SLCEDataset(basis, [mkdatum() for _ = 1:120])
        @test identifiability(ds2; torque_weight = 1.0,
                              asr = false).nullity == idT.nullity
        @test identifiability(ds2; force_weight = 1.0,
                              asr = false).nullity == idF.nullity
        @test identifiability(ds2; torque_weight = 0.4, force_weight = 0.6,
                              asr = false).nullity == id0.nullity
        @test identifiability(ds2; torque_weight = 0.4,
                              force_weight = 0.6).nullity == 0

        # (4) the deficiency is a property of the SAMPLING: generic (drifting)
        # displacements determine every direction with no constraint at all.
        @test identifiability(dsg; torque_weight = 0.4, force_weight = 0.6,
                              asr = false).nullity == 0
    end

    @testset "a lattice-only sector breaks the torque-channel coincidence" begin
        # Same cell plus a SPIN-FREE (force-constant) sector. Torque is blind to
        # all spin-independent content, so its deficiency is rank(A) PLUS the
        # spin-free feasible directions — and the ASR cannot cure that part.
        # Forces can, which is why the plan-B pair is still fully determined.
        specL = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)])
        bL = SLCEBasis(cr, specL)
        mL = n_salcs(bL)
        repL = build_asr(bL)
        qL = size(repL.Z, 2)
        spin_free = findall(s -> !any(has_spin, s.key.decors), salcs(bL))
        dfree = _feasible_within(repL.Z, spin_free)
        @test (mL, repL.rank, qL, length(spin_free), dfree) == (219, 150, 69, 21, 6)

        betaL = repL.Z * (repL.Z' * randn(rng, mL))
        modelL = SLCEModel(bL, 0.1, betaL)
        dataL = map(1:80) do _
            e = _id_cfg(rng, nat)
            u = 0.08 * randn(rng, 3, nat)
            u = u .- mean(u; dims = 2)
            TrainingDatum(; energy = predict_energy(modelL, e, u), directions = e,
                          magmoms = ones(nat), displacements = u,
                          forces = predict_force(modelL, e, u),
                          torques = predict_torque(modelL, e, u), provenance = prov)
        end
        dsL = SLCEDataset(bL, dataL)

        # the general rule, both halves
        @test identifiability(dsL; torque_weight = 1.0,
                              asr = false).nullity == repL.rank + dfree
        @test identifiability(dsL; torque_weight = 1.0).nullity == dfree
        # ... and the plan-B pair closes it anyway
        idLz = identifiability(dsL; torque_weight = 0.4, force_weight = 0.6)
        @test idLz.ncols == qL && idLz.nullity == 0
        fL = fit(SLCEFit, dsL, OLS(); torque_weight = 0.4, force_weight = 0.6)
        @test maximum(abs, fL.jphi .- betaL) < 1e-8
        @test maximum(abs, sum(predict_force(SLCEModel(fL), _id_cfg(rng, nat),
                                             0.05 * randn(rng, 3, nat)); dims = 2)) <
              1e-12
    end

    # --- plan B proper: fit on derivatives alone -----------------------------
    f = fit(SLCEFit, ds, OLS(); torque_weight = 0.4, force_weight = 0.6)
    fu = fit(SLCEFit, ds, OLS(); torque_weight = 0.4, force_weight = 0.6,
             asr = false)
    e0 = _id_cfg(rng, nat)
    u0 = 0.08 * randn(rng, 3, nat)
    u0 = u0 .- mean(u0; dims = 2)                      # a held-out COM-free point
    tvec = [0.09, -0.05, 0.04]

    @testset "forces + torques alone recover the model (ASR)" begin
        # zero energy weight: the energy data enter only through the analytic j0
        Xw, _, _, _, _ = _assemble_problem(ds, 0.4, 0.6, ds.asr)
        @test all(iszero, @view Xw[1:length(ds.y_E), :])   # energy block carries no weight
        @test maximum(abs, f.jphi .- beta_true) < 1e-8
        @test abs(f.j0 - 0.2) < 1e-10
        @test rmse_force(f) < 1e-10 && rmse_torque(f) < 1e-10
        @test rmse_energy(f) < 1e-10                        # energies follow for free
        mf = SLCEModel(f)
        Escale = max(abs(predict_energy(mf, e0, u0)), 1.0)
        @test abs(predict_energy(mf, e0, u0 .+ tvec) -
                  predict_energy(mf, e0, u0)) / Escale < 1e-12
        @test maximum(abs, sum(predict_force(mf, e0, u0); dims = 2)) < 1e-12
        @test asr_residual(mf) < 1e-13
        @test identifiability(f).nullity == 0
    end

    @testset "without the ASR the same data admit a different model" begin
        # The unconstrained fit reproduces every training derivative to machine
        # precision — and lands on a materially different coefficient vector.
        @test rmse_force(fu) < 1e-10 && rmse_torque(fu) < 1e-10
        @test maximum(abs, fu.jphi .- beta_true) > 1e-3
        @test identifiability(fu).nullity > 0
        mu = SLCEModel(fu)
        @test asr_residual(mu) > 1e-8
        # on the sampled (COM-free) slice the two models agree — the flat
        # direction is unobservable there ...
        @test abs(predict_energy(mu, e0, u0) - predict_energy(model, e0, u0)) < 1e-8
        @test maximum(abs, predict_force(mu, e0, u0) -
                           predict_force(model, e0, u0)) < 1e-8
        # ... and disagree the moment the configuration leaves it (a net drift),
        # which is exactly what the constraint buys: predictions off the slice.
        udrift = u0 .+ 0.05
        @test abs(predict_energy(mu, e0, udrift) -
                  predict_energy(model, e0, udrift)) > 1e-6
        # its Σ_a f_a vanishes on the slice too — that IS why the force channel
        # cannot see the violation there — and only breaks once u drifts.
        @test maximum(abs, sum(predict_force(mu, e0, u0); dims = 2)) < 1e-12
        @test maximum(abs, sum(predict_force(mu, e0, udrift); dims = 2)) > 1e-8
        @test maximum(abs, sum(predict_force(SLCEModel(f), e0, udrift);
                               dims = 2)) < 1e-12
    end

    @testset "identifiability API" begin
        # the fit method reassembles exactly what the fit solved
        @test identifiability(f) == identifiability(ds; torque_weight = 0.4,
                                                    force_weight = 0.6)
        @test identifiability(fu) == identifiability(ds; torque_weight = 0.4,
                                                     force_weight = 0.6,
                                                     asr = false)
        # a refit's support is not recorded on SLCEFit, so the report stays the
        # full-design one — a bound in the safe direction (dropping columns never
        # raises the nullity), as the docstring states
        fr = refit(f; threshold = 0.5 * maximum(abs, f.jphi))
        @test identifiability(fr) == identifiability(f)
        idr = identifiability(ds; torque_weight = 0.4, force_weight = 0.6)
        @test idr.sigma_max >= idr.sigma_min > 0
        # the cut is `min(size)`-based (LinearAlgebra.rank's), NOT `max(size)`:
        # the blocks are whitened by 1/√n, so a max-based tol would grow with the
        # row count and could flip a determined direction to flat
        Xr = _assemble_problem(ds, 0.4, 0.6, ds.asr)[1]
        @test idr.sigma_cut == minimum(size(Xr)) * eps() * idr.sigma_max
        @test idr.sigma_cut < maximum(size(Xr)) * eps() * idr.sigma_max
        # rtol is a knob: a cut above the whole spectrum kills every direction,
        # and an invalid one is refused
        @test identifiability(ds; torque_weight = 0.4, force_weight = 0.6,
                              rtol = 1.1).rank == 0
        @test_throws ArgumentError identifiability(ds; rtol = -1.0)
        # a pure-spin dataset: unconstrained coordinates either way
        bspin = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
        ms = SLCEModel(bspin, 0.0, randn(rng, n_salcs(bspin)))
        cfgs = [_id_cfg(rng, nat) for _ = 1:40]
        dss = SLCEDataset(bspin, cfgs, predict_energy(ms, cfgs))
        ids = identifiability(dss)
        @test ids.ncols == n_salcs(bspin) && ids.nullity == 0
        # fewer rows than columns is a (structural) deficiency, not an error —
        # and the energy block is CENTERED, so n rows carry only n − 1 ranks
        @test identifiability(dss[1:3]).rank == 2
        @test identifiability(dss[1:3]).nullity == n_salcs(bspin) - 2
        @test identifiability(dss[1:3]).sigma_min == 0.0
        # the error surface is the fit's, raised before any assembly
        @test_throws ArgumentError identifiability(ds; torque_weight = 1.2)
        @test_throws ArgumentError identifiability(ds; torque_weight = 0.6,
                                                   force_weight = 0.6)
        @test_throws ArgumentError identifiability(dss; force_weight = 0.5)
        @test_throws ArgumentError identifiability(ds; asr = false,
                                                   torque_weight = -0.1)
    end

    @testset "standing dead-column warning" begin
        # force-only: every pure-spin column is exactly zero
        @test_logs (:warn, r"carry no information") match_mode = :any fit(
            SLCEFit, ds, OLS(); force_weight = 1.0, asr = false)
        # ... and stays so under the constraint (γ coordinates)
        @test_logs (:warn, r"carry no information") match_mode = :any fit(
            SLCEFit, ds, OLS(); force_weight = 1.0)
        # the cut must be RELATIVE: on COM-free samples the SALCs proportional to
        # Σ_a u_a evaluate to ~1e-19 in the torque design — not to 0 — and they
        # are exactly the columns `build_asr` reports as structurally dead.
        XT, = _assemble_problem(ds, 1.0, 0.0)
        nrm = [norm(@view XT[:, j]) for j in axes(XT, 2)]
        near = findall(<=(1e-12 * maximum(nrm)), nrm)
        @test !isempty(near) && !any(j -> all(iszero, @view XT[:, j]), near)
        @test near == findall(j -> norm(@view rep.Z[j, :]) < 1e-12, 1:m)
        @test_logs (:warn, r"carry no information") match_mode = :any fit(
            SLCEFit, ds, OLS(); torque_weight = 1.0, asr = false)
        # a fit whose data touch every column is silent (a rank-deficient design
        # is NOT reported here — that is `identifiability`'s job): the plan-B
        # design is deficient by 54 directions and yet has no dead COLUMN
        @test_logs fit(SLCEFit, ds, OLS(); torque_weight = 0.4, force_weight = 0.6)
        @test_logs fit(SLCEFit, ds, OLS(); torque_weight = 0.4, force_weight = 0.6,
                       asr = false)
    end
end
