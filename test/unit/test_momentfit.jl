# The moment channel's dataset + fit (src/fitting/momentfit.jl): target assembly
# under the mode rule (hand-arithmetic oracle), mode-1 ≡ mode-4 identity
# equivalence, the decomposability gate (|M| sin²θ, the |M| = 0 pass, gated vs
# ungated disclosure, the antiparallel census), mode-1 zero-axis exclusion,
# per-orbit survival (single- and multi-orbit) + the coverage-floor refusal, the
# loud requirement checks (incl. nonzero displacements and mixed references),
# dataset-level time-reversal covariance (bitwise, both modes), the per-orbit μ₀
# intercept (residual invariance under a per-orbit constant shift), the
# vanishing-column freeze, and the predict_moment door.
#
# NOTE: the planted-model "recovery" gates the mask/solve/predict PLUMBING — the
# data are generated through _design_moment itself. The design matrix has its own
# independent oracles in test_momentbasis.jl.
#
# Reuses the FeGe B20 fixture (_mb_fege / _MBFixedSG / _mb_unit) from
# test_momentbasis.jl — runtests.jl includes that file first.

using Test
using SLCE
using SLCE: _design_moment, n_salcs
using LinearAlgebra
using Random

function _mf_datum(e::Matrix{Float64}; M::Matrix{Float64}, mode::Int,
                   axes::Union{Matrix{Float64},Nothing} = nothing,
                   setup::String = "mfit")
    return TrainingDatum(; energy = 0.0, directions = e,
                         magmoms = ones(size(e, 2)), moments_bare = M,
                         constraint_mode = mode, constraint_axes = axes,
                         provenance = DatumProvenance(; setup_id = setup))
end

# The FeGe primitive pointed design at the fixture spec carries one structural
# tie dependency (columns 2–3, named by moment_resolvability and warned by the
# MomentDataset ctor), so every construction warns "structurally dependent" and
# every OLS solve warns rank-deficient. The helpers assert exactly that (two
# solves per fit: gated + ungated) and keep the suite quiet; match_mode :any lets
# the drop @info pass through where it also fires.
_mf_ds(mbx, data; kw...) =
    @test_logs (:warn, r"structurally dependent") match_mode = :any MomentDataset(
        mbx, data; kw...)
_mf_fit(ds) = @test_logs (:warn, r"rank deficient") (:warn, r"rank deficient") fit(
    MomentFit, ds)

@testset "moment dataset + fit" begin
    xt, sg = _mb_fege()
    bk = _MBFixedSG(sg)
    nat = 8
    spec = MomentSpec(; lmax_env = [1, 1], sampled = [true, true], lmax_mark = 1,
                      nbody = 2, cutoff_pair = 3.0, marked = [true, false])
    mb = MomentBasis(xt, spec; backend = bk)
    p = n_salcs(mb)
    marked = mb.marked_atoms
    nm = length(marked)
    @test nm == 4                      # the Fe 4a orbit; Ge is unmarked
    rng = MersenneTwister(2026)
    beta = randn(rng, p)               # the planted pointed model

    # Decomposable synthetic data from the planted model: y = X·β exactly, M built
    # back as y·ê (+ optional transverse remainder t and target corruption δ on the
    # `dirty` configs — a dirty row has g ≈ |t| and a target the model does NOT fit,
    # so the gate's keep/reject decision is observable in the residuals).
    function _mf_data(ncfg; mode::Int = 4, ndirty::Int = 0, tperp::Float64 = 0.3,
                      delta::Float64 = 0.5, setup::String = "mfit")
        data = TrainingDatum[]
        for c = 1:ncfg
            e = _mb_unit(rng, nat)
            yv = _design_moment(mb, [e], [e]) * beta
            M = zeros(3, nat)
            for (ai, a) in enumerate(marked)
                y0 = yv[ai]
                if c <= ndirty
                    y0 += delta * randn(rng)
                    t = randn(rng, 3)
                    t .-= dot(t, e[:, a]) .* e[:, a]
                    t .*= tperp / norm(t)
                    M[:, a] .= y0 .* e[:, a] .+ t
                else
                    M[:, a] .= y0 .* e[:, a]
                end
            end
            push!(data, _mf_datum(e; M, mode,
                                  axes = mode == 1 ? copy(e) : nothing, setup))
        end
        return data
    end

    @testset "target assembly (hand oracle) + mode rule" begin
        e = _mb_unit(rng, nat)
        M = randn(rng, 3, nat)
        ax = _mb_unit(rng, nat)
        d4 = _mf_datum(e; M, mode = 4)
        d1 = _mf_datum(e; M, mode = 1, axes = copy(ax))
        ds = _mf_ds(mb, [d4, d1]; gate_eps = 1e6, coverage_floor = 0.0)
        @test size(ds.X) == (2nm, p)
        for (ai, a) in enumerate(marked)
            # mode 4: ê = the datum's direction; mode 1: ê = the constraint axis —
            # both targets and gates recomputed here by hand arithmetic
            y4 = e[1, a] * M[1, a] + e[2, a] * M[2, a] + e[3, a] * M[3, a]
            y1 = ax[1, a] * M[1, a] + ax[2, a] * M[2, a] + ax[3, a] * M[3, a]
            mm = norm(M[:, a])
            @test ds.y[ai] == y4
            @test ds.y[nm + ai] == y1
            # independent gate oracle: |M| sin²θ via the angle, not the code's form
            th4 = acos(clamp(y4 / mm, -1, 1))
            th1 = acos(clamp(y1 / mm, -1, 1))
            @test ds.gate[ai] ≈ mm * sin(th4)^2 rtol = 1e-9
            @test ds.gate[nm + ai] ≈ mm * sin(th1)^2 rtol = 1e-9
            @test ds.row_config[ai] == 1 && ds.row_atom[ai] == a
            @test ds.row_config[nm + ai] == 2 && ds.row_atom[nm + ai] == a
        end
        # the design rows are exactly the marked-column-substituted evaluations
        @test ds.X == vcat(_design_moment(mb, [e], [e]),
                           _design_moment(mb, [e], [ax]))
        # everything decomposes at a huge eps; nothing is undefined
        @test all(ds.defined) && all(ds.keep)
    end

    @testset "mode-1 with axes = directions ≡ mode-4 (identity substitution)" begin
        e = _mb_unit(rng, nat)
        M = randn(rng, 3, nat)
        ds4 = _mf_ds(mb, [_mf_datum(e; M, mode = 4)];
                     gate_eps = 1e6, coverage_floor = 0.0)
        ds1 = _mf_ds(mb, [_mf_datum(e; M, mode = 1, axes = copy(e))];
                     gate_eps = 1e6, coverage_floor = 0.0)
        @test ds1.X == ds4.X
        @test ds1.y == ds4.y
    end

    @testset "planted-model recovery (prediction equivalence)" begin
        data = _mf_data(40)
        ds = _mf_ds(mb, data; gate_eps = 1e-8)
        @test all(ds.keep)
        # the OLS rank warning asserted by _mf_fit has a name: the structural
        # resolvability rank of this cell is 4 of 5 columns
        mr = moment_resolvability(mb)
        @test mr.rank == 4
        @test mr.rank + length(mr.null_combinations) == length(mr.kept)
        f = _mf_fit(ds)
        @test norm(residuals(f)) < 1e-7
        @test rmse_moment(f) < 1e-8
        # held-out configuration: the fitted model reproduces the planted model's
        # predictions (rank-robust — any flat direction of the training design is
        # structural and annihilates the held-out rows too)
        eh = _mb_unit(rng, nat)
        model = MomentModel(f)
        @test predict_moment(model, eh) ≈ _design_moment(mb, [eh], [eh]) * beta atol = 1e-7
        @test predict_moment(f, eh) == predict_moment(model, eh)
        @test coef(f) === f.coeffs
        @test model.provenance.setup_id == "mfit"
    end

    @testset "decomposability gate: keep/reject + disclosure + |M| = 0 pass" begin
        data = _mf_data(40; ndirty = 10)
        ds = @test_logs (:warn, r"structurally dependent") (:info, r"excluded") match_mode =
            :any MomentDataset(mb, data; gate_eps = 1e-6)
        # exactly the dirty configs' rows are rejected (g ≈ tperp ≫ eps on dirty,
        # ≈ 0 on clean), and none are undefined
        @test all(ds.defined)
        @test count(ds.keep) == 30 * nm
        @test all(r -> ds.keep[r] == (ds.row_config[r] > 10), 1:length(ds.keep))
        t = only(ds.orbit_report)
        @test t.n_rows == 40 * nm && t.n_kept == 30 * nm && t.survival == 0.75
        @test t.atoms == marked
        @test t.n_anti == 0                      # mode 4: axis IS the direction
        # hand oracle for the disclosed transverse rms: 40 of 160 defined rows
        # carry |M⊥| = 0.3 exactly (the dirty construction), the rest ≈ 0
        @test t.mperp_rms ≈ sqrt(40 * 0.3^2 / 160) rtol = 1e-9
        # the gate is doing physics: the gated solve fits exactly, the ungated one
        # is polluted by the corrupted dirty targets
        f = _mf_fit(ds)
        @test rmse_moment(f) < 1e-8
        @test rmse_moment(f; gated = false) > 1e-2
        @test f.coeffs != f.coeffs_ungated
        # a quenched moment (M = 0) has g = 0 and passes with target 0
        e = _mb_unit(rng, nat)
        M = randn(rng, 3, nat)
        M[:, marked[1]] .= 0.0
        dq = @test_logs (:warn, r"structurally dependent") (:info, r"excluded") match_mode =
            :any MomentDataset(mb, [_mf_datum(e; M, mode = 4)];
                               gate_eps = 0.0, coverage_floor = 0.0)
        @test dq.keep[1] && dq.y[1] == 0.0 && dq.gate[1] == 0.0
    end

    @testset "mode-1 zero-axis rows are excluded, not fitted" begin
        data = _mf_data(20; mode = 1)
        # unconstrain atom marked[2] in the first 4 configs
        for c = 1:4
            data[c].constraint_axes[:, marked[2]] .= 0.0
        end
        ds = @test_logs (:warn, r"structurally dependent") (:info,
            r"undefined-axis") match_mode = :any MomentDataset(mb, data;
                                                               gate_eps = 1e-8)
        bad = findall(r -> !ds.defined[r], 1:length(ds.defined))
        @test length(bad) == 4
        @test all(r -> ds.row_atom[r] == marked[2] && ds.row_config[r] <= 4, bad)
        @test all(isnan, ds.y[bad]) && all(isnan, ds.gate[bad])
        @test count(ds.keep) == 20nm - 4
        t = only(ds.orbit_report)
        @test t.n_defined == 20nm - 4 && t.n_kept == 20nm - 4
        # the placeholder design row is the identity-substituted evaluation
        e1 = data[1].directions
        @test ds.X[bad[1], :] == vec(_design_moment(mb, [e1], [e1])[
            findfirst(==(marked[2]), marked), :])
        # both fits run on the defined rows only, and still recover the model
        f = _mf_fit(ds)
        @test length(residuals(f)) == 20nm - 4
        @test length(residuals(f; gated = false)) == 20nm - 4
        @test rmse_moment(f) < 1e-8
    end

    @testset "coverage floor + loud requirements" begin
        # 60 % dirty at a tight eps → survival 0.4 < floor 0.5 → refuse
        data = _mf_data(20; ndirty = 12)
        @test_throws ArgumentError MomentDataset(mb, data; gate_eps = 1e-6)
        # ... but an explicitly lowered floor accepts the same data
        ds = @test_logs (:warn, r"structurally dependent") (:info, r"excluded") match_mode =
            :any MomentDataset(mb, data; gate_eps = 1e-6, coverage_floor = 0.3)
        @test count(ds.keep) == 8 * nm
        # requirement checks
        e = _mb_unit(rng, nat)
        M = randn(rng, 3, nat)
        @test_throws ArgumentError MomentDataset(mb, TrainingDatum[]; gate_eps = 1e-6)
        @test_throws UndefKeywordError MomentDataset(mb, [_mf_datum(e; M, mode = 4)])
        @test_throws ArgumentError MomentDataset(mb, [_mf_datum(e; M, mode = 4)];
                                                 gate_eps = -1.0)
        @test_throws ArgumentError MomentDataset(mb, [_mf_datum(e; M, mode = 4)];
                                                 gate_eps = 1e-6,
                                                 coverage_floor = 1.5)
        no_m = TrainingDatum(; energy = 0.0, directions = e, magmoms = ones(nat),
                             constraint_mode = 4)
        @test_throws ArgumentError MomentDataset(mb, [no_m]; gate_eps = 1e-6)
        no_mode = TrainingDatum(; energy = 0.0, directions = e, magmoms = ones(nat),
                                moments_bare = M)
        @test_throws ArgumentError MomentDataset(mb, [no_mode]; gate_eps = 1e-6)
        # setup uniformity (same check as SLCEDataset)
        mixed = [_mf_datum(e; M, mode = 4, setup = "a"),
                 _mf_datum(e; M, mode = 4, setup = "b")]
        @test_throws ArgumentError MomentDataset(mb, mixed; gate_eps = 1e6,
                                                 coverage_floor = 0.0)
        # atom-count mismatch dies at the door
        e4 = _mb_unit(rng, 4)
        small = _mf_datum(e4; M = randn(rng, 3, 4), mode = 4)
        @test_throws ArgumentError MomentDataset(mb, [small]; gate_eps = 1e-6)
        # gate_eps = 0 with a floor of 0 constructs an empty-keep dataset; the fit
        # door refuses it (the ctor's floor would otherwise have fired first)
        dirty = _mf_data(5; ndirty = 5)
        ds0 = @test_logs (:warn, r"structurally dependent") (:info,
            r"excluded") match_mode = :any MomentDataset(mb, dirty; gate_eps = 0.0,
                                                         coverage_floor = 0.0)
        @test !any(ds0.keep)
        @test_throws ArgumentError fit(MomentFit, ds0)
    end

    @testset "time reversal at the dataset level (bitwise)" begin
        data = _mf_data(6)
        flipped = [TrainingDatum(; energy = d.energy, directions = -d.directions,
                                 magmoms = d.magmoms,
                                 moments_bare = -d.moments_bare,
                                 constraint_mode = 4, provenance = d.provenance)
                   for d in data]
        ds = _mf_ds(mb, data; gate_eps = 1e-8)
        dsf = _mf_ds(mb, flipped; gate_eps = 1e-8)
        # even-Σl labels (mark rank counts): every design entry is TR-even exactly,
        # and y = ê·M is invariant under (ê, M) → (−ê, −M)
        @test dsf.X == ds.X
        @test dsf.y == ds.y
        # same statement for mode 1 (axes flip with everything else)
        data1 = _mf_data(4; mode = 1)
        flip1 = [TrainingDatum(; energy = d.energy, directions = -d.directions,
                               magmoms = d.magmoms, moments_bare = -d.moments_bare,
                               constraint_mode = 1,
                               constraint_axes = -d.constraint_axes,
                               provenance = d.provenance) for d in data1]
        dsa = _mf_ds(mb, data1; gate_eps = 1e-8)
        dsb = _mf_ds(mb, flip1; gate_eps = 1e-8)
        @test dsb.X == dsa.X
        @test dsb.y == dsa.y
    end

    @testset "per-orbit μ₀ intercept absorbs a constant shift" begin
        data = _mf_data(30)
        shifted = TrainingDatum[]
        c0 = 0.37
        for d in data
            M = copy(d.moments_bare)
            for a in marked
                M[:, a] .+= c0 .* d.directions[:, a]     # y → y + c0, M⊥ unchanged
            end
            push!(shifted, TrainingDatum(; energy = d.energy,
                                         directions = d.directions,
                                         magmoms = d.magmoms, moments_bare = M,
                                         constraint_mode = 4,
                                         provenance = d.provenance))
        end
        f = _mf_fit(_mf_ds(mb, data; gate_eps = 1e-8))
        fs = _mf_fit(_mf_ds(mb, shifted; gate_eps = 1e-8))
        # the l = 0 1-body [MARK] μ₀ column of the (single) Fe orbit absorbs the
        # shift: residuals are invariant (which coefficients move is not asserted)
        @test maximum(abs, residuals(fs) - residuals(f)) < 1e-9
    end

    @testset "predict_moment plumbing" begin
        f = _mf_fit(_mf_ds(mb, _mf_data(12); gate_eps = 1e-8))
        model = MomentModel(f)
        e = _mb_unit(rng, nat)
        ax = _mb_unit(rng, nat)
        # explicit axes reproduce the training-row evaluation; default is axes = e
        @test predict_moment(model, e; axes = ax) ==
              vec(_design_moment(mb, [e], [ax]) * model.coeffs)
        @test predict_moment(model, e) == predict_moment(model, e; axes = e)
        @test_throws DimensionMismatch predict_moment(model, e;
                                                      axes = _mb_unit(rng, 4))
        @test_throws DimensionMismatch MomentModel(mb, randn(rng, p + 1),
                                                   model.provenance)
        # the DOOR: off-norm e is refused; off-norm MARKED axes column is refused;
        # unmarked axes columns are free and provably never read
        ebad = copy(e); ebad[:, 1] .*= 1.5
        @test_throws ArgumentError predict_moment(model, ebad)
        axbad = copy(e); axbad[:, marked[1]] .*= 1.5
        @test_throws ArgumentError predict_moment(model, e; axes = axbad)
        ge = first(setdiff(1:nat, marked))
        ax2 = copy(e); ax2[:, ge] .= [10.0, -3.0, 0.5]
        @test predict_moment(model, e; axes = ax2) == predict_moment(model, e)
    end

    @testset "resolvability wiring + vanishing-column freeze" begin
        ds = _mf_ds(mb, _mf_data(10); gate_eps = 1e-8)
        mr = moment_resolvability(mb)
        @test ds.vanishing == mr.vanishing == Int[]
        @test ds.dependent == mr.null_combinations
        @test length(only(ds.dependent)) == 2      # the columns-2/3 tie
        # freeze MECHANISM (no vanishing column exists on this fixture, so drive
        # the branch directly through the inner ctor): fit must return exact 0.0
        # for the frozen column and the reduced solve for the rest
        ds2 = MomentDataset(ds.basis, ds.X, ds.y, ds.defined, ds.keep, ds.gate,
                            ds.row_config, ds.row_atom, ds.orbit_rep,
                            ds.orbit_report, ds.order, [2], ds.dependent,
                            ds.gate_eps, ds.coverage_floor, ds.provenance)
        f2 = fit(MomentFit, ds2)     # dropping col 2 removes the tie: no warning
        @test f2.coeffs[2] == 0.0 && f2.coeffs_ungated[2] == 0.0
        act = [1, 3, 4, 5]
        @test f2.coeffs[act] ≈ ds.X[ds.keep, act] \ ds.y[ds.keep] rtol = 1e-10
    end

    @testset "antiparallel census (mode-1 gauge rows)" begin
        data = _mf_data(6; mode = 1)
        for c = 1:3
            data[c].constraint_axes[:, marked[1]] .*= -1.0   # re-gauged axis
        end
        ds = _mf_ds(mb, data; gate_eps = 1e-8)
        t = only(ds.orbit_report)
        @test t.n_anti == 3
        @test all(ds.keep)          # the gate is even in y: gauge rows still pass
    end

    @testset "displacement + reference doors" begin
        e = _mb_unit(rng, nat)
        M = randn(rng, 3, nat)
        moved = TrainingDatum(; energy = 0.0, directions = e, magmoms = ones(nat),
                              moments_bare = M, constraint_mode = 4,
                              displacements = 0.01 * randn(rng, 3, nat))
        @test_throws ArgumentError MomentDataset(mb, [moved]; gate_eps = 1e6,
                                                 coverage_floor = 0.0)
        atref = TrainingDatum(; energy = 0.0, directions = e, magmoms = ones(nat),
                              moments_bare = M, constraint_mode = 4,
                              displacements = zeros(3, nat))
        dz = _mf_ds(mb, [atref]; gate_eps = 1e6, coverage_floor = 0.0)
        @test size(dz.X, 1) == nm    # u ≡ 0 asserts the reference — accepted
        mixed_ref = [_mf_datum(e; M, mode = 4),
                     TrainingDatum(; energy = 0.0, directions = e,
                                   magmoms = ones(nat), moments_bare = M,
                                   constraint_mode = 4,
                                   provenance = DatumProvenance(;
                                       setup_id = "mfit", reference_id = "other"))]
        @test_throws ArgumentError MomentDataset(mb, mixed_ref; gate_eps = 1e6,
                                                 coverage_floor = 0.0)
    end

    @testset "multi-orbit bookkeeping (Fe + Ge marked)" begin
        spec2 = MomentSpec(; lmax_env = [1, 1], sampled = [true, true],
                           lmax_mark = 1, nbody = 2, cutoff_pair = 3.0,
                           marked = [true, true])
        mb2 = MomentBasis(xt, spec2; backend = bk)
        m2 = mb2.marked_atoms
        @test m2 == collect(1:8)
        e = _mb_unit(rng, nat)
        # Fe rows decomposable, Ge rows carry a large transverse remainder
        M = zeros(3, nat)
        for a = 1:4
            M[:, a] .= 1.3 .* e[:, a]
        end
        for a = 5:8
            t = randn(rng, 3); t .-= dot(t, e[:, a]) .* e[:, a]
            M[:, a] .= 0.2 .* e[:, a] .+ 0.4 .* t ./ norm(t)
        end
        d = _mf_datum(e; M, mode = 4)
        # per-orbit disaggregation: Fe orbit (rep 1) clean, Ge orbit (rep 5) dirty
        ds = @test_logs (:warn, r"structurally dependent") (:info, r"excluded") match_mode =
            :any MomentDataset(mb2, [d]; gate_eps = 1e-6, coverage_floor = 0.0)
        @test [t.orbit for t in ds.orbit_report] == [1, 5]
        @test ds.orbit_report[1].atoms == [1, 2, 3, 4]
        @test ds.orbit_report[2].atoms == [5, 6, 7, 8]
        @test ds.orbit_rep == [1, 1, 1, 1, 5, 5, 5, 5]
        @test ds.orbit_report[1].n_kept == 4 && ds.orbit_report[2].n_kept == 0
        @test ds.orbit_report[2].mperp_rms ≈ 0.4 rtol = 1e-9
        @test length(ds.dependent) == 2
        # the per-orbit coverage refusal names the failing (Ge) orbit
        err = try
            MomentDataset(mb2, [d]; gate_eps = 1e-6)
            nothing
        catch ex
            ex
        end
        @test err isa ArgumentError && occursin("orbit 5", err.msg)
    end

    @testset "fast design path ≡ full evaluation" begin
        # the member-index path must be value-identical to evaluating every
        # member (dead members only ever add exact zeros)
        for trial = 1:3
            e = _mb_unit(rng, nat)
            ax = trial == 1 ? e : _mb_unit(rng, nat)
            Xf = SLCE._design_moment(mb, [e], [ax])
            Xs = SLCE._design_moment(mb, [e], [ax]; member_index = false)
            @test Xf == Xs
        end
        # and through a 3-body star basis (multi-member orbits, both species env)
        psp = MomentSpec(; lmax_env = [1, 1], sampled = [true, true],
                         lmax_mark = 1, nbody = 3, cutoff_pair = 3.0,
                         marked = [true, false])
        mbs = MomentBasis(xt, psp; backend = bk)
        e = _mb_unit(rng, nat)
        @test SLCE._design_moment(mbs, [e], [e]) ==
              SLCE._design_moment(mbs, [e], [e]; member_index = false)
    end

    @testset "band-profile diagnostic" begin
        data = _mf_data(24)
        ds = _mf_ds(mb, data; gate_eps = 1e-8)
        f = _mf_fit(ds)
        # the order coordinate is the marked-sublattice |⟨e⟩| — hand oracle
        for c in (1, 24)
            m = sum(data[c].directions[:, a] for a in marked)
            @test ds.order[c] ≈ norm(m) / nm rtol = 1e-12
        end
        prof = moment_band_profile(f)
        # hand re-computation of the whole profile from public fields
        pred = ds.X * coef(f)
        mres = [sum((ds.y .- pred)[(c-1)*nm+1:c*nm]) / nm for c = 1:24]
        @test prof.mean_residual ≈ mres atol = 1e-12
        @test sum(b.n for b in prof.bands) == 24
        @test prof.bands[1].hi <= prof.bands[end].lo   # ordered bins
        # bin-free line: independent least-squares via the normal equations
        A = [ones(24) ds.order]
        ab = (A' * A) \ (A' * mres)
        @test prof.intercept ≈ ab[1] atol = 1e-10
        @test prof.slope ≈ ab[2] atol = 1e-10
        @test abs(prof.r) <= 1 + 1e-12
        # a planted linear-in-order corruption is recovered by the slope
        data2 = TrainingDatum[]
        for (c, d) in enumerate(data)
            M = copy(d.moments_bare)
            for a in marked
                M[:, a] .+= (0.5 * ds.order[c]) .* d.directions[:, a]
            end
            push!(data2, TrainingDatum(; energy = d.energy,
                                       directions = d.directions,
                                       magmoms = d.magmoms, moments_bare = M,
                                       constraint_mode = 4,
                                       provenance = d.provenance))
        end
        ds2 = _mf_ds(mb, data2; gate_eps = 1e-8)
        prof2 = moment_band_profile(MomentModel(f), ds2)
        @test prof2.slope ≈ prof.slope + 0.5 atol = 1e-6
        # gated rows are excluded from the profile: a config whose rows are all
        # gate-rejected disappears
        data3 = _mf_data(6; ndirty = 1)
        ds3 = @test_logs (:warn, r"structurally dependent") (:info,
            r"excluded") match_mode = :any MomentDataset(mb, data3;
                                                         gate_eps = 1e-6,
                                                         coverage_floor = 0.5)
        prof3 = moment_band_profile(MomentModel(f), ds3)
        @test length(prof3.mean_residual) == 5
        @test_throws ArgumentError moment_band_profile(f; nbins = 0)
        # fewer configs than bins: every config still lands in a band (the naive
        # div-by-nbins binning dropped the high-|⟨e⟩| end — review 2026-08-20)
        ds4 = _mf_ds(mb, _mf_data(3); gate_eps = 1e-8)
        prof4 = moment_band_profile(MomentModel(f), ds4; nbins = 4)
        @test sum(b.n for b in prof4.bands) == 3
        @test length(prof4.bands) == 3
        # a model on a different basis object is refused (same-width nonsense door)
        spec_alt = MomentSpec(; lmax_env = [1, 1], sampled = [true, true],
                              lmax_mark = 1, nbody = 2, cutoff_pair = 3.0,
                              marked = [true, false])
        mb_alt = MomentBasis(xt, spec_alt; backend = bk)
        malt = MomentModel(mb_alt, coef(f), f.dataset.provenance)
        @test_throws ArgumentError moment_band_profile(malt, ds3)
    end
end
