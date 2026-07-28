# Volume grids (src/slce/strainedmodels.jl) — `StrainedModels`, the K(ε) container
# (design record §9a, §14 M5-3).
#
# THE ACCEPTANCE GATE, and what it is made of. On a grid there are two strain derivatives
# that must agree: the INTRA-MODEL INCREMENTAL one, taken inside a single model with its
# coefficients fixed, and the GRID FINITE DIFFERENCE, taken along the grid, which also
# carries the drift of the coefficients. The record predicted this one test would catch "a
# wrong shear factor, a mislabelled η and basis truncation" — and building it caught two of
# the three for real, which is why the fixture below looks the way it does:
#
#   • BASIS TRUNCATION. The grid's basis must be closed under RE-EXPANSION. A model at
#     scale s is the reference model re-expanded around the scaled geometry, and that shift
#     is lower-triangular in degree: `degree = 2` content generates degrees 0, 1 and 2, and
#     spin-dressed `degree = 1` content generates a PURE-SPIN term. A grid whose points lack
#     those degrees cannot represent its own neighbours, and the two derivatives then
#     disagree by percent — measured: 5% with the pure-spin sector missing, 1% with the
#     lattice `degree = 1` sector missing, 4e-7 (the interpolation error) with both present.
#   • A MISLABELLED η. `dE/dη` and `dE/ds` differ by exactly the factor `s`, because η is
#     measured from the reference the derivative is taken AT and the total scale is
#     `s(1 + η)`. Dropping it agrees at the unstrained point and is off by the strain
#     everywhere else — the shape of error that looks like agreement on a first look.
#
# The residual 4e-7 is the coefficient interpolation error and nothing else, which the
# convergence test below establishes by halving the grid's span.

using Test
using SLCE
using SLCE: build_asr, _lift_gamma, _scaled_basis, _scaled_spec, salcs
using LinearAlgebra
using Random
using StaticArrays

const _SM_A0 = 3.0

_sm_crystal(s) = Crystal(Lattice(Matrix(s * _SM_A0 * I(3))),
                         [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])

# Every sector's cutoff scales with `s` — the key-stability pin (§9a). The four sectors are
# the closure requirement above: pure spin, spin × degree 1, lattice degree 1, lattice
# degree 2. Drop any one and the acceptance gate fails by percent rather than by 1e-7.
function _sm_spec(cr, s)
    rc = s * 1.1 * _SM_A0 / 3
    return BasisSpec(cr; lmax = 2, pmax = 2,
                     sectors = [Sector(spin = (sites = 1:2,), cutoff = rc),
                                Sector(spin = [1, 1], disp = (degree = 1,), sites = 1:2,
                                       cutoff = rc),
                                Sector(disp = (degree = 1,), sites = 1:2, cutoff = rc),
                                Sector(disp = (degree = 2,), sites = 1:2, cutoff = rc)])
end

_sm_basis(s) = (cr = _sm_crystal(s); SLCEBasis(cr, _sm_spec(cr, s)))

_sm_unit(rng) = normalize(randn(rng, 3))
_sm_cfg(rng, n) = reduce(hcat, [_sm_unit(rng) for _ = 1:n])

# One ground truth for the whole grid: a model at the reference cell, evaluated at each
# grid point through the affine path. That is the only way to generate data at several
# volumes from ONE physical energy surface, which is what makes the gate meaningful.
function _sm_truth(rng)
    b = _sm_basis(1.0)
    rep = build_asr(b)
    return SLCEModel(b, -1.5, _lift_gamma(rep, randn(rng, size(rep.Z, 2))))
end

function _sm_fit_at(rng, truth, s; n = 220)
    b = _sm_basis(s)
    cr = _sm_crystal(s)
    prov = DatumProvenance(; reference_id = "s$s",
                           reference_fingerprint = crystal_fingerprint(cr))
    M = (s - 1) * Matrix(1.0I(3))
    data = map(1:n) do _
        e = _sm_cfg(rng, 2)
        u = 0.05 .* randn(rng, 3, 2)
        TrainingDatum(; energy = affine_energy(truth, e, M; base = u), directions = e,
                      magmoms = ones(2), displacements = u, provenance = prov)
    end
    ds = SLCEDataset(b, data; use_torque = false, use_force = false)
    return SLCEModel(fit(SLCEFit, ds, OLS(); asr = true))
end

function _sm_grid(rng, truth, grid)
    return StrainedModels([_sm_fit_at(rng, truth, s) for s in grid], collect(grid))
end

@testset "volume grids (StrainedModels)" begin

    @testset "the acceptance gate: intra-model incremental ≡ grid finite difference" begin
        rng = MersenneTwister(0x5347)
        truth = _sm_truth(rng)
        grid = [0.98, 0.99, 1.0, 1.01, 1.02]
        sm = _sm_grid(rng, truth, grid)
        @test length(sm) == 5
        @test SLCE.scales(sm) ≈ grid
        e = _sm_cfg(rng, 2)
        worst = 0.0
        for s in (0.985, 0.99, 1.0, 1.005, 1.01)
            D = strain_derivatives(model_at(sm, s); spins = e, order = 1)
            intra = D[1, 1] + D[2, 2] + D[3, 3]          # ε = ηI contracts to the trace
            gridd = grid_strain_derivative(sm, s; spins = e)
            @test intra ≈ gridd rtol = 1e-5
            worst = max(worst, abs(intra - gridd) / abs(intra))
        end
        # ...and the residual really is the coefficient interpolation, not a bug: halve the
        # span and it falls, because the same polynomial degree now covers a quarter of the
        # curvature
        tight = _sm_grid(rng, truth, [0.99, 0.995, 1.0, 1.005, 1.01])
        wt = 0.0
        for s in (0.9925, 1.0, 1.0075)
            D = strain_derivatives(model_at(tight, s); spins = e, order = 1)
            intra = D[1, 1] + D[2, 2] + D[3, 3]
            g = grid_strain_derivative(tight, s; spins = e)
            wt = max(wt, abs(intra - g) / abs(intra))
        end
        @test wt < worst
    end

    @testset "η is measured from the reference the derivative is taken at" begin
        # `dE/dη = s·dE/ds`. Both derivatives use the same convention, so their RATIO is 1
        # at every scale; a version that dropped the factor would sit at 1 for s = 1 and at
        # s elsewhere, which is the mislabelled-η failure this pins.
        rng = MersenneTwister(0x5348)
        truth = _sm_truth(rng)
        sm = _sm_grid(rng, truth, [0.98, 0.99, 1.0, 1.01, 1.02])
        e = _sm_cfg(rng, 2)
        for s in (0.99, 1.01)
            D = strain_derivatives(model_at(sm, s); spins = e, order = 1)
            intra = D[1, 1] + D[2, 2] + D[3, 3]
            gridd = grid_strain_derivative(sm, s; spins = e)
            @test intra / gridd ≈ 1.0 rtol = 1e-4
            @test !isapprox(intra / (gridd / s), 1.0; rtol = 1e-3)   # the factor is real
        end
    end

    @testset "model_at reproduces the grid points, and the surgery is the real basis" begin
        rng = MersenneTwister(0x5349)
        truth = _sm_truth(rng)
        grid = [0.98, 1.0, 1.02]
        models = [_sm_fit_at(rng, truth, s) for s in grid]
        sm = StrainedModels(models, grid)
        for (i, s) in enumerate(grid)
            m = model_at(sm, s)
            @test m.j0 ≈ models[i].j0 atol = 1e-10
            @test norm(m.jphi - models[i].jphi) < 1e-9 * max(1.0, norm(models[i].jphi))
        end
        # the surgery (`_scaled_basis`) must reproduce a basis BUILT at that scale — cell,
        # keys, members and the folded tensors, which is what licenses interpolating
        # coefficients across the grid at all
        for s in grid
            cut = _scaled_basis(models[2].basis, s / 1.0)   # models[2] is the s = 1 point
            built = _sm_basis(s)
            @test cut.crystal.lattice.vectors ≈ built.crystal.lattice.vectors
            @test cut.salc_basis.keys == built.salc_basis.keys
            @test cut.spec.cutoff ≈ built.spec.cutoff
            for (sa, sb) in zip(salcs(cut), salcs(built))
                @test [m.atoms for m in sa.members] == [m.atoms for m in sb.members]
                @test [m.shifts for m in sa.members] == [m.shifts for m in sb.members]
                @test all(isapprox(ta.folded, tb.folded; rtol = 1e-12, atol = 1e-14)
                          for (ma, mb) in zip(sa.members, sb.members)
                          for (ta, tb) in zip(ma.terms, mb.terms))
            end
        end
        @test SLCE.volumes(sm) ≈ [(s * _SM_A0)^3 for s in grid]
        @test occursin("StrainedModels", sprint(show, sm))
    end

    @testset "what the constructor refuses" begin
        rng = MersenneTwister(0x534a)
        truth = _sm_truth(rng)
        m1 = _sm_fit_at(rng, truth, 1.0)
        m2 = _sm_fit_at(rng, truth, 1.02)
        @test_throws ArgumentError StrainedModels([m1], [1.0])
        @test_throws ArgumentError StrainedModels([m1, m2], [1.0, 1.02]; abscissa = :area)
        @test_throws ArgumentError StrainedModels([m1, m2], [1.0, 1.02]; degree = 5)
        @test_throws DimensionMismatch StrainedModels([m1, m2], [1.0])
        # a scale label that disagrees with its own cell: the label names a geometry
        @test_throws ArgumentError StrainedModels([m1, m2], [1.0, 1.05])
        # an ABSOLUTE cutoff — the shell crosses it as the volume changes and the key sets
        # diverge, which is exactly what §9a's key-stability pin is about
        crb = _sm_crystal(1.02)
        fixed = SLCEBasis(crb, BasisSpec(crb; lmax = 2, pmax = 2,
            sectors = [Sector(spin = (sites = 1:2,), cutoff = 1.1),
                       Sector(spin = [1, 1], disp = (degree = 1,), sites = 1:2,
                              cutoff = 1.1),
                       Sector(disp = (degree = 1,), sites = 1:2, cutoff = 1.1),
                       Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
        mf = SLCEModel(fixed, 0.0, zeros(n_salcs(fixed)))
        err = try
            StrainedModels([m1, mf], [1.0, 1.02])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("cutoff", err.msg) || occursin("SALC key set", err.msg)
        # a cell that is not an isotropic scaling
        crt = Crystal(Lattice(SMatrix{3,3,Float64}([3.0 0 0; 0 3.0 0; 0 0 3.12])),
                      [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        bt = SLCEBasis(crt, _sm_spec(crt, 1.02))
        mt = SLCEModel(bt, 0.0, zeros(n_salcs(bt)))
        @test_throws ArgumentError StrainedModels([m1, mt], [1.0, 1.02])
        # `model_at` / `grid_strain_derivative` on a nonsense scale
        sm = StrainedModels([m1, m2], [1.0, 1.02])
        @test_throws ArgumentError model_at(sm, -1.0)
        @test_throws ArgumentError grid_strain_derivative(sm, 0.0)
        @test_throws ArgumentError grid_strain_derivative(sm, 1.0)   # spins are required
    end

    @testset "gate (w): the abscissa is a modelling choice, and it is measured" begin
        # Leave-one-out against a DIRECTLY fitted point — the record's own prescription for
        # this gate. The result is not a tie: the map relating two grid points is the §9d
        # re-expansion, which is exactly polynomial in the affine displacement and hence in
        # the LINEAR scale, of the same degree as the displacement content. In the volume it
        # is a polynomial in `s³`, which no polynomial in `V` reproduces. That is why the
        # default is `:linear`, and why the choice is a field rather than a convention.
        rng = MersenneTwister(0x5350)
        truth = _sm_truth(rng)
        grid = [0.98, 0.99, 1.0, 1.01, 1.02]
        models = [_sm_fit_at(rng, truth, s) for s in grid]
        keep = [1, 2, 4, 5]
        direct = models[3]
        err = Dict{Symbol,Float64}()
        for ab in (:linear, :volume, :logvolume)
            loo = StrainedModels(models[keep], grid[keep]; abscissa = ab)
            m = model_at(loo, 1.0)
            err[ab] = norm(m.jphi - direct.jphi) / norm(direct.jphi)
        end
        @test err[:linear] < 1e-11                       # exact, up to the fits' own noise
        @test err[:volume] > 100 * err[:linear]          # the choice is not cosmetic
        @test err[:volume] < 1e-5                       # ...but each choice is usable here
        @test err[:logvolume] < 1e-7
        # The degree is the other half of the modelling choice, and the two abscissas make
        # the point from opposite sides. In `:linear` the map between grid points is a
        # polynomial of the displacement content's own degree — 2 here — so degree 2 is
        # already exact and raising it buys nothing: every value sits at the fits' noise
        # floor. In `:volume` the same coefficients are polynomials in `s³`, no polynomial
        # in `V` reproduces them, and the error falls monotonically with degree, which is
        # what a modelling choice looks like when it actually is one.
        direct995 = _sm_fit_at(rng, truth, 0.995)          # ONE reference, not one per loop
        scale995 = norm(direct995.jphi)
        for d in (2, 3, 4)
            lin = StrainedModels(models, grid; degree = d, abscissa = :linear)
            @test norm(model_at(lin, 0.995).jphi - direct995.jphi) < 1e-9 * scale995
        end
        prev = Inf
        for d in (2, 3, 4)
            vol = StrainedModels(models, grid; degree = d, abscissa = :volume)
            e = norm(model_at(vol, 0.995).jphi - direct995.jphi)
            @test e < prev
            prev = e
        end
    end

    @testset "the frozen disp_scale and the home-image condition" begin
        # Both are asserted in the constructor and neither can be triggered from the public
        # API today: `disp_scale ≠ 1` is refused at spec construction (the assertion is
        # there for the day that lifts), and along a SIMILARITY grid the member images
        # cannot drift on their own — the check exists for a grid assembled from
        # independently standardized cells, which no public constructor produces. What is
        # testable is that the invariants hold on a real grid, i.e. that the checks are
        # comparing something.
        rng = MersenneTwister(0x534b)
        truth = _sm_truth(rng)
        sm = StrainedModels([_sm_fit_at(rng, truth, s) for s in (1.0, 1.02)], [1.0, 1.02])
        a, b = sm.models[1].basis, sm.models[2].basis
        @test a.spec.disp_scale == b.spec.disp_scale
        @test a.salc_basis.keys == b.salc_basis.keys
        @test a.salc_basis.fingerprint == b.salc_basis.fingerprint
        @test all(m1.shifts == m2.shifts
                  for (s1, s2) in zip(salcs(a), salcs(b))
                  for (m1, m2) in zip(s1.members, s2.members))
        @test _scaled_spec(a.spec, 1.02).cutoff ≈ b.spec.cutoff
    end
end
