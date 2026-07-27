using Test
using SLCE
using LinearAlgebra
using Random

# The training-data boundary enforces the spin-config contract the harmonic kernels
# assume (3 × n_atoms, finite, unit-norm columns). These checks pin that malformed
# input fails loudly at construction / prediction instead of silently biasing the fit.

@testset "data-boundary validation" begin
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = false)
    basis = SLCEBasis(crystal, interaction)
    nat = n_atoms(crystal)

    rng = MersenneTwister(1)
    good = [randcfg(rng, nat) for _ = 1:8]
    energies = randn(rng, length(good))

    @testset "valid input constructs / predicts" begin
        ds = SLCEDataset(basis, good, energies)
        @test nobs(fit(SLCEFit, ds, OLS())) == length(good)
        model = SLCEModel(fit(SLCEFit, ds, OLS()))
        @test predict_energy(model, good) == [predict_energy(model, c) for c in good]
        @test length(predict_torque(model, good)) == length(good)
    end

    model = SLCEModel(fit(SLCEFit, SLCEDataset(basis, good, energies), OLS()))

    @testset "SLCEDataset rejects malformed configs" begin
        # non-unit column
        bad = deepcopy(good); bad[3][:, 1] .*= 1.5
        @test_throws ArgumentError SLCEDataset(basis, bad, energies)
        # NaN
        bad = deepcopy(good); bad[2][1, 2] = NaN
        @test_throws ArgumentError SLCEDataset(basis, bad, energies)
        # wrong atom count
        @test_throws DimensionMismatch SLCEDataset(basis, [randcfg(rng, nat + 1)], [0.0])
        # wrong row count
        @test_throws ArgumentError SLCEDataset(basis, [randn(rng, 2, nat)], [0.0])
        # config / energy length mismatch
        @test_throws DimensionMismatch SLCEDataset(basis, good, energies[1:end-1])
    end

    @testset "torque dataset rejects malformed configs / lengths" begin
        torques = [zeros(3, nat) for _ in good]
        @test nobs(fit(SLCEFit, SLCEDataset(basis, good, energies, torques), OLS())) == length(good)
        bad = deepcopy(good); bad[1][:, 2] .= 0.0           # zero-norm column
        @test_throws ArgumentError SLCEDataset(basis, bad, energies, torques)
        @test_throws ArgumentError SLCEDataset(basis, good, energies, torques[1:end-1])
    end

    @testset "predict rejects malformed configs (scalar + batch)" begin
        @test_throws ArgumentError predict_energy(model, 2.0 .* good[1])
        @test_throws ArgumentError predict_torque(model, 2.0 .* good[1])
        @test_throws DimensionMismatch predict_energy(model, randcfg(rng, nat + 1))
        # batch path validates serially up front (clean error, not a TaskFailedException)
        bad = deepcopy(good); bad[4][:, 1] .*= 3.0
        @test_throws ArgumentError predict_energy(model, bad)
        @test_throws ArgumentError predict_torque(model, bad)
    end

    @testset "atol keyword loosens / tightens the unit-norm check" begin
        nearly = deepcopy(good); nearly[1][:, 1] .*= (1 + 1e-4)
        @test_throws ArgumentError SLCEDataset(basis, nearly, energies)         # default 1e-6
        @test nobs(fit(SLCEFit, SLCEDataset(basis, nearly, energies; atol = 1e-3), OLS())) == length(good)
    end

    @testset "a component outside [-1,1] is refused, and atol cannot license it" begin
        # The kernel's Legendre recursion has the HARD domain |e_z| ≤ 1, so this used
        # to pass the boundary and throw a DomainError from inside the threaded design
        # assembly, naming neither the config nor the atom. Tightening `atol` does not
        # help: ‖e‖ − 1 = 5e-9 clears a 1e-8 band and still throws.
        pole = deepcopy(good); pole[1][:, 1] = [0.0, 0.0, 1 + 5e-9]
        @test abs(norm(pole[1][:, 1]) - 1) < 1e-8          # inside any sane atol band
        @test_throws ArgumentError SLCEDataset(basis, pole, energies)
        @test_throws ArgumentError SLCEDataset(basis, pole, energies; atol = 1e-3)
        @test_throws ArgumentError predict_energy(model, pole[1])
        # ... while an off-norm column whose components all sit inside the domain is
        # still accepted, so this rejects nothing the tolerance was meant to allow
        tilt = deepcopy(good); tilt[1][:, 1] = [0.6, 0.8, 0.0] .* (1 + 1e-7)
        @test maximum(abs, tilt[1][:, 1]) < 1
        @test SLCEDataset(basis, tilt, energies) isa SLCEDataset
    end
end
