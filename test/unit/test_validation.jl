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
    interaction = BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], isotropy = true)
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
end
