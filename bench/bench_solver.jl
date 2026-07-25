# Regression solve — `solve_coefficients` for OLS vs Ridge across matrix sizes.
#
#   julia --project=bench bench/bench_solver.jl
#
# Synthetic, column-centred design matrices (the solver contract: `X` is already
# centred and adds no intercept). Cheap enough for `@belapsed` sampling.

using SLCE
include(joinpath(@__DIR__, "fixtures.jl"))      # bench_header, @belapsed, mean, MersenneTwister

bench_header("solve_coefficients — OLS vs Ridge")

for (M, P) in ((2_000, 200), (5_000, 500), (10_000, 800))
    rng = MersenneTwister(0)
    X = randn(rng, M, P)
    X .-= mean(X; dims = 1)                     # column-centred (solver contract)
    y = randn(rng, M)
    t_ols   = @belapsed solve_coefficients(OLS(), $X, $y) samples = 3 evals = 1
    t_ridge = @belapsed solve_coefficients(Ridge(lambda = 1e-3), $X, $y) samples = 3 evals = 1
    @printf("M=%-6d P=%-5d   OLS=%8.3f ms   Ridge=%8.3f ms   ratio=%.2f\n",
            M, P, 1e3 * t_ols, 1e3 * t_ridge, t_ridge / t_ols)
end
