# Persistence + TOML input end-to-end: build a basis from a human-authored
# `input.toml`, fit a model, save it to a self-contained TOML document, reload it,
# and confirm the reloaded model predicts identically.
#
# Run:  julia --project=examples examples/persist_and_input.jl

using SLCE
import Spglib           # `import` (not `using`): activate the SpglibBackend extension
using LinearAlgebra
using Random

rng = MersenneTwister(2026)
randcfg(nat) = mapreduce(_ -> (v = randn(rng, 3); v / norm(v)), hcat, 1:nat)

# --- 1. a human-authored input file (inline crystal + interaction + symmetry) ----
input_toml = """
[structure]
lattice = [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0], [0.0, 0.0, 10.0]]   # each entry = one lattice vector
positions = [[0.0, 0.0, 0.0], [0.0, 0.0, 0.25], [0.0, 0.0, 0.5], [0.0, 0.0, 0.75]]
species = [1, 1, 1, 1]
species_labels = ["Fe"]

[interaction]
nbody = 2
cutoff = 2.6
lmax = [1]
soc = false

[symmetry]
backend = "spglib"
tol = 1.0e-5
"""

dir = mktempdir()
input_path = joinpath(dir, "input.toml")
write(input_path, input_toml)

basis = SLCEBasis(input_path)        # reads the [symmetry] backend/tol from the file
println("from input.toml → space group ", basis.spacegroup.symbol,
        " (#", basis.spacegroup.number, "), ", n_salcs(basis), " SALC(s)")

# --- 2. fit synthetic Heisenberg data --------------------------------------------
heis = SLCE.salcs(basis)[1]   # public-unexported: call it qualified
J_true = 0.0137
configs = [randcfg(4) for _ = 1:40]
E = [J_true * sum(dot(c[:, m.atoms[1]], c[:, m.atoms[2]]) for m in heis.members)
     for c in configs]
f = fit(SLCEFit, SLCEDataset(basis, configs, E), OLS())
model = SLCEModel(f)
println("fitted J = ", round(2 * sqrt(3) * coef(f)[1]; digits = 6), "  (true ", J_true, ")")
@assert isapprox(2 * sqrt(3) * coef(f)[1], J_true; rtol = 1e-6)

# the coefficients as a Tables.jl source — DataFrame(coeftable(f)) / CSV.write(...) also work
println("\n", coeftable(f))

# --- 3. save → reload → predict identically --------------------------------------
model_path = joinpath(dir, "model.toml")
SLCE.save(model_path, model)
reloaded = SLCE.load(SLCEModel, model_path)

test = [randcfg(4) for _ = 1:10]
@assert predict_energy(reloaded, test) == predict_energy(model, test)
@assert reloaded.jphi == model.jphi
println("✓ reloaded model predicts identically (", filesize(model_path), "-byte TOML)")

# the document is human-readable — show its head
println("\n--- model.toml (first lines) ---")
foreach(println, Iterators.take(eachline(model_path), 12))

# a basis alone round-trips too
basis_path = joinpath(dir, "basis.toml")
SLCE.save(basis_path, basis)
@assert SLCE.load(SLCEBasis, basis_path).salc_basis.keys == basis.salc_basis.keys
println("\n✓ basis round-trips as well")
