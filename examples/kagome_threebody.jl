# Three-body SLCE on a kagome lattice: the three sites of a kagome triangle are
# symmetry-equivalent (C₃ᵥ permutes them), so a 3-body term with unequal site angular
# momenta — e.g. l = (1,1,2) — must combine the three l-orderings into one
# multi-term SALC. This example builds the arbitrary-body-order basis, shows the
# multi-term channel, and recovers a synthetic in-span 3-body model from energy data.
#
# Run:  julia --project=examples examples/kagome_threebody.jl

using SLCE
import Spglib
using LinearAlgebra
using Random

rng = MersenneTwister(2026)
randcfg(nat) = reduce(hcat, (v = randn(rng, 3); v / norm(v)) for _ = 1:nat)

# Kagome: hexagonal cell, three equivalent sites forming triangles.
a, c = 2.0, 8.0
lat = Lattice([a -a/2 0.0; 0.0 a*sqrt(3)/2 0.0; 0.0 0.0 c])
frac = [0.0 0.5 0.5; 0.5 0.0 0.5; 0.0 0.0 0.0]
kagome = Crystal(lat, frac, [1, 1, 1], ["Fe"])

interaction = BasisSpec(; nbody = 3, cutoff = 1.2, lmax = [2])   # up to 3-body, l ≤ 2
basis = SLCEBasis(kagome, interaction; backend = SpglibBackend())
println("space group : ", basis.spacegroup.symbol, " (#", basis.spacegroup.number, ")")
println("# SALCs     : ", n_salcs(basis))

# A 3-body, unequal-l channel is a multi-term SALC (combines l-orderings).
# (`salcs` is public but unexported — qualify it.)
s112 = first(s for s in SLCE.salcs(basis)
             if s.key.body == 3 && SLCE.spin_ls(s.key) == [1, 1, 2] && s.Lf == 0)
println("3-body ls=[1,1,2], Lf=0 SALC: ", length(s112.members[1].terms), " orderings (terms) per member")

# Synthetic in-span data: random true couplings, recover them from energies + torques.
m = n_salcs(basis)
configs = [randcfg(3) for _ = 1:200]
skel = SLCEDataset(basis, configs, zeros(length(configs)))
J_true = randn(rng, m)
j0_true = 0.5
energies = j0_true .+ skel.X_E * J_true
model0 = SLCEModel(basis, j0_true, J_true)   # a synthetic model from hand-set couplings
torques = [predict_torque(model0, c) for c in configs]

f = fit(SLCEFit, SLCEDataset(basis, configs, energies, torques), OLS(); torque_weight = 0.3)
println("\nR² energy   : ", round(r2_energy(f); digits = 12))
println("R² torque   : ", round(r2_torque(f); digits = 12))
println("max |ΔJ|    : ", round(maximum(abs, coef(f) .- J_true); sigdigits = 3))
@assert isapprox(coef(f), J_true; atol = 1e-6)
@assert isapprox(predict_torque(f, configs[1]), torques[1]; atol = 1e-6)
println("✓ recovered the 3-body couplings (energy + torque co-fit)")
