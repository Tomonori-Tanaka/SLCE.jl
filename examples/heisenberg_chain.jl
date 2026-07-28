# Heisenberg chain end-to-end: build the SLCE basis for a spin chain, fit synthetic
# energies E = J Σ_⟨ij⟩ e_i·e_j, and recover the coupling J.
#
# Run:  julia --project=examples examples/heisenberg_chain.jl

using SLCE
import Spglib           # `import` (not `using`): just load it to activate the
                        # SpglibBackend extension, without importing names that
                        # clash with SLCE's `Lattice`/`Crystal`.
using LinearAlgebra
using Random

rng = MersenneTwister(2026)
randspin() = (v = randn(rng, 3); v / norm(v))
randcfg(nat) = mapreduce(_ -> randspin(), hcat, 1:nat)

# A 4-atom chain of identical spins along z (so neighboring spins differ).
lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
frac = [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75]
chain = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])

# Nearest-neighbor 2-body interaction, isotropic (Heisenberg) channel only.
interaction = BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], soc = false)
basis = SLCEBasis(chain, interaction; backend = SpglibBackend())
println("space group : ", basis.spacegroup.symbol, " (#", basis.spacegroup.number, ")")
println("# SALCs     : ", n_salcs(basis))

# Synthetic Heisenberg training data. (`salcs` is public but unexported — qualify it.)
# Canonical members list each physical bond once, so Σ over members is Σ_{⟨ij⟩}.
heis = SLCE.salcs(basis)[1]            # the nearest-neighbor Heisenberg SALC
J_true = 0.0137
configs = [randcfg(4) for _ = 1:40]
E = [J_true * sum(dot(c[:, m.atoms[1]], c[:, m.atoms[2]]) for m in heis.members)
     for c in configs]

# Fit and recover J (the SALC normalization gives J = 2√3 · jphi).
f = fit(SLCEFit, SLCEDataset(basis, configs, E), OLS())
J_recovered = 2 * sqrt(3) * coef(f)[1]

println("R²          : ", round(r2_energy(f); digits = 12))
println("J (true)    : ", J_true)
println("J (recovered): ", J_recovered)
@assert isapprox(J_recovered, J_true; rtol = 1e-6)
println("✓ recovered the Heisenberg coupling")

# --- energy + torque co-fit ---------------------------------------------------
# The same coupling fixes the per-atom torque τ_a = −e_a × ∂E/∂e_a (the physical /
# Landau–Lifshitz torque). For the Heisenberg energy above,
# ∂E/∂e_a = J Σ_{members} (δ_{a,i} e_j + δ_{a,j} e_i).
function heisenberg_torque(c, J)
    nat = size(c, 2)
    G = zeros(3, nat)
    for m in heis.members
        i, j = m.atoms[1], m.atoms[2]
        G[:, i] .+= J .* c[:, j]
        G[:, j] .+= J .* c[:, i]
    end
    return reduce(hcat, cross(G[:, a], c[:, a]) for a = 1:nat)   # τ = ∇E × e = −e × ∇E
end
torques = [heisenberg_torque(c, J_true) for c in configs]

fc = fit(SLCEFit, SLCEDataset(basis, configs, E, torques), OLS(); torque_weight = 0.5)
println("\nco-fit R² (energy): ", round(r2_energy(fc); digits = 12))
println("co-fit R² (torque): ", round(r2_torque(fc); digits = 12))
@assert isapprox(predict_torque(fc, configs[1]), torques[1]; atol = 1e-6)
println("✓ predict_torque matches the analytic Heisenberg torque")
