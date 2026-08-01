# Design-matrix assembly — energy (`evaluate_salc`) and torque (`accumulate_grad!`)
# kernels over all SALC columns × configurations.
#
#   julia --project=bench bench/bench_design_matrix.jl [n] [m] [lmax] [cutoff]
#
# `n` supercell size (default 4 → 128 atoms), `m` number of spin configurations
# (default 100), `lmax` angular cutoff, `cutoff` pair cutoff in Å (default 6.0,
# multi-shell). The torque block is the heavier of the two (gradient + cross product
# per atom, 3·n_atoms rows per config).

using SLCE
import Spglib
include(joinpath(@__DIR__, "fixtures.jl"))

n      = argn(1, 4)
m      = argn(2, 100)
lmax   = argn(3, 2)
cutoff = argf(4, 6.0)

bench_header("design matrices — bcc Fe $(n)×$(n)×$(n), $m configs, lmax=$lmax, " *
             "cutoff=$cutoff")
cr   = bcc_fe(n)
spec = basis_spec(; nbody = 2, cutoff = cutoff, lmax = lmax)
b    = SLCEBasis(cr, spec; backend = SpglibBackend())
configs = rand_configs(cr, m)
println("atoms = $(n_atoms(cr))   n_salcs = $(n_salcs(b))   configs = $m")

# `_design_energy` / `_design_torque` are internal kernels (not exported).
bench_one("_design_energy", () -> SLCE._design_energy(b, configs))
bench_one("_design_torque", () -> SLCE._design_torque(b, configs))
