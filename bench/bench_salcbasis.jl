# SALC basis construction — the dominant hotspot (symmetry-adapted projection).
#
#   julia --project=bench bench/bench_salcbasis.jl [n] [lmax] [cutoff]
#
# `n` is the bcc Fe supercell size (default 4 → 128 atoms), `lmax` the per-species
# angular cutoff (default 3 — the seconds-scale stress baseline), `cutoff` the pair
# cutoff in Å (default 6.0, multi-shell). Pass `2 2 2.6` for the first-shell smoke
# case.

using SLCE
import Spglib                                   # activate the SpglibBackend extension
include(joinpath(@__DIR__, "fixtures.jl"))

n      = argn(1, 4)
lmax   = argn(2, 3)
cutoff = argf(3, 6.0)

bench_header("build_salc_basis — bcc Fe $(n)×$(n)×$(n), lmax=$lmax, cutoff=$cutoff")
cr   = bcc_fe(n)
spec = basis_spec(; nbody = 2, cutoff = cutoff, lmax = lmax)
println("atoms = $(n_atoms(cr))")

# Stage the inputs the SALC build consumes (these are cheap relative to it).
sg = analyze_symmetry(SpglibBackend(), cr)
nl = build_neighbor_list(cr, SLCE._superset_cutoff(spec), MinimumImage())
cs = build_clusters(cr, nl, sg; nbody = spec.nbody, selection = MinimumImage(),
                         cutoff = spec.cutoff)

bench_one("build_salc_basis", () -> build_salc_basis(cr, sg, cs;
              lmax_by_species = spec.lmax, scalar_only = !spec.soc))

b = build_salc_basis(cr, sg, cs; lmax_by_species = spec.lmax, scalar_only = !spec.soc)
println("→ n_salcs = $(length(b))")
