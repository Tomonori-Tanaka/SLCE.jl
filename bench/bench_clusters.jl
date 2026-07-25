# Cluster enumeration — neighbour list + symmetry-orbit reduction.
#
#   julia --project=bench bench/bench_clusters.jl [n] [nbody] [cutoff]
#
# The N≥3 clique check (all C(N,2) edges simultaneously at minimum image) is the
# costly part, so the default is `nbody = 3` on the 128-atom cell (`n = 4`) with a
# multi-shell `cutoff = 6.0` Å. Pass `2 2 2.6` for the old first-shell smoke case.

using SLCE
import Spglib
include(joinpath(@__DIR__, "fixtures.jl"))

n      = argn(1, 4)
nbody  = argn(2, 3)
cutoff = argf(3, 6.0)

bench_header("cluster enumeration — bcc Fe $(n)×$(n)×$(n), nbody=$nbody, " *
             "cutoff=$cutoff")
cr   = bcc_fe(n)
spec = basis_spec(; nbody = nbody, cutoff = cutoff)
println("atoms = $(n_atoms(cr))")

sg = analyze_symmetry(SpglibBackend(), cr)

bench_one("build_neighbor_list",
          () -> build_neighbor_list(cr, SLCE._superset_cutoff(spec), MinimumImage()))
nl = build_neighbor_list(cr, SLCE._superset_cutoff(spec), MinimumImage())
println("→ neighbour pairs = $(length(nl.pairs))")

bench_one("build_clusters",
          () -> build_clusters(cr, nl, sg; nbody = nbody, selection = MinimumImage(),
                         cutoff = spec.cutoff))
cs = build_clusters(cr, nl, sg; nbody = nbody, selection = MinimumImage(),
                         cutoff = spec.cutoff)
println("→ orbits by body = $(sort(collect(k => length(v) for (k, v) in cs.by_body)))")
