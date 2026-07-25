# Nd2Fe14B — the realistic stress case: 68 atoms, 9 sublattice species, tetragonal
# P4_2/mnm (16 ops — low symmetry compared to bcc), per-species lmax with a
# non-magnetic species (B). Structure in bench/assets/nd2fe14b.toml. Benches the
# full pipeline the way a user runs it: basis build, design matrices, and the
# energy+torque co-fit sized like the real training set (103 configs, per-atom
# torques → 103·(1 + 3·68) rows).
#
#   julia --project=bench bench/bench_nd2fe14b.jl [nbody] [m] [cutoff]
#
# Defaults: nbody = 3, m = 103, cutoff = 4.0 Å — 3-body like the real l044 run, with
# the cutoff keeping the cliques compact (SLCE has no `lsum` cap, so an
# unbounded 3-body basis would blow up combinatorially). Pass `2 103 inf` for the
# every-resolvable-pair, many-orbit variant (205 pair orbits — Magesty's -1
# sentinel). `lmax` and `soc` come from the asset file
# ([4,4,2,2,2,2,2,2,0], soc = false — the isotropic/scalar channel).

using SLCE
import Spglib
include(joinpath(@__DIR__, "fixtures.jl"))

nbody  = argn(1, 3)
m      = argn(2, 103)
cutoff = argf(3, 4.0)

inp  = read_setup(joinpath(@__DIR__, "assets", "nd2fe14b.toml"))
cr   = inp.crystal
spec = BasisSpec(; nbody = nbody, cutoff = cutoff, lmax = inp.spec.lmax,
                 soc = inp.spec.soc)

bench_header("Nd2Fe14B — nbody=$nbody, $m configs, cutoff=$cutoff, " *
             "lmax=$(spec.lmax), soc=$(spec.soc)")
println("atoms = $(n_atoms(cr))   species = $(join(cr.species_labels, ","))")

# Stage-level timings (the SALC projection is the expected hotspot at 16 ops).
sg = analyze_symmetry(SpglibBackend(), cr; tol = inp.tol)
println("symmetry ops = $(SLCE.n_ops(sg))")
bench_one("build_neighbor_list",
          () -> build_neighbor_list(cr, SLCE._superset_cutoff(spec), MinimumImage()))
nl = build_neighbor_list(cr, SLCE._superset_cutoff(spec), MinimumImage())
bench_one("build_clusters",
          () -> build_clusters(cr, nl, sg; nbody = spec.nbody, selection = MinimumImage(),
                         cutoff = spec.cutoff))
cs = build_clusters(cr, nl, sg; nbody = spec.nbody, selection = MinimumImage(),
                         cutoff = spec.cutoff)
println("→ orbits by body = $(sort(collect(k => length(v) for (k, v) in cs.by_body)))")
bench_one("build_salc_basis", () -> build_salc_basis(cr, sg, cs;
              lmax_by_species = spec.lmax, isotropy = !spec.soc))

b = SLCEBasis(cr, spec; backend = SpglibBackend(), tol = inp.tol)
println("→ n_salcs = $(n_salcs(b))")

# Design matrices + fit, sized like the real l044 training set.
cfgs = rand_configs(cr, m)
rng  = MersenneTwister(2)
tqs  = [randn(rng, 3, n_atoms(cr)) for _ = 1:m]  # random targets — timing only

bench_one("_design_energy", () -> SLCE._design_energy(b, cfgs))
bench_one("_design_torque", () -> SLCE._design_torque(b, cfgs))

ds  = SLCEDataset(b, cfgs, randn(rng, m))
dst = SLCEDataset(b, cfgs, randn(rng, m), tqs)
println("energy rows = $m   torque rows = $(length(dst.y_T))")

bench_one("fit (OLS, energy)", () -> fit(SLCEFit, ds, OLS()))
bench_one("fit (OLS, co-fit w=0.5)",
          () -> fit(SLCEFit, dst, OLS(); torque_weight = 0.5))
bench_one("fit (Ridge, co-fit w=0.5)",
          () -> fit(SLCEFit, dst, Ridge(lambda = 1e-3); torque_weight = 0.5))
