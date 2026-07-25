using Test

# Construction internals are public but unexported (the `SLCEBasis` constructor drives them
# for you); the suite reaches them by qualification. Brought into the test module here so
# every included unit file sees them without each repeating the import.
using SLCE: build_neighbor_list, build_clusters, build_salc_basis, evaluate_salc,
    salcs, SALC, SALCKey, SALCBasis, NeighborPair, NeighborList, ClusterMember,
    ClusterOrbit, ClusterSet, analyze_symmetry, n_ops, SymOp, SpaceGroup,
    interplanar_spacing, solve_coefficients

const TEST_MODE = get(ENV, "TEST_MODE", "default")

@testset "SLCE.jl" begin
    if TEST_MODE in ("default", "all", "unit")
        include("unit/testutils.jl")     # shared helpers (rand_unit / randcfg /
                                         # canonical-member gates)
        include("unit/test_geometry.jl")
        include("unit/test_harmonics.jl")
        include("unit/test_solidharmonics.jl")
        include("unit/test_countingoracle.jl")
        include("unit/test_angmom.jl")
        include("unit/test_coupledbasis.jl")
        include("unit/test_symmetry.jl")
        include("unit/test_clusters.jl")
        include("unit/test_imageselection.jl")
        include("unit/test_truncation.jl")
        include("unit/test_ws_nbody.jl")
        include("unit/test_decor.jl")
        include("unit/test_salc.jl")
        include("unit/test_mixedsalc.jl")
        include("unit/test_sectorbasis.jl")
        include("unit/test_fit.jl")
        include("unit/test_selection.jl")
        include("unit/test_validation.jl")
        include("unit/test_torque.jl")
        include("unit/test_nbody.jl")
        include("unit/test_persist.jl")
        include("unit/test_input.jl")
        include("unit/test_dftsource.jl")
        include("unit/test_dataset.jl")
        include("unit/test_embset.jl")
        include("unit/test_coeftable.jl")
        include("unit/test_sunny.jl")
        include("unit/test_introspect.jl")
        include("unit/test_threading.jl")
    end
    if TEST_MODE in ("default", "all", "aqua")
        include("aqua.jl")
    end
    if TEST_MODE in ("all", "jet")
        include("jet.jl")
    end
end
