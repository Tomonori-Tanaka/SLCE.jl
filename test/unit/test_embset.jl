# The EMBSET reader (src/io/embset.jl): Magesty's legacy training-set format into
# SpinDatum. The format rules (comment/blank filtering, block-shape auto-detection,
# ≥7-column atom lines, index-column check) are pinned here on hand-written files;
# agreement with Magesty's own reader is pinned in test/oracle (which carries the
# pinned Magesty dependency the core suite deliberately omits).

using Test
using SLCE
using LinearAlgebra

# 2 configs × 3 atoms; atom 1 of config 1 carries an extra (ignored) 8th column,
# atom 3 is non-magnetic (zero moment AND zero field), indentation and blank lines
# vary on purpose.
const _EMBSET_SAMPLE = """
# config 1
  -12.5
  1   2.0 0.0 0.0   0.0 0.5 0.0   0.123
  2   0.0 1.5 0.0   0.0 0.0 0.0
  3   0.0 0.0 0.0   0.0 0.0 0.0

# config 2
  -13.0
1 0.0 0.0 2.0 0.1 0.0 0.0
2 1.5 0.0 0.0 0.0 0.2 0.0
3 0.0 0.0 0.0 0.0 0.0 0.0
"""

@testset "EMBSET reader" begin
    dir = mktempdir()
    path = joinpath(dir, "EMBSET")
    write(path, _EMBSET_SAMPLE)

    @testset "values / derived quantities" begin
        data = read_embset(path)
        @test length(data) == 2
        @test data[1].energy == -12.5 && data[2].energy == -13.0
        @test data[1].directions[:, 1] ≈ [1.0, 0.0, 0.0]
        @test data[1].magmoms ≈ [2.0, 1.5, 0.0]
        @test data[1].field[:, 1] ≈ [0.0, 0.5, 0.0]
        # τ = m × B: (2,0,0) × (0,0.5,0) = (0,0,1)
        @test data[1].torques[:, 1] ≈ [0.0, 0.0, 1.0]
        # non-magnetic atom: ẑ placeholder, zero torque
        @test data[1].directions[:, 3] ≈ [0.0, 0.0, 1.0]
        @test data[1].torques[:, 3] ≈ zeros(3)
        @test data[2].directions[:, 2] ≈ [1.0, 0.0, 0.0]
        @test data[2].magmoms[1] ≈ 2.0
    end

    @testset "detection variants" begin
        base = read_embset(path)
        # explicit n_atoms matches the auto-detection
        d2 = read_embset(path; n_atoms = 3)
        @test [d.energy for d in d2] == [d.energy for d in base]
        @test d2[1].directions == base[1].directions
        # a file with no `#` separators parses identically (token-based detection)
        stripped = join([l for l in split(_EMBSET_SAMPLE, '\n')
                         if !startswith(strip(l), "#")], '\n')
        p2 = joinpath(dir, "EMBSET_nocomment")
        write(p2, stripped)
        d3 = read_embset(p2)
        @test [d.energy for d in d3] == [d.energy for d in base]
        @test d3[2].field == base[2].field
    end

    @testset "EmbsetFile source → dataset" begin
        src = EmbsetFile(path)
        @test read_configs(src)[1].energy == -12.5
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2 0.5; 0.0 0.0 0.5; 0.0 0.0 0.5],
                     [1, 1, 2], ["Fe", "B"])
        basis = SLCEBasis(cr, BasisSpec(cr; nbody = 2, cutoff = 1.5,
                                       lmax = ["Fe" => 1, "B" => 0], isotropy = true))
        ds = SLCEDataset(basis, src)                  # zero-moment B passes the guard
        @test length(ds) == 2 && has_torque(ds)
        @test ds.y_E == [-12.5, -13.0]
    end

    @testset "error paths" begin
        @test_throws ArgumentError read_embset(joinpath(dir, "missing"))
        write(joinpath(dir, "empty"), "# nothing\n\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "empty"))
        write(joinpath(dir, "noenergy"), "1 1.0 0.0 0.0 0.0 0.0 0.0\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "noenergy"))
        # dropping one atom line breaks the block divisibility
        broken = join(split(strip(_EMBSET_SAMPLE), '\n')[1:(end - 1)], '\n')
        write(joinpath(dir, "broken"), broken)
        @test_throws ArgumentError read_embset(joinpath(dir, "broken"))
        write(joinpath(dir, "short"), "-1.0\n1 1.0 0.0 0.0 0.0 0.0\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "short"))   # 6 columns
        write(joinpath(dir, "permuted"),
              "-1.0\n2 1.0 0.0 0.0 0.0 0.0 0.0\n1 0.0 1.0 0.0 0.0 0.0 0.0\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "permuted"))
        write(joinpath(dir, "nonnum"), "-1.0\n1 1.0 abc 0.0 0.0 0.0 0.0\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "nonnum"))
        # non-finite values (a failed SCF) must not become silent training rows
        write(joinpath(dir, "nanenergy"), "NaN\n1 1.0 0.0 0.0 0.0 0.0 0.0\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "nanenergy"))
        write(joinpath(dir, "infmom"), "-1.0\n1 Inf 0.0 0.0 0.0 0.0 0.0\n")
        @test_throws ArgumentError read_embset(joinpath(dir, "infmom"))
        @test_throws ArgumentError read_embset(path; n_atoms = 2)  # 8 lines % 3 ≠ 0
        @test_throws ArgumentError read_embset(path; n_atoms = 0)
    end
end
