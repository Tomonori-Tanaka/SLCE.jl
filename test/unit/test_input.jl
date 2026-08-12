using Test
using SLCE
using LinearAlgebra
using StaticArrays

const _INPUT_FULL = """
[structure]
lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
positions = [[0.2, 0.0, 0.0], [0.8, 0.0, 0.0]]
species = [1, 1]
species_labels = ["Fe"]
pbc = [true, true, true]

[interaction]
nbody = 2
cutoff = 1.5
lmax = [2]
soc = true

[symmetry]
backend = "none"
tol = 1.0e-5
"""

# minimal: no [symmetry], no soc, no pbc → defaults apply
const _INPUT_MINIMAL = """
[structure]
lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]
positions = [[0.0, 0.0, 0.0]]
species = [1]
species_labels = ["Fe"]

[interaction]
nbody = 1
cutoff = 1.5
lmax = [2]
"""

_writetoml(s) = (p = tempname() * ".toml"; write(p, s); p)

@testset "TOML input files" begin
    @testset "read_setup parses structure / interaction / symmetry" begin
        inp = read_setup(_writetoml(_INPUT_FULL))
        @test n_atoms(inp.crystal) == 2
        @test inp.crystal.species_labels == ["Fe"]
        @test inp.crystal.species == [1, 1]
        @test inp.crystal.lattice.vectors == SMatrix{3,3,Float64}(3.0 * I)
        @test inp.crystal.frac_positions[:, 1] ≈ [0.2, 0.0, 0.0]
        @test inp.crystal.frac_positions[:, 2] ≈ [0.8, 0.0, 0.0]
        @test inp.crystal.lattice.pbc == SVector{3,Bool}(true, true, true)
        @test inp.spec.nbody == 2
        @test inp.spec.cutoff[1][1, 1] == 1.5
        @test inp.spec.lmax == [2]
        @test inp.spec.soc == true
        @test inp.backend isa NoSymmetry
        @test inp.tol == 1.0e-5
    end

    @testset "SLCEBasis(path) == building from the same Crystal/BasisSpec" begin
        path = _writetoml(_INPUT_FULL)
        b_file = SLCEBasis(path)
        inp = read_setup(path)
        b_manual = SLCEBasis(inp.crystal, inp.spec)
        @test b_file.salc_basis.keys == b_manual.salc_basis.keys
        @test n_salcs(b_file) == n_salcs(b_manual)
    end

    @testset "defaults for omitted keys" begin
        inp = read_setup(_writetoml(_INPUT_MINIMAL))
        @test inp.backend isa NoSymmetry          # no [symmetry] → NoSymmetry
        @test inp.tol == 1.0e-5                    # default tol
        @test inp.spec.soc == true    # default soc
        @test inp.crystal.lattice.pbc == SVector{3,Bool}(true, true, true)  # default pbc
        @test inp.tie_tol == 1.0e-8                # default same-distance band
    end

    # `tie_tol` changes the emitted basis (a widened band merges near-tie shells), so
    # a setup that needed one must round-trip through its own file — same rule as
    # `images`. The keyword overrides the file, and the file value reaches the
    # constructor's validation (the cap refuses).
    @testset "[interaction].tie_tol is carried and overridable" begin
        s = replace(_INPUT_FULL, "nbody = 2" => "nbody = 2\ntie_tol = 1e-5")
        inp = read_setup(_writetoml(s))
        @test inp.tie_tol == 1.0e-5
        @test SLCEBasis(_writetoml(s)) isa SLCEBasis                  # builds with it
        @test SLCEBasis(_writetoml(s); tie_tol = 1e-7) isa SLCEBasis  # override accepted
        bad = replace(_INPUT_FULL, "nbody = 2" => "nbody = 2\ntie_tol = 0.5")
        @test_throws ArgumentError SLCEBasis(_writetoml(bad))         # cap enforced
    end

    @testset "keyword arguments override the file's [symmetry]" begin
        path = _writetoml(_INPUT_FULL)
        @test SLCEBasis(path; tol = 1e-3).spacegroup.tol == 1e-3
        @test SLCEBasis(path).spacegroup.tol == 1e-5
    end

    @testset "backend name maps to the requested backend type" begin
        s = replace(_INPUT_FULL, "backend = \"none\"" => "backend = \"spglib\"")
        inp = read_setup(_writetoml(s))           # mapping only; building would need `using Spglib`
        @test inp.backend isa SpglibBackend
    end

    @testset "error paths" begin
        only_interaction = "[interaction]\nnbody = 1\ncutoff = 1.5\nlmax = [2]\n"
        @test_throws ArgumentError read_setup(_writetoml(only_interaction))   # no [structure]

        only_structure = """
        [structure]
        lattice = [[3.0,0.0,0.0],[0.0,3.0,0.0],[0.0,0.0,3.0]]
        positions = [[0.0,0.0,0.0]]
        species = [1]
        species_labels = ["Fe"]
        """
        @test_throws ArgumentError read_setup(_writetoml(only_structure))     # no [interaction]

        # missing required key inside a section
        no_lattice = replace(_INPUT_FULL,
            "lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]\n" => "")
        @test_throws ArgumentError read_setup(_writetoml(no_lattice))

        # lmax length ≠ number of species
        two_species = replace(_INPUT_FULL, "species_labels = [\"Fe\"]" => "species_labels = [\"Fe\", \"Pt\"]")
        @test_throws ArgumentError read_setup(_writetoml(two_species))        # lmax=[2] for 2 species

        # unrecognized backend
        bad_backend = replace(_INPUT_FULL, "backend = \"none\"" => "backend = \"xml\"")
        @test_throws ArgumentError read_setup(_writetoml(bad_backend))

        # malformed lattice (only 2 vectors)
        bad_lattice = replace(_INPUT_FULL,
            "lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]" =>
                "lattice = [[3.0, 0.0, 0.0], [0.0, 3.0, 0.0]]")
        @test_throws ArgumentError read_setup(_writetoml(bad_lattice))
    end
end
