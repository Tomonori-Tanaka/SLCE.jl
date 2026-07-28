# The phonopy export, gated against phonopy itself.
#
# `FORCE_CONSTANTS` is a positional format: a matrix over supercell atom indices, with
# the supercell built by phonopy from the unit cell and `--dim`. Disagree with its
# ordering and the export is a PERMUTED force-constant matrix — which still
# diagonalizes, still has three acoustic modes at Γ, and is simply wrong. No
# self-contained round trip can see that; only phonopy can. Hence this suite, and
# hence its own environment: phonopy is a Python package the core suite must not need.
#
# Run it with a Python that has phonopy importable:
#
#     PHONOPY_PYTHON=/path/to/venv/bin/python julia --project=test/phonopy \
#         test/phonopy/runtests.jl
#
# THE FIXTURE IS LOAD-BEARING, and three properties of it are not decoration:
#
#  1. **Mass-weighted comparison, with two species of different mass.** phonopy's
#     phase factor depends only on the lattice translation, so permuting the basis
#     atoms within a cell is a similarity transform and leaves the spectrum alone.
#     Measured: dropping the species permutation entirely moves an UNWEIGHTED
#     comparison by 8e-16 — i.e. not at all — and a mass-weighted one by 1.7e-2.
#  2. **Interleaved species** (`[Fe, Ni, Fe]`), so the POSCAR's species grouping is a
#     real permutation rather than the identity.
#  3. **Lattice shifts that are not symmetric under swapping the a₁ and a₂ axes.** The
#     first fixture tried here had `R ∈ {(0,0,0), ±(1,1,0)}`, which IS symmetric, and
#     reversing the lattice-point axis order produced a BYTE-IDENTICAL file. The
#     positions below place one bond wrapping in x only and another in y only.
#
# Mutation-tested, all three against this fixture: correct 7.4e-16; lattice-point axis
# order reversed 1.8e-1; atom and lattice-point nesting swapped 3.2e-1; species
# permutation dropped 1.7e-2.

using Test
using SLCE
using LinearAlgebra
using Random

const PY = get(ENV, "PHONOPY_PYTHON", "python3")

function _have_phonopy()
    try
        return success(pipeline(`$PY -c "import phonopy"`; stdout = devnull,
                                stderr = devnull))
    catch
        return false
    end
end

_have_phonopy() || error("""
    phonopy is not importable from `$PY`. Point PHONOPY_PYTHON at an interpreter
    that has it, e.g. `python3 -m venv env && ./env/bin/pip install phonopy` then
    `PHONOPY_PYTHON=\$PWD/env/bin/python julia --project=test/phonopy \\
        test/phonopy/runtests.jl`.""")

const MASSES = [55.845, 58.6934, 55.845]        # Fe, Ni, Fe — see property 1 above

function _fixture()
    lat = Lattice([3.3 0.0 0.0; 0.0 3.9 0.0; 0.0 0.0 7.4])
    # atom 2 sits next to atom 1 across the x boundary, atom 3 across the y boundary,
    # so the shift set contains (±1, 0, 0) AND (0, ±1, 0) — property 3.
    cr = Crystal(lat, [0.06 0.93 0.10; 0.10 0.14 0.88; 0.20 0.23 0.26],
                 [1, 2, 1], ["Fe", "Ni"])
    b = SLCEBasis(cr, BasisSpec(cr; lmax = 0, pmax = 2,
                                sectors = [Sector(disp = (degree = 2,), cutoff = 3.0)]))
    m = SLCEModel(b, 0.0, randn(MersenneTwister(5), n_salcs(b)))
    return force_constants(m; order = 2)
end

const COMPARE = raw"""
import numpy as np, os, sys
os.chdir(sys.argv[1])
from phonopy import Phonopy
from phonopy.interface.calculator import read_crystal_structure
from phonopy.file_IO import parse_FORCE_CONSTANTS
cell, _ = read_crystal_structure("POSCAR", interface_mode="vasp")
ph = Phonopy(cell, supercell_matrix=[int(x) for x in sys.argv[2].split()],
             primitive_matrix='P')
ph.force_constants = parse_FORCE_CONSTANTS("FORCE_CONSTANTS")
worst = 0.0
for line in open("qref.txt"):
    v = [float(x) for x in line.split()]
    q, ref = v[:3], np.array(sorted(v[3:]))
    ph.run_qpoints([q], with_dynamical_matrices=True)
    dm = ph.get_qpoints_dict()["dynamical_matrices"][0]
    ev = np.sort(np.linalg.eigvalsh(dm).real)
    worst = max(worst, np.max(np.abs(ev - ref)) / max(1e-30, np.max(np.abs(ref))))
print(worst)
"""

@testset "phonopy export" begin
    fcs = _fixture()
    Rs = unique([Tuple(Int.(k[2][2] - k[2][1])) for k in keys(fcs.constants)])
    @test (1, 0, 0) in Rs && (0, 1, 0) in Rs      # property 3, asserted not assumed
    @test length(unique(fcs.crystal.species)) == 2 &&
          fcs.crystal.species != sort(fcs.crystal.species)   # property 2

    mktempdir() do dir
        out = write_phonopy(dir, fcs)
        @test out.dim == (3, 3, 1)
        @test out.n_super == 3 * prod(out.dim)
        @test isfile(out.poscar) && isfile(out.force_constants)

        qs = [[0.0, 0.0, 0.0], [0.13, 0.29, 0.41], [0.5, 0.17, 0.06],
              [0.31, 0.05, 0.44]]
        open(joinpath(dir, "qref.txt"), "w") do io
            for q in qs
                ev = sort(real.(eigvals(dynamical_matrix(fcs, q; masses = MASSES))))
                println(io, join(q, " "), "  ", join(ev, " "))
            end
        end
        script = joinpath(dir, "compare.py")
        write(script, COMPARE)
        err = parse(Float64, strip(read(`$PY $script $dir "3 3 1"`, String)))
        @info "phonopy vs dynamical_matrix: worst relative eigenvalue error" err
        # Machine precision, not a tolerance with slack: the two are the same matrix.
        # A convention error lands three to fourteen orders of magnitude above this.
        @test err < 1e-12
    end
end
