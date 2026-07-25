module SLCESpglibExt

# Provides `analyze_symmetry(::SpglibBackend, …)` when Spglib is loaded. The
# SpglibBackend type itself lives in the core package, so it can be named (and
# dispatched on) without this extension; only the actual space-group analysis
# lights up here.

using SLCE: Crystal, SpglibBackend, _assemble_spacegroup
import SLCE: analyze_symmetry
using Spglib

function analyze_symmetry(::SpglibBackend, crystal::Crystal; tol::Real = 1e-5)
    lattice = Matrix(crystal.lattice.vectors)
    positions = collect(eachcol(crystal.frac_positions))
    cell = Spglib.Cell(lattice, positions, crystal.species)
    ds = Spglib.get_dataset(cell, tol)
    ds === nothing &&
        error("Spglib.get_dataset returned nothing; check the structure or `tol`.")
    return _assemble_spacegroup(crystal, ds.rotations, ds.translations,
                                ds.international_symbol, ds.spacegroup_number; tol = tol)
end

end # module SLCESpglibExt
