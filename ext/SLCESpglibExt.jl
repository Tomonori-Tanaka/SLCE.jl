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
    _warn_if_looser_tol_finds_more(cell, tol, ds)
    return _assemble_spacegroup(crystal, ds.rotations, ds.translations,
                                ds.international_symbol, ds.spacegroup_number; tol = tol)
end

# A search at too small a `tol` for the residue of a structure returns a smaller group
# without complaint — and a smaller group builds a larger basis, fits equivalent sites
# independently, and is then exactly what `symmetrize` makes the crystal symmetric under.
# Nothing downstream can see it, so repeat the search ten times looser and say so when it
# finds more operations (Magesty's `_warn_if_looser_symprec_raises_symmetry`). A looser
# search that fails or finds the same count is silent; one that finds FEWER is not a
# concern here. The factor is a probe, not a recommendation: the message names both
# groups and leaves the choice to the caller.
function _warn_if_looser_tol_finds_more(cell, tol::Real, ds)
    loose_tol = 10 * tol
    loose = try
        Spglib.get_dataset(cell, loose_tol)
    catch err
        err isa InterruptException && rethrow()
        nothing
    end
    (loose === nothing || loose.n_operations <= ds.n_operations) && return nothing
    @warn "A symmetry search ten times looser finds more operations: " *
          "tol = $(Float64(tol)) Å gives $(ds.international_symbol) " *
          "(no. $(ds.spacegroup_number), $(ds.n_operations) operations), " *
          "tol = $(Float64(loose_tol)) Å gives $(loose.international_symbol) " *
          "(no. $(loose.spacegroup_number), $(loose.n_operations) operations). The " *
          "structure may be a slightly distorted copy of the more symmetric one; " *
          "under the smaller group equivalent sites are fitted independently and the " *
          "basis is larger. If the larger group is the intended one, pass that `tol`, " *
          "or `symmetrize(crystal; backend, tol)` at it and build with the default."
    return nothing
end

end # module SLCESpglibExt
