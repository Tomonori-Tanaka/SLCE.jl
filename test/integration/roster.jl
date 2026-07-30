# The roster: named real crystals, each one row of the coverage matrix.
#
# A row exists because of a STRUCTURAL FEATURE that changes which code path fires —
# not to add another cell shape. The features, and the row that carries each:
#
#   * a Wigner-Seitz boundary tie in a conventional cell        bcc-Fe
#   * the same crystal in a cell where the tie is gone          bcc-Fe-2x2x2
#   * two species / two magnetic sublattices, cubic B2          B2-FeRh
#   * a hexagonal (non-orthogonal) lattice, two like sites      hcp-Co
#   * NO inversion centre, four atoms, lattice-only entry       wz-GaN
#   * an all-non-orthogonal primitive cell, and the same-atom
#     pair the minimum-image convention cannot express          rs-MnO
#
# Cell parameters are experimental room-temperature values, rounded; they are
# fixtures, not measurements, and nothing here depends on their accuracy. The
# space-group symbol / number / operation count of each row are stated from the
# International Tables and checked against the backend — that is the census
# column's oracle, and it is external to this package.

using SLCE
using LinearAlgebra: I, Diagonal

# One row of the matrix. `runs` and `skips` together must cover every column in
# `COLUMNS` (the driver refuses a row that leaves one unaccounted for), and every
# skip carries the reason in the file, where a reader will look for it.
struct Row
    id::Symbol
    title::String
    crystal::Crystal
    spec::BasisSpec
    symbol::String                  # international symbol, from the Tables
    number::Int                     # space-group number, from the Tables
    n_ops::Int                      # point ops × lattice points in THIS cell
    runs::Vector{Symbol}
    skips::Dict{Symbol,String}
end

# Tile a crystal `dims` times along its own axes. Fractional coordinates divide;
# species repeat. Used only for the tie-free bcc row.
function tile(cr::Crystal, dims::NTuple{3,Int})
    all(>(0), dims) || throw(ArgumentError("dims must be positive"))
    L = Matrix(cr.lattice.vectors) * Diagonal(collect(Float64, dims))
    sp = Int[]
    cols = Vector{Vector{Float64}}()
    for i = 0:dims[1]-1, j = 0:dims[2]-1, k = 0:dims[3]-1, a = 1:n_atoms(cr)
        f = cr.frac_positions[:, a]
        push!(cols, [(f[1] + i) / dims[1], (f[2] + j) / dims[2], (f[3] + k) / dims[3]])
        push!(sp, cr.species[a])
    end
    return Crystal(Lattice(L), reduce(hcat, cols), sp, cr.species_labels)
end

# ---------------------------------------------------------------------------
# Crystals
# ---------------------------------------------------------------------------

# bcc Fe, conventional cubic cell (a = 2.87 Å): two Fe, 1NN at √3a/2 = 2.485 Å.
const CR_FE = Crystal(Lattice(Matrix(2.87 * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5],
                      [1, 1], ["Fe"])
const CR_FE222 = tile(CR_FE, (2, 2, 2))

# B2 FeRh (a = 2.99 Å): Fe at the corner, Rh at the body centre. 1NN is the
# Fe-Rh bond at 2.589 Å; the 2NN shell at 2.99 Å is Fe-Fe and Rh-Rh.
const CR_FERH = Crystal(Lattice(Matrix(2.99 * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5],
                        [1, 2], ["Fe", "Rh"])

# hcp Co (a = 2.507 Å, c = 4.069 Å): the hexagonal lattice is non-orthogonal, and
# the two Co sites are the same species at inequivalent positions. In-plane and
# out-of-plane 1NN are nearly degenerate (2.507 / 2.497 Å) — both are inside the
# row's cutoff, so the row does not depend on which comes first.
const CR_CO = let a = 2.507, c = 4.069
    Crystal(Lattice([a -a/2 0.0; 0.0 a*sqrt(3)/2 0.0; 0.0 0.0 c]),
            [1/3 2/3; 2/3 1/3; 1/4 3/4], [1, 1], ["Co"])
end

# Wurtzite GaN (a = 3.189 Å, c = 5.185 Å, u = 0.377): four atoms and NO inversion
# centre, which is the row's reason for existing — odd-degree displacement
# invariants are allowed here and forbidden in every centrosymmetric row.
const CR_GAN = let a = 3.189, c = 5.185, u = 0.377
    Crystal(Lattice([a -a/2 0.0; 0.0 a*sqrt(3)/2 0.0; 0.0 0.0 c]),
            [1/3 2/3 1/3 2/3; 2/3 1/3 2/3 1/3; 0.0 0.5 u 0.5+u],
            [1, 1, 2, 2], ["Ga", "N"])
end

# Rocksalt MnO, CONVENTIONAL cubic cell (a = 4.446 Å): four Mn and four O, so
# the Mn-Mn 2NN pair joins two DISTINCT atoms and the model has content. The
# primitive cell below is the same crystal with one cation, where that pair joins
# an atom to an image of itself and the magnetic basis comes out empty — the
# `:selfpair` column builds it there and nowhere else.
const CR_MNO = let a = 4.446
    Crystal(Lattice(Matrix(a * I(3))),
            [0.0 0.5 0.5 0.0 0.5 0.5 0.0 0.0;
             0.0 0.5 0.0 0.5 0.5 0.0 0.5 0.0;
             0.0 0.0 0.5 0.5 0.5 0.0 0.0 0.5],
            [1, 1, 1, 1, 2, 2, 2, 2], ["Mn", "O"])
end

# The primitive cell of the same crystal: all three axes non-orthogonal (60°),
# one Mn and one O. Used only by `:selfpair`.
const CR_MNO_PRIM = let a = 4.446
    Crystal(Lattice([0.0 a/2 a/2; a/2 0.0 a/2; a/2 a/2 0.0]),
            [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 2], ["Mn", "O"])
end

# ---------------------------------------------------------------------------
# Specs
# ---------------------------------------------------------------------------

# The joint spec every magnetic row uses: a pure-spin pair sector plus a
# spin-dressed displacement sector at degree 2 (the shape that makes force
# constants a function of the magnetic state). `pmax = 2` supplies the on-site
# partners the ASR's difference invariants need — without them the whole
# displacement channel is structurally excluded by the sum rule.
joint_spec(cr::Crystal, cutoff::Real; lmax = 1) =
    BasisSpec(cr; lmax = lmax, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = cutoff),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = cutoff)])

# FeRh adds the degree-1 spin-dressed sector: that is the ε-linear content the
# magnetoelastic tier rides on, and the Fe-Rh bond is the case where it survives
# the sum rule — no operation exchanges the bond's two ends, so the partner can
# be the same channel with the displacement on the other atom.
ferh_spec(cr::Crystal, cutoff::Real) =
    BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = cutoff),
        Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = cutoff),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = cutoff)])

# The lattice-only spec: no spin content anywhere, which is also the entry path a
# phonon-only user takes. `degree = 2:3` reaches the cubic anharmonicity whose
# EXISTENCE is the inversion-parity question of the `:parity` column.
lattice_spec(cr::Crystal, cutoff::Real; degree = 2:3) =
    BasisSpec(cr; lmax = 0, pmax = 2,
              sectors = [Sector(disp = (degree = degree,), sites = 2, cutoff = cutoff)])

# ---------------------------------------------------------------------------
# The matrix
# ---------------------------------------------------------------------------

# Every column the tier knows how to run. A row must account for all of them.
const COLUMNS = (:census, :structure, :resolvability, :invariance, :affine, :asr,
                 :recovery, :phonons, :fd_hessian, :effective, :restrict, :persist,
                 :terms, :magnetoelastic, :strain, :selection, :selfpair)

# `bcc-Fe-2x2x2` runs the same spec as `bcc-Fe` on a cell that resolves it; the
# deliverable columns would re-run identical code on identical content, so they
# are delegated rather than duplicated. This is the reason string for that.
const _DELEGATED = "same spec as bcc-Fe on a larger cell — this row exists for the " *
                   "tie contrast (resolvability / recovery / phonons), and the " *
                   "deliverable columns would re-run identical content"

const ROSTER = Row[
    Row(:bcc_Fe, "bcc Fe, conventional 2-atom cell", CR_FE, joint_spec(CR_FE, 2.6),
        "Im-3m", 229, 96,
        [:census, :structure, :resolvability, :invariance, :affine, :asr, :recovery,
         :phonons, :fd_hessian, :effective, :restrict, :persist, :terms, :strain],
        Dict(:magnetoelastic =>
                 "no degree-1 spin-dressed sector: the ε-linear content B₁/B₂ ride " *
                 "on is absent by construction — B2-FeRh carries that sector, and " *
                 "carries both of the states a real cell puts it in",
             :selection =>
                 "the fold machinery is content-blind; bcc-Fe-2x2x2 and hcp-Co run " *
                 "it, at the two extremes of free-parameter count",
             :selfpair =>
                 "the same-atom pair limitation is measured on B2-FeRh (its 2NN " *
                 "shell is exactly that pair) and on rs-MnO (where it empties the " *
                 "magnetic basis outright)")),
    Row(:bcc_Fe_222, "bcc Fe, 2×2×2 of the conventional cell (tie-free)", CR_FE222,
        joint_spec(CR_FE222, 2.6), "Im-3m", 229, 768,
        [:census, :structure, :resolvability, :invariance, :asr, :recovery, :phonons,
         :persist, :selection],
        Dict(:affine => _DELEGATED, :fd_hessian =>
                 "a 48×48 central-difference Hessian is 9216 energy evaluations for " *
                 "a claim bcc-Fe already gates on the same coefficients",
             :effective => _DELEGATED, :restrict => _DELEGATED, :terms => _DELEGATED,
             :strain => _DELEGATED,
             :magnetoelastic => "no degree-1 spin-dressed sector (see bcc-Fe)",
             :selfpair => "measured on B2-FeRh and rs-MnO")),
    Row(:B2_FeRh, "B2 FeRh, two species and two magnetic sublattices", CR_FERH,
        ferh_spec(CR_FERH, 2.7), "Pm-3m", 221, 48,
        [:census, :structure, :resolvability, :invariance, :affine, :asr, :recovery,
         :phonons, :fd_hessian, :effective, :restrict, :persist, :terms,
         :magnetoelastic, :strain, :selfpair],
        Dict(:selection => "run on bcc-Fe-2x2x2 and hcp-Co")),
    Row(:hcp_Co, "hcp Co, hexagonal lattice", CR_CO, joint_spec(CR_CO, 2.6),
        "P6_3/mmc", 194, 24,
        [:census, :structure, :resolvability, :invariance, :affine, :asr, :recovery,
         :phonons, :fd_hessian, :effective, :restrict, :persist, :terms, :strain,
         :selection],
        Dict(:magnetoelastic =>
                 "`magnetoelastic_constants` pins the CUBIC B₁/B₂ convention; the " *
                 "hexagonal constants are a different parameterization this package " *
                 "does not claim",
             :selfpair => "measured on B2-FeRh and rs-MnO")),
    Row(:wz_GaN, "wurtzite GaN, no inversion centre, lattice-only", CR_GAN,
        lattice_spec(CR_GAN, 2.0), "P6_3mc", 186, 12,
        [:census, :structure, :resolvability, :invariance, :affine, :asr, :recovery,
         :phonons, :fd_hessian, :effective, :restrict, :persist, :terms,
         :magnetoelastic, :strain],
        Dict(:selection => "run on bcc-Fe-2x2x2 and hcp-Co",
             :selfpair => "measured on B2-FeRh and rs-MnO")),
    Row(:rs_MnO, "rocksalt MnO, conventional 8-atom cell", CR_MNO,
        lattice_spec(CR_MNO, 3.2; degree = 2), "Fm-3m", 225, 192,
        [:census, :structure, :resolvability, :invariance, :affine, :asr, :recovery,
         :phonons, :fd_hessian, :persist, :selfpair],
        Dict(:effective =>
                 "exercised on the four rows whose displacement content is larger; " *
                 "this row is here for the cation-cation shell and its primitive-cell " *
                 "counterpart",
             :restrict => "no spin content; wz-GaN carries the lattice-only " *
                          "`restrict` contract",
             :terms => "no spin content; the decorated-term reconstruction is " *
                       "exercised on wz-GaN (displacement slots) and on the joint rows",
             :magnetoelastic => "no spin content (see the wz-GaN refusal face)",
             :strain => "wz-GaN carries the lattice-only strain column",
             :selection => "run on bcc-Fe-2x2x2 and hcp-Co")),
]
