# Nothing the package decides may depend on how the crystal is turned against the
# Cartesian axes, and a lattice that has its symmetry only to the tolerance must be
# treated as the symmetric lattice it is meant to be (OPEN-WORK S-2, 2026-11-15;
# ported from SphericalTensorFC `311e08f`).
#
# Both were false. Measured on `eea5c16` over 455 turned crystals (seven bases incl.
# spin × displacement columns with and without SOC, five axes, thirteen angles from
# 1e-12 to 1 rad): 72 returned a wrong sum-rule rank without a word and 157 were
# refused. Four causes, each with a gate here:
#   1. constraint rows normalized one by one (`_asr_nullspace`);
#   2. absolute cuts on tensor entries and on cancellation (`_zero_roundoff!`,
#      `_ASR_CANCELLATION_RTOL`);
#   3. a Gram–Schmidt pivot of 1e-8 that amplified round-off by 1/θ
#      (`_GAUGE_PIVOT_MIN`);
#   4. Cartesian rotations `A·W·A⁻¹` of a lattice symmetric only to `tol`, which are
#      not orthogonal (`_idealized_cartesian_rotations`).
#
# The groups are assembled by hand (the core suite carries no Spglib). Turning a
# crystal leaves its fractional operations unchanged, so the SAME integer matrices and
# translations describe every orientation; only the lattice moves.

using SLCE: _assemble_spacegroup, build_asr, _idealized_cartesian_rotations,
            _symmetric_lattice, symmetrized_lattice
using Base.CoreLogging: with_logger, NullLogger
using LinearAlgebra
using Random
using StaticArrays

_rodrigues(axis, θ) = (k = normalize(axis); K = [0 -k[3] k[2]; k[3] 0 -k[1]; -k[2] k[1] 0];
                       I + sin(θ) * K + (1 - cos(θ)) * K * K)

# A supercell `n` (isotropic) of the cubic lattice `A0` with `basis` atoms, and its space
# group, found by brute force: the Cartesian point operations `R` of the cubic lattice
# give the integer matrices `W = A0⁻¹ R A0`, and for each `W` every translation
# `t = x_b − W·x_1` that maps the whole atom set onto itself is kept. That covers the
# nonsymmorphic case (diamond: the operations outside T_d carry (1/4, 1/4, 1/4)).
function _translate_group_crystal(A0, basis, n, species_labels)
    pos = Vector{Float64}[]
    for i = 0:(n[1] - 1), j = 0:(n[2] - 1), k = 0:(n[3] - 1), b in axes(basis, 2)
        push!(pos, (basis[:, b] .+ [i, j, k]) ./ collect(n))
    end
    A = A0 * Diagonal(collect(n))
    on_atom(x) = any(p -> maximum(abs.(mod.(x .- p .+ 0.5, 1.0) .- 0.5)) < 1e-9, pos)
    rots, trans = SMatrix{3,3,Float64,9}[], SVector{3,Float64}[]
    for R in oh48_matrices()
        W = SMatrix{3,3,Float64}(round.(inv(A0) * R * A0))
        for p in pos
            t = mod.(p .- W * pos[1], 1.0)
            all(q -> on_atom(W * q .+ t), pos) || continue
            push!(rots, W)
            push!(trans, SVector{3,Float64}(t))
        end
    end
    return A, reduce(hcat, pos), rots, trans
end

# A backend that hands back fixed fractional operations, so the production
# `SLCEBasis` path (and `_assemble_spacegroup`'s validation and idealization) runs on
# them exactly as it does on Spglib's.
struct _OrientationOps <: AbstractSymmetryBackend
    rots::Vector{SMatrix{3,3,Float64,9}}
    trans::Vector{SVector{3,Float64}}
end
SLCE.analyze_symmetry(b::_OrientationOps, cr::Crystal; tol::Real = 1e-5) =
    _assemble_spacegroup(cr, b.rots, b.trans, "manual", 0; tol = tol)

_oriented_basis(A, pos, rots, trans, spec_of; tol = 1e-8) =
    (cr = Crystal(Lattice(A), pos, ones(Int, size(pos, 2)), ["X"]);
     SLCEBasis(cr, spec_of(cr); backend = _OrientationOps(rots, trans), tol = tol))

_quiet_asr(b) = with_logger(() -> build_asr(b; warn = false), NullLogger())

# Three axes and nine angles: the ones where each cause failed on `eea5c16` (cause 2
# at 1e-10, cause 3 between 1e-8 and 1e-3 about x and [110], cause 1 everywhere
# between), plus a large angle. The full grid (5 axes × 13 angles × 7 bases, Spglib)
# was run by hand on 2026-11-15: 455 of 455 correct after the fix.
const _ORI_AXES = ([1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [1.0, 2.0, 3.0])
const _ORI_ANGLES = (1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 0.3)

@testset "the sum rule does not depend on the orientation of the crystal" begin
    # The rank of the translation constraint is a property of the crystal: turning it
    # turns the displacements with it and changes nothing else. So the rank of the
    # crystal along the axes is the answer for every orientation — an oracle that
    # needs no second implementation. Small angles are the point: a crystal turned by
    # θ has tensor entries and constraint rows of size θ and θ², and each zero
    # decision along the way was once wrong for some range of θ.
    disp_only(cut) = cr -> BasisSpec(cr; lmax = 0, pmax = 3,
        sectors = [Sector(disp = (degree = 1:3,), sites = 1:2, cutoff = cut)])
    joint(soc) = cr -> BasisSpec(cr; lmax = 2, pmax = 2,
        sectors = [Sector(spin = (sites = 1:2,), cutoff = 0.9),
                   Sector(spin = [1, 1], disp = (degree = 1:2,), sites = 2, cutoff = 0.9,
                          soc = soc),
                   Sector(disp = (degree = 1:2,), sites = 1:2, cutoff = 0.9)])
    fcc = [0 0.5 0.5; 0.5 0 0.5; 0.5 0.5 0]
    bcc = [0.0 0.5; 0.0 0.5; 0.0 0.5]
    # (name, A0, basis, supercell, spec, n_columns, rank along the axes). The columns
    # and ranks are those of the aligned crystal, which turned crystals must
    # reproduce. The joint SOC row is the one cause 3 alone broke (refused for θ in
    # 1e-8…1e-5 about x and [110]); diamond is the one the sum rule's cancellation cut
    # broke alone (`_ASR_CANCELLATION_RTOL` at 1e-10: refused for θ in 1e-6…1e-4 about
    # [110] and [123]) — the cubic Bravais rows do not see that cut.
    cases = (("sc 3x3x3", Matrix(1.0I, 3, 3), zeros(3, 1), (3, 3, 3), disp_only(1.05), 6,
              2),
             ("fcc 3x3x3", fcc, zeros(3, 1), (3, 3, 3), disp_only(0.75), 9, 3),
             ("diamond 2x2x2", fcc, [0.0 0.25; 0.0 0.25; 0.0 0.25], (2, 2, 2),
              disp_only(0.45), 8, 3),
             ("bcc 2x2x2 spin×disp soc", Matrix(1.0I, 3, 3), bcc, (2, 2, 2), joint(true),
              34, 14),
             ("bcc 2x2x2 spin×disp no soc", Matrix(1.0I, 3, 3), bcc, (2, 2, 2),
              joint(false), 14, 3))
    for (name, A0, basis, n, spec_of, n_columns, rank_along_axes) in cases
        A, pos, rots, trans = _translate_group_crystal(A0, basis, n, ["X"])
        aligned = _oriented_basis(A, pos, rots, trans, spec_of)
        @test n_salcs(aligned) == n_columns
        @test _quiet_asr(aligned).rank == rank_along_axes
        wrong = String[]
        for axis in _ORI_AXES, θ in _ORI_ANGLES
            result = try
                b = _oriented_basis(_rodrigues(axis, θ) * A, pos, rots, trans, spec_of)
                rep = _quiet_asr(b)
                (n_salcs(b), rep.rank, length(rep.free))
            catch err
                err isa ArgumentError || rethrow()
                (-1, -1, -1)
            end
            result == (n_columns, rank_along_axes, n_columns) ||
                push!(wrong, "$name axis=$axis θ=$θ → $result")
        end
        isempty(wrong) || @info "orientation-dependent sum rule" wrong
        @test isempty(wrong)
    end
end

@testset "rotations of the symmetrized lattice" begin
    rng = MersenneTwister(81)
    # A non-orthogonal cell, so A·W·A⁻¹ carries round-off of its own: the fcc
    # primitive cell, turned to a generic orientation.
    fcc = [0 0.5 0.5; 0.5 0 0.5; 0.5 0.5 0]
    A = _rodrigues([0.3, -0.7, 0.6], 0.4) * (3.6 .* fcc)
    Ws = [SMatrix{3,3,Float64}(round.(inv(fcc) * R * fcc)) for R in oh48_matrices()]
    integers(W) = round.(Int, W)
    index_of = Dict(integers(W) => i for (i, W) in enumerate(Ws))
    raw(L) = [L * W * inv(L) for W in Ws]
    # An exactly symmetric lattice is its own symmetrized lattice, up to round-off.
    @test maximum(abs, _symmetric_lattice(A, Ws) - A) <= 1e-13 * maximum(abs, A)
    ideal = _idealized_cartesian_rotations(A, Ws)
    @test maximum(maximum(abs, R - Q) for (R, Q) in zip(ideal, raw(A))) <= 1e-13
    # A lattice sheared by 2e-6 (inside a search tolerance) keeps the same integer
    # operations, but A·W·A⁻¹ is no longer a rotation. The idealized matrices are
    # orthogonal, multiply exactly as the integer matrices do, and are the rotations
    # of the unsheared lattice up to the square of the shear.
    shear = Symmetric(randn(rng, 3, 3))
    sheared = (I + 2e-6 * shear / opnorm(shear)) * A
    @test maximum(maximum(abs, R' * R - I) for R in raw(sheared)) > 1e-7
    ideal_s = _idealized_cartesian_rotations(sheared, Ws)
    @test maximum(maximum(abs, R' * R - I) for R in ideal_s) <= 1e-13
    @test maximum(maximum(abs, ideal_s[a] * ideal_s[b] -
                               ideal_s[index_of[integers(Ws[a] * Ws[b])]])
                  for a in eachindex(Ws), b in eachindex(Ws)) <= 1e-13
    @test maximum(maximum(abs, R - Q) for (R, Q) in zip(ideal_s, ideal)) <= 1e-10
end

@testset "operations that are not symmetries of the lattice are refused" begin
    # Averaging the metric over ANY set of integer matrices makes the idealized
    # rotations orthogonal, so whether the operations are symmetries of THIS lattice
    # must be checked before, on A·W·A⁻¹ of the lattice as given. The cubic point
    # operations offered for a tetragonal lattice (c/a = 1.3): the ones that exchange
    # z with x or y do not preserve its metric. A check moved after the idealization
    # would let them through.
    tet = Crystal(Lattice(Diagonal([3.0, 3.0, 3.9])), zeros(3, 1), [1], ["X"])
    cubic_ops = oh48_matrices()
    zeros_t = [SVector{3,Float64}(0, 0, 0) for _ in cubic_ops]
    @test_throws r"not orthogonal" _assemble_spacegroup(tet, cubic_ops, zeros_t, "wrong",
                                                       0; tol = 1e-5)
    # its own operations (those that keep z) go through
    own = [W for W in cubic_ops if abs(W[3, 3]) == 1]
    @test n_ops(_assemble_spacegroup(tet, own, zeros_t[1:length(own)], "P4/mmm", 123;
                                     tol = 1e-5)) == 16
end

@testset "a lattice symmetric only to the tolerance gives the symmetric basis" begin
    # bcc 2x2x2 with its lattice sheared by 5e-7 — the residue of a relaxed cell,
    # inside the tolerance of the symmetry search. Before the rotations were
    # idealized this was refused inside `wignerD_real` ("sampling is degenerate", the
    # wrong cause). Now the basis is the one of the unsheared crystal: same keys,
    # same values on a common configuration to the size of the shear, same rank.
    rng = MersenneTwister(82)
    spec_of = cr -> BasisSpec(cr; lmax = 2, pmax = 2,
        sectors = [Sector(spin = (sites = 1:2,), cutoff = 0.9),
                   Sector(spin = [2, 2], disp = (degree = 1:2,), sites = 2, cutoff = 0.9),
                   Sector(disp = (degree = 1:2,), sites = 1:2, cutoff = 0.9)])
    A, pos, rots, trans = _translate_group_crystal(2.87 .* Matrix(1.0I, 3, 3),
                                                   [0.0 0.5; 0.0 0.5; 0.0 0.5],
                                                   (2, 2, 2), ["Fe"])
    shear = Symmetric(randn(rng, 3, 3))
    sheared = (I + 5e-7 * shear / opnorm(shear)) * A
    exact = _oriented_basis(A, pos, rots, trans, spec_of; tol = 1e-5)
    noisy = @test_logs((:info, r"only to the search tolerance"), match_mode = :any,
                       _oriented_basis(sheared, pos, rots, trans, spec_of; tol = 1e-5))
    @test noisy.salc_basis.keys == exact.salc_basis.keys
    nat = size(pos, 2)
    e = randcfg(rng, nat)
    u = 0.1 .* randn(rng, 3, nat)
    vals(b) = [evaluate_salc(s, e, u) for s in salcs(b)]
    @test maximum(abs, vals(noisy) - vals(exact)) <= 1e-8 * maximum(abs, vals(exact))
    @test _quiet_asr(noisy).rank == _quiet_asr(exact).rank
    # What the user is told: the lattice given and the one the rotations come from.
    s = symmetrized_lattice(noisy)
    @test s.input_lattice == sheared
    @test 1e-8 < s.metric_change < 1e-5
    @test s.lattice_shift <= 5e-7 * maximum(norm, eachcol(sheared)) * 2
    # the symmetrized metric is invariant under every operation, exactly in the
    # integer matrices' arithmetic up to round-off
    G = s.symmetric_lattice' * s.symmetric_lattice
    @test maximum(maximum(abs, W' * G * W - G) for W in unique(rots)) <=
          1e-13 * maximum(abs, G)
    # an exact lattice reports nothing to report
    @test symmetrized_lattice(exact).metric_change <= 1e-14
end

@testset "SLCEBasis says which lattice the rotations come from" begin
    # An hcp lattice typed to six decimals: symmetric to 1e-7, not exactly. The
    # report is made before the build (so it is on screen if the build is then
    # refused for ties split by the same asymmetry); an exact lattice is silent.
    A = [2.5 -1.25 0.0; 0.0 2.5 * sqrt(3) / 2 0.0; 0.0 0.0 4.07]
    # the 12 point operations of the hexagonal lattice that keep z (D6), fractional
    six = SMatrix{3,3,Float64}([1 -1 0; 1 0 0; 0 0 1])
    two = SMatrix{3,3,Float64}([0 1 0; 1 0 0; 0 0 -1])
    D6 = [SMatrix{3,3,Float64}(six^k * two^m) for k = 0:5 for m = 0:1]
    spec_of = cr -> BasisSpec(cr; lmax = 0, pmax = 2,
        sectors = [Sector(disp = (degree = 2,), sites = 1, cutoff = 1.0)])
    mk(L) = (cr = Crystal(Lattice(L), zeros(3, 1), [1], ["Co"]);
             SLCEBasis(cr, spec_of(cr); backend = _OrientationOps(
                 D6, [SVector{3,Float64}(0, 0, 0) for _ in D6])))
    @test_logs mk(A)                                    # exact: silent
    @test_logs (:info, r"symmetric under the 12 operations.*only.*to the search " *
                       r"tolerance"s) mk(round.(A; digits = 6))
end
