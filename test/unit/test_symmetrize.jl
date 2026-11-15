# `symmetrize`, the distance of a crystal from its space group, the atom map's
# acceptance, and the closure refusal's diagnosis (OPEN-WORK S-5; ported from
# SphericalTensorFC `4bb2929` and adapted). The core suite has no Spglib, so the groups
# are assembled by hand through a test backend that hands fixed operations to
# `_assemble_spacegroup`; the Spglib side (noisy-cell sweep, the looser-`tol` warning)
# lives in the integration tier.
#
# The oracles do not use the code under test: the symmetry of a result is checked by
# applying the operations and searching for the images by brute force, and expected
# positions are worked out by hand for crystals whose symmetric configurations are
# known in closed form.

using SLCE: symmetrize, analyze_symmetry, n_ops, _symmetry_residual,
            _assemble_spacegroup, AbstractSymmetryBackend
using Base.CoreLogging: with_logger, NullLogger

# A backend that returns the operations it was built with.
struct _GivenOps <: AbstractSymmetryBackend
    rotations::Vector{Matrix{Float64}}
    translations::Vector{Vector{Float64}}
    symbol::String
    number::Int
end
SLCE.analyze_symmetry(b::_GivenOps, crystal::Crystal; tol::Real = 1e-5) =
    _assemble_spacegroup(crystal, b.rotations, b.translations, b.symbol, b.number;
                         tol = tol)

# Pm-3m: the 48 signed permutation matrices, no translation.
_pm3m() = _GivenOps([Matrix(W) for W in oh48_matrices()], [zeros(3) for _ = 1:48],
                    "Pm-3m", 221)
# Fm-3m in the conventional cell: Pm-3m times the four face centrings.
function _fm3m()
    centrings = ([0.0, 0, 0], [0.0, 0.5, 0.5], [0.5, 0.0, 0.5], [0.5, 0.5, 0.0])
    rots = Matrix{Float64}[]
    trans = Vector{Float64}[]
    for W in oh48_matrices(), c in centrings
        push!(rots, Matrix(W))
        push!(trans, collect(c))
    end
    return _GivenOps(rots, trans, "Fm-3m", 225)
end
# R-3m in rhombohedral axes: the six permutations of the axes, with and without −1.
function _r3m()
    rots = Matrix{Float64}[]
    for p in ([1, 2, 3], [2, 3, 1], [3, 1, 2], [1, 3, 2], [3, 2, 1], [2, 1, 3]),
        s in (1.0, -1.0)
        P = zeros(3, 3)
        for i = 1:3
            P[i, p[i]] = s
        end
        push!(rots, P)
    end
    return _GivenOps(rots, [zeros(3) for _ = 1:12], "R-3m", 166)
end

_quiet_symmetrize(crystal; kwargs...) =
    with_logger(() -> symmetrize(crystal; kwargs...), NullLogger())
_moved(c, shifts) = Crystal(c.lattice, c.frac_positions + shifts, c.species,
                            c.species_labels)
# Cartesian noise of amplitude `a` Å per component, as fractional shifts.
_noise(rng, c, a) = c.lattice.vectors \ (a .* (2 .* rand(rng, 3, n_atoms(c)) .- 1))
_reduced(d) = d .- round.(d)

_b2(a = 2.99) = Crystal(Lattice(a * Matrix(1.0I, 3, 3)), [0.0 0.5; 0.0 0.5; 0.0 0.5],
                        [1, 2], ["Fe", "Rh"])
function _rocksalt(a = 5.6)
    fcc = [0.0 0.0 0.5 0.5; 0.0 0.5 0.0 0.5; 0.0 0.5 0.5 0.0]
    return Crystal(Lattice(a * Matrix(1.0I, 3, 3)), hcat(fcc, fcc .+ 0.5),
                   [1, 1, 1, 1, 2, 2, 2, 2], ["Na", "Cl"])
end
# Rhombohedral cell of edge `a` and angle `α`, atoms at ±(x, x, x).
function _rhombo(α = deg2rad(80.0); a = 3.0, x = 0.23)
    c = cos(α)
    s = sin(α)
    cy = (c - c^2) / s
    A = a * [1 c c; 0 s cy; 0 0 sqrt(1 - c^2 - cy^2)]
    return Crystal(Lattice(A), [x -x; x -x; x -x], [1, 1], ["Bi"])
end

# How far `crystal` is from having the operations of `backend` — found without the
# atom map and without the stored translations: for every operation try each atom as
# the image of atom 1 (which fixes the translation), keep the candidate under which the
# atoms land best on atoms of their species; report the worst landing (Å) and the
# largest relative change of the metric.
function _brute_force_asymmetry(crystal, backend)
    A = Matrix(crystal.lattice.vectors)
    x = crystal.frac_positions
    metric = A' * A
    worst_position = 0.0
    worst_metric = 0.0
    for W in backend.rotations
        worst_metric = max(worst_metric,
                           maximum(abs, W' * metric * W - metric) / maximum(abs, metric))
        best = Inf
        for candidate = 1:n_atoms(crystal)
            crystal.species[candidate] == crystal.species[1] || continue
            translation = x[:, candidate] - W * x[:, 1]
            landing = 0.0
            for i = 1:n_atoms(crystal)
                image = W * x[:, i] + translation
                landing = max(landing, minimum(
                    norm(A * _reduced(image - x[:, j])) for j = 1:n_atoms(crystal)
                    if crystal.species[j] == crystal.species[i]))
            end
            best = min(best, landing)
        end
        worst_position = max(worst_position, best)
    end
    return (; position = worst_position, metric = worst_metric)
end

@testset "symmetrize (S-5)" begin
    @testset "a symmetrized crystal has its operations exactly" begin
        rng = MersenneTwister(20261115)
        stretch = Diagonal([1 + 1e-6, 1.0, 1 - 5e-7])
        stretched(c) = Crystal(Lattice(Matrix(stretch * c.lattice.vectors)), c.frac_positions,
                               c.species, c.species_labels)
        cases = ((_moved(_b2(), _noise(rng, _b2(), 3e-6)), _pm3m()),
                 (stretched(_moved(_b2(), _noise(rng, _b2(), 1e-6))), _pm3m()),
                 (_moved(_rocksalt(), _noise(rng, _rocksalt(), 3e-6)), _fm3m()),
                 (_moved(_rhombo(), _noise(rng, _rhombo(), 3e-6)), _r3m()))
        for (crystal, backend) in cases
            before = _brute_force_asymmetry(crystal, backend)
            @test max(before.position, before.metric) > 1e-8     # the case tests something
            symmetric = _quiet_symmetrize(crystal; backend = backend)
            after = _brute_force_asymmetry(symmetric, backend)
            @test after.position < 1e-12
            @test after.metric < 1e-12
            @test symmetric.species == crystal.species
            # the centroid has not moved; the handedness is kept
            moved = _reduced(symmetric.frac_positions - crystal.frac_positions)
            @test maximum(abs, sum(moved; dims = 2)) < 1e-12
            @test det(symmetric.lattice.vectors) * det(crystal.lattice.vectors) > 0
            # done once is done
            again = _quiet_symmetrize(symmetric; backend = backend)
            @test maximum(abs, _reduced(again.frac_positions -
                                        symmetric.frac_positions)) < 1e-13
            @test again.lattice.vectors ≈ symmetric.lattice.vectors rtol = 1e-13
        end
    end

    @testset "symmetrized positions and lattices, by hand" begin
        rng = MersenneTwister(7)
        # B2: both atoms on fixed points of Pm-3m, so the symmetry leaves the positions
        # no freedom. The symmetric configuration with the same centroid is the ideal
        # crystal shifted as a whole by the mean of the noise.
        b2 = _b2()
        noise = _noise(rng, b2, 3e-6)
        symmetric = _quiet_symmetrize(_moved(b2, noise); backend = _pm3m())
        expected = b2.frac_positions .+ sum(noise; dims = 2) ./ 2
        @test maximum(abs, _reduced(symmetric.frac_positions - expected)) < 1e-13

        # Rhombohedral, atoms at ±(x, x, x): one free coordinate. The operations that
        # keep an atom permute its three coordinates, the others exchange the atoms and
        # reverse the sign. Moving atom 1 by (d, 0, 0) therefore has the symmetric part
        # ±(d/6)(1, 1, 1) — x goes to x + d/6 — and the centroid, moved by (d/2, 0, 0),
        # stays there.
        x, d = 0.23, 3e-6
        rh = _rhombo(; x = x)
        symmetric = _quiet_symmetrize(_moved(rh, [d 0.0; 0.0 0.0; 0.0 0.0]);
                                      backend = _r3m(), tol = 1e-4)
        expected = [x + d / 6, x + d / 6, x + d / 6] * [1 -1] .+ [d / 2, 0.0, 0.0]
        @test maximum(abs, _reduced(symmetric.frac_positions - expected)) < 1e-14
        # (d, −d, 0) has no symmetric part: the ideal crystal shifted by half of it.
        symmetric = _quiet_symmetrize(_moved(rh, [d 0.0; -d 0.0; 0.0 0.0]);
                                      backend = _r3m(), tol = 1e-4)
        expected = rh.frac_positions .+ [d / 2, -d / 2, 0.0]
        @test maximum(abs, _reduced(symmetric.frac_positions - expected)) < 1e-14
        # What the symmetry leaves free is kept: another free coordinate, edge and
        # angle are another rhombohedral crystal and come back as they are.
        other = _rhombo(deg2rad(80.0) + 1e-5; a = 3.0 + 2e-5, x = 0.23 + 4e-6)
        symmetric = _quiet_symmetrize(other; backend = _r3m())
        @test maximum(abs, _reduced(symmetric.frac_positions - other.frac_positions)) <
              1e-14
        @test symmetric.lattice.vectors ≈ other.lattice.vectors rtol = 1e-13

        # A cubic lattice that lost its symmetry: diag(1 + ε₁, 1, 1 + ε₃)·a. Averaging
        # the metric over the 48 signed permutations makes it a²·m·I with m the mean of
        # the three squared stretches; the result is oriented like the given lattice,
        # so the two differ by a symmetric stretch (no rotation) of the size of ε.
        ε1, ε3 = 1e-6, -5e-7
        a = 2.99
        given = Crystal(Lattice(Matrix(a * Diagonal([1 + ε1, 1.0, 1 + ε3]))),
                        b2.frac_positions, b2.species, b2.species_labels)
        symmetric = _quiet_symmetrize(given; backend = _pm3m())
        metric = symmetric.lattice.vectors' * symmetric.lattice.vectors
        m = a^2 * ((1 + ε1)^2 + 1 + (1 + ε3)^2) / 3
        @test metric ≈ m * I atol = 1e-13
        S = symmetric.lattice.vectors / given.lattice.vectors
        @test maximum(abs, S - S') < 1e-13
        @test 1e-7 < maximum(abs, S - I) < 2 * ε1
    end

    @testset "what symmetrize leaves alone, and what it reports" begin
        for (crystal, backend) in ((_b2(), _pm3m()), (_rocksalt(), _fm3m()),
                                   (_rhombo(), _r3m()))
            symmetric = _quiet_symmetrize(crystal; backend = backend)
            @test maximum(abs, _reduced(symmetric.frac_positions -
                                        crystal.frac_positions)) < 1e-14
            @test symmetric.lattice.vectors ≈ crystal.lattice.vectors rtol = 1e-14
        end
        noisy = _moved(_b2(), _noise(MersenneTwister(3), _b2(), 3e-6))
        @test _quiet_symmetrize(noisy; backend = NoSymmetry()).frac_positions ==
              noisy.frac_positions
        @test _quiet_symmetrize(noisy; backend = _pm3m()).frac_positions !=
              noisy.frac_positions
        slab = Crystal(Lattice(Matrix(_b2().lattice.vectors); pbc = (true, true, false)),
                       noisy.frac_positions, noisy.species, noisy.species_labels)
        @test _quiet_symmetrize(slab; backend = NoSymmetry()).lattice.pbc ==
              slab.lattice.pbc
        # the group that was applied is announced, with the residue it removed
        @test_logs (:info, r"symmetrized onto Pm-3m \(no\. 221, 48 operations\)") begin
            symmetrize(noisy; backend = _pm3m())
        end
    end

    @testset "the distance of a crystal from its symmetry, by hand" begin
        b2 = _b2()
        exact = _symmetry_residual(b2, analyze_symmetry(_pm3m(), b2))
        @test exact.position < 1e-14 && exact.lattice < 1e-14
        # Atom 2 moved along z by d Å: an operation that keeps both atoms and reverses
        # z puts atom 2 at −d instead of +d, a mismatch of 2d, which the refitted
        # translation shares between the two atoms: d each. No operation does worse.
        d = 2e-6
        moved = _moved(b2, [0.0 0.0; 0.0 0.0; 0.0 d / 2.99])
        residual = _symmetry_residual(moved, analyze_symmetry(_pm3m(), moved))
        @test residual.position ≈ d rtol = 1e-6
        @test residual.lattice < 1e-14
        brute = _brute_force_asymmetry(moved, _pm3m())       # fits to one atom: ≤ 2×
        @test residual.position <= brute.position * (1 + 1e-6) <= 2.01 * residual.position
        # a₁ stretched by (1 + ε): the operation exchanging a₁ and a₂ changes g₁₁ by
        # a²((1 + ε)² − 1); the largest metric entry is a²(1 + ε)².
        ε = 2e-6
        st = Crystal(Lattice(Matrix(2.99 * Diagonal([1 + ε, 1.0, 1.0]))), b2.frac_positions,
                     b2.species, b2.species_labels)
        residual = _symmetry_residual(st, analyze_symmetry(_pm3m(), st))
        @test residual.lattice ≈ ((1 + ε)^2 - 1) / (1 + ε)^2 rtol = 1e-6
        @test residual.position < 1e-14
    end

    @testset "atoms are matched to their images by distance in Å, unambiguously" begin
        # Atom 2 of B2 moved along z by d: under the operations reversing z it is 2d from
        # its image. Twice `tol` is accepted (the search decides which operations there
        # are; the map guards against one that does not belong): 1.6 tol passes and
        # 3 tol does not, whatever the size of the cell.
        tol = 1e-5
        for (scale, d, accepted) in ((1.0, 0.8e-5, true), (1.0, 1.5e-5, false),
                                     (100.0, 0.8e-5, true), (100.0, 1.5e-5, false),
                                     (0.01, 0.8e-5, true), (0.01, 1.5e-5, false))
            a = 2.99 * scale
            c = Crystal(Lattice(a * Matrix(1.0I, 3, 3)), [0.0 0.5; 0.0 0.5; 0.0 0.5 + d / a],
                        [1, 2], ["Fe", "Rh"])
            if accepted
                @test n_ops(analyze_symmetry(_pm3m(), c; tol = tol)) == 48
            else
                @test_throws "has no image under operation" analyze_symmetry(_pm3m(), c;
                                                                             tol = tol)
            end
        end
        # The wider band does not widen what counts as resolved: two same-species atoms
        # both within 2·tol of one image make the map ambiguous, and that is refused
        # rather than settled by picking one. Here the atoms are 1.5e-5 Å apart: inside
        # 2·tol = 2e-5 Å at tol = 1e-5 (refused), outside 1e-5 Å at tol = 5e-6 (each
        # image within reach of its own atom only).
        close = Crystal(Lattice(3.0 * Matrix(1.0I, 3, 3)), [0.0 5e-6; 0.0 0.0; 0.0 0.0],
                        [1, 1], ["X"])
        @test_throws "both lie within" _assemble_spacegroup(close, [Matrix(1.0I, 3, 3)],
                                                            [zeros(3)], "P1", 1; tol = 1e-5)
        sg = _assemble_spacegroup(close, [Matrix(1.0I, 3, 3)], [zeros(3)], "P1", 1;
                                  tol = 5e-6)
        @test sg.map_sym[:, 1] == [1, 2]
        # translations are kept as given: an exactly symmetric crystal whose origin is
        # off the symmetry elements by less than `tol` (here 3e-6 Å) has translations
        # smaller than `tol`, and they are not noise
        shift = 1e-6
        shifted = _moved(_b2(), fill(shift, 3, 2))
        backend = _GivenOps(_pm3m().rotations,
                            [shift .* (ones(3) - W * ones(3)) for W in _pm3m().rotations],
                            "Pm-3m", 221)
        sg = analyze_symmetry(backend, shifted)
        x = shifted.frac_positions
        @test maximum(norm(_reduced(op.rotation_frac * x[:, i] + op.translation_frac -
                                    x[:, sg.map_sym[i, g]]))
                      for (g, op) in enumerate(sg.ops), i = 1:2) < 1e-14
        @test _symmetry_residual(shifted, sg).position < 1e-14
    end

    @testset "the closure refusal names its cause, and symmetrize is the remedy" begin
        # B2 with noise: the group is accepted at `tol`, but the eight nearest
        # neighbours of an atom are no longer equally far, so the cluster enumeration
        # loses closure at the default tie band.
        spec = BasisSpec(; nbody = 2, cutoff = 2.7, lmax = [1, 1], soc = false)
        ideal = SLCEBasis(_b2(), spec; backend = _pm3m())
        noisy = _moved(_b2(), _noise(MersenneTwister(5), _b2(), 3e-7))
        @test n_ops(analyze_symmetry(_pm3m(), noisy)) == 48
        message = try
            SLCEBasis(noisy, spec; backend = _pm3m())
            nothing
        catch err
            sprint(showerror, err)
        end
        @test message !== nothing
        @test occursin("not closed under the space group", message)
        @test occursin("only approximately", message)
        @test occursin("symmetrize(crystal", message)
        # the stated residue against the independent brute-force measure, which fits
        # each operation's translation to one atom and so sees between 1× and 2× of it
        stated = parse(Float64, match(r"by up to ([0-9.e+-]+) Å", message)[1])
        brute = _brute_force_asymmetry(noisy, _pm3m()).position
        @test stated <= brute * (1 + 1e-6) <= 2.01 * stated
        # the remedy: the symmetrized crystal builds the ideal crystal's columns
        basis = SLCEBasis(_quiet_symmetrize(noisy; backend = _pm3m()), spec;
                          backend = _pm3m())
        @test n_salcs(basis) == n_salcs(ideal)
        @test basis.salc_basis.keys == ideal.salc_basis.keys
    end
end
