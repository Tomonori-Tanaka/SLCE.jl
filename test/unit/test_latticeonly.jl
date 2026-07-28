# The lattice-only entry path: a user who wants force constants and nothing magnetic.
#
# Every requirement this package places on such a user has to be one they can actually
# meet. The package's centre of gravity is a spin–lattice expansion, and four of its
# entry points used to demand a magnetic state from callers who have none — so the
# only way through was to invent one, which is the failure this file exists to keep
# closed: a fabricated `+ẑ` ferromagnet is indistinguishable from a real one once it
# reaches a basis that reads spins.
#
# Gated here: `_basis_has_spin` (the predicate all of it turns on, and the one that
# must never be confused with `is_soc_free`), `LatticeDatum`, the `use_torque`
# resolution, `spins`/`e` omission in `force_constants` / `predict_*`, the
# provenance-vs-torque_qualified collision, and the structurally-dead-SALC diagnostic.

using Test
using SLCE
using SLCE: _basis_has_spin, _basis_has_disp, _identically_zero_salcs, is_soc_free,
            _assemble_spacegroup, build_neighbor_list, build_clusters,
            build_salc_basis, _superset_cutoff, salcs
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

# bcc Fe, conventional 2-atom cell, with Im-3m assembled by hand: the core test env
# carries no Spglib, and O_h is exactly the 48 signed permutation matrices, so the 96
# operations are a two-line generation that `_assemble_spacegroup` then validates.
function _bcc_im3m()
    cr = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5],
                 [1, 1], ["Fe"])
    pts = SMatrix{3,3,Float64}[]
    for p in ([1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]),
        s in Iterators.product(-1:2:1, -1:2:1, -1:2:1)

        M = zeros(3, 3)
        for i = 1:3
            M[i, p[i]] = s[i]
        end
        push!(pts, SMatrix{3,3,Float64}(M))
    end
    trs = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0.5, 0.5, 0.5)]
    sg = _assemble_spacegroup(cr, [P for P in pts for _ in trs],
                              [t for _ in pts for t in trs], "Im-3m", 229; tol = 1e-8)
    return cr, sg
end

function _lo_basis(cr, sg, spec)
    nl = build_neighbor_list(cr, _superset_cutoff(spec), MinimumImage())
    cl = build_clusters(cr, nl, sg; nbody = spec.nbody, selection = MinimumImage(),
                        cutoff = spec.cutoff)
    sb = build_salc_basis(cr, sg, cl, spec; neighbors = nl, selection = MinimumImage())
    return SLCEBasis(cr, sg, sb, spec)
end

# Minimal reader for phonopy's full-form FORCE_CONSTANTS, so the export is checked
# against a parse rather than against the writer's own buffer.
function _read_force_constants(path::AbstractString)
    toks = split(read(path, String))
    n1 = parse(Int, toks[1])
    n2 = parse(Int, toks[2])
    n1 == n2 || error("expected the square (full) form; got $n1 × $n2")
    Φ = zeros(Float64, 3n1, 3n1)
    k = 3
    for _ = 1:(n1 * n2)
        i = parse(Int, toks[k])
        j = parse(Int, toks[k + 1])
        k += 2
        for α = 1:3, β = 1:3
            Φ[3(i - 1) + α, 3(j - 1) + β] = parse(Float64, toks[k])
            k += 1
        end
    end
    return Φ, n1
end

@testset "the lattice-only entry path" begin
    cr, sg = _bcc_im3m()
    nat = 2
    lattice_only = _lo_basis(cr, sg, BasisSpec(cr; lmax = 0, pmax = 2,
        sectors = [Sector(disp = (degree = 2,), cutoff = 2.7)]))
    joint = _lo_basis(cr, sg, BasisSpec(cr; lmax = 2, pmax = 2, sectors = [
        Sector(spin = (nbody = 1:2, lmax = 2), cutoff = 2.7),
        Sector(disp = (degree = 2,), cutoff = 2.7),
        Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 2.7)]))

    @testset "_basis_has_spin is not is_soc_free" begin
        @test !_basis_has_spin(lattice_only)
        @test _basis_has_spin(joint)
        @test _basis_has_disp(lattice_only) && _basis_has_disp(joint)
        # The trap this predicate exists to avoid. `is_soc_free` asks whether a
        # label's total spin rank vanishes, which most ordinary `soc = false` spin
        # labels satisfy — so a "has no spin" test built on it would wave a
        # spin-carrying model through and evaluate it at a fabricated state. Assert
        # the two really do disagree here, or the distinction is untested folklore.
        spinful = _lo_basis(cr, sg, BasisSpec(cr; lmax = 2, sectors =
            [Sector(spin = (nbody = 1:2, lmax = 2), soc = false, cutoff = 2.7)]))
        @test _basis_has_spin(spinful)
        @test all(s -> is_soc_free(s.key.L_S), salcs(spinful))   # ...yet all soc-free
        # and the spec, not just the surviving SALCs, decides: a spin sector that
        # symmetry annihilates still means the data were generated magnetically
        @test _basis_has_spin(_lo_basis(cr, sg, BasisSpec(cr; lmax = 2, pmax = 2,
            sectors = [Sector(spin = (nbody = 1, lmax = 1), cutoff = 2.7),
                       Sector(disp = (degree = 2,), cutoff = 2.7)])))
    end

    @testset "LatticeDatum: no magnetic state to invent" begin
        rng = MersenneTwister(0x10)
        # zero on the identically-zero SALC (index 3, see the last testset): a
        # coefficient no configuration can express is not recoverable by construction
        truth = SLCEModel(lattice_only, 0.0, [0.01, -0.02, 0.0])
        us = [0.05 .* randn(rng, 3, nat) for _ = 1:40]
        # the whole chain, with no spin argument anywhere
        data = [LatticeDatum(predict_energy(truth, nothing, u); displacements = u,
                             forces = predict_force(truth, nothing, u), reference = cr)
                for u in us]
        d = data[1]
        @test d.directions == repeat([0.0, 0.0, 1.0], 1, nat)   # the inert placeholder
        @test all(iszero, d.magmoms)                            # EXACTLY zero: see below
        @test d.provenance.reference_fingerprint == crystal_fingerprint(cr)
        @test size(d.forces) == (3, nat)
        # `use_torque` resolves from the basis: a lattice-only expansion has no torque
        # block to build, so demanding torque data for it is unsatisfiable by design
        ds = SLCEDataset(lattice_only, data)
        @test !has_torque(ds) && has_force(ds)
        f = fit(SLCEFit, ds, OLS(); force_weight = 0.5, asr = false)
        @test maximum(abs, f.jphi .- truth.jphi) < 1e-9

        # The load-bearing half of the placeholder: feed the same data to a basis that
        # DOES read spins and the zero-moment invariant rejects it by name. The
        # placeholder can never be mistaken for a ferromagnet, which is why filling it
        # in silently is safe here and would not be with a plausible-looking moment.
        @test_throws ArgumentError SLCEDataset(joint, data)

        # atom count inference, and the one case that cannot be inferred
        @test size(LatticeDatum(1.0; reference = cr).directions, 2) == nat
        @test size(LatticeDatum(1.0; n_atoms = 5).directions, 2) == 5
        @test size(LatticeDatum(1.0; forces = zeros(3, 3)).directions, 2) == 3
        @test_throws ArgumentError LatticeDatum(1.0)
        @test_throws ArgumentError LatticeDatum(1.0; n_atoms = 0)
    end

    @testset "`spins` / `e` may be omitted only when there is none" begin
        rng = MersenneTwister(0x11)
        ml = SLCEModel(lattice_only, 0.0, randn(rng, n_salcs(lattice_only)))
        mj = SLCEModel(joint, 0.0, randn(rng, n_salcs(joint)))
        u = 0.05 .* randn(rng, 3, nat)
        ez = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])

        # omission is exactly equivalent to passing the all-zero marker, and the spin
        # state genuinely cannot matter: two different states give the same numbers
        @test force_constants(ml; order = 2).constants ==
              force_constants(ml; spins = zeros(3, nat), order = 2).constants
        @test predict_energy(ml, nothing, u) == predict_energy(ml, ez, u)
        @test predict_force(ml, nothing, u) == predict_force(ml, ez, u)
        # a lattice-only model takes a non-unit spin matrix without complaint (nothing
        # reads it) but still checks the shape
        @test predict_energy(ml, zeros(3, nat), u) == predict_energy(ml, nothing, u)
        @test_throws DimensionMismatch predict_energy(ml, zeros(3, nat + 1), u)

        # against a spin-carrying basis omission is an error, never a default state
        @test_throws ArgumentError force_constants(mj; order = 2)
        @test_throws ArgumentError predict_energy(mj, nothing, u)
        @test_throws ArgumentError predict_force(mj, nothing, u)
        # ...and the model that DOES depend on spins proves the refusal is not
        # pedantry: the two states disagree
        @test predict_energy(mj, ez, u) != predict_energy(mj, -ez, u) ||
              predict_energy(mj, ez, u) != predict_energy(mj, circshift(ez, (1, 0)), u)
    end

    @testset "a hand-built provenance no longer suppresses the torque channel" begin
        # The collision: the displacement channel REQUIRES a provenance carrying the
        # reference identity, and building one used to discard the automatic
        # `torque_qualified` that explicit torques earn — so a datum with
        # displacements AND torques failed the dataset build with "pass
        # use_torque = false", the exact opposite of the caller's intent.
        fp = crystal_fingerprint(cr)
        dirs = reduce(hcat, [normalize(randn(MersenneTwister(3), 3)) for _ = 1:nat])
        ref = DatumProvenance(; reference_id = "r", reference_fingerprint = fp)
        @test !ref.torque_qualified                       # the struct default
        d = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = ones(nat),
                          displacements = zeros(3, nat), torques = ones(3, nat),
                          provenance = ref)
        @test d.provenance.torque_qualified               # upgraded
        @test d.provenance.reference_fingerprint == fp    # and the stamp survives
        # a nonzero field earns it the same way, on both construction paths
        B = [0.0 0.1; 0.1 0.0; 0.0 0.0]
        @test TrainingDatum(; energy = 0.0, directions = dirs, magmoms = ones(nat),
                            field = B, provenance = ref).provenance.torque_qualified
        @test SpinDatum(0.0, [2.0 0.0; 0.0 1.5; 0.0 0.0], B;
                        provenance = ref).provenance.torque_qualified
        # upgrade-only: an explicit `true` with no torque evidence is never revoked
        yes = DatumProvenance(; torque_qualified = true, reference_id = "r",
                              reference_fingerprint = fp)
        @test TrainingDatum(; energy = 0.0, directions = dirs, magmoms = ones(nat),
                            displacements = zeros(3, nat),
                            provenance = yes).provenance.torque_qualified
        # and no evidence, no upgrade
        @test !TrainingDatum(; energy = 0.0, directions = dirs, magmoms = ones(nat),
                             displacements = zeros(3, nat),
                             provenance = ref).provenance.torque_qualified
    end

    # A design column can be zero because the DATA say nothing about it (collect more)
    # or because the SALC is zero for EVERY configuration (nothing will ever help).
    # The old single message asserted the first for both, which on this cell is advice
    # that cannot be taken: one of the three lattice-only SALCs has its minimum-image
    # ties cancel identically.
    @testset "identically-zero SALCs are named as such, not blamed on the data" begin
        @test n_salcs(lattice_only) == 3
        @test _identically_zero_salcs(lattice_only, 1:3) == [3]
        # the classifier only ever reports members of what it was handed
        @test isempty(_identically_zero_salcs(lattice_only, [1, 2]))
        # Three probe configurations decide a claim about ALL configurations, so pin
        # the claim against a 200-sample design matrix built through the OTHER code
        # path (`_design_energy`, not `evaluate_salc`): the two index sets agree
        # exactly. On this cell that is 14 of 23 joint SALCs — small cells lose far
        # more of `n_salcs` to minimum-image cancellation than the count suggests.
        mj = n_salcs(joint)
        rp = MersenneTwister(99)
        cf = [reduce(hcat, [normalize(randn(rp, 3)) for _ = 1:nat]) for _ = 1:200]
        uf = [0.07 .* randn(rp, 3, nat) for _ = 1:200]
        Xd = SLCE._design_energy(joint, cf, uf)
        nrm = [norm(@view Xd[:, j]) for j = 1:mj]
        @test findall(<=(1e-12 * maximum(nrm)), nrm) ==
              _identically_zero_salcs(joint, 1:mj)
        # not degenerate in either direction: some columns survive, and the dead set
        # is a proper subset rather than "everything the probe happened not to hit"
        deadj = _identically_zero_salcs(joint, 1:mj)
        @test 0 < length(deadj) < mj

        rng = MersenneTwister(0x12)
        truth = SLCEModel(lattice_only, 0.0, [0.01, -0.02, 0.0])
        data = [begin
                    u = 0.05 .* randn(rng, 3, nat)
                    LatticeDatum(predict_energy(truth, nothing, u); displacements = u,
                                 forces = predict_force(truth, nothing, u),
                                 reference = cr)
                end for _ = 1:40]
        ds = SLCEDataset(lattice_only, data)
        @test_logs((:warn, r"identically zero on this cell"), match_mode = :any,
                   fit(SLCEFit, ds, OLS(); force_weight = 0.5, asr = false))

        # ...and the other branch still fires, on the case the message names: a
        # force-only fit puts zero weight on the energy block, so every pure-spin
        # column of a JOINT design is dead — for want of data, not structurally.
        rng2 = MersenneTwister(0x13)
        tj = SLCEModel(joint, 0.0, randn(rng2, n_salcs(joint)))
        fp = crystal_fingerprint(cr)
        jd = [begin
                  e = reduce(hcat, [normalize(randn(rng2, 3)) for _ = 1:nat])
                  u = 0.05 .* randn(rng2, 3, nat)
                  TrainingDatum(; energy = predict_energy(tj, e, u), directions = e,
                                magmoms = ones(nat), displacements = u,
                                forces = predict_force(tj, e, u),
                                provenance = DatumProvenance(; reference_id = "r",
                                                             reference_fingerprint = fp))
              end for _ = 1:40]
        dsj = SLCEDataset(joint, jd; use_torque = false)
        @test_logs((:warn, r"carry no information"), match_mode = :any,
                   fit(SLCEFit, dsj, OLS(); force_weight = 1.0, asr = false))
    end

    # `write_phonopy`'s real gate — that the supercell ordering agrees with phonopy's —
    # lives in `test/phonopy/`, because only phonopy can settle it: a permuted export
    # still diagonalizes and still has three acoustic modes at Γ. What IS checkable
    # here is everything on this side of that boundary: the tiling, the shift-to-
    # supercell arithmetic, the species permutation, and the refusals.
    @testset "write_phonopy: the half that does not need phonopy" begin
        rng = MersenneTwister(0x20)
        mj = SLCEModel(joint, 0.0, randn(rng, n_salcs(joint)))
        ez = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])
        fcs = force_constants(mj; spins = ez, order = 2)
        shifts = unique([Tuple(Int.(k[2][2] - k[2][1])) for k in keys(fcs.constants)])
        want = ntuple(k -> 2 * maximum(abs(R[k]) for R in shifts) + 1, 3)

        mktempdir() do dir
            out = write_phonopy(dir, fcs)
            @test out.dim == want                      # smallest wraparound-free box
            @test out.n_super == nat * prod(want)
            # The written FORCE_CONSTANTS reproduces `dynamical_matrix` when D(q) is
            # rebuilt from it by the standard supercell sum. This does NOT pin the
            # ordering (any consistent map passes), which is the whole reason the
            # phonopy suite exists — but it does pin the tiling and the wraparound.
            Φ, n = _read_force_constants(out.force_constants)
            @test n == out.n_super
            ncell = prod(want)
            for q in ([0.0, 0.0, 0.0], [0.21, 0.34, 0.11])
                D = zeros(ComplexF64, 3nat, 3nat)
                for a = 1:nat, b = 1:nat, l3 = 0:(want[3] - 1),
                    l2 = 0:(want[2] - 1), l1 = 0:(want[1] - 1)

                    i = (a - 1) * ncell + 1                       # lattice point 0
                    j = (b - 1) * ncell + (l1 + want[1] * (l2 + want[2] * l3)) + 1
                    # the shift a supercell atom sits at, folded to (−n/2, n/2]
                    R = ntuple(k -> (l = (l1, l2, l3)[k];
                                     l > want[k] ÷ 2 ? l - want[k] : l), 3)
                    ph = cis(2π * (q[1] * R[1] + q[2] * R[2] + q[3] * R[3]))
                    for α = 1:3, β = 1:3
                        D[3(a - 1) + α, 3(b - 1) + β] +=
                            Φ[3(i - 1) + α, 3(j - 1) + β] * ph
                    end
                end
                @test maximum(abs, D .- dynamical_matrix(fcs, q)) <
                      1e-10 * max(1.0, norm(D))
            end
        end

        # The POSCAR groups species and the force constants follow it. `joint`'s
        # crystal is single-species, so build one that actually interleaves.
        inter = Crystal(Lattice(Matrix(3.0 * I(3))),
                        [0.0 0.25 0.5; 0.0 0.25 0.5; 0.0 0.25 0.5], [1, 2, 1],
                        ["Fe", "Ni"])
        @test SLCE._species_grouped_perm(inter) == [1, 3, 2]
        mktempdir() do dir
            fake = ForceConstantSet(2, inter, zeros(3, 3),
                Dict(([1, 2], [SVector{3,Int}(0, 0, 0), SVector{3,Int}(0, 0, 0)]) =>
                     ones(3, 3)))
            write_phonopy(dir, fake)
            lines = readlines(joinpath(dir, "POSCAR"))
            @test split(lines[6]) == ["Fe", "Ni"]
            @test split(lines[7]) == ["2", "1"]
        end

        # refusals and the aliasing warning
        @test_throws ArgumentError write_phonopy(mktempdir(),
                                                 force_constants(mj; spins = ez,
                                                                 order = 3))
        @test_throws ArgumentError write_phonopy(mktempdir(), fcs; dim = (0, 1, 1))
        # a dim below the wraparound-free one folds shifts, and says how many
        @test_logs((:warn, r"too small"), match_mode = :any,
                   write_phonopy(mktempdir(), fcs; dim = (1, 1, 1)))
    end
end
