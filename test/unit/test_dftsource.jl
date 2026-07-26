# The code-agnostic DFT data boundary (src/io/dftsource.jl): `SpinDatum`,
# `read_configs`, and `SLCEDataset(basis, data/src)`. The headline gate is the torque
# convention `τ_a = m_a × B_a` (the physical / Landau–Lifshitz torque) — CLAUDE.md
# designates this file as the convention source every DFT adapter must match, so a
# closed-form sign check lives here, next to the definition. A silent sign flip
# would bias every energy+torque co-fit while all self-consistency tests still pass.

using Test
using SLCE
using LinearAlgebra
using Random

# A minimal in-memory source, exercising the `read_configs` extension seam the way a
# real adapter (e.g. SLCETools.VASP.Oszicar) does.
struct _MemSource <: AbstractDFTSource
    data::Vector{TrainingDatum}
end
SLCE.read_configs(s::_MemSource) = s.data

struct _EmptySource <: AbstractDFTSource end   # no read_configs method on purpose

@testset "DFT data boundary (SpinDatum / read_configs)" begin
    @testset "torque convention: τ = m × B, closed form" begin
        # m = m·x̂, B = B·ŷ ⇒ τ = m×B = mB·ẑ (and NOT −mB·ẑ: the old, pre-LL sign)
        moments = reshape([2.0, 0.0, 0.0], 3, 1)
        field = reshape([0.0, 0.5, 0.0], 3, 1)
        d = SpinDatum(-1.0, moments, field)
        @test d.torques[:, 1] ≈ [0.0, 0.0, 1.0]
        @test d.directions[:, 1] ≈ [1.0, 0.0, 0.0]
        @test d.magmoms[1] ≈ 2.0
        @test d.energy == -1.0
        # generic cross-product check on random data
        rng = MersenneTwister(11)
        mo = randn(rng, 3, 4)
        Bf = randn(rng, 3, 4)
        dr = SpinDatum(0.0, mo, Bf)
        for a = 1:4
            @test dr.torques[:, a] ≈ cross(mo[:, a], Bf[:, a]) atol = 1e-14
            @test dr.directions[:, a] ≈ mo[:, a] ./ norm(mo[:, a]) atol = 1e-14
        end
    end

    @testset "zero-moment placeholder: ẑ direction, zero torque" begin
        moments = [1.0 0.0; 0.0 0.0; 0.0 0.0]         # atom 2 is quenched
        field = [0.0 1.0; 1.0 1.0; 0.0 1.0]
        d = SpinDatum(0.0, moments, field)
        @test d.directions[:, 2] ≈ [0.0, 0.0, 1.0]     # the documented ẑ placeholder
        @test d.torques[:, 2] ≈ zeros(3) atol = 1e-15  # m = 0 ⇒ τ = 0 exactly
        @test d.magmoms[2] == 0.0
    end

    @testset "constructor / dataset validation throws" begin
        @test_throws ArgumentError SpinDatum(0.0, zeros(2, 3), zeros(2, 3))   # not 3 × n
        @test_throws ArgumentError SpinDatum(0.0, zeros(3, 2), zeros(3, 3))   # shape mismatch
        @test_throws ArgumentError read_configs(_EmptySource())               # no method
        # dataset-level guards
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        basis = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                       soc = false))
        @test_throws ArgumentError SLCEDataset(basis, TrainingDatum[])         # empty data
        # a zero constraining field derives torque_qualified = false, so with
        # use_torque = true no configuration qualifies — fail loudly
        rng = MersenneTwister(3)
        nofield = [SpinDatum(0.1, randn(rng, 3, 2), zeros(3, 2)) for _ = 1:3]
        @test all(d -> !d.provenance.torque_qualified, nofield)
        @test_throws ArgumentError SLCEDataset(basis, nofield)
        ds = SLCEDataset(basis, nofield; use_torque = false)                   # energy-only OK
        @test !has_torque(ds)
        # the explicit unconstrained-stationarity override admits the τ = 0 rows
        okprov = DatumProvenance(; torque_qualified = true)
        qual = [SpinDatum(0.1, randn(rng, 3, 2), zeros(3, 2); provenance = okprov)
                for _ = 1:3]
        dq = @test_logs (:warn, r"exactly zero") SLCEDataset(basis, qual)
        @test has_torque(dq) && all(iszero, dq.y_T)
    end

    @testset "zero-moment guard: referenced atoms must stay magnetic" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        # Fe + B; B is removed from the basis via lmax = 0 (non-magnetic species),
        # so only the Fe single-ion SALCs reference an atom (no Fe–Fe pair exists:
        # a single Fe atom has no self-pair under MinimumImage).
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 2], ["Fe", "B"])
        basis = SLCEBasis(cr, BasisSpec(cr; nbody = 2, cutoff = 1.5,
                                       lmax = ["Fe" => 2, "B" => 0]))
        # quenched B is fine — the basis never reads it
        m_okay = [2.0 0.0; 0.0 0.0; 0.0 0.0]
        ds = SLCEDataset(basis, [SpinDatum(0.0, m_okay, zeros(3, 2))];
                        use_torque = false)
        @test length(ds) == 1
        # quenched Fe is an error naming the config, atom, and species
        m_bad = [0.0 0.0; 0.0 1.0; 0.0 0.0]
        err = try
            SLCEDataset(basis, [SpinDatum(0.0, m_okay, zeros(3, 2)),
                               SpinDatum(0.0, m_bad, zeros(3, 2))];
                       use_torque = false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("config 2", err.msg) && occursin("(Fe)", err.msg)
        # atom-count mismatch is caught at the same boundary
        @test_throws DimensionMismatch SLCEDataset(
            basis, [SpinDatum(0.0, zeros(3, 3) .+ 1.0, zeros(3, 3))];
            use_torque = false)
    end

    @testset "source → dataset round trip carries directions / energies / torques" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        basis = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                       soc = false))
        rng = MersenneTwister(5)
        data = [SpinDatum(0.1 * k, randn(rng, 3, 2), 0.1 .* randn(rng, 3, 2)) for k = 1:4]
        src = _MemSource(data)
        @test read_configs(src) === data
        ds = SLCEDataset(basis, src)
        @test has_torque(ds)
        @test ds.y_E == [0.1 * k for k = 1:4]
        @test ds.configs == [d.directions for d in data]
        # flattened config-major / atom-major / xyz torque layout
        @test ds.y_T == reduce(vcat, [vec(d.torques) for d in data])
        @test ds.torque_config == repeat(1:4; inner = 6)
    end

    @testset "TrainingDatum construction / validation" begin
        dirs = [0.0 0.0; 0.0 0.0; 1.0 -1.0]                  # Ising ±ẑ is legal input
        d = TrainingDatum(; energy = 0.5, directions = dirs, magmoms = [2.0, 2.0])
        @test d.energy == 0.5 && d.displacements === nothing && d.forces === nothing
        @test d.field === nothing && d.torques === nothing
        # torques derived from a present field via τ = m × B (magmoms · directions)
        B = [0.1 0.0; 0.0 0.2; 0.0 0.0]
        df = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = [2.0, 1.5],
                           field = B)
        @test df.torques[:, 1] ≈ cross(2.0 .* dirs[:, 1], B[:, 1])
        @test df.torques[:, 2] ≈ cross(1.5 .* dirs[:, 2], B[:, 2])
        # explicit torques take precedence (direct-torque codes)
        τ = randn(MersenneTwister(1), 3, 2)
        dt = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = [1.0, 1.0],
                           field = B, torques = τ)
        @test dt.torques == τ
        # u = 0 forces without displacements are a legitimate datum
        dfor = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = [1.0, 1.0],
                             forces = zeros(3, 2))
        @test dfor.forces !== nothing && dfor.displacements === nothing
        # validation: signed magmoms, non-unit directions, shape/finiteness
        @test_throws ArgumentError TrainingDatum(; energy = 0.5, directions = dirs,
                                                 magmoms = [2.0, -2.0])
        @test_throws ArgumentError TrainingDatum(; energy = 0.5,
                                                 directions = 2.0 .* dirs,
                                                 magmoms = [1.0, 1.0])
        @test_throws ArgumentError TrainingDatum(; energy = NaN, directions = dirs,
                                                 magmoms = [1.0, 1.0])
        @test_throws ArgumentError TrainingDatum(; energy = 0.0, directions = dirs,
                                                 magmoms = [1.0, 1.0],
                                                 forces = zeros(3, 3))
        @test_throws ArgumentError TrainingDatum(; energy = 0.0, directions = dirs,
                                                 magmoms = [1.0, 1.0],
                                                 displacements = fill(NaN, 3, 2))
    end

    @testset "provenance derivation and overrides" begin
        m = [2.0 0.0; 0.0 1.5; 0.0 0.0]
        d = SpinDatum(0.0, m, [0.0 0.1; 0.1 0.0; 0.0 0.0])
        @test d.provenance.constrained && d.provenance.torque_qualified
        dz = SpinDatum(0.0, m, zeros(3, 2))
        @test !dz.provenance.constrained && !dz.provenance.torque_qualified
        @test dz.torques !== nothing                        # present (observed) zeros
        d2 = SpinDatum(0.0, m)                              # field-less: nothing at all
        @test d2.field === nothing && d2.torques === nothing
        @test !d2.provenance.torque_qualified
        p = DatumProvenance(; constrained = false, torque_qualified = true,
                            setup_id = "vasp-ncl", soc = true)
        dov = SpinDatum(0.0, m, zeros(3, 2); provenance = p)
        @test dov.provenance.torque_qualified && dov.provenance.setup_id == "vasp-ncl"
        @test DatumProvenance().reference_id === nothing
        # the keyword constructor derives the SAME gate as SpinDatum — the two
        # construction paths of one type must not disagree on torque_qualified
        dirs = [1.0 0.0; 0.0 0.0; 0.0 1.0]
        B = [0.0 0.1; 0.1 0.0; 0.0 0.0]
        kw = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = [2.0, 1.5],
                           field = B)
        @test kw.provenance.torque_qualified && kw.provenance.constrained
        kwz = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = [2.0, 1.5],
                            field = zeros(3, 2))
        @test !kwz.provenance.torque_qualified
        # explicitly passed torques qualify (the caller owns the convention)
        kwt = TrainingDatum(; energy = 0.0, directions = dirs, magmoms = [2.0, 1.5],
                            torques = zeros(3, 2))
        @test kwt.provenance.torque_qualified && !kwt.provenance.constrained
    end

    @testset "crystal_fingerprint" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.0 0.5; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        fp = crystal_fingerprint(cr)
        @test fp isa String && length(fp) == 16
        # deterministic across independent builds
        @test fp == crystal_fingerprint(Crystal(lat, [0.0 0.5; 0.0 0.0; 0.0 0.0],
                                                [1, 1], ["Fe"]))
        # sensitive to geometry, species, labels, and lattice
        @test fp != crystal_fingerprint(Crystal(lat, [0.01 0.5; 0.0 0.0; 0.0 0.0],
                                                [1, 1], ["Fe"]))
        @test fp != crystal_fingerprint(Crystal(lat, [0.0 0.5; 0.0 0.0; 0.0 0.0],
                                                [1, 1], ["Co"]))
        @test fp != crystal_fingerprint(Crystal(Lattice(Matrix(3.1 * I(3))),
                                                [0.0 0.5; 0.0 0.0; 0.0 0.0],
                                                [1, 1], ["Fe"]))
        # label serialization is length-prefixed: no concatenation collisions
        @test crystal_fingerprint(Crystal(lat, [0.0 0.5; 0.0 0.0; 0.0 0.0], [1, 2],
                                          ["Fe", "eO"])) !=
              crystal_fingerprint(Crystal(lat, [0.0 0.5; 0.0 0.0; 0.0 0.0], [1, 2],
                                          ["Fee", "O"]))
        # the fractional wrap boundary folds: mod(-1e-17, 1.0) == 1.0 stores as 1.0,
        # but must fingerprint as 0.0
        crW = Crystal(lat, [-1e-17 0.5; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        @test crW.frac_positions[1, 1] == 1.0               # the hazard is real
        @test crystal_fingerprint(crW) == fp
        # -0.0 in the lattice folds; sub-quantum (< 1e-10) noise folds
        @test crystal_fingerprint(Crystal(Lattice([3.0 0.0 0.0; -0.0 3.0 0.0;
                                                   0.0 0.0 3.0]),
                                          [0.0 0.5; 0.0 0.0; 0.0 0.0], [1, 1],
                                          ["Fe"])) == fp
        @test crystal_fingerprint(Crystal(lat, [3e-11 0.5; 0.0 0.0; 0.0 0.0], [1, 1],
                                          ["Fe"])) == fp
        # accepted limitation, pinned: two values STRADDLING a 1e-10 grid line
        # fingerprint differently (a loud false mismatch, never a silent match)
        fa = crystal_fingerprint(Crystal(lat, [0.15 + 4.9e-11 0.5; 0.0 0.0; 0.0 0.0],
                                         [1, 1], ["Fe"]))
        fb = crystal_fingerprint(Crystal(lat, [0.15 + 5.1e-11 0.5; 0.0 0.0; 0.0 0.0],
                                         [1, 1], ["Fe"]))
        @test fa != fb
    end

    @testset "one dataset = one computational setup" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        basis = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                       soc = false))
        rng = MersenneTwister(9)
        mk() = 2.0 .* (m = randn(rng, 3, 2); m ./ mapslices(norm, m; dims = 1))
        pA = DatumProvenance(; setup_id = "collinear", soc = false)
        pB = DatumProvenance(; setup_id = "ncl-soc", soc = true)
        mixed = [SpinDatum(0.1, mk(); provenance = pA),
                 SpinDatum(0.2, mk(); provenance = pB)]
        err = try
            SLCEDataset(basis, mixed; use_torque = false)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("computational setup", err.msg)
        # same setup_id but contradictory soc flags is a mixture too
        pB2 = DatumProvenance(; setup_id = "collinear", soc = true)
        @test_throws ArgumentError SLCEDataset(
            basis, [SpinDatum(0.1, mk(); provenance = pA),
                    SpinDatum(0.2, mk(); provenance = pB2)]; use_torque = false)
        # uniform (untagged) data pass
        okds = SLCEDataset(basis, [SpinDatum(0.1, mk()), SpinDatum(0.2, mk())];
                          use_torque = false)
        @test length(okds) == 2
        # the invariant survives vcat (the incremental-addition path): two datasets
        # from different setups must not concatenate
        dsA = SLCEDataset(basis, [SpinDatum(0.1, mk(); provenance = pA)
                                  for _ = 1:2]; use_torque = false)
        dsB = SLCEDataset(basis, [SpinDatum(0.2, mk(); provenance = pB)
                                  for _ = 1:2]; use_torque = false)
        @test dsA.provenance.setup_id == "collinear"
        errv = try
            vcat(dsA, dsB)
            nothing
        catch e
            e
        end
        @test errv isa ArgumentError && occursin("setup/reference identity", errv.msg)
        # slicing carries the identity, so slice-then-vcat still enforces it
        @test dsA[1:1].provenance == dsA.provenance
        @test_throws ArgumentError vcat(dsA[1:1], dsB[2:2])
        @test length(vcat(dsA[1:1], dsA[2:2])) == 2         # same setup concatenates
        @test_throws ArgumentError vcat(okds, dsA)          # untagged vs tagged
    end

    @testset "reference pinning: basis-keyed double-counting invariant" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        # displacement-decorated (p ≥ 1) basis via the sector table
        bdisp = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.5),
            Sector(spin = [1, 1], disp = (degree = 1:2,), cutoff = 1.5)]))
        @test SLCE._basis_has_disp(bdisp)
        bspin = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                       soc = false))
        @test !SLCE._basis_has_disp(bspin)
        fp = crystal_fingerprint(cr)
        rng = MersenneTwister(17)
        mk() = 2.0 .* (m = randn(rng, 3, 2); m ./ mapslices(norm, m; dims = 1))
        pin = DatumProvenance(; reference_id = "ref", reference_fingerprint = fp)
        pinned = [SpinDatum(0.1, mk(); provenance = pin) for _ = 1:2]
        unpinned = [SpinDatum(0.1, mk()) for _ = 1:2]
        # p ≥ 1 basis: unpinned data (even spin-only — they assert u = 0) rejected
        err = try
            SLCE._check_reference(bdisp, unpinned)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("reference_id", err.msg)
        # wrong fingerprint rejected with the mismatch message
        bad = DatumProvenance(; reference_id = "ref",
                              reference_fingerprint = "0000000000000000")
        err2 = try
            SLCE._check_reference(bdisp, [SpinDatum(0.1, mk(); provenance = bad)])
            nothing
        catch e
            e
        end
        @test err2 isa ArgumentError && occursin("different crystal", err2.msg)
        # two different reference labels rejected even with matching fingerprints
        pin2 = DatumProvenance(; reference_id = "other", reference_fingerprint = fp)
        @test_throws ArgumentError SLCE._check_reference(
            bdisp, [pinned[1], SpinDatum(0.1, mk(); provenance = pin2)])
        # correctly pinned data pass the reference gate (the dataset then still
        # refuses the p ≥ 1 design build — that path lands with X_F)
        @test SLCE._check_reference(bdisp, pinned) === nothing
        err3 = try
            SLCEDataset(bdisp, pinned; use_torque = false)
            nothing
        catch e
            e
        end
        @test err3 isa ArgumentError && occursin("pure-spin", err3.msg)
        # pure-spin basis: displaced data rejected, u-free data need no pin
        disp = TrainingDatum(; energy = 0.0, directions = [0.0 0.0; 0.0 0.0; 1.0 1.0],
                             magmoms = [2.0, 2.0], displacements = 0.01 .* ones(3, 2))
        @test_throws ArgumentError SLCE._check_reference(bspin, [disp])
        @test SLCE._check_reference(bspin, unpinned) === nothing
        # forces against a pure-spin basis are ignored with a warning
        withf = TrainingDatum(; energy = 0.0, directions = [0.0 0.0; 0.0 0.0; 1.0 1.0],
                              magmoms = [2.0, 2.0], forces = zeros(3, 2))
        @test_logs (:warn, r"pure-spin") SLCE._check_reference(bspin, [withf])
    end
end
