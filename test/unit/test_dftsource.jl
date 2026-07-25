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
    data::Vector{SpinDatum}
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
                                       isotropy = true))
        @test_throws ArgumentError SLCEDataset(basis, SpinDatum[])             # empty data
        # all-zero torque targets with use_torque = true must fail loudly
        rng = MersenneTwister(3)
        nofield = [SpinDatum(0.1, randn(rng, 3, 2), zeros(3, 2)) for _ = 1:3]
        @test_throws ArgumentError SLCEDataset(basis, nofield)
        ds = SLCEDataset(basis, nofield; use_torque = false)                   # energy-only OK
        @test !has_torque(ds)
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
                                       isotropy = true))
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
    end
end
