# Sunny-export validation: build a Sunny.System from a fitted SLCEModel and confirm
# its classical energy reproduces the SLCE energy. Runs in a separate environment
# (heavy Sunny dependency), mirroring `test/oracle/`. The conversion math itself is
# already covered Sunny-free in `test/unit/test_sunny.jl`; here we check the actual
# Sunny.System assembly end-to-end.

using Test
using Random
using LinearAlgebra
using StaticArrays
import SLCE
import Sunny   # triggers SLCESunnyExt
import Spglib  # triggers SLCESpglibExt (real space groups for the unfold)

const MR = SLCE

_rdir(rng) = (v = randn(rng, 3); v / norm(v))
_rcfg(rng, n) = reshape(reduce(vcat, (_rdir(rng) for _ = 1:n)), 3, n)

# Max |Sunny energy − (predict_energy − j0)| over random configurations.
function energy_error(model, S, mode; seed = 5, ntrial = 20, placement = :auto)
    sys = MR.to_sunny(model; spin_length = S, mode = mode, placement = placement)
    nat = MR.n_atoms(model.basis.crystal)
    rng = MersenneTwister(seed)
    me = 0.0
    for _ = 1:ntrial
        e = _rcfg(rng, nat)
        for i = 1:nat
            Sunny.set_dipole!(sys, e[:, i], (1, 1, 1, i))
        end
        me = max(me, abs(Sunny.energy(sys) - (MR.predict_energy(model, e) - model.j0)))
    end
    return me
end

@testset "Sunny export vs SLCE" begin
    lat = MR.Lattice(Matrix(3.0 * I(3)))
    cr2 = MR.Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    rng = MersenneTwister(1)

    @testset "bilinear exchange: energy matches across modes and spins" begin
        b = MR.SLCEBasis(cr2, MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                            soc = true))
        model = MR.SLCEModel(b, 0.5, randn(rng, MR.n_salcs(b)), b.salc_basis.keys)
        @test energy_error(model, 1.5, :dipole_uncorrected) < 1e-10
        @test energy_error(model, 1.0, :dipole) < 1e-10            # exchange is mode-independent
        @test energy_error(model, 2.5, :dipole) < 1e-10           # and spin-length-independent
    end

    @testset "single-ion anisotropy: classical and quantum-mode energies" begin
        cr1 = MR.Crystal(lat, reshape([0.0, 0, 0], 3, 1), [1], ["Fe"])
        bo = MR.SLCEBasis(cr1, MR.BasisSpec(; nbody = 1, cutoff = 1.5, lmax = [2],
                                             soc = true))
        mo = MR.SLCEModel(bo, 0.0, randn(rng, MR.n_salcs(bo)), bo.salc_basis.keys)
        @test energy_error(mo, 1.5, :dipole_uncorrected) < 1e-10
        @test energy_error(mo, 1.0, :dipole) < 1e-10              # s ≥ 1: quantum factor exact
        @test_throws ErrorException MR.to_sunny(mo; spin_length = 0.5, mode = :dipole)   # spin-1/2 guard
    end

    @testset "explicit (supercell) placement: energy matches" begin
        b = MR.SLCEBasis(cr2, MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                            soc = true))
        model = MR.SLCEModel(b, 0.3, randn(rng, MR.n_salcs(b)), b.salc_basis.keys)
        @test energy_error(model, 1.5, :dipole_uncorrected; placement = :explicit) < 1e-10
    end

    @testset "pair + single-ion together (supported channels) match" begin
        # lmax=[2] also brings unsupported ls=[2,2] pairs; zero those so the exported
        # System and the model carry the same energy, then check consistency.
        b = MR.SLCEBasis(cr2, MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                                            soc = true))
        keys = b.salc_basis.keys
        jphi = randn(rng, MR.n_salcs(b))
        for k in eachindex(keys)
            MR._classify_salc(keys[k].decors) === :unsupported && (jphi[k] = 0.0)
        end
        model = MR.SLCEModel(b, 0.2, jphi, keys)
        @test energy_error(model, 1.5, :dipole_uncorrected) < 1e-10
    end

    @testset "unsupported channels trigger a warning" begin
        b = MR.SLCEBasis(cr2, MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                                            soc = true))
        model = MR.SLCEModel(b, 0.0, ones(MR.n_salcs(b)), b.salc_basis.keys)
        @test_logs (:warn,) match_mode = :any MR.to_sunny(model; spin_length = 1.5,
                                                          mode = :dipole_uncorrected)
    end

    @testset "spin_length as a per-species mapping" begin
        b = MR.SLCEBasis(cr2, MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                            soc = false))
        model = MR.SLCEModel(b, 0.0, [0.01], b.salc_basis.keys)
        @test energy_error(model, Dict("Fe" => 1.5), :dipole_uncorrected) < 1e-10
    end

    # The `:coupling` route keeps `Moment` at a placeholder s₀ = 1 and folds the physical
    # S_eff into the couplings, so it works for a non-half-integer (itinerant) S_eff. The
    # whole energy landscape is then scaled by s₀/S, so `energy(sys) = (predict − j0)/S`.
    function coupling_energy_error(model, S, mode; seed = 7, ntrial = 20)
        sys = MR.to_sunny(model; spin_length = S, mode = mode, scaling = :coupling,
                          placement = :explicit)
        nat = MR.n_atoms(model.basis.crystal)
        rng = MersenneTwister(seed)
        me = 0.0
        for _ = 1:ntrial
            e = _rcfg(rng, nat)
            for i = 1:nat
                Sunny.set_dipole!(sys, e[:, i], (1, 1, 1, i))
            end
            me = max(me, abs(Sunny.energy(sys) - (MR.predict_energy(model, e) - model.j0) / S))
        end
        return me
    end

    @testset "spin scaling routes (:moment / :coupling)" begin
        b = MR.SLCEBasis(cr2, MR.BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                            soc = true))
        model = MR.SLCEModel(b, 0.5, randn(rng, MR.n_salcs(b)), b.salc_basis.keys)
        # :coupling carries S_eff in the couplings; energy is rescaled by 1/S. Works for a
        # NON-half-integer S_eff (1.1) that Sunny's `Moment` (the :moment route) rejects.
        @test coupling_energy_error(model, 1.1, :dipole_uncorrected) < 1e-10
        @test coupling_energy_error(model, 1.5, :dipole_uncorrected) < 1e-10
        # Forcing :moment on a non-half-integer S_eff is a clear, early error.
        @test_throws ErrorException MR.to_sunny(model; spin_length = 1.1, scaling = :moment)
        # Default :auto: non-half-integer builds (→ :coupling), half-integer is energy-exact
        # (→ :moment), so the existing energy-matching property is preserved.
        @test MR.to_sunny(model; spin_length = 1.1) isa Sunny.System
        @test energy_error(model, 1.5, :dipole) < 1e-10

        # Single-ion: the placeholder Moment can carry the classical (uncorrected) term,
        # but not the quantum (:dipole) quadrupole — that combination is rejected.
        cr1 = MR.Crystal(lat, reshape([0.0, 0, 0], 3, 1), [1], ["Fe"])
        bo = MR.SLCEBasis(cr1, MR.BasisSpec(; nbody = 1, cutoff = 1.5, lmax = [2],
                                             soc = true))
        mo = MR.SLCEModel(bo, 0.0, randn(rng, MR.n_salcs(bo)), bo.salc_basis.keys)
        @test coupling_energy_error(mo, 1.1, :dipole_uncorrected) < 1e-10
        @test_throws ErrorException MR.to_sunny(mo; spin_length = 1.1, scaling = :coupling,
                                                mode = :dipole)
    end

    @testset "coupling and moment give the same magnon dispersion" begin
        spg = MR.SpglibBackend()
        lat3 = MR.Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        crc = MR.Crystal(lat3, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
        bc = MR.SLCEBasis(crc, MR.BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1],
                                             soc = false); backend = spg)
        # Ferromagnetic nearest-neighbor coupling: aligned spins are the ground state.
        mc = MR.SLCEModel(bc, 0.0, [-0.02], bc.salc_basis.keys)

        function chain_dispersion(S, scaling)
            sys = MR.to_sunny(mc; spin_length = S, scaling = scaling, placement = :primitive)
            Sunny.polarize_spins!(sys, (0, 0, 1))
            swt = Sunny.SpinWaveTheory(sys; measure = nothing)
            qs = Sunny.q_space_path(sys.crystal, [[0, 0, 0], [0, 0, 1 / 2], [0, 0, 1]], 16)
            return Sunny.dispersion(swt, qs)
        end
        # At a half-integer S_eff both routes are valid and must yield the SAME dispersion
        # (the magnon frequency is invariant under the overall spin scale s₀/S).
        d_moment = chain_dispersion(3 / 2, :moment)
        d_coupling = chain_dispersion(3 / 2, :coupling)
        @test maximum(abs, d_moment .- d_coupling) < 1e-7
        # And a non-half-integer S_eff (the whole point) runs only through :coupling.
        @test chain_dispersion(1.1, :coupling) isa AbstractMatrix
    end

    # Copy a supercell spin config onto a reshaped (primitive→supercell) system by
    # matching each site's Cartesian position to a supercell atom.
    function _set_reshaped!(sys_r, model, e)
        cr = model.basis.crystal
        A = SMatrix{3,3,Float64}(cr.lattice.vectors)
        nat = MR.n_atoms(cr)
        key(f) = Tuple(mod.(round.(Int, 1000 .* f), 1000))
        lut = Dict(key(SVector{3,Float64}(cr.frac_positions[:, a])) => a for a = 1:nat)
        for site in Sunny.eachsite(sys_r)
            f = A \ SVector{3,Float64}(Sunny.global_position_at(sys_r, site))
            Sunny.set_dipole!(sys_r, e[:, lut[key(f)]], site)
        end
    end

    # The primitive system, reshaped back to the training supercell, must reproduce the
    # SLCE energy on arbitrary (non-uniform) configurations — i.e. the unfolded bonds
    # and their offsets are correct, not just the per-cell sums.
    function primitive_energy_error(model, S, mode; seed = 9, ntrial = 12)
        sys_p = MR.to_sunny(model; spin_length = S, mode = mode, placement = :primitive)
        prim = MR._sunny_primitive(model)
        sys_r = Sunny.reshape_supercell(sys_p, Matrix(prim.reshape))
        nat = MR.n_atoms(model.basis.crystal)
        rng = MersenneTwister(seed)
        me = 0.0
        for _ = 1:ntrial
            e = _rcfg(rng, nat)
            _set_reshaped!(sys_r, model, e)
            me = max(me, abs(Sunny.energy(sys_r) - (MR.predict_energy(model, e) - model.j0)))
        end
        return me, prim
    end

    @testset "primitive unfold: reshaped system reproduces the SLCE energy (offsets)" begin
        spg = MR.SpglibBackend()
        # 4-atom chain = 4× a 1-atom primitive chain
        lat = MR.Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        crc = MR.Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
        bc = MR.SLCEBasis(crc, MR.BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1],
                                             soc = true); backend = spg)
        mc = MR.SLCEModel(bc, 0.5, randn(rng, MR.n_salcs(bc)), bc.salc_basis.keys)
        me, prim = primitive_energy_error(mc, 1.5, :dipole_uncorrected)
        @test prim.clean && length(prim.positions) == 1
        @test me < 1e-10

        # bcc conventional (2 atoms) = 2× a 1-atom primitive
        latb = MR.Lattice(Matrix(3.0 * I(3)))
        crb = MR.Crystal(latb, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        bb = MR.SLCEBasis(crb, MR.BasisSpec(; nbody = 2, cutoff = 2.7, lmax = [1],
                                             soc = true); backend = spg)
        mb = MR.SLCEModel(bb, 0.0, randn(rng, MR.n_salcs(bb)), bb.salc_basis.keys)
        meb, primb = primitive_energy_error(mb, 2.0, :dipole)
        @test primb.clean
        @test meb < 1e-10

        # 2×2×1 simple-cubic supercell: the recovered primitive cell is left-handed
        # before the right-handedness fix — the default to_sunny (placement=:auto,
        # mode=:dipole) must build without a Sunny "not right-handed" crash.
        crs = MR.Crystal(MR.Lattice([6.0 0 0; 0 6.0 0; 0 0 3.0]),
                         [0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.5; 0.0 0.0 0.0 0.0],
                         [1, 1, 1, 1], ["Fe"])
        bs = MR.SLCEBasis(crs, MR.BasisSpec(; nbody = 2, cutoff = 3.1, lmax = [1],
                                             soc = false); backend = spg)
        ms = MR.SLCEModel(bs, 0.0, [0.01], bs.salc_basis.keys)
        sys = MR.to_sunny(ms; spin_length = 1.5)                # default placement/mode: must not throw
        @test sys isa Sunny.System
        mes, prims = primitive_energy_error(ms, 1.5, :dipole_uncorrected)
        @test prims.clean
        @test mes < 1e-10
    end
end
