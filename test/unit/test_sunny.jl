using Test
using SLCE
using SLCE: _l1_pair_matrix, _l2_onsite_matrix, _classify_salc,
    _bilinear_terms, _reconstruct_energy, _sunny_primitive, _assemble_spacegroup,
    Harmonics
using StaticArrays
using LinearAlgebra
using Random

# These exercise the Sunny-export conversion math in the core — the part that must
# be numerically correct — without loading Sunny. The actual Sunny.System assembly
# is tested in the separate `test/sunny/` environment.

_rdir(rng) = (v = randn(rng, 3); v / norm(v))
_rcfg(rng, n) = reshape(reduce(vcat, (_rdir(rng) for _ = 1:n)), 3, n)

@testset "Sunny export (core conversion math)" begin
    Z = Harmonics.Zlm
    rng = MersenneTwister(20)

    @testset "_l1_pair_matrix reproduces the Z₁ contraction" begin
        me = 0.0
        for _ = 1:200
            folded = randn(rng, 3, 3)
            M = _l1_pair_matrix(folded)
            ea, eb = _rdir(rng), _rdir(rng)
            ref = sum(folded[m1 + 2, m2 + 2] * Z(1, m1, SVector{3}(ea)) * Z(1, m2, SVector{3}(eb))
                      for m1 = -1:1, m2 = -1:1)
            me = max(me, abs(dot(ea, M * eb) - ref))
        end
        @test me < 1e-12
    end

    @testset "_l2_onsite_matrix reproduces the Z₂ contraction; symmetric traceless" begin
        me = 0.0
        for _ = 1:200
            folded = randn(rng, 5)
            A = _l2_onsite_matrix(folded)
            @test isapprox(A, A'; atol = 1e-12)
            @test abs(tr(A)) < 1e-12
            e = _rdir(rng)
            ref = sum(folded[m + 3] * Z(2, m, SVector{3}(e)) for m = -2:2)
            me = max(me, abs(dot(e, A * e) - ref))
        end
        @test me < 1e-12
    end

    @testset "_classify_salc" begin
        sd(ls...) = SLCE.spin_decors(collect(Int, ls))
        @test _classify_salc(sd(1, 1)) === :pair
        @test _classify_salc(sd(2)) === :onsite
        @test _classify_salc(sd(2, 2)) === :unsupported
        @test _classify_salc(sd(1, 1, 2)) === :unsupported
        # any displacement decoration disqualifies the bilinear extraction
        @test _classify_salc([SLCE.SiteDecor(; spin = 1),
                              SLCE.SiteDecor(; spin = 1, disp = (0, 1))]) ===
              :unsupported
        @test _classify_salc([SLCE.SiteDecor(; disp = (0, 1))]) === :unsupported
    end

    # Reconstruct the SLCE energy from the exported matrices: for a model whose only
    # channels are ls=[1,1] / ls=[2], this must equal predict_energy − j0.
    function _recon_max_err(basis; seed = 7, ntrial = 30)
        rng = MersenneTwister(seed)
        jphi = randn(rng, n_salcs(basis))
        j0 = 0.37
        model = SLCEModel(basis, j0, jphi, basis.salc_basis.keys)
        terms = _bilinear_terms(model)
        nat = n_atoms(basis.crystal)
        me = 0.0
        for _ = 1:ntrial
            e = _rcfg(rng, nat)
            me = max(me, abs(_reconstruct_energy(terms, e) - (predict_energy(model, e) - j0)))
        end
        return me, terms
    end

    @testset "energy reconstruction == predict − j0 (supported-only models)" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        me, terms = _recon_max_err(
            SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1], soc = true)))
        @test me < 1e-12
        @test isempty(terms.skipped)
        @test !isempty(terms.pairs)

        cr1 = Crystal(lat, reshape([0.0, 0, 0], 3, 1), [1], ["Fe"])
        meo, termso = _recon_max_err(
            SLCEBasis(cr1, BasisSpec(; nbody = 1, cutoff = 1.5, lmax = [2], soc = true)))
        @test meo < 1e-12
        @test isempty(termso.skipped)
        @test !isempty(termso.onsites)
    end

    @testset "Heisenberg pair → isotropic exchange matrix (M = J·I on the bond)" begin
        # the nearest-neighbor Heisenberg SALC alone: its exported 3×3 must be a
        # scalar multiple of the identity (no DM, no Γ).
        lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
        b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], soc = false))
        model = SLCEModel(b, 0.0, [0.0137], b.salc_basis.keys)
        terms = _bilinear_terms(model)
        for (_, M) in terms.pairs
            @test isapprox(M, (M[1, 1]) * I; atol = 1e-12)   # diagonal isotropic
        end
    end

    @testset "unsupported channels are reported, not silently dropped" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = true))
        model = SLCEModel(b, 0.0, ones(n_salcs(b)), b.salc_basis.keys)
        terms = _bilinear_terms(model)
        @test !isempty(terms.skipped)                         # ls=[2,2] pairs reported
        @test all(s -> occursin("not representable", s), terms.skipped)
    end

    @testset "primitive unfold (Sunny-free): clean fold reproduces the supercell" begin
        # a 4-atom chain that is a 4× supercell of a 1-atom primitive chain, with the
        # pure z-translations supplied as a manual space group (no Spglib in this env).
        lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.25 0.5 0.75], [1, 1, 1, 1], ["Fe"])
        rots = fill(SMatrix{3,3,Float64}(I), 4)
        trans = [SVector{3,Float64}(0, 0, t) for t in (0.0, 0.25, 0.5, 0.75)]
        sg = _assemble_spacegroup(cr, rots, trans, "chainZ4", 1; tol = 1e-8)
        nl = build_neighbor_list(cr, 2.6, MinimumImage())
        cls = build_clusters(cr, nl, sg; nbody = 2, selection = MinimumImage())
        salcs = build_salc_basis(cr, sg, cls; lmax_by_species = [1], scalar_only = false)
        basis = SLCEBasis(cr, sg, salcs,
                         BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [1], soc = true))
        rng = MersenneTwister(3)
        model = SLCEModel(basis, 0.5, randn(rng, n_salcs(basis)), basis.salc_basis.keys)

        prim = _sunny_primitive(model)
        @test prim.clean
        @test length(prim.positions) == 1                     # one sublattice
        ntran = round(Int, abs(det(Matrix(prim.reshape))))
        @test ntran == 4
        @test length(prim.bonds) * ntran ==
              length(_bilinear_terms(model).pairs)      # each prim bond ↔ ntran cells

        # uniform-config energy round-trip: supercell == ntran · (per-primitive-cell)
        terms = _bilinear_terms(model)
        e0 = (v = randn(rng, 3); v / norm(v))
        eu = repeat(e0, 1, n_atoms(cr))
        e_prim = sum(dot(e0, M * e0) for (_, M) in prim.bonds; init = 0.0)
        @test isapprox(_reconstruct_energy(terms, eu), ntran * e_prim; atol = 1e-10)
    end

    @testset "primitive unfold: two sublattices, no inversion" begin
        # The fixture above has ONE sublattice, so the `sublattice` translation-orbit
        # grouping, the per-sublattice origin, and `cellof`'s subtraction of it are all
        # exercised at their trivial value — a bug that mixes sublattices up cannot
        # show. It is also centrosymmetric, so a bond and its reverse are related by a
        # symmetry the unfold does not have to distinguish.
        #
        # This one is a 2× supercell of a TWO-atom primitive cell whose basis is
        # Fe at 0 and Co at 0.3 along z: two sublattices of different species, hence
        # no inversion centre anywhere (inversion would have to exchange Fe with Co).
        lat = Lattice([8.0 0 0; 0 8.0 0; 0 0 10.0])
        cr = Crystal(lat, [0 0 0 0; 0 0 0 0; 0.0 0.15 0.5 0.65],
                     [1, 2, 1, 2], ["Fe", "Co"])
        rots = fill(SMatrix{3,3,Float64}(I), 2)
        trans = [SVector{3,Float64}(0, 0, t) for t in (0.0, 0.5)]
        sg = _assemble_spacegroup(cr, rots, trans, "chainZ2-FeCo", 1; tol = 1e-8)
        nl = build_neighbor_list(cr, 3.6, MinimumImage())
        cls = build_clusters(cr, nl, sg; nbody = 2, selection = MinimumImage())
        salcs = build_salc_basis(cr, sg, cls; lmax_by_species = [1, 1], scalar_only = false)
        basis = SLCEBasis(cr, sg, salcs,
                         BasisSpec(; nbody = 2, cutoff = 3.6, lmax = [1, 1], soc = true))
        rng = MersenneTwister(11)
        model = SLCEModel(basis, 0.25, randn(rng, n_salcs(basis)), basis.salc_basis.keys)

        prim = _sunny_primitive(model)
        @test prim.clean
        @test length(prim.positions) == 2                     # two sublattices
        @test Set(prim.types) == Set(["Fe", "Co"])            # and they are told apart
        @test all(p -> all(0 .<= p .< 1), prim.positions)      # inside the primitive cell
        ntran = round(Int, abs(det(Matrix(prim.reshape))))
        @test ntran == 2
        @test length(prim.bonds) * ntran == length(_bilinear_terms(model).pairs)
        # every primitive bond joins two REAL sublattices at a resolvable offset
        for ((i, j, n), _) in prim.bonds
            @test 1 <= i <= 2 && 1 <= j <= 2
            @test (i, j, Tuple(n)) != (i, i, (0, 0, 0))       # no self-bond at zero
        end

        terms = _bilinear_terms(model)
        e0 = (v = randn(rng, 3); v / norm(v))
        eu = repeat(e0, 1, n_atoms(cr))
        e_prim = sum(dot(e0, M * e0) for (_, M) in prim.bonds; init = 0.0)
        @test isapprox(_reconstruct_energy(terms, eu), ntran * e_prim; atol = 1e-10)
    end

    @testset "no symmetry ⇒ primitive fold is the supercell itself (clean, trivial)" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1], soc = true))
        model = SLCEModel(b, 0.0, ones(n_salcs(b)), b.salc_basis.keys)
        prim = _sunny_primitive(model)               # NoSymmetry ⇒ only the identity translation
        @test prim.clean
        @test length(prim.positions) == n_atoms(cr)
        @test round(Int, abs(det(Matrix(prim.reshape)))) == 1
    end

    @testset "to_sunny without Sunny gives a helpful error" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        b = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1], soc = false))
        model = SLCEModel(b, 0.0, zeros(n_salcs(b)), b.salc_basis.keys)
        @test_throws ErrorException to_sunny(model; spin_length = 1)
    end
end
