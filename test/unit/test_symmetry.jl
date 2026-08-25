using Test
using SLCE
using SLCE: _assemble_spacegroup, n_ops, SpaceGroup
using StaticArrays
using LinearAlgebra

# A backend that hands back {E, m_z} — the mirror through z = 0 — however the crystal
# is placed. It stands in for Spglib in the aperiodic-axis tests below: Spglib is not
# available in this environment, and the point of those tests is what the assembler
# does with a backend's answer, not which answer Spglib gives.
struct _MirrorZBackend <: SLCE.AbstractSymmetryBackend end
SLCE.analyze_symmetry(::_MirrorZBackend, c::Crystal; tol::Real = 1e-5) =
    _assemble_spacegroup(c, [SMatrix{3,3,Float64}(I),
                             SMatrix{3,3,Float64}(Diagonal([1.0, 1.0, -1.0]))],
                         [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)],
                         "Pm", 6; tol = tol)

@testset "symmetry" begin
    lat = Lattice(Matrix(3.0 * I(3)))

    @testset "NoSymmetry → P1" begin
        crystal = Crystal(lat, [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
        sg = analyze_symmetry(NoSymmetry(), crystal)
        @test sg.symbol == "P1"
        @test sg.number == 1
        @test length(sg.ops) == 1
        @test sg.ops[1].is_translation
        @test sg.ops[1].is_proper
        @test sg.map_sym == reshape([1, 2], 2, 1)
        @test sg.translation_ops == [1]
    end

    @testset "assembler: identity + inversion on a centrosymmetric pair" begin
        cryst = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        rots = [SMatrix{3,3,Float64}(I), SMatrix{3,3,Float64}(-I)]
        trans = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0, 0, 0)]
        sg = _assemble_spacegroup(cryst, rots, trans, "manual", 0; tol = 1e-6)
        @test sg.ops[1].is_proper            # identity proper
        @test sg.ops[2].is_proper == false   # inversion improper
        @test isapprox(sg.ops[2].rotation_cart, SMatrix{3,3,Float64}(-I); atol = 1e-12)
        @test sg.map_sym[:, 1] == [1, 2]     # identity
        @test sg.map_sym[:, 2] == [2, 1]     # inversion swaps the pair
        for o = 1:n_ops(sg)
            @test sort(sg.map_sym[:, o]) == [1, 2]   # each op is a permutation
        end
    end

    @testset "the assembler refuses a set that is not a space group" begin
        # Downstream code consumes a `SpaceGroup` AS a group — orbits are reduced by
        # stabilizer counting and SALCs are projected with (1/|G|) Σ_g — so a merely
        # plausible list of matrices does not fail, it silently produces wrong
        # multiplicities and a non-idempotent projector. Each of these used to load.
        cryst = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        E = SMatrix{3,3,Float64}(I)
        z0 = SVector{3,Float64}(0, 0, 0)
        asm(rots, trans; tol = 1e-6) =
            _assemble_spacegroup(cryst, rots, trans, "bad", 0; tol = tol)

        # a 3-fold rotation is not an operation of a cubic lattice: integral it is not
        cz = SMatrix{3,3,Float64}([cos(2π/3) -sin(2π/3) 0; sin(2π/3) cos(2π/3) 0; 0 0 1])
        @test_throws ArgumentError asm([E, cz, cz * cz], fill(z0, 3))
        # integral and det = 1, but a shear: it does not preserve the lattice metric,
        # which integrality alone never implies
        sh = SMatrix{3,3,Float64}([1 1 0; 0 1 0; 0 0 1])
        @test_throws ArgumentError asm([E, sh], fill(z0, 2))
        # ... and no value of `tol` buys a pass: whether a matrix is integral is a
        # question about a DIMENSIONLESS quantity, while `tol` is the backend's
        # symprec, a distance in Å. A rotation 0.3 away from an integer matrix is not
        # an operation of any lattice at any tolerance. (Feeding `tol` in as the matrix
        # tolerance used to round this one to the identity and let it through at
        # `tol = 0.5`.)
        bogus = SMatrix{3,3,Float64}([1 0.3 0; 0 1 0; 0 0 1])
        @test_throws ArgumentError asm([E, bogus], fill(z0, 2); tol = 0.5)
        # not a bijection of the lattice
        @test_throws ArgumentError asm([E, SMatrix{3,3,Float64}(2I)], fill(z0, 2))
        # the identity is what every projector and every orbit is anchored on
        @test_throws ArgumentError asm([SMatrix{3,3,Float64}(-I)], [z0])
        # duplicates inflate every orbit multiplicity
        @test_throws ArgumentError asm([E, E], fill(z0, 2))
        @test_throws ArgumentError asm([E, E], [z0, SVector{3,Float64}(1, 0, 0)])
        # closed under inverses (each is an involution) but not under composition:
        # diag(-1,1,1)·diag(1,-1,1) = diag(-1,-1,1) is absent
        mx = SMatrix{3,3,Float64}(Diagonal([-1.0, 1, 1]))
        my = SMatrix{3,3,Float64}(Diagonal([1.0, -1, 1]))
        @test_throws ArgumentError asm([E, mx, my], fill(z0, 3))
        # rotation parts closed AND every inverse present, yet (W|t) is not:
        # (C2z|0) ∘ (I|½x) = (C2z|−½x) is absent
        c2z = SMatrix{3,3,Float64}(Diagonal([-1.0, -1, 1]))
        hx = SVector{3,Float64}(0.5, 0, 0)
        @test_throws ArgumentError asm([E, E, c2z], [z0, hx, z0])
        # ... and the honest group of that shape loads, on a crystal it maps onto
        halved = Crystal(lat, [0.25 0.75; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        sg = _assemble_spacegroup(halved, [E, E, c2z, c2z], [z0, hx, z0, hx],
                                  "P2/m-like", 0; tol = 1e-6)
        @test n_ops(sg) == 4

        # `map_sym` columns are documented as permutations; two atoms sharing an image
        # means `tol` is too loose for this structure, which the orbit code would then
        # miscount rather than reject. The separation here is 1e-3 × 3 Å = 3e-3 Å, so
        # `tol = 1e-2` Å resolves nothing and `tol = 1e-6` Å resolves both.
        near = Crystal(lat, [0.0 1e-3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        @test _assemble_spacegroup(near, [E], [z0], "P1", 1; tol = 1e-6) isa SpaceGroup
        @test_throws ErrorException _assemble_spacegroup(near, [E], [z0], "P1", 1;
                                                         tol = 1e-2)
    end

    @testset "`tol` is a Cartesian distance, not a fractional one" begin
        # `tol` is the backend's symprec: a distance in Å. Whether two atoms are close
        # enough to collide under an operation is therefore a statement about Å, and
        # its verdict cannot depend on the SIZE or the SHAPE of the cell those Å happen
        # to be expressed in. Both properties below follow from what the number means,
        # not from what this code computes — and each one is violated by a fractional
        # comparison (measured on the fractional form: the cubic case collides at
        # L = 24 Å and not at L = 3 Å; the hexagonal case collides along `c` and not
        # along `a`, a 4× spread in effective tolerance across directions of one cell).
        E = SMatrix{3,3,Float64}(I)
        z0 = SVector{3,Float64}(0, 0, 0)
        collides(lat, dfrac, tol) =
            try
                _assemble_spacegroup(Crystal(lat, hcat([0.0, 0.0, 0.0], dfrac),
                                             [1, 1], ["Fe"]),
                                     [E], [z0], "P1", 1; tol = tol)
                false
            catch
                true
            end

        sep = 5e-3                    # Å between the two Fe
        tol = 1e-3                    # Å, five times smaller: they are resolved
        a, c = 3.0, 12.0
        hex = Lattice([a -a / 2 0.0; 0.0 a * sqrt(3) / 2 0.0; 0.0 0.0 c])

        # (i) cell size: the same two atoms, the same Å apart, in a 3 Å cell and a 24 Å
        # one — the second is a plausible 8× supercell of the first.
        for L in (3.0, 24.0)
            @test !collides(Lattice(Matrix(L * I(3))), [sep / L, 0.0, 0.0], tol)
        end
        # (ii) cell shape: `sep` along `a` and `sep` along `c` of a γ = 120° cell with
        # a ≠ c are the same distance, so they get the same answer.
        @test !collides(hex, [sep / a, 0.0, 0.0], tol)
        @test !collides(hex, [0.0, 0.0, sep / c], tol)

        # ... and the tolerance still bites, everywhere, when the separation really is
        # below it. Without this half the gate above would also pass a `tol` that had
        # been quietly disabled.
        small = tol / 2
        for L in (3.0, 24.0)
            @test collides(Lattice(Matrix(L * I(3))), [small / L, 0.0, 0.0], tol)
        end
        @test collides(hex, [small / a, 0.0, 0.0], tol)
        @test collides(hex, [0.0, 0.0, small / c], tol)
    end

    @testset "unloaded backend → friendly error" begin
        crystal = Crystal(lat, reshape([0.0, 0.0, 0.0], 3, 1), [1], ["Fe"])
        # Spglib is not loaded in the core test env, so SpglibBackend falls through
        # to the abstract-supertype method, which errors clearly.
        @test_throws ErrorException analyze_symmetry(SpglibBackend(), crystal)
    end
    @testset "an aperiodic axis restricts the group to the operations it really has" begin
        # A backend analyses the cell as a fully periodic 3D crystal — Spglib is not
        # told about `Lattice(...; pbc)`. Its answer must therefore be intersected
        # with the declared periodicity before anything downstream (`build_clusters`
        # demands closure under the group; the neighbour list refuses images along an
        # aperiodic axis, so an operation that closes only through the artificial
        # periodicity surfaces as a closure failure blamed on the tie tolerance).
        #
        # All three cases below are decided by hand from the geometry, not read off
        # the implementation. `slab` is a tetragonal cell with `c = 12`; `m_z` is the
        # mirror through `z = 0`, which the backend reports with `t = 0`.
        tall = Lattice([3.0 0 0; 0 3.0 0; 0 0 12.0]; pbc = (true, true, false))
        tallp = Lattice([3.0 0 0; 0 3.0 0; 0 0 12.0])

        # (a) CENTRED at z = 1/2: the three layers sit at 3/8, 1/2, 5/8, which IS
        # mirror-symmetric — about z = 1/2, not about z = 0. The two mirrors differ by
        # the lattice translation `c`, which along an aperiodic axis is not an
        # identification one may make: exactly one of them is the operation the finite
        # slab has. So nothing may be dropped, and the surviving mirror must carry the
        # representative `t_z = 1` (`m_z` about z = 0 composed with +c), which is what
        # makes `round(W·x_a + t − x_b)` vanish along z for every atom.
        centred = Crystal(tall, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.375 0.5 0.625],
                          [1, 1, 1], ["Fe"])
        sg_c = analyze_symmetry(_MirrorZBackend(), centred)
        @test n_ops(sg_c) == 2
        @test !occursin("pbc subgroup", sg_c.symbol)
        gi = findfirst(o -> o.rotation_frac[3, 3] < 0, sg_c.ops)
        mirror = sg_c.ops[gi]
        @test mirror.translation_frac[3] ≈ 1.0
        # Stated where it is physical: in Cartesian terms the layers sit at 4.5, 6.0
        # and 7.5 Å, and the mirror at z = 6 Å sends them to 7.5, 6.0, 4.5 — every
        # image an atom that is there, with no cell translation added. The backend's
        # `t_z = 0` representative sends 4.5 Å to −4.5 Å, which is not.
        cart = cartesian_positions(centred)
        A = centred.lattice.vectors
        for a = 1:3
            img = A * (mirror.rotation_frac * centred.frac_positions[:, a] +
                       mirror.translation_frac)
            @test isapprox(img, cart[:, sg_c.map_sym[a, gi]]; atol = 1e-12)
        end

        # (b) STRADDLING z = 0: layers at 7/8, 0, 1/8. Under the artificial
        # periodicity that is the same slab as (a) shifted, and `m_z` maps 1/8 to
        # −1/8 ≡ 7/8. Along a FINITE z it is three layers at 0, 1.5, 10.5 Å, and no
        # mirror plane fixes that set (a plane at 0.75 Å would send 1.5 → 0 but 10.5
        # to −9). So `m_z` must go, leaving the identity alone.
        straddle = Crystal(tall, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.875 0.0 0.125],
                           [1, 1, 1], ["Fe"])
        sg_s = @test_logs (:warn,) match_mode = :any analyze_symmetry(_MirrorZBackend(),
                                                                      straddle)
        @test n_ops(sg_s) == 1
        @test sg_s.ops[1].is_translation                     # the identity
        @test endswith(sg_s.symbol, "(pbc subgroup)")

        # (c) the SAME atoms declared fully periodic keep both operations and the
        # backend's symbol: the restriction is a consequence of the declaration, not a
        # change of behaviour for a periodic crystal.
        periodic = Crystal(tallp, [0.0 0.0 0.0; 0.0 0.0 0.0; 0.875 0.0 0.125],
                           [1, 1, 1], ["Fe"])
        sg_p = analyze_symmetry(_MirrorZBackend(), periodic)
        @test n_ops(sg_p) == 2
        @test sg_p.symbol == "Pm"

        # (d) end to end: (b) used to build a neighbour list that cannot produce the
        # z-shifted image `m_z` implies, and died in `build_clusters` with "check the
        # image selection and the tie tolerance" — a cause it does not have.
        @test SLCEBasis(straddle, BasisSpec(; nbody = 2, cutoff = 3.2, lmax = [1],
                                           soc = false);
                       backend = _MirrorZBackend()) isa SLCEBasis
    end

    @testset "an aperiodic axis is never wrapped, so positions may span cells" begin
        # There is no period to wrap with along an aperiodic axis, so `Crystal` leaves
        # those coordinates as given and a legitimate structure may list positions
        # outside `[0, 1)` there. Two atoms an exact cell apart are then indivisible to
        # a mod-1 comparison — which is what the matcher uses to find a candidate — so
        # the whole-operation integer shift has to be searched over, not read off the
        # first candidate atom 1 happens to hit.
        #
        # `c = 6` Å with atoms at `z = 0` and `z = 1`, i.e. Cartesian 0 and 6 Å.
        # By hand: the two points are mirror-symmetric about `z = 3` Å, the plane
        # `z_frac = 1/2`. The backend reports that mirror as the one at `z = 0` with
        # `t_z = 0` (the same operation modulo `c`), so the physical representative is
        # `t_z = 1`. Both operations are genuine and neither may be dropped — the
        # identity least of all.
        span = Lattice([3.0 0 0; 0 3.0 0; 0 0 6.0]; pbc = (true, true, false))
        cryst = Crystal(span, [0.0 0.0; 0.0 0.0; 0.0 1.0], [1, 1], ["Fe"])
        @test cryst.frac_positions[3, :] == [0.0, 1.0]        # not wrapped
        sg = analyze_symmetry(_MirrorZBackend(), cryst)
        @test n_ops(sg) == 2
        @test !occursin("pbc subgroup", sg.symbol)
        mirror = sg.ops[findfirst(o -> o.rotation_frac[3, 3] < 0, sg.ops)]
        @test mirror.translation_frac[3] ≈ 1.0
        @test sg.map_sym[:, findfirst(o -> o.rotation_frac[3, 3] < 0, sg.ops)] == [2, 1]
        # and the identity is still the identity, not "atom 2 maps to atom 1"
        @test sg.map_sym[:, findfirst(o -> o.is_translation, sg.ops)] == [1, 2]
    end

    @testset "an operation that mixes a periodic and an aperiodic axis is refused" begin
        # `x ↔ z` on a cell with `a = c` permutes the two atoms below exactly, so the
        # atom-image test alone would keep it. It still cannot be a symmetry when `c`
        # is aperiodic: it sends the lattice translation `a` into a direction with no
        # lattice translations to receive it, so the declared translation lattice is
        # not mapped onto itself. Refused on the rotation alone, before any atom is
        # looked at.
        cube = Lattice(Matrix(3.0 * I(3)); pbc = (true, true, false))
        cryst = Crystal(cube, [0.0 0.5; 0.0 0.0; 0.0 0.5], [1, 1], ["Fe"])
        swap = SMatrix{3,3,Float64}([0.0 0 1; 0 1 0; 1 0 0])
        z0 = SVector{3,Float64}(0, 0, 0)
        sg = @test_logs (:warn,) match_mode = :any _assemble_spacegroup(
            cryst, [SMatrix{3,3,Float64}(I), swap], [z0, z0], "manual", 0; tol = 1e-6)
        @test n_ops(sg) == 1
        @test endswith(sg.symbol, "(pbc subgroup)")
        # ... and it survives when all three axes are periodic
        cubep = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 0.5; 0.0 0.0; 0.0 0.5],
                        [1, 1], ["Fe"])
        @test n_ops(_assemble_spacegroup(cubep, [SMatrix{3,3,Float64}(I), swap],
                                         [z0, z0], "manual", 0; tol = 1e-6)) == 2
    end
end
