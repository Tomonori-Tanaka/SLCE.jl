# `symmetrize` and the symmetry door on noisy real crystals, against Spglib (OPEN-WORK
# S-5). The core suite gates the same functions on hand-assembled groups; what only a
# real backend can show is (a) that every cell Spglib accepts is also accepted by the
# assembler, (b) the looser-`tol` warning, and (c) the user's path: a noisy crystal,
# symmetrized, builds the ideal crystal's basis.

_frac(v...) = collect(Float64, v)
function _sym_crystals()
    fcc = hcat(_frac(0, 0, 0), _frac(0, 0.5, 0.5), _frac(0.5, 0, 0.5), _frac(0.5, 0.5, 0))
    cubic(a) = a * Matrix(1.0I, 3, 3)
    a, c = 2.5, 2.5 * 1.633
    hexagonal = [a -a/2 0; 0 a*sqrt(3)/2 0; 0 0 c]
    α = deg2rad(80.0)
    cα = cos(α)
    sα = sin(α)
    cy = (cα - cα^2) / sα
    rhombohedral = 3.0 * [1 cα cα; 0 sα cy; 0 0 sqrt(1 - cα^2 - cy^2)]
    return ("fcc" => Crystal(Lattice(cubic(3.6)), fcc, ones(Int, 4), ["Cu"]),
            "bcc" => Crystal(Lattice(cubic(2.87)), hcat(_frac(0, 0, 0), _frac(0.5, 0.5, 0.5)),
                             [1, 1], ["Fe"]),
            "hcp" => Crystal(Lattice(hexagonal),
                             hcat(_frac(1/3, 2/3, 1/4), _frac(2/3, 1/3, 3/4)), [1, 1], ["Co"]),
            "rocksalt" => Crystal(Lattice(cubic(5.6)), hcat(fcc, fcc .+ 0.5),
                                  [1, 1, 1, 1, 2, 2, 2, 2], ["Na", "Cl"]),
            "B2" => Crystal(Lattice(cubic(2.99)), hcat(_frac(0, 0, 0), _frac(0.5, 0.5, 0.5)),
                            [1, 2], ["Fe", "Rh"]),
            "diamond" => Crystal(Lattice(cubic(5.43)), hcat(fcc, fcc .+ 0.25), ones(Int, 8),
                                 ["Si"]),
            "rhombohedral" => Crystal(Lattice(rhombohedral),
                                      hcat(_frac(0.23, 0.23, 0.23), _frac(-0.23, -0.23, -0.23)),
                                      [1, 1], ["Bi"]))
end
_sym_moved(c, s) = Crystal(c.lattice, c.frac_positions + s, c.species, c.species_labels)
_spg_ops(c, tol) = Spglib.get_dataset(Spglib.Cell(Matrix(c.lattice.vectors),
                                                  collect(eachcol(c.frac_positions)),
                                                  c.species), tol)

@testset "symmetrize and the symmetry door on noisy crystals (S-5)" begin
    @testset "every cell Spglib accepts, the assembler accepts" begin
        # Cartesian noise 1e-6 … 1e-5 Å per component, three origins (none, 3e-6
        # fractional off the symmetry elements, a general one), at tol = 1e-5. Before
        # S-5 the assembler refused 117 of 1260 such cells (no image within exactly
        # `tol`; translations snapped to zero breaking the group closure) — measured on
        # the full sweep, of which this is a fifth.
        tol = 1e-5
        accepted = 0
        refused = String[]
        for (name, c0) in _sym_crystals(), amp in (1e-6, 3e-6, 1e-5), origin = 1:3,
            seed = 1:4
            rng = MersenneTwister(hash((name, amp, origin, seed)))
            shift = origin == 1 ? zeros(3) : origin == 2 ? fill(3e-6, 3) : 0.1 .* rand(rng, 3)
            noise = c0.lattice.vectors \ (amp .* (2 .* rand(rng, 3, n_atoms(c0)) .- 1))
            c = _sym_moved(c0, noise .+ shift)
            _spg_ops(c, tol) === nothing && continue
            try
                SLCE.analyze_symmetry(SpglibBackend(), c; tol = tol)
                accepted += 1
            catch err
                push!(refused, "$name $amp $origin $seed: " * first(sprint(showerror, err), 100))
            end
        end
        @test isempty(refused)
        @test accepted == 7 * 3 * 3 * 4
    end

    @testset "a noisy crystal, symmetrized, builds the ideal basis" begin
        name_crystal = _sym_crystals()
        hcp = last(name_crystal[3])
        spec = BasisSpec(; nbody = 2, cutoff = 2.6, lmax = [2], soc = false)
        ideal = SLCEBasis(hcp, spec; backend = SpglibBackend())
        jitter = 3e-7 .* (2 .* rand(MersenneTwister(1), 3, 2) .- 1)     # Å
        noisy = _sym_moved(hcp, hcp.lattice.vectors \ jitter)
        @test _spg_ops(noisy, 1e-5).n_operations == 24
        # refused at the default tie band, with the residue and the remedy named
        err = try
            SLCEBasis(noisy, spec; backend = SpglibBackend())
            nothing
        catch e
            sprint(showerror, e)
        end
        @test err !== nothing && occursin("symmetrize(crystal", err)
        symmetric = @test_logs (:info, r"symmetrized onto P6_3/mmc") symmetrize(noisy;
                                                                    backend = SpglibBackend())
        @test _spg_ops(symmetric, 1e-9).n_operations == 24
        basis = SLCEBasis(symmetric, spec; backend = SpglibBackend())
        @test SLCE.n_ops(basis.spacegroup) == SLCE.n_ops(ideal.spacegroup) == 24
        @test basis.salc_basis.keys == ideal.salc_basis.keys
    end

    @testset "a looser search that finds more operations is reported" begin
        # fcc with 3e-5 Å noise: at tol = 1e-5 Spglib finds a small group, at 1e-4 the
        # full one — the warning fires and names both. The ideal crystal stays silent.
        fcc = last(first(_sym_crystals()))
        jitter = 3e-5 .* (2 .* rand(MersenneTwister(2), 3, 4) .- 1)     # Å
        noisy = _sym_moved(fcc, fcc.lattice.vectors \ jitter)
        small = _spg_ops(noisy, 1e-5).n_operations
        @test small < _spg_ops(noisy, 1e-4).n_operations
        @test_logs (:warn, r"ten times looser finds more operations") match_mode = :any begin
            SLCE.analyze_symmetry(SpglibBackend(), noisy; tol = 1e-5)
        end
        @test_logs SLCE.analyze_symmetry(SpglibBackend(), fcc; tol = 1e-5)
        # and symmetrize at the tolerance that finds the full group gets it back
        full = @test_logs (:info, r"symmetrized onto Fm-3m") match_mode = :any symmetrize(
            noisy; backend = SpglibBackend(), tol = 1e-4)
        @test _spg_ops(full, 1e-9).n_operations == 192
    end
end
