# The extended-XYZ training-data container (src/io/extxyz.jl) and the moment
# channel's axis-consistency gates (check_moment_gates, src/io/dftsource.jl), plus
# the legacy EMBSET pair reader (read_embset_pair, src/io/embset.jl). The headline
# gates: (1) every stored value survives the file bit-exactly (shortest-round-trip
# printing); (2) spin-only vs joint is MEASURED from positions, and a config_type
# claim contradicting the measurement is a loud error; (3) the axis gates fire on
# generation AND on load — archived constraint axes are re-verified, never believed.

using Test
using SLCE
using SLCE: DatumProvenance
using LinearAlgebra
using Random

@testset "extxyz container + moment gates + EMBSET pair" begin
    tmp = mktempdir()
    xt = Crystal(Lattice(3.0 .* [1.0 0 0; 0 1 0; 0 0 1]),
                 [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 2], ["Fe", "Ge"])
    nat = 2
    rng = MersenneTwister(0x5CE2)
    mkdirs() = (m = randn(rng, 3, nat);
                for a = 1:nat; m[:, a] ./= norm(m[:, a]); end; m)
    function mkdat(i; joint::Bool = false, mode = 4)
        dirs = mkdirs()
        mags = [1.2, 0.1] .+ 0.01 .* rand(rng, nat)
        M = 0.9 .* (mags' .* dirs) .+ 0.001 .* randn(rng, 3, nat)
        prov = joint ?
               DatumProvenance(; reference_id = "ref",
                               reference_fingerprint = crystal_fingerprint(xt),
                               setup_id = "s1", soc = false) :
               DatumProvenance(; setup_id = "s1", soc = false)
        TrainingDatum(; energy = -1.0 - 0.1i, directions = dirs, magmoms = mags,
                      field = 0.01 .* randn(rng, 3, nat), moments_bare = M,
                      constraint_axes = dirs, constraint_mode = mode,
                      displacements = joint ? 0.01 .* randn(rng, 3, nat) : nothing,
                      forces = joint ? randn(rng, 3, nat) : nothing,
                      provenance = prov)
    end

    @testset "spin-only round trip: stored values survive bit-exactly" begin
        data = [mkdat(i) for i = 1:4]
        f = joinpath(tmp, "spin.extxyz")
        write_extxyz(f, data, xt; field_sign = "vasp:+B", source = "test")
        back = read_extxyz(f)
        @test length(back) == 4
        for i = 1:4
            @test back[i].energy == data[i].energy                    # bitwise
            @test back[i].field == data[i].field                      # bitwise
            @test back[i].moments_bare == data[i].moments_bare        # bitwise
            @test back[i].constraint_axes == data[i].constraint_axes  # bitwise
            @test back[i].constraint_mode == 4
            @test back[i].displacements === nothing
            @test back[i].torques !== nothing            # derived from the field
            # directions/magmoms are re-derived from the written moment vectors —
            # exact up to the unit normalization, deliberately not bitwise
            @test maximum(abs, back[i].directions .- data[i].directions) < 1e-14
            @test maximum(abs, back[i].magmoms .- data[i].magmoms) < 1e-14
            @test back[i].provenance.setup_id == "s1"
            @test back[i].provenance.soc == false
        end
        # a second round trip is stable: the stored channels stay bitwise (the mw
        # column re-derives from magmoms .* directions, so IT may move by an ulp —
        # which is exactly why the bare channels are stored as their own columns)
        f2 = joinpath(tmp, "spin2.extxyz")
        write_extxyz(f2, back, xt; field_sign = "vasp:+B", source = "test")
        back2 = read_extxyz(f2)
        for i = 1:4
            @test back2[i].energy == back[i].energy
            @test back2[i].field == back[i].field
            @test back2[i].moments_bare == back[i].moments_bare
            @test back2[i].constraint_axes == back[i].constraint_axes
            @test maximum(abs, back2[i].directions .- back[i].directions) < 1e-14
        end
    end

    @testset "joint round trip: u measured against the reference" begin
        dj = [mkdat(i; joint = true) for i = 1:3]
        f = joinpath(tmp, "joint.extxyz")
        write_extxyz(f, dj, xt)
        bj = read_extxyz(f; reference = xt)
        for i = 1:3
            @test bj[i].displacements !== nothing
            @test maximum(abs, bj[i].displacements .- dj[i].displacements) < 1e-12
            @test bj[i].forces == dj[i].forces                        # bitwise
            @test bj[i].provenance.reference_fingerprint == crystal_fingerprint(xt)
        end
        # joint file without a reference: loud, names the remedy
        err = try
            read_extxyz(f)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("reference", err.msg)
        # spin-only data read WITH a reference: u ≡ 0 measured → spin-only datums
        ds = [mkdat(i) for i = 1:2]
        fs = joinpath(tmp, "spin_ref.extxyz")
        write_extxyz(fs, ds, xt)
        bs = read_extxyz(fs; reference = xt)
        @test all(d.displacements === nothing for d in bs)
    end

    @testset "u is measured with a band: round-off is not a displacement field" begin
        # This is an INTERCHANGE format, so a file's writer is routinely a different
        # build, and the reference side is a freshly recomputed `vectors * frac` whose
        # last bits depend on the arithmetic path. Measuring u exactly turned a 1-ulp
        # mismatch into a JOINT dataset carrying a displacement field of round-off.
        # The distinction is about >~ 1e-3 A; both scales are asserted so the band
        # cannot drift into either.
        ds = [mkdat(i) for i = 1:2]
        fb = joinpath(tmp, "band.extxyz")
        write_extxyz(fb, ds, xt)
        frac = Matrix(xt.frac_positions)
        frac[1, 2] += eps(1.5) / 3.0            # one ulp of a 1.5 A coordinate
        near = Crystal(xt.lattice, frac, xt.species, xt.species_labels)
        @test cartesian_positions(near) != cartesian_positions(xt)   # a real last bit
        @test all(d.displacements === nothing for d in read_extxyz(fb; reference = near))
        # Pinned at the band's OWN scale, keyed to the constant: a tenth of the band
        # is still no displacement. (The other side, ten times the band, is asserted
        # below on the joint file — a spin-only file read against a reference far
        # enough to measure joint is caught one gate later by its config_type claim.)
        atol = SLCE._REF_GEOM_ATOL
        frac[1, 2] = xt.frac_positions[1, 2] + 0.1 * atol / 3.0
        just_in = Crystal(xt.lattice, frac, xt.species, xt.species_labels)
        @test all(d.displacements === nothing
                  for d in read_extxyz(fb; reference = just_in))

        # The other end of the band: a 1e-3 A shift of the reference must move the
        # measured u by exactly that, not be swallowed. Measured on a JOINT file, since
        # a spin-only file read against a displaced reference is caught one gate later
        # by its own config_type claim.
        frac[1, 2] = xt.frac_positions[1, 2] + 1e-3 / 3.0            # 1e-3 A
        far = Crystal(xt.lattice, frac, xt.species, xt.species_labels)
        dj2 = [mkdat(1; joint = true)]
        fj = joinpath(tmp, "band_joint.extxyz")
        write_extxyz(fj, dj2, xt)
        u_ref = read_extxyz(fj; reference = xt)[1].displacements
        u_far = read_extxyz(fj; reference = far)[1].displacements
        delta = u_far - u_ref
        @test delta[1, 2] ≈ -1e-3 rtol = 1e-9
        @test maximum(abs, delta[:, 1]) == 0.0
        # and the one-ulp reference moves the same joint file's u only by round-off
        @test maximum(abs, read_extxyz(fj; reference = near)[1].displacements - u_ref) <
              1e-14
        # ten times the band IS a displacement, measured at its own scale — the
        # assertion that fails if the band is ever quietly widened
        frac[1, 2] = xt.frac_positions[1, 2] + 10 * atol / 3.0
        just_out = Crystal(xt.lattice, frac, xt.species, xt.species_labels)
        u_out = read_extxyz(fj; reference = just_out)[1].displacements
        @test (u_out - u_ref)[1, 2] ≈ -10 * atol rtol = 1e-9
    end

    @testset "loud checks: claims never override measurements" begin
        data = [mkdat(i) for i = 1:2]
        f = joinpath(tmp, "claims.extxyz")
        write_extxyz(f, data, xt)
        txt = read(f, String)
        # config_type flag contradicting the measured spin-only answer
        bad = joinpath(tmp, "claims_bad.extxyz")
        write(bad, replace(txt, "config_type=spin-only" => "config_type=joint"))
        @test_throws ArgumentError read_extxyz(bad)
        # units_field: the "T" header mislabel is refused
        write(bad, replace(txt, "units_field=eV/muB" => "units_field=T"))
        @test_throws ArgumentError read_extxyz(bad)
        # mconstr columns without a constraint_mode key
        write(bad, replace(txt, " constraint_mode=4" => ""))
        @test_throws ArgumentError read_extxyz(bad)
        # truncated file
        write(bad, join(split(txt, "\n")[1:3], "\n"))
        @test_throws ArgumentError read_extxyz(bad)
        # writer refuses mixed channel presence (one file = one observation set)
        nofield = TrainingDatum(; energy = -1.0, directions = mkdirs(),
                                magmoms = [1.0, 1.0])
        @test_throws ArgumentError write_extxyz(joinpath(tmp, "mix.extxyz"),
                                                [data[1], nofield], xt)
    end

    @testset "axis gates: generation-time and load-time, loud" begin
        zhat = repeat([0.0, 0.0, 1.0], 1, nat)
        mags = [1.0, 1.0]
        M = 0.9 .* zhat
        # consistent mode-1 datum passes
        d_ok = TrainingDatum(; energy = -1.0, directions = zhat, magmoms = mags,
                             moments_bare = M, constraint_axes = zhat,
                             constraint_mode = 1)
        @test SLCE.check_moment_gates([d_ok]) === nothing
        # a NEGATIVE readout with the converged direction on the negative side is
        # consistent (the sign gate tests agreement, not positivity)
        d_neg = TrainingDatum(; energy = -1.0, directions = -zhat, magmoms = mags,
                              moments_bare = -M, constraint_axes = zhat,
                              constraint_mode = 1)
        @test SLCE.check_moment_gates([d_neg]) === nothing
        # sign flip: y > 0 but the converged direction points the other way
        d_flip = TrainingDatum(; energy = -1.0, directions = -zhat, magmoms = mags,
                               moments_bare = M, constraint_axes = zhat,
                               constraint_mode = 1)
        err = try
            SLCE.check_moment_gates([d_flip])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("sign-consistency", err.msg)
        # rows below the gate floor carry no sign information and are not gated
        d_small = TrainingDatum(; energy = -1.0, directions = -zhat, magmoms = mags,
                                moments_bare = 1e-4 .* zhat, constraint_axes = zhat,
                                constraint_mode = 1)
        @test SLCE.check_moment_gates([d_small]) === nothing
        # axis-angle p99 (mode 4): a 10° stale axis is loud at the 5° default
        th = deg2rad(10.0)
        rot = [cos(th) 0.0 sin(th); 0.0 1.0 0.0; -sin(th) 0.0 cos(th)]
        d_ang = TrainingDatum(; energy = -1.0, directions = rot * zhat,
                              magmoms = mags, moments_bare = M,
                              constraint_axes = zhat, constraint_mode = 4)
        err = try
            SLCE.check_moment_gates([d_ang])
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("axis-angle", err.msg)
        @test SLCE.check_moment_gates([d_ang]; axis_angle_p99_max = 15.0) === nothing
        # the gate is a PERCENTILE: one collapse-row outlier among many clean rows
        # passes (the measured FeGe τ0.5 situation), a systematic offset does not
        many = [TrainingDatum(; energy = -1.0, directions = zhat, magmoms = mags,
                              moments_bare = M, constraint_axes = zhat,
                              constraint_mode = 4) for _ = 1:100]
        @test SLCE.check_moment_gates(vcat(many, [d_ang])) === nothing
        # the writer runs the gates: a violating set never becomes a file
        @test_throws ArgumentError write_extxyz(joinpath(tmp, "gate.extxyz"),
                                                [d_flip], xt)
        # ... and the reader re-runs them: a file corrupted after generation is
        # caught at load. Two corruptions, one gauge: per-atom columns are
        # species pos(3) mw(3) bcon(3) mint(3) mconstr(3).
        f = joinpath(tmp, "gated.extxyz")
        dm1 = TrainingDatum(; energy = -1.0, directions = zhat, magmoms = mags,
                            field = zeros(3, nat), moments_bare = M,
                            constraint_axes = zhat, constraint_mode = 1)
        write_extxyz(f, [dm1], xt)
        lines = split(read(f, String), "\n")
        # (a) flipping the WHOLE mconstr axis flips y with it — a mode-1 axis sign
        # is a gauge, and the gate must NOT fire on it (that is the physics: the
        # transverse-penalty constraint prescribes an axis, not an orientation)
        toka = split(lines[3])
        toka[16] = "-1.0"                       # mconstr z of atom 1 → axis flipped
        gauge = joinpath(tmp, "gated_gauge.extxyz")
        write(gauge, join([lines[1], lines[2], join(toka, " "), lines[4]], "\n"))
        @test length(read_extxyz(gauge)) == 1
        # (b) flipping the BARE MOMENT alone breaks the sign consistency between
        # the converged direction and the readout — loud at load
        tokb = split(lines[3])
        tokb[13] = "-0.9"                       # mint z of atom 1 → flipped
        bad = joinpath(tmp, "gated_bad.extxyz")
        write(bad, join([lines[1], lines[2], join(tokb, " "), lines[4]], "\n"))
        err = try
            read_extxyz(bad)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("sign-consistency", err.msg)
    end

    @testset "hardening: duplicates, string columns, writer free text, byte-copy pair" begin
        # [backported from SCEFitting.jl f966417] a file that says two things is
        # refused, not arbitrated; the writer only produces files its reader loads
        data = [mkdat(i) for i = 1:2]
        f = joinpath(tmp, "harden.extxyz")
        write_extxyz(f, data, xt)
        lines = split(read(f, String), "\n")
        function edited(edit)
            L = copy(lines); edit(L)
            q = joinpath(tmp, "harden_edit.extxyz"); write(q, join(L, "\n")); q
        end
        @test length(read_extxyz(edited(L -> nothing))) == 2           # the editor itself
        # duplicate info key (frame 1): last-wins Dict semantics refused
        @test_throws ArgumentError read_extxyz(edited(L -> L[2] = L[2] * " energy=0.0"))
        # duplicate property block: the second would silently overwrite the first
        @test_throws ArgumentError read_extxyz(edited(L -> begin
            L[2] = replace(L[2], ":mconstr:R:3" => ":mconstr:R:3:mw:R:3")
            L[3] = L[3] * " 0.0 0.0 0.0"; L[4] = L[4] * " 0.0 0.0 0.0"
        end))
        # a second string column would be read INTO species
        @test_throws ArgumentError read_extxyz(edited(L -> begin
            L[2] = replace(L[2], ":mconstr:R:3" => ":mconstr:R:3:tag:S:1")
            L[3] = L[3] * " x"; L[4] = L[4] * " y"
        end))
        # free-text values: whitespace is quoted so the file reloads; a quote or a
        # line break inside a value is refused up front (the lexer has no escape)
        fq = joinpath(tmp, "quoted.extxyz")
        write_extxyz(fq, data, xt; source = "C:\\Program Files\\x",
                     field_sign = "vasp +B", comment = "two words")
        @test length(read_extxyz(fq)) == 2
        @test occursin("source=\"C:\\Program Files\\x\"", read(fq, String))
        @test_throws ArgumentError write_extxyz(fq, data, xt; comment = "say \"hi\"")
        @test_throws ArgumentError write_extxyz(fq, data, xt; source = "a\nb")
        # provenance strings go through the same door
        dsp = [TrainingDatum(; energy = d.energy, directions = d.directions,
                             magmoms = d.magmoms, field = d.field,
                             moments_bare = d.moments_bare,
                             constraint_axes = d.constraint_axes,
                             constraint_mode = d.constraint_mode,
                             provenance = DatumProvenance(; setup_id = "run 7"))
               for d in data]
        write_extxyz(fq, dsp, xt)
        @test occursin("setup_id=\"run 7\"", read(fq, String))
        @test read_extxyz(fq)[1].provenance.setup_id == "run 7"
        dsq = [TrainingDatum(; energy = d.energy, directions = d.directions,
                             magmoms = d.magmoms, field = d.field,
                             moments_bare = d.moments_bare,
                             constraint_axes = d.constraint_axes,
                             constraint_mode = d.constraint_mode,
                             provenance = DatumProvenance(; setup_id = "run \"7\""))
               for d in data]
        @test_throws ArgumentError write_extxyz(fq, dsq, xt)
        # EMBSET pair: the same file twice, or a byte copy, is not an MW / M_int pair
        eb = joinpath(tmp, "EMBSET_b")
        open(eb, "w") do io
            for c = 1:3
                println(io, -1.0 - c)
                for a = 1:2
                    println(io, "$a 0.0 0.0 1.2 0.01 0.02 0.0")
                end
            end
        end
        @test_throws ArgumentError read_embset_pair(eb, eb; constraint_mode = 4)
        cp(eb, joinpath(tmp, "EMBSET_copy"); force = true)
        @test_throws ArgumentError read_embset_pair(eb, joinpath(tmp, "EMBSET_copy");
                                                    constraint_mode = 4)
    end

    @testset "EMBSET pair: loud sibling checks, energies uncompared" begin
        e1 = joinpath(tmp, "EMBSET")
        e2 = joinpath(tmp, "EMBSET_mint")
        wemb(p, mz, e0; nconf = 3, bfy = 0.02) = open(p, "w") do io
            for c = 1:nconf
                println(io, e0 - c)
                for a = 1:2
                    println(io, "$a 0.0 0.0 $mz 0.01 $bfy 0.0")
                end
            end
        end
        wemb(e1, 1.2, -1.0)
        wemb(e2, 1.1, -2.0)                     # energies differ ON PURPOSE
        pd = read_embset_pair(e1, e2; constraint_mode = 4)
        @test length(pd) == 3
        @test pd[1].magmoms[1] == 1.2           # MW file: decomposition source
        @test pd[1].moments_bare[3, 1] == 1.1   # mint file: bare target
        @test pd[1].energy == -2.0              # the MW-file energy is used
        @test pd[1].constraint_mode == 4
        # config-count mismatch (the EMBSET_mint_100 provenance-bug class)
        wemb(e2, 1.1, -2.0; nconf = 2)
        @test_throws ArgumentError read_embset_pair(e1, e2; constraint_mode = 4)
        # field-block mismatch: not siblings
        wemb(e2, 1.1, -2.0; bfy = 0.03)
        err = try
            read_embset_pair(e1, e2; constraint_mode = 4)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError && occursin("field", err.msg)
        # per-config constraint axes attach, and the gates run over them
        wemb(e2, 1.1, -2.0)
        ax = repeat([0.0, 0.0, 1.0], 1, 2)
        pd1 = read_embset_pair(e1, e2; constraint_mode = 1, constraint_axes = ax)
        @test all(d.constraint_axes == ax for d in pd1)
        axv = [copy(ax) for _ = 1:3]
        pd2 = read_embset_pair(e1, e2; constraint_mode = 1, constraint_axes = axv)
        @test all(d.constraint_axes == ax for d in pd2)
        @test_throws ArgumentError read_embset_pair(e1, e2; constraint_mode = 1,
                                                    constraint_axes = axv[1:2])
        # mode 1 without axes dies at the datum ctor (loud, not a silent fallback)
        @test_throws ArgumentError read_embset_pair(e1, e2; constraint_mode = 1)
    end

    @testset "ExtxyzFile source → SLCEDataset (E path inert to the trio)" begin
        data = [mkdat(i) for i = 1:4]
        f = joinpath(tmp, "src.extxyz")
        write_extxyz(f, data, xt)
        b = SLCEBasis(xt, BasisSpec(; nbody = 2, lmax = [1, 1], cutoff = 3.0))
        ds = SLCEDataset(b, ExtxyzFile(f); use_torque = false)
        @test length(ds.configs) == 4
        # identical to the direct-datum path
        ds2 = SLCEDataset(b, data; use_torque = false)
        @test maximum(abs, ds.X_E .- ds2.X_E) < 1e-14
        @test ds.y_E == ds2.y_E
    end
end
