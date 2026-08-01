# Periodic resolvability: which SALCs a finite reference cell can resolve at all.
#
# A SALC's value on training data is its orbit sum, and on a finite cell several
# members of one orbit can join the SAME atoms of the reference cell — the
# Wigner–Seitz boundary ties `theory/resolvability.md` describes, where one atom
# pair carries several equidistant minimum images. A `TrainingDatum` carries one
# spin and one displacement per reference-cell atom, so all tied members see
# identical arguments and the orbit sum can annihilate the invariant outright.
# Those coefficients are UNIDENTIFIABLE, not physically zero: the same basis
# functions are nonzero under `affine_energy` and in a Monte-Carlo supercell.
#
# The fixtures build their space group from HAND-WRITTEN operations (the 48 signed
# permutation matrices of m-3m, and the 16 of them that keep the z axis) rather
# than from a symmetry backend: the core test environment has no Spglib, and an
# explicit group is the better oracle anyway.
#
# Gates, none of them the classifier's own output:
#   (A) INDEPENDENT IMPLEMENTATION — the evaluator. Each SALC's orbit sum is
#       compared against the sum of its single-member values on the same
#       configuration, the scale a cancellation must be judged against.
#       `unresolvable_columns` reaches its verdict through a symbolic monomial
#       expansion that shares no code with `evaluate_salc`.
#   (B) THE PROPERTY THE DESIGN RESTS ON — freezing whole COLUMNS is only exact if
#       the canonical gauge already separates the kernel, i.e. if the basis has no
#       null COMBINATION beyond its null columns. Gated by the rank deficiency of a
#       sampled value matrix (again the evaluator, not the symbolic expansion).
#   (C) PHYSICS — unidentifiable is not zero: a classified column has a nonzero
#       ε-linear response, exactly linear in ε.
#   (D) THE REMEDY — a cell in which the pair's minimum image is unique resolves the
#       channel, and breaking the tie in only one direction does not.
#   (E) THE ASR SIDE — the differentiated expansion cancels where the undifferentiated
#       one does, so an all-residue constraint matrix must impose NOTHING. Counted
#       against the production gradient kernel (`accumulate_grad!`), and read back
#       through the public `asr_residual`, which physical consumers gate on.

using Test
using SLCE
using SLCE: salcs, SALC, SALCScratch, evaluate_salc, _signature_matrix,
    _assemble_spacegroup, _superset_cutoff, build_neighbor_list, build_clusters,
    build_salc_basis, build_asr, _asr_matrix, _asr_expansion, _asr_nullspace,
    _is_staged, accumulate_grad!, has_disp, crystal_fingerprint,
    _unresolvable_expanded, _has_boundary_tie
using LinearAlgebra
using Random
using StaticArrays

# The 48 signed permutation matrices: the full cubic point group m-3m, written out.
function _cubic_ops()
    ops = SMatrix{3,3,Float64,9}[]
    for perm in ([1, 2, 3], [1, 3, 2], [2, 1, 3], [2, 3, 1], [3, 1, 2], [3, 2, 1]),
        s1 in (1, -1), s2 in (1, -1), s3 in (1, -1)

        M = zeros(3, 3)
        sgn = (s1, s2, s3)
        for (row, col) in enumerate(perm)
            M[row, col] = sgn[row]
        end
        push!(ops, SMatrix{3,3,Float64}(M))
    end
    return ops
end
# ... and the 16 of them that keep the z axis (up to sign): tetragonal 4/mmm.
_tetragonal_ops() = [o for o in _cubic_ops() if abs(o[3, 3]) == 1]

function _basis_with_ops(cr::Crystal, spec::BasisSpec, rots)
    trans = [SVector{3,Float64}(0, 0, 0) for _ in rots]
    sg = _assemble_spacegroup(cr, rots, trans, "manual", 0; tol = 1e-6)
    nl = build_neighbor_list(cr, _superset_cutoff(spec), MinimumImage())
    cl = build_clusters(cr, nl, sg; nbody = spec.nbody, selection = MinimumImage(),
                        cutoff = spec.cutoff)
    sb = build_salc_basis(cr, sg, cl, spec; neighbors = nl, selection = MinimumImage())
    return SLCEBasis(cr, sg, sb, spec)
end

# Gate (A)/(B) source: the evaluator's own view of the basis. `V` holds each SALC's
# value over random configurations, column-scaled by the sum of |single-member|
# values so that a cancelled column sits at roundoff and a surviving one at O(1) —
# the distinction a global threshold cannot make on a basis mixing spin and
# high-degree displacement channels.
function _evaluator_view(b; nprobe = 240, seed = 2024)
    ss = salcs(b)
    p = length(ss)
    nat = n_atoms(b.crystal)
    rng = MersenneTwister(seed)
    scratch = SALCScratch()
    V = zeros(nprobe, p)
    scale = zeros(p)
    for i = 1:nprobe
        e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])
        u = 0.05 .* randn(rng, 3, nat)
        for j = 1:p
            s = ss[j]
            V[i, j] = evaluate_salc(s, e, u, scratch)
            gross = 0.0
            for mem in s.members
                gross += abs(evaluate_salc(SALC(s.key, s.body, s.decors, s.L_S, s.Lf,
                                                [mem]), e, u, scratch))
            end
            scale[j] = max(scale[j], gross)
        end
    end
    for j = 1:p
        scale[j] > 0 && (@views V[:, j] ./= scale[j])
    end
    ratio = [maximum(abs, @view V[:, j]) for j = 1:p]
    return V, ratio
end

# Gate (E) source: the translation image Σ_a ∂Φ_j/∂u_a through the PRODUCTION
# gradient kernel (`accumulate_grad!`'s `Gu` column sum), sampled at random (e, u).
# It shares no code with the symbolic ASR builder, so its rank is the
# algorithm-independent count of real constraints.
function _res_translation_image(b; nprobe = 24, seed = 7)
    ss = salcs(b)
    nat = n_atoms(b.crystal)
    rng = MersenneTwister(seed)
    B = zeros(3 * nprobe, length(ss))
    Ge = zeros(3, nat)
    Gu = zeros(3, nat)
    for t = 1:nprobe
        e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])
        u = 0.05 .* randn(rng, 3, nat)
        for j in eachindex(ss)
            fill!(Ge, 0.0)
            fill!(Gu, 0.0)
            accumulate_grad!(Ge, Gu, ss[j], e, u, 1.0)
            B[(3 * (t - 1) + 1):(3 * t), j] .= vec(sum(Gu; dims = 2))
        end
    end
    return B
end
_res_rank(B) = (s = svdvals(B); (isempty(s) || s[1] == 0.0) ? 0 :
                count(>(maximum(size(B)) * eps() * s[1]), s))

_bcc() = Crystal(Lattice(Matrix(3.0 * I(3))), [0.0 0.5; 0.0 0.5; 0.0 0.5], [1, 1], ["Fe"])
_harmonic(cr; degree = 2, cutoff = 2.7) =
    BasisSpec(cr; lmax = 0, pmax = degree,
              sectors = [Sector(disp = (degree = degree,), sites = 2, cutoff = cutoff)])

function _fixtures()
    bcc = _bcc()
    z2 = Crystal(Lattice([3.0 0 0; 0 3.0 0; 0 0 6.0]),
                 [0.0 0.5 0.0 0.5; 0.0 0.5 0.0 0.5; 0.0 0.25 0.5 0.75],
                 [1, 1, 1, 1], ["Fe"])
    sc = Crystal(Lattice(Matrix(6.0 * I(3))),
                 reduce(hcat, [[x, y, z] ./ 2 .+ o
                               for o in ([0.0, 0, 0], [0.25, 0.25, 0.25])
                               for x = 0:1, y = 0:1, z = 0:1]),
                 fill(1, 16), ["Fe"])
    return [
        ("bcc harmonic", _basis_with_ops(bcc, _harmonic(bcc), _cubic_ops())),
        ("bcc degree 3", _basis_with_ops(bcc, _harmonic(bcc; degree = 3), _cubic_ops())),
        ("bcc pure spin soc",
         _basis_with_ops(bcc, BasisSpec(bcc; lmax = 2, pmax = 0, soc = true,
                                       sectors = [Sector(spin = (sites = 1:2,),
                                                         cutoff = 2.7)]), _cubic_ops())),
        ("bcc spin x disp",
         _basis_with_ops(bcc, BasisSpec(bcc; lmax = 1, pmax = 2,
                                       sectors = [Sector(spin = [1, 1],
                                                         disp = (degree = 1,), sites = 2,
                                                         cutoff = 2.7)]), _cubic_ops())),
        ("bcc doubled along z only",
         _basis_with_ops(z2, _harmonic(z2), _tetragonal_ops())),
        ("bcc as 2x2x2", _basis_with_ops(sc, _harmonic(sc), _cubic_ops())),
    ]
end

@testset "periodic resolvability" begin
    fixtures = _fixtures()

    @testset "(A) symbolic classification == the evaluator's verdict" begin
        for (name, b) in fixtures
            _, ratio = _evaluator_view(b)
            num = [j for j in eachindex(ratio) if ratio[j] <= 1e-10]
            @test unresolvable_columns(b) == num
            # Both routes through the classifier must agree with it: the public one,
            # which short-circuits on the cheap "is any orbit tied at all?" pre-check,
            # and the expansion it guards. A tie is NECESSARY, so the short-circuit is
            # exact — and this is where that claim is checked, on tied AND untied
            # fixtures (the untied one is the case the pre-check answers alone).
            @test _unresolvable_expanded(b) == num
            @test _has_boundary_tie(b) || isempty(num)
            # A cancelled column is at roundoff and a surviving one is nowhere near
            # the cut: the criterion is well posed here, not threshold-tuned.
            for j in eachindex(ratio)
                @test ratio[j] <= 1e-12 || ratio[j] >= 1e-3
            end
            S, gross = _signature_matrix(b)
            for j in eachindex(gross)
                gross[j] == 0.0 && continue
                r = norm(@view S[:, j]) / gross[j]
                @test r <= 1e-12 || r >= 1e-3
            end
        end
    end

    @testset "(B) no null combination beyond the null columns" begin
        # Freezing whole columns is exact only if the kernel is spanned by columns.
        # Measured through the sampled value matrix, so the gate does not consult the
        # symbolic expansion the classifier uses.
        for (name, b) in fixtures
            V, _ = _evaluator_view(b)
            p = n_salcs(b)
            sv = svdvals(V)
            rank = count(>(1e-10), sv)
            @test p - rank == length(unresolvable_columns(b))
            # The same property as the classifier reports it. These fixtures are face
            # (a) only, which is exactly the case `_unresolvable_split` used to skip —
            # so this assertion was passing on an untouched initial value and asserting
            # nothing. It is a real check only because the computation is no longer
            # conditioned on face (b) having fired.
            @test SLCE._unresolvable_split(b).residual_flat == 0
        end
    end

    @testset "(C) unidentifiable, not zero — a uniform strain sees it" begin
        b = _basis_with_ops(_bcc(), BasisSpec(_bcc(); lmax = 1, pmax = 2,
                                             sectors = [Sector(spin = [1, 1],
                                                               disp = (degree = 1,),
                                                               sites = 2, cutoff = 2.7)]),
                            _cubic_ops())
        dead = unresolvable_columns(b)
        @test !isempty(dead)
        rng = MersenneTwister(9)
        e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:n_atoms(b.crystal)])
        nonzero = 0
        for j in dead
            m = SLCEModel(b, 0.0, [i == j ? 1.0 : 0.0 for i = 1:n_salcs(b)])
            a1 = affine_energy(m, e, [0 0 0; 0 0 0; 0 0 0.01])
            a2 = affine_energy(m, e, [0 0 0; 0 0 0; 0 0 0.02])
            if abs(a1) > 1e-8
                nonzero += 1
                @test a2 ≈ 2 * a1 rtol = 1e-10
            end
        end
        @test nonzero > 0
    end

    @testset "(D) the remedy, and that one direction is not enough" begin
        byname = Dict(fixtures)
        # bcc's cross pair sits at all eight WS corners of the cubic cell. Doubling
        # along z alone leaves a four-fold tie, so the channel stays unresolvable;
        # 2x2x2 puts the pair strictly inside the WS cell and resolves it.
        @test !isempty(unresolvable_columns(byname["bcc harmonic"]))
        @test !isempty(unresolvable_columns(byname["bcc doubled along z only"]))
        @test isempty(unresolvable_columns(byname["bcc as 2x2x2"]))
    end

    @testset "(E) cancellation residue never becomes a constraint" begin
        # The differentiated expansion cancels wherever the undifferentiated one does,
        # so a tied cell can hand `build_asr` a constraint matrix that is residue from
        # end to end. Judged against its own maximum that matrix looks full-strength,
        # and the row normalization then promotes BLAS rounding to unit constraints.
        # Judged per entry against its own gross accumulation it is what it is: nothing.
        for (name, b) in fixtures
            any(s -> any(has_disp, s.key.decors), salcs(b)) || continue
            rep = build_asr(b; warn = false)
            # The number of real constraints, counted through the production gradient
            # kernel. Only over the FREE columns: a frozen column's kernel value is
            # roundoff, and a rank count cut relative to the matrix's own maximum reads
            # those as real (measured on the two all-frozen fixtures below: rank 8 and
            # 5 over all columns, on bases whose every column is identically zero).
            # That is the same blindness, one layer out.
            B = _res_translation_image(b)
            @test rep.rank == _res_rank(B[:, rep.free])
        end
        byname = Dict(fixtures)
        for name in ("bcc degree 3", "bcc spin x disp")
            b = byname[name]
            @test unresolvable_columns(b) == collect(1:n_salcs(b))
            A0, G = _asr_expansion(b)
            @test !isempty(A0)                             # the expansion DID deposit
            @test maximum(abs, A0) <= 1e-12 * maximum(G)   # and every deposit cancelled
            @test isempty(_asr_matrix(b))                  # so there is no constraint
            @test build_asr(b; warn = false).rank == 0
            # Every column is identically zero on this cell, so a hand-built model's
            # energy is too, and it is trivially translation invariant. The public
            # verifier must agree: under the cut against `A`'s own maximum it read
            # 0.24 (degree 3) and 0.36 (spin × disp) instead, and the physical
            # consumers — force constants, dynamical matrix, strain — gate on it.
            rng = MersenneTwister(11)
            nat = n_atoms(b.crystal)
            m = SLCEModel(b, 0.0, randn(rng, n_salcs(b)))
            e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])
            u = 0.05 .* randn(rng, 3, nat)
            @test abs(predict_energy(m, e, u)) < 1e-14
            @test abs(predict_energy(m, e, u .+ [0.013, -0.007, 0.021]) -
                      predict_energy(m, e, u)) < 1e-14
            @test asr_residual(m) == 0.0
        end
    end

    @testset "(F) the physical readouts name the frozen channel" begin
        # The freeze is silent in the fit and LOUD in the readouts: they differentiate
        # the individual cluster members, where the tie does not cancel. This fixture
        # has both kinds of column (6 frozen, 4 free), so a model can be built the way
        # a fit on this cell would return one.
        b = Dict(fixtures)["bcc doubled along z only"]
        frozen = unresolvable_columns(b)
        free = setdiff(1:n_salcs(b), frozen)
        @test !isempty(frozen) && !isempty(free)
        rng = MersenneTwister(3)
        beta = zeros(n_salcs(b))
        beta[free] .= 0.05 .* randn(rng, length(free))
        fitted = SLCEModel(b, 0.0, beta)
        # (i) held at zero: the deliverable is MISSING those channels, and they are not
        # physically zero. Every readout that reads the monomials says so.
        @test_logs (:warn, r"cannot resolve") match_mode = :any force_constants(fitted)
        @test_logs (:warn, r"cannot resolve") match_mode = :any decorated_terms(fitted)
        # (ii) a value supplied from outside the fit — legal, and unvalidated.
        beta2 = copy(beta)
        beta2[frozen[1]] = 0.3
        supplied = SLCEModel(b, 0.0, beta2)
        @test_logs (:warn, r"NONZERO coefficient") match_mode = :any force_constants(supplied)
        # ...and the physics that warning claims, measured through two paths that share
        # no code: the ENERGY on cell-periodic configurations cannot tell the two models
        # apart (that is what "unidentifiable" means), while the force constants can —
        # and precisely at q ≠ 0, since Σ_R Φ(R) is the Hessian of that same energy.
        nat = n_atoms(b.crystal)
        for _ = 1:5
            e = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])
            u = 0.03 .* randn(rng, 3, nat)
            @test abs(predict_energy(supplied, e, u) - predict_energy(fitted, e, u)) < 1e-12
        end
        D0a = dynamical_matrix(force_constants(fitted; order = 2), [0.0, 0.0, 0.0])
        D0b = dynamical_matrix(force_constants(supplied; order = 2), [0.0, 0.0, 0.0])
        Dqa = dynamical_matrix(force_constants(fitted; order = 2), [0.3, 0.1, 0.25])
        Dqb = dynamical_matrix(force_constants(supplied; order = 2), [0.3, 0.1, 0.25])
        @test norm(D0b - D0a) <= 1e-12 * max(norm(D0a), 1.0)
        @test norm(Dqb - Dqa) > 1e-3 * max(norm(Dqa), 1.0)
    end

    @testset "the freeze is wired into the reparameterization" begin
        byname = Dict(fixtures)
        b = byname["bcc harmonic"]
        frozen = unresolvable_columns(b)
        @test frozen == [2]                     # 1 of 2, agreed with the evaluator above
        # Two diagnostics, in this order: the freeze names its own columns and its own
        # remedy, and only then does the ASR speak about what is left. The second one
        # is the LOUD case (`rank == ndisp` over the free columns): with the pair's
        # even-Lf column frozen, the surviving harmonic column has no translation-
        # invariant content, so this basis can express no lattice coupling at all.
        rep = @test_logs((:warn, r"cannot resolve"), (:warn, r"no translation-invariant"),
                         match_mode = :any, build_asr(b))
        @test rep.free == [1]
        @test size(rep.Z, 2) == 0               # no free parameter survives
        # Before the freeze this basis got `Z = [0; 1]`: the only ASR-feasible model was
        # a multiple of a function that is identically zero, reported through the mild
        # "some columns are structurally zeroed" message. Whatever else changes, a
        # feasible model on this basis must not be able to carry the frozen column.
        @test all(norm(@view rep.Z[j, :]) == 0.0 for j in frozen)
        @test all(rep.beta_p[j] == 0.0 for j in frozen)
    end

    @testset "no unresolvable column => the reparameterization is unchanged" begin
        # Byte-neutrality: where nothing is frozen, `build_asr` must reproduce exactly
        # what it produced before the freeze existed — the null space of the whole
        # row-normalized constraint matrix, with every column free.
        b = Dict(fixtures)["bcc as 2x2x2"]
        @test isempty(unresolvable_columns(b))
        A = _asr_matrix(b)
        An = A ./ [norm(@view A[r, :]) for r in axes(A, 1)]
        Z0, rank0 = _asr_nullspace(An)
        rep = build_asr(b; warn = false)
        @test rep.free == collect(1:n_salcs(b))
        @test rep.rank == rank0
        @test rep.Z == Z0                       # bitwise
        @test rep.A == An
    end

    # The three blind spots a review found: every one of these passed before the fixes
    # and each names a path that reached a frozen column with the freeze switched off.
    @testset "the freeze survives every fit path" begin
        bcc = _bcc()
        # (a) a PURE-SPIN basis. `build_asr` used to return `nothing` for one
        # unconditionally, and the pure-spin `SLCEDataset` constructors hard-coded
        # `asr = nothing`, so a pure-spin cell's frozen columns were fitted freely — and
        # came back at ~1e-18, which is NOT the exact zero `select_support`'s alive rule
        # and SLCEMonteCarlo's term prune both test for.
        bs = _basis_with_ops(bcc, BasisSpec(bcc; lmax = 2, pmax = 0, soc = true,
                                            sectors = [Sector(spin = (sites = 1:2,),
                                                              cutoff = 2.7)]),
                             _cubic_ops())
        sfrozen = unresolvable_columns(bs)
        @test !isempty(sfrozen)
        rng = MersenneTwister(5)
        st = SLCEModel(bs, 0.0, 0.1 .* randn(rng, n_salcs(bs)))
        configs = [reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:2]) for _ = 1:150]
        dss = SLCEDataset(bs, configs, [predict_energy(st, e) for e in configs])
        @test dss.asr !== nothing
        @test dss.asr.free == setdiff(1:n_salcs(bs), sfrozen)
        # `asr = true` and `asr = false` must agree: the freeze is a property of the
        # basis, not of the constraint. Before the fix they disagreed in both directions
        # — exact zeros only under `asr = false`, and `dof` 8 versus 4.
        for flag in (true, false)
            f = fit(SLCEFit, dss, OLS(); asr = flag)
            for j in sfrozen
                @test coef(f)[j] === 0.0
            end
            @test dof(f) == length(dss.asr.free) + 1     # + the analytic intercept
        end

        # (b) a STAGED fit. `sector_columns` resolves selectors from key content alone,
        # so a mask naming a frozen column used to hand it a free null-space direction on
        # an identically-zero design column — an unbounded solve, measured at 3.7e14 in
        # `jphi` with `asr_residual` still reporting 0.0 and the junk reaching
        # `force_constants`.
        jspec = BasisSpec(bcc; lmax = 1, pmax = 2, sectors = [
            Sector(spin = [1, 1], sites = 2, cutoff = 2.7),
            Sector(disp = (degree = 2,), sites = 2, cutoff = 2.7)])
        bj = _basis_with_ops(bcc, jspec, _cubic_ops())
        jfrozen = unresolvable_columns(bj)
        @test !isempty(jfrozen)
        rng2 = MersenneTwister(4)
        tj = SLCEModel(bj, 0.0, 0.1 .* randn(rng2, n_salcs(bj)))
        fp = crystal_fingerprint(bcc)
        data = [begin
                    e = reduce(hcat, [normalize(randn(rng2, 3)) for _ = 1:2])
                    u = 0.05 .* randn(rng2, 3, 2)
                    TrainingDatum(; energy = predict_energy(tj, e, u), directions = e,
                                  magmoms = ones(2), displacements = u,
                                  forces = predict_force(tj, e, u),
                                  provenance = DatumProvenance(;
                                      reference_id = "r", reference_fingerprint = fp))
                end for _ = 1:120]
        dsj = SLCEDataset(bj, data; use_torque = false)
        f1 = fit(SLCEFit, dsj, OLS(); sector_mask = :spin, force_weight = 0.0)
        f2 = fit(SLCEFit, dsj, OLS(); sector_mask = :lattice, frozen = SLCEModel(f1),
                 force_weight = 0.5)
        for j in jfrozen
            @test coef(f2)[j] === 0.0
            @test norm(@view f2.reparam.Z[j, :]) == 0.0
        end
        # an explicit mask that names a frozen column says so rather than fitting it
        @test_logs((:warn, r"unresolvable on this reference cell"), match_mode = :any,
                   fit(SLCEFit, dsj, OLS(); sector_mask = collect(1:n_salcs(bj)),
                       force_weight = 0.5))

        # (c) `asr = false` must not be mistaken for a STAGED fit. The freeze-only
        # reparameterization is a fresh object, so the old `reparam !== dataset.asr`
        # identity test read "staged" and `select_support` refused an ordinary ablation
        # fit with a message about staging that was simply false.
        fabl = fit(SLCEFit, dss, OLS(); asr = false)
        @test !_is_staged(fabl)
        @test _is_staged(f2)
        @test select_support(fabl; npoints = 3) isa SupportPath

        # (d) WHICH COORDINATES the dead-column warning reports its indices in. A
        # freeze-only reparameterization has no constraint rows, so `Z` is exactly the
        # selection matrix of `free` and γ direction k IS `jphi` column `free[k]` — while
        # under the real constraint the two spaces are genuinely different and only the γ
        # statement is true. Reporting a γ index in the first case hands the caller a
        # number that indexes nothing they hold. The oracle for the expected set is
        # structural, from the KEYS: a force-only fit sees no pure-spin column.
        purespin = [j for (j, s) in enumerate(salcs(bj)) if !any(has_disp, s.key.decors)]
        expected = setdiff(purespin, jfrozen)
        @test !isempty(expected)
        function _deadrec(flag)
            lg = Test.TestLogger()
            Base.CoreLogging.with_logger(lg) do
                fit(SLCEFit, dsj, OLS(); asr = flag, force_weight = 1.0)
            end
            recs = [r for r in lg.logs if occursin("carry no information", r.message)]
            @test length(recs) == 1
            return recs[1]
        end
        rb = _deadrec(false)
        @test rb.kwargs[:coordinates] == "jphi (β)"
        @test rb.kwargs[:columns] ⊆ expected
        @test occursin("$(length(expected)) design column", rb.message)
        @test occursin("refit", rb.message)              # β-space advice, applicable
        rg = _deadrec(true)
        @test rg.kwargs[:coordinates] == "reparameterized (γ)"
        @test occursin("null-space basis", rg.message)   # and NOT the β-space advice
        @test !occursin("refit", rg.message)
        # This fixture's ASR branch is also the ALL-ZERO design: its single feasible
        # direction is pure spin, hence invisible to forces, so EVERY column is dead —
        # the case `_warn_unidentified` used to return from without a word, i.e. the fit
        # that determined nothing was the one that said nothing.
        qdim = size(fit(SLCEFit, dsj, OLS(); force_weight = 1.0).reparam.Z, 2)
        @test rg.kwargs[:columns] == collect(1:qdim)
    end

    # (G) THE UNFUSED TIE. Everything above is the case where the point group permutes
    # the tied images, so they share one orbit and the odd content cancels — a null
    # COLUMN. In low symmetry the same tie puts them in DIFFERENT orbits with
    # independent couplings: every column is nonzero and a null COMBINATION appears
    # instead. The old within-SALC pre-check could not see that (asserted below as the
    # kill-shot), so nine flat directions reached `dynamical_matrix(q ≠ 0)` with a 52 %
    # error while every gate was green.
    #
    # Oracles, none of them the classifier: the geometric tie is computed from the
    # lattice; which columns SHOULD go is derived from the SALC keys; the "no data can
    # see it" claim is the production evaluator; the "the readouts can" claim is
    # `dynamical_matrix`; and the recovery claim is a synthetic ground truth.
    @testset "(G) a tie symmetry does not fuse drops the whole interaction" begin
        lat = Lattice(Matrix(3.0 * I(3)))
        # P1 (identity only): atoms 1,2 differ by EXACTLY 0.5 in fractional x.
        cr = Crystal(lat, [0.0 0.5 0.25; 0.0 0.2 0.3; 0.0 0.1 0.42], [1, 1, 1], ["Fe"])
        spec = BasisSpec(cr; lmax = 1, pmax = 2,
                         sectors = [Sector(disp = (degree = 1:2,), sites = 1:2,
                                           cutoff = 2.6)])
        b = _basis_with_ops(cr, spec, [SMatrix{3,3,Float64}(Matrix(1.0 * I(3)))])
        p = n_salcs(b)

        # The fixture really is tied, from the lattice alone: the two images of the
        # (1,2) pair are equidistant. If this ever stops holding the gate is vacuous.
        A = cr.lattice.vectors
        d(R) = norm(A * ([0.5, 0.2, 0.1] .+ R))
        @test d([0, 0, 0]) ≈ d([-1, 0, 0])
        @test d([0, 0, 0]) < 2.6 && d([0, 1, 0]) > 2.6

        # ... and the two tied images land in TWO orbits, while no single SALC repeats
        # an atom multiset. That second half is the kill-shot: it is exactly the
        # condition the within-SALC scan tests, so a pre-check that looks only inside
        # one SALC returns `false` here and the freeze silently stops working.
        orbs = unique([(s.key.body, s.key.orbit_id) for s in salcs(b)
                       for m in s.members if sort(m.atoms) == [1, 2]])
        @test length(orbs) == 2
        @test all(s -> allunique([sort(m.atoms) for m in s.members]), salcs(b))
        @test _has_boundary_tie(b)

        # Which columns go is derived from the keys, not read back from the classifier.
        want = [j for (j, s) in enumerate(salcs(b))
                if (s.key.body, s.key.orbit_id) in orbs]
        frozen = unresolvable_columns(b)
        @test frozen == want
        @test !isempty(frozen) && length(frozen) < p
        # Face (b), not face (a): each dropped column is individually NONZERO, so the
        # null-column test alone reports nothing at all.
        @test isempty(_unresolvable_expanded(b))
        split = SLCE._unresolvable_split(b)
        @test isempty(split.vanishing) && split.undetermined == want
        @test split.multisets == [[1, 2]]

        # Nothing flat is left behind: the structural expansion over the kept columns
        # has full column rank (the same statement gate (B) makes, by sampling).
        S, = _signature_matrix(b)
        kept = setdiff(1:p, frozen)
        sv = svdvals(S[:, kept])
        @test count(>(1e-9 * sv[1]), sv) == length(kept)
        @test split.residual_flat == 0

        # WHAT the drop costs, exactly. The frozen block splits in half: the SUMS are
        # determined and the DIFFERENCES are not. For a two-fold tie each channel of one
        # orbit pairs with exactly one channel of the other, so the undetermined
        # subspace is half the block — that ratio is the oracle, not a pinned 9.
        Sf = S[:, frozen]
        svf = svdvals(Sf)
        rank_f = count(>(1e-9 * svf[1]), svf)
        @test rank_f == length(frozen) ÷ 2
        @test length(frozen) - rank_f == length(frozen) ÷ 2

        # The DIFFERENCE directions are what no cell-periodic observable can see — the
        # production evaluator says so — while the read-outs do see them, which is why
        # the split had to be refused rather than chosen. (Arbitrary values on the frozen
        # columns are visible; that is the determined half, and dropping it is the
        # deliberate price.)
        rng = MersenneTwister(0x0b)
        N = nullspace(Sf; rtol = 1e-9)
        @test size(N, 2) == length(frozen) - rank_f
        v = zeros(p)
        v[frozen] .= N * randn(rng, size(N, 2))
        m0 = SLCEModel(b, 0.0, zeros(p))
        m1 = SLCEModel(b, 0.0, 0.7 .* v)
        for _ = 1:5
            u = 0.03 .* randn(rng, 3, 3)
            @test abs(predict_energy(m1, nothing, u) -
                      predict_energy(m0, nothing, u)) < 1e-14
            @test norm(predict_force(m1, nothing, u) -
                       predict_force(m0, nothing, u)) < 1e-13
        end
        mass = fill(55.845, 3)
        Dof(m, q) = dynamical_matrix(force_constants(m; order = 2), q; masses = mass)
        @test norm(Dof(m1, [0.0, 0.0, 0.0]) - Dof(m0, [0.0, 0.0, 0.0])) < 1e-12
        @test norm(Dof(m1, [0.3, 0.1, 0.2]) - Dof(m0, [0.3, 0.1, 0.2])) > 1e-3

        # The fit is now identified: a truth drawn from the RETAINED span comes back
        # exactly, and `identifiability` reports no flat direction.
        rep = build_asr(b; warn = false)
        truth = SLCEModel(b, 0.0, rep.beta_p .+ rep.Z * randn(rng, size(rep.Z, 2)))
        fp = crystal_fingerprint(cr)
        mkdata(model) = [begin
                             u = 0.03 .* randn(rng, 3, 3)
                             lattice_datum(predict_energy(model, nothing, u);
                                           displacements = u,
                                           forces = predict_force(model, nothing, u),
                                           reference = cr)
                         end for _ = 1:120]
        f = fit(SLCEFit, SLCEDataset(b, mkdata(truth)), OLS(); force_weight = 0.4)
        @test rmse_energy(f) < 1e-12
        @test maximum(abs, f.jphi .- truth.jphi) < 1e-10
        @test identifiability(f).nullity == 0
        @test all(==(0.0), f.jphi[frozen])          # exactly zero, for the MC prune

        # And the price is paid LOUDLY: data that contains the dropped interaction can
        # no longer be fitted exactly. A silent split is what this replaces.
        touched = SLCEModel(b, 0.0, randn(MersenneTwister(3), p))
        @test norm(touched.jphi[frozen]) > 0.1      # the fixture must exercise it
        f2 = fit(SLCEFit, SLCEDataset(b, mkdata(touched)), OLS(); force_weight = 0.4)
        @test r2_energy(f2) < 0.99
    end

    # Gate (H). The two faces in ONE basis. Every other fixture here has exactly one of
    # them — the cubic ones are face (a) only, gate (G) is face (b) only — and that split
    # left `_unresolvable_split`'s `!(j in vset)` guard, the single line deciding what
    # happens where the faces intersect, untested: with face (a) alone nothing reads
    # `undetermined`, and with face (b) alone `vanishing` is empty, so deleting the guard
    # (or reversing the precedence) failed no assertion.
    #
    # The fixture makes the group fuse a tie only PARTIALLY. The two atoms differ by half
    # a cell in BOTH x and y, a four-fold tie; `m_y` is a real symmetry of the positions
    # and permutes the y images (face a), while nothing swaps the x images (face b).
    @testset "(H) both faces of a tie in one basis" begin
        cr = Crystal(Lattice(Matrix(Diagonal([3.0, 3.0, 6.0]))),
                     [0.1 0.6; 0.0 0.5; 0.03 0.28], [1, 1], ["Fe"])
        my = SMatrix{3,3,Float64}(Diagonal([1.0, -1.0, 1.0]))
        spec = BasisSpec(cr; lmax = 0, pmax = 2,
                         sectors = [Sector(disp = (degree = 1:2,), sites = 1:2,
                                           cutoff = 2.7)])
        b = _basis_with_ops(cr, spec, [SMatrix{3,3,Float64}(Matrix(1.0 * I(3))), my])
        p = n_salcs(b)
        split = SLCE._unresolvable_split(b)

        # The fixture must exercise BOTH, or every assertion below is vacuous.
        @test !isempty(split.vanishing)
        @test !isempty(split.undetermined)
        # ... and the guard's actual content: a column is classified once. Dropping
        # `!(j in vset)` puts the vanishing columns into `undetermined` as well, which
        # this pair of assertions is the only thing in the suite to notice.
        @test isempty(intersect(split.vanishing, split.undetermined))
        @test allunique(split.columns)
        @test split.columns == sort(vcat(split.vanishing, split.undetermined))

        # The evaluator, not the classifier, says how much is really flat. The freeze is
        # deliberately COARSER than that (whole orbits go), and it must still leave
        # nothing flat behind — the property gate (B) states, on a basis with both faces.
        V, = _evaluator_view(b; nprobe = 400, seed = 4)
        sv = svdvals(V)
        nullity = p - count(>(1e-10 * sv[1]), sv)
        @test 0 < nullity < length(split.columns)
        kept = setdiff(1:p, split.columns)
        svk = svdvals(V[:, kept])
        @test count(>(1e-10 * svk[1]), svk) == length(kept)
        @test split.residual_flat == 0
    end

    # Gate (I). Face (b) at N = 3 — the same phenomenon on a THREE-body atom multiset,
    # which no other fixture reaches (every other one truncates at pairs). The classifier
    # is body-agnostic by construction, and this is what says so out loud.
    @testset "(I) face (b) on a three-body cluster" begin
        # P1, and TWO tied edges: (1,2) and (2,3) are each separated by exactly a/2 in x.
        cr = Crystal(Lattice(Matrix(3.0 * I(3))),
                     [0.0 0.5 0.0; 0.0 0.13 0.31; 0.0 0.07 0.44], [1, 1, 1], ["Fe"])
        spec = BasisSpec(cr; lmax = 0, pmax = 3, nbody = 3,
                         sectors = [Sector(disp = (degree = 3,), sites = 1:3,
                                           cutoff = 2.2)])
        b = _basis_with_ops(cr, spec, [SMatrix{3,3,Float64}(Matrix(1.0 * I(3)))])
        p = n_salcs(b)
        split = SLCE._unresolvable_split(b)

        # The tie is geometric, computed from the lattice alone.
        A = cr.lattice.vectors
        @test norm(A * [0.5, 0.13, 0.07]) ≈ norm(A * [-0.5, 0.13, 0.07])

        # It is genuinely a THREE-body statement: a 3-atom multiset is reached by more
        # than one orbit, and 3-body columns are among those frozen. Without these two
        # the testset would be an expensive restatement of gate (G).
        @test [1, 2, 3] in split.multisets
        @test 3 in [salcs(b)[j].key.body for j in split.columns]
        @test isempty(split.vanishing)          # low symmetry: face (b), never face (a)

        # Which columns go is derived from the KEYS, as in gate (G) — not read back.
        orbs = Set((salcs(b)[j].key.body, salcs(b)[j].key.orbit_id) for j in split.columns)
        @test split.columns == [j for (j, s) in enumerate(salcs(b))
                                if (s.key.body, s.key.orbit_id) in orbs]

        # The evaluator's verdict: really flat, and strictly less than what was frozen
        # (the freeze drops determined content on purpose), with nothing left over.
        V, = _evaluator_view(b; nprobe = 3 * p, seed = 11)
        sv = svdvals(V)
        nullity = p - count(>(1e-10 * sv[1]), sv)
        @test 0 < nullity < length(split.columns)
        kept = setdiff(1:p, split.columns)
        svk = svdvals(V[:, kept])
        @test count(>(1e-10 * svk[1]), svk) == length(kept)
        @test split.residual_flat == 0

        # End to end, as at N = 2: a truth on the retained span comes back exactly and
        # the frozen coefficients are exactly zero, while data containing the dropped
        # interaction fails loudly instead of being split silently.
        rng = MersenneTwister(5)
        rep = build_asr(b; warn = false)
        truth = SLCEModel(b, 0.0, rep.beta_p .+ rep.Z * randn(rng, size(rep.Z, 2)))
        mkdata(model) = [begin
                             u = 0.03 .* randn(rng, 3, 3)
                             lattice_datum(predict_energy(model, nothing, u);
                                           displacements = u,
                                           forces = predict_force(model, nothing, u),
                                           reference = cr)
                         end for _ = 1:150]
        f = fit(SLCEFit, SLCEDataset(b, mkdata(truth)), OLS(); force_weight = 0.4)
        @test rmse_energy(f) < 1e-12
        @test maximum(abs, f.jphi .- truth.jphi) < 1e-10
        @test identifiability(f).nullity == 0
        @test all(==(0.0), f.jphi[split.columns])
        touched = SLCEModel(b, 0.0, randn(MersenneTwister(3), p))
        @test norm(touched.jphi[split.columns]) > 0.1
        @test r2_energy(fit(SLCEFit, SLCEDataset(b, mkdata(touched)), OLS();
                            force_weight = 0.4)) < 0.99
    end

    # Gate (J). The SCOPE of the freeze at N >= 3, which is a real limit and is easy to
    # read as a bug. The compact-cluster criterion admits a cluster only when all C(N,2)
    # edges are simultaneously minimum-image, so a tie's NON-congruent sibling (same
    # atoms, but one of its other edges lands on a longer shell) is rejected there and
    # the tie leaves no trace at all — no tie reported, nothing frozen. That is aliasing
    # and not indeterminacy, and the oracle for "nothing is lost" is built here from
    # scratch: a 3-body degree-(1,1,1) energy is a trilinear form in the three atoms'
    # displacements, so the whole function space is spanned by the 27 monomials
    # u_{1a} u_{2b} u_{3c}, written out independently of the package.
    @testset "(J) at N >= 3 a non-congruent sibling is aliased, not frozen" begin
        cr = Crystal(Lattice(Matrix(3.0 * I(3))),
                     [0.0 0.5 0.25; 0.0 0.2 0.3; 0.0 0.1 0.42], [1, 1, 1], ["Fe"])
        spec = BasisSpec(cr; lmax = 0, pmax = 3, nbody = 3,
                         sectors = [Sector(disp = (degree = 3,), sites = 3,
                                           cutoff = 2.6)])
        b = _basis_with_ops(cr, spec, [SMatrix{3,3,Float64}(Matrix(1.0 * I(3)))])
        p = n_salcs(b)

        # The (1,2) edge IS tied, from the lattice alone — the pre-check's silence below
        # is a statement about clusters, not about the geometry.
        A = cr.lattice.vectors
        @test norm(A * [0.5, 0.2, 0.1]) ≈ norm(A * [-0.5, 0.2, 0.1])
        @test !_has_boundary_tie(b)
        @test isempty(unresolvable_columns(b))

        # Nothing is lost: the basis spans exactly the full trilinear space.
        rng = MersenneTwister(23)
        U = [0.05 .* randn(rng, 3, 3) for _ = 1:400]
        scratch = SALCScratch()
        X = [evaluate_salc(salcs(b)[j], zeros(3, 3), u, scratch)
             for u in U, j = 1:p]
        T = zeros(length(U), 27)
        for a = 1:3, c = 1:3, d = 1:3
            col = 9(a - 1) + 3(c - 1) + d
            for (i, u) in enumerate(U)
                T[i, col] = u[a, 1] * u[c, 2] * u[d, 3]
            end
        end
        rk(M) = (s = svdvals(M); count(>(1e-10 * s[1]), s))
        @test rk(T) == 27
        @test rk(X) == p
        @test rk(hcat(X, T)) == 27          # identical span: every coupling representable
    end

    @testset "the classification is structural, not sampled" begin
        b = _basis_with_ops(_bcc(), _harmonic(_bcc()), _cubic_ops())
        first = unresolvable_columns(b)
        @test !isempty(first)
        rand(MersenneTwister(1), 100)          # ambient RNG state must not matter
        Random.seed!(12345)
        @test unresolvable_columns(b) == first
        @test unresolvable_columns(b) == first
    end
end
