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
        cfgs = [reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:2]) for _ = 1:150]
        dss = SLCEDataset(bs, cfgs, [predict_energy(st, e) for e in cfgs])
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
