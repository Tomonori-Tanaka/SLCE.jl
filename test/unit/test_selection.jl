using Test
using SLCE
using SLCE: solve_coefficients, salc_groups, group_costs, cost_weights, with_lambda
using LinearAlgebra
using Statistics
using Random
import Tables

# A centered synthetic regression fixture: X column-centered, y centered, so the
# estimator contract (`solve_coefficients` receives a centered problem) holds.
function _centered_problem(rng, n, btrue; noise = 0.01)
    p = length(btrue)
    X = randn(rng, n, p)
    X .-= mean(X; dims = 1)
    y = X * btrue .+ noise .* randn(rng, n)
    y .-= mean(y)
    return X, y
end

@testset "cost-weighted group selection" begin
    @testset "GroupAdaptiveRidge construction and validation" begin
        estimator = GroupAdaptiveRidge([1, 1, 2], [1.0, 2.0]; lambda = 0.5)
        @test estimator.lambda === 0.5
        @test estimator.column_groups == [1, 1, 2]
        @test estimator.group_weights == [1.0, 2.0]
        @test estimator.group_sizes == [2, 1]
        @test estimator.epsilon === 1e-8 && estimator.max_iter === 50 && estimator.tol === 1e-6
        @test SLCE.islinear(estimator)

        # invalid scalars
        @test_throws ArgumentError GroupAdaptiveRidge([1], [1.0]; lambda = -1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1], [1.0]; lambda = Inf)
        @test_throws ArgumentError GroupAdaptiveRidge([1], [1.0];
                                                      lambda = 1.0, epsilon = 0.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1], [1.0];
                                                      lambda = 1.0, max_iter = 0)
        @test_throws ArgumentError GroupAdaptiveRidge([1], [1.0]; lambda = 1.0, tol = 0.0)
        # invalid labels / weights
        @test_throws ArgumentError GroupAdaptiveRidge(Int[], Float64[]; lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1, 3, 3], [1.0, 1.0, 1.0];
                                                      lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([0, 1], [1.0]; lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1, 2], [1.0]; lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1, 2], [1.0, -1.0]; lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1, 2], [1.0, 0.0]; lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge([1, 2], [1.0, NaN]; lambda = 1.0)

        # construction copies: caller mutation cannot leak in
        cg = [1, 1, 2]
        wv = [1.0, 2.0]
        est2 = GroupAdaptiveRidge(cg, wv; lambda = 1.0)
        cg[1] = 99
        wv[1] = -5.0
        @test est2.column_groups == [1, 1, 2]
        @test est2.group_weights == [1.0, 2.0]

        # label length must match the design-matrix column count
        rng = MersenneTwister(1)
        X, y = _centered_problem(rng, 20, zeros(4))
        @test_throws DimensionMismatch solve_coefficients(estimator, X, y)   # 3 labels, 4 cols

        # compact show (never dumps the label vectors)
        s = sprint(show, estimator)
        @test occursin("GroupAdaptiveRidge(3 columns in 2 groups", s)
        @test occursin("lambda=0.5", s)
    end

    @testset "singleton groups + unit weights reproduce AdaptiveRidge" begin
        rng = MersenneTwister(42)
        btrue = zeros(10)
        btrue[2] = 1.5
        btrue[7] = -2.0
        X, y = _centered_problem(rng, 80, btrue)
        lam = 1e-3
        b_ar = solve_coefficients(AdaptiveRidge(; lambda = lam), X, y)
        b_gar = solve_coefficients(
            GroupAdaptiveRidge(collect(1:10), ones(10); lambda = lam), X, y)
        # mathematically identical iterations (wⱼ = 1/(βⱼ² + ε), same init, same
        # convergence test) through separate accumulation code — pin tightly, not bitwise
        @test isapprox(b_gar, b_ar; rtol = 1e-10)
        # exact lambda == 0 routes to the same QR OLS path
        est0 = GroupAdaptiveRidge(collect(1:10), ones(10); lambda = 0.0)
        @test solve_coefficients(est0, X, y) == solve_coefficients(OLS(), X, y)
    end

    @testset "group-sparse recovery and weight monotonicity" begin
        rng = MersenneTwister(7)
        labels = repeat(1:4; inner = 3)          # p = 12, four groups of three
        btrue = zeros(12)
        btrue[4:6] .= [1.0, -0.8, 0.6]           # signal on group 2 only
        X, y = _centered_problem(rng, 120, btrue)
        b = solve_coefficients(GroupAdaptiveRidge(labels, ones(4); lambda = 1e-2), X, y)
        active = b[4:6]
        inactive = b[[1:3; 7:12]]
        @test minimum(abs, active) > 50 * maximum(abs, inactive)   # off-groups crushed
        @test isapprox(active, btrue[4:6]; rtol = 0.05)

        # a heavier fixed weight on a group strictly shrinks that group's norm
        btrue2 = copy(btrue)
        btrue2[7:9] .= [0.5, 0.4, -0.3]          # add signal on group 3
        y2 = X * btrue2 .+ 0.01 .* randn(rng, 120)
        y2 .-= mean(y2)
        w_hi = [1.0, 1.0, 2.0, 1.0]
        b_lo = solve_coefficients(GroupAdaptiveRidge(labels, ones(4); lambda = 0.05), X, y2)
        b_hi = solve_coefficients(GroupAdaptiveRidge(labels, w_hi; lambda = 0.05), X, y2)
        @test norm(b_hi[7:9]) < norm(b_lo[7:9])
    end

    # --- the penalty metric ---------------------------------------------------------
    @testset "penalty metric: unpenalized columns are exactly OLS" begin
        # Orthogonal design ⇒ `X'X + λD` is diagonal ⇒ every coefficient is
        # `x_j'y / (‖x_j‖² + λ·D_j)` in closed form, so a column with `D_j = 0` must
        # equal its OLS value for ANY λ and any weight map. Analytic oracle.
        rng = MersenneTwister(9001)
        n, p = 30, 6
        Q = Matrix(qr(randn(rng, n, p)).Q)
        scales = [0.4, 1.0, 2.5, 7.0, 0.9, 3.3]
        X = Q .* scales'
        y = randn(rng, n)
        m = [0.0, 1.3, 0.7, 0.0, 2.0, 0.5]          # two unpenalized columns
        free = [1, 4]
        ols = [dot(view(X, :, j), y) / sum(abs2, view(X, :, j)) for j = 1:p]
        for lam in (1e-3, 1.0, 1e4)
            for est in (Ridge(; lambda = lam, metric = m),
                        AdaptiveRidge(; lambda = lam, metric = m),
                        GroupAdaptiveRidge([1, 1, 2, 2, 3, 3], ones(3);
                                           lambda = lam, metric = m))
                b = solve_coefficients(est, X, y)
                @test b[free] ≈ ols[free] rtol = 1e-13
                # and the penalized ones did move
                @test all(abs.(b[[2, 3, 5, 6]]) .< abs.(ols[[2, 3, 5, 6]]))
            end
        end
        # ... and under a FREEZE reparameterization (`Z` a column selection, the only
        # kind a pure-spin basis produces) the same statement holds in γ space: the
        # unpenalized β column that survives the freeze keeps its OLS value.
        keep = [1, 2, 3, 5, 6]                       # column 4 frozen out
        Z = zeros(p, length(keep))
        for (k, j) in enumerate(keep)
            Z[j, k] = 1.0
        end
        for est in (Ridge(; lambda = 1.0, metric = m),
                    AdaptiveRidge(; lambda = 1.0, metric = m),
                    GroupAdaptiveRidge([1, 1, 2, 2, 3, 3], ones(3); lambda = 1.0,
                                       metric = m))
            g = solve_coefficients(est, X * Z, y; nullspace = Z)
            @test (Z * g)[1] ≈ ols[1] rtol = 1e-13
        end
    end

    @testset "penalty metric: the penalized fit is scale invariant" begin
        # Rescaling column j by c_j and the metric by c_j² leaves the penalty
        # `λ Σ m_j β_j²` unchanged, so the FITTED FUNCTION `Xβ̂` must not move. This is
        # the property the metric exists for; the uniform-metric control shows the
        # gate has teeth.
        rng = MersenneTwister(9002)
        n, p = 60, 8
        X = randn(rng, n, p)
        y = randn(rng, n)
        m = 0.3 .+ 2 .* rand(rng, p)
        groups = [1, 1, 1, 2, 2, 3, 3, 3]
        # deliberately INHOMOGENEOUS within each group: a group-uniform rescaling
        # would not separate "metric in the denominator" from "metric outside it"
        C = exp.(range(-1.5, 1.5; length = p))
        X2 = X .* C'
        m2 = m .* C .^ 2
        # `tol` well below the assertion: the iteration is exactly equivariant in real
        # arithmetic, but a knife-edge rounding difference in the stopping test can
        # cost one extra step in one of the two runs, moving β by O(tol)
        for (e1, e2) in ((Ridge(; lambda = 0.7, metric = m),
                          Ridge(; lambda = 0.7, metric = m2)),
                         (AdaptiveRidge(; lambda = 0.05, tol = 1e-12, metric = m),
                          AdaptiveRidge(; lambda = 0.05, tol = 1e-12, metric = m2)),
                         (GroupAdaptiveRidge(groups, [1.0, 2.0, 0.5]; lambda = 0.05,
                                             tol = 1e-12, metric = m),
                          GroupAdaptiveRidge(groups, [1.0, 2.0, 0.5]; lambda = 0.05,
                                             tol = 1e-12, metric = m2)))
            b1 = solve_coefficients(e1, X, y)
            b2 = solve_coefficients(e2, X2, y)
            @test X2 * b2 ≈ X * b1 rtol = 1e-7
            @test b2 ≈ b1 ./ C rtol = 1e-7
        end
        # control: without a metric the same rescaling moves the fit
        bu1 = solve_coefficients(Ridge(; lambda = 0.7), X, y)
        bu2 = solve_coefficients(Ridge(; lambda = 0.7), X2, y)
        @test !isapprox(X2 * bu2, X * bu1; rtol = 1e-3)
    end

    @testset "penalty metric: the group-L0 fixed point survives" begin
        # The property that makes v_g a group-L0 weight: the converged penalty
        # contribution of an ALIVE group, `Σ_{j∈g} D_j·β_j²`, tends to `v_g` once its
        # metric-weighted norm dominates `p_g·ε`. Placing the metric anywhere but the
        # denominator of the weight map would leave `v_g·⟨m⟩_g` here instead.
        rng = MersenneTwister(9003)
        n, p = 80, 6
        X = randn(rng, n, p)
        btrue = [1.0, -0.8, 0.6, 0.0, 0.0, 0.0]
        y = X * btrue .+ 0.001 .* randn(rng, n)
        groups = [1, 1, 1, 2, 2, 2]
        vg = [1.0, 3.0]
        m = [0.2, 5.0, 1.0, 0.7, 2.0, 0.3]          # strongly non-uniform within group 1
        est = GroupAdaptiveRidge(groups, vg; lambda = 1e-4, metric = m)
        b = solve_coefficients(est, X, y)
        _, D = SLCE._penalty_diagonal(est, b)
        contrib1 = sum(D[j] * b[j]^2 for j = 1:3)
        @test contrib1 ≈ vg[1] rtol = 1e-6         # alive group: → v_g, not v_g·⟨m⟩
        # the fixed point is `v_g·N/(N + p_g·ε)`, so the relative gap is `p_g·ε/N`:
        # the assertion above is only meaningful while N clears `p_g·ε / rtol`
        @test sum(m[j] * b[j]^2 for j = 1:3) > 3 * est.epsilon / 1e-6
    end

    @testset "penalty metric: refusals" begin
        rng = MersenneTwister(9004)
        X = randn(rng, 20, 4)
        y = randn(rng, 20)
        @test_throws ArgumentError Ridge(; lambda = 1.0, metric = zeros(4))
        @test_throws ArgumentError Ridge(; lambda = 1.0, metric = [1.0, -1.0, 1.0, 1.0])
        @test_throws ArgumentError Ridge(; lambda = 1.0, metric = [1.0, NaN, 1.0, 1.0])
        # length is checked against the DESIGN, at the solve door
        @test_throws DimensionMismatch solve_coefficients(
            Ridge(; lambda = 1.0, metric = ones(3)), X, y)
        # a rank-deficient unpenalized block leaves the fit unidentified
        Xd = copy(X)
        Xd[:, 2] = Xd[:, 1]
        @test_throws ArgumentError solve_coefficients(
            Ridge(; lambda = 1.0, metric = [0.0, 0.0, 1.0, 1.0]), Xd, y)
        # `nothing` (uniform) reproduces the unweighted penalty exactly
        @test solve_coefficients(Ridge(; lambda = 0.3, metric = ones(4)), X, y) ==
              solve_coefficients(Ridge(; lambda = 0.3), X, y)
        @test solve_coefficients(AdaptiveRidge(; lambda = 0.3, metric = ones(4)), X, y) ==
              solve_coefficients(AdaptiveRidge(; lambda = 0.3), X, y)
        @test_throws ArgumentError SLCE._checked_metric_keyword(:bases)
    end

    @testset "OLS ignores the metric (normal equations, not a captured value)" begin
        rng = MersenneTwister(9005)
        X = randn(rng, 25, 5)
        y = randn(rng, 25)
        b = solve_coefficients(OLS(), X, y)
        @test norm(X' * (y .- X * b)) <= 1e-10 * norm(X' * y)
        C = exp.(range(-1.0, 1.0; length = 5))
        b2 = solve_coefficients(OLS(), X .* C', y)
        @test (X .* C') * b2 ≈ X * b rtol = 1e-10
        # lambda = 0 routes to OLS whatever the metric says
        @test solve_coefficients(Ridge(; lambda = 0.0, metric = 0.1 .+ rand(rng, 5)),
                                 X, y) == b
    end

    @testset "the metric under a reparameterization: Z'DZ, and λI when uniform" begin
        # The reparameterized path has its own penalty form, so the metric has to be
        # checked there rather than only on the plain design.
        rng = MersenneTwister(9006)
        n, p, q = 40, 7, 5
        X = randn(rng, n, p)
        y = randn(rng, n)
        Z = Matrix(qr(randn(rng, p, q)).Q)[:, 1:q]
        Xt = X * Z
        lam = 0.4
        # (a) With a metric the γ-space penalty is `λ·Z'·Diagonal(m)·Z` — written out
        #     here from the definition `λ Σ mⱼβⱼ²` with β = Z·γ, not from the code.
        m = 0.3 .+ 2 .* rand(rng, p)
        g = solve_coefficients(Ridge(; lambda = lam, metric = m), Xt, y; nullspace = Z)
        ref = Symmetric(Xt' * Xt + lam * (Z' * Diagonal(m) * Z)) \ (Xt' * y)
        @test g ≈ ref rtol = 1e-10
        # (b) With NO metric the penalty is exactly `λ·I` in γ space (Z orthonormal),
        #     which is what the pre-metric code solved.
        g0 = solve_coefficients(Ridge(; lambda = lam), Xt, y; nullspace = Z)
        @test g0 == Symmetric(Xt' * Xt + lam * I(q)) \ (Xt' * y)
        # (c) an explicit all-ones metric is the same problem, to rounding
        g1 = solve_coefficients(Ridge(; lambda = lam, metric = ones(p)), Xt, y;
                                nullspace = Z)
        @test g1 ≈ g0 rtol = 1e-10
        # (d) the metric is indexed by the BASIS columns, so its length is p, not q
        @test_throws DimensionMismatch solve_coefficients(
            Ridge(; lambda = lam, metric = ones(q)), Xt, y; nullspace = Z)
    end

    # --- basis-driven helpers -------------------------------------------------------
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    # NoSymmetry (P1): anisotropic basis with several (orbit, ls) groups split into
    # Lf/block channels, plus a small isotropic basis for by-hand entry counting
    basis = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                                        soc = true))
    small = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [1],
                                        soc = false))

    @testset "penalty_metric: closed-form reference norms" begin
        # `small` carries exactly one SALC: the isotropic 2-body l = 1 pair, whose
        # normalization is derived by hand in test_normalization.jl ("O1"),
        #   Φ(e) = 2√3·(e₁·e₂).
        # For independent uniform directions u = e₁·e₂ is uniform on [−1, 1], so
        #   E[Φ] = 0 and Var[Φ] = 12·E[u²] = 12/3 = 4.
        @test n_salcs(small) == 1
        K = 20000
        m0 = penalty_metric(small; nconfig = K, seed = 7)
        sig = 12 * sqrt((4 / 45) / K)          # Var(u²) = 1/5 − 1/9 = 4/45
        @test abs(m0[1] - 4.0) < 5 * sig
        # Headroom: 5σ is ~3.2 % of the expected 4.0, while the errors this gate is
        # for are gross — dropping the ordering fold would be a factor of 4, and
        # dropping the torque row average below a factor of 3·n_atoms.
        @test 5 * sig / 4.0 < 0.05
        # Torque block: ∂Φ/∂e₁ = 2√3·e₂, so ‖τ₁‖² = 12(1 − u²) and likewise for atom 2,
        #   m(w = 1) = E[Σ_a ‖τ_a‖²] / (3·n_atoms) = 2·12·(1 − 1/3) / 6 = 8/3.
        m1 = penalty_metric(small; torque_weight = 1.0, nconfig = K, seed = 7)
        sigt = 4 * sqrt((4 / 45) / K)          # the per-config quantity is 4(1 − u²)
        @test abs(m1[1] - 8 / 3) < 5 * sigt
        # a co-fit metric is the convex combination of the two blocks at the same seed
        mh = penalty_metric(small; torque_weight = 0.25, nconfig = K, seed = 7)
        @test mh[1] ≈ 0.75 * m0[1] + 0.25 * m1[1] rtol = 1e-12
        # the generator is internal to the package, so the ensemble is reproducible
        @test penalty_metric(small; nconfig = 512, seed = 3) ==
              penalty_metric(small; nconfig = 512, seed = 3)
        @test penalty_metric(small; nconfig = 512, seed = 3) !=
              penalty_metric(small; nconfig = 512, seed = 4)
        @test_throws ArgumentError penalty_metric(small; nconfig = 1)
        @test_throws ArgumentError penalty_metric(small; torque_weight = 1.5)
    end

    @testset "penalty_metric refuses a displacement-carrying basis" begin
        # The reference ensemble is spin directions only; `|u|^{2k}·R_lm(u)` has no
        # value on one. Refuse by name rather than measure something arbitrary — and
        # refuse the same way through the basis-aware constructors, which is where a
        # joint user would meet it.
        jcr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0],
                      [1, 1], ["Fe"])
        jb = SLCEBasis(jcr, BasisSpec(jcr; lmax = 1, pmax = 1, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)]))
        @test !all(SLCE.is_pure_spin, jb.salc_basis.keys)
        @test_throws ArgumentError penalty_metric(jb)
        @test_throws ArgumentError Ridge(jb; lambda = 1.0)
        @test_throws ArgumentError GroupAdaptiveRidge(jb; lambda = 1.0)
        # ... and `metric = nothing` is the documented way through
        @test Ridge(jb; lambda = 1.0, metric = nothing).metric === nothing
    end

    @testset "salc_groups: contiguous labels matching (body, orbit_id, ls) runs" begin
        ks = basis.salc_basis.keys
        labels = salc_groups(basis)
        m = length(ks)
        @test length(labels) == m == n_salcs(basis)
        G = maximum(labels)
        @test sort(unique(labels)) == 1:G
        tup = [(k.body, k.orbit_id, k.decors) for k in ks]
        @test G == length(unique(tup))
        @test all((labels[i] == labels[j]) == (tup[i] == tup[j])
                  for i = 1:m for j = 1:m)
        # at least one group carries multiple Lf/block channels (else the fixture is
        # too weak to distinguish group from column selection)
        @test G < m
    end

    @testset "group_costs: brute-force union, additivity, validation" begin
        # both fixtures: `small` (single group) and `basis` (many (orbit, ls) groups,
        # so the additivity claim is exercised with G > 1)
        for b in (small, basis)
            labels = salc_groups(b)
            costs = group_costs(b, labels)
            G = maximum(labels)
            @test length(costs) == G

            # Independent brute force with a structurally different key encoding:
            # nested tuples rather than the vectors `_EntryKey` uses, and the slot
            # labels spelled out here rather than routed through `SLCE._slotkey`.
            bysets = [Set{Any}() for _ = 1:G]
            for (j, s) in enumerate(salcs(b))
                for mem in s.members, t in mem.terms
                    slotkey = Tuple((Int(sl.factor.channel), sl.site,
                                     sl.factor.k, sl.factor.l) for sl in t.slots)
                    for index in CartesianIndices(t.folded)
                        t.folded[index] == 0.0 && continue
                        key = (Tuple(mem.atoms), Tuple(Tuple.(mem.shifts)),
                               slotkey, Tuple(index))
                        push!(bysets[labels[j]], key)
                    end
                end
            end
            # the cost is the summed SLOT COUNT over distinct entries, not the entry
            # count: the MC emits one site program per slot (`_push_term_programs!`)
            slotsum(S) = sum(k -> length(k[3]), S; init = 0)
            @test costs == slotsum.(bysets)
            # and it is strictly larger than the bare entry count wherever a term has
            # more than one slot — the defect this pricing fixes
            @test all(costs .>= length.(bysets))
            @test any(costs .> length.(bysets))

            # additivity: groups never share entries, so the sum is the global
            # cost — equal to the single-group partition's cost
            total = group_costs(b, ones(Int, n_salcs(b)))
            @test length(total) == 1
            @test sum(costs) == total[1]
            @test total[1] == slotsum(union(bysets...))

            # default labels argument = salc_groups partition
            @test group_costs(b) == costs
        end
        @test maximum(salc_groups(basis)) > 1     # the multi-group case was real

        # A JOINT basis: this used to be refused outright ("the MC entry-key model is
        # pure-spin"). The slot-keyed entry is what makes it well-defined — keying on
        # the pure-spin `ls` would erase every displacement slot, collapsing distinct
        # lattice groups onto one key and destroying the additivity asserted below.
        jcr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0],
                      [1, 1], ["Fe"])
        jbasis = SLCEBasis(jcr, BasisSpec(jcr; lmax = 1, pmax = 1, sectors = [
            Sector(spin = (sites = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1)]))
        jlab = salc_groups(jbasis)
        jcost = group_costs(jbasis, jlab)
        @test length(jcost) == maximum(jlab)
        @test all(>(0), jcost)
        # the basis really is mixed, and really does carry groups whose slot count
        # exceeds the body order (a site holding both a spin and a displacement
        # factor) — the case a `body`-based multiplier would get wrong
        ks = jbasis.salc_basis.keys
        @test !all(SLCE.is_pure_spin, ks) && any(SLCE.is_pure_spin, ks)
        @test any(length(t.slots) > s.body
                  for s in salcs(jbasis) for m in s.members for t in m.terms)
        # additivity survives on the joint basis too
        @test sum(jcost) == group_costs(jbasis, ones(Int, n_salcs(jbasis)))[1]
        # and pure-spin groups are still priced at |entries|·body
        for g in unique(jlab)
            cols = findall(==(g), jlab)
            all(SLCE.is_pure_spin, ks[cols]) || continue
            nent = sum(count(!=(0.0), t.folded)
                       for j in cols for m in salcs(jbasis)[j].members
                       for t in m.terms)
            @test jcost[g] <= nent * ks[cols[1]].body      # ≤: the union dedups
        end

        # malformed labels
        @test_throws ArgumentError group_costs(small, ones(Int, n_salcs(small) + 1))
        badgap = ones(Int, n_salcs(small))
        badgap[1] = 3                          # skips label 2
        @test_throws ArgumentError group_costs(small, badgap)
    end

    @testset "cost_weights and the basis convenience constructor" begin
        labels = salc_groups(basis)
        c = group_costs(basis, labels)
        G = maximum(labels)
        p = [count(==(g), labels) for g = 1:G]

        lw0 = cost_weights(basis; cost_exponent = 0.0)
        @test lw0.column_groups == labels
        @test lw0.weights == sqrt.(p)             # exactly √p_g at cost_exponent = 0

        lw1 = cost_weights(basis; cost_exponent = 1.0)
        @test lw1.weights ≈ sqrt.(p) .* (c ./ mean(c))

        lw5 = cost_weights(basis; cost_exponent = 0.5)
        @test all(min.(lw0.weights, lw1.weights) .<= lw5.weights .+ 1e-12) &&
              all(lw5.weights .<= max.(lw0.weights, lw1.weights) .+ 1e-12)

        @test_throws ArgumentError cost_weights(basis; cost_exponent = -0.1)
        @test_throws ArgumentError cost_weights(basis; cost_exponent = 1.1)

        estimator = GroupAdaptiveRidge(basis; lambda = 1e-3, cost_exponent = 1.0)
        @test estimator.column_groups == labels
        @test estimator.group_weights == lw1.weights
        @test estimator.lambda === 1e-3
    end

    # --- GCV / effective degrees of freedom -----------------------------------------
    @testset "effective_dof / gcv match the dense hat matrix (n > p)" begin
        rng = MersenneTwister(23)
        nconf = 30
        configs = [Matrix(randcfg(rng, 2)) for _ = 1:nconf]
        energies = randn(rng, nconf)
        ds = SLCEDataset(small, configs, energies)
        X, y, _, _, _ = SLCE._assemble_problem(ds, 0.0)
        n = size(X, 1)
        @test n > size(X, 2)                       # overdetermined fixture

        lam = 0.3
        fr = fit(SLCEFit, ds, Ridge(; lambda = lam))
        Hr = X * ((X' * X + lam * I) \ X')
        @test effective_dof(fr) ≈ tr(Hr) + 1 rtol = 1e-8
        rss_r = sum(abs2, y .- X * coef(fr))
        @test gcv(fr) ≈ n * rss_r / (n - (tr(Hr) + 1))^2 rtol = 1e-8

        # adaptive members: converged weights recomputed from the fitted coefficients
        fa = fit(SLCEFit, ds, AdaptiveRidge(; lambda = 1e-3))
        Da = Diagonal(1.0 ./ (coef(fa) .^ 2 .+ 1e-8))
        Ha = X * ((X' * X + 1e-3 * Da) \ X')
        @test effective_dof(fa) ≈ tr(Ha) + 1 rtol = 1e-8

        glabels = salc_groups(small)
        fg = fit(SLCEFit, ds, GroupAdaptiveRidge(glabels, ones(maximum(glabels));
                                                lambda = 1e-3))
        beta = coef(fg)
        wg = [1.0 / (sum(abs2, beta[glabels .== glabels[j]]) +
                     count(==(glabels[j]), glabels) * 1e-8) for j in eachindex(beta)]
        Hg = X * ((X' * X + 1e-3 * Diagonal(wg)) \ X')
        @test effective_dof(fg) ≈ tr(Hg) + 1 rtol = 1e-8
        @test gcv(fg) ≈ n * sum(abs2, y .- X * beta) / (n - (tr(Hg) + 1))^2 rtol = 1e-8

        fo = fit(SLCEFit, ds, OLS())
        @test effective_dof(fo) ≈ rank(X) + 1
        @test isfinite(gcv(fo))
    end

    @testset "effective dof with unpenalized columns (dense hat reference)" begin
        rng = MersenneTwister(4127)
        # Independent oracle: the dense hat matrix of the penalized normal equations,
        # `tr(X(X'X + λD)⁻¹X')`, formed here without any of the split-block machinery.
        dense_df(X, lam, d) = tr(X * ((X' * X + lam * Diagonal(d)) \ X'))

        lam = 0.37
        # (a) overdetermined: no / one / several unpenalized columns, and the cached
        #     Gram keyword (whose `D^{-1/2}` form is the one that divides by zero)
        n, p = 40, 12
        X = randn(rng, n, p)
        base = 0.3 .+ rand(rng, p)
        for free in (Int[], [3], [1, 7, 12])
            d = copy(base)
            d[free] .= 0.0
            @test SLCE._effective_dof_gram(X, lam, d) ≈ dense_df(X, lam, d) rtol = 1e-9
            @test SLCE._effective_dof_gram(X, lam, d; XtX = X' * X) ≈
                  dense_df(X, lam, d) rtol = 1e-9
        end

        # (b) underdetermined: the dual side, with the unpenalized block still thin
        n2, p2 = 9, 25
        X2 = randn(rng, n2, p2)
        d2 = 0.3 .+ rand(rng, p2)
        d2[[2, 5]] .= 0.0
        @test SLCE._effective_dof_gram(X2, lam, d2) ≈ dense_df(X2, lam, d2) rtol = 1e-9

        # (c) λ → ∞ leaves exactly the unpenalized degrees of freedom, and an
        #     all-unpenalized diagonal is the plain design rank
        d3 = copy(base)
        d3[[2, 4, 9]] .= 0.0
        @test SLCE._effective_dof_gram(X, 1e12, d3) ≈ 3.0 rtol = 1e-6
        @test SLCE._effective_dof_gram(X, lam, zeros(p)) == Float64(p)

        # (d) a rank-deficient unpenalized block leaves the fit unidentified: refuse
        #     by name rather than report a finite dof for it
        Xd = copy(X)
        Xd[:, 5] = Xd[:, 3]
        dd = copy(base)
        dd[[3, 5]] .= 0.0
        @test_throws ArgumentError SLCE._effective_dof_gram(Xd, lam, dd)

        # (e) the reparameterized sibling has no split-block form, so it refuses an
        #     unpenalized column by name instead of silently factorizing a singular
        #     compressed penalty. No channel produces the combination today; the
        #     refusal is what keeps that true if one starts to.
        Z = Matrix(qr(randn(rng, p, 8)).Q)[:, 1:8]
        @test_throws ArgumentError SLCE._effective_dof_nullspace(X * Z, lam, d3, Z)
    end

    @testset "the unpenalized rank cut is min(size), not max(size)" begin
        # `_rank_df` feeds `effective_dof`/`gcv` for every OLS fit and every λ = 0 point
        # of a λ-path. The fixture above is well conditioned, so `min` and `max` agree
        # there and it cannot see the convention. Engineer a marginal singular value
        # that sits ABOVE the package's own documented cut (`min(size)·eps·σ₁`, the
        # convention `identifiability` picks and reasons about) and watch a `max`-based
        # cut drop it as soon as the row count exceeds the column count.
        p = 6
        for n in (10, 1000)
            U = qr(randn(MersenneTwister(4), n, p)).Q[:, 1:p]
            V = qr(randn(MersenneTwister(5), p, p)).Q
            # last singular value at 5× the min-based tolerance: determined by the
            # package's convention, flat under a max-based one once n ≫ p
            svals = [1.0, 0.5, 0.25, 0.1, 0.05, 5 * p * eps(Float64)]
            Xm = U * Diagonal(svals) * V'
            @test SLCE._rank_df(Xm) == Float64(p)
            @test SLCE._rank_df(Xm) == Float64(rank(Xm))   # agrees with LinearAlgebra
        end
    end

    @testset "gcv in the underdetermined regime and guards" begin
        rng = MersenneTwister(31)
        nconf = 12
        configs = [Matrix(randcfg(rng, 2)) for _ = 1:nconf]
        # an in-span group-sparse target: a pure-noise target would drive the adaptive
        # fit to interpolation (df → n) and the GCV guard to Inf by design
        ds0 = SLCEDataset(basis, configs, zeros(nconf))
        btrue = zeros(n_salcs(basis))
        btrue[1:3] .= [0.5, -0.3, 0.2]
        energies = 0.7 .+ ds0.X_E * btrue .+ 0.001 .* randn(rng, nconf)
        ds = SLCEDataset(basis, configs, energies)   # p = n_salcs(basis) > 12 rows
        @test n_salcs(basis) > nconf
        f = fit(SLCEFit, ds, GroupAdaptiveRidge(basis; lambda = 1e-2))
        @test effective_dof(f) - 1 < nconf
        @test isfinite(gcv(f))
        # near-interpolating: the guard returns Inf instead of a meaningless score
        # (column centering makes the rows sum to zero, so rank ≤ n − 1 and df + 1 → n)
        fni = fit(SLCEFit, ds, Ridge(; lambda = 1e-14))
        @test gcv(fni) == Inf
        # non-linear estimators are rejected
        fp = fit(SLCEFit, ds, FixedCoefficients(zeros(n_salcs(basis))))
        @test_throws ArgumentError gcv(fp)
        @test_throws ArgumentError effective_dof(fp)
    end

    @testset "gcv on a torque co-fit (smoke; grouped CV is the ground truth)" begin
        rng = MersenneTwister(37)
        nconf = 10
        configs = [Matrix(randcfg(rng, 2)) for _ = 1:nconf]
        energies = randn(rng, nconf)
        torques = [randn(rng, 3, 2) for _ = 1:nconf]
        ds = SLCEDataset(small, configs, energies, torques)
        f = fit(SLCEFit, ds, Ridge(; lambda = 0.1); torque_weight = 0.3)
        @test isfinite(gcv(f))
        @test effective_dof(f) > 1
    end

    # --- λ path + Pareto selection --------------------------------------------------
    @testset "_select_pareto rule" begin
        sp = SLCE._select_pareto
        scores = [1.0, 1.02, 1.5]
        costs = [100.0, 40.0, 10.0]
        @test sp(scores, costs, 0.05) == 2      # 1.02 within 5% of 1.0, cheaper
        @test sp(scores, costs, 0.6) == 3       # 1.5 admitted too, cheapest wins
        @test sp(scores, costs, 0.0) == 1       # only the exact minimum eligible
        @test sp([1.0, 1.0], [5.0, 5.0], 0.1) == 1          # cost tie → larger λ
        @test sp([Inf, 1.0, 1.1], [1.0, 9.0, 8.0], 0.2) == 3  # Inf never eligible
        @test_throws ArgumentError sp([Inf, Inf], [1.0, 2.0], 0.1)
        @test_throws ArgumentError sp(scores, costs, -0.1)
        @test_throws DimensionMismatch sp(scores, [1.0], 0.1)
    end

    # group-sparse in-span fixture shared by the path tests
    rngp = MersenneTwister(41)
    nconf_p = 40
    configs_p = [Matrix(randcfg(rngp, 2)) for _ = 1:nconf_p]
    ds0_p = SLCEDataset(basis, configs_p, zeros(nconf_p))
    labels_p = salc_groups(basis)
    btrue_p = zeros(n_salcs(basis))
    btrue_p[labels_p .== 1] .= 0.4
    btrue_p[labels_p .== 3] .= -0.25
    energies_p = 0.3 .+ ds0_p.X_E * btrue_p .+ 0.005 .* randn(rngp, nconf_p)
    ds_p = SLCEDataset(basis, configs_p, energies_p)
    est_p = GroupAdaptiveRidge(basis; lambda = 1.0)   # template λ is ignored

    @testset "penalty-metric provenance is checked at the fitting doors" begin
        good = GroupAdaptiveRidge(basis; lambda = 1e-3).metric
        fp = basis.salc_basis.fingerprint
        lab = salc_groups(basis)
        bad_basis = GroupAdaptiveRidge(
            lab, ones(maximum(lab)); lambda = 1e-3, metric = good,
            metric_provenance = MetricProvenance(:energy, 0.0, false, 2000, 1, fp + 0x1))
        @test_throws ArgumentError fit(SLCEFit, ds_p, bad_basis)
        bad_channel = GroupAdaptiveRidge(
            lab, ones(maximum(lab)); lambda = 1e-3, metric = good,
            metric_provenance = MetricProvenance(:moment, 0.0, true, 2000, 1, fp))
        @test_throws ArgumentError fit(SLCEFit, ds_p, bad_channel)
        # a metric taken at w = 0 is refused for a co-fit: the assembled design mixes
        # the two blocks by w, so the column scales move with it
        est_w0 = GroupAdaptiveRidge(basis; lambda = 1e-3, torque_weight = 0.0)
        @test_throws ArgumentError select_fit(ds_p, est_w0; lambdas = [1e-3],
                                              torque_weight = 0.5)
        # a mistyped sentinel is refused BY NAME wherever a metric is validated —
        # the plain keyword constructors included, which is where a user who read
        # `Ridge(basis; ...)` first would reach for it. Falling through would die
        # inside `Vector{Float64}(::Symbol)` with a bare MethodError.
        for bad_sentinel in (:base, :Basis, :default)
            @test_throws ArgumentError Ridge(; lambda = 1.0, metric = bad_sentinel)
            @test_throws ArgumentError AdaptiveRidge(; lambda = 1.0,
                                                     metric = bad_sentinel)
        end
        # a hand-built metric with no provenance is the caller's business and passes
        hand = GroupAdaptiveRidge(lab, ones(maximum(lab)); lambda = 1e-3, metric = good)
        @test fit(SLCEFit, ds_p, hand) isa SLCEFit
        # `with_lambda` carries the metric AND its provenance; a hand rebuild does not
        @test with_lambda(est_p, 0.5).metric == est_p.metric
        @test with_lambda(est_p, 0.5).metric_provenance === est_p.metric_provenance
        @test with_lambda(est_p, 0.5).lambda == 0.5
        @test GroupAdaptiveRidge(est_p.column_groups, est_p.group_weights;
                                 lambda = 0.5).metric === nothing
        # MetricProvenance validates its own fields
        @test_throws ArgumentError MetricProvenance(:torque, 0.0, false, 2000, 1, fp)
        @test_throws ArgumentError MetricProvenance(:energy, 1.5, false, 2000, 1, fp)
        @test_throws ArgumentError MetricProvenance(:energy, 0.0, false, 1, 1, fp)
        # `free_intercepts` is a moment-channel property (the μ₀ columns); the energy
        # channel has no such column, so recording it there is a wiring mistake
        @test_throws ArgumentError MetricProvenance(:energy, 0.0, true, 2000, 1, fp)
    end

    # The λ grids are MSE-relative, i.e. divided by `nconf_p`. `_assemble_problem` whitens
    # the energy block by `1/√n_E` at every weight setting, so the penalty a given λ applies
    # scales with `n_E`; a grid written in the old SSE units sits ~1.6 decades too high on
    # this fixture, where the IRLS lands on the all-zero fixed point (`edof ≈ 1`) and the
    # warm/cold agreement below stops testing anything.
    @testset "select_fit: warm-started path matches cold fits" begin
        lams = 10.0 .^ range(0, -6; length = 6) ./ nconf_p
        path = select_fit(ds_p, est_p; lambdas = lams, criterion = :gcv)
        @test path.lambda == sort(unique(Float64.(lams)); rev = true)
        # warm-started path entries reproduce cold single-λ fits: both converge to the
        # same fixed point but stop within the IRLS tol (1e-6 on coefficients), so the
        # scores agree to a few multiples of that tolerance, not exactly
        # `with_lambda`, NOT a hand-rebuilt GroupAdaptiveRidge: rebuilding from
        # `column_groups` / `group_weights` silently drops the basis-intrinsic penalty
        # metric `est_p` carries, and the cold fit would then solve a different
        # objective than the path did — which is exactly what this comparison would
        # otherwise report as a warm/cold disagreement.
        for i in (2, 4)
            est_i = with_lambda(est_p, path.lambda[i])
            fc = fit(SLCEFit, ds_p, est_i)
            @test isapprox(path.score[i], gcv(fc); rtol = 1e-4)
            @test isapprox(path.edof[i], effective_dof(fc); rtol = 1e-4)
        end
        # the selected fit is the cold fit at the selected λ, byte-for-byte
        est_s = with_lambda(est_p, path.lambda[path.selected])
        @test path.fit.jphi == fit(SLCEFit, ds_p, est_s).jphi
        # the selected row's score/edof are re-derived from that cold fit, so the
        # displayed row is self-consistent (not the warm path value)
        @test path.score[path.selected] ≈ gcv(path.fit) rtol = 1e-12
        @test path.edof[path.selected] ≈ effective_dof(path.fit) rtol = 1e-12
        # the returned fit carries the path's metric: the cold re-solve at the
        # selected λ rebuilds the estimator, and dropping the metric there would
        # leave the path solved one way and the returned model another
        @test path.fit.estimator.metric == est_p.metric
        @test path.fit.estimator.metric_provenance === est_p.metric_provenance
        # alive count grows (weakly) as λ decreases along the descending path
        @test all(diff(path.n_alive) .>= 0)
    end

    @testset "select_fit: end-to-end selection, tables, and validation" begin
        lams = 10.0 .^ range(1, -7; length = 10) ./ nconf_p
        path = select_fit(ds_p, est_p; lambdas = lams, criterion = :gcv)
        np = length(path.lambda)
        @test length(path.score) == length(path.edof) == np
        @test length(path.n_alive) == length(path.cost) == np
        @test 1 <= path.selected <= np
        @test isfinite(path.score[path.selected])
        # the relative alive floor discriminates: the cost column varies along the
        # path (a flat column would mean the cost axis is vacuous)
        @test path.cost[1] < path.cost[end]
        @test path.threshold > 0
        @test minimum(path.n_alive) < maximum(path.n_alive)
        # widening the accuracy tolerance can only cheapen the selection
        pwide = select_fit(ds_p, est_p; lambdas = lams, criterion = :gcv, score_rtol = 0.5)
        @test pwide.cost[pwide.selected] <= path.cost[path.selected]
        # de-bias on exactly the reported support and recover the data
        fr = refit(path.fit; threshold = path.threshold)
        @test r2_energy(fr) > 0.95
        # predicted cost at the selected point matches an independent recomputation
        # from the returned fit and threshold
        X, _, _, _, _ = SLCE._assemble_problem(ds_p, 0.0)
        cn = [norm(X[:, j]) for j = 1:size(X, 2)]
        cg = group_costs(basis, labels_p)
        aliveg = unique(labels_p[[j for j in eachindex(path.fit.jphi)
                                  if abs(path.fit.jphi[j]) * cn[j] > path.threshold]])
        @test path.cost[path.selected] == sum(cg[aliveg])
        @test path.n_alive[path.selected] == length(aliveg)
        # Tables view
        cols = Tables.columntable(path)
        @test keys(cols) == (:lambda, :score, :edof, :n_alive, :cost, :selected)
        @test count(cols.selected) == 1
        # show smoke
        @test occursin("← selected", sprint(show, MIME"text/plain"(), path))

        # an absolute threshold reproduces refit's rule verbatim; threshold = 0
        # counts every group alive (documented degenerate case)
        p0 = select_fit(ds_p, est_p; lambdas = [1.0, 1e-3], threshold = 0.0)
        @test all(==(maximum(labels_p)), p0.n_alive)
        @test p0.threshold === 0.0

        # grouped-CV criterion runs and selects a finite score
        pcv = select_fit(ds_p, est_p; lambdas = 10.0 .^ range(0, -5; length = 5),
                         criterion = :cv, nfolds = 4)
        @test all(isnan, pcv.edof)
        @test isfinite(pcv.score[pcv.selected])

        # validation errors
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = Float64[])
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [-1.0])
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0],
                                              criterion = :bogus)
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0],
                                              score_rtol = -0.1)
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0],
                                              costs = [1.0])
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0], nfolds = 1)
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0],
                                              threshold = -1.0)
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0],
                                              torque_weight = 1.5)
        @test_throws ArgumentError select_fit(ds_p, est_p; lambdas = [1.0],
                                              torque_weight = 0.5)   # no torque data
        # too few resampling units for grouped CV
        ds_tiny = SLCEDataset(basis, configs_p[1:5], energies_p[1:5])
        @test_throws ArgumentError select_fit(ds_tiny, est_p; lambdas = [1.0, 0.1],
                                              criterion = :cv)
    end

    @testset "select_support: threshold-swept refit front" begin
        ds_tr2 = ds_p[1:30]
        ds_ev = ds_p[31:40]                       # held-out evaluation slice
        f = fit(SLCEFit, ds_tr2, GroupAdaptiveRidge(basis; lambda = 1e-4))
        sp = select_support(f; npoints = 8, evalset = ds_ev)

        nt = length(sp.threshold)
        @test issorted(sp.threshold; rev = true)                  # sparsest first
        @test length(sp.n_alive) == length(sp.cost) == nt
        @test length(sp.score) == length(sp.rmse_energy) == nt
        @test issorted(sp.n_alive) && issorted(sp.cost)           # nested supports
        @test sp.threshold[end] == 0.0                            # full-support anchor
        @test sp.n_alive[end] == maximum(salc_groups(basis))
        @test minimum(sp.n_alive) < maximum(sp.n_alive)           # front discriminates
        @test all(isnan, sp.rmse_torque)                          # energy-only evalset
        @test sp.score ≈ sp.rmse_energy .^ 2                      # w = 0 objective
        @test 1 <= sp.selected <= nt
        @test isfinite(sp.score[sp.selected])
        # the returned fit is the refit at the selected threshold, byte-for-byte
        @test sp.fit.jphi == refit(f; threshold = sp.threshold[sp.selected]).jphi
        # in-span group-sparse target: the selected sparse refit generalizes, and
        # beats the full-support refit (which is an underdetermined min-norm OLS
        # here, n = 30 rows < p = 44 columns — it overfits the held-out slice)
        @test sp.rmse_energy[sp.selected] < 1e-2
        @test sp.rmse_energy[sp.selected] < sp.rmse_energy[end]
        # Pareto: widening score_rtol can only cheapen the selection
        spw = select_support(f; npoints = 8, evalset = ds_ev, score_rtol = 0.5)
        @test spw.cost[spw.selected] <= sp.cost[sp.selected]

        # explicit thresholds vector; single full-support point
        sp1 = select_support(f; thresholds = [0.0])
        @test length(sp1.threshold) == 1 && sp1.selected == 1

        # Tables + show smoke
        cols = Tables.columntable(sp)
        @test keys(cols) == (:threshold, :n_alive, :cost, :score, :rmse_energy,
                             :rmse_torque, :rmse_force, :selected)
        @test count(cols.selected) == 1
        @test occursin("← selected", sprint(show, MIME"text/plain"(), sp))

        # coupled-site pin: the selected row re-derives from the returned refit —
        # groups with a nonzero de-biased coefficient are exactly the alive groups
        lab_b = salc_groups(basis)
        cg_b = group_costs(basis, lab_b)
        ag = unique(lab_b[findall(!=(0.0), sp.fit.jphi)])
        @test length(ag) == sp.n_alive[sp.selected]
        @test sum(cg_b[ag]) == sp.cost[sp.selected]

        # the auto grid: midpoints between distinct magnitudes + the 0.0 anchor;
        # exact ties collapse (tied groups die together, no empty-support point)
        st = SLCE._support_thresholds
        @test st(5, [3.0, 2.0, 1.0]) == [2.5, 1.5, 0.0]
        @test st(4, [2.0]) == [0.0]                       # G = 1 → anchor only
        @test st(9, [1.0, 1.0, 0.5]) == [0.75, 0.0]       # tie: no t = 1.0 point
        @test_throws ArgumentError st(1, [1.0])

        # torque co-fit: rmse_torque populated; energy-only evalset rejected
        rngs = MersenneTwister(53)
        cfg_s = [Matrix(randcfg(rngs, 2)) for _ = 1:12]
        en_s = randn(rngs, 12)
        tq_s = [randn(rngs, 3, 2) for _ = 1:12]
        ds_cofit = SLCEDataset(small, cfg_s, en_s, tq_s)
        glab = salc_groups(small)
        fco = fit(SLCEFit, ds_cofit, GroupAdaptiveRidge(glab, ones(maximum(glab));
                                                       lambda = 1e-3);
                  torque_weight = 0.4)
        spc = select_support(fco; npoints = 3)
        @test all(isfinite, spc.rmse_torque)
        @test length(spc.threshold) == 1        # single-group basis: grid collapses
        @test_throws ArgumentError select_support(fco;
                                                  evalset = SLCEDataset(small, cfg_s, en_s))

        # validation errors
        @test_throws ArgumentError select_support(f; score_rtol = -0.1)
        @test_throws ArgumentError select_support(f; thresholds = Float64[])
        @test_throws ArgumentError select_support(f; thresholds = [-1.0])
        @test_throws ArgumentError select_support(f; npoints = 1)
        @test_throws ArgumentError select_support(f; costs = [1.0])
        # the count knob and the explicit vector are separate keywords: a scalar
        # passed to `thresholds` is a loud type error, not a silent 8-point grid
        @test_throws TypeError select_support(f; thresholds = 8)
        # and an explicit vector wins over the grid count
        @test length(select_support(f; npoints = 25,
                                    thresholds = [0.0]).threshold) == 1
        @test_throws ArgumentError select_support(f; estimator = FixedCoefficients(
            zeros(n_salcs(basis))))
        ds_other = SLCEDataset(small, [Matrix(randcfg(rngs, 2)) for _ = 1:4],
                              randn(rngs, 4))
        @test_throws ArgumentError select_support(f; evalset = ds_other)
    end

    @testset "cross_validate" begin
        # energy-only fixture: 12 configs so nfolds = 3 needs no reduction
        rngc = MersenneTwister(67)
        nconf_c = 12
        cfg_c = [Matrix(randcfg(rngc, 2)) for _ = 1:nconf_c]
        ds0_c = SLCEDataset(small, cfg_c, zeros(nconf_c))
        en_c = 0.2 .+ ds0_c.X_E * fill(0.3, n_salcs(small)) .+
               0.01 .* randn(rngc, nconf_c)
        ds_c = SLCEDataset(small, cfg_c, en_c)

        cv = cross_validate(ds_c, OLS(); nfolds = 3)
        @test cv.nfolds == 3 && cv.seed == 1 && cv.torque_weight == 0.0
        @test sum(cv.n_holdout) == nconf_c
        @test all(isfinite, cv.score) && all(isfinite, cv.rmse_energy)
        @test all(isnan, cv.rmse_torque) && isnan(cv.pooled_rmse_torque)
        # energy-only: score = MSE_E (rmse stores the sqrt, so equality is to 1 ulp)
        @test cv.score ≈ cv.rmse_energy .^ 2 rtol = 1e-14
        @test cv.pooled_score ≈ cv.pooled_rmse_energy^2 rtol = 1e-14

        # manual reconstruction of one fold: same folds, same fit, same residuals
        folds_c = SLCE._grouped_folds(collect(1:nconf_c), 3, 1)
        ho1 = findall(==(1), folds_c)
        f1 = fit(SLCEFit, ds_c[findall(!=(1), folds_c)], OLS())
        h1 = ds_c[ho1]
        @test cv.n_holdout[1] == length(ho1)
        @test cv.rmse_energy[1] ≈
              sqrt(mean(abs2, h1.y_E .- (f1.j0 .+ h1.X_E * f1.jphi))) rtol = 1e-12

        # pooled MSE = holdout-size-weighted mean of the per-fold MSEs
        @test cv.pooled_rmse_energy^2 ≈
              sum(cv.n_holdout .* cv.rmse_energy .^ 2) / nconf_c rtol = 1e-12

        # deterministic in the seed; a different seed re-deals the folds
        @test cross_validate(ds_c, OLS(); nfolds = 3).rmse_energy == cv.rmse_energy
        @test cross_validate(ds_c, OLS(); nfolds = 3, seed = 2).rmse_energy !=
              cv.rmse_energy

        # torque co-fit fixture: both error axes reported, even at torque_weight = 0
        tq_c = [randn(rngc, 3, 2) for _ = 1:nconf_c]
        ds_ct = SLCEDataset(small, cfg_c, en_c, tq_c)
        cvt = cross_validate(ds_ct, OLS(); torque_weight = 0.4, nfolds = 3)
        @test all(isfinite, cvt.rmse_torque) && isfinite(cvt.pooled_rmse_torque)
        @test cvt.score ≈ 0.6 .* cvt.rmse_energy .^ 2 .+ 0.4 .* cvt.rmse_torque .^ 2
        @test cvt.pooled_score ≈ 0.6 * cvt.pooled_rmse_energy^2 +
                                 0.4 * cvt.pooled_rmse_torque^2
        cvt0 = cross_validate(ds_ct, OLS(); torque_weight = 0.0, nfolds = 3)
        @test all(isfinite, cvt0.rmse_torque)           # measured despite w = 0
        @test cvt0.score ≈ cvt0.rmse_energy .^ 2 rtol = 1e-14

        # works with a penalized estimator (the intended comparison use)
        glab_c = salc_groups(small)
        est_c = GroupAdaptiveRidge(glab_c, ones(maximum(glab_c)); lambda = 1e-4)
        cvg = cross_validate(ds_c, est_c; nfolds = 3)
        @test all(isfinite, cvg.score)

        # fold reduction: 7 configs support only 2 folds of >= 3 configs
        ds7 = ds_c[1:7]
        cv7 = @test_logs (:warn, r"reducing CV folds") match_mode = :any begin
            cross_validate(ds7, OLS(); nfolds = 5)
        end
        @test cv7.nfolds == 2 && sum(cv7.n_holdout) == 7
        # minimum valid size: 6 configs at nfolds = 2 succeeds without a warning
        cv6 = @test_logs cross_validate(ds_c[1:6], OLS(); nfolds = 2)
        @test cv6.nfolds == 2 && sum(cv6.n_holdout) == 6

        # Tables + show smoke
        tbl = Tables.columntable(cv)
        @test keys(tbl) == (:fold, :n_holdout, :score, :rmse_energy, :rmse_torque,
                            :rmse_force)
        @test tbl.fold == [1, 2, 3] && tbl.rmse_energy == cv.rmse_energy
        # an energy-only dataset reports neither derivative axis
        @test all(isnan, cv.rmse_force) && isnan(cv.pooled_rmse_force)
        @test !occursin("rmse_F", sprint(show, MIME"text/plain"(), cv))
        @test occursin("pooled", sprint(show, MIME"text/plain"(), cvt))
        @test occursin("rmse_T", sprint(show, MIME"text/plain"(), cvt))
        @test !occursin("rmse_T", sprint(show, MIME"text/plain"(), cv))

        # validation errors
        @test_throws ArgumentError cross_validate(ds_c, OLS(); torque_weight = 1.5)
        @test_throws ArgumentError cross_validate(ds_c, OLS(); torque_weight = 0.5)
        @test_throws ArgumentError cross_validate(ds_c, OLS(); nfolds = 1)
        @test_throws ArgumentError cross_validate(ds_c[1:5], OLS())
        @test_throws ArgumentError cross_validate(ds_c, FixedCoefficients(
            zeros(n_salcs(small))))
        @test_throws ArgumentError cross_validate(ds_c, AdaptiveLasso(
            pilot = FixedCoefficients(zeros(n_salcs(small)))))
    end
end

@testset "diagnostics refuse a refit result (review 2026-08-11 M3)" begin
    # `effective_dof`/`gcv` reconstruct the FULL design from dataset + reparam; on a
    # refit that is not the problem that was solved — measured before the guard:
    # effective_dof(refit) = 40.0 where ≈ 5 was honest, gcv a silent Inf.
    lat = Lattice(Matrix(3.0 * I(3)))
    crystal = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    basis = SLCEBasis(crystal, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2],
                                        soc = false))
    rng = MersenneTwister(23)
    configs = [randcfg(rng, 2) for _ = 1:30]
    y = randn(rng, 30)
    ds = SLCEDataset(basis, configs, y)
    f = fit(SLCEFit, ds, Ridge(0.1))
    @test f.support === nothing
    @test isfinite(gcv(f)) && effective_dof(f) > 1.0     # the direct fit is served
    r = refit(f)
    @test r.support isa Vector{Int}
    for diag in (effective_dof, gcv)
        err = try
            diag(r)
            nothing
        catch e
            e
        end
        @test err isa ArgumentError
        @test occursin("refit", err.msg)
    end
    # a group estimator is refused up front with the actual cause named — it used
    # to die deeper with a DimensionMismatch blaming a "different SLCEBasis"
    gar = GroupAdaptiveRidge(basis; lambda = 1e-3)
    fg = fit(SLCEFit, ds, gar)
    errg = try
        refit(fg, gar)
        nothing
    catch e
        e
    end
    @test errg isa ArgumentError
    @test occursin("column_groups", errg.msg)
    @test refit(fg) isa SLCEFit                          # the default OLS de-bias works
end
