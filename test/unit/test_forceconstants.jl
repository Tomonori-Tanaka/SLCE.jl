# Force constants and the dynamical matrix (src/slce/forceconstants.jl) — the physics
# deliverables of the displacement channel. The constants are claimed to be EXACT
# derivatives of the model's energy, so every gate here is against the production
# evaluator: a finite-difference Hessian (order 2) and third derivative (order 3),
# the index-ordering symmetry, the reciprocal-space properties of D(q), and the
# acoustic modes an ASR-satisfying model must have at Γ.

using Test
using SLCE
using SLCE: build_asr, salcs
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_fc_cfg(rng, n) = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:n])

# Γ-restricted Hessian of the model's own energy: the uniform-per-atom displacement
# field the evaluator takes, differentiated by central differences.
function _fd_hessian(model, e, nat; h = 1e-5)
    H = zeros(3nat, 3nat)
    δ(i, x) = (z = zeros(3, nat); z[x, i] += h; z)
    for a = 1:nat, α = 1:3, b = 1:nat, β = 1:3
        d1, d2 = δ(a, α), δ(b, β)
        H[3(a - 1) + α, 3(b - 1) + β] =
            (predict_energy(model, e, d1 .+ d2) - predict_energy(model, e, d1 .- d2) -
             predict_energy(model, e, -d1 .+ d2) + predict_energy(model, e, -d1 .- d2)) /
            (4h^2)
    end
    return H
end

# Σ_R of the anchored constants, flattened atom-major / Cartesian-minor — the object
# that must equal the Γ-restricted derivative (each cell sees the same displacement,
# so every lattice shift folds onto the same pair of atoms).
function _gamma_sum(fcs, nat)
    S = zeros(ntuple(_ -> 3nat, fcs.order))
    for ((atoms, _), T) in fcs.constants
        for cidx in CartesianIndices(T)
            idx = ntuple(i -> 3 * (atoms[i] - 1) + cidx[i], fcs.order)
            S[idx...] += T[cidx]
        end
    end
    return S
end

@testset "force constants and the dynamical matrix" begin
    rng = MersenneTwister(5)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    nat = 2
    # pure-spin + coupled + lattice-only, so the constants pick up BOTH the
    # spin-dressed and the bare displacement channels
    b = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (nbody = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1),
        Sector(disp = (degree = 2,), nbody = 1:2, cutoff = 1.1)]))
    model = SLCEModel(b, 0.3, randn(rng, n_salcs(b)))
    e = _fc_cfg(rng, nat)
    fcs = force_constants(model; spins = e, order = 2)

    @testset "the constants are the energy's second derivatives" begin
        @test fcs isa ForceConstantSet && fcs.order == 2
        @test !isempty(fcs.constants)
        @test all(k -> iszero(k[2][1]), keys(fcs.constants))      # anchored on cell 0
        @test all(k -> length(k[1]) == length(k[2]) == 2, keys(fcs.constants))
        @test all(T -> size(T) == (3, 3), values(fcs.constants))
        H = _fd_hessian(model, e, nat)
        @test maximum(abs, _gamma_sum(fcs, nat) .- H) < 1e-4 * max(1.0, norm(H))
        @test norm(H) > 1.0                                       # not a trivial zero
        @test occursin("order = 2", sprint(show, fcs))
    end

    @testset "order = 1 is ∂E/∂u, i.e. MINUS predict_force" begin
        # `order >= 1` is explicitly permitted and was covered by nothing (the file
        # tested orders 2 and 3 and the order-0 refusal). A user asking for "the
        # forces" at order 1 gets them negated; the sign convention `f = −∂E/∂u` lives
        # in exactly two other places in the package, each with its own gate, so this
        # third entrance needs one too.
        # The file's main fixture carries displacement DEGREE 2 only, so its order-1
        # set is empty and the relation would hold vacuously. Build a degree-1 basis:
        # that content is exactly the reference-force channel this is about.
        b1 = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 1,), nbody = 2, cutoff = 1.1),
            Sector(disp = (degree = 1,), nbody = 1:2, cutoff = 1.1)]))
        m1 = SLCEModel(b1, 0.0, randn(rng, n_salcs(b1)))
        f1 = force_constants(m1; spins = e, order = 1)
        @test f1.order == 1
        g = zeros(3, nat)
        for ((atoms, _), T) in f1.constants
            g[:, atoms[1]] .+= T
        end
        @test maximum(abs, g) > 1e-6        # teeth: a real gradient, not all zeros
        @test isapprox(g, -predict_force(m1, e, zeros(3, nat)); atol = 1e-10)
        # and it is genuinely the OPPOSITE sign, not a coincidence of symmetry
        @test !isapprox(g, predict_force(m1, e, zeros(3, nat)); atol = 1e-6)
    end

    @testset "the R labels are real bond vectors, not bookkeeping" begin
        # `Σ_R Φ(R)`, the transpose relation and the Γ acoustic-mode gate are all blind
        # to the lattice shifts themselves: multiplying every stored `R` by 2 leaves all
        # of this file's other assertions green while producing a completely wrong
        # dispersion. The periodicity gate above cannot see it either — it constrains
        # the phase, not the labels. Pin the labels geometrically instead: every key
        # must describe a pair that really sits inside the cutoff the basis was built
        # with.
        A = cr.lattice.vectors
        pos = cartesian_positions(cr)
        cut = 1.1                                     # the fixture's sector cutoff
        ds = Float64[]
        for ((atoms, shifts), _) in fcs.constants
            d = norm(pos[:, atoms[2]] .+ A * shifts[2] .- pos[:, atoms[1]])
            push!(ds, d)
            @test d <= cut * (1 + 1e-8)
        end
        @test !isempty(ds)
        @test maximum(ds) > 0.5      # teeth: real bonds present, not an on-site-only set
    end

    @testset "index ordering: Φ[(a,0),(b,R)] = Φ[(b,0),(a,−R)]ᵀ" begin
        # the reverse ordering is a SEPARATE key, related by transposition — the
        # relation is a property to check, never a storage shortcut
        for ((atoms, shifts), T) in fcs.constants
            rk = ([atoms[2], atoms[1]], [zero(shifts[2]), -shifts[2]])
            @test haskey(fcs.constants, rk)
            @test fcs.constants[rk] == transpose(T)
        end
    end

    @testset "dynamical_matrix: Hermiticity, conjugation, Γ" begin
        q = [0.13, -0.27, 0.41]
        Dq = dynamical_matrix(fcs, q)
        @test size(Dq) == (3nat, 3nat)
        @test Dq ≈ Dq'                                        # Hermitian
        @test dynamical_matrix(fcs, -q) ≈ conj(Dq)
        @test real(dynamical_matrix(fcs, [0.0, 0.0, 0.0])) ≈ _gamma_sum(fcs, nat)
        # a nonzero q must actually change the matrix (the phase factor is live)
        @test !isapprox(Dq, dynamical_matrix(fcs, [0.0, 0.0, 0.0]))
        # PERIODICITY IN A RECIPROCAL LATTICE VECTOR. Everything above this line is
        # invariant under rescaling the phase argument — Hermiticity, D(−q) = conj D(q)
        # and "the phase is not constant" all survive `cis(2π q·R) → cis(α q·R)` for any
        # α, which is why the documented convention ("q in FRACTIONAL reciprocal
        # coordinates, phase exp(2πi q·R)") used to rest on nothing executable: the
        # mutation α = 1 passed all 77 assertions in this file. D(q + G) = D(q) for
        # integer G holds if and only if q is fractional AND the phase is exactly 2π.
        for G in ([1, 0, 0], [0, -1, 0], [2, 1, -3])
            @test dynamical_matrix(fcs, q .+ G) ≈ Dq
        end
        # mass weighting divides by √(MₐM_b) entrywise
        masses = [55.845, 26.982]
        Dm = dynamical_matrix(fcs, q; masses = masses)
        for a = 1:nat, α = 1:3, bb = 1:nat, β = 1:3
            @test Dm[3(a-1)+α, 3(bb-1)+β] ≈
                  Dq[3(a-1)+α, 3(bb-1)+β] / sqrt(masses[a] * masses[bb])
        end
        @test_throws DimensionMismatch dynamical_matrix(fcs, [0.0, 0.0])
        @test_throws DimensionMismatch dynamical_matrix(fcs, q; masses = [1.0])
        @test_throws ArgumentError dynamical_matrix(fcs, q; masses = [1.0, -1.0])
    end

    @testset "acoustic modes at Γ (the ASR payoff)" begin
        # A model in the ASR null space must have exactly three zero eigenvalues at
        # q = 0 — the acoustic branches — and per-atom row sums that vanish. An
        # unconstrained model of the SAME basis must not: that is the teeth, and the
        # physical statement of what `fit(...; asr = true)` buys.
        rep = build_asr(b)
        feas = SLCEModel(b, 0.0, rep.Z * randn(rng, size(rep.Z, 2)))
        ffeas = force_constants(feas; spins = e, order = 2)
        Sf = _gamma_sum(ffeas, nat)
        # columns run atom-major / Cartesian-minor, so β is the FAST index: the
        # reshape is (row, β, b) and the sum over lattice partners is over `b`
        @test maximum(abs, sum(reshape(Sf, 3nat, 3, nat); dims = 3)) <
              1e-10 * max(1.0, norm(Sf))                     # Σ_{b,R} Φ = 0
        ev = sort(real(eigvals(Hermitian(dynamical_matrix(ffeas, zeros(3))))))
        @test count(x -> abs(x) < 1e-9 * max(1.0, norm(Sf)), ev) == 3
        evb = sort(real(eigvals(Hermitian(dynamical_matrix(fcs, zeros(3))))))
        @test count(x -> abs(x) < 1e-9 * max(1.0, norm(_gamma_sum(fcs, nat))), evb) == 0
    end

    @testset "the constants depend on the spin configuration" begin
        # the whole point of a spin–lattice expansion: the lattice dynamics is a
        # function of the magnetic state
        e2 = _fc_cfg(rng, nat)
        f2 = force_constants(model; spins = e2, order = 2)
        @test maximum(abs, _gamma_sum(f2, nat) .- _gamma_sum(fcs, nat)) > 1e-3
        # and each is exact for ITS configuration
        @test maximum(abs, _gamma_sum(f2, nat) .- _fd_hessian(model, e2, nat)) < 1e-3
        # the configuration is recorded on the set
        @test f2.spins == e2 && fcs.spins == e
    end

    @testset "order 3 against a third derivative" begin
        b3 = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 3, sectors = [
            Sector(disp = (degree = 3,), nbody = 1:2, cutoff = 1.1)]))
        m3 = SLCEModel(b3, 0.0, randn(rng, n_salcs(b3)))
        f3 = force_constants(m3; spins = e, order = 3)
        @test f3.order == 3 && !isempty(f3.constants)
        @test all(T -> size(T) == (3, 3, 3), values(f3.constants))
        S3 = _gamma_sum(f3, nat)
        # a degree-3 model is exactly cubic in u, so the finite difference is exact
        h = 3e-3
        δ(i, x) = (z = zeros(3, nat); z[x, i] += h; z)
        function fd3(a, α, bb, β, c, γ)
            da, db, dc = δ(a, α), δ(bb, β), δ(c, γ)
            s = 0.0
            for sa in (1, -1), sb in (1, -1), sc in (1, -1)
                s += sa * sb * sc * predict_energy(m3, e, sa .* da .+ sb .* db .+ sc .* dc)
            end
            return s / (8h^3)
        end
        worst = maximum(abs(S3[3(a-1)+α, 3(bb-1)+β, 3(c-1)+γ] - fd3(a, α, bb, β, c, γ))
                        for a = 1:nat, α = 1:3, bb = 1:nat, β = 1:3, c = 1:nat, γ = 1:3)
        @test norm(S3) > 1.0
        @test worst < 1e-8 * norm(S3)
        # order 2 of a purely cubic model is empty: no term has displacement degree 2
        @test isempty(force_constants(m3; spins = e, order = 2).constants)
        @test_throws ArgumentError dynamical_matrix(f3, zeros(3))
    end

    @testset "refusals and degenerate inputs" begin
        # a pure-spin model has no displacement content at all
        bs = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
        ms = SLCEModel(bs, 0.0, randn(rng, n_salcs(bs)))
        @test isempty(force_constants(ms; spins = e, order = 2).constants)
        # and so does the clamped-ion restriction of a joint one
        @test isempty(force_constants(restrict(model, :spin); spins = e).constants)
        @test_throws ArgumentError force_constants(model; spins = e, order = 0)
        @test_throws DimensionMismatch force_constants(model; spins = zeros(3, 3))
        # a zero model has no constants (the all-cancelled tuples are dropped)
        @test isempty(force_constants(SLCEModel(b, 1.0, zeros(n_salcs(b)));
                                      spins = e).constants)
    end
end
