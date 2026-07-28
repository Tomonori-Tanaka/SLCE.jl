# Force constants and the dynamical matrix (src/slce/forceconstants.jl) — the physics
# deliverables of the displacement channel. The constants are claimed to be EXACT
# derivatives of the model's energy, so every gate here is against the production
# evaluator: a finite-difference Hessian (order 2) and third derivative (order 3),
# the index-ordering symmetry, the reciprocal-space properties of D(q), and the
# acoustic modes an ASR-satisfying model must have at Γ.

using Test
using SLCE
using SLCE: build_asr, salcs, _assemble_spacegroup, build_neighbor_list,
            build_clusters, build_salc_basis, _superset_cutoff, has_spin
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

# ---------------------------------------------------------------------------
# Magnetic-symmetry ledger (the stripe-AFM fixture used by the last testset).
# ---------------------------------------------------------------------------

# `SLCEBasis` derives the space group from the crystal, and the core test env has no
# Spglib — so assemble the group by hand and feed it to the same three builders the
# constructor calls. `_assemble_spacegroup` validates closure, so a generation that
# missed an element throws here rather than silently under-symmetrizing the basis.
function _basis_with_sg(cr, sg, spec)
    nl = build_neighbor_list(cr, _superset_cutoff(spec), MinimumImage())
    cl = build_clusters(cr, nl, sg; nbody = spec.nbody, selection = MinimumImage(),
                        cutoff = spec.cutoff)
    sb = build_salc_basis(cr, sg, cl, spec; neighbors = nl, selection = MinimumImage())
    return SLCEBasis(cr, sg, sb, spec)
end

# Simple-tetragonal a = 3, doubled in x and y: four Fe sites, P4/mmm, 64 operations
# (16 point ops × 4 sublattice translations). The spins are a stripe antiferromagnet
# along z whose sign follows x, so the doubling translations are exactly the elements
# that flip it — the antiunitary half of the magnetic group.
function _stripe_afm()
    lat = Lattice(SMatrix{3,3,Float64}([6.0 0 0; 0 6.0 0; 0 0 3.0]))
    frac = [0.0 0.5 0.0 0.5; 0.0 0.0 0.5 0.5; 0.0 0.0 0.0 0.0]
    cr = Crystal(lat, frac, [1, 1, 1, 1], ["Fe"])
    C4 = SMatrix{3,3,Float64}([0 -1 0; 1 0 0; 0 0 1])   # +90° about z, exact in frac
    Mx = SMatrix{3,3,Float64}([1 0 0; 0 -1 0; 0 0 -1])  # C2 about x
    pts = [SMatrix{3,3,Float64}(I)]
    for _ = 1:6, g in (C4, Mx, SMatrix{3,3,Float64}(-I)), p in copy(pts)
        q = g * p
        any(x -> x ≈ q, pts) || push!(pts, q)
    end
    trs = [SVector{3,Float64}(0, 0, 0), SVector{3,Float64}(0.5, 0, 0),
           SVector{3,Float64}(0, 0.5, 0), SVector{3,Float64}(0.5, 0.5, 0)]
    rots = [p for p in pts for _ in trs]
    tran = [t for _ in pts for t in trs]
    sg = _assemble_spacegroup(cr, rots, tran, "P4/mmm", 123; tol = 1e-8)
    e = [0.0 0.0 0.0 0.0; 0.0 0.0 0.0 0.0; 1.0 -1.0 1.0 -1.0]
    return cr, sg, e
end

_opdata(sg, o) = (Matrix(sg.ops[o].rotation_cart), sg.map_sym[:, o])

# Spins are AXIAL: `e → det(R)·R·e`, then permuted by the operation's site map. `:D`
# preserves the magnetic state, `:Dp` reverses it (so `g·T` is a symmetry and its
# rotation part still constrains a time-reversal-EVEN tensor), `:out` is neither.
function _spinclass(sg, o, e)
    R, p = _opdata(sg, o)
    ep = zeros(size(e))
    for a in axes(e, 2)
        ep[:, p[a]] = det(R) * R * e[:, a]
    end
    return isapprox(ep, e; atol = 1e-10) ? :D :
           isapprox(ep, -e; atol = 1e-10) ? :Dp : :out
end

# `H_{p(a)p(b)} = R H_{ab} Rᵀ` — the transformation a Γ-Hessian must satisfy.
function _act(H, R, p, nat)
    K = zeros(size(H))
    for a = 1:nat, b = 1:nat
        K[3(p[a] - 1) + 1:3p[a], 3(p[b] - 1) + 1:3p[b]] =
            R * H[3(a - 1) + 1:3a, 3(b - 1) + 1:3b] * R'
    end
    return K
end

# Dimension of the space of symmetric Γ-Hessians invariant under `ops`, as the trace
# of the group-averaging projector written in the symmetric-matrix basis.
function _invdim(ops, nat)
    n = 3nat
    bas = [(i, j) for i = 1:n for j = i:n]
    M = zeros(length(bas), length(bas))
    for (c, (i, j)) in enumerate(bas)
        H = zeros(n, n)
        H[i, j] += 1
        H[j, i] += 1
        i == j && (H[i, j] = 1)
        A = zeros(n, n)
        for (R, p) in ops
            A .+= _act(H, R, p, nat)
        end
        A ./= length(ops)
        for (r, (k, l)) in enumerate(bas)
            M[r, c] = A[k, l]
        end
    end
    return round(Int, tr(M))
end

# Σ_R of the anchored constants as a 3N × 3N matrix.
function _gamma_matrix(model, e, nat)
    H = zeros(3nat, 3nat)
    for ((atoms, _), T) in force_constants(model; spins = e, order = 2).constants
        for α = 1:3, β = 1:3
            H[3(atoms[1] - 1) + α, 3(atoms[2] - 1) + β] += T[α, β]
        end
    end
    return H
end

@testset "force constants and the dynamical matrix" begin
    rng = MersenneTwister(5)
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    nat = 2
    # pure-spin + coupled + lattice-only, so the constants pick up BOTH the
    # spin-dressed and the bare displacement channels
    b = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
        Sector(spin = (sites = 1:2,), cutoff = 1.1),
        Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),
        Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)]))
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
            Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = 1.1),
            Sector(disp = (degree = 1,), sites = 1:2, cutoff = 1.1)]))
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
            Sector(disp = (degree = 3,), sites = 1:2, cutoff = 1.1)]))
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
            Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
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

    # ---------------------------------------------------------------------
    # The package's headline physics claim, and the two ways to lose it.
    #
    # Force constants are time-reversal EVEN, so an antiunitary element `g·T` of the
    # magnetic space group constrains Φ through its rotation part `g` exactly as a
    # unitary element does. The correct invariance group is therefore `D ∪ D′`, not
    # the unitary halving subgroup `D` — and the joint path lands on it for free: the
    # SALCs are projected with the paramagnetic grey group `G × {1, T}`, and fixing
    # `spins` reduces that to the magnetic stabilizer with nothing declared.
    #
    # The ledger below is the whole argument in three numbers, and neither neighbour
    # is reachable by accident: a lattice-only basis imposes the paramagnetic group
    # (too large — it zeroes what the order breaks), and relabelling the sublattices
    # as distinct species would impose `D` alone (too small — it drops `D′`).
    # ---------------------------------------------------------------------
    @testset "magnetic symmetry: the joint path imposes D ∪ D′, exactly" begin
        cr4, sg4, eafm = _stripe_afm()
        nat4 = 4
        @test length(sg4.ops) == 64
        cls = [_spinclass(sg4, o, eafm) for o in eachindex(sg4.ops)]
        @test count(==(:D), cls) == 16
        @test count(==(:Dp), cls) == 16          # the antiunitary half is NOT empty
        @test count(==(:out), cls) == 32
        pick(k) = [_opdata(sg4, o) for o in eachindex(sg4.ops) if cls[o] == k]
        # 78 raw parameters in a symmetric 12 × 12; the three groups admit:
        @test _invdim([_opdata(sg4, o) for o in eachindex(sg4.ops)], nat4) == 7
        @test _invdim(vcat(pick(:D), pick(:Dp)), nat4) == 12    # physically correct
        @test _invdim(pick(:D), nat4) == 16                     # unitary only

        spec4 = BasisSpec(cr4; lmax = 2, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 3.1),
            Sector(spin = (sites = 2, lmax = 2), disp = (degree = 2,), sites = 2,
                   cutoff = 3.1),
            Sector(spin = (sites = 1, lmax = 2), disp = (degree = 2,), sites = 1:2,
                   cutoff = 3.1)])
        bj = _basis_with_sg(cr4, sg4, spec4)
        mj = SLCEModel(bj, 0.0, randn(MersenneTwister(11), n_salcs(bj)))
        Hj = _gamma_matrix(mj, eafm, nat4)
        viol(ops) = maximum(o -> maximum(abs, _act(Hj, o[1], o[2], nat4) - Hj), ops)
        tol = 1e-10 * norm(Hj)
        @test viol(pick(:D)) < tol
        @test viol(pick(:Dp)) < tol              # the kill shot: antiunitary too
        # and it is not invariant under everything — the constraint has content
        @test viol(pick(:out)) > 1e-3 * norm(Hj)

        # Over-symmetrization, made concrete. Every operation of the paramagnetic
        # group survives in a lattice-only basis, including the C4 that exchanges the
        # x and y axes — so its on-site Φ is forced isotropic in-plane no matter what
        # the magnetic order does. The stripe breaks that axis exchange, and the
        # joint model sees it.
        bl = _basis_with_sg(cr4, sg4, BasisSpec(cr4; lmax = 0, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 3.1)]))
        Hl = _gamma_matrix(SLCEModel(bl, 0.0, randn(MersenneTwister(12), n_salcs(bl))),
                           eafm, nat4)
        @test Hl[1, 1] == Hl[2, 2]               # exactly, by projection
        @test abs(Hj[1, 1] - Hj[2, 2]) > 1e-3 * norm(Hj)

        # Two magnetic states, one model: the constants really do depend on `spins`.
        Hfm = _gamma_matrix(mj, ones(3, nat4) ./ √3, nat4)
        @test norm(Hj - Hfm) > 1e-3 * norm(Hj)
    end

    # The other silent way to lose it: a joint basis whose only spin-carrying
    # displacement sector sits at `degree = 1`. That row feeds the FORCES; nothing
    # reaches order 2, so Φ comes out bit-identical for every magnetic state while
    # the fit reports a perfect r². The warning is the only signal a user gets.
    @testset "a spin-blind order: degree = 1 magnetoelastic warns, degree = 2 does not" begin
        cr4, sg4, eafm = _stripe_afm()
        # The quiet cases go FIRST: the warning carries `maxlog = 1`, so checking a
        # silent call after a noisy one would assert nothing.
        lat_only = BasisSpec(cr4; lmax = 0, pmax = 2,
                             sectors = [Sector(disp = (degree = 2,), sites = 1:2,
                                               cutoff = 3.1)])
        ml = SLCEModel(_basis_with_sg(cr4, sg4, lat_only), 0.0,
                       randn(MersenneTwister(13), n_salcs(_basis_with_sg(cr4, sg4,
                                                                        lat_only))))
        @test_logs force_constants(ml; spins = eafm)     # no spin content to lose
        @test_logs force_constants(restrict(model, :spin); spins = e)  # no disp content
        # the dJ/dr row at degree 2 does reach order 2, and says nothing
        seeing = _basis_with_sg(cr4, sg4, BasisSpec(cr4; lmax = 2, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 3.1),
            Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 3.1)]))
        ms = SLCEModel(seeing, 0.0, randn(MersenneTwister(13), n_salcs(seeing)))
        @test_logs force_constants(ms; spins = eafm)

        # ...and the same row at degree 1 — the spelling both the README and the
        # basis guide use for magnetoelastic coupling — does not.
        blind = _basis_with_sg(cr4, sg4, BasisSpec(cr4; lmax = 2, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), sites = 1:2, cutoff = 3.1),
            Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = 3.1)]))
        mb = SLCEModel(blind, 0.0, randn(MersenneTwister(13), n_salcs(blind)))
        @test any(s -> any(has_spin, s.decors), salcs(blind))   # spin content exists
        @test_logs (:warn, r"do not depend on `spins`") force_constants(mb; spins = eafm)
        # and it is not crying wolf: the constants ARE spin-independent here
        @test _gamma_matrix(mb, eafm, 4) == _gamma_matrix(mb, ones(3, 4) ./ √3, 4)
    end
end
