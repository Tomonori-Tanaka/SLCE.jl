# Synthetic recovery (plan A, engine level; zero core changes): Cartesian
# ground-truth spin–lattice models → sampled energies → OLS on the SALC design
# → held-out energies match to machine precision AND the Cartesian input
# constants are recovered through probe configurations. In-span recovery is a
# theoretical guarantee (the tesseral/Racah basis is an exact linear transform
# of Cartesian polynomials), so any drift here is a conventions bug in the
# construction/evaluation chain. Negative faces: content excluded by the
# selection rules (DMI under soc = false; the twist under L_S = 0) must be
# demonstrably irrecoverable, and in-span-but-absent channels must fit to
# zero coefficients (discrimination).

using Test
using SLCE
using SLCE: evaluate_salc, build_neighbor_list, build_clusters,
    _assemble_spacegroup
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

# Design matrix with intercept: X[i, 1] = 1, X[i, k+1] = Φ_k(e_i, u_i).
function _rc_design(svec, cfgs)
    X = Matrix{Float64}(undef, length(cfgs), length(svec) + 1)
    for (i, (e, u)) in enumerate(cfgs)
        X[i, 1] = 1.0
        for (k, s) in enumerate(svec)
            X[i, k + 1] = evaluate_salc(s, e, u)
        end
    end
    return X
end

_rc_pred(β, svec, e, u) =
    β[1] + sum(β[k + 1] * evaluate_salc(svec[k], e, u) for k in eachindex(svec))

# OLS fit on the first `nfit` configs, relative holdout residual on the rest.
# The fit design must be full column rank (unique β — the coefficient
# discrimination assertions depend on it), and the holdout nonempty.
function _rc_fit(svec, cfgs, E, nfit)
    @assert 1 <= nfit < length(cfgs)
    X = _rc_design(svec, cfgs)
    @test rank(X[1:nfit, :]) == size(X, 2)
    β = X[1:nfit, :] \ E[1:nfit]
    ho = X[(nfit + 1):end, :] * β .- E[(nfit + 1):end]
    return β, norm(ho) / max(norm(E[(nfit + 1):end]), 1e-30)
end

_rc_unit(rng) = normalize(randn(rng, 3))

@testset "synthetic recovery (plan A)" begin
    @testset "A1: exchange striction + DMI on the bent Fe–O–Fe unit" begin
        # Ground truth (Cartesian):
        #   E = E0 + [J + a_s (u₁−u₂)·x̂ + a_y (u₁+u₂)·ŷ + a_O u_O·ŷ]·(ê₁·ê₂)
        # — the gate-(f) content as one model: exchange + bond-stretch dJ/dr +
        # pair sway + ligand superexchange path. All soc = false in-span.
        # No ASR is imposed here (2a_y + a_O ≠ 0 deliberately): the basis
        # spans translation-non-invariant directions by design — the ASR
        # enters the fit layer as exact equality constraints at M3 (design
        # record §6), not the basis.
        L = 8.0
        c = SVector{3,Float64}(0.5, 0.5, 0.5)
        offs = [SVector{3,Float64}(1.0, 0.0, 0.0), SVector{3,Float64}(-1.0, 0.0, 0.0),
                SVector{3,Float64}(0.0, 1.0, 0.0)]
        frac = reduce(hcat, [c + o / L for o in offs])
        cr = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 1, 2], ["Fe", "O"])
        rots = [SMatrix{3,3,Float64}(I),
                SMatrix{3,3,Float64}([1.0 0 0; 0 1 0; 0 0 -1]),
                SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 -1]),
                SMatrix{3,3,Float64}([-1.0 0 0; 0 1 0; 0 0 1])]
        trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
        sg = _assemble_spacegroup(cr, rots, trans, "C2v", 0; tol = 1e-6)
        nl = build_neighbor_list(cr, 2.3)
        cs = build_clusters(cr, nl, sg; nbody = 3)
        mkspec(soc_pair) = BasisSpec(cr; lmax = ["Fe" => 1, "O" => 0],
                                     pmax = ["Fe" => 1, "O" => 1], sectors = [
            Sector(spin = [1, 1], nbody = 2, cutoff = 2.1, soc = soc_pair),
            Sector(spin = [1, 1], disp = (degree = 1,), nbody = 2, cutoff = 2.1,
                   soc = false),
            Sector(spin = [1, 1], disp = (degree = 1,), nbody = 3, cutoff = 2.3,
                   soc = false)])
        # (odd max displacement degree ⇒ the boundedness warning must fire)
        spec0 = @test_logs (:warn, r"odd") mkspec(false)
        b0 = SLCE.build_salc_basis(cr, sg, cs, spec0; neighbors = nl)
        # count pin (an extra spurious in-span column would fit to ~0 and pass
        # the holdout silently): 1 pure spin + 2 bond-stretch + 1 ligand
        @test length(b0.salcs) == 4
        @test all(s -> s.key.L_S == 0, b0.salcs)

        E0, J, a_s, a_y, a_O = 0.37, 1.0, 0.45, -0.28, 0.61
        gt(e, u) = E0 + (J + a_s * (u[1, 1] - u[1, 2]) + a_y * (u[2, 1] + u[2, 2]) +
                         a_O * u[2, 3]) * dot(e[:, 1], e[:, 2])
        rng = MersenneTwister(0xa1)
        cfgs = [(reduce(hcat, [_rc_unit(rng) for _ = 1:3]), randn(rng, 3, 3))
                for _ = 1:40]
        E = [gt(e, u) for (e, u) in cfgs]
        β, res = _rc_fit(b0.salcs, cfgs, E, 30)
        @test res < 1e-9
        # Cartesian constant recovery through probe configurations (the
        # back-transform face of the plan, realized operationally): each probe
        # isolates one input constant from the FITTED model.
        nprobe = 0
        for _ = 1:6
            e = reduce(hcat, [_rc_unit(rng) for _ = 1:3])
            sc = dot(e[:, 1], e[:, 2])
            abs(sc) < 0.1 && continue
            nprobe += 1
            J_rec = (_rc_pred(β, b0.salcs, e, zeros(3, 3)) - β[1]) / sc
            @test J_rec ≈ J rtol = 1e-8
            us = zeros(3, 3)
            us[1, 1], us[1, 2] = 0.5, -0.5              # unit bond stretch
            @test (_rc_pred(β, b0.salcs, e, us) - β[1]) / sc - J ≈ a_s rtol = 1e-7
            uy = zeros(3, 3)
            uy[2, 1] = uy[2, 2] = 0.5                   # unit pair sway
            @test (_rc_pred(β, b0.salcs, e, uy) - β[1]) / sc - J ≈ a_y rtol = 1e-7
            uo = zeros(3, 3)
            uo[2, 3] = 1.0                              # unit ligand shift
            @test (_rc_pred(β, b0.salcs, e, uo) - β[1]) / sc - J ≈ a_O rtol = 1e-7
        end
        @test nprobe >= 2
        # Negative: plant the C2v-allowed DMI D·(ê₁×ê₂)_z. Under soc = false
        # it is OUT of span — the holdout must fail loudly...
        D = 0.3
        gt2(e, u) = gt(e, u) + D * (e[1, 1] * e[2, 2] - e[2, 1] * e[1, 2])
        E2 = [gt2(e, u) for (e, u) in cfgs]
        _, res2 = _rc_fit(b0.salcs, cfgs, E2, 30)
        @test res2 > 1e-2
        # ...and with the pure-spin pair sector flipped to soc = true the very
        # same data recovers exactly, including the Cartesian D.
        spec1 = @test_logs (:warn, r"odd") mkspec(true)
        b1 = SLCE.build_salc_basis(cr, sg, cs, spec1; neighbors = nl)
        @test length(b1.salcs) == 7          # pure-spin pair L_S = 0/1/2 → 1/1/2
        @test count(s -> s.key.L_S == 1, b1.salcs) == 1
        β1, res1 = _rc_fit(b1.salcs, cfgs, E2, 30)
        @test res1 < 1e-9
        ed = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]    # ê₁ = x̂, ê₂ = ŷ (ê·ê = 0)
        D_rec = _rc_pred(β1, b1.salcs, ed, zeros(3, 3)) - β1[1]
        @test D_rec ≈ D rtol = 1e-8
    end

    @testset "A2: B₁/B₂ shell constants on the Fe(O)₆ unit" begin
        # Ground truth: per-bond C4v pair (transported over the O_h shell)
        #   q1_j = (3(n·v̂_j)² − 1)(u_j·v̂_j),  q2_j = (n·v̂_j)·(n·u_j^⊥)
        #   E = E0 + α Σ_j q1_j + β Σ_j q2_j
        # — the Cartesian two-constant magnetoelastic model of gate (e2).
        L = 10.0
        c = SVector{3,Float64}(0.5, 0.5, 0.5)
        dsh = 2.0
        shell = [SVector(1.0, 0.0, 0.0), SVector(-1.0, 0.0, 0.0),
                 SVector(0.0, 1.0, 0.0), SVector(0.0, -1.0, 0.0),
                 SVector(0.0, 0.0, 1.0), SVector(0.0, 0.0, -1.0)]
        offs = vcat([SVector{3,Float64}(0, 0, 0)], [dsh * v for v in shell])
        frac = reduce(hcat, [c + o / L for o in offs])
        cr = Crystal(Lattice(Matrix(L * I(3))), frac, [1, 2, 2, 2, 2, 2, 2],
                     ["Fe", "O"])
        rots = oh48_matrices()
        trans = [(SMatrix{3,3,Float64}(I) - R) * c for R in rots]
        sg = _assemble_spacegroup(cr, rots, trans, "Pm-3m", 0; tol = 1e-6)
        nl = build_neighbor_list(cr, 2.1)
        cs = build_clusters(cr, nl, sg; nbody = 2)
        spec = @test_logs (:warn, r"odd") BasisSpec(cr;
            lmax = ["Fe" => 2, "O" => 0], pmax = ["Fe" => 0, "O" => 1],
            sectors = [Sector(spin = [2], disp = (degree = 1,), nbody = 2,
                              cutoff = 2.1)])
        b = SLCE.build_salc_basis(cr, sg, cs, spec; neighbors = nl)
        @test length(b.salcs) == 2

        E0, α, βc = -0.11, 0.8, -0.55
        function gt(e, u)
            n = e[:, 1]
            acc = E0
            for (j, v) in enumerate(shell)
                uj = u[:, j + 1]
                nv = dot(n, v)
                uv = dot(uj, v)
                acc += α * (3 * nv^2 - 1) * uv + βc * nv * dot(n, uj .- uv .* v)
            end
            return acc
        end
        rng = MersenneTwister(0xa2)
        cfgs = [(reduce(hcat, [_rc_unit(rng) for _ = 1:7]), randn(rng, 3, 7))
                for _ = 1:24]
        E = [gt(e, u) for (e, u) in cfgs]
        β, res = _rc_fit(b.salcs, cfgs, E, 16)
        @test res < 1e-9
        # probe recovery of α and β on the +x bond alone
        n1 = n2 = 0
        for _ = 1:8
            e = reduce(hcat, [_rc_unit(rng) for _ = 1:7])
            n = e[:, 1]
            ux = zeros(3, 7)
            ux[1, 2] = 1.0                              # u_{+x} = x̂ (u·v̂ = 1)
            d1 = 3 * n[1]^2 - 1
            if abs(d1) > 0.1
                n1 += 1
                @test (_rc_pred(β, b.salcs, e, ux) - β[1]) / d1 ≈ α rtol = 1e-7
            end
            uy = zeros(3, 7)
            uy[2, 2] = 1.0                              # u_{+x} = ŷ (transverse)
            d2 = n[1] * n[2]
            if abs(d2) > 0.05
                n2 += 1
                @test (_rc_pred(β, b.salcs, e, uy) - β[1]) / d2 ≈ βc rtol = 1e-7
            end
        end
        @test n1 >= 2 && n2 >= 2
    end

    @testset "A3: chirality twist — recovery, discrimination, soc negative" begin
        # Ground truth: E = E0 + K (ê₁×ê₂)·(u₁×u₂) on the centrosymmetric
        # Fe–Fe bond (the gate-(g2) invariant, L_S = 1 alone). The fit must
        # put ALL weight on the (L_S = 1, Lf = 0) SALC and zero on the other
        # eight columns of the doubly-decorated family.
        latB = Lattice(Matrix(3.0 * I(3)))
        xtalB = Crystal(latB, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        rotsB = [R for R in oh48_matrices() if abs(R[1, 1]) == 1.0]
        trsB = [SVector{3,Float64}(0, 0, 0) for _ in rotsB]
        sgB = _assemble_spacegroup(xtalB, rotsB, trsB, "P4/mmm", 123; tol = 1e-5)
        nlB = build_neighbor_list(xtalB, 1.1)
        csB = build_clusters(xtalB, nlB, sgB; nbody = 2)
        mkspec(soc) = BasisSpec(xtalB; lmax = 1, pmax = 1, sectors = [
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1,
                   soc = soc)])
        b = SLCE.build_salc_basis(xtalB, sgB, csB, mkspec(true); neighbors = nlB)
        @test length(b.salcs) == 9

        E0, K = 0.05, 0.9
        gt(e, u) = E0 + K * dot(cross(e[:, 1], e[:, 2]), cross(u[:, 1], u[:, 2]))
        rng = MersenneTwister(0xa3)
        cfgs = [(reduce(hcat, [_rc_unit(rng) for _ = 1:2]), randn(rng, 3, 2))
                for _ = 1:60]
        E = [gt(e, u) for (e, u) in cfgs]
        β, res = _rc_fit(b.salcs, cfgs, E, 45)
        @test res < 1e-9
        ktw = findfirst(s -> s.key.L_S == 1 && s.key.Lf == 0, b.salcs)
        @test ktw !== nothing
        # discrimination: the twist column carries everything
        @test abs(β[ktw + 1]) > 1e-3
        for k in eachindex(b.salcs)
            k == ktw && continue
            @test abs(β[k + 1]) < 1e-8 * abs(β[ktw + 1])
        end
        # Cartesian K via the twist SALC's fixed evaluation ratio
        rats = Float64[]
        for _ = 1:6
            e = reduce(hcat, [_rc_unit(rng) for _ = 1:2])
            u = randn(rng, 3, 2) * 0.4
            F = dot(cross(e[:, 1], e[:, 2]), cross(u[:, 1], u[:, 2]))
            abs(F) < 1e-2 && continue
            push!(rats, evaluate_salc(b.salcs[ktw], e, u) / F)
        end
        @test length(rats) >= 2
        @test all(r -> isapprox(r, rats[1]; rtol = 1e-9), rats)   # fixed ratio
        K_rec = β[ktw + 1] * rats[1]
        @test K_rec ≈ K rtol = 1e-8
        # negative: the L_S = 0 subset cannot express the twist
        b0 = SLCE.build_salc_basis(xtalB, sgB, csB, mkspec(false); neighbors = nlB)
        @test all(s -> s.key.L_S == 0, b0.salcs)
        _, res0 = _rc_fit(b0.salcs, cfgs, E, 45)
        @test res0 > 1e-2
    end

    @testset "A4: spin-dependent force constants Φ⁰ + (ê·ê)Φ¹ at p = 2" begin
        # Ground truth on the D4h bond, relative-displacement form:
        #   E = E0 + u_r·Φ⁰·u_r + (ê₁·ê₂)(u_r·Φ¹·u_r),  u_r = u₁ − u₂,
        # with bond-frame tensors Φ⁰ = diag(a, b, b), Φ¹ = diag(cc, dd, dd).
        # In span of {pure-displacement degree-2 sectors} ∪ {spin pair ×
        # degree-2, soc = false}. The design is full column rank (β unique),
        # but the (a, b, cc, dd) → coefficient map is not diagonal, so the
        # PREDICTION is compared at probe configurations instead of the
        # per-column coefficients.
        latB = Lattice(Matrix(3.0 * I(3)))
        xtalB = Crystal(latB, [1 / 6 -1 / 6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
        rotsB = [R for R in oh48_matrices() if abs(R[1, 1]) == 1.0]
        trsB = [SVector{3,Float64}(0, 0, 0) for _ in rotsB]
        sgB = _assemble_spacegroup(xtalB, rotsB, trsB, "P4/mmm", 123; tol = 1e-5)
        nlB = build_neighbor_list(xtalB, 1.1)
        csB = build_clusters(xtalB, nlB, sgB; nbody = 2)
        spec = BasisSpec(xtalB; lmax = 1, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 2,), nbody = 2, cutoff = 1.1,
                   soc = false)])
        b = SLCE.build_salc_basis(xtalB, sgB, csB, spec; neighbors = nlB)
        @test !isempty(b.salcs)

        E0 = 0.2
        a, bb, cc, dd = 1.3, 0.7, -0.35, 0.15
        P0 = Diagonal([a, bb, bb])
        P1 = Diagonal([cc, dd, dd])
        function gt(e, u)
            ur = u[:, 1] .- u[:, 2]
            return E0 + dot(ur, P0 * ur) + dot(e[:, 1], e[:, 2]) * dot(ur, P1 * ur)
        end
        rng = MersenneTwister(0xa4)
        cfgs = [(reduce(hcat, [_rc_unit(rng) for _ = 1:2]), randn(rng, 3, 2))
                for _ = 1:80]
        E = [gt(e, u) for (e, u) in cfgs]
        β, res = _rc_fit(b.salcs, cfgs, E, 60)
        @test res < 1e-9
        # probe recovery: unit stretches along and across the bond, spins
        # aligned (+) and antialigned (−) separate Φ⁰ from Φ¹
        e1 = _rc_unit(rng)
        ep = hcat(e1, e1)
        em = hcat(e1, -e1)
        for (dir, l0, l1) in ((1, a, cc), (2, bb, dd), (3, bb, dd))
            us = zeros(3, 2)
            us[dir, 1], us[dir, 2] = 0.5, -0.5          # u_r = unit axis `dir`
            vp = _rc_pred(β, b.salcs, ep, us) - β[1]    # l0 + l1
            vm = _rc_pred(β, b.salcs, em, us) - β[1]    # l0 − l1
            @test (vp + vm) / 2 ≈ l0 rtol = 1e-7
            @test (vp - vm) / 2 ≈ l1 rtol = 1e-6
        end
    end
end
