# The per-site basis-row layout (src/slce/rowlayout.jl) — the contract a sampler
# builds its gather programs against. The decisive gate is an END-TO-END one: a
# miniature "consumer" that tabulates rows with `site_rows!`, addresses them with
# `row_index`, and contracts the published `decorated_terms` must reproduce
# `predict_energy`. If the row numbering and the row contents ever disagree, that
# sum is wrong — no amount of index bookkeeping catches it on its own.

using Test
using SLCE
using SLCE: RowLayout, row_layout, row_index, site_rows!, SiteFactor, SPIN, DISP, OCC
using LinearAlgebra
using Random
using StaticArrays

isdefined(@__MODULE__, :same_members) || include("testutils.jl")

_rl_cfg(rng, n) = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:n])

# The reference consumer: rows in, energy out, gathering exactly the way a sweep
# kernel does — one row per tensor axis, no channel branch inside the loop.
function _energy_via_rows(model, layout, e, u)
    nat = size(e, 2)
    rows = zeros(layout.nrows, nat)
    for a = 1:nat
        site_rows!(view(rows, :, a), layout, view(e, :, a), view(u, :, a))
    end
    tot = 0.0
    for t in decorated_terms(model)
        s = 0.0
        for idx in CartesianIndices(t.folded)
            w = t.folded[idx]
            w == 0.0 && continue
            for (i, sl) in enumerate(t.slots)
                m = idx[i] - sl.factor.l - 1
                w *= rows[row_index(layout, sl.factor, m), t.atoms[sl.site]]
            end
            s += w
        end
        tot += t.coef * t.scale * s
    end
    return tot
end

@testset "per-site basis-row layout" begin
    rng = MersenneTwister(31)
    cr = Crystal(Lattice(Matrix(3.0 * I(3))),
                 [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    nat = 2

    @testset "the field list is part of the contract" begin
        # `RowLayout`'s `==` and `hash` are written out field by field (the default
        # `==` would compare the vector fields by identity, so a freshly derived
        # layout would never equal a stored one — and "did the support move?" is
        # exactly what a consumer asks across a coefficient hot-swap or a
        # checkpoint reload). Hand-written means a NEW field is silently excluded:
        # two layouts differing only in it would compare equal, and every
        # layout-change gate downstream would go quiet. Adding a field is fine —
        # extend `==`, `hash`, and this tuple together.
        @test fieldnames(SLCE.RowLayout) ==
              (:nrows, :spin_lmax, :disp_offset, :disp_factors, :disp_starts)
    end

    @testset "pure spin: the layout IS the existing lm_index numbering" begin
        # the whole point of putting SPIN first at offset 0 — a spin-only consumer's
        # row tables must not move when the displacement channel is added to the
        # package, so this is an identity, not an isomorphism
        bs = SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = true))
        L = row_layout(bs)
        @test L.spin_lmax == 2
        @test L.nrows == (2 + 1)^2 == SLCE.Harmonics.num_lm(2)
        @test isempty(L.disp_factors)
        @test L.disp_offset == L.nrows
        # a SPIN factor has l ≥ 1 (l = 0 would be a constant), but the l = 0 row is
        # still part of the block so the numbering is `lm_index` itself
        for l = 1:2, m = -l:l
            @test row_index(L, SiteFactor(SPIN, 0, l), m) ==
                  SLCE.Harmonics.lm_index(l, m)
        end
        # and the filler is the plain tesseral row
        e = _rl_cfg(rng, nat)
        rows = zeros(L.nrows)
        site_rows!(rows, L, view(e, :, 1), zeros(3))
        for l = 0:2, m = -l:l
            @test rows[SLCE.Harmonics.lm_index(l, m)] ==
                  SLCE.Harmonics.Zlm(l, m, SVector{3}(e[:, 1]))
        end
        @test occursin("spin lmax = 2", sprint(show, L))
    end

    @testset "joint: blocks stack in channel order, sizes and offsets" begin
        b = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1),
            Sector(spin = [1, 1], disp = (degree = 1:2,), nbody = 2, cutoff = 1.1),
            Sector(disp = (degree = 1:2,), nbody = 1:2, cutoff = 1.1)]))
        L = row_layout(b)
        @test L.spin_lmax == 1
        @test L.disp_offset == 4                      # (1 + 1)² spin rows first
        @test !isempty(L.disp_factors)
        @test issorted(L.disp_factors)
        # the displacement blocks are contiguous, 2l+1 wide, starting after the spin
        # block and covering every row up to nrows
        off = L.disp_offset
        for (i, (k, l)) in pairs(L.disp_factors)
            @test L.disp_starts[i] == off
            @test row_index(L, SiteFactor(DISP, k, l), -l) == off + 1
            @test row_index(L, SiteFactor(DISP, k, l), l) == off + 2l + 1
            off += 2l + 1
        end
        @test off == L.nrows
        # every (k, l) the basis actually uses has a block
        used = Set((d.disp_k, d.disp_l) for kk in b.salc_basis.keys
                   for d in kk.decors if SLCE.has_disp(d))
        @test Set(L.disp_factors) == used
        # the spin rows are still exactly where they were
        for l = 1:1, m = -l:l
            @test row_index(L, SiteFactor(SPIN, 0, l), m) ==
                  SLCE.Harmonics.lm_index(l, m)
        end

        @testset "site_rows! contents" begin
            e = _rl_cfg(rng, nat)
            u = 0.07 .* randn(rng, 3, nat)
            rows = zeros(L.nrows)
            site_rows!(rows, L, view(e, :, 2), view(u, :, 2))
            uv = SVector{3}(u[:, 2])
            for (i, (k, l)) in pairs(L.disp_factors), m = -l:l
                @test rows[L.disp_starts[i] + m + l + 1] ≈
                      dot(uv, uv)^k * SLCE.SolidHarmonics.Rlm(l, m, uv)
            end
            # u = 0 zeroes every displacement row (each factor is homogeneous of
            # degree ≥ 1) while leaving the spin rows untouched
            z = zeros(L.nrows)
            site_rows!(z, L, view(e, :, 2), zeros(3))
            @test all(iszero, z[(L.disp_offset + 1):L.nrows])
            @test z[1:L.disp_offset] == rows[1:L.disp_offset]
        end

        @testset "a consumer gathering through the layout reproduces the energy" begin
            model = SLCEModel(b, 0.41, randn(rng, n_salcs(b)))
            for _ = 1:8
                e = _rl_cfg(rng, nat)
                u = 0.06 .* randn(rng, 3, nat)
                @test _energy_via_rows(model, L, e, u) ≈
                      predict_energy(model, e, u) - 0.41 atol = 1e-10
            end
            # the layout depends on the BASIS, not the coefficients: a sampler may
            # keep its row tables across a coefficient hot-swap
            @test row_layout(SLCEModel(b, 0.0, zeros(n_salcs(b)))) ==
                  row_layout(model)
        end
    end

    @testset "degenerate layouts and refusals" begin
        # lattice-only: no spin block at all
        bl = SLCEBasis(cr, BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
            Sector(disp = (degree = 2,), nbody = 1:2, cutoff = 1.1)]))
        Ll = row_layout(bl)
        @test Ll.spin_lmax == -1 && Ll.disp_offset == 0
        @test Ll.nrows == sum(2l + 1 for (_, l) in Ll.disp_factors)
        @test_throws ArgumentError row_index(Ll, SiteFactor(SPIN, 0, 1), 0)
        # a factor the layout does not carry is an error, never a fallback row
        bs = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
            Sector(spin = (nbody = 1:2,), cutoff = 1.1)]))
        Ls = row_layout(bs)
        @test_throws ArgumentError row_index(Ls, SiteFactor(SPIN, 0, 3), 0)
        @test_throws ArgumentError row_index(Ls, SiteFactor(DISP, 0, 1), 0)
        @test_throws ArgumentError row_index(Ls, SiteFactor(SPIN, 0, 1), 2)   # |m| > l
        @test_throws DimensionMismatch site_rows!(zeros(Ls.nrows - 1), Ls,
                                                  [1.0, 0, 0], zeros(3))
    end
end
