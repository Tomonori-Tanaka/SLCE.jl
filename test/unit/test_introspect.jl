# Fitted-model introspection (src/slce/introspect.jl): the code-neutral `multipole_terms` /
# `bilinear_terms` view downstream packages (e.g. the SLCETools.jl samplers) read instead of
# the SALC-basis internals. The gates are energy reconstructions: summing the per-term
# tesseral contraction must reproduce `predict_energy − j0`.

using Test
using SLCE
using LinearAlgebra
using Random
using StaticArrays

const Zlm = SLCE.Harmonics.Zlm

# A 2-body lmax=[2] noncollinear model carries [1,1] bilinear, [1,2]/[2,2] biquadratic, and
# [2] single-ion channels — every body order / l the introspection must round-trip.
function _multichannel_basis()
    lat = Lattice(Matrix(3.0 * I(3)))
    cr = Crystal(lat, [0.2 -0.2; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
    return SLCEBasis(cr, BasisSpec(; nbody = 2, cutoff = 1.5, lmax = [2], soc = true))
end

# Random unit-column spin configuration (3 × n).
function _rand_config(rng, n)
    e = randn(rng, 3, n)
    for a = 1:n
        e[:, a] ./= norm(@view e[:, a])
    end
    return e
end

# Energy from the neutral multipole terms: Σ_term coef·(4π)^(N/2)·Σ_μ folded[μ] ∏ Z.
function _energy_from_terms(terms, e)
    tot = 0.0
    for t in terms
        N = t.body
        s = 0.0
        for idx in CartesianIndices(t.folded)
            w = t.folded[idx]
            w == 0.0 && continue
            for i = 1:N
                μ = idx[i] - t.ls[i] - 1
                ei = SVector{3,Float64}(e[1, t.atoms[i]], e[2, t.atoms[i]], e[3, t.atoms[i]])
                w *= Zlm(t.ls[i], μ, ei)
            end
            s += w
        end
        tot += t.coef * (4π)^(N / 2) * s
    end
    return tot
end

# Energy from the bilinear / single-ion 3×3 matrices: Σ eₐ'·M·e_b + Σ eₐ'·A·eₐ.
function _energy_from_bilinear(bt, e)
    tot = 0.0
    for ((a, b, _), M) in bt.pairs
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        eb = SVector{3,Float64}(e[1, b], e[2, b], e[3, b])
        tot += dot(ea, M * eb)
    end
    for (a, A) in bt.onsites
        ea = SVector{3,Float64}(e[1, a], e[2, a], e[3, a])
        tot += dot(ea, A * ea)
    end
    return tot
end

@testset "fitted-model introspection" begin
    rng = MersenneTwister(2024)
    b = _multichannel_basis()
    keys = b.salc_basis.keys
    K = n_salcs(b)

    @testset "n_atoms(model)" begin
        model = SLCEModel(b, 0.0, randn(rng, K), keys)
        @test n_atoms(model) == 2
    end

    @testset "multipole_terms reproduces predict_energy − j0" begin
        j0 = 0.37
        model = SLCEModel(b, j0, 0.1 .* randn(rng, K), keys)
        terms = multipole_terms(model)
        @test !isempty(terms)
        # The raw coefficient is the fitted jϕ (no (4π)^(N/2) baked in); the body order spans
        # both 1-body (single-ion) and 2-body channels.
        @test Set(t.body for t in terms) == Set([1, 2])
        @test intercept(model) ≈ j0
        for _ = 1:8
            e = _rand_config(rng, 2)
            @test _energy_from_terms(terms, e) ≈ predict_energy(model, e) - j0 atol = 1e-10
        end
    end

    @testset "MultipoleTerm field contract (downstream consumers pin on this)" begin
        # SLCETools.jl's bridge reads exactly these fields; a rename/retype must fail here
        # (and then be synchronized downstream), not slip through the energy gates.
        @test fieldnames(MultipoleTerm) == (:coef, :body, :atoms, :shifts, :ls, :folded)
        terms = multipole_terms(SLCEModel(b, 0.0, randn(MersenneTwister(3), K), keys))
        @test terms isa Vector{MultipoleTerm}
        t = terms[1]
        @test t.coef isa Float64 && t.body isa Int
        @test t.atoms isa Vector{Int} && t.ls isa Vector{Int}
        @test t.shifts isa Vector{SVector{3,Int}}
        @test t.folded isa Array{Float64}
        @test length(t.atoms) == length(t.shifts) == length(t.ls) == t.body == ndims(t.folded)
    end

    @testset "zero-coefficient SALCs are dropped" begin
        jphi = zeros(K)
        jphi[1] = 0.5
        terms = multipole_terms(SLCEModel(b, 0.0, jphi, keys))
        @test all(t -> t.coef == 0.5, terms)        # only the one nonzero SALC's members
    end

    @testset "bilinear_terms reproduces a bilinear + single-ion model" begin
        # Keep only the Sunny-representable channels (ls = [1,1] pair, ls = [2] single-ion);
        # then the bilinear extraction captures the whole energy (the skipped channels exist
        # in the basis but carry a zero coefficient, so they contribute nothing).
        jphi = [SLCE.spin_ls(keys[k]) in ([1, 1], [2]) ? randn(rng) : 0.0 for k = 1:K]
        model = SLCEModel(b, 0.0, jphi, keys)
        bt = bilinear_terms(model)
        for _ = 1:8
            e = _rand_config(rng, 2)
            @test _energy_from_bilinear(bt, e) ≈ predict_energy(model, e) atol = 1e-10
        end
    end

    @testset "bilinear_terms reports the higher-order channels it drops" begin
        model = SLCEModel(b, 0.0, ones(K), keys)
        bt = bilinear_terms(model)
        @test !isempty(bt.skipped)                  # the [1,2] / [2,2] biquadratic channels
    end
end
