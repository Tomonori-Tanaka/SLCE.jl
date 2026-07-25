# Shared unit-test helpers, included exactly once by `runtests.jl` before the unit
# files. They used to be redefined per file in the shared test module — textually
# identical copies that overwrote one another (method-overwrite warnings on every
# run), where an edit to one copy would silently win for every later-included file.
# The bodies are kept byte-identical to the originals so every seeded fixture draws
# the same RNG stream.

using StaticArrays: SVector, SMatrix, @SMatrix
using LinearAlgebra: norm, I
using Random: AbstractRNG

# A uniform random unit direction.
rand_unit(rng::AbstractRNG) =
    (v = SVector{3,Float64}(randn(rng), randn(rng), randn(rng)); v / norm(v))

# Proper rotation from a random axis-angle (Rodrigues).
function rand_rotation(rng::AbstractRNG)
    a = SVector{3,Float64}(randn(rng), randn(rng), randn(rng))
    a = a / norm(a)
    θ = 2π * rand(rng)
    Kc = @SMatrix [0.0 -a[3] a[2]; a[3] 0.0 -a[1]; -a[2] a[1] 0.0]
    return SMatrix{3,3,Float64}(I) + sin(θ) * Kc + (1 - cos(θ)) * (Kc * Kc)
end

# A random spin configuration: `3 × nat`, unit columns (fresh draw per column, so
# neighboring spins differ and odd-Lf / anisotropic channels do not vanish).
function randcfg(rng::AbstractRNG, nat::Integer)
    M = Matrix{Float64}(undef, 3, nat)
    for a = 1:nat
        v = randn(rng, 3)
        M[:, a] = v / norm(v)
    end
    return M
end

# --- canonical-member gates (used by test_salc.jl and test_nbody.jl) -----------

# Canonical-member structural invariants: sites sorted by (atom, shift) and
# unique, shifts anchored (shifts[1] == 0), one member per physical instance, one
# term per l-assignment (unique, sorted, nonzero).
function check_canonical_members(salc)
    seen = Set{Tuple{Vector{Int},Vector{SVector{3,Int}}}}()
    for m in salc.members
        sk = [(m.atoms[i], Tuple(m.shifts[i])) for i in eachindex(m.atoms)]
        @test issorted(sk)
        @test allunique(sk)
        @test m.shifts[1] == SVector(0, 0, 0)
        @test (m.atoms, m.shifts) ∉ seen
        push!(seen, (m.atoms, m.shifts))
        lss = [Tuple(t.ls) for t in m.terms]
        @test issorted(lss)
        @test allunique(lss)
        @test all(t -> any(!=(0.0), t.folded), m.terms)
    end
end

same_members(a, b) =
    length(a) == length(b) &&
    all(ma.atoms == mb.atoms && ma.shifts == mb.shifts &&
        length(ma.terms) == length(mb.terms) &&
        all(ta.ls == tb.ls && ta.folded == tb.folded
            for (ta, tb) in zip(ma.terms, mb.terms))
        for (ma, mb) in zip(a, b))

# Split every member of `salc` into two site-permuted half-weight copies (`perma`,
# `permb`, per body order), de-anchor one by a constant lattice shift, and check
# `_canonicalize_members` folds the mess back to the original members exactly
# (0.5·T + 0.5·T == T bitwise). Exercises axis transposition + re-anchoring.
function split_roundtrip_exact(salc, perms_by_body::Dict{Int,NTuple{2,Vector{Int}}})
    red = SLCE.SALCMember[]
    R0 = SVector(1, -2, 3)
    for m in salc.members
        N = length(m.atoms)
        pa, pb = perms_by_body[N]
        for (p, deanchor) in ((pa, false), (pb, true))
            sh = [deanchor ? m.shifts[i] + R0 : m.shifts[i] for i in p]
            terms = SLCE.SALCTerm[
                SLCE.SALCTerm(t.ls[p], 0.5 .* permutedims(t.folded, p))
                for t in m.terms]
            push!(red, SLCE.SALCMember(m.atoms[p], sh, terms))
        end
    end
    @test same_members(SLCE._canonicalize_members(red), salc.members)
end
