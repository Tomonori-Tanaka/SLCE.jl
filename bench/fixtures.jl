# Shared fixtures and timing helpers for the SLCE benchmark scripts.
#
# `include`d (not `import`ed) by each bench/bench_*.jl. Each script does
# `using SLCE` first; this file then layers the fixtures and the small
# `@timed`/`@allocations` harness on top. Mirrors the standalone-script style of
# Magesty.jl/bench (no package coupling) while keeping the common pieces DRY.

using SLCE
# Public (unexported) staging functions the stage-level benches drive directly.
using SLCE: analyze_symmetry, build_neighbor_list, build_clusters, build_salc_basis
using LinearAlgebra: norm
using Statistics: median, mean
using Printf: @printf
using Random: MersenneTwister
using BenchmarkTools: @belapsed, @benchmark

# ---------------------------------------------------------------------------
# Fixtures — single-species cubic supercells of tunable size.
# ---------------------------------------------------------------------------

"""
    bcc_fe(n; a = 2.87) -> Crystal

A conventional bcc Fe cell tiled `n × n × n` (`2n³` atoms, one species `"Fe"`).
`n = 2 → 16`, `n = 3 → 54`, `n = 4 → 128` atoms. The cell is cubic with edge `n·a`
(Å); the bcc nearest-neighbour distance is `a√3/2 ≈ 2.49 Å`, so `cutoff = 2.6`
keeps the first shell.
"""
function bcc_fe(n::Integer; a::Real = 2.87)
    L = n * a
    lat = Lattice(Float64[L 0 0; 0 L 0; 0 0 L])
    motif = ((0.0, 0.0, 0.0), (0.5, 0.5, 0.5))     # bcc, conventional cell
    cols = Vector{Float64}[]
    for i = 0:n-1, j = 0:n-1, k = 0:n-1, (bx, by, bz) in motif
        push!(cols, [(i + bx) / n, (j + by) / n, (k + bz) / n])
    end
    frac = reduce(hcat, cols)                       # 3 × natoms
    nat = size(frac, 2)
    return Crystal(lat, frac, fill(1, nat), ["Fe"])
end

"""
    rand_configs(crystal, m; seed = 1) -> Vector{Matrix{Float64}}

`m` random spin configurations (`3 × n_atoms`, unit columns) for `crystal`.
"""
function rand_configs(crystal, m::Integer; seed::Integer = 1)
    rng = MersenneTwister(seed)
    nat = n_atoms(crystal)
    return [reduce(hcat, ((v = randn(rng, 3); v ./ norm(v)) for _ = 1:nat)) for _ = 1:m]
end

"""
    basis_spec(; nbody = 2, cutoff = 2.6, lmax = 1, isotropy = false) -> BasisSpec

A [`BasisSpec`](@ref) for the single-species fixtures (`lmax` is broadcast to the one
species).
"""
basis_spec(; nbody = 2, cutoff = 2.6, lmax = 1, isotropy = false) =
    BasisSpec(; nbody = nbody, cutoff = cutoff, lmax = [lmax], isotropy = isotropy)

# ---------------------------------------------------------------------------
# Timing harness.
# ---------------------------------------------------------------------------

"""
    bench_header(title)

Print a banner with the machine/Julia context for a benchmark script.
"""
function bench_header(title::AbstractString)
    println("=" ^ 72)
    println(title)
    println("julia $(VERSION)   threads=$(Threads.nthreads())")
    println("=" ^ 72)
end

"""
    bench_one(label, f; ntrials = 3) -> NamedTuple

Warm up `f()` once (compile), then run `ntrials` timed passes; report median/min
wall time plus the per-call allocation count and median bytes. For hot paths too
heavy for `@benchmark` sampling (the SALC build, design matrices, end-to-end).
"""
function bench_one(label::AbstractString, f; ntrials::Integer = 3)
    f()                                              # warm-up / compile
    times = Float64[]
    bytes = Float64[]
    for _ = 1:ntrials
        s = @timed f()
        push!(times, s.time)
        push!(bytes, Float64(s.bytes))
    end
    allocs = @allocations f()
    @printf("%-32s  min=%9.3f ms  med=%9.3f ms  allocs=%-11d mem=%8.2f MiB\n",
            label, 1e3 * minimum(times), 1e3 * median(times), allocs,
            median(bytes) / 2^20)
    return (; label = String(label), min_s = minimum(times), med_s = median(times),
            allocs = allocs)
end

# Parse positional ARGS with defaults: argn(2, 30) → 2nd arg or 30 (Int);
# argf(3, 6.0) → 3rd arg or 6.0 (Float64, accepts `inf`).
argn(i::Integer, default::Integer) = length(ARGS) >= i ? parse(Int, ARGS[i]) : default
argf(i::Integer, default::Real) =
    length(ARGS) >= i ? parse(Float64, ARGS[i]) : Float64(default)
