# Design-matrix assembly: build X_E / X_T (and flatten torque targets) from a
# dataset's spin configurations. Split out of model.jl (pipeline types).

# Energy design matrix: X_E[config, salc] = Φ_salc(config).
function _design_energy(basis::SLCEBasis, cfgs::Vector{Matrix{Float64}})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(cfgs)
    m = length(salcs)
    X = Matrix{Float64}(undef, n, m)
    # Columns are independent; each task owns whole columns, so the writes are
    # disjoint (no false sharing) and the result is identical at any thread count.
    Threads.@threads for j = 1:m
        scratch = SALCScratch()      # task-local workspace (dnPl + harmonic tables)
        @inbounds for i = 1:n        # column-major: stride-1 writes down each column
            X[i, j] = evaluate_salc(salcs[j], cfgs[i], scratch)
        end
    end
    return X
end

# Torque design matrix: for each SALC column, each config, each atom, the three
# components of τ_a = −e_a × ∂Φ/∂e_a (the physical / Landau–Lifshitz torque) stacked
# config-major / atom-major / xyz.
function _design_torque(basis::SLCEBasis, cfgs::Vector{Matrix{Float64}})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(cfgs)
    m = length(salcs)
    nat = isempty(cfgs) ? 0 : size(cfgs[1], 2)
    block = 3 * nat
    X = Matrix{Float64}(undef, n * block, m)
    # Columns are independent; each task owns whole columns. `G` must be
    # task-local (a shared buffer would race across threads).
    Threads.@threads for j = 1:m     # column per SALC (column-major writes)
        G = Matrix{Float64}(undef, 3, nat)
        scratch = SALCScratch()      # task-local workspace (dnPl + harmonic tables)
        @inbounds for ci = 1:n
            c = cfgs[ci]
            fill!(G, 0.0)
            accumulate_grad!(G, salcs[j], c, 1.0, scratch)
            row_off = block * (ci - 1)
            for a = 1:nat
                ea = SVector{3,Float64}(c[1, a], c[2, a], c[3, a])
                ga = SVector{3,Float64}(G[1, a], G[2, a], G[3, a])
                t = cross(ga, ea)        # τ = ∇Φ × e = −e × ∇Φ  (physical / LL torque)
                rb = row_off + 3 * (a - 1)
                X[rb + 1, j] = t[1]
                X[rb + 2, j] = t[2]
                X[rb + 3, j] = t[3]
            end
        end
    end
    return X
end

# Flatten per-config torque targets in the same row order as `_design_torque`.
function _flatten_torques(torques::AbstractVector, cfgs::Vector{Matrix{Float64}})::Vector{Float64}
    nat = isempty(cfgs) ? 0 : size(cfgs[1], 2)
    block = 3 * nat
    y = Vector{Float64}(undef, length(cfgs) * block)
    @inbounds for ci in eachindex(torques)
        τ = torques[ci]
        size(τ) == (3, nat) ||
            throw(ArgumentError("torque block $ci must be 3 × $nat (got $(size(τ)))"))
        rb = block * (ci - 1)
        for a = 1:nat, d = 1:3
            y[rb + 3 * (a - 1) + d] = τ[d, a]
        end
    end
    return y
end
