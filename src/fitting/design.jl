# Design-matrix assembly: build X_E / X_T (and flatten torque targets) from a
# dataset's spin configurations. Split out of model.jl (pipeline types).

# Energy design matrix: X_E[config, salc] = Φ_salc(config).
function _design_energy(basis::SLCEBasis, configs::Vector{Matrix{Float64}})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(configs)
    m = length(salcs)
    X = Matrix{Float64}(undef, n, m)
    # Columns are independent; each task owns whole columns, so the writes are
    # disjoint (no false sharing) and the result is identical at any thread count.
    Threads.@threads for j = 1:m
        scratch = SALCScratch()      # task-local workspace (dnPl + harmonic tables)
        @inbounds for i = 1:n        # column-major: stride-1 writes down each column
            X[i, j] = evaluate_salc(salcs[j], configs[i], scratch)
        end
    end
    return X
end

# Torque design matrix: for each SALC column, each config, each atom, the three
# components of τ_a = −e_a × ∂Φ/∂e_a (the physical / Landau–Lifshitz torque) stacked
# config-major / atom-major / xyz.
function _design_torque(basis::SLCEBasis, configs::Vector{Matrix{Float64}})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(configs)
    m = length(salcs)
    nat = isempty(configs) ? 0 : size(configs[1], 2)
    block = 3 * nat
    X = Matrix{Float64}(undef, n * block, m)
    # Columns are independent; each task owns whole columns. `G` must be
    # task-local (a shared buffer would race across threads).
    Threads.@threads for j = 1:m     # column per SALC (column-major writes)
        G = Matrix{Float64}(undef, 3, nat)
        scratch = SALCScratch()      # task-local workspace (dnPl + harmonic tables)
        @inbounds for ci = 1:n
            c = configs[ci]
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

# --- joint (spin + displacement) designs -----------------------------------------
# Same layouts as the spin-only builders, evaluated at each configuration's (e, u)
# through the joint kernels (which share the `(4π)^(n_spin/2)` scale and the
# `_fill_ztables_mixed!` tables with the spin-only path — bit-identical on
# pure-spin SALCs, so a joint basis' pure-spin columns match the spin-only design).

function _design_energy(basis::SLCEBasis, configs::Vector{Matrix{Float64}},
                        disps::Vector{Matrix{Float64}})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(configs)
    m = length(salcs)
    X = Matrix{Float64}(undef, n, m)
    Threads.@threads for j = 1:m         # columns independent (cf. spin-only form)
        scratch = SALCScratch()
        @inbounds for i = 1:n
            X[i, j] = evaluate_salc(salcs[j], configs[i], disps[i], scratch)
        end
    end
    return X
end

# Joint torque design, restricted to the spin-referenced atoms `tatoms` (rows the
# model is structurally blind to — a displacement-only ligand site has no SPIN slot,
# so its torque row is exactly zero in X_T AND in y_T (τ = m × B with m = 0) — are
# excluded, never padded, exactly like the force block; padding would dilute the
# `√(w_T/n_T)` whitening). The pure-spin path keeps its all-atom layout (every atom
# of a pure-spin basis is either spin-referenced or genuinely observed).
function _design_torque(basis::SLCEBasis, configs::Vector{Matrix{Float64}},
                        disps::Vector{Matrix{Float64}},
                        tatoms::Vector{Int})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(configs)
    m = length(salcs)
    nat = isempty(configs) ? 0 : size(configs[1], 2)
    block = 3 * length(tatoms)
    X = Matrix{Float64}(undef, n * block, m)
    Threads.@threads for j = 1:m         # columns independent; buffers task-local
        Ge = Matrix{Float64}(undef, 3, nat)
        Gu = Matrix{Float64}(undef, 3, nat)
        scratch = SALCScratch()
        @inbounds for ci = 1:n
            c = configs[ci]
            fill!(Ge, 0.0)
            fill!(Gu, 0.0)
            accumulate_grad!(Ge, Gu, salcs[j], c, disps[ci], 1.0, scratch)
            row_off = block * (ci - 1)
            for (k, a) in enumerate(tatoms)
                ea = SVector{3,Float64}(c[1, a], c[2, a], c[3, a])
                ga = SVector{3,Float64}(Ge[1, a], Ge[2, a], Ge[3, a])
                t = cross(ga, ea)        # τ = ∇Φ × e = −e × ∇Φ  (physical / LL torque)
                rb = row_off + 3 * (k - 1)
                X[rb + 1, j] = t[1]
                X[rb + 2, j] = t[2]
                X[rb + 3, j] = t[3]
            end
        end
    end
    return X
end

# Compact force design block: for the displacement-active SALC columns `cols` only
# (a pure-spin SALC has ∂Φ/∂u ≡ 0 — its zero columns are never materialized) and
# the displacement-referenced atoms `fatoms` only (rows the model is structurally
# blind to are excluded, never padded). Entry = −∂Φ/∂u (the force convention
# `f_a = −∂E/∂u_a`, design record §6), rows config-major / fatoms-major / xyz.
function _design_force(basis::SLCEBasis, configs::Vector{Matrix{Float64}},
                       disps::Vector{Matrix{Float64}}, cols::Vector{Int},
                       fatoms::Vector{Int})::Matrix{Float64}
    salcs = basis.salc_basis.salcs
    n = length(configs)
    nat = isempty(configs) ? 0 : size(configs[1], 2)
    block = 3 * length(fatoms)
    X = Matrix{Float64}(undef, n * block, length(cols))
    Threads.@threads for jj in eachindex(cols)   # columns independent; buffers task-local
        j = cols[jj]
        Ge = Matrix{Float64}(undef, 3, nat)
        Gu = Matrix{Float64}(undef, 3, nat)
        scratch = SALCScratch()
        @inbounds for ci = 1:n
            fill!(Ge, 0.0)
            fill!(Gu, 0.0)
            accumulate_grad!(Ge, Gu, salcs[j], configs[ci], disps[ci], 1.0, scratch)
            row_off = block * (ci - 1)
            for (k, a) in enumerate(fatoms)
                rb = row_off + 3 * (k - 1)
                X[rb + 1, jj] = -Gu[1, a]
                X[rb + 2, jj] = -Gu[2, a]
                X[rb + 3, jj] = -Gu[3, a]
            end
        end
    end
    return X
end

# Flatten per-config 3 × n_atoms target blocks restricted to the atom subset
# `atoms`, config-major / subset-atom-major / xyz — the shared row order of the
# restricted derivative designs (`_design_force`, the joint `_design_torque`).
function _flatten_atom_rows(blocks::AbstractVector, atoms::Vector{Int}, nat::Int,
                            what::String)::Vector{Float64}
    block = 3 * length(atoms)
    y = Vector{Float64}(undef, length(blocks) * block)
    @inbounds for ci in eachindex(blocks)
        F = blocks[ci]
        size(F) == (3, nat) ||
            throw(ArgumentError("$what block $ci must be 3 × $nat (got $(size(F)))"))
        rb = block * (ci - 1)
        for (k, a) in enumerate(atoms), d = 1:3
            y[rb + 3 * (k - 1) + d] = F[d, a]
        end
    end
    return y
end

# Flatten per-config force targets in the same row order as `_design_force`.
_flatten_forces(forces::AbstractVector, fatoms::Vector{Int}, nat::Int)::Vector{Float64} =
    _flatten_atom_rows(forces, fatoms, nat, "force")

# Flatten per-config torque targets in the same row order as `_design_torque`.
function _flatten_torques(torques::AbstractVector, configs::Vector{Matrix{Float64}})::Vector{Float64}
    nat = isempty(configs) ? 0 : size(configs[1], 2)
    block = 3 * nat
    y = Vector{Float64}(undef, length(configs) * block)
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
