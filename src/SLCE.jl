"""
    SLCE

Clean, extensible, Julia-native rebuild of `Magesty.jl`: fit spin-cluster-expansion
(SCE) models to noncollinear DFT data. The numerical core is reimplemented from
scratch; `Magesty.jl` serves only as a pinned numerical oracle in `test/oracle/`.

The v0 feature set (geometry, symmetry, cluster/SALC basis, fitting, prediction,
diagnostics, persistence, Sunny export, introspection) is realized; see `SPEC.md`.
"""
module SLCE

using LinearAlgebra: norm, det, I, eigen, eigvals, svdvals, Symmetric, Diagonal, dot, cross
using StaticArrays
using Statistics: mean
using Random: AbstractRNG, default_rng
import TOML
import Tables
# Extend the StatsAPI generics rather than shadow them, so `coef` / `fit` / `nobs` /
# `dof` / `coeftable` / `islinear` / `residuals` / `predict` / `r2` compose with the
# StatsBase / GLM ecosystem instead of clashing on `using`.
import StatsAPI: coef, fit, nobs, dof, coeftable, islinear, residuals, predict, r2

# --- geometry ---
include("geometry/lattice.jl")
include("geometry/crystal.jl")
include("geometry/neighborlist.jl")

# --- symmetry (backend-pluggable; Spglib lives in an extension) ---
include("symmetry/types.jl")
include("symmetry/backend.jl")

# --- basis: numeric kernels (self-contained submodules) ---
include("basis/Harmonics.jl")
include("basis/SolidHarmonics.jl")
include("basis/AngularMomentum.jl")
include("basis/coupledbasis.jl")
include("basis/decor.jl")

# --- clusters: enumeration + symmetry orbit reduction ---
include("clusters/enumerate.jl")
include("clusters/orbits.jl")

# --- SALC basis: symmetry-adapted, time-reversal-even invariants ---
include("basis/salc.jl")
include("basis/salcbasis.jl")

# --- fitting + high-level SCE API ---
include("fitting/estimators.jl")
include("slce/truncation.jl")     # BasisSpec sugar → dense canonical resolution
include("slce/model.jl")          # pipeline types + constructors + config validation
include("basis/sectorbasis.jl")   # sector-table → decor-engine basis construction
include("fitting/design.jl")     # design-matrix assembly (X_E / X_T)
include("fitting/fit.jl")        # fit / refit / predict
include("fitting/diagnostics.jl")  # coef / intercept / residuals / R² / RMSE
include("fitting/selection.jl")  # MC-cost group labels/costs, GCV, λ-path + Pareto

# --- tabular results (Tables.jl source) ---
include("slce/coeftable.jl")

# --- bilinear / single-ion extraction (tesseral → Cartesian), a core capability shared
# by the introspection below and the Sunny interop.
include("slce/bilinear.jl")

# --- fitted-model introspection: a code-neutral view of the multipole / bilinear terms
# (consumed by downstream packages such as the SLCETools.jl samplers).
include("slce/introspect.jl")

# --- external-engine interop: conversion math in core, engine assembly in extensions ---
# Sunny export (Sunny.System assembled in SLCESunnyExt), consuming the bilinear
# extraction above.
include("interop/sunny.jl")

# --- I/O: persistence (TOML model schema), TOML input files, and the code-agnostic DFT
# data boundary. Concrete DFT-code adapters (e.g. the VASP reader/writer) live in
# SLCETools.jl; the SCE pipeline only ever sees `SpinDatum` / `SLCEDataset`.
include("io/persist.jl")
include("io/input.jl")
include("io/dftsource.jl")
include("io/embset.jl")

# --- Public API (exported) --------------------------------------------------------
# The fitting workflow a user reaches for. Construction internals (cluster / neighbor /
# SALC builders, symmetry analysis) are *public but unexported* — see the block below.

# geometry the user builds
export Lattice, Crystal, n_atoms, cartesian_positions
# how periodic images / symmetry are chosen (passed into `SLCEBasis`)
export AbstractImageSelection, MinimumImage, AllImages
export AbstractSymmetryBackend, NoSymmetry, SpglibBackend
# the SCE pipeline
export BasisSpec, Sector, SLCEBasis, SLCEDataset, SLCEModel, SLCEFit, fit, refit,
    n_salcs, read_setup
export predict_energy, predict_torque, has_torque
# estimators
export AbstractEstimator, OLS, Ridge, ElasticNet, Lasso, AdaptiveLasso, AdaptiveRidge,
    GroupAdaptiveRidge, PrecomputedPilot
# fit diagnostics (predict / residuals / r2 are StatsAPI generics defaulting to the
# energy block; the explicit *_energy / *_torque forms are the full surface)
export coef, intercept, nobs, dof, predict, residuals, r2,
    r2_energy, rmse_energy, r2_torque, rmse_torque,
    rss_energy, rss_torque, residuals_energy, residuals_torque
export coeftable, SCECoefficients
# model selection: GCV / effective dof (linear estimators), the cost-aware λ path,
# and the threshold-swept refit front
export gcv, effective_dof, select_fit, SelectionPath, select_support, SupportPath
export cross_validate, CVResult
export to_sunny
# Fitted-model introspection: a code-neutral view of the multipole / bilinear terms of a
# fitted SCE, the stable contract downstream packages (e.g. the SLCETools.jl mean-field
# samplers) read instead of the SALC-basis internals.
export MultipoleTerm, multipole_terms, bilinear_terms
# DFT data I/O: only the code-agnostic boundary is exported; per-code adapters live in
# downstream packages as namespaced submodules (e.g. `SLCETools.VASP.read_poscar`), so
# adding a code touches neither the core nor this export list. The one in-core format
# is Magesty's EMBSET training set — code-agnostic (it carries exactly what SpinDatum
# stores), kept here for legacy-data reuse.
export AbstractDFTSource, SpinDatum, read_configs
export EmbsetFile, read_embset

# --- Public, unexported -----------------------------------------------------------
# Reachable as `SLCE.<name>` (and documented), but kept out of the flat `using`
# namespace: the `SLCEBasis` constructor already drives them for you. Power users and the
# test suite reach them by qualification. Declared with the `public` keyword so the
# tier is machine-checkable (`Base.ispublic`, Aqua) instead of a comment-only promise.
public Harmonics, SolidHarmonics, AngularMomentum                    # numeric kernels
public build_neighbor_list, NeighborPair, NeighborList, interplanar_spacing
public analyze_symmetry, n_ops, SymOp, SpaceGroup, AbstractTrainingDatum
public build_clusters, ClusterMember, ClusterOrbit, ClusterSet
public build_salc_basis, evaluate_salc, salcs, SALC, SALCKey, SALCBasis
public Channel, SPIN, DISP, OCC, SiteFactor, SiteDecor                # decoration labels
public SectorRule                                                     # resolved sector row
public has_spin, has_disp, spin_rank, disp_degree, factors, is_pure_spin
public spin_decors, spin_ls, rep_scale
public islinear, solve_coefficients
public salc_groups, group_costs, cost_weights                        # MC-cost grouping
public save, load                                                    # TOML persistence

end # module SLCE
