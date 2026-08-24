"""
    SLCE

Clean, extensible, Julia-native rebuild of `Magesty.jl`: fit spin–lattice cluster-expansion
(SLCE) models to noncollinear DFT data. The numerical core is reimplemented from
scratch; `Magesty.jl` serves only as a pinned numerical oracle in `test/oracle/`.

Two channels are fitted. The **spin** channel (Drautz–Fähnle, tesseral harmonics over
clusters of spin directions) trains on energies and per-atom torques; the **displacement**
channel decorates the same clusters with solid-harmonic factors and trains on forces as
well, under the acoustic sum rule. Realized: geometry, symmetry, cluster/SALC basis with a
`Sector` truncation table, the three-block (energy / torque / force) fit with staged and
cost-weighted selection, prediction, diagnostics and identifiability, persistence, Sunny
export, and the derivative readouts — force constants and dynamical matrices, strain
derivatives, magnetoelastic constants, magnon–phonon vertices, re-expanded effective
models, volume grids, and phonopy / ALAMODE export. See `SPEC.md`.
"""
module SLCE

using LinearAlgebra: norm, det, I, eigen, eigvals, svd, svdvals, cholesky, Symmetric,
    Diagonal, dot, cross, qr, ColumnNorm
using StaticArrays
using Statistics: mean, quantile, std
using Random: AbstractRNG, default_rng, MersenneTwister
using Printf: @printf
import TOML
import Tables
# Extend the StatsAPI generics rather than shadow them, so `coef` / `fit` / `nobs` /
# `dof` / `coeftable` / `islinear` / `residuals` / `predict` / `r2` compose with the
# StatsBase / GLM ecosystem instead of clashing on `using`.
import StatsAPI: coef, fit, nobs, dof, coeftable, islinear, residuals, predict, r2

# --- units: the kelvin ↔ model-energy boundary the family's samplers share ---
include("units.jl")

# --- spin directions: the unit-norm invariant, carried by a type rather than by a
# rule each entry point re-states. Deliberately BELOW `basis/Harmonics.jl`, so the
# harmonic kernels can name the type their preconditions are about.
include("direction.jl")

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

# --- fitting + high-level SLCE API ---
include("fitting/estimators.jl")
include("slce/truncation.jl")     # BasisSpec sugar → dense canonical resolution
include("io/provenance.jl")       # DatumProvenance (SLCEDataset stores an identity summary)
include("slce/model.jl")          # pipeline types + constructors + config validation
include("basis/sectorbasis.jl")   # sector-table → decor-engine basis construction
include("basis/resolvability.jl")  # which SALCs a finite reference cell can resolve
include("basis/momentbasis.jl")    # pointed (site-marked) basis for adiabatic moments
include("fitting/asr.jl")        # ASR constraint builder + null-space machinery
include("fitting/design.jl")     # design-matrix assembly (X_E / X_T / X_F)
include("fitting/staged.jl")     # sector selectors + staged (frozen) reparameterization
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

# --- the per-site basis-row layout: the contract a sampler builds its gather
# programs against (SLCEMonteCarlo's sweep kernels are the consumer).
include("slce/rowlayout.jl")

# --- affine (non-periodic) displacement fields: the evaluation path a rigid rotation
# or a homogeneous strain needs, and the finite-ω rotational-invariance diagnostic.
include("slce/affine.jl")

# --- physics deliverables of the displacement channel: exact force constants at a
# spin configuration, and their reciprocal-space form.
include("slce/forceconstants.jl")

# --- homogeneous-strain response: the SAME monomial coefficients, contracted with site
# positions instead of being differentiated. Rides on the affine path above, because a
# strain field is not cell-periodic.
include("slce/strain.jl")

# --- the ε-LINEAR magnetoelastic tier that rides on it: the pinned B₁/B₂ convention and
# the strain derivatives of the bilinear couplings. First order in ε and nowhere else —
# both the measure argument (§13 risk 2) and the origin-independence one (§9d) stop there.
include("slce/magnetoelastic.jl")

# --- the mixed derivative that couples the two channels: one u index, one (tangential)
# spin index. Not a strain object at all — no absolute positions, hence no ASR precondition
# — but it belongs to the same deliverable tier.
include("slce/magnonphonon.jl")

# --- volume grids: the K(ε) container. The one strain family that preserves the point
# group, so the SALC keys survive and coefficients can be interpolated across it.
include("slce/strainedmodels.jl")

# Re-expansion around a displaced reference (§9d): the same homogeneous-polynomial
# fact the force constants ride on, used to MOVE the expansion point rather than to
# differentiate at it.
include("slce/effective.jl")

# --- external-engine interop: conversion math in core, engine assembly in extensions ---
# Sunny export (Sunny.System assembled in SLCESunnyExt), consuming the bilinear
# extraction above.
include("interop/sunny.jl")

# --- I/O: persistence (TOML model schema), TOML input files, and the code-agnostic DFT
# data boundary. Concrete DFT-code adapters (e.g. the VASP reader/writer) live in
# SLCETools.jl; the SLCE pipeline only ever sees `spin_datum` / `SLCEDataset`.
include("io/persist.jl")
include("io/input.jl")
include("io/dftsource.jl")
include("io/embset.jl")
include("io/extxyz.jl")    # extended-XYZ training-data container (ASE dialect)
include("fitting/momentfit.jl")  # MomentDataset / fit / MomentModel (needs TrainingDatum)
include("io/phonopy.jl")   # (see below)
include("io/alamode.jl")   # harmonic + anharmonic -> ALAMODE FCSXML for anphon

# --- Public API (exported) --------------------------------------------------------
# The fitting workflow a user reaches for. Construction internals (cluster / neighbor /
# SALC builders, symmetry analysis) are *public but unexported* — see the block below.

# geometry the user builds
export UnitVector3, SpinConfiguration, Trusted
export Lattice, Crystal, n_atoms, cartesian_positions
# how periodic images / symmetry are chosen (passed into `SLCEBasis`)
export AbstractImageSelection, MinimumImage, AllImages
export AbstractSymmetryBackend, NoSymmetry, SpglibBackend
# the SLCE pipeline
export BasisSpec, Sector, SLCEBasis, SLCEDataset, SLCEModel, SLCEFit, fit, refit,
    n_salcs, read_setup
export predict_energy, predict_torque, predict_force, has_torque, has_force
export asr_residual
export unresolvable_columns
# affine-field evaluation and the rotational-invariance diagnostic (NOT a constraint)
export affine_energy, rotational_residual, rotation_transfer_residual
# estimators
export AbstractEstimator, OLS, Ridge, ElasticNet, Lasso, AdaptiveLasso, AdaptiveRidge,
    GroupAdaptiveRidge, FixedCoefficients
# fit diagnostics (predict / residuals / r2 are StatsAPI generics defaulting to the
# energy block; the explicit *_energy / *_torque / *_force forms are the full surface)
export identifiability
export coef, intercept, nobs, dof, predict, residuals, r2,
    r2_energy, rmse_energy, r2_torque, rmse_torque, r2_force, rmse_force,
    rss_energy, rss_torque, rss_force, residuals_energy, residuals_torque,
    residuals_force
export coeftable, SLCECoefficients
# model selection: GCV / effective dof (linear estimators), the cost-aware λ path,
# and the threshold-swept refit front
export gcv, effective_dof, select_fit, LambdaPath, select_support, SupportPath
export cross_validate, CVResult
export to_sunny
# Fitted-model introspection: a code-neutral view of the multipole / bilinear terms of a
# fitted SLCE, the stable contract downstream packages (e.g. the SLCETools.jl mean-field
# samplers) read instead of the SALC-basis internals.
export SpinMultipoleTerm, spin_multipole_terms, bilinear_terms
# The general (channel-decorated) successor of the multipole view, plus the bridge that
# turns a joint model into one the frozen pure-spin surfaces accept.
export DecoratedTerm, decorated_terms, restrict
# lattice dynamics from the displacement channel (spin-configuration dependent)
export ForceConstantSet, force_constants, dynamical_matrix
# homogeneous-strain response (clamped ion, Biot measure); ASR is a hard precondition
export strain_derivatives
# the ε-linear magnetoelastic tier: the pinned cubic B₁/B₂ convention (clamped ion)
export magnetoelastic_constants, exchange_strain_derivatives, ExchangeStrainDerivatives
# the adiabatic magnon-phonon vertices: ∂²E/∂u ∂e, with the spin derivative tangential
export MagnonPhononVertices, magnon_phonon_vertices
# volume grids (K(ε)): the isotropic-strain container and its two readouts
export StrainedModels, model_at, grid_strain_derivative
# the same coefficient set re-expanded around a displaced structure (a relaxed cell, a
# condensed soft mode) — exact, and deliberately NOT an SLCEModel
export EffectiveModel, EffectiveTerm, effective_model
# DFT data I/O: only the code-agnostic boundary is exported; per-code adapters live in
# downstream packages as namespaced submodules (e.g. `SLCETools.VASP.read_poscar`), so
# adding a code touches neither the core nor this export list. The one in-core format
# is Magesty's EMBSET training set — code-agnostic (it carries exactly what a
# spin-only TrainingDatum stores), kept here for legacy-data reuse.
export AbstractDFTSource, TrainingDatum, DatumProvenance,
       spin_datum, lattice_datum, joint_datum, read_configs
export crystal_fingerprint
export write_phonopy, write_alamode
export EmbsetFile, read_embset, read_embset_pair
export ExtxyzFile, read_extxyz, write_extxyz
export MomentSpec, MomentBasis, moment_resolvability
export MomentDataset, MomentFit, MomentModel, predict_moment, rmse_moment,
    moment_band_profile, moment_local_field, moment_coverage, moment_simple_floor

# --- Public, unexported -----------------------------------------------------------
# Reachable as `SLCE.<name>` (and documented), but kept out of the flat `using`
# namespace: the `SLCEBasis` constructor already drives them for you. Power users and the
# test suite reach them by qualification. Declared with the `public` keyword so the
# tier is machine-checkable (`Base.ispublic`, Aqua) instead of a comment-only promise.
public Harmonics, SolidHarmonics, AngularMomentum                    # numeric kernels
public build_neighbor_list, NeighborPair, NeighborList, interplanar_spacing
public analyze_symmetry, n_ops, SymOp, SpaceGroup
public build_clusters, ClusterMember, ClusterOrbit, ClusterSet
public build_salc_basis, evaluate_salc, salcs, SALC, SALCKey, SALCBasis
# the volume grid's own geometry accessors (generic names — reachable, not exported)
public scales, volumes
public Channel, SPIN, DISP, OCC, SiteFactor, SiteDecor                # decoration labels
public Slot                                       # a DecoratedTerm's axis label
public SectorRule                                                     # resolved sector row
public has_spin, has_disp, spin_rank, disp_degree, factors, is_pure_spin, is_soc_free
public spin_decors, spin_ls, rep_scale
public islinear, solve_coefficients
public salc_groups, group_costs, cost_weights, group_freedom        # MC-cost grouping
public ASRReparam, build_asr                                         # ASR machinery
public UnclassifiableBasis          # the resolvability / fitting-door refusal type
public sector_columns                                                # staged-fit selectors
public RowLayout, row_layout, row_index, site_rows!   # sampler row-table contract
# The kelvin ↔ model-energy conversion every downstream sampler shares. Unexported: the
# fitting core has no temperature of its own, it only owns the convention.
public KB_EV, resolve_kt
public save, load                                                    # TOML persistence
public check_moment_gates                     # the moment channel's axis-consistency gates

end # module SLCE
