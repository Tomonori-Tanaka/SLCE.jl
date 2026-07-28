# Reading a fitted model

A fitted [`SLCEModel`](@ref) predicts energies, torques and forces, but sooner or
later you want the *terms* — to hand the Hamiltonian to a sampler, to compare a
coupling against a published number, to see which interactions survived the fit.
That is what this page is about. Downstream packages read a model through these
functions and never through `model.basis.salc_basis.salcs`, so the basis
representation can change without breaking them.

| function | what you get |
|:--|:--|
| [`decorated_terms`](@ref) | every term, any channel — the general surface |
| [`spin_multipole_terms`](@ref) | the frozen pure-spin view (refuses displacement models) |
| [`bilinear_terms`](@ref) | `ls = [1,1]` pairs and `ls = [2]` single-ion terms as Cartesian `3×3` matrices |
| [`restrict`](@ref) | the clamped-ion (`u = 0`) sub-model of a joint model |
| [`to_sunny`](@ref) | a `Sunny.System` (see [Sunny export](sunny.md)) |

## The scale rule

Every term carries a scale factor that the *consumer* applies:

```
E_term = coef · scale · Σ_μ folded[μ] ∏ᵢ fᵢ(μ_i)
```

with one factor `fᵢ` per tensor axis — `Z_{l,m}(ê_a)` on a spin axis,
`|u_a|^{2k} R_{l,m}(u_a)` on a displacement one.

**The scale is `(4π)^(n_spin_slots / 2)`**: one `√(4π)` per *spin* factor. The 4π
is an artifact of the spin-sphere measure, and the displacement kernel is
normalized 4π-free, so displacement axes contribute none.

The pure-spin-era shortcut `(4π)^(body/2)` — one factor per *site* — agrees with
this only when every site carries exactly one spin factor and nothing else. On a
joint model it is simply wrong, and wrong in a way that still produces
plausible-looking numbers. [`DecoratedTerm`](@ref) therefore ships `scale` as a
field: read it, never re-derive it from `body` or `length(atoms)`.

```@example introspect
using SLCE, LinearAlgebra, Random

cr = Crystal(Lattice(Matrix(3.0 * I(3))), [1/6 -1/6; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
spec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
    Sector(spin = (sites = 1:2,), cutoff = 1.1),                      # pure spin
    Sector(spin = [1, 1], disp = (degree = 2,), sites = 2, cutoff = 1.1),  # coupled
    Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)])         # force constants
basis = SLCEBasis(cr, spec)
model = SLCEModel(basis, 0.0, randn(MersenneTwister(1), n_salcs(basis)))

terms = decorated_terms(model)
sig(t) = (t.body, [s.factor.channel for s in t.slots])   # one line per slot signature

println(rpad("body", 6), rpad("slots", 34), rpad("scale", 10), "(4π)^(body/2)")
for s in unique(sig.(terms))
    t = terms[findfirst(x -> sig(x) == s, terms)]
    chans = [Symbol(s.factor.channel) for s in t.slots]
    println(rpad(t.body, 6), rpad(string(chans), 34),
            rpad(round(t.scale; sigdigits = 5), 10),
            round((4π)^(t.body / 2); sigdigits = 5),
            t.scale ≈ (4π)^(t.body / 2) ? "" : "   ← differs")
end
```

The rows where the two rules part ways are the ones with a site that carries **no**
spin factor — a force-constant term, whose sites are pure displacement. There the
shortcut invents a `√(4π)` per site out of nothing. (A coupled term whose every site
happens to hold a spin factor as well as a displacement one agrees by coincidence,
which is exactly why the shortcut survived so long.)

A displacement-decorated model is refused by [`spin_multipole_terms`](@ref) rather than
mis-scaled, and the error names the two ways forward: `decorated_terms(model)`, or
`spin_multipole_terms(restrict(model, :spin))`.

## `restrict` is not a refit

[`restrict(model, :spin)`](@ref restrict) returns the joint model evaluated at
`u = 0` — the clamped-ion sub-model. It is exact: `predict_energy(restrict(m,
:spin), e)` equals `predict_energy(m, e, zeros(3, n))` bitwise, and the result is
an ordinary pure-spin model that every spin-only surface accepts.

!!! warning "Clamped-ion couplings are not spin-only-fit couplings"
    The couplings `restrict` hands back are the values each spin interaction takes
    with every atom at its **ideal reference site**. They are *not* the couplings
    you get by fitting a spin-only model to the same data, and not the ones a
    relaxed or finite-temperature structure exhibits.

    A spin-only model has no displacement coordinate to attribute anything to, so
    whatever the lattice was doing in the training structures is absorbed into its
    spin couplings: the fitted `J` silently carries a `⟨u⟩`- and
    `⟨uu⟩`-renormalized contribution. `restrict` drops exactly that. The gap
    between the two is the physics the joint model exists to separate — not a
    discrepancy to reconcile. If you want renormalized couplings at some
    displacement state, that is a thermodynamic average over the joint model, not
    a restriction of it.

Here is the difference, measured. We generate data from a known joint model on
displaced structures, then fit two models to the *same* energies — one joint, one
spin-only:

```@example introspect
nat = 2
fp = crystal_fingerprint(cr)
prov = DatumProvenance(; reference_id = "ref", reference_fingerprint = fp)
rng = MersenneTwister(7)
randcfg() = reduce(hcat, [normalize(randn(rng, 3)) for _ = 1:nat])

data = map(1:200) do _
    e = randcfg()
    u = 0.08 .* randn(rng, 3, nat)          # the structures are displaced
    TrainingDatum(; energy = predict_energy(model, e, u), directions = e,
                  magmoms = ones(nat), displacements = u, provenance = prov)
end

jds = SLCEDataset(basis, data; use_torque = false, use_force = false)
joint = SLCEModel(fit(SLCEFit, jds, OLS(); asr = false))
clamped = restrict(joint, :spin)             # exact: the joint model at u = 0

# the same energies, fitted with no displacement coordinate at all
sbasis = SLCEBasis(cr, BasisSpec(cr; lmax = 1, sectors = [
    Sector(spin = (sites = 1:2,), cutoff = 1.1)]))
sonly = SLCEModel(fit(SLCEFit, SLCEDataset(sbasis, [d.directions for d in data],
                                           [d.energy for d in data]), OLS()))

println("clamped-ion (restrict):  ", round.(coef(clamped); sigdigits = 3))
println("spin-only refit:         ", round.(coef(sonly); sigdigits = 3))
println("max |difference|:        ",
        round(maximum(abs, coef(clamped) .- coef(sonly)); sigdigits = 3))
```

The spin-only fit reproduces the *energies* it was given about as well as the joint
one does, and lands on different couplings — which is the whole point: those two
numbers answer different questions, and only the joint model can tell you which
part of the coupling came from the lattice.

## Handing the lattice channel to phonopy

[`write_phonopy`](@ref) writes a `ForceConstantSet` as a phonopy calculation — a
`POSCAR` and a `FORCE_CONSTANTS` — so band structures, DOS, thermal properties,
group velocities and thermal conductivity come from phonopy rather than from
reimplementations here:

```julia
fcs = force_constants(model; spins = afm, order = 2)
out = write_phonopy("phonons/afm", fcs)     # out.dim is the --dim to pass
```

The magnetic state is already baked into `fcs`, and phonopy has no notion of one:
export two states to two directories and the difference between the band structures
*is* the magnetoelastic content of the model. Note the two files must be used
together — the `POSCAR` is what fixes the atom order the `FORCE_CONSTANTS` is written
in, and phonopy's supercell ordering is pinned by a test that runs phonopy itself
(`test/phonopy/`), because a permuted export still diagonalizes and still has three
acoustic modes at Γ.

### Anharmonic constants: ALAMODE

phonopy takes the harmonic channel and stops. [`write_alamode`](@ref) writes an
ALAMODE `FCSXML` carrying the harmonic set *and* any anharmonic ones, which is what
`anphon` needs for relaxation-time thermal conductivity, self-consistent phonons and
Grüneisen parameters:

```julia
write_alamode("si.xml", force_constants(model; spins = e, order = 2),
                        force_constants(model; spins = e, order = 3))
```

All sets must come from one crystal and one spin state — they are the expansion of a
single energy surface, and `anphon` has no notion of a magnetic state.

## Force constants and phonons

[`force_constants`](@ref) differentiates the energy with respect to displacements at
`u = 0`, **with the spins held fixed** — so the lattice dynamics you get out is the
lattice dynamics *of that magnetic state*. The derivatives are exact, not finite
differences: every displacement factor is a homogeneous polynomial, so the constants
are read off its monomial coefficients.

```@example introspect
spins = randcfg()
fcs = force_constants(joint; spins = spins, order = 2)

for ((atoms, shifts), Φ) in sort(collect(fcs.constants); by = k -> (k[1][1], k[1][2]))
    println("Φ[(", atoms[1], ",0), (", atoms[2], ",", Tuple(shifts[2]), ")]  ‖Φ‖ = ",
            round(norm(Φ); sigdigits = 4))
end
```

Indices follow the lattice-dynamics convention `Φ[(a,0),(b,R)] = ∂²E/∂uₐ(0)∂u_b(R)`,
anchored so the first index is in the home cell. The reverse ordering is a separate
key, equal to the transpose.

[`dynamical_matrix`](@ref) folds them into reciprocal space at a wavevector given in
**fractional reciprocal coordinates**. Pass `masses` (in **amu**) to get an `ω²`
spectrum; omit it for the bare force-constant matrix:

```@example introspect
D0 = dynamical_matrix(fcs, [0.0, 0.0, 0.0])
println("eigenvalues of D(0): ", round.(sort(real(eigvals(Hermitian(D0)))); sigdigits = 3))
```

That call passed no `masses`, so those are eigenvalues of the **bare** force-constant
matrix, in eV/Å² — not frequencies of anything. Pass `masses` in amu and they become
`ω²` **in eV/Å²/amu**, which is still not a frequency until you convert it:

```@example introspect
λ = sort(real(eigvals(Hermitian(dynamical_matrix(fcs, [0.0, 0.0, 0.0];
                                                 masses = [55.845, 55.845])))))
signed_sqrt(x) = x < 0 ? -sqrt(-x) : sqrt(x)          # imaginary modes stay negative
println("ω² [eV/Å²/amu]: ", round.(λ; sigdigits = 3))
println("ν  [THz]      : ", round.(signed_sqrt.(λ) .* 15.633302; sigdigits = 3))
```

The `15.633302` is phonopy's `VaspToTHz` (the `2π` is already in it, so this is the
ordinary frequency `ν`, not the angular `ω`); a further `× 33.35641` gives cm⁻¹. The
`test/alamode/` suite pins both against `anphon`'s own cm⁻¹ output.

None of those are zero — and they should be. Three eigenvalues of `D(0)` must vanish:
those are the acoustic branches, the statement that translating the whole crystal
costs no energy. They do not here because `joint` above was deliberately fitted with
`asr = false`. Fit the same data under the acoustic sum rule — the default — and they
appear:

```@example introspect
constrained = SLCEModel(fit(SLCEFit, jds, OLS(); asr = true))
Dc = dynamical_matrix(force_constants(constrained; spins = spins), [0.0, 0.0, 0.0])
println("asr = true:   ", round.(sort(real(eigvals(Hermitian(Dc)))); sigdigits = 3))
println("asr_residual: ", round(asr_residual(constrained); sigdigits = 3),
        "   vs unconstrained ", round(asr_residual(joint); sigdigits = 3))
```

That is what `fit(...; asr = true)` buys, and [`asr_residual`](@ref) is the direct
check on any model before you trust its phonons. (Zero acoustic frequencies are not
the same as *stable* phonons: a negative eigenvalue is a real instability of the
fitted model, not a bug in the fold.)

The magnetic contribution to the dynamics is the difference between two spin states:

```@example introspect
D_other = dynamical_matrix(force_constants(joint; spins = randcfg()), [0.0, 0.0, 0.0])
println("max |ΔD(0)| between two magnetic states: ",
        round(maximum(abs, D_other .- D0); sigdigits = 4))
```

## Homogeneous strain

Force constants answer "what does it cost to move one atom?". The other lattice
question is "what does it cost to deform the cell?", and that one cannot go through
[`predict_energy`](@ref) at all: a strain moves the atom at `R + L` by `ε·(R + L)`,
which grows with the lattice vector, so the field is not cell-periodic. It goes through
the affine path instead — the same machinery
[rotational invariance](fitting.md#Rotational-invariance-(measured,-not-imposed)) is
measured with.

[`strain_derivatives`](@ref) returns the exact derivatives at the model's own reference:

```@example introspect
D1 = strain_derivatives(constrained; spins = spins, order = 1)
println("∂E/∂ε  (= V·σ)  = ", round.(D1; sigdigits = 3))
```

`D1` is exactly zero, and not by luck. A displacement site factor is a homogeneous
polynomial of degree `2k + l`, so substituting the affine field turns a term of total
displacement degree `n` into a degree-`n` polynomial in `ε` — the order-`n` strain
derivative collects terms of degree `n` and nothing else. The basis on this page declares
`disp = (degree = 2,)` only, so it has no ε-linear content and its reference is
stress-free in the displacement channel by construction. The ε-linear magnetoelastic
constants live in the `degree = 1` (relative-displacement) sectors — the same ones that
feed the forces — never in single-site `degree = 2` monomials, which enter the strain
expansion only at `O(ε²)`.

The strain measure is **Biot** (Seth–Hill `m = 1`): the deformation is *defined* as
`F = I + ε` with `ε` symmetric, which makes the substitution `u_i − u_j = ε·d_ij` exact
rather than a linearization. `order = 1` is the same number in every member of the
Seth–Hill family; `order = 2` and beyond are not, so record the measure — and the spin
state — with any elastic constant you quote.

### The acoustic sum rule is required, and at `order ≥ 2` it is not enough

A strain displaces site `i` by `ε·(R_i − origin)`. Moving the origin adds the uniform
translation `ε·t` to every displacement, changing the energy by `ε : (Σ_i ∇_i E)` — so
the strain response is origin-independent exactly when `Σ_i ∇_i E = 0`, which is the
acoustic sum rule. On a model that does not satisfy it there is no answer to give: the
origin is unbounded, so the number would be whatever coordinate system you happened to
write the crystal in. [`strain_derivatives`](@ref) therefore *throws* on `joint`, the
unconstrained fit from earlier on this page:

```@example introspect
try
    strain_derivatives(joint; spins = spins)
catch err
    println(join(first(split(err.msg, ". "), 2), ". "), ".")
end
```

The tolerance it enforces is tighter than the one you would use to judge
[`asr_residual`](@ref) elsewhere, because the strain path weights the residual by the
site positions `|R_i|` before it reaches the answer.

At `order = 2` the sum rule stops being sufficient, and this is worth understanding
before quoting an elastic constant. The ASR is the identity `Σ_a ∂E/∂u_a ≡ 0` in the
**atom** variables — an identity on *cell-periodic* fields. At `ε = 0` the affine field
is zero, hence periodic, hence covered; one order further out it is not, and what would
be needed is the stronger per-**site** statement, which the constraint does not imply.
The gap is the home-**image gauge**: a term's content is anchored at whichever image the
cluster list calls "home", while its translation partners sit on the images the clusters
actually reach. When those differ, the cancellation is split across cells and the
per-cell energy picks up a piece linear in the cell's position.

The crystal on this page is exactly that case — its two atoms sit at `±1/6`, so each
atom's bonded partner is a *far* image of the one in the home cell — and
`strain_derivatives` refuses rather than returning a description-dependent number:

```@example introspect
try
    strain_derivatives(constrained; spins = spins, order = 2)
catch err
    println(first(split(err.msg, ". "), 1)[1], ".")
end
```

It is not a small correction. The escape hatch exists to make that measurable, never to
be used for a result:

```@example introspect
raw = strain_derivatives(constrained; spins = spins, order = 2, check_origin = false)
alt = strain_derivatives(constrained; spins = spins, order = 2,
                         origin = [7.0, -3.0, 2.5], check_origin = false)
println("‖ΔD²‖ / ‖D²‖ between two origins: ", round(norm(alt - raw) / norm(raw);
                                                    sigdigits = 3))
```

The fix is the crystal **description**, not the fit: place the atoms so that each one's
home representative is the image its clusters use. The same chain written with the
bonded partner in the home cell answers, and its check passes.

## Magnetoelastic coupling: the ε-linear tier

`order = 1` is where the magnetoelastic deliverables live, and the two paragraphs above
are the reason it is a *tier* and not just the first term of a series: the sum rule buys
origin independence exactly where the affine field is periodic, and the Seth–Hill measure
drops out of ε-linear content unconditionally. Everything second order in `ε` is
conditional on both; nothing here is.

The basis on this page carries no `degree = 1` content, so it has no magnetoelastic
coupling at all. Here is one that does — the same chain with a bilinear `ls = [1,1]` pair
dressed by a relative displacement, fitted under the sum rule:

```@example introspect
mespec = BasisSpec(cr; lmax = 1, pmax = 2, sectors = [
    Sector(spin = [1, 1], disp = (degree = 1,), sites = 2, cutoff = 1.1),  # magnetoelastic
    Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1)])              # force constants
mebasis = SLCEBasis(cr, mespec)
metruth = SLCEModel(mebasis, 0.0, randn(MersenneTwister(4), n_salcs(mebasis)))
medata = map(1:300) do _
    e = randcfg()
    u = 0.08 .* randn(rng, 3, nat)
    TrainingDatum(; energy = predict_energy(metruth, e, u), directions = e,
                  magmoms = ones(nat), displacements = u, provenance = prov)
end
memodel = SLCEModel(fit(SLCEFit, SLCEDataset(mebasis, medata; use_torque = false,
                                             use_force = false), OLS(); asr = true))
dJ = exchange_strain_derivatives(memodel)
println(dJ)
for ((a, b, R), T) in dJ.pairs
    println("bond ($a, $b, $R):  ∂M^{xx}/∂ε = ", round.(T[1, 1, :, :]; sigdigits = 3))
end
```

[`exchange_strain_derivatives`](@ref) is the model's `dJ/dr`, resolved per bond and per
strain component instead of collapsed onto a bond length: `T[α, β, γ, δ]` is
`∂M_ab^{αβ}/∂ε_{γδ}`, with `M_ab` the bilinear matrix [`bilinear_terms`](@ref) reports.
The `γ, δ` pair is **unsymmetrized** — the derivative with respect to a general affine
map — so contract it with `ε` as it stands.

Those bond couplings are not an independent readout: contracting them with a spin
configuration has to rebuild the cell's total ε-linear response, which
[`strain_derivatives`](@ref) reaches by an entirely different route (monomial
coefficients contracted with site positions, never touching the tesseral-to-Cartesian
conversion). That identity is the acceptance gate on this deliverable:

```@example introspect
e = randcfg()
recon = zeros(3, 3)
for ((a, b, _), T) in dJ.pairs, γ = 1:3, δ = 1:3
    recon[γ, δ] += dot(e[:, a], T[:, :, γ, δ] * e[:, b])
end
D = strain_derivatives(memodel; spins = e, order = 1, symmetrize = false)
println("‖rebuilt − total‖ / ‖total‖ = ", round(norm(recon - D) / norm(D); sigdigits = 3))
```

!!! warning "The per-bond split is origin-dependent unless the bond's own content is relative"
    A strain moves each site by `ε·(R_s − origin)`, so every term carries an *absolute*
    position. The origin cancels from the total by the sum rule — but that is a statement
    about the whole model, not about one bond. A bond whose displacement content is not
    purely relative (`u_b − u_a`) has an `∂M/∂ε` that depends on where the origin was put,
    and is then not a property of the bond at all. Bond orbits with a site-swap operation
    admit only the relative combination, so the check normally passes silently; when it
    does not, the function throws rather than handing back a convention-dependent
    coupling.

### The two cubic constants, and knowing when they mean anything

[`magnetoelastic_constants`](@ref) compresses the same content into `B₁` and `B₂`, in a
convention this package pins and gates (`test/unit/test_magnetoelastic.jl`):

```
E_me / V = B₁ Σ_i ε_ii (α_i² − 1/3) + 2 B₂ Σ_{i<j} ε_ij α_i α_j
```

**tensor** shear `ε_ij` (engineering `γ = 2ε` is an I/O view, never the internal object),
that summation range, that sign, `E_me` an energy *density*. Every one of those is a
convention that differs between papers, and each disagreement is a clean factor of two or
a flipped sign in a published number.

The form is cubic. This chain is not, and the result says so rather than returning two
plausible numbers:

```@example introspect
c = magnetoelastic_constants(memodel)
println("B₁ = ", round(c.B1; sigdigits = 3), "   B₂ = ", round(c.B2; sigdigits = 3),
        "   ion = ", c.ion, "   residual = ", round(c.residual; sigdigits = 3))
```

`residual` is the fraction of the model's *magnetization-dependent* ε-linear response that
the two-constant cubic form does not explain — computed by projecting the exact response
at 19 magnetization directions onto that form. Near zero, `B₁`/`B₂` *are* the coupling;
`O(1)`, as here, and they are a summary of something else, which is what the warning
says. Read `ion = :clamped` as part of the answer: no internal-strain relaxation has been
applied, and the clamped-versus-relaxed difference is routinely a factor ~2.

## Magnon–phonon vertices

The force constants differentiate the energy twice in `u`; the bilinear couplings twice in
the spin directions. What couples the two subsystems is the **mixed** derivative, and it is
its own deliverable:

```@example introspect
V = magnon_phonon_vertices(memodel; spins = e)
println(V)
for ((a, b, R), T) in sort(collect(V.vertices); by = first)
    println("displace $a, vary spin $b at $R:  ‖∂²E/∂u∂e‖ = ", round(norm(T); sigdigits = 3),
            "   ‖T·ê_b‖ = ", round(norm(T * e[:, b]); sigdigits = 3))
end
```

The pair is **ordered** — `a` is displaced, `b` is the magnetic site — unlike the
undirected bond keys of [`bilinear_terms`](@ref). And the second column is the point: the
spin derivative is *tangential*, with the radial direction projected out, because a magnon
amplitude cannot change the length of a spin. That is what lets the result come back in
Cartesian components without pinning a local-frame convention on the caller: project
`T[α, :]` onto whichever `(f₁, f₂)` your magnon basis uses and the answer is the same.

Only terms with exactly one **degree-1** displacement factor *and* a spin factor
contribute. A magnetoelastic sector declared at `disp = (degree = 2,)` produces none — that
content feeds the spin-dependent force constants instead — which is the same trap
[`force_constants`](@ref) warns about from the other side.

"Adiabatic" in the docstring is a scope statement, not a hedge about accuracy: these are
derivatives of the static energy surface, so retardation, the Berry-phase term that carries
phonon angular momentum, and spin-lattice relaxation lie outside a static cluster expansion
by construction — no basis or truncation choice brings them in.

## Volume grids: when one model is not enough

Everything above lives at one volume. A model's coefficients are themselves functions of
the cell — that is what magnetovolume coupling *is* — and capturing it needs a grid of fits
at scaled cells, which is what [`StrainedModels`](@ref) holds.

Isotropic scaling is the one strain family that preserves the point group, so the SALC keys
survive and coefficients can be interpolated. Two rules make the grid assemblable: the
cutoffs must be expressed in units of the (strained) `d_NN` so that no neighbour shell
crosses one as the volume changes, and every point's basis must be closed under
**re-expansion** — a model at scale `s` is the reference re-expanded around the scaled
geometry, and that shift generates lower degrees, so a `degree = 2` sector needs
`degree = 1` and a spin-dressed `degree = 1` sector needs a pure-spin one:

```@example introspect
gcr(s) = Crystal(Lattice(Matrix(3.0s * I(3))), [0.0 1/3; 0.0 0.0; 0.0 0.0], [1, 1], ["Fe"])
gspec(cr, s) = BasisSpec(cr; lmax = 2, pmax = 2, sectors = [
    Sector(spin = (sites = 1:2,), cutoff = 1.1s),                            # closure
    Sector(spin = [1, 1], disp = (degree = 1,), sites = 1:2, cutoff = 1.1s),
    Sector(disp = (degree = 1,), sites = 1:2, cutoff = 1.1s),                # closure
    Sector(disp = (degree = 2,), sites = 1:2, cutoff = 1.1s)])
gbasis(s) = (c = gcr(s); SLCEBasis(c, gspec(c, s)))

gtruth = SLCEModel(gbasis(1.0), -1.5, 0.2 .* randn(MersenneTwister(5), n_salcs(gbasis(1.0))))

function fit_at(s)
    b = gbasis(s)
    pv = DatumProvenance(; reference_id = "s$s", reference_fingerprint = crystal_fingerprint(gcr(s)))
    d = map(1:200) do _
        ec = randcfg()
        u = 0.05 .* randn(rng, 3, 2)
        TrainingDatum(; energy = affine_energy(gtruth, ec, (s - 1) * Matrix(1.0I(3)); base = u),
                      directions = ec, magmoms = ones(2), displacements = u, provenance = pv)
    end
    return SLCEModel(fit(SLCEFit, SLCEDataset(b, d; use_torque = false, use_force = false),
                         OLS(); asr = true))
end

grid = [0.98, 0.99, 1.0, 1.01, 1.02]
sm = StrainedModels([fit_at(s) for s in grid], grid)
println(sm)
```

Now the check that makes the grid trustworthy. There are two strain derivatives here and
they are different objects: the **intra-model incremental** one, taken inside a single model
with its coefficients held fixed, and the **grid finite difference**, taken along the grid,
which also carries the drift of those coefficients. They have to agree:

```@example introspect
es = randcfg()
for s in (0.99, 1.0, 1.01)
    D = strain_derivatives(model_at(sm, s); spins = es, order = 1)
    intra = D[1, 1] + D[2, 2] + D[3, 3]              # ε = ηI contracts to the trace
    println("s = $s:  intra-model = ", round(intra; sigdigits = 8),
            "   grid = ", round(grid_strain_derivative(sm, s; spins = es); sigdigits = 8))
end
```

That agreement says the expansion captures the strain response through its *displacement
channel* rather than through coefficient drift, which is the condition under which one grid
point may be used at finite strain. A disagreement is not noise: drop the pure-spin sector
above and it becomes 5%, drop the lattice `degree = 1` sector and 1%. Both derivatives are
`dE/dη` with `η` measured from the reference the derivative is taken *at* — `dE/dη` and
`dE/ds` differ by exactly the factor `s`, which agrees at the unstrained point and is off by
the strain everywhere else.
