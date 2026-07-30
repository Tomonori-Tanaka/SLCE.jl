# Verification: angular momentum

```@meta
CurrentModule = SLCE
```

The **spin** channel's SALC projection rests on three pieces of angular-momentum
machinery, all implemented from scratch in the public (unexported)
[`AngularMomentum`](@ref) submodule: the Clebsch–Gordan coefficients (Racah single-sum formula,
Condon–Shortley phase), the real Wigner-``D`` matrices in the tesseral basis, and
the complex→real unitary ``U^{(l)}``. This page renders the checks of
`test/unit/test_angmom.jl` in human-readable form — every table and summary below
is **computed when the documentation is built**, and each section ends in an
assertion, so a regression fails the (strict) docs build just as it fails the test
suite. The unit tests remain the machine gate; this page is the same evidence,
made readable.

!!! note "Scope: the spin channel's kernels only"
    The displacement channel's own machinery — the ``4\pi``-free solid harmonics and the
    monomial-coefficient function four consumers share (the ASR builder,
    `force_constants`, `strain_derivatives`, `effective_model`), the mixed-channel
    counting oracle, and the ASR rank identity — is gated in the test suite but has no
    readable chapter here yet.

## Clebsch–Gordan coefficients: known values

Closed-form values from the standard tables (Varshalovich, *Quantum Theory of
Angular Momentum*), including three selection-rule zeros. The notation
`⟨j₁ m₁; j₂ m₂ → J M⟩` stands for ``\langle j_1 m_1\, j_2 m_2 | J M \rangle``:

```@example am
using SLCE
using Printf, Markdown

const CG = SLCE.AngularMomentum.clebsch_gordan

known = [
    (1, 0, 1, 0, 0, 0,   "−1/√3",              -1 / sqrt(3)),
    (1, 1, 1, -1, 0, 0,  "1/√3",               1 / sqrt(3)),
    (1, 1, 1, -1, 1, 0,  "1/√2",               1 / sqrt(2)),
    (1, 1, 1, -1, 2, 0,  "1/√6",               1 / sqrt(6)),
    (2, 0, 2, 0, 0, 0,   "1/√5",               1 / sqrt(5)),
    (1, 1, 1, 1, 2, 2,   "1 (stretched)",      1.0),
    (1, 1, 1, 1, 0, 0,   "0 (M conservation)", 0.0),
    (1, 0, 1, 0, 3, 0,   "0 (triangle rule)",  0.0),
    (1, 0, 1, 0, 1, 0,   "0 (odd-J symmetry)", 0.0),
]

function known_value_table(rows; atol = 1e-14)
    io = IOBuffer()
    println(io, "| coefficient | exact | computed | abs. error | pass |")
    println(io, "|:---|:---|---:|---:|:---:|")
    worst = 0.0
    for (j1, m1, j2, m2, J, M, expr, exact) in rows
        val = CG(j1, m1, j2, m2, J, M)
        err = abs(val - exact)
        worst = max(worst, err)
        @printf(io, "| ⟨%d %d; %d %d → %d %d⟩ | %s | %.15f | %.1e | %s |\n",
                j1, m1, j2, m2, J, M, expr, val, err, err <= atol ? "✔" : "✗")
    end
    worst <= atol || error("Clebsch–Gordan known-value check failed (worst $worst)")
    return Markdown.parse(String(take!(io)))
end

known_value_table(known)
```

## Clebsch–Gordan coefficients: orthogonality sweeps

Individual values can be right by accident; the two orthogonality relations tie
the *whole* coefficient family together. First,

```math
\sum_{m_1 m_2} \langle j_1 m_1\, j_2 m_2 | J M \rangle
              \langle j_1 m_1\, j_2 m_2 | J' M \rangle = \delta_{J J'},
```

swept over every ``(j_1, j_2, J, J', M)`` with ``j_1, j_2 \le 3``:

```@example am
function cg_orthonormality(jmax)
    checked = 0
    maxdev = 0.0
    for j1 = 0:jmax, j2 = 0:jmax
        for J = abs(j1 - j2):(j1 + j2), Jp = abs(j1 - j2):(j1 + j2), M = -J:J
            s = 0.0
            for m1 = -j1:j1, m2 = -j2:j2
                s += CG(j1, m1, j2, m2, J, M) * CG(j1, m1, j2, m2, Jp, M)
            end
            maxdev = max(maxdev, abs(s - (J == Jp ? 1.0 : 0.0)))
            checked += 1
        end
    end
    return (sums_checked = checked, max_deviation = maxdev)
end

ortho = cg_orthonormality(3)
```

and the completeness relation in the other direction,

```math
\sum_{J M} \langle j_1 m_1\, j_2 m_2 | J M \rangle
           \langle j_1 m_1'\, j_2 m_2' | J M \rangle
         = \delta_{m_1 m_1'} \delta_{m_2 m_2'},
```

swept over every ``(m_1, m_2, m_1', m_2')`` for the same ``j_1, j_2 \le 3``:

```@example am
function cg_completeness(jmax)
    checked = 0
    maxdev = 0.0
    for j1 = 0:jmax, j2 = 0:jmax
        for m1 = -j1:j1, m2 = -j2:j2, m1p = -j1:j1, m2p = -j2:j2
            s = 0.0
            for J = abs(j1 - j2):(j1 + j2)
                s += CG(j1, m1, j2, m2, J, m1 + m2) *
                     CG(j1, m1p, j2, m2p, J, m1 + m2)
            end
            expected = (m1 == m1p && m2 == m2p) ? 1.0 : 0.0
            maxdev = max(maxdev, abs(s - expected))
            checked += 1
        end
    end
    return (sums_checked = checked, max_deviation = maxdev)
end

complete = cg_completeness(3)
```

```@example am
max(ortho.max_deviation, complete.max_deviation) <= 1e-12 ||
    error("Clebsch–Gordan orthogonality sweep failed")
println("both sweeps pass at the 1e-12 tolerance")
```

## Real Wigner-D matrices

[`wignerD_real`](@ref SLCE.AngularMomentum.wignerD_real) must satisfy the
functional identity ``Z_{l m}(R\hat{\boldsymbol u}) = \sum_{m'} \Delta^{(l)}_{m m'}(R)\,
Z_{l m'}(\hat{\boldsymbol u})`` for *every* direction — the SALC projector applies it to
arbitrary spin configurations. The sweep below checks that identity on fresh
random directions (never used in the internal least-squares construction), for
proper **and** improper operations (the second half of the rotations is composed
with the inversion), plus orthogonality ``\Delta \Delta^{\mathsf T} = I`` and the
inversion parity ``\Delta(-I) = (-1)^l I``:

```@example am
using LinearAlgebra, Random
using SLCE: Harmonics

const Dreal = SLCE.AngularMomentum.wignerD_real

rand_unit(rng) = normalize(randn(rng, 3))
function rand_rotation(rng)
    Q = Matrix(qr(randn(rng, 3, 3)).Q)
    Q[:, 1] .*= sign(det(Q))            # make it proper; improper = -Q below
    return Q
end

function wignerD_deviations(; lmax = 4, nrot = 20, npts = 5, seed = 7)
    rng = MersenneTwister(seed)
    dev_orth = 0.0
    dev_fun = 0.0
    for k = 1:2nrot
        R = rand_rotation(rng)
        k > nrot && (R = -R)            # improper half: inversion ∘ rotation
        for l = 0:lmax
            Δ = Dreal(l, R)
            dev_orth = max(dev_orth, opnorm(Δ * Δ' - I))
            for _ = 1:npts
                u = rand_unit(rng)
                lhs = [Harmonics.Zlm(l, m, R * u) for m = -l:l]
                rhs = Δ * [Harmonics.Zlm(l, m, u) for m = -l:l]
                dev_fun = max(dev_fun, maximum(abs.(lhs - rhs)))
            end
        end
    end
    parity = maximum(0:lmax) do l
        opnorm(Dreal(l, -Matrix(1.0I, 3, 3)) - (-1.0)^l * I)
    end
    return (orthogonality = dev_orth, functional_identity = dev_fun,
            inversion_parity = parity)
end

wd = wignerD_deviations()
```

```@example am
maximum(wd) <= 1e-9 || error("Wigner-D verification failed")
println("all Wigner-D checks pass at the 1e-9 tolerance ",
        "(200 operations × l ≤ 4 × 5 fresh directions)")
```

## Complex→real unitary

[`c2r_matrix`](@ref SLCE.AngularMomentum.c2r_matrix) is compared against the
closed form of the tesseral transformation (the paper's Eq. B2) — a
convention-independent anchor written out digit by digit, plus unitarity:

```@example am
const c2r = SLCE.AngularMomentum.c2r_matrix

function c2r_deviations(lmax)
    dev_form = 0.0
    dev_unit = 0.0
    for l = 0:lmax
        U = c2r(l)
        ix(m) = m + l + 1
        ref = zeros(ComplexF64, 2l + 1, 2l + 1)
        ref[ix(0), ix(0)] = 1
        for m = 1:l
            s = 1 / sqrt(2.0)
            ref[ix(m), ix(m)] = (-1.0)^m * s
            ref[ix(m), ix(-m)] = s
            ref[ix(-m), ix(-m)] = im * s
            ref[ix(-m), ix(m)] = -im * (-1.0)^m * s
        end
        dev_form = max(dev_form, maximum(abs.(U - ref)))
        dev_unit = max(dev_unit, opnorm(U' * U - I))
    end
    return (closed_form = dev_form, unitarity = dev_unit)
end

u = c2r_deviations(3)
```

```@example am
maximum(u) <= 1e-12 || error("complex→real unitary verification failed")
println("closed form and unitarity pass at the 1e-12 tolerance (l ≤ 3)")
```

## Worked example: tesseral CG coefficients for ``l_1 = l_2 = 1``

The SALC construction couples products of *tesseral* harmonics, so it needs the
CG coefficients **in the real basis** — this section derives them from the
familiar complex ones for the simplest case, two ``l = 1`` sites, using the exact
code path of the basis build
([`coeff_tensor_complex`](@ref SLCE.AngularMomentum.coeff_tensor_complex) →
[`complex_to_real_tensor`](@ref SLCE.AngularMomentum.complex_to_real_tensor)).

**Step 1 — the complex CG coefficients.** The standard coupling

```math
|L\, M\rangle = \sum_{m_1 m_2}
    \langle 1\, m_1\, 1\, m_2 | L\, M \rangle\, |1\, m_1\rangle |1\, m_2\rangle,
\qquad L \in \{0, 1, 2\},
```

conserves ``M = m_1 + m_2``, so the nonzero coefficients form one ``3 \times 3``
matrix per ``L``: ``C^{(L)}_{m_1 m_2} = \langle 1 m_1\, 1 m_2 | L,\, m_1{+}m_2
\rangle``, rows ``m_1 = -1, 0, +1`` top to bottom and columns ``m_2`` likewise.
Computed from the package's `clebsch_gordan` and pretty-printed by matching each
entry against ``\pm p/\sqrt{q}`` (an unmatched entry throws — the display *is* a
check):

```@example am
using LaTeXStrings

function exact_string(x; tol = 1e-12)
    abs(x) < tol && return "0"
    for q = 1:12, p = 1:3
        if abs(abs(x) - p / sqrt(q)) < tol
            sign = x < 0 ? "-" : ""
            return q == 1 ? string(sign, p) :
                   string(sign, "\\frac{", p, "}{\\sqrt{", q, "}}")
        end
    end
    error("no exact form ±p/√q matched for $x")
end

function texmatrix(A)
    rows = [join([exact_string(A[i, j]) for j = 1:size(A, 2)], " & ")
            for i = 1:size(A, 1)]
    return "\\begin{pmatrix} " * join(rows, " \\\\ ") * " \\end{pmatrix}"
end

displaymath(tex) = LaTeXString("\$\$\\begin{gathered}" * tex * "\\end{gathered}\$\$")

Ccg = [[CG(1, m1, 1, m2, L, m1 + m2) for m1 = -1:1, m2 = -1:1] for L = 0:2]

displaymath("C^{(0)} = " * texmatrix(Ccg[1]) *
            " \\qquad C^{(1)} = " * texmatrix(Ccg[2]) *
            " \\\\[0.8em] C^{(2)} = " * texmatrix(Ccg[3]))
```

**Step 2 — the complex→real change of basis.** The tesseral harmonics are
``Z_{1,\mu} = \sum_m U^{(1)}_{\mu m} Y_1^m`` with the (verified-above) closed form

```math
\begin{pmatrix} Z_{1,-1} \\ Z_{1,0} \\ Z_{1,+1} \end{pmatrix}
  = U^{(1)}
\begin{pmatrix} Y_1^{-1} \\ Y_1^{0} \\ Y_1^{+1} \end{pmatrix},
\qquad
U^{(1)} = \frac{1}{\sqrt{2}}
\begin{pmatrix} i & 0 & i \\ 0 & \sqrt{2} & 0 \\ 1 & 0 & -1 \end{pmatrix},
```

which is exactly the statement ``\bigl(Z_{1,-1}, Z_{1,0}, Z_{1,+1}\bigr) \propto
(y, z, x)``. Inverting (``U`` is unitary, ``Y = U^\dagger Z``) and substituting
into Step 1 turns the coupled function ``T_{L M} = \sum C^{(L)} Y_1^{m_1}
Y_1^{m_2}`` into an expansion over products ``Z_{1\mu_1} Z_{1\mu_2}`` — each site
axis picks up ``\overline{U^{(1)}}`` — and rotating the multiplet axis
``\tilde{T}_{L \tilde{M}} = \sum_M U^{(L)}_{\tilde{M} M} T_{L M}`` makes the
resulting family itself tesseral:

```math
\tilde{C}^{(L)}_{\mu_1 \mu_2, \tilde{M}}
  = i^{\,L - l_1 - l_2} \sum_{m_1 m_2 M}
    \overline{U^{(1)}_{\mu_1 m_1}}\, \overline{U^{(1)}_{\mu_2 m_2}}\,
    U^{(L)}_{\tilde{M} M}\,
    \langle 1\, m_1\, 1\, m_2 | L\, M \rangle .
```

Without the prefactor the entries come out purely real for even ``l_1 + l_2 - L``
and purely imaginary for odd; the global phase ``i^{L - l_1 - l_2}`` (one number
per ``L``, physically irrelevant) makes every sector real. This is precisely what
`complex_to_real_tensor` implements — it errors if any imaginary residue survives.

**Step 3 — the tesseral CG coefficients.** Running that pipeline
(`build_real_bases([1, 1])`) and printing each ``\tilde{M}`` slice:

```@example am
const AM = SLCE.AngularMomentum
real11 = Dict(Lf => T for (_, Lf, T) in AM.build_real_bases([1, 1]))

slices(L) = ["\\tilde C^{(" * string(L) * ")}_{\\tilde M = " * string(k - L - 1) *
             "} = " * texmatrix(real11[L][:, :, k]) for k = 1:(2L + 1)]

displaymath(slices(0)[1] *
            " \\\\[0.8em] " * join(slices(1), " \\quad ") *
            " \\\\[0.8em] " * join(slices(2)[1:3], " \\quad ") *
            " \\\\[0.8em] " * join(slices(2)[4:5], " \\quad "))
```

Rows and columns are ``\mu_1, \mu_2 = -1, 0, +1``, i.e. the ``(y, z, x)``
components of the two spins. Reading the slices back as functions of two unit
vectors:

- ``L = 0``: ``\tilde{C}^{(0)} = \delta_{\mu_1 \mu_2} / \sqrt{3}`` — the
  Heisenberg invariant ``\hat{\boldsymbol e}_1 \cdot \hat{\boldsymbol e}_2 / \sqrt{3}``.
- ``L = 1``: the three slices are the Levi-Civita tensor, ``\tilde{C}^{(1)}
  = \varepsilon / \sqrt{2}`` — the cross product
  ``(\hat{\boldsymbol e}_1 \times \hat{\boldsymbol e}_2) / \sqrt{2}``, an **axial** vector:
  its parity is ``(-1)^{l_1 + l_2} = +1``, not ``(-1)^L``. That mismatch is what the
  even-``\sum_i l_i`` time-reversal screen buys back. The SALC projector never forms
  ``\Delta^{(L)}(R)`` at all: it rotates each *site* axis with the full matrix (improper
  operations included) and reads the ``L_f`` action off by contraction, and because the
  screen makes ``\det(R)^{\sum_i l_i} \equiv +1``, the axial spin action equals the polar
  one. So one polar Wigner-``D`` cache is exact for both channels and **no proper /
  improper case distinction exists anywhere in the projector** — the same cache also
  serves the polar displacement axes of a joint basis.
- ``L = 2``: the five quadrupole combinations — reading ``\tilde{M} = -2 \dots 2``:
  ``xy``-, ``yz``-type, ``(2 z_1 z_2 - x_1 x_2 - y_1 y_2)/\sqrt{6}``, ``zx``-type,
  and ``(x_1 x_2 - y_1 y_2)/\sqrt{2}``.

The closed forms and the defining property — the coupled products transform under
simultaneous (proper) rotation of both spins exactly like ``Z_{L, \tilde{M}}``,

```math
\tilde{T}_{L \tilde M}(R \hat{\boldsymbol e}_1, R \hat{\boldsymbol e}_2)
  = \sum_{\tilde M'} \Delta^{(L)}_{\tilde M \tilde M'}(R)\,
    \tilde{T}_{L \tilde M'}(\hat{\boldsymbol e}_1, \hat{\boldsymbol e}_2)
```

— are asserted:

```@example am
function coupled_eval(T, e1, e2)
    z1 = [Harmonics.Zlm(1, m, e1) for m = -1:1]
    z2 = [Harmonics.Zlm(1, m, e2) for m = -1:1]
    return [z1' * T[:, :, k] * z2 for k = 1:size(T, 3)]
end

function equivariance_deviation(real11; nrot = 20, seed = 11)
    rng = MersenneTwister(seed)
    dev = 0.0
    for _ = 1:nrot
        R = rand_rotation(rng)         # proper only: the L = 1 channel is axial
        e1, e2 = rand_unit(rng), rand_unit(rng)
        for (Lf, T) in real11
            lhs = coupled_eval(T, R * e1, R * e2)
            rhs = Dreal(Lf, R) * coupled_eval(T, e1, e2)
            dev = max(dev, maximum(abs.(lhs - rhs)))
        end
    end
    return dev
end

# δ/√3 and ε/√2 closed forms; μ = (-1, 0, +1) ↔ the (y, z, x) = xyz-axes (2, 3, 1)
ax = (2, 3, 1)
eps3 = zeros(3, 3, 3)
for (i, j, k) in ((1, 2, 3), (2, 3, 1), (3, 1, 2))
    eps3[i, j, k] = 1.0
    eps3[i, k, j] = -1.0
end
dot_dev = maximum(abs.(real11[0][:, :, 1] - Matrix(1.0I, 3, 3) / sqrt(3)))
cross_dev = maximum(abs(real11[1][b, c, a] - eps3[ax[a], ax[b], ax[c]] / sqrt(2))
                    for a = 1:3, b = 1:3, c = 1:3)
eq_dev = equivariance_deviation(real11)

max(dot_dev, cross_dev) <= 1e-12 || error("tesseral CG closed-form check failed")
eq_dev <= 1e-9 || error("tesseral CG equivariance check failed")
(dot_form = dot_dev, cross_form = cross_dev, rotation_equivariance = eq_dev)
```

## What this page does and does not gate

- Every number above is recomputed on each docs build; `warnonly = false` means a
  failed assertion breaks the build, so the published page can never show a stale
  green.
- The sweeps here mirror (and slightly extend — the completeness relation, ``j
  \le 3``) the unit tests in `test/unit/test_angmom.jl`; the test suite is still
  the primary, faster gate run on every change.
- Downstream consistency — that these pieces assemble into symmetry-invariant
  SALCs — is checked elsewhere: the invariance and idempotency gates of
  `test/unit/test_salc.jl` / `test_nbody.jl` and the pinned-oracle comparison in
  `test/oracle/`.
