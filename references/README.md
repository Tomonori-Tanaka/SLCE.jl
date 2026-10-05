# References

Supporting literature for this package — papers and useful web resources.

## Layout

- `*.md` (this directory) — **one note per source**: citation, link, a short
  summary, and why it matters here. **Tracked in git** (own notes, no copyright
  issue).
- `papers/` — the actual paper **PDFs**. **Gitignored**; local-only.

## Index

### Papers
- [`drautz-2020-tesseral.md`](drautz-2020-tesseral.md) — Drautz 2020, "Atomic
  cluster expansion of scalar, vector and tensor properties…" — the real
  (tesseral) spherical-harmonic `Zₗₘ` normalization/sign convention used here.
- [`weissenhofer-2023-rotational-slc.md`](weissenhofer-2023-rotational-slc.md) — Weißenhofer
  et al. 2023, PRB 108, L060404 — rotationally invariant spin–lattice coupling
  (local-frame construction; `B^as ω` rotation-tensor term; SM `B₁/B₂` sums =
  our affine path). Resolves the paper's `todoRotationalME` together with Melcher.
- Drautz & Fähnle 2004, *Phys. Rev. B* **69**, 104404 — the spin-cluster
  expansion energy model `E = j0 + Σ Jφ Φφ`.
- Tanaka & Gohda 2026, *Phys. Rev. Research* **8**, 023300 — SCE from
  noncollinear spin DFT (the method this package fits).

## Adding a reference

1. Copy an existing note to `<short-key>.md` and fill it in.
2. Put the PDF at `papers/<short-key>.pdf` (stays local, gitignored).
3. Add a line to the index above.
