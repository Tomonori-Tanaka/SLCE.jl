using SLCE
import Spglib   # activates the SpglibBackend extension for the executed `@example` blocks
using Documenter
using Documenter: Remotes

DocMeta.setdocmeta!(SLCE, :DocTestSetup, :(using SLCE);
                    recursive = true)

makedocs(;
    sitename = "SLCE.jl",
    modules = [SLCE],
    repo = Remotes.GitHub("Tomonori-Tanaka", "SLCE.jl"),
    format = Documenter.HTML(;
        prettyurls = get(ENV, "CI", "false") == "true",
        mathengine = Documenter.MathJax3(),
        canonical = "https://tomonori-tanaka.github.io/SLCE.jl/dev",
        edit_link = "main",
        footer = "Built with [Documenter.jl](https://documenter.juliadocs.org).",
        # api.md is one deliberate flat reference page; it crossed the default
        # 200 KiB HTML threshold with the joint/ASR docstrings.
        size_threshold = 400 * 2^10,
    ),
    pages = [
        "Home" => "index.md",
        "Getting started" => "getting_started.md",
        "Guide" => [
            "guide/basis.md",
            "guide/io.md",
            "guide/fitting.md",
            "guide/joint.md",
            "guide/introspection.md",
            "guide/lattice_dynamics.md",
            "guide/strain.md",
            "guide/sunny.md",
        ],
        "Tutorials" => [
            "tutorials/index.md",
            "tutorials/heisenberg_chain.md",
            "tutorials/kagome_threebody.md",
            "tutorials/case1_bcc_fe.md",
        ],
        "Theory" => [
            "theory/index.md",
            "theory/slce.md",
            "theory/resolvability.md",
            "theory/architecture.md",
        ],
        "Verification" => [
            "verification/angular_momentum.md",
        ],
        "API reference" => "api.md",
    ],
    warnonly = false,   # strict: any @example error / missing docstring fails the build
    # :public, not :exports — the unexported `public` surface (SolidHarmonics,
    # build_asr, sector_columns, salc_groups, …) is API too: it is what the
    # downstream packages and the staged-fit plans call, so a missing docstring
    # there has to fail the build like any other.
    checkdocs = :public,
    doctest = false,
)

# Publishes to https://tomonori-tanaka.github.io/SLCE.jl/ from the `documentation build`
# CI job (which needs `permissions: contents: write`). Outside CI this is a no-op, so a
# local `julia --project=docs docs/make.jl` still just builds into `docs/build/`.
deploydocs(;
    repo = "github.com/Tomonori-Tanaka/SLCE.jl",
    devbranch = "main",
    push_preview = false,
)
