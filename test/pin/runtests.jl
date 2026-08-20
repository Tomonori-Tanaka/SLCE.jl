# The pin driver.  Independent of `test/runtests.jl` on purpose: it needs a
# symmetry backend, and folding it into `TEST_MODE` would need an env var that
# also disables the core suite's thread gates.
#
#   julia --project=test/pin -t 4 test/pin/runtests.jl
#   julia --project=test/pin -t 1 test/pin/runtests.jl     # both, every time
#
# THESE ARE CHANGE DETECTORS, NOT CORRECTNESS EVIDENCE.  Every expected value in
# `pins/*.toml` was produced by this package; a green run says "the output has
# not moved since capture", nothing more.  Correctness lives in
# `test/unit/test_normalization.jl` (closed forms) and `test/oracle/`.
#
# When it goes red it prints WHICH LAYER moved and the recapture rule that
# applies.  That printout is the only enforcement this tier has -- there is no
# branch protection and no pin-only-commit job -- so read it before touching a
# `.toml`.  The layers and the rules are in PIN.md.
using Test, TOML, LinearAlgebra
include(joinpath(@__DIR__, "payload.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))

# Exact on the platform the pin was captured on; a tolerance elsewhere, because
# the chain runs through LAPACK (`eigen` in the SALC projector) and BLAS, whose
# kernels are architecture-specific.  Whether the values are in fact bitwise
# portable across platforms is UNMEASURED; until it is, this is the honest rule,
# and the capture platform is recorded in each pin's [meta].  Byte-stability
# across THREAD COUNTS is measured, and is why the runner is run at -t 4 and -t 1.
const HERE = string(Sys.ARCH, "-", Sys.KERNEL)
const L1_RTOL = 1e-12               # used only off the capture platform

const MOVED = Dict("L0" => String[], "L0prime" => String[], "L1" => String[],
                   "L2" => String[])
note!(layer, fid, what) = push!(MOVED[layer], string(fid, ": ", what))

# `@test` the comparison AND record which layer it belongs to, so the verdict at
# the end can name the layer without re-deriving it from the failure list.
macro pin(layer, fid, what, ex)
    quote
        ok = $(esc(ex))
        @test ok
        ok || note!($(esc(layer)), $(esc(fid)), $(esc(what)))
        ok
    end
end

function run_pins()
    @testset "pins" begin
        Threads.nthreads() == 1 &&
            @info "pin run is single-threaded; run it with -t 4 as well (the basis " *
                  "build is threaded, and byte-stability across thread counts is " *
                  "part of what these detect)"
        for fx in PIN_FIXTURES
            path = joinpath(@__DIR__, "pins", fx.id * ".toml")
            @testset "$(fx.id)" begin
                @test isfile(path)
                isfile(path) || continue
                want = TOML.parsefile(path)
                @test want["schema"] == PIN_SCHEMA
                got = pin_payload(fx)
                exact_here = get(get(want, "meta", Dict()), "platform", "") == HERE

                # -- the fixture itself.  A pin whose fixture drifted is comparing two
                #    different crystals and every later layer is meaningless, so this
                #    is a PRECONDITION, checked before anything numeric.
                @pin("L0", fx.id, "fixture definition", got["fixture"] == want["fixture"]) ||
                    continue

                # -- L0 structure: integers and labels only.  Exact on every platform.
                #    It is also the precondition for L0'/L1: those two are indexed by
                #    position, so comparing them across a changed structure compares
                #    unrelated numbers.
                l0 = true
                for k in ("n_salcs", "keys", "members", "terms")
                    l0 &= @pin("L0", fx.id, k, got["L0"][k] == want["L0"][k])
                end

                if l0
                    # -- L0' coarse mask: the sign/support pattern of `folded` at a
                    #    threshold with nine decades of headroom (PIN.md).  Exact
                    #    everywhere: it survives any rounding two platforms differ by,
                    #    so a red L0' is a structural change, not arithmetic noise.
                    @pin("L0prime", fx.id, "eps",
                         got["L0prime"]["eps"] == want["L0prime"]["eps"])
                    @pin("L0prime", fx.id, "support pattern",
                         got["L0prime"]["support"] == want["L0prime"]["support"])

                    # -- L1 values.
                    if @pin("L1", fx.id, "folded length",
                            length(got["L1"]["folded"]) == length(want["L1"]["folded"]))
                        if exact_here
                            @pin("L1", fx.id, "folded (bitwise, capture platform)",
                                 got["L1"]["folded"] == want["L1"]["folded"])
                        else
                            @pin("L1", fx.id, "folded (rtol $L1_RTOL, off-platform)",
                                 all(isapprox(unhexf(a), unhexf(b); rtol = L1_RTOL,
                                              atol = 1e-300)
                                     for (a, b) in zip(got["L1"]["folded"],
                                                       want["L1"]["folded"])))
                        end
                    end
                end

                # -- L2 physics.  Gauge-DEPENDENT by construction (`coef` is one
                #    representative of a non-unique solution when columns are frozen),
                #    so a legitimate gauge change trips this layer and only this layer.
                for k in ("design_frob2", "r2")
                    @pin("L2", fx.id, k, exact_here ? got["L2"][k] == want["L2"][k] :
                         isapprox(unhexf(got["L2"][k]), unhexf(want["L2"][k]); rtol = 1e-10))
                end
                for k in ("coef", "heldout")
                    @pin("L2", fx.id, "$k length",
                         length(got["L2"][k]) == length(want["L2"][k])) || continue
                    @pin("L2", fx.id, k, exact_here ? got["L2"][k] == want["L2"][k] :
                         all(isapprox(unhexf(a), unhexf(b); rtol = 1e-8, atol = 1e-12)
                             for (a, b) in zip(got["L2"][k], want["L2"][k])))
                end
            end
        end
    end

end

# --- the verdict.  Rule numbers refer to PIN.md "recapture rules". -----------
function print_verdict()
    any(!isempty, values(MOVED)) || return
    println("\n", "="^72)
    println("PIN VERDICT — which layer moved decides what you are allowed to do.")
    println("="^72)
    for (layer, rule) in (("L0", "rule 2 — treat as a BUG. Recapture forbidden."),
                          ("L0prime", "rule 2 — treat as a BUG. Recapture forbidden."),
                          ("L1", "rule 4 — recapture allowed, on a pin-only commit " *
                                 "whose body carries the moving change's hash and " *
                                 "the measurement that L0/L2 did not move. " *
                                 "\"to make the suite green\" is not a reason."),
                          ("L2", "rule 3 — a bug, OR a design change. Recapture only " *
                                 "with SPEC.md + CLAUDE.md coupling-site updates and " *
                                 "a re-measured real-data parity in the CHANGELOG."))
        isempty(MOVED[layer]) && continue
        println("\n", layer, " MOVED (", length(MOVED[layer]), "):")
        for m in MOVED[layer]
            println("    ", m)
        end
        println("  → ", rule)
    end
    println("\nrule 5 — recapture happens only on an explicit human instruction.")
    println("rule 7 — before recapturing, confirm test/unit/test_normalization.jl " *
            "is green:\n         those closed forms are geometry, not a pin, and " *
            "they do not move.")
    println("="^72)
    return
end

# `finally`, so the verdict is printed even when the testset throws at the end —
# which it always does when a pin moves, and which would otherwise swallow the
# one thing this tier exists to say.
try
    run_pins()
finally
    print_verdict()
end
