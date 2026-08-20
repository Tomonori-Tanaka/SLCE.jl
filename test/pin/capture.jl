# Regenerate the pins.  MANUAL: this is never run by the suite -- a pin that
# rewrites itself detects nothing.  See PIN.md for when recapture is allowed.
#
#   julia --project=test/pin -t 4 test/pin/capture.jl
include(joinpath(@__DIR__, "payload.jl"))
include(joinpath(@__DIR__, "fixtures.jl"))
mkpath(joinpath(@__DIR__, "pins"))
for fx in PIN_FIXTURES
    d = pin_payload(fx)
    d["meta"] = Dict{String,Any}(
        # Both are passed in, not sniffed: recapturing is a deliberate act, and
        # the entry in PIN.md that explains WHY is written at the same moment.
        "captured" => get(ENV, "PIN_DATE", "SET-ME"),
        "julia" => string(VERSION),
        "platform" => string(Sys.ARCH, "-", Sys.KERNEL),
        "threads" => Threads.nthreads(),
        "commit" => get(ENV, "PIN_COMMIT", "SET-ME"))
    open(joinpath(@__DIR__, "pins", fx.id * ".toml"), "w") do io
        TOML.print(io, d; sorted = true)
    end
    println("captured ", fx.id, "  n_salcs = ", d["L0"]["n_salcs"],
            "  folded = ", length(d["L1"]["folded"]))
end
