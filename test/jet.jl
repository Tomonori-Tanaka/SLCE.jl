using SLCE
using JET

@testset "JET" begin
    JET.test_package(SLCE; target_modules = (SLCE,))
end
