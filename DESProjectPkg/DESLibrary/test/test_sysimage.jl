println("=== Sysimage Test ===")
@time using Statistics, HypothesisTests
@time include("../src/DESLibrary.jl")
@time using .DESLibrary
println("Sysimage test complete")