println("=== Normal Julia Test ===")
@time using Statistics, HypothesisTests
@time include("../src/DESLibrary.jl")
@time using .DESLibrary
println("Normal Julia test complete")