#!/usr/bin/env julia

"""
Build script for creating optimized DESLibrary system image using PackageCompiler.jl
"""

using PackageCompiler
using Pkg

println("Building DESLibrary System Image")
println("=" ^ 60)

# Ensure we have all required packages
required_packages = [
    "DataStructures",
    "Random", 
    "Distributions",
    "Statistics",
    "SpecialFunctions",
    "HypothesisTests",
    "Printf"
]

println("Checking required packages...")
for pkg in required_packages
    try
        eval(:(using $(Symbol(pkg))))
        println("  OK $pkg")
    catch e
        println("  Installing $pkg...")
        Pkg.add(pkg)
        println("  OK $pkg installed")
    end
end

# Build configuration
const SYSIMAGE_PATH = "DESLibrary_compiled.so"
const PRECOMPILE_SCRIPT = "precompile/precompile_workload.jl"

println("\nBuild Configuration:")
println("  Output path: $SYSIMAGE_PATH")
println("  Precompile script: $PRECOMPILE_SCRIPT")
println("  Julia version: $(VERSION)")
println("  Available threads: $(Threads.nthreads())")

# Check if precompile script exists
if !isfile(PRECOMPILE_SCRIPT)
    error("Precompile script not found: $PRECOMPILE_SCRIPT")
end

println("\nStarting system image compilation...")
println("This may take 5-15 minutes depending on your system...")

start_time = time()

try
    # Create system image with precompilation
    create_sysimage(
        [:DataStructures, :Distributions, :SpecialFunctions, :HypothesisTests];
        sysimage_path = SYSIMAGE_PATH,
        precompile_execution_file = PRECOMPILE_SCRIPT,
        cpu_target = "generic",
        include_transitive_dependencies = true,
        incremental = false
    )
    
    
    
    compile_time = time() - start_time
    
    println("\nSystem image compilation successful!")
    println("Compilation Statistics:")
    println("  Compilation time: $(round(compile_time, digits=1)) seconds")
    println("  System image size: $(round(stat(SYSIMAGE_PATH).size / 1024^2, digits=1)) MB")
    println("  Output location: $(abspath(SYSIMAGE_PATH))")
    
    println("\nUsage Instructions:")
    println("  To use the compiled system image:")
    println("  julia --sysimage $(SYSIMAGE_PATH) your_script.jl")
    println()
    println("  For batch simulations:")
    println("  julia --sysimage $(SYSIMAGE_PATH) -t 4 run_batch_analysis.jl")
    println("  julia --sysimage $(SYSIMAGE_PATH) -t 4 run_csmacd_analysis.jl")
    
catch e
    println("\nSystem image compilation failed!")
    println("Error: $e")
    
    println("\nTroubleshooting:")
    println("  1. Ensure all dependencies are installed")
    println("  2. Check that precompile script runs successfully:")
    println("     julia precompile/precompile_workload.jl")
    println("  3. Try with fewer optimizations:")
    println("     Set cpu_target=\"generic\" (already set)")
    println("  4. Check available disk space (>2GB recommended)")
    
    exit(1)
end