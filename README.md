ID: 124102872
Name: Kamthe Siddhesh Sanjay

=============================
Project title: Discrete Event Simulation in Julia
=============================
Current and Final Implementation in DESLibrary folder.

# Start repl Normal

julia --project

# With sysimage

julia --project -J DESLibrary_compiled.so

# To run scripts in Repl use below commands

using Pkg; Pkg.instantiate()
include("examples/CSMACD/csmacd_enhanced.jl")

# Commands to run on terminal

cd "DESProjectPkg\DESLibrary"
julia examples\Batch_Simulations\mmc_batch_simulation.jl
julia examples\Batch_Simulations\run_batch_analysis.jl
julia examples\Batch_Simulations\run_csmacd_analysis.jl
julia examples\CSMACD\csmacd_example.jl
julia examples\CSMACD\csmacd_enhanced.jl
julia examples\CSMACD\csmacd_simjulia.jl
julia examples\MMC\mmc_deslibrary.jl
julia examples\MMC\mmc_simjulia.jl

cd "DESProjectPkg\DESLibrary"
julia --sysimage DESLibrary_compiled.so examples\Batch_Simulations\mmc_batch_simulation.jl
julia --sysimage DESLibrary_compiled.so examples\Batch_Simulations\run_batch_analysis.jl
julia --sysimage DESLibrary_compiled.so examples\Batch_Simulations\run_csmacd_analysis.jl
julia --sysimage DESLibrary_compiled.so examples\CSMACD\csmacd_example.jl
julia --sysimage DESLibrary_compiled.so examples\CSMACD\csmacd_simjulia.jl
julia --sysimage DESLibrary_compiled.so examples\MMC\mmc_deslibrary.jl
julia --sysimage DESLibrary_compiled.so examples\MMC\mmc_simjulia.jl

# Sysimage too large to upload build using this command

cd "DESProjectPkg\DESLibrary"
julia build_sysimage.jl
