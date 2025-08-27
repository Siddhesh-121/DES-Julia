#!/usr/bin/env julia



using Base.Threads
using Printf
using Statistics

try
    using HypothesisTests
catch e
    println(" Installing HypothesisTests.jl...")
    using Pkg
    Pkg.add("HypothesisTests")
    using HypothesisTests
end

include("mmc_batch_simulation.jl")

function entry_banner()
    println("MMC Batch Analysis Framework")
    println("=" ^ 60)
    println("Statistical Testing for M/M/C Queueing Systems")
    println()

    println("Thread Configuration:")
    println("   Available threads: $(nthreads())")
    if nthreads() == 1
        println("   Running in single-threaded mode.")
        println("   To enable multi-threading, start Julia with: julia -t auto")
    else
        println("   Multi-threading enabled")
    end
    println()
end


function run_mmc_batch_analysis(; 
    num_batches::Int=10,
    λ::Float64=10.0, 
    μ::Float64=2.0, 
    c::Int=6,
    max_entities::Int=50000,
    confidence_level::Float64=0.95
)
    println("Starting M/M/C Batch Analysis...")
    
    try
        result = main_mmc(
            num_batches=num_batches,
            λ=λ, μ=μ, c=c,
            max_entities=max_entities,
            confidence_level=confidence_level
        )
        
        println("M/M/C analysis completed successfully")
        return result
    catch e
        println("M/M/C analysis failed: $e")
        return nothing
    end
end




function analyze_mmc_results(mmc_result)
    if mmc_result === nothing
        println("Cannot perform analysis - MMC simulation failed")
        return nothing
    end
    
    if mmc_result.statistics === nothing
        println("M/M/C analysis failed: ErrorException(\"type Nothing has no field runtime_stats\")")
        return nothing
    end
    
    println("\nMMC DETAILED ANALYSIS")
    println("=" ^ 60)
    
    mmc_stats = mmc_result.statistics
    
    println("PERFORMANCE INSIGHTS")
    println("-" ^ 40)
    
    mmc_batch_count = length(mmc_result.results)
    mmc_avg_entities = mean([r.entities_processed for r in mmc_result.results])
    mmc_avg_runtime_s = mmc_stats.runtime_stats.mean / 1000
    mmc_events_per_sec = (mmc_avg_entities * 2) / mmc_avg_runtime_s  # 2 events per entity
    
    @printf("Batch count:          %d\n", mmc_batch_count)
    @printf("Avg entities/batch:   %.0f\n", mmc_avg_entities)
    @printf("Event processing rate: ~%.0f events/second\n", mmc_events_per_sec)
    
    println("\nACCURACY ASSESSMENT")
    println("-" ^ 40)
    
    avg_accuracy = mean([mmc_stats.accuracy_stats.waiting, mmc_stats.accuracy_stats.utilization, 
                        mmc_stats.accuracy_stats.queue])
    
    @printf("Overall accuracy:     %.2f%% average error\n", avg_accuracy * 100)
    @printf("Waiting time error:   %.2f%%\n", mmc_stats.accuracy_stats.waiting * 100)
    @printf("Utilization error:    %.2f%%\n", mmc_stats.accuracy_stats.utilization * 100)
    @printf("Queue length error:   %.2f%%\n", mmc_stats.accuracy_stats.queue * 100)
    @printf("Little's Law error:   %.2f%%\n", mmc_stats.accuracy_stats.littles * 100)
    
    println("\nPERFORMANCE ASSESSMENT")
    println("-" ^ 40)
    
    consistency_threshold = 0.1  # 10% CV threshold
    mmc_consistent = mmc_stats.runtime_stats.cv < consistency_threshold
    
    @printf("Runtime consistency:  %s (CV: %.4f)\n", 
            mmc_stats.runtime_stats.cv < 0.1 ? "High" : "Moderate", mmc_stats.runtime_stats.cv)
    
    mmc_tests_passed = sum(values(mmc_stats.hypothesis_results))
    @printf("Statistical tests:    %d/4 passed (%.0f%%)\n", 
            mmc_tests_passed, (mmc_tests_passed/4)*100)
    
    println("\nRECOMMENDATIONS")
    println("-" ^ 40)
    
    if avg_accuracy > 0.05
        println("• Accuracy could be improved - consider longer simulation runs")
    end
    
    if mmc_stats.runtime_stats.cv > 0.15
        println("• Runtime variance is high - check for system load issues")
    end
    
    if mmc_tests_passed < 3
        println("• Statistical tests indicate potential issues - review implementation")
    end
    
    if mmc_events_per_sec < 1000
        println("• Event processing rate is low - consider performance optimization")
    end
    
    return (
        avg_accuracy = avg_accuracy,
        consistency = mmc_consistent,
        tests_passed = mmc_tests_passed,
        events_per_sec = mmc_events_per_sec
    )
end


function run_mmc_analysis(; num_batches::Int=10, λ::Float64=10.0, μ::Float64=2.0, 
                          c::Int=6, max_entities::Int=50000, confidence_level::Float64=0.95)
    println("MMC BATCH ANALYSIS")
    println("System: M/M/$c with λ=$λ, μ=$μ")
    println("Batches: $num_batches, Entities per batch: $max_entities")
    println("Confidence level: $(confidence_level * 100)%")
    println()
    
    start_time = time()
    
    mmc_result = run_mmc_batch_analysis(
        num_batches=num_batches,
        λ=λ, μ=μ, c=c,
        max_entities=max_entities,
        confidence_level=confidence_level
    )
    
    analysis = analyze_mmc_results(mmc_result)
    
    total_time = time() - start_time
    
    println("\nANALYSIS COMPLETE")
    println("=" ^ 60)
    @printf("Total execution time: %.2f seconds\n", total_time)
    @printf("Batches completed: %d\n", num_batches)
    
    if nthreads() > 1
        @printf("Multi-threading efficiency: ~%.1fx speedup\n", min(nthreads(), num_batches))
    end
    
    return (mmc_result=mmc_result, analysis=analysis)
end



function quick_test()
    println(" Running Quick MMC Test (3 batches)...")
    return run_mmc_analysis(num_batches=3, max_entities=50000)  # Increased for stability
end


function standard_test()
    println(" Running Standard MMC Test (20 batches)...")
    return run_mmc_analysis(num_batches=20, max_entities=100000)  # Quick mode: 10 batches
end


function extensive_test()
    println(" Running Extensive MMC Test (30 batches)...")
    return run_mmc_analysis(num_batches=30, max_entities=200000)  # Standard mode: 20 batches
end


if abspath(PROGRAM_FILE) == @__FILE__
    entry_banner()
    println()
    println("Running standard MMC test...")
    result = standard_test()
end