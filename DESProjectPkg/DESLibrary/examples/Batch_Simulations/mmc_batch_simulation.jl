#!/usr/bin/env julia



using Base.Threads
using Statistics
using Printf
using Random
using HypothesisTests
using Distributions

if !isdefined(Main, :DESLibrary)
    include("../../src/DESLibrary.jl")
    using .DESLibrary
else
    using .DESLibrary
end


"""
Configuration structure for batch simulation parameters
"""
mutable struct BatchConfig
    λ::Float64                    # Arrival rate
    μ::Float64                    # Service rate  
    c::Int                        # Number of servers
    max_entities::Int             # Entities per simulation
    warmup_fraction::Float64      # Warmup period
    
    num_batches::Int              # Number of simulation batches
    num_threads::Int              # Number of threads to use
    confidence_level::Float64     # Statistical confidence level
    
    base_seed::Int                # Base random seed
    
    function BatchConfig(;
        λ=10.0, μ=2.0, c=6, max_entities=50000, warmup_fraction=0.1,
        num_batches=10, num_threads=nthreads(), confidence_level=0.95,
        base_seed=2024
    )
        new(λ, μ, c, max_entities, warmup_fraction, num_batches, 
            num_threads, confidence_level, base_seed)
    end
end


struct BatchResult
    batch_id::Int
    thread_id::Int
    
    runtime_ms::Float64           # Wall-clock time in milliseconds
    entities_processed::Int       # Number of entities processed
    throughput::Float64           # Entities per second
    
    avg_waiting_time::Float64
    avg_service_time::Float64
    avg_time_in_system::Float64
    server_utilization::Float64
    avg_queue_length::Float64
    max_queue_length::Int
    
    theoretical_waiting_time::Float64
    theoretical_utilization::Float64
    theoretical_queue_length::Float64
    
    waiting_time_error::Float64   # Relative error vs theoretical
    utilization_error::Float64    # Relative error vs theoretical
    queue_length_error::Float64   # Relative error vs theoretical
    
    littles_law_valid::Bool
    littles_law_error::Float64
end



function calculate_mmc_theoretical(λ::Float64, μ::Float64, c::Int)
    ρ = λ / μ                    # Traffic intensity
    ρ_per_server = λ / (c * μ)   # Per-server utilization
    
    if ρ_per_server >= 1.0
        return (
            stable = false,
            waiting_time = Inf,
            utilization = 1.0,
            queue_length = Inf,
            service_time = 1.0 / μ
        )
    end
    
    sum_term = sum((ρ^k) / factorial(k) for k in 0:(c-1))
    erlang_c_numerator = (ρ^c) / factorial(c)
    erlang_c_denominator = sum_term + erlang_c_numerator / (1 - ρ_per_server)
    erlang_c = erlang_c_numerator / (erlang_c_denominator * (1 - ρ_per_server))
    
    waiting_time = erlang_c / (c * μ * (1 - ρ_per_server))
    service_time = 1.0 / μ
    time_in_system = waiting_time + service_time
    queue_length = erlang_c * ρ_per_server / (1 - ρ_per_server)
    utilization = ρ_per_server
    
    return (
        stable = true,
        waiting_time = waiting_time,
        utilization = utilization,
        queue_length = queue_length,
        service_time = service_time,
        time_in_system = time_in_system,
        erlang_c = erlang_c
    )
end



function run_single_simulation(config::BatchConfig, batch_id::Int)
    thread_id = threadid()
    
    simulation_seed = config.base_seed + batch_id * 1000
    
    theoretical = calculate_mmc_theoretical(config.λ, config.μ, config.c)
    
    if !theoretical.stable
        error("System is unstable (ρ/c ≥ 1.0). Cannot run simulation.")
    end
    
    start_time = time()
    
    engine = DESEngine(config.max_entities, simulation_seed, config.warmup_fraction)
    model_id = Symbol("mmc_batch_$batch_id")
    model = MMCModel(config.λ, config.μ, config.c, model_id)
    add_model!(engine, model_id, model)
    
    simulate!(engine, verbose=false)
    
    end_time = time()
    runtime_ms = (end_time - start_time) * 1000
    
    results = get_results(engine)[model_id]
    
    throughput = results.entities_processed / (runtime_ms / 1000)
    
    waiting_time_error = abs(results.avg_waiting_time - theoretical.waiting_time) / theoretical.waiting_time
    utilization_error = abs(results.server_utilization - theoretical.utilization) / theoretical.utilization
    queue_length_error = abs(results.avg_queue_length - theoretical.queue_length) / theoretical.queue_length
    
    arrival_rate = results.entities_processed / results.simulation_time
    littles_law_L = results.avg_queue_length
    littles_law_λW = arrival_rate * results.avg_waiting_time
    littles_law_error = abs(littles_law_L - littles_law_λW) / max(littles_law_L, littles_law_λW, 1e-10)
    littles_law_valid = littles_law_error < 0.05  # 5% tolerance
    
    return BatchResult(
        batch_id, thread_id, runtime_ms, results.entities_processed, throughput,
        results.avg_waiting_time, results.avg_service_time, results.avg_time_in_system,
        results.server_utilization, results.avg_queue_length, results.max_queue_length,
        theoretical.waiting_time, theoretical.utilization, theoretical.queue_length,
        waiting_time_error, utilization_error, queue_length_error,
        littles_law_valid, littles_law_error
    )
end



function run_batch_simulations(config::BatchConfig)
    println("🚀 Starting MMC Batch Simulations")
    println("=" ^ 60)
    println("Configuration:")
    println("  M/M/$(config.c) system: λ=$(config.λ), μ=$(config.μ)")
    println("  Entities per simulation: $(config.max_entities)")
    println("  Number of batches: $(config.num_batches)")
    println("  Available threads: $(nthreads())")
    println("  Using threads: $(config.num_threads)")
    println("  Confidence level: $(config.confidence_level * 100)%")
    
    ρ_per_server = config.λ / (config.c * config.μ)
    println("  Traffic intensity per server: $(round(ρ_per_server, digits=3))")
    println("  System status: $(ρ_per_server < 1.0 ? "✅ Stable" : "❌ Unstable")")
    
    if ρ_per_server >= 1.0
        error("System is unstable. Please adjust parameters.")
    end
    
    println("\n🔥 Per-thread warmup (eliminating cold start JIT compilation)...")
    
    warmup_tasks = Vector{Task}(undef, min(config.num_threads, nthreads()))
    warmup_config = BatchConfig(
        λ=config.λ, μ=config.μ, c=config.c,
        max_entities=1000,  # Small warmup
        num_batches=1,
        num_threads=1,      # Single batch per warmup
        base_seed=config.base_seed + 9999
    )
    
    if config.num_threads > 1
        for i in 1:min(config.num_threads, nthreads())
            warmup_tasks[i] = @spawn begin
                run_single_simulation(warmup_config, i)
                println("Thread $(threadid()) warmed up ✓")
            end
        end
        
        for task in warmup_tasks
            wait(task)
        end
        println("✅ All $(min(config.num_threads, nthreads())) threads warmed up")
    end
    
    println("\n⏱️  Executing batch simulations...")
    
    threads_used = Set{Int}()
    thread_lock = ReentrantLock()
    
    batch_start_time = time()
    
    results = Vector{BatchResult}(undef, config.num_batches)
    
    if config.num_threads == 1
        for i in 1:config.num_batches
            lock(thread_lock) do
                push!(threads_used, threadid())
            end
            print(".")  # Progress indicator
            results[i] = run_single_simulation(config, i)
        end
    elseif config.num_threads >= nthreads()
        @threads for i in 1:config.num_batches
            lock(thread_lock) do
                push!(threads_used, threadid())
            end
            print(".")  # Progress indicator
            results[i] = run_single_simulation(config, i)
        end
    else
        semaphore = Base.Semaphore(config.num_threads)
        tasks = Vector{Task}(undef, config.num_batches)
        
        for i in 1:config.num_batches
            tasks[i] = @spawn begin
                Base.acquire(semaphore)
                try
                    lock(thread_lock) do
                        push!(threads_used, threadid())
                    end
                    print(".")  # Progress indicator
                    result = run_single_simulation(config, i)
                    results[i] = result
                finally
                    Base.release(semaphore)
                end
            end
        end
        
        for task in tasks
            wait(task)
        end
    end
    
    batch_end_time = time()
    total_batch_time = batch_end_time - batch_start_time
    
    println("\n✅ Completed $(config.num_batches) batches in $(round(total_batch_time, digits=2))s")
    
    actual_threads_used = length(threads_used)
    max_concurrent = min(config.num_threads, config.num_batches, nthreads())
    max_unique_possible = min(config.num_batches, nthreads())
    
    println("📊 Thread usage analysis:")
    println("   Configured max concurrent: $(config.num_threads)")
    println("   Effective max concurrent: $(max_concurrent)")
    println("   Unique threads used: $(actual_threads_used)")
    println("   Thread IDs used: $(sort(collect(threads_used)))")
    
    if config.num_threads == 1 && actual_threads_used == 1
        println("✅ Single-threaded execution verified")
    elseif config.num_threads >= nthreads() && actual_threads_used <= nthreads()
        println("✅ Multi-threaded execution using available threads")
    elseif actual_threads_used <= max_unique_possible
        println("✅ Thread usage within expected bounds")
        if actual_threads_used > config.num_threads
            println("   Note: Julia scheduler may use more unique threads than max concurrent")
        end
    else
        println("⚠️  Unexpected thread usage pattern")
    end
    
    return results
end



function perform_statistical_analysis(results::Vector{BatchResult}, config::BatchConfig)
    println("\n📊 STATISTICAL ANALYSIS")
    println("=" ^ 60)
    
    runtimes = [r.runtime_ms for r in results]
    throughputs = [r.throughput for r in results]
    waiting_times = [r.avg_waiting_time for r in results]
    utilizations = [r.server_utilization for r in results]
    queue_lengths = [r.avg_queue_length for r in results]
    
    waiting_errors = [r.waiting_time_error for r in results]
    utilization_errors = [r.utilization_error for r in results]
    queue_errors = [r.queue_length_error for r in results]
    littles_errors = [r.littles_law_error for r in results]
    
    println("\n🕐 RUNTIME PERFORMANCE")
    println("-" ^ 40)
    @printf("Mean runtime: %.2f ± %.2f ms\n", mean(runtimes), std(runtimes))
    @printf("Median runtime: %.2f ms\n", median(runtimes))
    @printf("Min/Max runtime: %.2f / %.2f ms\n", minimum(runtimes), maximum(runtimes))
    @printf("CV (coefficient of variation): %.4f\n", std(runtimes) / mean(runtimes))
    
    println("\n📊 INDIVIDUAL BATCH RUNTIMES")
    println("-" ^ 50)
    println("Batch | Runtime (ms) | Thread ID | Entities | Deviation")
    println("-" ^ 55)
    
    mean_runtime = mean(runtimes)
    for (i, result) in enumerate(results)
        deviation_pct = ((result.runtime_ms - mean_runtime) / mean_runtime) * 100
        deviation_symbol = if abs(deviation_pct) < 10
            "  ≈"
        elseif deviation_pct > 0
            "  ↑"
        else
            "  ↓"
        end
        
        @printf("%5d | %11.2f | %9d | %8d | %6.1f%% %s\n", 
                i, result.runtime_ms, result.thread_id, 
                result.entities_processed, deviation_pct, deviation_symbol)
    end
    
    println("\n📈 RUNTIME DISTRIBUTION ANALYSIS")
    println("-" ^ 40)
    
    fast_threshold = mean_runtime - std(runtimes)
    slow_threshold = mean_runtime + std(runtimes)
    
    fast_runs = [r for r in runtimes if r < fast_threshold]
    normal_runs = [r for r in runtimes if fast_threshold <= r <= slow_threshold]  
    slow_runs = [r for r in runtimes if r > slow_threshold]
    
    @printf("Fast runs (< %.0f ms): %d batches (%.1f%%)\n", 
            fast_threshold, length(fast_runs), (length(fast_runs)/length(runtimes))*100)
    @printf("Normal runs (%.0f-%.0f ms): %d batches (%.1f%%)\n", 
            fast_threshold, slow_threshold, length(normal_runs), (length(normal_runs)/length(runtimes))*100)
    @printf("Slow runs (> %.0f ms): %d batches (%.1f%%)\n", 
            slow_threshold, length(slow_runs), (length(slow_runs)/length(runtimes))*100)
    
    if length(fast_runs) > 0 && length(slow_runs) > 0
        println("\n🔍 VARIANCE PATTERN ANALYSIS")
        println("-" ^ 30)
        
        thread_runtimes = Dict{Int, Vector{Float64}}()
        for result in results
            if !haskey(thread_runtimes, result.thread_id)
                thread_runtimes[result.thread_id] = Float64[]
            end
            push!(thread_runtimes[result.thread_id], result.runtime_ms)
        end
        
        println("Runtime by thread:")
        for (thread_id, times) in sort(collect(thread_runtimes))
            @printf("Thread %d: %.1f ± %.1f ms (CV: %.3f)\n", 
                    thread_id, mean(times), std(times), std(times)/mean(times))
        end
        
        println("\nTrend analysis:")
        first_half = runtimes[1:div(length(runtimes), 2)]
        second_half = runtimes[div(length(runtimes), 2)+1:end]
        
        if length(first_half) > 0 && length(second_half) > 0
            @printf("First half mean: %.1f ms\n", mean(first_half))
            @printf("Second half mean: %.1f ms\n", mean(second_half))
            trend = mean(second_half) - mean(first_half)
            @printf("Trend: %+.1f ms (%s)\n", trend, 
                    abs(trend) < 20 ? "stable" : trend > 0 ? "slowing down" : "speeding up")
        end
        
        thread_first_runs = Dict{Int, Float64}()
        thread_later_runs = Dict{Int, Vector{Float64}}()
        
        for result in results
            if !haskey(thread_first_runs, result.thread_id)
                thread_first_runs[result.thread_id] = result.runtime_ms
                thread_later_runs[result.thread_id] = Float64[]
            else
                push!(thread_later_runs[result.thread_id], result.runtime_ms)
            end
        end
        
        first_run_times = collect(values(thread_first_runs))
        later_run_times = Float64[]
        for runs in values(thread_later_runs)
            append!(later_run_times, runs)
        end
        
        if length(first_run_times) > 1 && length(later_run_times) > 1
            mean_first = mean(first_run_times)
            mean_later = mean(later_run_times)
            
            if mean_first > mean_later * 1.5  # First runs 50% slower
                println("\n🎯 JIT COMPILATION PATTERN DETECTED!")
                println("-" ^ 40)
                @printf("First runs per thread: %.1f ms average\n", mean_first)
                @printf("Subsequent runs: %.1f ms average\n", mean_later)
                @printf("Cold start penalty: %.1f ms (%.1f%% slower)\n", 
                        mean_first - mean_later, ((mean_first / mean_later - 1) * 100))
                println("\n💡 SOLUTION: Per-thread warmup implemented!")
                println("   Each thread now compiles code before timing begins.")
                return  # Skip generic variance analysis since we found the specific cause
            end
        end
        
        if maximum(runtimes) > 2 * minimum(runtimes)
            println("\n⚠️ POTENTIAL CAUSES OF HIGH VARIANCE:")
            println("• Memory pressure or garbage collection")
            println("• Thread scheduling variations")
            println("• System load fluctuations")
            println("• Cache effects (first runs vs. later runs)")
            println("💡 Consider: longer warmup, larger heap, process priority")
        end
    end
    
    println("\n⚡ THROUGHPUT ANALYSIS")
    println("-" ^ 40)
    @printf("Mean throughput: %.0f ± %.0f entities/sec\n", mean(throughputs), std(throughputs))
    @printf("Throughput CV: %.4f\n", std(throughputs) / mean(throughputs))
    
    theoretical = calculate_mmc_theoretical(config.λ, config.μ, config.c)
    
    println("\n🎯 ACCURACY ANALYSIS")
    println("-" ^ 40)
    @printf("Waiting time error: %.4f ± %.4f (%.2f%%)\n", 
            mean(waiting_errors), std(waiting_errors), mean(waiting_errors) * 100)
    @printf("Utilization error: %.4f ± %.4f (%.2f%%)\n", 
            mean(utilization_errors), std(utilization_errors), mean(utilization_errors) * 100)
    @printf("Queue length error: %.4f ± %.4f (%.2f%%)\n", 
            mean(queue_errors), std(queue_errors), mean(queue_errors) * 100)
    @printf("Little's Law error: %.4f ± %.4f (%.2f%%)\n", 
            mean(littles_errors), std(littles_errors), mean(littles_errors) * 100)
    
    littles_valid_count = sum(r.littles_law_valid for r in results)
    @printf("Little's Law validation success rate: %.1f%% (%d/%d)\n", 
            (littles_valid_count / length(results)) * 100, littles_valid_count, length(results))
    
    println("\n🧪 HYPOTHESIS TESTING")
    println("-" ^ 40)
    
    cv_threshold = 0.2  # 20% coefficient of variation (achievable with warmup)
    runtime_cv = length(runtimes) > 1 ? std(runtimes) / mean(runtimes) : 0.0
    println("H₁: Runtime consistency (CV < $(cv_threshold))")
    if length(runtimes) > 1
        @printf("   Result: CV = %.4f %s\n", runtime_cv, 
                runtime_cv < cv_threshold ? "✅ PASS" : "❌ FAIL")
        if runtime_cv > 0.4
            println("   ⚠️  High variability may indicate per-thread JIT compilation")
            println("       Per-thread warmup should resolve this issue")
        elseif runtime_cv > cv_threshold
            println("   ⚠️  Moderate variability - consider longer simulations")
        end
    else
        @printf("   Result: CV = N/A (single sample) ❌ FAIL\n")
    end
    
    accuracy_threshold = 0.15  # 15% error tolerance (realistic for batch simulations)
    max_error = maximum([maximum(waiting_errors), maximum(utilization_errors), maximum(queue_errors)])
    println("H₂: Accuracy within tolerance (max error < $(accuracy_threshold))")
    @printf("   Result: Max error = %.4f %s\n", max_error, 
            max_error < accuracy_threshold ? "✅ PASS" : "❌ FAIL")
    if max_error > 0.1
        println("   💡 Consider increasing simulation size for better accuracy")
    end
    
    α = 1 - config.confidence_level
    println("H₃: Waiting time matches theoretical value")
    waiting_time_test_passed = false
    
    if length(waiting_times) >= 2 && std(waiting_times) > 1e-10
        try
            waiting_time_test = OneSampleTTest(waiting_times, theoretical.waiting_time)
            test_pvalue = pvalue(waiting_time_test)
            waiting_time_test_passed = test_pvalue > α
            @printf("   t-statistic: %.4f, p-value: %.6f\n", waiting_time_test.t, test_pvalue)
            @printf("   Result: %s (α = %.3f)\n", 
                    waiting_time_test_passed ? "✅ ACCEPT H₀" : "❌ REJECT H₀", α)
        catch e
            println("   ⚠️ t-test failed: $e")
            @printf("   Result: ❌ FAIL (test error)\n")
            waiting_time_test_passed = false
        end
    else
        println("   ⚠️ Insufficient data for t-test (need ≥2 samples with non-zero variance)")
        @printf("   Result: ❌ FAIL (insufficient data)\n")
        waiting_time_test_passed = false
    end
    
    if length(runtimes) >= 8  # Minimum samples for normality test
        println("H₄: Runtime follows normal distribution")
        try
            runtime_normality = ExactOneSampleKSTest(runtimes, Normal(mean(runtimes), std(runtimes)))
            @printf("   KS test p-value: %.6f\n", pvalue(runtime_normality))
            @printf("   Result: %s\n", 
                    pvalue(runtime_normality) > α ? "✅ Normal distribution" : "⚠️ Non-normal distribution")
        catch e
            println("   ⚠️ Normality test failed: $e")
        end
    end
    
    littles_success_rate = littles_valid_count / length(results)
    success_threshold = 0.95  # 95% success rate expected
    println("H₅: Little's Law validation success rate ≥ $(success_threshold)")
    @printf("   Result: %.2f %s\n", littles_success_rate, 
            littles_success_rate >= success_threshold ? "✅ PASS" : "❌ FAIL")
    
    return (
        runtime_stats = (mean=mean(runtimes), std=std(runtimes), cv=runtime_cv),
        accuracy_stats = (waiting=mean(waiting_errors), utilization=mean(utilization_errors), 
                         queue=mean(queue_errors), littles=mean(littles_errors)),
        hypothesis_results = (
            runtime_consistency = runtime_cv < cv_threshold,
            accuracy_tolerance = max_error < accuracy_threshold,
            waiting_time_match = waiting_time_test_passed,
            littles_success = littles_success_rate >= success_threshold
        )
    )
end



function main_mmc(; num_batches::Int=10, λ::Float64=10.0, μ::Float64=2.0, c::Int=6, 
                  max_entities::Int=50000, confidence_level::Float64=0.95)
    
    println("🏗️  MMC BATCH SIMULATION FRAMEWORK")
    println("=" ^ 60)
    println("Multi-threaded Statistical Analysis for M/M/C Queueing Systems")
    println("Threads available: $(nthreads())")
    println()
    
    config = BatchConfig(
        λ=λ, μ=μ, c=c, max_entities=max_entities,
        num_batches=num_batches, confidence_level=confidence_level
    )
    
    results = run_batch_simulations(config)
    
    stats = perform_statistical_analysis(results, config)
    
    println("\n📈 SUMMARY REPORT")
    println("=" ^ 60)
    @printf("✅ Completed %d batches successfully\n", length(results))
    @printf("⏱️  Mean runtime: %.2f ms (CV: %.4f)\n", 
            stats.runtime_stats.mean, stats.runtime_stats.cv)
    @printf("🎯 Mean accuracy: %.2f%% error\n", 
            mean([stats.accuracy_stats.waiting, stats.accuracy_stats.utilization, 
                  stats.accuracy_stats.queue]) * 100)
    @printf("🧪 Hypothesis tests passed: %d/5\n", 
            sum(values(stats.hypothesis_results)))
    
    println("\n✅ MMC Batch Simulation completed successfully!")
    
    return (config=config, results=results, statistics=stats)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main_mmc(
        num_batches=10,      # Number of simulation batches
        λ=10.0,              # Arrival rate
        μ=2.0,               # Service rate per server
        c=6,                 # Number of servers
        max_entities=50000,  # Entities per simulation
        confidence_level=0.95 # Statistical confidence level
    )
end