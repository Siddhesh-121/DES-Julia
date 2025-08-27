#!/usr/bin/env julia



using Base.Threads
using Printf
using Statistics
using Random
using Distributions

try
    using HypothesisTests
catch e
    println(" Installing HypothesisTests.jl...")
    using Pkg
    Pkg.add("HypothesisTests")
    using HypothesisTests
end

if !isdefined(Main, :DESLibrary)
    include("../../src/DESLibrary.jl")
    using .DESLibrary
else
    using .DESLibrary
end


const LOCAL_NUM_NODES = 10
const LOCAL_LAMBDA = 5.0
const LOCAL_TX_TIME = 0.01
const LOCAL_PROP_DELAY = 0.005
const LOCAL_SLOT_TIME = 0.005
const LOCAL_MAX_RETRY_BASE = 16
const LOCAL_MAX_RETRY_SCALE = max(16, LOCAL_NUM_NODES * 2)
const LOCAL_MAX_BACKOFF_EXPONENT = 8
const LOCAL_FAIRNESS_THRESHOLD = 5
const LOCAL_STARVATION_THRESHOLD = 10

struct ThreadLocalValues
    exponentials::Vector{Float64}
    backoffs::Vector{Int}
    exp_counter::Ref{Int}
    backoff_counter::Ref{Int}
end

function ThreadLocalValues(seed::Int)
    Random.seed!(seed)
    ThreadLocalValues(
        [randexp() / LOCAL_LAMBDA for _ in 1:10000],
        [rand(0:255) for _ in 1:10000],
        Ref(1),
        Ref(1)
    )
end

const thread_local_values = Dict{Int, ThreadLocalValues}()
const thread_local_lock = ReentrantLock()

function get_thread_local_values()
    tid = threadid()
    lock(thread_local_lock) do
        if !haskey(thread_local_values, tid)
            thread_local_values[tid] = ThreadLocalValues(2024 + tid * 1000)
        end
        return thread_local_values[tid]
    end
end

function local_deterministic_exponential()
    vals = get_thread_local_values()
    val = vals.exponentials[vals.exp_counter[]]
    vals.exp_counter[] = vals.exp_counter[] % length(vals.exponentials) + 1
    return val
end

function local_deterministic_backoff(max_slots::Int)
    vals = get_thread_local_values()
    val = vals.backoffs[vals.backoff_counter[]]
    vals.backoff_counter[] = vals.backoff_counter[] % length(vals.backoffs) + 1
    return val % max_slots
end

function reset_thread_local_counters!()
    vals = get_thread_local_values()
    vals.exp_counter[] = 1
    vals.backoff_counter[] = 1
end

mutable struct LocalEnhancedCSMACDModel <: DESLibrary.SimulationModel
    num_nodes::Int
    model_id::Symbol
    max_frames::Int  
    
    node_states::Vector{Symbol}
    
    packet_queues::Vector{Vector{Float64}}
    
    ongoing_transmissions::Dict{Int, Float64}  
    
    retry_counters::Vector{Int}
    
    node_success_counts::Vector{Int}            
    node_drop_counts::Vector{Int}              
    node_generated_counts::Vector{Int}         
    last_fairness_check::Float64               
    
    successful_transmissions::Int
    total_collisions::Int
    total_frames_generated::Int                 
    transmission_delays::Vector{Float64}
    max_queue_lengths::Vector{Int}              
    
    function LocalEnhancedCSMACDModel(num_nodes::Int, model_id::Symbol, max_frames::Int)
        new(
            num_nodes,
            model_id,
            max_frames,
            fill(:idle, num_nodes),
            [Vector{Float64}() for _ in 1:num_nodes],
            Dict{Int, Float64}(),
            zeros(Int, num_nodes),
            zeros(Int, num_nodes),
            zeros(Int, num_nodes),
            zeros(Int, num_nodes),
            0.0,
            0,
            0,
            0,
            Vector{Float64}(),
            zeros(Int, num_nodes)
        )
    end
end

function DESLibrary.initialize_model!(model::LocalEnhancedCSMACDModel, engine::DESEngine)
    for node in 1:model.num_nodes
        interarrival = local_deterministic_exponential()
        event = create_generic_event(interarrival, node, model.model_id, :frame_generated)
        DESLibrary.schedule_event!(engine, event)
    end
    println(" Initialized $(model.num_nodes) nodes with frame generation (target: $(model.max_frames) frames)")
end

function DESLibrary.finalize_model!(model::LocalEnhancedCSMACDModel, engine::DESEngine)
    return nothing
end

function DESLibrary.process_event!(model::LocalEnhancedCSMACDModel, event::GenericEvent, engine::DESEngine)
    
    current_time = DESLibrary.get_current_time(engine)
    
    if model.total_frames_generated >= model.max_frames
        total_processed = model.successful_transmissions + sum(model.node_drop_counts)
        total_queued = sum(length(queue) for queue in model.packet_queues)
        total_transmitting = length(model.ongoing_transmissions)
        
        if total_processed >= model.total_frames_generated && total_queued == 0 && total_transmitting == 0
            return  # Stop processing - simulation complete
        end
    end
    node = event.entity_id
    
    if event.event_type == :frame_generated
        _handle_frame_generation!(model, node, current_time, engine)
    elseif event.event_type == :transmission_complete
        _handle_transmission_complete!(model, node, current_time, engine)
    elseif event.event_type == :backoff_complete
        _handle_backoff_complete!(model, node, current_time, engine)
    end
end

function _handle_frame_generation!(model::LocalEnhancedCSMACDModel, node::Int, current_time::Float64, engine::DESEngine)
    if model.total_frames_generated >= model.max_frames
        return
    end
    
    push!(model.packet_queues[node], current_time)
    model.node_generated_counts[node] += 1  # Track per-node generation
    model.total_frames_generated += 1       # Track global generation count
    
    current_queue_length = length(model.packet_queues[node])
    if current_queue_length > model.max_queue_lengths[node]
        model.max_queue_lengths[node] = current_queue_length
    end
    
    if model.total_frames_generated < model.max_frames
        interarrival = local_deterministic_exponential()
        next_event = create_generic_event(current_time + interarrival, node, model.model_id, :frame_generated)
        DESLibrary.schedule_event!(engine, next_event)
    end
    
    if model.node_states[node] == :idle && !isempty(model.packet_queues[node])
        _attempt_transmission!(model, node, current_time, engine)
    end
end

function _attempt_transmission!(model::LocalEnhancedCSMACDModel, node::Int, current_time::Float64, engine::DESEngine)
    carrier_busy = false
    for (other_node, start_time) in model.ongoing_transmissions
        sense_time = start_time + LOCAL_PROP_DELAY
        if current_time >= sense_time
            carrier_busy = true
            break
        end
    end
    
    if carrier_busy
        _apply_backoff!(model, node, current_time, engine)
        return
    end
    
    model.node_states[node] = :transmitting
    model.ongoing_transmissions[node] = current_time
    
    tx_complete_time = current_time + LOCAL_TX_TIME
    event = create_generic_event(tx_complete_time, node, model.model_id, :transmission_complete)
    DESLibrary.schedule_event!(engine, event)
end

function _handle_transmission_complete!(model::LocalEnhancedCSMACDModel, node::Int, current_time::Float64, engine::DESEngine)
    
    if !haskey(model.ongoing_transmissions, node)
        model.node_states[node] = :idle
        return
    end
    
    collision_detected = length(model.ongoing_transmissions) > 1
    
    delete!(model.ongoing_transmissions, node)
    
    if collision_detected
        model.total_collisions += 1
        model.node_states[node] = :idle
        _apply_backoff!(model, node, current_time, engine)
        
        other_nodes = collect(keys(model.ongoing_transmissions))
        for other_node in other_nodes
            delete!(model.ongoing_transmissions, other_node)
            model.node_states[other_node] = :idle
            _apply_backoff!(model, other_node, current_time, engine)
        end
    else
        if !isempty(model.packet_queues[node])
            frame_time = popfirst!(model.packet_queues[node])
            delay = current_time - frame_time
            push!(model.transmission_delays, delay)
            
            model.successful_transmissions += 1
            model.node_success_counts[node] += 1
            model.retry_counters[node] = 0  # Reset retry counter
        end
        
        model.node_states[node] = :idle
        
        if !isempty(model.packet_queues[node])
            _attempt_transmission!(model, node, current_time, engine)
        end
    end
end

function _apply_backoff!(model::LocalEnhancedCSMACDModel, node::Int, current_time::Float64, engine::DESEngine)
    model.retry_counters[node] += 1
    
    max_retries = LOCAL_MAX_RETRY_SCALE
    if model.retry_counters[node] > max_retries
        if !isempty(model.packet_queues[node])
            popfirst!(model.packet_queues[node])
            model.node_drop_counts[node] += 1
        end
        model.retry_counters[node] = 0
        model.node_states[node] = :idle
        return
    end
    
    backoff_exponent = min(model.retry_counters[node], LOCAL_MAX_BACKOFF_EXPONENT)
    max_slots = 2^backoff_exponent
    slots = local_deterministic_backoff(max_slots)
    backoff_time = slots * LOCAL_SLOT_TIME
    
    model.node_states[node] = :backoff
    
    event = create_generic_event(current_time + backoff_time, node, model.model_id, :backoff_complete)
    DESLibrary.schedule_event!(engine, event)
end

function _handle_backoff_complete!(model::LocalEnhancedCSMACDModel, node::Int, current_time::Float64, engine::DESEngine)
    model.node_states[node] = :idle
    
    if !isempty(model.packet_queues[node])
        _attempt_transmission!(model, node, current_time, engine)
    end
end

function get_statistics(model::LocalEnhancedCSMACDModel, engine::DESEngine)
    sim_time = DESLibrary.get_current_time(engine)
    
    throughput = sim_time > 0 ? model.successful_transmissions / sim_time : 0.0
    collision_rate = sim_time > 0 ? model.total_collisions / sim_time : 0.0
    avg_delay = !isempty(model.transmission_delays) ? sum(model.transmission_delays) / length(model.transmission_delays) : 0.0
    
    total_drops = sum(model.node_drop_counts)
    drop_rate = sim_time > 0 ? total_drops / sim_time : 0.0
    
    if model.successful_transmissions > 0
        sum_successes = sum(model.node_success_counts)
        sum_squares = sum(x^2 for x in model.node_success_counts)
        fairness_index = sum_successes^2 / (model.num_nodes * sum_squares)
    else
        fairness_index = 1.0
    end
    
    queue_lengths = [length(q) for q in model.packet_queues]
    max_queue = maximum(queue_lengths)
    min_queue = minimum(queue_lengths)
    avg_queue = sum(queue_lengths) / length(queue_lengths)
    
    return Dict(
        :simulation_time => sim_time,
        :successful_transmissions => model.successful_transmissions,
        :total_collisions => model.total_collisions,
        :total_drops => total_drops,
        :throughput => throughput,
        :collision_rate => collision_rate,
        :average_delay => avg_delay,
        :drop_rate => drop_rate,
        :fairness_index => fairness_index,
        :max_queue_length => max_queue,
        :min_queue_length => min_queue,
        :avg_queue_length => avg_queue,
        :node_success_counts => copy(model.node_success_counts),
        :node_drop_counts => copy(model.node_drop_counts),
        :node_generated_counts => copy(model.node_generated_counts),
        :max_queue_lengths => copy(model.max_queue_lengths)
    )
end

mutable struct CSMACDBatchConfig
    num_nodes::Int
    λ::Float64
    max_frames::Int
    num_batches::Int
    num_threads::Int
    confidence_level::Float64
    base_seed::Int
    
    function CSMACDBatchConfig(;
        num_nodes=10, λ=5.0, max_frames=100000,  # Enhanced model - configurable
        num_batches=20, num_threads=nthreads(), confidence_level=0.95,
        base_seed=2024
    )
        new(num_nodes, λ, max_frames, num_batches, num_threads, confidence_level, base_seed)
    end
end

struct CSMACDBatchResult
    batch_id::Int
    thread_id::Int
    runtime_ms::Float64
    successful_transmissions::Int
    total_collisions::Int
    total_drops::Int
    throughput::Float64
    collision_rate::Float64
    drop_rate::Float64
    avg_delay::Float64
    fairness_index::Float64
end

function run_single_csmacd_simulation(config::CSMACDBatchConfig, batch_id::Int)
    try
        thread_id = threadid()
        simulation_seed = config.base_seed + batch_id * 1000
        
        start_time = time()
        
        
        engine_capacity = max(10000, config.max_frames ÷ 10)  # Scale capacity based on target
        engine = DESEngine(engine_capacity, simulation_seed)
        model_id = Symbol("csmacd_batch_$batch_id")
        
        thread_rng = MersenneTwister(simulation_seed)
        
        model = LocalEnhancedCSMACDModel(config.num_nodes, model_id, config.max_frames)
        add_model!(engine, model_id, model)
        
        simulate!(engine, verbose=false)
        
        end_time = time()
        runtime_ms = (end_time - start_time) * 1000
        
        stats = get_statistics(model, engine)
        
        successful_transmissions = stats[:successful_transmissions]
        total_collisions = stats[:total_collisions]
        total_drops = stats[:total_drops]
        avg_delay = stats[:average_delay]
        
        throughput = stats[:throughput]
        collision_rate = stats[:collision_rate]
        drop_rate = stats[:drop_rate]
        
        fairness_index = stats[:fairness_index]
        
        return CSMACDBatchResult(
            batch_id, thread_id, runtime_ms,
            successful_transmissions,
            total_collisions,
            total_drops,
            throughput, collision_rate, drop_rate,
            avg_delay,
            fairness_index
        )
        
    catch e
        println("\n Error in batch $batch_id on thread $(threadid()): $e")
        rethrow(e)
    end
end

function run_csmacd_batch_simulations(config::CSMACDBatchConfig)
    println("Starting CSMA/CD Batch Simulations")
    println("=" ^ 60)
    println("Configuration:")
    println("  Network nodes: $(config.num_nodes)")
    println("  Frame rate: $(config.λ) frames/time unit")
            println("  Max frames: $(config.max_frames)")
    println("  Number of batches: $(config.num_batches)")
    println("  Available threads: $(nthreads())")
    println("  Using threads: $(config.num_threads)")
    
    println("\nExecuting batch simulations...")
    
    results = Vector{CSMACDBatchResult}(undef, config.num_batches)
    
    println("\n Per-thread warmup (eliminating cold start JIT compilation)...")
    warmup_start_time = time()
    warmup_tasks = Vector{Task}(undef, min(config.num_threads, nthreads()))
    warmup_runtimes = Vector{Float64}(undef, min(config.num_threads, nthreads()))
    
    if config.num_threads > 1
        for i in 1:min(config.num_threads, nthreads())
            warmup_tasks[i] = @spawn begin
                try
                    thread_warmup_start = time()
                    
                    warmup_seed = config.base_seed + i * 10000
                    warmup_engine = DESEngine(1000, warmup_seed)
                    warmup_model_id = Symbol("warmup_$i")
                    warmup_model = LocalEnhancedCSMACDModel(config.num_nodes, warmup_model_id, 100)
                    add_model!(warmup_engine, warmup_model_id, warmup_model)
                    
                    simulate!(warmup_engine, verbose=false)
                    
                    thread_warmup_time = (time() - thread_warmup_start) * 1000
                    println("Thread $(threadid()) warmed up ✓ ($(round(thread_warmup_time, digits=1))ms)")
                    return thread_warmup_time
                catch e
                    println("Warmup failed on thread $(threadid()): $e")
                    return 0.0
                end
            end
        end
        
        for (i, task) in enumerate(warmup_tasks)
            warmup_runtimes[i] = fetch(task)
        end
        
        warmup_total_time = (time() - warmup_start_time) * 1000
        warmup_mean = mean(warmup_runtimes[warmup_runtimes .> 0])
        
        println(" All $(min(config.num_threads, nthreads())) threads warmed up")
        println("   Warmup statistics:")
        println("   - Total warmup time: $(round(warmup_total_time, digits=1))ms")
        println("   - Mean warmup per thread: $(round(warmup_mean, digits=1))ms")
        println("   - Individual warmup times: [$(join([round(t, digits=1) for t in warmup_runtimes], ", "))]ms")
    end
    
    batch_start_time = time()
    
    if config.num_threads == 1
        for i in 1:config.num_batches
            print(".")
            results[i] = run_single_csmacd_simulation(config, i)
        end
    elseif config.num_threads >= nthreads()
        @threads for i in 1:config.num_batches
            print(".")
            results[i] = run_single_csmacd_simulation(config, i)
        end
    else
        semaphore = Base.Semaphore(config.num_threads)
        tasks = Vector{Task}(undef, config.num_batches)
        
        for i in 1:config.num_batches
            tasks[i] = @spawn begin
                Base.acquire(semaphore)
                try
                    print(".")
                    result = run_single_csmacd_simulation(config, i)
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
    
    println("\nCompleted $(config.num_batches) batches in $(round(total_batch_time, digits=2))s")
    
    return results
end

function entry_banner_csmacd()
    println("CSMA/CD Batch Analysis Framework")
    println("=" ^ 60)
    println("Statistical Testing for CSMA/CD Network Protocol")
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


function run_csmacd_batch_analysis(;
    num_batches::Int=20,
    num_nodes::Int=10,
    λ::Float64=5.0,
    max_frames::Int=100000,
    confidence_level::Float64=0.95
)
    println("Starting CSMA/CD Batch Analysis...")
    
    try
        config = CSMACDBatchConfig(
            num_nodes=num_nodes,
            λ=λ,
            max_frames=max_frames,
            num_batches=num_batches,
            num_threads=nthreads(),  # Use available threads
            confidence_level=confidence_level
        )
        
        results = run_csmacd_batch_simulations(config)
        
        runtimes = [r.runtime_ms for r in results]
        throughputs = [r.throughput for r in results]
        collisions = [r.total_collisions for r in results]
        drops = [r.total_drops for r in results]
        
        result = (
            results = results,
            statistics = (
                runtime_stats = (mean = mean(runtimes), std = std(runtimes)),
                throughput_stats = (mean = mean(throughputs), std = std(throughputs)),
                collision_stats = (mean = mean(collisions), std = std(collisions)),
                drop_stats = (mean = mean(drops), std = std(drops))
            ),
            config = config
        )
        
        println("CSMA/CD analysis completed successfully")
        return result
        
    catch e
        println("CSMA/CD analysis failed: $e")
        @show e
        return nothing
    end
end


function analyze_csmacd_results(csmacd_result)
    if csmacd_result === nothing
        println("Cannot perform analysis - CSMA/CD simulation failed")
        return nothing
    end
    
    println("\nCSMA/CD DETAILED ANALYSIS")
    println("=" ^ 60)
    
    csmacd_stats = csmacd_result.statistics
    
    println("NETWORK PERFORMANCE INSIGHTS")
    println("-" ^ 40)
    
    batch_count = length(csmacd_result.results)
    avg_successes = mean([r.successful_transmissions for r in csmacd_result.results])
    avg_runtime_s = csmacd_stats.runtime_stats.mean / 1000
    
    avg_collisions = mean([r.total_collisions for r in csmacd_result.results])
    avg_drops = mean([r.total_drops for r in csmacd_result.results])
    total_network_events = avg_successes + avg_collisions + avg_drops
    events_per_sec = total_network_events / avg_runtime_s
    
    @printf("Batch count:          %d\n", batch_count)
    @printf("Avg successes/batch:  %.0f\n", avg_successes)
    @printf("Avg collisions/batch: %.0f\n", avg_collisions)
    @printf("Avg drops/batch:      %.0f\n", avg_drops)
    @printf("Network event rate:   ~%.0f events/second\n", events_per_sec)
    
    println("\nINDIVIDUAL BATCH RUNTIMES")
    println("-" ^ 40)
    println("Batch | Runtime (ms) | Thread ID | Successes | Collisions | Drops")
    println("-" ^ 65)
    
    runtimes = [r.runtime_ms for r in csmacd_result.results]
    mean_runtime = mean(runtimes)
    
    for (i, result) in enumerate(csmacd_result.results)
        deviation_pct = ((result.runtime_ms - mean_runtime) / mean_runtime) * 100
        deviation_symbol = if abs(deviation_pct) < 10 "  ~" elseif deviation_pct > 0 "  +" else "  -" end
        @printf("%5d | %11.2f | %9d | %9d | %10d | %5d %s\n",
                i, result.runtime_ms, result.thread_id,
                result.successful_transmissions, result.total_collisions, 
                result.total_drops, deviation_symbol)
    end
    
    @printf("\nRuntime Statistics:\n")
    @printf("  Mean:   %.2f ms\n", mean_runtime)
    runtime_std = std(runtimes)
    runtime_cv = runtime_std / mean_runtime  # Calculate coefficient of variation
    @printf("  Std:    %.2f ms\n", runtime_std)
    @printf("  CV:     %.4f\n", runtime_cv)
    @printf("  Min:    %.2f ms\n", minimum(runtimes))
    @printf("  Max:    %.2f ms\n", maximum(runtimes))
    
    println("\nEFFICIENCY ASSESSMENT")
    println("-" ^ 40)
    
    throughput = csmacd_stats.throughput_stats.mean
    collision_rate = csmacd_stats.collision_stats.mean
    drop_rate = csmacd_stats.drop_stats.mean
    
    total_attempts = avg_successes + avg_collisions
    efficiency = total_attempts > 0 ? avg_successes / total_attempts : 0.0
    
    loss_rate = avg_successes > 0 ? (avg_drops / avg_successes) * 100 : 0.0
    
    fairness = mean([r.fairness_index for r in csmacd_result.results])
    
    @printf("Network efficiency:   %.2f%% (successes vs total attempts)\n", efficiency * 100)
    @printf("Throughput:           %.2f frames/time\n", throughput)
    @printf("Collision rate:       %.2f collisions/time\n", collision_rate)
    @printf("Frame loss rate:      %.2f%%\n", loss_rate)
    @printf("Fairness index:       %.4f (1.0 = perfect)\n", fairness)
    
    println("\nCONSISTENCY ASSESSMENT")
    println("-" ^ 40)
    
    runtime_cv = csmacd_stats.runtime_stats.mean > 0 ? csmacd_stats.runtime_stats.std / csmacd_stats.runtime_stats.mean : 0.0
    throughput_cv = csmacd_stats.throughput_stats.mean > 0 ? csmacd_stats.throughput_stats.std / csmacd_stats.throughput_stats.mean : 0.0
    
    consistency_threshold = 0.15  # 15% CV threshold for CSMA/CD
    csmacd_consistent = runtime_cv < consistency_threshold
    
    @printf("Runtime consistency:  %s (CV: %.4f)\n", 
            csmacd_consistent ? "High" : "Moderate", runtime_cv)
    @printf("Throughput consistency: CV: %.4f\n", throughput_cv)
    
    tests_passed = 0
    total_tests = 4
    
    if runtime_cv < 0.2; tests_passed += 1; end
    if throughput_cv < 0.2; tests_passed += 1; end  
    if efficiency > 0.3; tests_passed += 1; end  # At least 30% efficiency
    if fairness > 0.8; tests_passed += 1; end   # Good fairness
    
    @printf("Statistical tests:    %d/%d passed (%.0f%%)\n", 
            tests_passed, total_tests, (tests_passed/total_tests)*100)
    
    if fairness < 0.8
        println("\nFAIRNESS ISSUES DETECTED")
        println("-" ^ 40)
        @printf("Network fairness index: %.4f (below 0.8 threshold)\n", fairness)
        println("Some nodes may be experiencing unequal access to the channel")
    else
        println("\n FAIRNESS ASSESSMENT")
        println("-" ^ 40)
        println("All nodes performing within acceptable range")
    end
    
    println("\nRECOMMENDATIONS")
    println("-" ^ 40)
    
    if efficiency < 0.5
        println("• Network efficiency is low - consider reducing load or optimizing protocol")
    end
    
    if collision_rate > throughput * 2
        println("• High collision rate - network may be overloaded")
    end
    
    if loss_rate > 10.0
        println("• High frame loss rate - consider increasing retry limits")
    end
    
    if fairness < 0.8
        println("• Poor fairness - some nodes may be starved")
    end
    
    if runtime_cv > 0.2
        println("• Runtime variance is high - consider larger sample sizes")
    end
    
    if tests_passed < 3
        println("• Multiple statistical tests failed - review implementation")
    end
    
    if events_per_sec < 100
        println("• Low event processing rate - consider optimization")
    end
    
    
    return (
        efficiency = efficiency,
        throughput = throughput,
        fairness = fairness,
        consistency = csmacd_consistent,
        tests_passed = tests_passed,
        events_per_sec = events_per_sec
    )
end


function run_csmacd_analysis(; num_batches::Int=20, num_nodes::Int=10, λ::Float64=5.0, 
                              max_frames::Int=100000, confidence_level::Float64=0.95)
    println("CSMA/CD BATCH ANALYSIS")
    println("Network: $num_nodes nodes with λ=$λ frames/time")
    println("Target: $max_frames frames generated per batch")
    println("Batches: $num_batches")
    println("Confidence level: $(confidence_level * 100)%")
    println()
    
    start_time = time()
    
    csmacd_result = run_csmacd_batch_analysis(
        num_batches=num_batches,
        num_nodes=num_nodes,
        λ=λ,
        max_frames=max_frames,
        confidence_level=confidence_level
    )
    
    analysis = analyze_csmacd_results(csmacd_result)
    
    total_time = time() - start_time
    
    println("\nANALYSIS COMPLETE")
    println("=" ^ 60)
    @printf("Total execution time: %.2f seconds\n", total_time)
    @printf("Batches completed: %d\n", num_batches)
    
    if nthreads() > 1
        @printf("Multi-threading efficiency: ~%.1fx speedup\n", min(nthreads(), num_batches))
    end
    
    return (csmacd_result=csmacd_result, analysis=analysis)
end



function quick_test()
    println("Running Quick CSMA/CD Test (3 batches)...")
    return run_csmacd_analysis(num_batches=3, max_frames=100000)
end


function standard_test()
    println("Running Standard CSMA/CD Test (20 batches)...")
    return run_csmacd_analysis(num_batches=20, max_frames=100000)
end


function extensive_test()
    println("Running Extensive CSMA/CD Test (30 batches)...")
    return run_csmacd_analysis(num_batches=30, max_frames=100000)
end


if abspath(PROGRAM_FILE) == @__FILE__
    entry_banner_csmacd()
    println()
    println("Running standard CSMA/CD test...")
    println("🔧 Testing with multi-threading...")
    result = run_csmacd_analysis(num_batches=20, max_frames=100000)
end