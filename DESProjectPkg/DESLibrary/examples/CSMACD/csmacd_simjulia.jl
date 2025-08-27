#!/usr/bin/env julia



using SimJulia
using ResumableFunctions
using Random
using Statistics
using Printf

const NUM_NODES = 10
const LAMBDA = 5.0
const TX_TIME = 0.01
const PROP_DELAY = 0.005
const SLOT_TIME = 0.005
const SEED = 2024
const MAX_SUCCESSES = 100000

const MAX_RETRY_SCALE = max(16, NUM_NODES * 2)      # 30 for 15 nodes
const MAX_BACKOFF_EXPONENT = 8                      # Cap backoff growth at 2^8
const FAIRNESS_THRESHOLD = 5                        # Queue length threshold for priority
const STARVATION_THRESHOLD = 10                     # Emergency intervention threshold

const PRECOMPUTED_EXPONENTIALS = Ref{Vector{Float64}}(Vector{Float64}())
const PRECOMPUTED_BACKOFFS = Ref{Vector{Int}}(Vector{Int}())

const exp_counter = Ref(0)
const backoff_counter = Ref(0)
const total_generated = Ref(0)
const final_sim_time = Ref(0.0)


function deterministic_exponential()
    exp_counter[] += 1
    local arr = PRECOMPUTED_EXPONENTIALS[]
    return arr[(exp_counter[] - 1) % length(arr) + 1]
end


function deterministic_backoff_int(max_val::Int)
    backoff_counter[] += 1
    local arr = PRECOMPUTED_BACKOFFS[]
    return arr[(backoff_counter[] - 1) % length(arr) + 1] % max_val
end


mutable struct CSMACDModel
    num_nodes::Int
    packet_queues::Vector{Vector{Float64}}
    node_states::Vector{Symbol}  # :idle, :transmitting, :backoff
    ongoing_transmissions::Dict{Int, Float64}  # node_id -> start_time
    retry_counters::Vector{Int}
    successful_transmissions::Int
    total_collisions::Int
    transmission_delays::Vector{Float64}
    node_success_counts::Vector{Int}
    node_drop_counts::Vector{Int}
    dropped_frame_delays::Vector{Float64}  # Track dropped frame delays for bias analysis
    
    function CSMACDModel(num_nodes::Int)
        new(
            num_nodes,
            [Float64[] for _ in 1:num_nodes],
            fill(:idle, num_nodes),
            Dict{Int, Float64}(),
            zeros(Int, num_nodes),
            0,
            0,
            Float64[],
            zeros(Int, num_nodes),
            zeros(Int, num_nodes),
            Float64[]
        )
    end
end


function calculate_fairness_factor(model::CSMACDModel, node::Int)
    queue_len = length(model.packet_queues[node])
    
    if queue_len >= STARVATION_THRESHOLD
        return 0.1  # 90% backoff reduction
    end
    
    avg_queue_len = sum(length(q) for q in model.packet_queues) / model.num_nodes
    
    if avg_queue_len > 0
        queue_ratio = queue_len / avg_queue_len
        return max(0.3, min(2.0, 2.0 - queue_ratio))
    else
        return 1.0  # No adjustment when all queues empty
    end
end


function calculate_enhanced_backoff(model::CSMACDModel, node::Int)
    retry_count = model.retry_counters[node]
    
    k = min(retry_count, MAX_BACKOFF_EXPONENT)
    max_slots = 2^k
    backoff_slots = deterministic_backoff_int(max_slots)
    
    base_backoff = backoff_slots * SLOT_TIME
    
    fairness_factor = calculate_fairness_factor(model, node)
    
    node_jitter = (node - 1) * SLOT_TIME * 0.01  # Convert to 0-based for consistency
    random_jitter = deterministic_backoff_int(100) / 100.0 * SLOT_TIME * 0.1
    
    total_backoff = base_backoff * fairness_factor + node_jitter + random_jitter
    
    return max(SLOT_TIME * 0.1, total_backoff)  # Minimum backoff
end


function get_priority_delay(model::CSMACDModel, node::Int)
    queue_len = length(model.packet_queues[node])
    
    if queue_len >= STARVATION_THRESHOLD
        sense_delay = SLOT_TIME / 1000  # Emergency priority
    elseif queue_len >= FAIRNESS_THRESHOLD
        sense_delay = SLOT_TIME / 200   # High priority
    else
        sense_delay = SLOT_TIME / 100   # Normal priority
    end
    
    node_jitter = (node - 1) * SLOT_TIME * 0.001  # Convert to 0-based for consistency
    random_jitter = deterministic_backoff_int(100) / 100.0 * SLOT_TIME * 0.01
    
    return sense_delay + node_jitter + random_jitter
end


@resumable function frame_generation_process(sim::Simulation, model::CSMACDModel, node::Int)
    @yield timeout(sim, deterministic_exponential())
    
    while total_generated[] < MAX_SUCCESSES
        gen_time = now(sim)
        push!(model.packet_queues[node], gen_time)
        total_generated[] += 1
        
        if model.node_states[node] == :idle
            Process(transmission_attempt_process, sim, model, node)
        end
        
        if total_generated[] < MAX_SUCCESSES
            @yield timeout(sim, deterministic_exponential())
        else
            break
        end
    end
end


@resumable function transmission_attempt_process(sim::Simulation, model::CSMACDModel, node::Int)
    if model.node_states[node] != :idle || isempty(model.packet_queues[node])
        return
    end
    
    priority_delay = get_priority_delay(model, node)
    @yield timeout(sim, priority_delay)
    
    @yield Process(start_transmission_process, sim, model, node)
end


@resumable function start_transmission_process(sim::Simulation, model::CSMACDModel, node::Int)
    if model.node_states[node] != :idle || isempty(model.packet_queues[node])
        return
    end
    
    current_time = now(sim)
    
    collided_nodes = Int[]
    for (other_node, start_time) in model.ongoing_transmissions
        if other_node != node && abs(current_time - start_time) < 2 * PROP_DELAY
            push!(collided_nodes, other_node)
        end
    end
    
    if !isempty(collided_nodes)
        push!(collided_nodes, node)  # Add current node
        @yield Process(handle_collision_process, sim, model, collided_nodes)
    else
        model.node_states[node] = :transmitting
        model.ongoing_transmissions[node] = current_time
        
        Process(transmission_end_process, sim, model, node)
        @yield timeout(sim, 0)  # Make it a resumable function
    end
end


@resumable function transmission_end_process(sim::Simulation, model::CSMACDModel, node::Int)
    @yield timeout(sim, TX_TIME)
    
    if model.node_states[node] == :transmitting && haskey(model.ongoing_transmissions, node)
        finish_time = now(sim)
        
        if !isempty(model.packet_queues[node])
            gen_time = popfirst!(model.packet_queues[node])
            delay = finish_time - gen_time
            push!(model.transmission_delays, delay)
            
            model.successful_transmissions += 1
            model.node_success_counts[node] += 1
            model.retry_counters[node] = 0  # Reset retry counter
        end
        
        delete!(model.ongoing_transmissions, node)
        model.node_states[node] = :idle
        
        if !isempty(model.packet_queues[node])
            Process(transmission_attempt_process, sim, model, node)
        end
    end
end


@resumable function handle_collision_process(sim::Simulation, model::CSMACDModel, collided_nodes::Vector{Int})
    model.total_collisions += 1
    
    for node in collided_nodes
        if haskey(model.ongoing_transmissions, node)
            delete!(model.ongoing_transmissions, node)
        end
        model.node_states[node] = :backoff
        
        model.retry_counters[node] += 1
        
        queue_len = length(model.packet_queues[node])
        if queue_len >= STARVATION_THRESHOLD
            adaptive_retry_limit = MAX_RETRY_SCALE + 10  # Extra attempts for starved nodes
        else
            adaptive_retry_limit = MAX_RETRY_SCALE
        end
        
        if model.retry_counters[node] > adaptive_retry_limit
            if !isempty(model.packet_queues[node])
                dropped_gen_time = popfirst!(model.packet_queues[node])  # Drop frame
                model.node_drop_counts[node] += 1
                
                dropped_delay = now(sim) - dropped_gen_time
                push!(model.dropped_frame_delays, dropped_delay)
            end
            model.retry_counters[node] = 0
            model.node_states[node] = :idle
            
            if !isempty(model.packet_queues[node])
                Process(transmission_attempt_process, sim, model, node)
            end
        else
            backoff_time = calculate_enhanced_backoff(model, node)
            
            Process(backoff_expired_process, sim, model, node, backoff_time)
        end
    end
    @yield timeout(sim, 0)  # Make it a resumable function
end


@resumable function backoff_expired_process(sim::Simulation, model::CSMACDModel, node::Int, backoff_time::Float64)
    @yield timeout(sim, backoff_time)
    
    model.node_states[node] = :idle
    
    if !isempty(model.packet_queues[node])
        Process(transmission_attempt_process, sim, model, node)
    end
end


@resumable function simulation_monitor(sim::Simulation, model::CSMACDModel)
    last_report_time = 0.0
    
    while true
        @yield timeout(sim, 1.0)  # Check every simulation time unit
        
        current_time = now(sim)
        frames_queued = sum(length(q) for q in model.packet_queues)
        
        if isfinite(current_time)
            final_sim_time[] = current_time
        end
        
        if current_time - last_report_time >= 2.0
            if frames_queued > 0
                # progress monitor suppressed
            end
            last_report_time = current_time
        end
        
        if total_generated[] >= MAX_SUCCESSES && frames_queued == 0
            println(" All nodes idle with empty queues - simulation complete")
            final_sim_time[] = current_time
            break
        end
        
        if current_time > 5000.0
            @printf("\n Safety break at time %.2f - simulation may be stuck\n", current_time)
            final_sim_time[] = current_time
            break
        end
    end
end


function run_simjulia_simulation()
    println("\nCSMA/CD Simulation using SimJulia")
    println("=" ^ 50)
    println("Nodes: $NUM_NODES, λ: $LAMBDA, Target: $MAX_SUCCESSES frames")
    println("TX time: $TX_TIME, Prop delay: $PROP_DELAY, Slot time: $SLOT_TIME")
    println("Adaptive retry limits: $MAX_RETRY_SCALE, Fairness threshold: $FAIRNESS_THRESHOLD")

    Random.seed!(SEED)
    PRECOMPUTED_EXPONENTIALS[] = [randexp() / LAMBDA for _ in 1:10000]
    PRECOMPUTED_BACKOFFS[] = [rand(0:255) for _ in 1:10000]
    exp_counter[] = 0
    backoff_counter[] = 0
    total_generated[] = 0
    final_sim_time[] = 0.0

    sim = Simulation()
    
    model = CSMACDModel(NUM_NODES)
    
    for node in 1:NUM_NODES
        Process(frame_generation_process, sim, model, node)
    end
    
    Process(simulation_monitor, sim, model)

    start_time = time()
    
    run(sim)
    
    wall_clock = (time() - start_time) * 1000
    sim_time = now(sim)
    
    if !isfinite(sim_time)
        sim_time = final_sim_time[]
        println("WARNING: SimJulia returned Inf time, using last tracked time: $sim_time")
    end

    throughput = model.successful_transmissions / sim_time
    collision_rate = model.total_collisions / sim_time
    avg_delay = !isempty(model.transmission_delays) ? mean(model.transmission_delays) : 0.0
    
    total_drops = sum(model.node_drop_counts)
    drop_rate = total_drops / sim_time
    
    if model.successful_transmissions > 0
        sum_successes = sum(model.node_success_counts)
        sum_squares = sum(x^2 for x in model.node_success_counts)
        fairness_index = sum_successes^2 / (model.num_nodes * sum_squares)
    else
        fairness_index = 1.0
    end
    
    queue_lengths = [length(q) for q in model.packet_queues]
    max_queue = !isempty(queue_lengths) ? maximum(queue_lengths) : 0
    min_queue = !isempty(queue_lengths) ? minimum(queue_lengths) : 0
    avg_queue = !isempty(queue_lengths) ? mean(queue_lengths) : 0.0

    println("\n  SIMJULIA SIMULATION RESULTS")
    println("=" ^ 50)
    @printf("Wall-clock time: %.2f ms\n", wall_clock)
    @printf("Simulation time: %.2f time units\n", sim_time)
    @printf("Successful transmissions: %d\n", model.successful_transmissions)
    @printf("Total collisions: %d\n", model.total_collisions)
    @printf("Total generated frames: %d\n", total_generated[])
    @printf("Total dropped frames: %d\n", total_drops)
    @printf("Frames still queued: %d\n", sum(queue_lengths))
    @printf("Frame loss rate: %.3f%%\n", total_drops / total_generated[] * 100)

    println("\nPERFORMANCE METRICS")
    println("=" ^ 50)
    @printf("Throughput: %.4f frames/time unit\n", throughput)
    @printf("Collision rate: %.4f collisions/time unit\n", collision_rate)
    @printf("Drop rate: %.4f drops/time unit\n", drop_rate)
    @printf("Average delay: %.6f time units\n", avg_delay)
    @printf("Fairness index: %.4f (1.0 = perfect fairness)\n", fairness_index)
    
    if !isempty(model.dropped_frame_delays)
        avg_dropped_delay = mean(model.dropped_frame_delays)
        max_dropped_delay = maximum(model.dropped_frame_delays)
        println("\nDELAY ANALYSIS (Bias Investigation)")
        println("=" ^ 50)
        @printf("Successful frame avg delay: %.6f time units\n", avg_delay)
        @printf("Dropped frame avg delay: %.6f time units\n", avg_dropped_delay)
        @printf("Max dropped frame delay: %.6f time units\n", max_dropped_delay)
        @printf("Dropped frames: %d\n", length(model.dropped_frame_delays))
        if avg_delay > 0
            @printf("Delay bias factor: %.2fx\n", avg_dropped_delay / avg_delay)
            println("This explains why avg delay seems low - high-delay frames are dropped!")
        end
    end

    # println("\nQUEUE DISTRIBUTION")
    # println("=" ^ 50)
    # @printf("Max queue length: %d\n", max_queue)
    # @printf("Min queue length: %d\n", min_queue)
    # @printf("Average queue length: %.2f\n", avg_queue)
    # @printf("Queue imbalance: %d\n", max_queue - min_queue)

    println("\nPER-NODE STATISTICS")
    println("=" ^ 50)
    println("Node | Successes | Drops | Queued | Success Rate")
    println("-" ^ 50)
    
    for i in 1:NUM_NODES
        successes = model.node_success_counts[i]
        drops = model.node_drop_counts[i]
        queued = length(model.packet_queues[i])
        total_attempts = successes + drops
        success_rate = total_attempts > 0 ? successes / total_attempts * 100 : 0.0
        
        @printf("%4d | %9d | %5d | %6d | %11.3f%%\n", i, successes, drops, queued, success_rate)
    end

    println("\nSimJulia CSMA/CD simulation completed!")
    
    expected_collision_rate_15_nodes = 68.0  # Based on previous 15-node results
    match_quality = collision_rate / expected_collision_rate_15_nodes
    
    
    # println("\nRandom sequence usage:")
    # @printf("Exponentials used: %d/%d\n", exp_counter[], length(PRECOMPUTED_EXPONENTIALS[]))
    # @printf("Backoffs used: %d/%d\n", backoff_counter[], length(PRECOMPUTED_BACKOFFS[]))
    
    
    return model, sim_time, wall_clock
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_simjulia_simulation()
end