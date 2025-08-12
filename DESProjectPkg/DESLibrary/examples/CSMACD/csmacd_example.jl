using Random

if !isdefined(Main, :DESLibrary)
    include("../../src/DESLibrary.jl")
end
using .DESLibrary

const NUM_NODES = 10
const LAMBDA = 5.0          # Frame generation rate (frames/time unit)
const TX_TIME = 0.01        # Transmission time
const PROP_DELAY = 0.005    # Propagation delay
const SLOT_TIME = 0.005     # Slot time for backoff
const MAX_SUCCESSES = 100000    # Target successful transmissions
const MAX_ENTITIES = 10000  # DES engine capacity
const SEED = 2024          # Random seed
const WARMUP_FRACTION = 0.1

const MAX_RETRY_BASE = 16                           # Base retry limit
const MAX_RETRY_SCALE = max(16, NUM_NODES * 2)      # Adaptive retry limit
const MAX_BACKOFF_EXPONENT = 8                      # Cap backoff growth at 2^8
const FAIRNESS_THRESHOLD = 5                        # Queue length threshold for priority
const STARVATION_THRESHOLD = 10                     # Emergency intervention threshold

println("\n🏗️  CSMA/CD Simulation using DESLibrary")
println("=" ^ 50)
println("Nodes: $NUM_NODES, λ: $LAMBDA, Target: $MAX_SUCCESSES frames")
println("TX time: $TX_TIME, Prop delay: $PROP_DELAY, Slot time: $SLOT_TIME")

Random.seed!(SEED)
const PRECOMPUTED_EXPONENTIALS = [randexp() / LAMBDA for _ in 1:10000]
const PRECOMPUTED_BACKOFFS = [rand(0:255) for _ in 1:10000]

exp_counter = Ref(1)
backoff_counter = Ref(1)
total_generated = Ref(0)


function deterministic_exponential()
    val = PRECOMPUTED_EXPONENTIALS[exp_counter[]]
    exp_counter[] = exp_counter[] % length(PRECOMPUTED_EXPONENTIALS) + 1
    return val
end


function deterministic_backoff(max_slots::Int)
    val = PRECOMPUTED_BACKOFFS[backoff_counter[]]
    backoff_counter[] = backoff_counter[] % length(PRECOMPUTED_BACKOFFS) + 1
    return val % max_slots
end


mutable struct EnhancedCSMACDModel <: DESLibrary.SimulationModel
    num_nodes::Int
    model_id::Symbol
    
    node_states::Vector{Symbol}
    
    packet_queues::Vector{Vector{Float64}}
    
    ongoing_transmissions::Dict{Int, Float64}  # node_id -> start_time
    
    retry_counters::Vector{Int}
    
    node_success_counts::Vector{Int}            # Successful transmissions per node
    node_drop_counts::Vector{Int}               # Dropped frames per node
    node_generated_counts::Vector{Int}          # Total frames generated per node
    last_fairness_check::Float64                # Time of last fairness adjustment
    
    successful_transmissions::Int
    total_collisions::Int
    transmission_delays::Vector{Float64}
    max_queue_lengths::Vector{Int}              # Track max queue length per node
    
    function EnhancedCSMACDModel(num_nodes::Int, model_id::Symbol = :csmacd)
        new(
            num_nodes,
            model_id,
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
            Vector{Float64}(),
            zeros(Int, num_nodes)
        )
    end
end


function calculate_fairness_factor(model::EnhancedCSMACDModel, node::Int)
    queue_len = length(model.packet_queues[node])
    
    if queue_len >= STARVATION_THRESHOLD
        return 0.1  # 90% backoff reduction
    end
    
    avg_queue_len = sum(length.(model.packet_queues)) / model.num_nodes
    
    if avg_queue_len > 0
        queue_ratio = queue_len / avg_queue_len
        return max(0.3, min(2.0, 2.0 - queue_ratio))
    else
        return 1.0  # No adjustment when all queues empty
    end
end


function calculate_enhanced_backoff(model::EnhancedCSMACDModel, node::Int)
    retry_count = model.retry_counters[node]
    
    k = min(retry_count, MAX_BACKOFF_EXPONENT)
    max_slots = 2^k
    backoff_slots = deterministic_backoff(max_slots)
    
    base_backoff = backoff_slots * SLOT_TIME
    
    fairness_factor = calculate_fairness_factor(model, node)
    
    node_jitter = (node - 1) * SLOT_TIME * 0.01
    random_jitter = rand() * SLOT_TIME * 0.1
    
    total_backoff = base_backoff * fairness_factor + node_jitter + random_jitter
    
    return max(SLOT_TIME * 0.1, total_backoff)  # Minimum backoff
end


function initialize_model!(model::EnhancedCSMACDModel, engine::DESEngine)
    for node in 1:model.num_nodes
        interarrival = deterministic_exponential()
        event = create_generic_event(interarrival, node, model.model_id, :frame_generated)
        DESLibrary.schedule_event!(engine, event)
    end
    println("✅ Initialized $(model.num_nodes) nodes with frame generation")
end


function safe_schedule_event!(engine::DESEngine, event::GenericEvent)
    current_time = DESLibrary.get_current_time(engine)
    if event.time >= current_time
        DESLibrary.schedule_event!(engine, event)
    else
        adjusted_event = create_generic_event(
            current_time + 1e-9, event.entity_id, event.model_id, event.event_type
        )
        DESLibrary.schedule_event!(engine, adjusted_event)
    end
end


function handle_collision!(model::EnhancedCSMACDModel, engine::DESEngine, collided_nodes::Vector{Int}, time::Float64)
    model.total_collisions += 1
    
    for node in collided_nodes
        delete!(model.ongoing_transmissions, node)
        model.node_states[node] = :backoff
        
        model.retry_counters[node] += 1
        
        queue_len = length(model.packet_queues[node])
        adaptive_retry_limit = if queue_len >= STARVATION_THRESHOLD
            MAX_RETRY_SCALE + 10  # Extra attempts for starved nodes
        else
            MAX_RETRY_SCALE
        end
        
        if model.retry_counters[node] > adaptive_retry_limit
            if !isempty(model.packet_queues[node])
                popfirst!(model.packet_queues[node])  # Drop frame
                model.node_drop_counts[node] += 1
                println("📉 Node $node drops frame after $adaptive_retry_limit retries (queue: $queue_len)")
            end
            model.retry_counters[node] = 0
            model.node_states[node] = :idle
            
            if !isempty(model.packet_queues[node])
                schedule_transmission_attempt!(model, engine, node, time)
            end
        else
            backoff_time = calculate_enhanced_backoff(model, node)
            
            event = create_generic_event(time + backoff_time, node, model.model_id, :backoff_expired)
            safe_schedule_event!(engine, event)
        end
    end
end


function schedule_transmission_attempt!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, time::Float64)
    if model.node_states[node] == :idle && !isempty(model.packet_queues[node])
        queue_len = length(model.packet_queues[node])
        
        if queue_len >= STARVATION_THRESHOLD
            sense_delay = SLOT_TIME / 1000  # Very short delay
        elseif queue_len >= FAIRNESS_THRESHOLD
            sense_delay = SLOT_TIME / 200
        else
            sense_delay = SLOT_TIME / 100
        end
        
        node_jitter = (node - 1) * SLOT_TIME * 0.001
        random_jitter = rand() * SLOT_TIME * 0.01
        
        total_delay = sense_delay + node_jitter + random_jitter
        event = create_generic_event(time + total_delay, node, model.model_id, :transmission_attempt)
        safe_schedule_event!(engine, event)
    end
end


function start_transmission!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, time::Float64)
    if model.node_states[node] != :idle || isempty(model.packet_queues[node])
        return
    end
    
    collided_nodes = Int[]
    for (other_node, start_time) in model.ongoing_transmissions
        if other_node != node && abs(time - start_time) < 2 * PROP_DELAY
            push!(collided_nodes, other_node)
        end
    end
    
    if !isempty(collided_nodes)
        push!(collided_nodes, node)  # Add current node
        handle_collision!(model, engine, collided_nodes, time)
    else
        model.node_states[node] = :transmitting
        model.ongoing_transmissions[node] = time
        
        end_event = create_generic_event(time + TX_TIME, node, model.model_id, :transmission_end)
        safe_schedule_event!(engine, end_event)
    end
end


function complete_transmission!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, time::Float64)
    if model.node_states[node] != :transmitting || !haskey(model.ongoing_transmissions, node)
        return
    end
    
    if !isempty(model.packet_queues[node])
        gen_time = popfirst!(model.packet_queues[node])
        delay = time - gen_time
        push!(model.transmission_delays, delay)
        
        model.successful_transmissions += 1
        model.node_success_counts[node] += 1  # Track per-node successes
        model.retry_counters[node] = 0  # Reset retry counter
        
        
        mark_entity_completed!(engine, node)
        
        if model.successful_transmissions >= MAX_SUCCESSES
            println("🎯 Target reached! $(model.successful_transmissions) successful transmissions")
            DESLibrary.stop_simulation!(engine)
            return
        end
    end
    
    delete!(model.ongoing_transmissions, node)
    model.node_states[node] = :idle
    
    if !isempty(model.packet_queues[node])
        schedule_transmission_attempt!(model, engine, node, time)
    end
end


function process_event!(model::EnhancedCSMACDModel, event::GenericEvent, engine::DESEngine)
    node = event.entity_id
    time = event.time
    event_type = event.event_type
    
    if event_type == :frame_generated
        if total_generated[] < MAX_SUCCESSES
            push!(model.packet_queues[node], time)
            total_generated[] += 1
            model.node_generated_counts[node] += 1  # Track per-node generation
            
            current_queue_length = length(model.packet_queues[node])
            if current_queue_length > model.max_queue_lengths[node]
                model.max_queue_lengths[node] = current_queue_length
            end
            
            increment_arrivals!(engine)
            
            
            if model.node_states[node] == :idle
                schedule_transmission_attempt!(model, engine, node, time)
            end
            
            if total_generated[] < MAX_SUCCESSES
                interarrival = deterministic_exponential()
                next_event = create_generic_event(time + interarrival, node, model.model_id, :frame_generated)
                safe_schedule_event!(engine, next_event)
            end
        end
        
    elseif event_type == :transmission_attempt
        start_transmission!(model, engine, node, time)
        
    elseif event_type == :transmission_end
        complete_transmission!(model, engine, node, time)
        
    elseif event_type == :backoff_expired
        model.node_states[node] = :idle
        if !isempty(model.packet_queues[node])
            schedule_transmission_attempt!(model, engine, node, time)
        end
    end
end


function finalize_model!(model::EnhancedCSMACDModel, engine::DESEngine)
end


function get_statistics(model::EnhancedCSMACDModel, engine::DESEngine)
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
        :total_generated => total_generated[],
        :frames_queued => sum(queue_lengths),
        :throughput => throughput,
        :collision_rate => collision_rate,
        :average_delay => avg_delay,
        :total_drops => total_drops,
        :drop_rate => drop_rate,
        :fairness_index => fairness_index,
        :max_queue_length => max_queue,
        :min_queue_length => min_queue,
        :avg_queue_length => avg_queue,
        :node_success_counts => copy(model.node_success_counts),
        :node_drop_counts => copy(model.node_drop_counts),
        :node_generated_counts => copy(model.node_generated_counts)
    )
end

println("\n🚀 Starting simulation...")

engine = DESEngine(MAX_ENTITIES, SEED, WARMUP_FRACTION)
model = EnhancedCSMACDModel(NUM_NODES)
add_model!(engine, :csmacd, model)

initialize_model!(model, engine)
start_time = time()

engine.entities_processed = 0
engine.entities_arrived = 0
engine.stop_simulation = false

last_monitor_time = 0.0
while !engine.stop_simulation
    event = DESLibrary.get_next_event!(engine.event_queue)
    
    if event === nothing
        all_idle = all(state == :idle for state in model.node_states)
        total_queued = sum(length.(model.packet_queues))
        
        if all_idle && total_queued == 0
            println("✅ All nodes idle with empty queues - simulation complete")
            break
        elseif all_idle && total_queued > 0
            for node in 1:model.num_nodes
                if !isempty(model.packet_queues[node])
                    current_time = DESLibrary.get_current_time(engine)
                    schedule_transmission_attempt!(model, engine, node, current_time)
                end
            end
        else
            println("⚠️  No events but nodes not all idle - terminating")
            break
        end
        continue
    end
    
    if haskey(engine.models, event.model_id)
        process_event!(engine.models[event.model_id], event, engine)
    end
    
    current_time = DESLibrary.get_current_time(engine)
    if current_time - last_monitor_time >= 2.0
        global last_monitor_time = current_time
        total_queued = sum(length.(model.packet_queues))
        
        if total_queued > 0
            queue_status = [length(q) for q in model.packet_queues]
            println("📊 Time $(round(current_time, digits=1)): Completed $(model.successful_transmissions), Queued $total_queued, Queues: $queue_status")
        end
        
        if current_time > 5000.0
            println("⏰ Safety timeout reached")
            break
        end
    end
end

end_time = time()
stats = get_statistics(model, engine)

println("\n⏱️  SIMULATION RESULTS")
println("=" ^ 50)
println("Wall-clock time: $(round((end_time - start_time) * 1000, digits=2)) ms")
println("Simulation time: $(round(stats[:simulation_time], digits=2)) time units")
println("Successful transmissions: $(stats[:successful_transmissions])")
println("Total collisions: $(stats[:total_collisions])")
println("Total generated frames: $(stats[:total_generated])")
println("Total dropped frames: $(stats[:total_drops])")
println("Frames still queued: $(stats[:frames_queued])")
println("Frame loss rate: $(round(stats[:total_drops] / stats[:total_generated] * 100, digits=3))%")

println("\n📊 PERFORMANCE METRICS")
println("=" ^ 50)
println("Throughput: $(round(stats[:throughput], digits=4)) frames/time unit")
println("Collision rate: $(round(stats[:collision_rate], digits=4)) collisions/time unit")
println("Drop rate: $(round(stats[:drop_rate], digits=4)) drops/time unit")
println("Average delay: $(round(stats[:average_delay], digits=6)) time units")
println("Fairness index: $(round(stats[:fairness_index], digits=4)) (1.0 = perfect fairness)")

println("\n📈 QUEUE DISTRIBUTION")
println("=" ^ 50)

println("\n📦 PER-NODE MAXIMUM QUEUE LENGTHS")
println("-" ^ 40)
for i in 1:model.num_nodes
    max_q = model.max_queue_lengths[i]
    println("Node $i: $max_q frames")
end

println("\n🎯 PER-NODE STATISTICS")
println("=" ^ 80)
println("Node | Generated | Successes | Drops | Max Queue | Success Rate")
println("-" ^ 80)
for i in 1:model.num_nodes
    generated = stats[:node_generated_counts][i]
    successes = stats[:node_success_counts][i]
    drops = stats[:node_drop_counts][i]
    max_queue = model.max_queue_lengths[i]
    success_rate = generated > 0 ? (successes / generated) * 100 : 0.0
    
    println("$(lpad(i, 4)) | $(lpad(generated, 9)) | $(lpad(successes, 9)) | $(lpad(drops, 5)) | $(lpad(max_queue, 9)) | $(lpad(round(success_rate, digits=3), 11))%")
end

total_generated = sum(stats[:node_generated_counts])
total_successes = sum(stats[:node_success_counts])
total_drops = sum(stats[:node_drop_counts])
total_queued = sum(length(queue) for queue in model.packet_queues)

println("\n📊 FRAME ACCOUNTING SUMMARY")
println("-" ^ 40)
println("Total frames generated: $total_generated")
println("Total frames succeeded: $total_successes")
println("Total frames dropped:   $total_drops")
println("Total frames queued:    $total_queued")
println("Frames processed:       $(total_successes + total_drops)")
println("Overall success rate:   $(round((total_successes/total_generated)*100, digits=3))%")

if stats[:frames_queued] > 0
    println("\n📦 REMAINING FRAMES BY NODE")
    for (i, queue) in enumerate(model.packet_queues)
        if !isempty(queue)
            println("Node $i: $(length(queue)) frames")
        end
    end
end

println("\n✅ DES Julia CSMA/CD simulation completed!")