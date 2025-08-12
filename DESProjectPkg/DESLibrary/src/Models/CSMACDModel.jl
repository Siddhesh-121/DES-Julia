using Random

const MAX_RETRY = 16   # Standard value for Ethernet CSMA/CD
const EPSILON = 1e-7
const CARRIER_SENSING_DELAY = EPSILON / 10  # Smaller carrier sensing delay to match SimPy

mutable struct CSMACDModel <: SimulationModel
    num_nodes::Int
    frame_generation_rate::Float64
    transmission_time::Float64
    propagation_delay::Float64
    slot_time::Float64
    model_id::Symbol

    # State
    node_states::Vector{Symbol}         # :idle, :transmitting, :backoff
    backoff_timers::Vector{Float64}
    retry_counters::Vector{Int}
    transmission_start_times::Dict{Int, Float64}  # node_id => start_time (allows multiple concurrent)
    transmission_end_times::Dict{Int, Float64}    # node_id => end_time
    event_log::Vector{Dict{Symbol, Any}}
    packet_queues::Vector{Vector{Float64}}  # FIFO queue of frame generation times
    transmission_delays::Vector{Float64}
    retry_counts_per_success::Vector{Int}
    drop_counts::Vector{Int}
    collisions::Int
    successful_transmissions::Int
    rng::AbstractRNG
    total_generated_frames::Int
end

function CSMACDModel(num_nodes, frame_generation_rate, transmission_time, propagation_delay, slot_time, model_id=:csma_cd; rng=MersenneTwister(2024))
    node_states = fill(:idle, num_nodes)
    backoff_timers = zeros(Float64, num_nodes)
    retry_counters = zeros(Int, num_nodes)
    transmission_start_times = Dict{Int, Float64}()
    transmission_end_times = Dict{Int, Float64}()
    event_log = Vector{Dict{Symbol, Any}}()
    packet_queues = [Vector{Float64}() for _ in 1:num_nodes]
    transmission_delays = Float64[]
    retry_counts_per_success = Int[]
    drop_counts = zeros(Int, num_nodes)
    collisions = 0
    successful_transmissions = 0
    total_generated_frames = 0
    return CSMACDModel(num_nodes, frame_generation_rate, transmission_time, propagation_delay, slot_time, model_id, node_states, backoff_timers, retry_counters, transmission_start_times, transmission_end_times, event_log, packet_queues, transmission_delays, retry_counts_per_success, drop_counts, collisions, successful_transmissions, rng, total_generated_frames)
end

# Helper to prevent scheduling events beyond simulation end time or at the current time
function safe_schedule_event!(engine, event)
    now = get_current_time(engine)
    if event.time > now && event.time < engine.simulation_end_time
        schedule_event!(engine, event)
    end
end

function initialize_model!(model::CSMACDModel, engine::DESEngine)
    for node in 1:model.num_nodes
        t = rand(model.rng, Exponential(1.0 / model.frame_generation_rate))
        event = create_generic_event(t, node, model.model_id, :frame_generated)
        safe_schedule_event!(engine, event)
    end
    model.node_states .= :idle
    model.backoff_timers .= 0.0
    model.retry_counters .= 0
    model.drop_counts .= 0
    empty!(model.event_log)
    empty!(model.transmission_start_times)
    empty!(model.transmission_end_times)
    empty!(model.transmission_delays)
    empty!(model.retry_counts_per_success)
    for q in model.packet_queues
        empty!(q)
    end
    model.collisions = 0
    model.successful_transmissions = 0
    model.total_generated_frames = 0
end

function _handle_collision!(model::CSMACDModel, nodes, t, engine)
    unique_nodes = unique(nodes)
    
    # Count ONE collision per collision event (not per node like before)
    model.collisions += 1
    
    for n in unique_nodes
        model.retry_counters[n] += 1
        
        if model.retry_counters[n] > MAX_RETRY
            # Drop packet after max retries
            if !isempty(model.packet_queues[n])
                popfirst!(model.packet_queues[n])
            end
            model.drop_counts[n] += 1
            model.retry_counters[n] = 0
        else
            # Apply backoff with same algorithm as SimPy
            _apply_backoff!(model, n, t, engine)
        end
        
        # Clean up transmission state
        delete!(model.transmission_start_times, n)
        delete!(model.transmission_end_times, n)
        model.node_states[n] = :idle
        
        # Try to transmit next packet if available after backoff/drop
        if !isempty(model.packet_queues[n])
            if model.retry_counters[n] == 0  # Just dropped a packet
                event_attempt = create_generic_event(t + EPSILON, n, model.model_id, :transmission_attempt)
                safe_schedule_event!(engine, event_attempt)
            end
        end
    end
end

function _apply_backoff!(model::CSMACDModel, node::Int, t::Float64, engine::DESEngine)
    # Exact same binary exponential backoff as SimPy
    k = min(model.retry_counters[node], 8)
    r = rand(model.rng, 0:2^k-1)
    backoff = r * (model.slot_time * 0.75)
    backoff = max(backoff, model.slot_time * 0.5)
    
    model.backoff_timers[node] = t + backoff
    model.node_states[node] = :backoff
    event_backoff = create_generic_event(t + backoff, node, model.model_id, :backoff_expired)
    safe_schedule_event!(engine, event_backoff)
end

function _start_transmission!(model::CSMACDModel, node::Int, t::Float64, engine::DESEngine)
    # Match SimPy logic exactly: check for collisions first, then start transmission
    if model.node_states[node] == :idle
        # Check for collision before starting (exactly like SimPy's carrier sense)
        colliding_nodes = Int[]
        for (other, start) in model.transmission_start_times
            if other != node && abs(t - start) < 2 * model.propagation_delay
                push!(colliding_nodes, other)
            end
        end
        
        if !isempty(colliding_nodes)
            # Collision detected - handle it and don't start transmission
            push!(colliding_nodes, node)
            _handle_collision!(model, colliding_nodes, t, engine)
            return
        end
        
        # No collision - start transmission (like SimPy)
        model.node_states[node] = :transmitting
        model.transmission_start_times[node] = t
        model.transmission_end_times[node] = t + model.transmission_time
        
        # Schedule mid-transmission collision check (like SimPy)
        mid_tx_time = t + model.transmission_time / 2
        event_check = create_generic_event(mid_tx_time, node, model.model_id, :check_collision)
        safe_schedule_event!(engine, event_check)
        
        # Schedule transmission end
        event_end = create_generic_event(t + model.transmission_time, node, model.model_id, :transmission_end)
        safe_schedule_event!(engine, event_end)
    else
        # Node not idle - apply backoff
        _apply_backoff!(model, node, t, engine)
    end
end

function process_event!(model::CSMACDModel, event::GenericEvent, engine::DESEngine)
    if model.successful_transmissions >= 1000
        return
    end
    
    node = event.entity_id
    t = event.time
    etype = event.event_type
    now = get_current_time(engine)

    if etype == :frame_generated
        # Independent Poisson frame generation per node
        push!(model.packet_queues[node], t)
        model.total_generated_frames += 1
        increment_arrivals!(engine)
        
        # Schedule next frame generation
        next_t = t + rand(model.rng, Exponential(1.0 / model.frame_generation_rate))
        if model.successful_transmissions < 1000 && next_t < engine.simulation_end_time
            event_next = create_generic_event(next_t, node, model.model_id, :frame_generated)
            safe_schedule_event!(engine, event_next)
        end
        
        # If node is idle, immediately try to transmit
        if model.node_states[node] == :idle && !isempty(model.packet_queues[node])
            event_attempt = create_generic_event(now + EPSILON, node, model.model_id, :transmission_attempt)
            safe_schedule_event!(engine, event_attempt)
        end

    elseif etype == :transmission_attempt
        if model.node_states[node] == :idle && !isempty(model.packet_queues[node])
            # Schedule carrier sense with small delay (like SimPy)
            event_sense = create_generic_event(t + CARRIER_SENSING_DELAY, node, model.model_id, :carrier_sense)
            safe_schedule_event!(engine, event_sense)
        end

    elseif etype == :carrier_sense
        # Attempt to start transmission (collision detection happens in _start_transmission!)
        _start_transmission!(model, node, t, engine)

    elseif etype == :check_collision
        # Mid-transmission collision check (only if still transmitting)
        if haskey(model.transmission_start_times, node) && model.node_states[node] == :transmitting
            start_time = model.transmission_start_times[node]
            
            # Find overlapping transmissions (original SimPy logic)
            collided_nodes = Int[]
            for (other, other_start) in model.transmission_start_times
                if other != node && abs(other_start - start_time) < 2 * model.propagation_delay
                    push!(collided_nodes, other)
                end
            end
            
            if !isempty(collided_nodes)
                # Only handle collision if we found truly overlapping transmissions
                push!(collided_nodes, node)
                _handle_collision!(model, collided_nodes, t, engine)
            end
        end

    elseif etype == :transmission_end
        # Complete successful transmission (only if still transmitting)
        if haskey(model.transmission_start_times, node) && model.node_states[node] == :transmitting
            if !isempty(model.packet_queues[node])
                gen_time = popfirst!(model.packet_queues[node])
                delay = t - gen_time
                push!(model.transmission_delays, delay)
                push!(model.retry_counts_per_success, model.retry_counters[node])
                model.retry_counters[node] = 0
                model.successful_transmissions += 1
                mark_entity_completed!(engine, node)
            end
            
            # Clean up transmission state
            delete!(model.transmission_start_times, node)
            delete!(model.transmission_end_times, node)
            model.node_states[node] = :idle
            
            # Try to transmit next packet if available
            if !isempty(model.packet_queues[node])
                event_attempt = create_generic_event(now + EPSILON, node, model.model_id, :transmission_attempt)
                safe_schedule_event!(engine, event_attempt)
            end
        end

    elseif etype == :backoff_expired
        model.node_states[node] = :idle
        if !isempty(model.packet_queues[node])
            event_attempt = create_generic_event(now + EPSILON, node, model.model_id, :transmission_attempt)
            safe_schedule_event!(engine, event_attempt)
        end
    end

    # Log the event
    push!(model.event_log, Dict(:time => t, :node => node, :event => etype))
end

function finalize_model!(model::CSMACDModel, engine::DESEngine)
    # No special finalization needed
end

function get_statistics(model::CSMACDModel, engine::DESEngine)
    total_events = length(model.event_log)
    transmissions = model.successful_transmissions
    backoffs = count(e -> e[:event] == :backoff_expired, model.event_log)
    collisions = model.collisions
    sim_time = get_current_time(engine)
    throughput = sim_time > 0 ? transmissions / sim_time : 0.0
    collision_rate = sim_time > 0 ? collisions / sim_time : 0.0
    avg_delay = isempty(model.transmission_delays) ? 0.0 : mean(model.transmission_delays)
    avg_retries = isempty(model.retry_counts_per_success) ? 0.0 : mean(model.retry_counts_per_success)
    total_drops = sum(model.drop_counts)
    
    per_node_stats = Dict{Symbol, Float64}()
    for n in 1:model.num_nodes
        per_node_stats[Symbol("node$(n)_drops")] = Float64(model.drop_counts[n])
    end
    per_node_stats[:total_generated_frames] = Float64(model.total_generated_frames)
    
    # Count frames still in queues
    frames_in_queues = 0
    for q in model.packet_queues
        frames_in_queues += length(q)
    end
    per_node_stats[:frames_in_queues] = Float64(frames_in_queues)
    
    return SimulationResults(
        0.0, 0.0, 0.0, 0.0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, transmissions, sim_time,
        merge(Dict(
            :total_events => Float64(total_events),
            :transmissions => Float64(transmissions),
            :backoffs => Float64(backoffs),
            :collisions => Float64(collisions),
            :throughput => throughput,
            :collision_rate => collision_rate,
            :avg_delay => avg_delay,
            :avg_retries => avg_retries,
            :total_drops => Float64(total_drops)
        ), per_node_stats)
    )
end

function reset_model!(model::CSMACDModel)
    model.node_states .= :idle
    model.backoff_timers .= 0.0
    model.retry_counters .= 0
    model.drop_counts .= 0
    empty!(model.event_log)
    empty!(model.transmission_start_times)
    empty!(model.transmission_end_times)
    empty!(model.transmission_delays)
    empty!(model.retry_counts_per_success)
    for q in model.packet_queues
        empty!(q)
    end
    model.collisions = 0
    model.successful_transmissions = 0
    model.total_generated_frames = 0
end

function reset_statistics!(model::CSMACDModel, engine::DESEngine)
    empty!(model.event_log)
    empty!(model.transmission_start_times)
    empty!(model.transmission_end_times)
    empty!(model.transmission_delays)
    empty!(model.retry_counts_per_success)
    model.drop_counts .= 0
    for q in model.packet_queues
        empty!(q)
    end
    model.collisions = 0
    model.successful_transmissions = 0
    model.total_generated_frames = 0
end

# Required method for simulation engine
function has_entities_in_service(model::CSMACDModel)::Bool
    # Check if any node is currently transmitting
    return !isempty(model.transmission_start_times)
end

# Optional helper method for consistency with MMCModel
function count_entities_in_service(model::CSMACDModel)::Int
    return length(model.transmission_start_times)
end

# --- Extensibility notes ---
# To add capture effect: add a probability in _try_transmit! for one node to succeed in a collision.
# To add multiple channels: make active_transmissions a vector of sets, and assign nodes to channels.
