mutable struct SimulationEngine
    event_queue::EventQueue
    waiting_queue::Vector{Int}
    server_busy::Bool
    entities_processed::Int
    max_entities::Int
    
    # Time-weighted queue length tracking
    total_queue_length_time::Float64  # Sum of (queue_length * time_duration)
    last_queue_change_time::Float64   # Last time queue length changed
    current_queue_length::Int         # Current queue length
    
    # Legacy simple average (for comparison)
    total_queue_length::Float64
    queue_length_samples::Int
    
    total_waiting_time::Float64
    entity_arrival_times::Dict{Int, Float64}
    server_busy_time::Float64
    last_state_change_time::Float64
    queue_length_history::Vector{Tuple{Float64, Int}}
    waiting_times::Vector{Float64}
    max_waiting_time::Float64
    
    # Time in system tracking
    total_time_in_system::Float64
    time_in_system_history::Vector{Float64}
    max_time_in_system::Float64
    
    # Service time tracking
    total_service_time::Float64
    service_times::Vector{Float64}
    max_service_time::Float64
    
    # Statistics for validation
    total_customers_served::Int
    simulation_start_time::Float64
end

function SimulationEngine(max_entities::Int=1500)
    SimulationEngine(
        EventQueue(), Int[], false, 0, max_entities,
        0.0, 0.0, 0,  # time-weighted fields
        0.0, 0,       # legacy fields
        0.0, Dict{Int, Float64}(), 0.0, 0.0, 
        Tuple{Float64, Int}[], Float64[], 0.0,
        0.0, Float64[], 0.0,
        0.0, Float64[], 0.0,
        0, 0.0        # statistics fields
    )
end

function run!(engine::SimulationEngine, rng_manager::RNGManager)
    while engine.entities_processed < engine.max_entities
        event = get_next_event!(engine.event_queue)
        if event === nothing
            break
        end
        process_event!(event, engine, rng_manager)
        if engine.entities_processed >= engine.max_entities
            println("\nReached maximum number of entities ($(engine.max_entities)). Stopping simulation.")
            break
        end
    end
end

function stop!(engine::SimulationEngine)
    engine.running = false
end

function get_current_time(engine::SimulationEngine)
    return get_current_time(engine.event_queue)
end

function get_entities_processed(engine::SimulationEngine)
    return engine.entities_processed
end

function record_queue_length!(engine::SimulationEngine)
    current_time = get_current_time(engine)
    
    # Update time-weighted queue length
    if engine.last_queue_change_time > 0.0
        time_duration = current_time - engine.last_queue_change_time
        engine.total_queue_length_time += engine.current_queue_length * time_duration
    end
    
    # Update current queue length
    engine.current_queue_length = length(engine.waiting_queue)
    engine.last_queue_change_time = current_time
    
    # Legacy simple average (keep for comparison)
    engine.total_queue_length += engine.current_queue_length
    engine.queue_length_samples += 1
    push!(engine.queue_length_history, (current_time, engine.current_queue_length))
end

function initialize_queue_tracking!(engine::SimulationEngine)
    current_time = get_current_time(engine)
    engine.simulation_start_time = current_time
    engine.last_queue_change_time = current_time
    engine.current_queue_length = length(engine.waiting_queue)
end

function finalize_queue_tracking!(engine::SimulationEngine)
    current_time = get_current_time(engine)
    if engine.last_queue_change_time > 0.0
        time_duration = current_time - engine.last_queue_change_time
        engine.total_queue_length_time += engine.current_queue_length * time_duration
    end
end

function get_time_weighted_average_queue_length(engine::SimulationEngine)
    # Finalize the calculation first
    finalize_queue_tracking!(engine)
    
    total_time = get_current_time(engine) - engine.simulation_start_time
    if total_time <= 0.0
        return 0.0
    end
    
    return engine.total_queue_length_time / total_time
end

function get_average_queue_length(engine::SimulationEngine)
    if engine.queue_length_samples == 0
        return 0.0
    end
    engine.total_queue_length / engine.queue_length_samples
end

function get_simulation_statistics(engine::SimulationEngine)
    current_time = get_current_time(engine)
    total_time = current_time - engine.simulation_start_time
    
    # Calculate effective arrival rate
    arrival_rate = total_time > 0.0 ? engine.entities_processed / total_time : 0.0
    
    # Get time-weighted queue length
    time_weighted_queue_length = get_time_weighted_average_queue_length(engine)
    
    # Get average waiting time
    avg_waiting_time = get_average_waiting_time(engine)
    
    # Validate Little's Law: L = λW
    littles_law_left = time_weighted_queue_length
    littles_law_right = arrival_rate * avg_waiting_time
    littles_law_error = abs(littles_law_left - littles_law_right)
    
    return (
        arrival_rate = arrival_rate,
        time_weighted_queue_length = time_weighted_queue_length,
        simple_average_queue_length = get_average_queue_length(engine),
        avg_waiting_time = avg_waiting_time,
        littles_law_left = littles_law_left,
        littles_law_right = littles_law_right,
        littles_law_error = littles_law_error,
        total_simulation_time = total_time
    )
end

function record_arrival_time!(engine::SimulationEngine, entity_id::Int, time::Float64)
    engine.entity_arrival_times[entity_id] = time
end

function record_waiting_time!(engine::SimulationEngine, entity_id::Int, service_start_time::Float64)
    arrival_time = get(engine.entity_arrival_times, entity_id, service_start_time)
    waiting_time = service_start_time - arrival_time
    engine.total_waiting_time += waiting_time
    push!(engine.waiting_times, waiting_time)
    engine.max_waiting_time = max(engine.max_waiting_time, waiting_time)
end

function get_average_waiting_time(engine::SimulationEngine)
    if engine.entities_processed == 0
        return 0.0
    end
    engine.total_waiting_time / engine.entities_processed
end

function get_max_waiting_time(engine::SimulationEngine)
    return engine.max_waiting_time
end

function get_waiting_time_std(engine::SimulationEngine)
    if length(engine.waiting_times) < 2
        return 0.0
    end
    mean_wait = mean(engine.waiting_times)
    variance = sum((wt - mean_wait)^2 for wt in engine.waiting_times) / (length(engine.waiting_times) - 1)
    return sqrt(variance)
end

function get_queue_length_history(engine::SimulationEngine)
    return engine.queue_length_history
end

function get_waiting_times(engine::SimulationEngine)
    return engine.waiting_times
end

function record_server_state_change!(engine::SimulationEngine, new_busy_state::Bool, current_time::Float64)
    if engine.server_busy != new_busy_state
        if engine.server_busy
            engine.server_busy_time += (current_time - engine.last_state_change_time)
        end
        engine.server_busy = new_busy_state
        engine.last_state_change_time = current_time
    end
end

function get_server_utilization(engine::SimulationEngine)
    current_time = get_current_time(engine)
    if engine.server_busy
        engine.server_busy_time += (current_time - engine.last_state_change_time)
        engine.last_state_change_time = current_time
    end
    
    if current_time == 0.0
        return 0.0
    end
    engine.server_busy_time / current_time
end

function record_departure_time!(engine::SimulationEngine, entity_id::Int, departure_time::Float64)
    arrival_time = get(engine.entity_arrival_times, entity_id, departure_time)
    time_in_system = departure_time - arrival_time
    engine.total_time_in_system += time_in_system
    push!(engine.time_in_system_history, time_in_system)
    engine.max_time_in_system = max(engine.max_time_in_system, time_in_system)
    
    # Clean up arrival time record
    delete!(engine.entity_arrival_times, entity_id)
end

function get_average_time_in_system(engine::SimulationEngine)
    if engine.entities_processed == 0
        return 0.0
    end
    engine.total_time_in_system / engine.entities_processed
end

function get_max_time_in_system(engine::SimulationEngine)
    return engine.max_time_in_system
end

function get_time_in_system_std(engine::SimulationEngine)
    if length(engine.time_in_system_history) < 2
        return 0.0
    end
    mean_time = mean(engine.time_in_system_history)
    variance = sum((t - mean_time)^2 for t in engine.time_in_system_history) / (length(engine.time_in_system_history) - 1)
    return sqrt(variance)
end

function get_time_in_system_history(engine::SimulationEngine)
    return engine.time_in_system_history
end

function record_service_time!(engine::SimulationEngine, service_time::Float64)
    engine.total_service_time += service_time
    push!(engine.service_times, service_time)
    engine.max_service_time = max(engine.max_service_time, service_time)
end

function get_average_service_time(engine::SimulationEngine)
    if engine.entities_processed == 0
        return 0.0
    end
    engine.total_service_time / engine.entities_processed
end

function get_max_service_time(engine::SimulationEngine)
    return engine.max_service_time
end

function get_service_time_std(engine::SimulationEngine)
    if length(engine.service_times) < 2
        return 0.0
    end
    mean_service = mean(engine.service_times)
    variance = sum((st - mean_service)^2 for st in engine.service_times) / (length(engine.service_times) - 1)
    return sqrt(variance)
end

function get_service_times(engine::SimulationEngine)
    return engine.service_times
end

function get_average_customers_in_system(engine::SimulationEngine)
    # Using Little's Law: L = λW where W is time in system
    current_time = get_current_time(engine)
    total_time = current_time - engine.simulation_start_time
    
    if total_time <= 0.0 || engine.entities_processed == 0
        return 0.0
    end
    
    arrival_rate = engine.entities_processed / total_time
    avg_time_in_system = get_average_time_in_system(engine)
    
    return arrival_rate * avg_time_in_system
end 