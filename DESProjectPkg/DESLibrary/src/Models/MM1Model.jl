mutable struct MM1Model <: QueueModel
    λ::Float64
    μ::Float64
    model_id::Symbol
    
    queue::Vector{Int}
    server_busy::Bool
    current_customer::Int
    
    arrival_times::Dict{Int, Float64}
    waiting_times::Vector{Float64}
    service_times::Vector{Float64}
    
    total_waiting_time::Float64
    total_service_time::Float64
    server_busy_time::Float64
    last_event_time::Float64
    max_queue_length::Int
    
    server_busy_time_stats::Float64
    
    queue_length_samples::Vector{Int}
    queue_length_times::Vector{Float64}
    cumulative_queue_area::Float64
    cumulative_queue_area_stats::Float64
    
    function MM1Model(λ::Float64, μ::Float64, model_id::Symbol=:mm1)
        new(
            λ, μ, model_id,
            Int[], false, 0,
            Dict{Int, Float64}(),
            Vector{Float64}(),
            Vector{Float64}(),
            0.0, 0.0, 0.0, 0.0, 0,
            0.0,
            Vector{Int}(),
            Vector{Float64}(),
            0.0,
            0.0
        )
    end
end

function initialize_model!(model::MM1Model, engine::DESEngine)
    set_arrival_rate!(engine.rng, model.λ)
    set_service_rate!(engine.rng, model.μ)
    
    first_arrival_time = next_arrival_time!(engine.rng)
    first_entity_id = next_entity_id!(engine.rng)
    schedule_arrival!(engine, first_entity_id, model.model_id, first_arrival_time)
    
    model.last_event_time = 0.0
    model.cumulative_queue_area = 0.0
    
    push!(model.queue_length_samples, 0)
    push!(model.queue_length_times, 0.0)
end

function process_event!(model::MM1Model, event::ArrivalEvent, engine::DESEngine)
    current_time = get_current_time(engine)
    entity_id = get_entity_id(event)
    
    increment_arrivals!(engine)
    
    model.arrival_times[entity_id] = current_time
    
    if model.server_busy
        model.server_busy_time += (current_time - model.last_event_time)
        if is_warmup_complete(engine)
            model.server_busy_time_stats += (current_time - model.last_event_time)
        end
    end
    
    if !isempty(model.queue_length_times)
        time_duration = current_time - model.last_event_time
        current_queue_size = length(model.queue)
        model.cumulative_queue_area += current_queue_size * time_duration
        if is_warmup_complete(engine)
            model.cumulative_queue_area_stats += current_queue_size * time_duration
        end
    end
    
    if !model.server_busy
        model.server_busy = true
        model.current_customer = entity_id
        
        if is_warmup_complete(engine)
            push!(model.waiting_times, 0.0)
        end
        
        service_time = next_service_time!(engine.rng)
        
        if is_warmup_complete(engine)
            push!(model.service_times, service_time)
        end
        model.total_service_time += service_time
        
        schedule_departure!(engine, entity_id, model.model_id, current_time + service_time)
    else
        push!(model.queue, entity_id)
        model.max_queue_length = max(model.max_queue_length, length(model.queue))
    end
    
    push!(model.queue_length_samples, length(model.queue))
    push!(model.queue_length_times, current_time)
    
    if engine.entities_arrived < engine.max_entities
        next_arrival_time = current_time + next_arrival_time!(engine.rng)
        next_entity_id = next_entity_id!(engine.rng)
        schedule_arrival!(engine, next_entity_id, model.model_id, next_arrival_time)
    end
    
    model.last_event_time = current_time
end

function process_event!(model::MM1Model, event::DepartureEvent, engine::DESEngine)
    current_time = get_current_time(engine)
    entity_id = get_entity_id(event)
    
    model.server_busy_time += (current_time - model.last_event_time)
    
    if is_warmup_complete(engine)
        model.server_busy_time_stats += (current_time - model.last_event_time)
    end
    
    if !isempty(model.queue_length_times)
        time_duration = current_time - model.last_event_time
        current_queue_size = length(model.queue)
        model.cumulative_queue_area += current_queue_size * time_duration
        if is_warmup_complete(engine)
            model.cumulative_queue_area_stats += current_queue_size * time_duration
        end
    end
    
    increment_entities!(engine)
    
    if !isempty(model.queue)
        next_customer = popfirst!(model.queue)
        model.current_customer = next_customer
        
        arrival_time = model.arrival_times[next_customer]
        waiting_time = current_time - arrival_time
        
        if is_warmup_complete(engine)
            push!(model.waiting_times, waiting_time)
        end
        model.total_waiting_time += waiting_time
        
        service_time = next_service_time!(engine.rng)
        
        if is_warmup_complete(engine)
            push!(model.service_times, service_time)
        end
        model.total_service_time += service_time
        
        schedule_departure!(engine, next_customer, model.model_id, current_time + service_time)
    else
        model.server_busy = false
        model.current_customer = 0
    end
    
    push!(model.queue_length_samples, length(model.queue))
    push!(model.queue_length_times, current_time)
    
    delete!(model.arrival_times, entity_id)
    
    model.last_event_time = current_time
end

function process_event!(model::MM1Model, event::Event, engine::DESEngine)
    error("MM1Model does not handle event type $(typeof(event))")
end

function finalize_model!(model::MM1Model, engine::DESEngine)
    current_time = get_current_time(engine)
    
    if model.server_busy
        model.server_busy_time += (current_time - model.last_event_time)
        if is_warmup_complete(engine)
            model.server_busy_time_stats += (current_time - model.last_event_time)
        end
    end
    
    if !isempty(model.queue_length_times) && model.last_event_time < current_time
        time_duration = current_time - model.last_event_time
        current_queue_size = length(model.queue)
        model.cumulative_queue_area += current_queue_size * time_duration
        if is_warmup_complete(engine)
            model.cumulative_queue_area_stats += current_queue_size * time_duration
        end
    end
end

function get_statistics(model::MM1Model, engine::DESEngine)::SimulationResults
    current_time = get_current_time(engine)
    
    stats_collection_time = engine.warmup_completed ? 
        (current_time - engine.statistics_start_time) : current_time
    
    n_entities = length(model.waiting_times)
    avg_waiting_time = n_entities > 0 ? mean(model.waiting_times) : 0.0
    avg_service_time = length(model.service_times) > 0 ? mean(model.service_times) : 0.0
    avg_time_in_system = avg_waiting_time + avg_service_time
    
    server_utilization = if engine.warmup_completed && stats_collection_time > 0
        model.server_busy_time_stats / stats_collection_time
    else
        current_time > 0 ? model.server_busy_time / current_time : 0.0
    end
    
    avg_queue_length = if engine.warmup_completed && stats_collection_time > 0
        model.cumulative_queue_area_stats / stats_collection_time
    else
        current_time > 0 ? model.cumulative_queue_area / current_time : 0.0
    end
    
    waiting_time_std = length(model.waiting_times) > 1 ? std(model.waiting_times) : 0.0
    service_time_std = length(model.service_times) > 1 ? std(model.service_times) : 0.0
    max_waiting_time = length(model.waiting_times) > 0 ? maximum(model.waiting_times) : 0.0
    max_service_time = length(model.service_times) > 0 ? maximum(model.service_times) : 0.0
    
    entities_for_stats = engine.warmup_completed ? 
        max(0, engine.entities_processed - engine.warmup_entities) : 
        engine.entities_processed
    
    return SimulationResults(
        avg_waiting_time,
        avg_service_time,
        avg_time_in_system,
        server_utilization,
        avg_queue_length,
        model.max_queue_length,
        waiting_time_std,
        service_time_std,
        max_waiting_time,
        max_service_time,
        entities_for_stats,
        stats_collection_time
    )
end

function reset_statistics!(model::MM1Model, engine::DESEngine)
    empty!(model.waiting_times)
    empty!(model.service_times)
    
    model.total_waiting_time = 0.0
    model.total_service_time = 0.0
    model.server_busy_time_stats = 0.0
    model.max_queue_length = 0
    
    model.cumulative_queue_area = 0.0
    model.cumulative_queue_area_stats = 0.0
    
    model.last_event_time = get_current_time(engine)
end

function reset_model!(model::MM1Model)
    empty!(model.queue)
    model.server_busy = false
    model.current_customer = 0
    
    empty!(model.arrival_times)
    empty!(model.waiting_times)
    empty!(model.service_times)
    
    model.total_waiting_time = 0.0
    model.total_service_time = 0.0
    model.server_busy_time = 0.0
    model.server_busy_time_stats = 0.0
    model.last_event_time = 0.0
    model.max_queue_length = 0
    model.cumulative_queue_area = 0.0
    model.cumulative_queue_area_stats = 0.0
    
    empty!(model.queue_length_samples)
    empty!(model.queue_length_times)
end

function get_theoretical_results(model::MM1Model)::SimulationResults
    ρ = model.λ / model.μ
    
    if ρ >= 1.0
        return SimulationResults(
            Inf, 1.0/model.μ, Inf, 1.0, Inf, typemax(Int),
            0.0, 0.0, Inf, Inf, 0, 0.0
        )
    end
    
    avg_waiting_time = ρ / (model.μ * (1 - ρ))
    avg_service_time = 1.0 / model.μ
    avg_time_in_system = 1.0 / (model.μ - model.λ)
    server_utilization = ρ
    avg_queue_length = (ρ * ρ) / (1 - ρ)
    
    return SimulationResults(
        avg_waiting_time,
        avg_service_time,
        avg_time_in_system,
        server_utilization,
        avg_queue_length,
        typemax(Int),
        0.0, 0.0, 0.0, 0.0,
        0, 0.0
    )
end

function has_entities_in_service(model::MM1Model)::Bool
    return model.server_busy
end

function count_entities_in_service(model::MM1Model)::Int
    return model.server_busy ? 1 : 0
end 