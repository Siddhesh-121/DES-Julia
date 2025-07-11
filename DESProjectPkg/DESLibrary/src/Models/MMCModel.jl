using SpecialFunctions

mutable struct MMCModel <: QueueModel
    λ::Float64
    μ::Float64
    c::Int
    model_id::Symbol
    
    queue::Vector{Int}
    servers_busy::Vector{Bool}
    current_customers::Vector{Int}
    
    next_server_index::Int
    busy_servers::Set{Int}
    
    arrival_times::Dict{Int, Float64}
    waiting_times::Vector{Float64}
    service_times::Vector{Float64}
    
    total_waiting_time::Float64
    total_service_time::Float64
    server_busy_times::Vector{Float64}
    last_event_time::Float64
    max_queue_length::Int
    
    server_busy_times_stats::Vector{Float64}
    
    queue_length_buffer::Vector{Int}
    queue_length_times_buffer::Vector{Float64}
    buffer_size::Int
    buffer_index::Int
    buffer_wrapped::Bool
    
    cumulative_queue_area::Float64
    cumulative_queue_area_stats::Float64
    
    total_arrivals::Int
    total_departures::Int
    
    function MMCModel(λ::Float64, μ::Float64, c::Int, model_id::Symbol=:mmc; buffer_size::Int=10000)
        if c <= 0
            throw(ArgumentError("Number of servers (c) must be positive, got: $c"))
        end
        if λ <= 0
            throw(ArgumentError("Arrival rate (λ) must be positive, got: $λ"))
        end
        if μ <= 0
            throw(ArgumentError("Service rate (μ) must be positive, got: $μ"))
        end
        if c > 1000
            @warn "Large number of servers (c=$c) may impact performance"
        end
        if buffer_size <= 0
            throw(ArgumentError("Buffer size must be positive, got: $buffer_size"))
        end
        
        new(
            λ, μ, c, model_id,
            Int[], 
            fill(false, c),
            fill(0, c),
            1,
            Set{Int}(),
            Dict{Int, Float64}(),
            Vector{Float64}(),
            Vector{Float64}(),
            0.0, 0.0,
            fill(0.0, c),
            0.0, 0,
            fill(0.0, c),
            Vector{Int}(undef, buffer_size),
            Vector{Float64}(undef, buffer_size),
            buffer_size,
            1,
            false,
            0.0, 0.0,
            0, 0
        )
    end
end

MM1Model_via_MMC(λ::Float64, μ::Float64, model_id::Symbol=:mm1) = MMCModel(λ, μ, 1, model_id)
MM2Model(λ::Float64, μ::Float64, model_id::Symbol=:mm2) = MMCModel(λ, μ, 2, model_id)
MM3Model(λ::Float64, μ::Float64, model_id::Symbol=:mm3) = MMCModel(λ, μ, 3, model_id)
MM4Model(λ::Float64, μ::Float64, model_id::Symbol=:mm4) = MMCModel(λ, μ, 4, model_id)
MM5Model(λ::Float64, μ::Float64, model_id::Symbol=:mm5) = MMCModel(λ, μ, 5, model_id)

@inline function find_available_server(model::MMCModel)::Union{Int, Nothing}
    for i in 0:(model.c-1)
        server_idx = ((model.next_server_index - 1 + i) % model.c) + 1
        if !model.servers_busy[server_idx]
            model.next_server_index = (server_idx % model.c) + 1
            return server_idx
        end
    end
    return nothing
end

@inline function find_server_for_customer(model::MMCModel, entity_id::Int)::Union{Int, Nothing}
    for server_idx in model.busy_servers
        if model.current_customers[server_idx] == entity_id
            return server_idx
        end
    end
    return nothing
end

@inline function record_queue_length!(model::MMCModel, queue_length::Int, time::Float64)
    model.queue_length_buffer[model.buffer_index] = queue_length
    model.queue_length_times_buffer[model.buffer_index] = time
    
    model.buffer_index += 1
    if model.buffer_index > model.buffer_size
        model.buffer_index = 1
        model.buffer_wrapped = true
    end
end

@inline function update_busy_times!(model::MMCModel, current_time::Float64, warmup_complete::Bool)
    if !isempty(model.busy_servers)
        time_diff = current_time - model.last_event_time
        for server_idx in model.busy_servers
            model.server_busy_times[server_idx] += time_diff
            if warmup_complete
                model.server_busy_times_stats[server_idx] += time_diff
            end
        end
    end
end

@inline function update_queue_area!(model::MMCModel, current_time::Float64, warmup_complete::Bool)
    if current_time > model.last_event_time
        time_duration = current_time - model.last_event_time
        current_queue_size = length(model.queue)
        model.cumulative_queue_area += current_queue_size * time_duration
        if warmup_complete
            model.cumulative_queue_area_stats += current_queue_size * time_duration
        end
    end
end

function initialize_model!(model::MMCModel, engine::DESEngine)
    set_arrival_rate!(engine.rng, model.λ)
    set_service_rate!(engine.rng, model.μ)
    
    first_arrival_time = next_arrival_time!(engine.rng)
    first_entity_id = next_entity_id!(engine.rng)
    schedule_arrival!(engine, first_entity_id, model.model_id, first_arrival_time)
    
    model.last_event_time = 0.0
    model.cumulative_queue_area = 0.0
    model.cumulative_queue_area_stats = 0.0
    
    model.buffer_index = 1
    model.buffer_wrapped = false
    record_queue_length!(model, 0, 0.0)
    
    model.total_arrivals = 0
    model.total_departures = 0
end

function process_event!(model::MMCModel, event::ArrivalEvent, engine::DESEngine)
    current_time = get_current_time(engine)
    entity_id = get_entity_id(event)
    
    model.total_arrivals += 1
    
    increment_arrivals!(engine)
    
    model.arrival_times[entity_id] = current_time
    
    update_busy_times!(model, current_time, is_warmup_complete(engine))
    
    update_queue_area!(model, current_time, is_warmup_complete(engine))
    
    available_server = find_available_server(model)
    
    if available_server !== nothing
        model.servers_busy[available_server] = true
        model.current_customers[available_server] = entity_id
        push!(model.busy_servers, available_server)
        
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
    
    record_queue_length!(model, length(model.queue), current_time)
    
    if should_generate_more_arrivals(engine)
        next_arrival_time = current_time + next_arrival_time!(engine.rng)
        next_entity_id = next_entity_id!(engine.rng)
        schedule_arrival!(engine, next_entity_id, model.model_id, next_arrival_time)
    end
    
    model.last_event_time = current_time
end

function process_event!(model::MMCModel, event::DepartureEvent, engine::DESEngine)
    current_time = get_current_time(engine)
    entity_id = get_entity_id(event)
    
    model.total_departures += 1
    
    update_busy_times!(model, current_time, is_warmup_complete(engine))
    
    update_queue_area!(model, current_time, is_warmup_complete(engine))
    
    increment_entities!(engine)
    
    server_index = find_server_for_customer(model, entity_id)
    if server_index === nothing
        error("Departing entity $entity_id not found on any server")
    end
    
    model.servers_busy[server_index] = false
    model.current_customers[server_index] = 0
    delete!(model.busy_servers, server_index)
    
    if !isempty(model.queue)
        next_customer = popfirst!(model.queue)
        
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
        
        model.servers_busy[server_index] = true
        model.current_customers[server_index] = next_customer
        push!(model.busy_servers, server_index)
        
        schedule_departure!(engine, next_customer, model.model_id, current_time + service_time)
    end
    
    record_queue_length!(model, length(model.queue), current_time)
    
    delete!(model.arrival_times, entity_id)
    
    model.last_event_time = current_time
end

function process_event!(model::MMCModel, event::Event, engine::DESEngine)
    error("MMCModel does not handle event type $(typeof(event))")
end

function finalize_model!(model::MMCModel, engine::DESEngine)
    current_time = get_current_time(engine)
    
    update_busy_times!(model, current_time, is_warmup_complete(engine))
    
    update_queue_area!(model, current_time, is_warmup_complete(engine))
end

function get_statistics(model::MMCModel, engine::DESEngine)::SimulationResults
    current_time = get_current_time(engine)
    
    stats_collection_time = engine.warmup_completed ? 
        (current_time - engine.statistics_start_time) : current_time
    
    n_entities = length(model.waiting_times)
    avg_waiting_time = n_entities > 0 ? mean(model.waiting_times) : 0.0
    avg_service_time = length(model.service_times) > 0 ? mean(model.service_times) : 0.0
    avg_time_in_system = avg_waiting_time + avg_service_time
    
    if engine.warmup_completed && stats_collection_time > 0
        total_busy_time_stats = sum(model.server_busy_times_stats)
        server_utilization = total_busy_time_stats / (model.c * stats_collection_time)
    else
        total_busy_time = sum(model.server_busy_times)
        server_utilization = current_time > 0 ? total_busy_time / (model.c * current_time) : 0.0
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
    
    extra_metrics = Dict{Symbol, Float64}()
    
    for i in 1:model.c
        if engine.warmup_completed && stats_collection_time > 0
            util = model.server_busy_times_stats[i] / stats_collection_time
        else
            util = current_time > 0 ? model.server_busy_times[i] / current_time : 0.0
        end
        extra_metrics[Symbol("server$(i)_utilization")] = util
    end
    
    extra_metrics[:num_servers] = Float64(model.c)
    extra_metrics[:total_utilization] = server_utilization * model.c
    extra_metrics[:traffic_intensity] = model.λ / model.μ
    extra_metrics[:traffic_intensity_per_server] = model.λ / (model.c * model.μ)
    
    extra_metrics[:total_arrivals] = Float64(model.total_arrivals)
    extra_metrics[:total_departures] = Float64(model.total_departures)
    extra_metrics[:busy_servers_count] = Float64(length(model.busy_servers))
    
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
        stats_collection_time,
        extra_metrics
    )
end

function reset_statistics!(model::MMCModel, engine::DESEngine)
    empty!(model.waiting_times)
    empty!(model.service_times)
    
    model.total_waiting_time = 0.0
    model.total_service_time = 0.0
    fill!(model.server_busy_times_stats, 0.0)
    model.max_queue_length = 0
    
    model.cumulative_queue_area = 0.0
    model.cumulative_queue_area_stats = 0.0
    
    model.buffer_index = 1
    model.buffer_wrapped = false
    
    current_time = get_current_time(engine)
    model.last_event_time = current_time
    record_queue_length!(model, length(model.queue), current_time)
end

function reset_model!(model::MMCModel)
    empty!(model.queue)
    fill!(model.servers_busy, false)
    fill!(model.current_customers, 0)
    empty!(model.busy_servers)
    model.next_server_index = 1
    
    empty!(model.arrival_times)
    empty!(model.waiting_times)
    empty!(model.service_times)
    
    model.total_waiting_time = 0.0
    model.total_service_time = 0.0
    fill!(model.server_busy_times, 0.0)
    fill!(model.server_busy_times_stats, 0.0)
    model.last_event_time = 0.0
    model.max_queue_length = 0
    model.cumulative_queue_area = 0.0
    model.cumulative_queue_area_stats = 0.0
    
    model.buffer_index = 1
    model.buffer_wrapped = false
    
    model.total_arrivals = 0
    model.total_departures = 0
end

function has_entities_in_service(model::MMCModel)::Bool
    return !isempty(model.busy_servers)
end

function count_entities_in_service(model::MMCModel)::Int
    return length(model.busy_servers)
end

function get_theoretical_results(model::MMCModel)::SimulationResults
    ρ = model.λ / model.μ
    ρ_per_server = ρ / model.c
    
    if ρ_per_server >= 1.0
        return SimulationResults(
            Inf, 1.0/model.μ, Inf, 1.0, Inf, typemax(Int),
            0.0, 0.0, Inf, Inf, 0, 0.0
        )
    end
    
    if model.c == 1
        avg_waiting_time = ρ_per_server / (model.μ * (1 - ρ_per_server))
        avg_queue_length = (ρ_per_server^2) / (1 - ρ_per_server)
    else
        try
            log_sum_terms = Vector{Float64}()
            
            for k in 0:(model.c-1)
                log_term = k * log(ρ) - loggamma(k + 1)
                push!(log_sum_terms, log_term)
            end
            
            log_erlang_term = model.c * log(ρ) - loggamma(model.c + 1) - log(1 - ρ_per_server)
            push!(log_sum_terms, log_erlang_term)
            
            max_log_term = maximum(log_sum_terms)
            sum_exp_terms = sum(exp.(log_sum_terms .- max_log_term))
            P0 = exp(-max_log_term) / sum_exp_terms
            
            erlang_c = exp(log_erlang_term + log(P0))
            
            avg_waiting_time = erlang_c / (model.c * model.μ * (1 - ρ_per_server))
            
            avg_queue_length = erlang_c * ρ_per_server / (1 - ρ_per_server)
            
        catch e
            @warn "Numerical instability in M/M/C calculations for c=$(model.c), using approximation"
            avg_waiting_time = ρ / (model.c * model.μ * (1 - ρ_per_server))
            avg_queue_length = ρ^2 / (model.c * (1 - ρ_per_server))
        end
    end
    
    avg_service_time = 1.0 / model.μ
    avg_time_in_system = avg_waiting_time + avg_service_time
    server_utilization = ρ_per_server
    
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