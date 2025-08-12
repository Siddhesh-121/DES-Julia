mutable struct DESEngine
    event_queue::EventQueue
    rng::FastRNG
    
    models::Dict{Symbol, SimulationModel}
    
    max_entities::Int
    entities_processed::Int
    entities_arrived::Int
    simulation_end_time::Float64
    
    warmup_entities::Int
    warmup_completed::Bool
    statistics_start_time::Float64
    
    stop_simulation::Bool
    
    function DESEngine(max_entities::Int=1000, seed::Int=2024, warmup_fraction::Float64=0.1)
        warmup_entities = max(1, Int(round(max_entities * warmup_fraction)))
        new(
            EventQueue(),
            FastRNG(seed),
            Dict{Symbol, SimulationModel}(),
            max_entities,
            0,
            0,
            Inf,
            warmup_entities,
            false,
            0.0,
            false
        )
    end
end

function add_model!(engine::DESEngine, model_id::Symbol, model::SimulationModel)
    engine.models[model_id] = model
end

@inline function is_warmup_complete(engine::DESEngine)
    return engine.warmup_completed
end

@inline function complete_warmup!(engine::DESEngine)
    if !engine.warmup_completed && engine.entities_processed >= engine.warmup_entities
        engine.warmup_completed = true
        engine.statistics_start_time = get_current_time(engine)
        
        for (model_id, model) in engine.models
            try
                reset_statistics!(model, engine)
            catch e
                if isa(e, MethodError) && e.f === reset_statistics!
                    @warn "reset_statistics! not defined for model type $(typeof(model)). Statistics may be inaccurate."
                else
                    rethrow(e)
                end
            end
        end
    end
end

function all_entities_complete(engine::DESEngine)
    if engine.entities_processed < engine.max_entities
        return false
    end

    for (model_id, model) in engine.models
        if hasfield(typeof(model), :queue)
            if !isempty(getfield(model, :queue)) || has_entities_in_service(model)
                return false
            end
        elseif hasfield(typeof(model), :packet_queues)
            # For CSMACDModel: all per-node queues must be empty and no node transmitting
            if any(!isempty(q) for q in getfield(model, :packet_queues)) || has_entities_in_service(model)
                return false
            end
        else
            # Fallback: just check if any entities are in service
            if has_entities_in_service(model)
                return false
            end
        end
    end

    return true
end

function should_generate_more_arrivals(engine::DESEngine)
    return engine.entities_arrived < engine.max_entities
end

function simulate!(engine::DESEngine; verbose::Bool=false)
    for (model_id, model) in engine.models
        initialize_model!(model, engine)
    end
    
    engine.entities_processed = 0
    engine.entities_arrived = 0
    engine.stop_simulation = false
    engine.warmup_completed = false
    engine.statistics_start_time = 0.0
    
    events_processed = 0
    
    if verbose
        println("Starting simulation with $(engine.max_entities) entities...")
        println("Warmup period: $(engine.warmup_entities) entities ($(round(engine.warmup_entities/engine.max_entities*100, digits=1))%)")
    end
    
    while !engine.stop_simulation && !is_empty(engine.event_queue)
        event = get_next_event!(engine.event_queue)
        
        if event === nothing
            break
        end
        
        if event.time >= engine.simulation_end_time
            break
        end
        
        model_id = get_model_id(event)
        if haskey(engine.models, model_id)
            model = engine.models[model_id]
            process_event!(model, event, engine)
        else
            error("Unknown model: $model_id")
        end
        
        complete_warmup!(engine)
        
        events_processed += 1
        
        if verbose && events_processed % 1000 == 0
            warmup_status = engine.warmup_completed ? "✅" : "⏳"
            arrivals_status = engine.entities_arrived >= engine.max_entities ? "✅" : "⏳"
            print("Events: $events_processed, Arrived: $(engine.entities_arrived), Completed: $(engine.entities_processed), Warmup: $warmup_status, Arrivals: $arrivals_status\r")
            flush(stdout)
        end
        
        if all_entities_complete(engine)
            if verbose
                println("\n✅ All $(engine.max_entities) entities completed!")
            end
            break
        end
    end
    
    for (model_id, model) in engine.models
        finalize_model!(model, engine)
    end
    
    if verbose
        println("\nSimulation completed. Processed $(engine.entities_processed) entities in $events_processed events.")
        println("Warmup period: $(engine.warmup_entities) entities, Statistics collection: $(engine.max_entities - engine.warmup_entities) entities")
    end
end

@inline function schedule_event!(engine::DESEngine, event::Event)
    schedule_event!(engine.event_queue, event)
end

@inline function increment_entities!(engine::DESEngine)
    engine.entities_processed += 1
end

# Alias for models to mark entity completion
@inline function mark_entity_completed!(engine::DESEngine, _)
    increment_entities!(engine)
end

@inline function increment_arrivals!(engine::DESEngine)
    engine.entities_arrived += 1
end

@inline function get_current_time(engine::DESEngine)
    get_current_time(engine.event_queue)
end

@inline function stop_simulation!(engine::DESEngine)
    engine.stop_simulation = true
end

function set_end_time!(engine::DESEngine, end_time::Float64)
    engine.simulation_end_time = end_time
end

function get_results(engine::DESEngine)
    results = Dict{Symbol, Any}()
    for (model_id, model) in engine.models
        results[model_id] = get_statistics(model, engine)
    end
    return results
end

function reset!(engine::DESEngine, seed::Int=2024)
    clear_queue!(engine.event_queue)
    reset!(engine.rng, seed)
    engine.entities_processed = 0
    engine.entities_arrived = 0
    engine.warmup_completed = false
    engine.statistics_start_time = 0.0
    engine.stop_simulation = false
    
    for (model_id, model) in engine.models
        reset_model!(model)
    end
end 