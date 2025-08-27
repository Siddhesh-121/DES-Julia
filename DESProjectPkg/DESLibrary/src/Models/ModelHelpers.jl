function initialize_model!(model::QueueModel, engine::DESEngine)
    error("initialize_model! must be implemented for $(typeof(model))")
end

function process_event!(model::QueueModel, event::GenericEvent, engine::DESEngine)
    error("process_event! must be implemented for $(typeof(model))")
end

function finalize_model!(model::QueueModel, engine::DESEngine)
    error("finalize_model! must be implemented for $(typeof(model))")
end

function get_statistics(model::QueueModel, engine::DESEngine)::SimulationResults
    error("get_statistics must be implemented for $(typeof(model))")
end

function reset_model!(model::QueueModel)
    error("reset_model! must be implemented for $(typeof(model))")
end

function get_theoretical_results(model::QueueModel)::SimulationResults
    error("get_theoretical_results not implemented for $(typeof(model))")
end

function validate_model(model::QueueModel)::Bool
    return true
end

@inline function schedule_arrival!(engine::DESEngine, entity_id::Int, model_id::Symbol, time::Float64)
    event = create_arrival(entity_id, model_id, time)
    schedule_event!(engine, event)
end

@inline function schedule_departure!(engine::DESEngine, entity_id::Int, model_id::Symbol, time::Float64)
    event = create_departure(entity_id, model_id, time)
    schedule_event!(engine, event)
end

@inline function schedule_service_completion!(engine::DESEngine, entity_id::Int, model_id::Symbol, time::Float64)
    event = create_service_completion(entity_id, model_id, time)
    schedule_event!(engine, event)
end 