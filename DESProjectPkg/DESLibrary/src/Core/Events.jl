struct GenericEvent{P}
    time::Float64
    entity_id::Int
    model_id::Symbol
    event_type::Symbol
    payload::P
end

const DefaultEvent = GenericEvent{Nothing}

Base.isless(e1::GenericEvent, e2::GenericEvent) = e1.time < e2.time

@inline function create_arrival(entity_id::Int, model_id::Symbol, time::Float64)
    GenericEvent{Nothing}(time, entity_id, model_id, :arrival, nothing)
end

@inline function create_departure(entity_id::Int, model_id::Symbol, time::Float64)
    GenericEvent{Nothing}(time, entity_id, model_id, :departure, nothing)
end

@inline function create_service_completion(entity_id::Int, model_id::Symbol, time::Float64)
    GenericEvent{Nothing}(time, entity_id, model_id, :service_completion, nothing)
end

@inline function create_generic_event(time::Float64, entity_id::Int, model_id::Symbol, event_type::Symbol)
    GenericEvent{Nothing}(time, entity_id, model_id, event_type, nothing)
end

@inline get_time(event::GenericEvent) = event.time
@inline get_entity_id(event::GenericEvent) = event.entity_id
@inline get_model_id(event::GenericEvent) = event.model_id