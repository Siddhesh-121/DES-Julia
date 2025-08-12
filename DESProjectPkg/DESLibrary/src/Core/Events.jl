abstract type Event end

struct ArrivalEvent <: Event
    entity_id::Int
    model_id::Symbol
    time::Float64
end

struct DepartureEvent <: Event
    entity_id::Int
    model_id::Symbol
    time::Float64
end

struct ServiceCompletionEvent <: Event
    entity_id::Int
    model_id::Symbol
    time::Float64
end

struct GenericEvent <: Event
    time::Float64
    entity_id::Int
    model_id::Symbol
    event_type::Symbol
    data::Dict{Symbol, Any}
end

Base.isless(e1::Event, e2::Event) = e1.time < e2.time

@inline function create_arrival(entity_id::Int, model_id::Symbol, time::Float64)
    ArrivalEvent(entity_id, model_id, time)
end

@inline function create_departure(entity_id::Int, model_id::Symbol, time::Float64)
    DepartureEvent(entity_id, model_id, time)
end

@inline function create_service_completion(entity_id::Int, model_id::Symbol, time::Float64)
    ServiceCompletionEvent(entity_id, model_id, time)
end

@inline function create_generic_event(time::Float64, entity_id::Int, model_id::Symbol, event_type::Symbol, data::Dict{Symbol, Any}=Dict{Symbol, Any}())
    GenericEvent(time, entity_id, model_id, event_type, data)
end

@inline get_time(event::Event) = event.time
@inline get_entity_id(event::Event) = event.entity_id
@inline get_model_id(event::Event) = event.model_id 