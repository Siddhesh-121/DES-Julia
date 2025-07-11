using DataStructures

abstract type Event end

struct ArrivalEvent <: Event
    entity_id::Int
end

struct DepartureEvent <: Event
    entity_id::Int
end

mutable struct EventQueue
    events::BinaryMinHeap{Tuple{Float64, Int, Event}}
    current_time::Float64
    sequence::Int
end

function EventQueue()
    EventQueue(BinaryMinHeap{Tuple{Float64, Int, Event}}(), 0.0, 0)
end

function schedule_event!(queue::EventQueue, event::Event, time::Float64)
    queue.sequence += 1
    push!(queue.events, (time, queue.sequence, event))
end

function get_next_event!(queue::EventQueue)
    if isempty(queue.events)
        return nothing
    end
    time, _, event = pop!(queue.events)
    queue.current_time = time
    return event
end

function get_current_time(queue::EventQueue)
    return queue.current_time
end 