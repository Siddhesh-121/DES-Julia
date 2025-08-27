mutable struct EventQueue
    heap::BinaryMinHeap{GenericEvent}
    current_time::Float64
    
    function EventQueue()
        new(BinaryMinHeap{GenericEvent}(), 0.0)
    end
end

@inline function schedule_event!(queue::EventQueue, event::GenericEvent)
    push!(queue.heap, event)
    nothing
end

@inline function get_next_event!(queue::EventQueue)
    if isempty(queue.heap)
        return nothing
    end
    
    event = pop!(queue.heap)
    queue.current_time = event.time
    return event
end

@inline function is_empty(queue::EventQueue)
    isempty(queue.heap)
end

@inline function get_current_time(queue::EventQueue)
    queue.current_time
end

@inline function peek_next_time(queue::EventQueue)
    if isempty(queue.heap)
        return Inf
    end
    return top(queue.heap).time
end

@inline function queue_size(queue::EventQueue)
    length(queue.heap)
end

function clear_queue!(queue::EventQueue)
    queue.heap = BinaryMinHeap{Event}()
    queue.current_time = 0.0
end 