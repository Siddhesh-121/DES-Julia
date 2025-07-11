function process_event!(event::Event, engine::SimulationEngine, rng_manager::RNGManager)
    error("No handler defined for event type $(typeof(event))")
end

function process_event!(event::ArrivalEvent, engine::SimulationEngine, rng_manager::RNGManager)
    # println("Processing Arrival Event at time $(round(get_current_time(engine.event_queue), digits=2)) for entity $(event.entity_id)")
    record_queue_length!(engine)
    record_arrival_time!(engine, event.entity_id, get_current_time(engine.event_queue))
    if !engine.server_busy
        record_server_state_change!(engine, true, get_current_time(engine.event_queue))
        record_waiting_time!(engine, event.entity_id, get_current_time(engine.event_queue))
        service_time = schedule_departure!(engine.event_queue, rng_manager, event.entity_id)
        record_service_time!(engine, service_time)
    else
        push!(engine.waiting_queue, event.entity_id)
    end
    # println("Waiting queue length after arrival: $(length(engine.waiting_queue))")
    schedule_next_arrival!(engine.event_queue, rng_manager)
end

function process_event!(event::DepartureEvent, engine::SimulationEngine, rng_manager::RNGManager)
    # println("Processing Departure Event at time $(round(get_current_time(engine.event_queue), digits=2)) for entity $(event.entity_id)")
    record_queue_length!(engine)
    
    # Record departure time for time in system calculation
    record_departure_time!(engine, event.entity_id, get_current_time(engine.event_queue))
    
    engine.entities_processed += 1
    if !isempty(engine.waiting_queue)
        next_id = popfirst!(engine.waiting_queue)
        arrival_time = get(engine.entity_arrival_times, next_id, get_current_time(engine.event_queue))
        waiting_time = get_current_time(engine.event_queue) - arrival_time
        # println("Entity $(next_id) was waiting for $(round(waiting_time, digits=2)) time units.")
        record_waiting_time!(engine, next_id, get_current_time(engine.event_queue))
        service_time = schedule_departure!(engine.event_queue, rng_manager, next_id)
        record_service_time!(engine, service_time)
    else
        record_server_state_change!(engine, false, get_current_time(engine.event_queue))
    end
end 