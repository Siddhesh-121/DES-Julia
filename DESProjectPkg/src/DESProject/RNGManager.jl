using Random
using Distributions

mutable struct RNGManager
    rng::MersenneTwister
    arrival_distribution::Distribution
    service_distribution::Distribution
    next_id::Int
    
    function RNGManager(seed::Int=2024)
        rng = MersenneTwister(seed)
        arrival_dist = Exponential(0.222)
        service_dist = Exponential(0.2)
        new(rng, arrival_dist, service_dist, 1)
    end
end

function set_arrival_distribution!(manager::RNGManager, dist::Distribution)
    manager.arrival_distribution = dist
end

function set_service_distribution!(manager::RNGManager, dist::Distribution)
    manager.service_distribution = dist
end

function next_arrival_time(manager::RNGManager)
    round(rand(manager.rng, manager.arrival_distribution), digits=2)
end

function next_service_time(manager::RNGManager)
    round(rand(manager.rng, manager.service_distribution), digits=2)
end

function schedule_next_arrival!(queue::EventQueue, manager::RNGManager)
    current_time = get_current_time(queue)
    next_time = current_time + next_arrival_time(manager)
    entity_id = manager.next_id
    manager.next_id += 1
    
    event = ArrivalEvent(entity_id)
    schedule_event!(queue, event, next_time)
end

function schedule_departure!(queue::EventQueue, manager::RNGManager, entity_id::Int)
    current_time = get_current_time(queue)
    service_time = next_service_time(manager)
    next_time = current_time + service_time
    
    event = DepartureEvent(entity_id)
    schedule_event!(queue, event, next_time)
    
    return service_time
end

function initialize_simulation!(queue::EventQueue, manager::RNGManager)
    event = ArrivalEvent(manager.next_id)
    manager.next_id += 1
    schedule_event!(queue, event, 0.0)
end 