mutable struct FastRNG
    rng::MersenneTwister
    
    arrival_cache::Vector{Float64}
    arrival_cache_idx::Int
    service_cache::Vector{Float64}
    service_cache_idx::Int
    cache_size::Int
    
    arrival_rate::Float64
    service_rate::Float64
    
    next_entity_id::Int
    
    function FastRNG(seed::Int=2024, cache_size::Int=1000)
        rng = MersenneTwister(seed)
        arrival_cache = Vector{Float64}(undef, cache_size)
        service_cache = Vector{Float64}(undef, cache_size)
        
        new(rng, arrival_cache, cache_size + 1, service_cache, cache_size + 1, 
            cache_size, 4.5, 5.0, 1)
    end
end

@inline function set_arrival_rate!(rng::FastRNG, rate::Float64)
    rng.arrival_rate = rate
end

@inline function set_service_rate!(rng::FastRNG, rate::Float64)
    rng.service_rate = rate
end

@inline function next_arrival_time!(rng::FastRNG)
    if rng.arrival_cache_idx > rng.cache_size
        rand!(rng.rng, rng.arrival_cache)
        for i in 1:rng.cache_size
            rng.arrival_cache[i] = -log(rng.arrival_cache[i]) / rng.arrival_rate
        end
        rng.arrival_cache_idx = 1
    end
    
    val = rng.arrival_cache[rng.arrival_cache_idx]
    rng.arrival_cache_idx += 1
    return val
end

@inline function next_service_time!(rng::FastRNG)
    if rng.service_cache_idx > rng.cache_size
        rand!(rng.rng, rng.service_cache)
        for i in 1:rng.cache_size
            rng.service_cache[i] = -log(rng.service_cache[i]) / rng.service_rate
        end
        rng.service_cache_idx = 1
    end
    
    val = rng.service_cache[rng.service_cache_idx]
    rng.service_cache_idx += 1
    return val
end

@inline function next_entity_id!(rng::FastRNG)
    id = rng.next_entity_id
    rng.next_entity_id += 1
    return id
end

function reset!(rng::FastRNG, seed::Int)
    rng.rng = MersenneTwister(seed)
    rng.arrival_cache_idx = rng.cache_size + 1
    rng.service_cache_idx = rng.cache_size + 1
    rng.next_entity_id = 1
end 