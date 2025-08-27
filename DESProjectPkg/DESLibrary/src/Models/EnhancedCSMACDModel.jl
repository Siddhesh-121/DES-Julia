using Random
using DataStructures: Deque

@enum NodeState::UInt8 begin
    NODE_IDLE = 0
    NODE_TRANSMITTING = 1
    NODE_BACKOFF = 2
end

"""
    EnhancedCSMACDModel(
        num_nodes::Int;
        model_id::Symbol = :csmacd,
        frame_generation_rate::Float64 = 5.0,
        transmission_time::Float64 = 0.01,
        propagation_delay::Float64 = 0.005,
        slot_time::Float64 = 0.005,
        max_frames::Int = 100000,
        fairness_threshold::Int = 5,
        starvation_threshold::Int = 10,
        max_backoff_exponent::Int = 8,
        rng::AbstractRNG = MersenneTwister(2024)
    ) -> EnhancedCSMACDModel

High-performance CSMA/CD model with fairness-aware exponential backoff, deterministic arrivals/backoff, and O(1) queues.
Returns the initialized model.
"""
mutable struct EnhancedCSMACDModel <: SimulationModel
    num_nodes::Int
    model_id::Symbol

    frame_generation_rate::Float64
    transmission_time::Float64
    propagation_delay::Float64
    slot_time::Float64

    max_frames::Int
    fairness_threshold::Int
    starvation_threshold::Int
    max_backoff_exponent::Int

    node_states::Vector{NodeState}
    packet_queues::Vector{Deque{Float64}}
    ongoing_transmissions::Vector{Float64}
    retry_counters::Vector{Int}

    node_success_counts::Vector{Int}
    node_drop_counts::Vector{Int}
    node_generated_counts::Vector{Int}

    successful_transmissions::Int
    total_generated::Int
    total_collisions::Int
    transmission_delays::Vector{Float64}
    max_queue_lengths::Vector{Int}

    rng::AbstractRNG
    precomp_exp::Vector{Float64}
    precomp_backoff::Vector{Int}
    exp_idx::Int
    backoff_idx::Int
    precomp_jitter::Vector{Float64}
    jitter_idx::Int
end

function EnhancedCSMACDModel(
    num_nodes::Int;
    model_id::Symbol = :csmacd,
    frame_generation_rate::Float64 = 5.0,
    transmission_time::Float64 = 0.01,
    propagation_delay::Float64 = 0.005,
    slot_time::Float64 = 0.005,
    max_frames::Int = 100000,
    fairness_threshold::Int = 5,
    starvation_threshold::Int = 10,
    max_backoff_exponent::Int = 8,
    rng::AbstractRNG = MersenneTwister(2024)
)::EnhancedCSMACDModel
    node_states = fill(NODE_IDLE, num_nodes)
    packet_queues = [Deque{Float64}() for _ in 1:num_nodes]
    ongoing_transmissions = zeros(Float64, num_nodes)
    retry_counters = zeros(Int, num_nodes)
    node_success_counts = zeros(Int, num_nodes)
    node_drop_counts = zeros(Int, num_nodes)
    node_generated_counts = zeros(Int, num_nodes)
    transmission_delays = Float64[]
    max_queue_lengths = zeros(Int, num_nodes)
    precomp_len = 10_000
    precomp_exp = [randexp(rng) / frame_generation_rate for _ in 1:precomp_len]
    precomp_backoff = [rand(rng, 0:255) for _ in 1:precomp_len]
    precomp_jitter = [rand(rng) for _ in 1:precomp_len]
    return EnhancedCSMACDModel(
        num_nodes,
        model_id,
        frame_generation_rate,
        transmission_time,
        propagation_delay,
        slot_time,
        max_frames,
        fairness_threshold,
        starvation_threshold,
        max_backoff_exponent,
        node_states,
        packet_queues,
        ongoing_transmissions,
        retry_counters,
        node_success_counts,
        node_drop_counts,
        node_generated_counts,
        0,
        0,
        0,
        transmission_delays,
        max_queue_lengths,
        rng,
        precomp_exp,
        precomp_backoff,
        1,
        1,
        precomp_jitter,
        1
    )
end

@inline function _det_exp!(model::EnhancedCSMACDModel)::Float64
    i = model.exp_idx
    v = @inbounds model.precomp_exp[i]
    model.exp_idx = i == length(model.precomp_exp) ? 1 : i + 1
    return v
end

@inline function _det_backoff!(model::EnhancedCSMACDModel, max_slots::Int)::Int
    i = model.backoff_idx
    v = @inbounds model.precomp_backoff[i]
    model.backoff_idx = i == length(model.precomp_backoff) ? 1 : i + 1
    return max_slots > 0 ? (v % max_slots) : 0
end

@inline function _det_jitter!(model::EnhancedCSMACDModel)::Float64
    i = model.jitter_idx
    v = @inbounds model.precomp_jitter[i]
    model.jitter_idx = i == length(model.precomp_jitter) ? 1 : i + 1
    return v
end

@inline function _safe_schedule_event!(engine::DESEngine, event::GenericEvent)
    now = get_current_time(engine)
    if event.time >= now
        schedule_event!(engine, event)
    else
        adjusted = create_generic_event(now + 1e-9, event.entity_id, event.model_id, event.event_type)
        schedule_event!(engine, adjusted)
    end
end

"""
    initialize_model!(model::EnhancedCSMACDModel, engine::DESEngine)

Schedule initial frame generation for each node using deterministic interarrivals.
"""
function initialize_model!(model::EnhancedCSMACDModel, engine::DESEngine)
    for node in 1:model.num_nodes
        interarrival = _det_exp!(model)
        ev = create_generic_event(interarrival, node, model.model_id, :frame_generated)
        _safe_schedule_event!(engine, ev)
    end
end

@inline function _fairness_factor(model::EnhancedCSMACDModel, node::Int)::Float64
    qlen = length(@inbounds model.packet_queues[node])
    if qlen >= model.starvation_threshold
        return 0.1
    end
    total = 0
    for q in model.packet_queues
        total += length(q)
    end
    avg = total > 0 ? total / model.num_nodes : 0.0
    if avg > 0
        ratio = qlen / avg
        return max(0.3, min(2.0, 2.0 - ratio))
    end
    return 1.0
end

@inline function _enhanced_backoff(model::EnhancedCSMACDModel, node::Int)::Float64
    retry = @inbounds model.retry_counters[node]
    k = min(retry, model.max_backoff_exponent)
    max_slots = 2^k
    slots = _det_backoff!(model, max_slots)
    base_backoff = slots * model.slot_time
    fairness = _fairness_factor(model, node)
    node_jitter = (node - 1) * model.slot_time * 0.01
    random_jitter = _det_jitter!(model) * model.slot_time * 0.1
    total = base_backoff * fairness + node_jitter + random_jitter
    return max(model.slot_time * 0.1, total)
end

"""
    schedule_transmission_attempt!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, t::Float64)

Schedule a carrier-sense attempt for the node if it is idle and has queued frames.
"""
function schedule_transmission_attempt!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, t::Float64)
    if @inbounds model.node_states[node] == NODE_IDLE && !isempty(@inbounds model.packet_queues[node])
        qlen = length(@inbounds model.packet_queues[node])
        sense_delay = if qlen >= model.starvation_threshold
            model.slot_time / 1000
        elseif qlen >= model.fairness_threshold
            model.slot_time / 200
        else
            model.slot_time / 100
        end
        node_jitter = (node - 1) * model.slot_time * 0.001
        random_jitter = _det_jitter!(model) * model.slot_time * 0.01
        total_delay = sense_delay + node_jitter + random_jitter
        ev = create_generic_event(t + total_delay, node, model.model_id, :transmission_attempt)
        _safe_schedule_event!(engine, ev)
    end
end

@inline function _handle_collision!(model::EnhancedCSMACDModel, engine::DESEngine, nodes::Vector{Int}, t::Float64)
    model.total_collisions += 1
    for node in nodes
        @inbounds model.ongoing_transmissions[node] = 0.0
        @inbounds model.node_states[node] = NODE_BACKOFF
        @inbounds model.retry_counters[node] += 1
        qlen = length(@inbounds model.packet_queues[node])
        adaptive_retry_limit = max(16, model.num_nodes * 2)
        adaptive_retry_limit = qlen >= model.starvation_threshold ? (adaptive_retry_limit + 10) : adaptive_retry_limit
        if @inbounds model.retry_counters[node] > adaptive_retry_limit
            if !isempty(@inbounds model.packet_queues[node])
                @inbounds popfirst!(model.packet_queues[node])
                @inbounds model.node_drop_counts[node] += 1
            end
            @inbounds model.retry_counters[node] = 0
            @inbounds model.node_states[node] = NODE_IDLE
            if !isempty(@inbounds model.packet_queues[node])
                schedule_transmission_attempt!(model, engine, node, t)
            end
        else
            backoff_time = _enhanced_backoff(model, node)
            ev = create_generic_event(t + backoff_time, node, model.model_id, :backoff_expired)
            _safe_schedule_event!(engine, ev)
        end
    end
end

@inline function _start_transmission!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, t::Float64)
    if @inbounds model.node_states[node] != NODE_IDLE || isempty(@inbounds model.packet_queues[node])
        return
    end
    collided = Int[]
    @inbounds for other in 1:model.num_nodes
        start_time = model.ongoing_transmissions[other]
        if other != node && start_time > 0.0 && abs(t - start_time) < 2 * model.propagation_delay
            push!(collided, other)
        end
    end
    if !isempty(collided)
        push!(collided, node)
        _handle_collision!(model, engine, collided, t)
    else
        @inbounds model.node_states[node] = NODE_TRANSMITTING
        @inbounds model.ongoing_transmissions[node] = t
        ev_end = create_generic_event(t + model.transmission_time, node, model.model_id, :transmission_end)
        _safe_schedule_event!(engine, ev_end)
    end
end

@inline function _complete_transmission!(model::EnhancedCSMACDModel, engine::DESEngine, node::Int, t::Float64)
    if @inbounds model.node_states[node] != NODE_TRANSMITTING || @inbounds model.ongoing_transmissions[node] == 0.0
        return
    end
    if !isempty(@inbounds model.packet_queues[node])
        @inbounds gen_time = popfirst!(model.packet_queues[node])
        push!(model.transmission_delays, t - gen_time)
        model.successful_transmissions += 1
        @inbounds model.node_success_counts[node] += 1
        @inbounds model.retry_counters[node] = 0
        mark_entity_completed!(engine, node)
    end
    @inbounds model.ongoing_transmissions[node] = 0.0
    @inbounds model.node_states[node] = NODE_IDLE
    if !isempty(@inbounds model.packet_queues[node])
        schedule_transmission_attempt!(model, engine, node, t)
    end
end

"""
    process_event!(model::EnhancedCSMACDModel, event::GenericEvent, engine::DESEngine)

Process a single event for EnhancedCSMACDModel.
"""
function process_event!(model::EnhancedCSMACDModel, event::GenericEvent, engine::DESEngine)
    node = event.entity_id
    t = event.time
    etype = event.event_type
    if etype == :frame_generated
        if model.total_generated < model.max_frames
            @inbounds push!(model.packet_queues[node], t)
            @inbounds model.node_generated_counts[node] += 1
            model.total_generated += 1
            qlen = length(@inbounds model.packet_queues[node])
            if qlen > @inbounds model.max_queue_lengths[node]
                @inbounds model.max_queue_lengths[node] = qlen
            end
            increment_arrivals!(engine)
            if @inbounds model.node_states[node] == NODE_IDLE
                schedule_transmission_attempt!(model, engine, node, t)
            end
            if model.total_generated < model.max_frames
                next_ia = _det_exp!(model)
                next_ev = create_generic_event(t + next_ia, node, model.model_id, :frame_generated)
                _safe_schedule_event!(engine, next_ev)
            end
        end
    elseif etype == :transmission_attempt
        _start_transmission!(model, engine, node, t)
    elseif etype == :transmission_end
        _complete_transmission!(model, engine, node, t)
    elseif etype == :backoff_expired
        @inbounds model.node_states[node] = NODE_IDLE
        if !isempty(@inbounds model.packet_queues[node])
            schedule_transmission_attempt!(model, engine, node, t)
        end
    end
end

"""
    finalize_model!(model::EnhancedCSMACDModel, engine::DESEngine)

Finalize the model after simulation. No-op for this model.
"""
function finalize_model!(model::EnhancedCSMACDModel, engine::DESEngine)
end

"""
    get_statistics(model::EnhancedCSMACDModel, engine::DESEngine) -> Dict{Symbol, Any}

Return simulation metrics and per-node statistics.
"""
function get_statistics(model::EnhancedCSMACDModel, engine::DESEngine)
    sim_time = get_current_time(engine)
    throughput = sim_time > 0 ? model.successful_transmissions / sim_time : 0.0
    collision_rate = sim_time > 0 ? model.total_collisions / sim_time : 0.0
    avg_delay = isempty(model.transmission_delays) ? 0.0 : sum(model.transmission_delays) / length(model.transmission_delays)
    total_drops = sum(model.node_drop_counts)
    drop_rate = sim_time > 0 ? total_drops / sim_time : 0.0
    fairness_index = if model.successful_transmissions > 0
        s = sum(model.node_success_counts)
        sq = sum(x -> x*x, model.node_success_counts)
        sq > 0 ? (s*s) / (model.num_nodes * sq) : 1.0
    else
        1.0
    end
    queue_lengths = map(length, model.packet_queues)
    max_queue = isempty(queue_lengths) ? 0 : maximum(queue_lengths)
    min_queue = isempty(queue_lengths) ? 0 : minimum(queue_lengths)
    avg_queue = isempty(queue_lengths) ? 0.0 : sum(queue_lengths) / length(queue_lengths)
    return Dict(
        :simulation_time => sim_time,
        :successful_transmissions => model.successful_transmissions,
        :total_collisions => model.total_collisions,
        :total_generated => sum(model.node_generated_counts),
        :frames_queued => sum(queue_lengths),
        :throughput => throughput,
        :collision_rate => collision_rate,
        :average_delay => avg_delay,
        :total_drops => total_drops,
        :drop_rate => drop_rate,
        :fairness_index => fairness_index,
        :max_queue_length => max_queue,
        :min_queue_length => min_queue,
        :avg_queue_length => avg_queue,
        :node_success_counts => copy(model.node_success_counts),
        :node_drop_counts => copy(model.node_drop_counts),
        :node_generated_counts => copy(model.node_generated_counts)
    )
end

"""
    reset_model!(model::EnhancedCSMACDModel)

Reset the model to its initial state without changing configuration.
"""
function reset_model!(model::EnhancedCSMACDModel)
    model.node_states .= NODE_IDLE
    empty!.(model.packet_queues)
    fill!(model.ongoing_transmissions, 0.0)
    fill!(model.retry_counters, 0)
    fill!(model.node_success_counts, 0)
    fill!(model.node_drop_counts, 0)
    fill!(model.node_generated_counts, 0)
    empty!(model.transmission_delays)
    fill!(model.max_queue_lengths, 0)
    model.successful_transmissions = 0
    model.total_collisions = 0
    model.exp_idx = 1
    model.backoff_idx = 1
end

