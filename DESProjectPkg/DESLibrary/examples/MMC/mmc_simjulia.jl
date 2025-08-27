#!/usr/bin/env julia

using SimJulia
using ResumableFunctions
using Distributions
using Statistics
using Printf
using Random

const λ = 20.0
const μ = 2.0
const c = 15
const max_entities = 200000
const warmup_fraction = 0.1
const seed = 2024

const ρ = λ / μ
const ρ_per_server = λ / (c * μ)

mutable struct MMCStatistics
    warmup_entities::Int
    warmup_completed::Bool
    warmup_completion_time::Float64
    
    all_waiting_times::Vector{Float64}
    all_service_times::Vector{Float64}
    all_system_times::Vector{Float64}
    all_arrival_times::Vector{Float64}
    all_departure_times::Vector{Float64}
    
    waiting_times::Vector{Float64}
    service_times::Vector{Float64}
    system_times::Vector{Float64}
    arrival_times::Vector{Float64}
    departure_times::Vector{Float64}
    
    queue_length_samples::Vector{Int}
    queue_length_times::Vector{Float64}
    queue_area::Float64
    queue_area_stats::Float64
    last_queue_time::Float64
    last_queue_size::Int
    
    server_busy_times::Vector{Float64}
    server_busy_times_stats::Vector{Float64}
    server_start_times::Vector{Float64}
    servers_busy::Vector{Bool}
    
    total_entities::Int
    statistics_entities::Int
    max_queue_length::Int
    
    function MMCStatistics(warmup_entities::Int)
        new(warmup_entities, false, 0.0,
            Float64[], Float64[], Float64[], Float64[], Float64[],
            Float64[], Float64[], Float64[], Float64[], Float64[],
            Int[], Float64[], 0.0, 0.0, 0.0, 0,
            zeros(Float64, c), zeros(Float64, c), zeros(Float64, c), fill(false, c),
            0, 0, 0)
    end
end

function record_arrival!(stats::MMCStatistics, env::Simulation, entity_id::Int)
    current_time = now(env)
    push!(stats.all_arrival_times, current_time)
    
    if !stats.warmup_completed && entity_id >= stats.warmup_entities
        stats.warmup_completed = true
        stats.warmup_completion_time = current_time
        println("    Warmup completed at time $(round(current_time, digits=2)) after $(entity_id) entities")
    end
    
    if stats.warmup_completed
        push!(stats.arrival_times, current_time)
    end
end

function record_service_start!(stats::MMCStatistics, env::Simulation, entity_id::Int, server_id::Int)
    current_time = now(env)
    stats.servers_busy[server_id] = true
    stats.server_start_times[server_id] = current_time
end

function record_service_end!(stats::MMCStatistics, env::Simulation, entity_id::Int, server_id::Int, arrival_time::Float64, service_time::Float64)
    current_time = now(env)
    waiting_time = current_time - arrival_time - service_time
    system_time = current_time - arrival_time
    
    stats.server_busy_times[server_id] += service_time
    if stats.warmup_completed
        stats.server_busy_times_stats[server_id] += service_time
    end
    stats.servers_busy[server_id] = false
    
    push!(stats.all_waiting_times, waiting_time)
    push!(stats.all_service_times, service_time)
    push!(stats.all_system_times, system_time)
    push!(stats.all_departure_times, current_time)
    
    if stats.warmup_completed
        push!(stats.waiting_times, waiting_time)
        push!(stats.service_times, service_time)
        push!(stats.system_times, system_time)
        push!(stats.departure_times, current_time)
        stats.statistics_entities += 1
    end
    
    stats.total_entities += 1
end

function record_queue_length!(stats::MMCStatistics, env::Simulation, queue_length::Int)
    current_time = now(env)
    
    if stats.last_queue_time > 0
        time_duration = current_time - stats.last_queue_time
        stats.queue_area += stats.last_queue_size * time_duration
        if stats.warmup_completed
            stats.queue_area_stats += stats.last_queue_size * time_duration
        end
    end
    
    push!(stats.queue_length_samples, queue_length)
    push!(stats.queue_length_times, current_time)
    stats.last_queue_time = current_time
    stats.last_queue_size = queue_length
    stats.max_queue_length = max(stats.max_queue_length, queue_length)
end

@resumable function customer(env::Simulation, customer_id::Int, server::Resource, stats::MMCStatistics)
    arrival_time = now(env)
    record_arrival!(stats, env, customer_id)
    
    @yield request(server)
    
    queue_length = length(server.put_queue)
    record_queue_length!(stats, env, queue_length)
    
    server_id = (customer_id % c) + 1
    record_service_start!(stats, env, customer_id, server_id)
    
    service_time = rand(Exponential(1.0 / μ))
    
    @yield timeout(env, service_time)
    
    record_service_end!(stats, env, customer_id, server_id, arrival_time, service_time)
    
    @yield release(server)
    
    queue_length = length(server.put_queue)
    record_queue_length!(stats, env, queue_length)
end

@resumable function arrival_process(env::Simulation, server::Resource, stats::MMCStatistics)
    customer_id = 0
    
    while customer_id < max_entities
        inter_arrival = rand(Exponential(1.0 / λ))
        @yield timeout(env, inter_arrival)
        
        queue_length = length(server.put_queue)
        record_queue_length!(stats, env, queue_length)
        
        @process customer(env, customer_id, server, stats)
        customer_id += 1
    end
end

@resumable function termination_monitor(env::Simulation, stats::MMCStatistics)
    while stats.total_entities < max_entities
        @yield timeout(env, 1.0)
    end
end

function calculate_theoretical_values(λ::Float64, μ::Float64, c::Int)
    ρ = λ / μ
    ρ_per_server = λ / (c * μ)
    
    if ρ_per_server >= 1.0
        return (waiting_time = Inf, service_time = 1.0 / μ, system_time = Inf, 
                queue_length = Inf, utilization = 1.0, erlang_c = 1.0)
    end
    
    theoretical_service_time = 1.0 / μ
    theoretical_utilization = ρ_per_server
    
    if c == 1
        erlang_c = ρ_per_server
    else
        sum_term = sum((ρ^k) / factorial(k) for k in 0:(c-1))
        erlang_c_numerator = (ρ^c) / factorial(c)
        erlang_c_denominator = sum_term + erlang_c_numerator / (1 - ρ_per_server)
        erlang_c = erlang_c_numerator / (erlang_c_denominator * (1 - ρ_per_server))
    end
    
    theoretical_waiting_time = erlang_c / (c * μ * (1 - ρ_per_server))
    theoretical_system_time = theoretical_waiting_time + theoretical_service_time
    theoretical_queue_length = erlang_c * ρ_per_server / (1 - ρ_per_server)
    
    return (waiting_time = theoretical_waiting_time, service_time = theoretical_service_time,
            system_time = theoretical_system_time, queue_length = theoretical_queue_length,
            utilization = theoretical_utilization, erlang_c = erlang_c)
end

function main()
    println("M/M/C Queue Simulation using SimJulia")
    println("=" ^ 50)
    println("Configuration:")
    println("  Arrival rate (λ): $λ entities/time")
    println("  Service rate (μ): $μ entities/time per server")
    println("  Number of servers (c): $c")
    println("  Total entities: $max_entities")
    println("  Warmup period: $(round(warmup_fraction*100, digits=1))%")
    println("  Traffic intensity (ρ): $(round(ρ, digits=3))")
    println("  Traffic intensity per server (ρ/c): $(round(ρ_per_server, digits=3))")
    println("  System status: $(ρ_per_server < 1.0 ? "Stable" : "Unstable")")
    
    if ρ_per_server >= 1.0
        println("Warning: System is unstable (ρ/c ≥ 1.0). Results may be unreliable.")
    end
    
    println("\nStarting simulation...")
    
    Random.seed!(seed)
    
    env = Simulation()
    
    server = Resource(env, c)
    
    warmup_entities = max(1, Int(round(max_entities * warmup_fraction)))
    stats = MMCStatistics(warmup_entities)
    
    @process arrival_process(env, server, stats)
    @process termination_monitor(env, stats)
    
    start_time = time()
    run(env)
    end_time = time()
    wall_clock_time = end_time - start_time
    
    record_queue_length!(stats, env, length(server.put_queue))
    
    simulation_time = now(env)
    if !isfinite(simulation_time)
        simulation_time = !isempty(stats.departure_times) ? stats.departure_times[end] : (stats.last_queue_time > 0 ? stats.last_queue_time : 0.0)
    end
    if stats.warmup_completed
        stats_collection_time = simulation_time > stats.warmup_completion_time ? simulation_time - stats.warmup_completion_time : 0.0
    else
        stats_collection_time = simulation_time
    end
    
    total_throughput = wall_clock_time > 0 ? stats.total_entities / wall_clock_time : 0.0
    stats_throughput = wall_clock_time > 0 ? stats.statistics_entities / wall_clock_time : 0.0
    events_per_sec = wall_clock_time > 0 ? stats.total_entities * 2 / wall_clock_time : 0.0
    
    println("\nPERFORMANCE RESULTS")
    println("=" ^ 50)
    println("  Wall-clock time: $(round(wall_clock_time * 1000, digits=2)) ms")
    println("  Simulation time: $(round(simulation_time, digits=2)) time units")
    println("  Total entities processed: $(stats.total_entities)")
    println("  Entities for statistics: $(stats.statistics_entities) (after warmup)")
    println("  Warmup entities: $warmup_entities")
    println("  Total throughput: $(round(total_throughput, digits=0)) entities/sec")
    println("  Statistics throughput: $(round(stats_throughput, digits=0)) entities/sec")
    println("  Event processing rate: $(round(events_per_sec, digits=0)) events/sec")
    
    avg_waiting_time = length(stats.waiting_times) > 0 ? mean(stats.waiting_times) : 0.0
    avg_service_time = length(stats.service_times) > 0 ? mean(stats.service_times) : 0.0
    avg_system_time = length(stats.system_times) > 0 ? mean(stats.system_times) : 0.0
    
    avg_queue_length = if stats_collection_time > 0 && stats.queue_area_stats > 0
        stats.queue_area_stats / stats_collection_time
    elseif simulation_time > 0 && stats.queue_area > 0
        stats.queue_area / simulation_time
    else
        length(stats.queue_length_samples) > 0 ? mean(stats.queue_length_samples) : 0.0
    end
    
    total_server_busy_time = sum(stats.server_busy_times_stats)
    server_utilization = stats_collection_time > 0 ? total_server_busy_time / (c * stats_collection_time) : 0.0
    
    println("\nSIMULATION STATISTICS")
    println("=" ^ 50)
    println("  Average waiting time: $(round(avg_waiting_time, digits=4)) time units")
    println("  Average service time: $(round(avg_service_time, digits=4)) time units")
    println("  Average time in system: $(round(avg_system_time, digits=4)) time units")
    println("  Average queue length: $(round(avg_queue_length, digits=4)) entities")
    println("  Maximum queue length: $(stats.max_queue_length) entities")
    println("  Server utilization: $(round(server_utilization, digits=4)) ($(round(server_utilization*100, digits=2))%)")
    
    println("\nINDIVIDUAL SERVER UTILIZATIONS")
    println("=" ^ 50)
    server_utils = Float64[]
    for i in 1:c
        util = stats_collection_time > 0 ? stats.server_busy_times_stats[i] / stats_collection_time : 0.0
        push!(server_utils, util)
        println("  Server $i: $(round(util, digits=4)) ($(round(util*100, digits=2))%)")
    end
    
    server_cv = 0.0
    if length(server_utils) > 1
        server_mean = mean(server_utils)
        server_std = std(server_utils)
        server_cv = server_mean > 0 ? server_std / server_mean : 0.0
        println("  Server balance (CV): $(round(server_cv, digits=6)) (lower is better)")
        balance_status = server_cv < 0.01 ? "Excellent" : server_cv < 0.05 ? "Good" : "Fair"
        println("  Balance status: $balance_status")
    end
    
    theoretical = calculate_theoretical_values(λ, μ, c)
    
    println("\nTHEORETICAL CALCULATIONS")
    println("=" ^ 50)
    
    if ρ_per_server >= 1.0
        println("  System is unstable - theoretical values are infinite")
    else
        println("  Theoretical waiting time: $(round(theoretical.waiting_time, digits=4)) time units")
        println("  Theoretical service time: $(round(theoretical.service_time, digits=4)) time units")
        println("  Theoretical time in system: $(round(theoretical.system_time, digits=4)) time units")
        println("  Theoretical queue length: $(round(theoretical.queue_length, digits=4)) entities")
        println("  Theoretical utilization: $(round(theoretical.utilization, digits=4)) ($(round(theoretical.utilization*100, digits=2))%)")
        println("  Erlang-C (P(wait)): $(round(theoretical.erlang_c, digits=6))")
    end
    
    println("\nACCURACY VALIDATION")
    println("=" ^ 50)
    
    max_rel_error = 0.0
    
    if ρ_per_server < 1.0
        waiting_error = abs(avg_waiting_time - theoretical.waiting_time)
        service_error = abs(avg_service_time - theoretical.service_time)
        system_error = abs(avg_system_time - theoretical.system_time)
        queue_error = abs(avg_queue_length - theoretical.queue_length)
        utilization_error = abs(server_utilization - theoretical.utilization)
        
        waiting_rel_error = waiting_error / theoretical.waiting_time * 100
        service_rel_error = service_error / theoretical.service_time * 100
        system_rel_error = system_error / theoretical.system_time * 100
        queue_rel_error = queue_error / theoretical.queue_length * 100
        utilization_rel_error = utilization_error / theoretical.utilization * 100
        
        println("  Waiting time error: $(round(waiting_error, digits=6)) ($(round(waiting_rel_error, digits=2))%)")
        println("  Service time error: $(round(service_error, digits=6)) ($(round(service_rel_error, digits=2))%)")
        println("  System time error: $(round(system_error, digits=6)) ($(round(system_rel_error, digits=2))%)")
        println("  Queue length error: $(round(queue_error, digits=6)) ($(round(queue_rel_error, digits=2))%)")
        println("  Utilization error: $(round(utilization_error, digits=6)) ($(round(utilization_rel_error, digits=2))%)")
        
        max_rel_error = max(waiting_rel_error, service_rel_error, system_rel_error, 
                           queue_rel_error, utilization_rel_error)
        
        println("\n  Maximum relative error: $(round(max_rel_error, digits=2))%")
        if max_rel_error < 1.0
            println("  Accuracy status:  Excellent (< 1% error)")
        elseif max_rel_error < 5.0
            println("  Accuracy status:  Good (< 5% error)")
        elseif max_rel_error < 10.0
            println("  Accuracy status:  Fair (< 10% error)")
        else
            println("  Accuracy status:  Poor (≥ 10% error)")
        end
    else
        println("  System is unstable - accuracy validation not applicable")
    end
    
    println("\nLITTLE'S LAW VALIDATION")
    println("=" ^ 50)
    
    effective_arrival_rate = stats_collection_time > 0 ? stats.statistics_entities / stats_collection_time : 0.0
    L = avg_queue_length
    W = avg_waiting_time
    lambda_W = effective_arrival_rate * W
    
    littles_law_error = abs(L - lambda_W)
    littles_law_rel_error = L > 0 ? (littles_law_error / L) * 100 : 0.0
    
    println("  L (average queue length): $(round(L, digits=4))")
    println("  λ (effective arrival rate): $(round(effective_arrival_rate, digits=4))")
    println("  W (average waiting time): $(round(W, digits=4))")
    println("  λW (arrival rate × waiting time): $(round(lambda_W, digits=4))")
    println("  |L - λW| (absolute error): $(round(littles_law_error, digits=6))")
    println("  Relative error: $(round(littles_law_rel_error, digits=4))%")
    
    if littles_law_rel_error < 1.0
        println("  Little's Law:  Validated (< 1% error)")
    elseif littles_law_rel_error < 5.0
        println("  Little's Law:  Acceptable (< 5% error)")
    else
        println("  Little's Law:  Questionable (≥ 5% error)")
    end
    
    println("\nSYSTEM PERFORMANCE SUMMARY")
    println("=" ^ 50)
    println("  Implementation: SimJulia M/M/C")
    println("  Configuration: λ=$λ, μ=$μ, c=$c")
    println("  Entities processed: $(stats.total_entities)")
    println("  Runtime: $(round(wall_clock_time * 1000, digits=2)) ms")
    println("  Throughput: $(round(total_throughput, digits=0)) entities/sec")
    println("  Memory efficiency: No special optimization")
    println("  Server balance: $(length(server_utils) > 1 ? "$(round(server_cv, digits=6)) CV" : "N/A")")
    accuracy_text = ρ_per_server < 1.0 ? "$(round(max_rel_error, digits=2))% max error" : "N/A (unstable)"
    println("  Theoretical accuracy: $accuracy_text")
    println("  Little's Law: $(round(littles_law_rel_error, digits=4))% error")
    
    println("\nSimJulia M/M/C simulation completed successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end