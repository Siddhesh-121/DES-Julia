#!/usr/bin/env julia

if !isdefined(Main, :DESLibrary)
    include("../../src/DESLibrary.jl")
end
using .DESLibrary

const λ = 20.0
const μ = 2.0
const c = 15
const max_entities = 200000
const warmup_fraction = 0.1
const seed = 2024

const ρ = λ / μ
const ρ_per_server = λ / (c * μ)

function run_mmc_deslibrary()
    println("M/M/C Queue Simulation using DESLibrary")
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

    engine = DESEngine(max_entities, seed, warmup_fraction)
    model = MMCModel(λ, μ, c, :mmc_queue)
    add_model!(engine, :mmc_queue, model)

    start_time = time()
    simulate!(engine, verbose=false)
    end_time = time()
    wall_clock_time = end_time - start_time

    results = get_results(engine)[:mmc_queue]

    total_entities = engine.entities_arrived
    warmup_entities = engine.warmup_entities
    statistics_entities = results.entities_processed
    simulation_time = results.simulation_time
    stats_collection_time = results.simulation_time

    total_throughput = total_entities / wall_clock_time
    stats_throughput = statistics_entities / wall_clock_time
    events_per_sec = total_entities * 2 / wall_clock_time

    println("\nPERFORMANCE RESULTS")
    println("=" ^ 50)
    println("  Wall-clock time: $(round(wall_clock_time * 1000, digits=2)) ms")
    println("  Simulation time: $(round(simulation_time, digits=2)) time units")
    println("  Total entities processed: $total_entities")
    println("  Entities for statistics: $statistics_entities (after warmup)")
    println("  Warmup entities: $warmup_entities")
    println("  Total throughput: $(round(total_throughput, digits=0)) entities/sec")
    println("  Statistics throughput: $(round(stats_throughput, digits=0)) entities/sec")
    println("  Event processing rate: $(round(events_per_sec, digits=0)) events/sec")

    println("\nSIMULATION STATISTICS")
    println("=" ^ 50)
    println("  Average waiting time: $(round(results.avg_waiting_time, digits=4)) time units")
    println("  Average service time: $(round(results.avg_service_time, digits=4)) time units")
    println("  Average time in system: $(round(results.avg_time_in_system, digits=4)) time units")
    println("  Average queue length: $(round(results.avg_queue_length, digits=4)) entities")
    println("  Maximum queue length: $(results.max_queue_length) entities")
    println("  Server utilization: $(round(results.server_utilization, digits=4)) ($(round(results.server_utilization*100, digits=2))%)")

    println("\nINDIVIDUAL SERVER UTILIZATIONS")
    println("=" ^ 50)
    server_utils = []
    for i in 1:c
        key = Symbol("server$(i)_utilization")
        if haskey(results.extra_metrics, key)
            util = results.extra_metrics[key]
            push!(server_utils, util)
            println("  Server $i: $(round(util, digits=4)) ($(round(util*100, digits=2))%)")
        end
    end

    server_cv = 0.0
    if length(server_utils) > 1
        server_mean = mean(server_utils)
        server_std = std(server_utils)
        server_cv = server_std / server_mean
        println("  Server balance (CV): $(round(server_cv, digits=6)) (lower is better)")
        println("  Balance status: $(server_cv < 0.01 ? "Excellent" : server_cv < 0.05 ? "Good" : "Fair")")
    end

    println("\nTHEORETICAL CALCULATIONS")
    println("=" ^ 50)

    theoretical_waiting_time = 0.0
    theoretical_service_time = 0.0
    theoretical_time_in_system = 0.0
    theoretical_queue_length = 0.0
    theoretical_utilization = 0.0
    erlang_c = 0.0
    sum_term = 0.0

    if ρ_per_server >= 1.0
        println("  System is unstable - theoretical values are infinite")
        theoretical_waiting_time = Inf
        theoretical_service_time = 1.0 / μ
        theoretical_time_in_system = Inf
        theoretical_queue_length = Inf
        theoretical_utilization = 1.0
        erlang_c = 1.0
    else
        theoretical_service_time = 1.0 / μ
        theoretical_utilization = ρ_per_server
        
        if c == 1
            erlang_c = ρ_per_server
        else
            sum_term = 0.0
            for k in 0:(c-1)
                sum_term += (ρ^k) / factorial(k)
            end
            
            erlang_c_numerator = (ρ^c) / factorial(c)
            erlang_c_denominator = sum_term + erlang_c_numerator / (1 - ρ_per_server)
            erlang_c = erlang_c_numerator / (erlang_c_denominator * (1 - ρ_per_server))
        end
        
        theoretical_waiting_time = erlang_c / (c * μ * (1 - ρ_per_server))
        theoretical_time_in_system = theoretical_waiting_time + theoretical_service_time
        theoretical_queue_length = erlang_c * ρ_per_server / (1 - ρ_per_server)
        
        println("  Theoretical waiting time: $(round(theoretical_waiting_time, digits=4)) time units")
        println("  Theoretical service time: $(round(theoretical_service_time, digits=4)) time units")
        println("  Theoretical time in system: $(round(theoretical_time_in_system, digits=4)) time units")
        println("  Theoretical queue length: $(round(theoretical_queue_length, digits=4)) entities")
        println("  Theoretical utilization: $(round(theoretical_utilization, digits=4)) ($(round(theoretical_utilization*100, digits=2))%)")
        println("  Erlang-C (P(wait)): $(round(erlang_c, digits=6))")
    end

    println("\nACCURACY VALIDATION")
    println("=" ^ 50)

    max_rel_error = 0.0

    if ρ_per_server < 1.0
        waiting_error = abs(results.avg_waiting_time - theoretical_waiting_time)
        service_error = abs(results.avg_service_time - theoretical_service_time)
        system_error = abs(results.avg_time_in_system - theoretical_time_in_system)
        queue_error = abs(results.avg_queue_length - theoretical_queue_length)
        utilization_error = abs(results.server_utilization - theoretical_utilization)
        
        waiting_rel_error = waiting_error / theoretical_waiting_time * 100
        service_rel_error = service_error / theoretical_service_time * 100
        system_rel_error = system_error / theoretical_time_in_system * 100
        queue_rel_error = queue_error / theoretical_queue_length * 100
        utilization_rel_error = utilization_error / theoretical_utilization * 100
        
        println("  Waiting time error: $(round(waiting_error, digits=6)) ($(round(waiting_rel_error, digits=2))%)")
        println("  Service time error: $(round(service_error, digits=6)) ($(round(service_rel_error, digits=2))%)")
        println("  System time error: $(round(system_error, digits=6)) ($(round(system_rel_error, digits=2))%)")
        println("  Queue length error: $(round(queue_error, digits=6)) ($(round(queue_rel_error, digits=2))%)")
        println("  Utilization error: $(round(utilization_error, digits=6)) ($(round(utilization_rel_error, digits=2))%)")
        
        max_rel_error = max(waiting_rel_error, service_rel_error, system_rel_error, 
                           queue_rel_error, utilization_rel_error)
        
        println("\n  Maximum relative error: $(round(max_rel_error, digits=2))%")
        if max_rel_error < 1.0
            println("  Accuracy status: Excellent (< 1% error)")
        elseif max_rel_error < 5.0
            println("  Accuracy status: Good (< 5% error)")
        elseif max_rel_error < 10.0
            println("  Accuracy status: Fair (< 10% error)")
        else
            println("  Accuracy status: Poor (≥ 10% error)")
        end
    else
        println("  System is unstable - accuracy validation not applicable")
    end

    println("\nLITTLE'S LAW VALIDATION")
    println("=" ^ 50)

    effective_arrival_rate = statistics_entities / stats_collection_time
    L = results.avg_queue_length
    W = results.avg_waiting_time
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
        println("  Little's Law: Validated (< 1% error)")
    elseif littles_law_rel_error < 5.0
        println("  Little's Law: Acceptable (< 5% error)")
    else
        println("  Little's Law: Questionable (≥ 5% error)")
    end

    println("\nSYSTEM PERFORMANCE SUMMARY")
    println("=" ^ 50)
    println("  Implementation: DESLibrary M/M/C")
    println("  Configuration: λ=$λ, μ=$μ, c=$c")
    println("  Entities processed: $total_entities")
    println("  Runtime: $(round(wall_clock_time * 1000, digits=2)) ms")
    println("  Throughput: $(round(total_throughput, digits=0)) entities/sec")
    println("  Memory efficiency: $(model.buffer_wrapped ? "Circular buffer active" : "No buffer wrapping")")
    println("  Server balance: $(length(server_utils) > 1 ? "$(round(server_cv, digits=6)) CV" : "N/A")")
    println("  Theoretical accuracy: $(ρ_per_server < 1.0 ? "$(round(max_rel_error, digits=2))% max error" : "N/A (unstable)")")
    println("  Little's Law: $(round(littles_law_rel_error, digits=4))% error")

    println("\nDESLibrary M/M/C simulation completed successfully!")
end

if abspath(PROGRAM_FILE) == @__FILE__
    run_mmc_deslibrary()
end