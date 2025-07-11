# Statistical utility functions

# Calculate theoretical MM1 values
function theoretical_mm1(λ::Float64, μ::Float64)
    ρ = λ / μ
    
    if ρ >= 1.0
        return (
            utilization = 1.0,
            waiting_time = Inf,
            service_time = 1.0/μ,
            time_in_system = Inf,
            queue_length = Inf,
            customers_in_system = Inf
        )
    end
    
    return (
        utilization = ρ,
        waiting_time = ρ / (μ * (1 - ρ)),
        service_time = 1.0/μ,
        time_in_system = 1.0/(μ - λ),
        queue_length = (ρ * ρ) / (1 - ρ),
        customers_in_system = ρ / (1 - ρ)
    )
end

# Calculate theoretical MM2 values
function theoretical_mm2(λ::Float64, μ::Float64)
    ρ = λ / (2 * μ)  # Traffic intensity per server
    
    if ρ >= 1.0
        return (
            utilization = 1.0,
            waiting_time = Inf,
            service_time = 1.0/μ,
            time_in_system = Inf,
            queue_length = Inf,
            customers_in_system = Inf
        )
    end
    
    # MM2 calculations
    ρ_total = λ / μ  # Total traffic intensity
    P0 = 1.0 / (1.0 + ρ_total + (ρ_total^2) / (2 * (1 - ρ)))
    Lq = (P0 * (ρ_total^2) * ρ) / (2 * (1 - ρ)^2)
    Wq = Lq / λ
    
    return (
        utilization = ρ,
        waiting_time = Wq,
        service_time = 1.0/μ,
        time_in_system = Wq + 1.0/μ,
        queue_length = Lq,
        customers_in_system = Lq + ρ_total
    )
end

# Calculate errors between simulation and theoretical values
function calculate_errors(sim_results::SimulationResults, theoretical)
    return (
        utilization_error = abs(sim_results.server_utilization - theoretical.utilization),
        waiting_time_error = abs(sim_results.avg_waiting_time - theoretical.waiting_time),
        service_time_error = abs(sim_results.avg_service_time - theoretical.service_time),
        time_in_system_error = abs(sim_results.avg_time_in_system - theoretical.time_in_system),
        queue_length_error = abs(sim_results.avg_queue_length - theoretical.queue_length)
    )
end

# Calculate confidence intervals
function confidence_interval(data::Vector{Float64}, confidence_level::Float64=0.95)
    if length(data) < 2
        return (mean(data), 0.0, 0.0)
    end
    
    m = mean(data)
    s = std(data)
    n = length(data)
    
    # t-distribution critical value (approximation for large n)
    α = 1 - confidence_level
    t_critical = 1.96  # Approximate for 95% confidence
    
    margin_error = t_critical * s / sqrt(n)
    
    return (m, m - margin_error, m + margin_error)
end

# Performance summary
function performance_summary(times::Vector{Float64}, entities_processed::Int)
    total_time = sum(times)
    avg_time = mean(times)
    std_time = std(times)
    min_time = minimum(times)
    max_time = maximum(times)
    
    entities_per_second = entities_processed / avg_time
    time_per_entity_μs = (avg_time * 1_000_000) / entities_processed
    
    return (
        total_time = total_time,
        avg_time = avg_time,
        std_time = std_time,
        min_time = min_time,
        max_time = max_time,
        entities_per_second = entities_per_second,
        time_per_entity_μs = time_per_entity_μs
    )
end

# Little's Law validation
function validate_littles_law(results::SimulationResults, λ::Float64)
    # L = λW (queue length = arrival rate × waiting time)
    L_simulated = results.avg_queue_length
    λW_calculated = λ * results.avg_waiting_time
    error = abs(L_simulated - λW_calculated)
    relative_error = L_simulated > 0 ? error / L_simulated : 0.0
    
    return (
        L = L_simulated,
        λW = λW_calculated,
        error = error,
        relative_error = relative_error,
        is_valid = relative_error < 0.02  # FIXED: Tighter tolerance (2% instead of 5%)
    )
end

# Print formatted results
function print_results(results::SimulationResults, model_name::String="")
    println("=== $model_name SIMULATION RESULTS ===")
    println("Entities processed: $(results.entities_processed)")
    println("Simulation time: $(round(results.simulation_time, digits=2))")
    println("Average waiting time: $(round(results.avg_waiting_time, digits=4))")
    println("Average service time: $(round(results.avg_service_time, digits=4))")
    println("Average time in system: $(round(results.avg_time_in_system, digits=4))")
    println("Server utilization: $(round(results.server_utilization, digits=4))")
    println("Average queue length: $(round(results.avg_queue_length, digits=4))")
    println("Max queue length: $(results.max_queue_length)")
    println("Max waiting time: $(round(results.max_waiting_time, digits=4))")
    println("Waiting time std dev: $(round(results.waiting_time_std, digits=4))")
    
    # Print extra metrics if available
    if !isempty(results.extra_metrics)
        println("\n=== ADDITIONAL METRICS ===")
        for (key, value) in results.extra_metrics
            println("$key: $(round(value, digits=4))")
        end
    end
    println()
end

# Compare simulation vs theoretical
function print_comparison(sim_results::SimulationResults, theoretical, model_name::String="")
    errors = calculate_errors(sim_results, theoretical)
    
    println("=== $model_name COMPARISON ===")
    println("Metric                 | Simulation | Theoretical | Error")
    println("-" ^ 60)
    println("Server utilization     | $(rpad(round(sim_results.server_utilization, digits=4), 10)) | $(rpad(round(theoretical.utilization, digits=4), 11)) | $(round(errors.utilization_error, digits=4))")
    println("Average waiting time   | $(rpad(round(sim_results.avg_waiting_time, digits=4), 10)) | $(rpad(round(theoretical.waiting_time, digits=4), 11)) | $(round(errors.waiting_time_error, digits=4))")
    println("Average service time   | $(rpad(round(sim_results.avg_service_time, digits=4), 10)) | $(rpad(round(theoretical.service_time, digits=4), 11)) | $(round(errors.service_time_error, digits=4))")
    println("Average time in system | $(rpad(round(sim_results.avg_time_in_system, digits=4), 10)) | $(rpad(round(theoretical.time_in_system, digits=4), 11)) | $(round(errors.time_in_system_error, digits=4))")
    println("Average queue length   | $(rpad(round(sim_results.avg_queue_length, digits=4), 10)) | $(rpad(round(theoretical.queue_length, digits=4), 11)) | $(round(errors.queue_length_error, digits=4))")
    println()
end 