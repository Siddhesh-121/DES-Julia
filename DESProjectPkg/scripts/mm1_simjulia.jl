using SimJulia
using ResumableFunctions
using Distributions, Random
using Statistics

@resumable function customer(env::Simulation, server::Resource, results_chan::Channel, μ::Float64)
    arrive = now(env)
    @yield request(server)
    wait_time = now(env) - arrive
    
    service_time = rand(Exponential(1/μ))
    @yield timeout(env, service_time)
    
    @yield release(server)
    
    total_time = now(env) - arrive
    
    put!(results_chan, (wait_time, service_time, total_time))
end

@resumable function arrival_process(env::Simulation, server::Resource, results_chan::Channel, λ::Float64, μ::Float64, num_customers::Int, last_arrival_time::Ref{Float64})
    for i in 1:num_customers
        @process customer(env, server, results_chan, μ)
        interarrival = rand(Exponential(1/λ))
        @yield timeout(env, interarrival)
        last_arrival_time[] = now(env)
    end
end

function run_mm1_simulation(λ::Float64, μ::Float64, num_customers::Int)
    env = Simulation()
    server = Resource(env, 1)
    
    results_chan = Channel{Tuple{Float64, Float64, Float64}}(num_customers)
    
    last_arrival_time = Ref(0.0)
    
    @process arrival_process(env, server, results_chan, λ, μ, num_customers, last_arrival_time)
    run(env)
    
    close(results_chan)

    return results_chan, now(env), last_arrival_time[]
end

function analyze_and_print_results(results_channel, total_simulation_time, last_arrival_time, num_customers)
    all_results = collect(results_channel)
    waiting_times = [res[1] for res in all_results]
    service_times = [res[2] for res in all_results]
    times_in_system = [res[3] for res in all_results]

    total_busy_time = 0.0
    for t in service_times
        total_busy_time += t
    end

    effective_simulation_time = last_arrival_time
    server_utilization = effective_simulation_time > 0 ? total_busy_time / effective_simulation_time : 0.0

    arrival_rate = effective_simulation_time > 0 ? num_customers / effective_simulation_time : 0.0
    avg_wait_time = !isempty(waiting_times) ? mean(waiting_times) : 0.0
    avg_service_time = !isempty(service_times) ? mean(service_times) : 0.0
    avg_time_in_system = !isempty(times_in_system) ? mean(times_in_system) : 0.0
    
    avg_customers_waiting = arrival_rate * avg_wait_time
    avg_customers_in_system = arrival_rate * avg_time_in_system
    avg_customers_in_queue = avg_customers_in_system - server_utilization

    max_wait_time = !isempty(waiting_times) ? maximum(waiting_times) : 0.0
    max_service_time = !isempty(service_times) ? maximum(service_times) : 0.0
    max_time_in_system = !isempty(times_in_system) ? maximum(times_in_system) : 0.0
    
    std_dev_wait = length(waiting_times) > 1 ? std(waiting_times) : 0.0
    std_dev_service = length(service_times) > 1 ? std(service_times) : 0.0
    std_dev_time_in_system = length(times_in_system) > 1 ? std(times_in_system) : 0.0
    
    cv_wait = avg_wait_time > 0 ? std_dev_wait / avg_wait_time : 0.0
    cv_service = avg_service_time > 0 ? std_dev_service / avg_service_time : 0.0
    cv_time_in_system = avg_time_in_system > 0 ? std_dev_time_in_system / avg_time_in_system : 0.0

    λ_theoretical = 4.5
    μ_theoretical = 5.0
    ρ_theoretical = λ_theoretical / μ_theoretical
    
    theoretical_wait_time = ρ_theoretical / (μ_theoretical * (1 - ρ_theoretical))
    theoretical_service_time = 1 / μ_theoretical
    theoretical_time_in_system = 1 / (μ_theoretical - λ_theoretical)
    theoretical_queue_length = (ρ_theoretical^2) / (1 - ρ_theoretical)
    theoretical_customers_in_system = ρ_theoretical / (1 - ρ_theoretical)

    println("--- SimJulia M/M/1 Comprehensive Results ---")
    println("SIMULATION RESULTS:")
    println("Server Utilization: $(round(server_utilization, digits=3))")
    println("Average Queue Length: $(round(avg_customers_in_queue, digits=3))")
    println("Average Customers in System: $(round(avg_customers_in_system, digits=3))")
    println("Average Customers Waiting: $(round(avg_customers_waiting, digits=3))")
    println("Average Waiting Time: $(round(avg_wait_time, digits=3))")
    println("Average Service Time: $(round(avg_service_time, digits=3))")
    println("Average Time in System: $(round(avg_time_in_system, digits=3))")
    println("Max Waiting Time: $(round(max_wait_time, digits=3))")
    println("Max Service Time: $(round(max_service_time, digits=3))")
    println("Max Time in System: $(round(max_time_in_system, digits=3))")
    println("Waiting Time Std Dev: $(round(std_dev_wait, digits=3))")
    println("Service Time Std Dev: $(round(std_dev_service, digits=3))")
    println("Time in System Std Dev: $(round(std_dev_time_in_system, digits=3))")
    println("Waiting Time CV: $(round(cv_wait, digits=3))")
    println("Service Time CV: $(round(cv_service, digits=3))")
    println("Time in System CV: $(round(cv_time_in_system, digits=3))")
    println("Effective Simulation Time: $(round(effective_simulation_time, digits=3))")
    println("Effective Arrival Rate: $(round(arrival_rate, digits=3))")

    println("\nTHEORETICAL M/M/1 VALUES:")
    println("Theoretical Utilization (ρ): $(round(ρ_theoretical, digits=3))")
    println("Theoretical Avg Queue Length: $(round(theoretical_queue_length, digits=3))")
    println("Theoretical Avg Customers in System: $(round(theoretical_customers_in_system, digits=3))")
    println("Theoretical Avg Waiting Time: $(round(theoretical_wait_time, digits=3))")
    println("Theoretical Avg Service Time: $(round(theoretical_service_time, digits=3))")
    println("Theoretical Avg Time in System: $(round(theoretical_time_in_system, digits=3))")

    println("\nVERIFICATION:")
    println("Utilization Error: $(round(abs(server_utilization - ρ_theoretical), digits=3))")
    println("Queue Length Error: $(round(abs(avg_customers_in_queue - theoretical_queue_length), digits=3))")
    println("Customers in System Error: $(round(abs(avg_customers_in_system - theoretical_customers_in_system), digits=3))")
    println("Waiting Time Error: $(round(abs(avg_wait_time - theoretical_wait_time), digits=3))")
    println("Service Time Error: $(round(abs(avg_service_time - theoretical_service_time), digits=3))")
    println("Time in System Error: $(round(abs(avg_time_in_system - theoretical_time_in_system), digits=3))")
end

λ = 4.5
μ = 5.0
num_customers = 100000
Random.seed!(2024)

# Add timing here
start_time = time()
results_channel, total_simulation_time, last_arrival_time = run_mm1_simulation(λ, μ, num_customers)
end_time = time()
execution_time = end_time - start_time

println("SimJulia execution time: $(round(execution_time, digits=4)) seconds")

analyze_and_print_results(results_channel, total_simulation_time, last_arrival_time, num_customers)