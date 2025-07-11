using Pkg
Pkg.activate(".")

using Statistics
import DESProject as DES

println("Creating simulation components...")
engine = DES.SimulationEngine(100000)
rng_manager = DES.RNGManager(2024)

println("Initializing simulation...")
DES.initialize_simulation!(engine.event_queue, rng_manager)
DES.initialize_queue_tracking!(engine)

println("Starting simulation...")
start_time = time()
DES.run!(engine, rng_manager)
end_time = time()
execution_time = end_time - start_time

stats = DES.get_simulation_statistics(engine)

println("\nSimulation completed.")
println("Execution time: $(round(execution_time, digits=4)) seconds")
println("Total entities processed: $(DES.get_entities_processed(engine))")
println("Final simulation time: $(round(DES.get_current_time(engine), digits=2))")
println("Time-weighted average queue length: $(round(stats.time_weighted_queue_length, digits=3))")
println("Simple average queue length: $(round(stats.simple_average_queue_length, digits=3))")
println("Average customers in system: $(round(DES.get_average_customers_in_system(engine), digits=3))")
println("Average waiting time: $(round(stats.avg_waiting_time, digits=3))")
println("Average service time: $(round(DES.get_average_service_time(engine), digits=3))")
println("Average time in system: $(round(DES.get_average_time_in_system(engine), digits=3))")
println("Server utilization: $(round(DES.get_server_utilization(engine), digits=3))")
println("Effective arrival rate: $(round(stats.arrival_rate, digits=2))")

println("\n=== VARIANCE METRICS ===")
println("Max waiting time: $(round(DES.get_max_waiting_time(engine), digits=2))")
println("Waiting time std dev: $(round(DES.get_waiting_time_std(engine), digits=3))")
println("Waiting time coefficient of variation: $(round(DES.get_waiting_time_std(engine) / DES.get_average_waiting_time(engine), digits=3))")
println("Max service time: $(round(DES.get_max_service_time(engine), digits=2))")
println("Service time std dev: $(round(DES.get_service_time_std(engine), digits=3))")
println("Service time coefficient of variation: $(round(DES.get_service_time_std(engine) / DES.get_average_service_time(engine), digits=3))")
println("Max time in system: $(round(DES.get_max_time_in_system(engine), digits=2))")
println("Time in system std dev: $(round(DES.get_time_in_system_std(engine), digits=3))")
println("Time in system coefficient of variation: $(round(DES.get_time_in_system_std(engine) / DES.get_average_time_in_system(engine), digits=3))")

println("\n=== LITTLE'S LAW VALIDATION ===")
println("L (time-weighted queue length): $(round(stats.littles_law_left, digits=3))")
println("λW (arrival rate × waiting time): $(round(stats.littles_law_right, digits=3))")
println("Little's Law Error |L - λW|: $(round(stats.littles_law_error, digits=6))")
println("Little's Law Status: $(stats.littles_law_error < 0.1 ? "✅ VALID" : "❌ INVALID")")

ρ = 4.5/5.0
μ = 5.0

theoretical_waiting_time = ρ / (μ * (1 - ρ))
theoretical_queue_length = ρ^2/(1-ρ)
theoretical_time_in_system = 1/(μ - 4.5)
theoretical_service_time = 1/μ
theoretical_customers_in_system = ρ/(1-ρ)

println("\n=== THEORETICAL vs SIMULATION COMPARISON ===")
println("Theoretical utilization (ρ = λ/μ): $(ρ)")
println("Theoretical waiting time: $(round(theoretical_waiting_time, digits=3))")
println("Theoretical service time: $(round(theoretical_service_time, digits=3))")
println("Theoretical time in system: $(round(theoretical_time_in_system, digits=3))")
println("Theoretical queue length: $(round(theoretical_queue_length, digits=3))")
println("Theoretical customers in system: $(round(theoretical_customers_in_system, digits=3))")
println("\nSimulation results:")
println("  Server utilization: $(round(DES.get_server_utilization(engine), digits=3))")
println("  Average waiting time: $(round(DES.get_average_waiting_time(engine), digits=3))")
println("  Average service time: $(round(DES.get_average_service_time(engine), digits=3))")
println("  Average time in system: $(round(DES.get_average_time_in_system(engine), digits=3))")
println("  Time-weighted queue length: $(round(stats.time_weighted_queue_length, digits=3))")
println("  Average customers in system: $(round(DES.get_average_customers_in_system(engine), digits=3))")
println("  Simple average queue length: $(round(stats.simple_average_queue_length, digits=3))")
println("  Max waiting time: $(round(DES.get_max_waiting_time(engine), digits=2))")
println("  Max service time: $(round(DES.get_max_service_time(engine), digits=2))")
println("  Max time in system: $(round(DES.get_max_time_in_system(engine), digits=2))")
println("  Waiting time std dev: $(round(DES.get_waiting_time_std(engine), digits=3))")
println("  Service time std dev: $(round(DES.get_service_time_std(engine), digits=3))")
println("  Time in system std dev: $(round(DES.get_time_in_system_std(engine), digits=3))")

println("\n=== ERROR ANALYSIS ===")
println("Utilization error: $(round(abs(DES.get_server_utilization(engine) - ρ), digits=3))")
println("Waiting time error: $(round(abs(DES.get_average_waiting_time(engine) - theoretical_waiting_time), digits=3))")
println("Service time error: $(round(abs(DES.get_average_service_time(engine) - theoretical_service_time), digits=3))")
println("Time in system error: $(round(abs(DES.get_average_time_in_system(engine) - theoretical_time_in_system), digits=3))")
println("Queue length error (time-weighted): $(round(abs(stats.time_weighted_queue_length - theoretical_queue_length), digits=3))")
println("Customers in system error: $(round(abs(DES.get_average_customers_in_system(engine) - theoretical_customers_in_system), digits=3))")
println("Queue length error (simple avg): $(round(abs(stats.simple_average_queue_length - theoretical_queue_length), digits=3))")

queue_history = DES.get_queue_length_history(engine)
println("\n=== QUEUE LENGTH ANALYSIS ===")
println("Total queue length samples: $(length(queue_history))")
println("Max queue length observed: $(maximum(qh[2] for qh in queue_history))")
println("Queue length variance: $(round(var(qh[2] for qh in queue_history), digits=3))")

println("\n=== QUEUE LENGTH OVER TIME (Sample) ===")
println("Time\tQueue Length")
for i in 1:min(20, length(queue_history))
    time, qlen = queue_history[i]
    println("$(round(time, digits=1))\t$qlen")
end
if length(queue_history) > 20
    println("... (showing first 20 samples)")
end

println("\n=== SUMMARY ===")
println("Time-weighted queue length is more accurate than simple average.")
println("Little's Law validation helps ensure simulation correctness.")
println("Note: ρ = 0.25 < 1, so the system should be stable") 