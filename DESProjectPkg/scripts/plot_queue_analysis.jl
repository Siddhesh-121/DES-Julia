using Pkg
Pkg.activate(".")

try
    using Plots
catch
    Pkg.add("Plots")
    using Plots
end

using Statistics

import DESProject as DES

println("Creating simulation components...")
engine = DES.SimulationEngine(500)
rng_manager = DES.RNGManager(2024)

println("Initializing simulation...")
DES.initialize_simulation!(engine.event_queue, rng_manager)

println("Starting simulation...")
DES.run!(engine, rng_manager)

println("\nSimulation completed. Creating plots...")

queue_history = DES.get_queue_length_history(engine)
times = [qh[1] for qh in queue_history]
queue_lengths = [qh[2] for qh in queue_history]

waiting_times = DES.get_waiting_times(engine)

p1 = plot(times, queue_lengths, 
    label="Queue Length", 
    xlabel="Time", 
    ylabel="Queue Length",
    title="Queue Length Over Time",
    linewidth=1,
    markersize=2)

p2 = histogram(waiting_times, 
    label="Waiting Times", 
    xlabel="Waiting Time", 
    ylabel="Frequency",
    title="Waiting Time Distribution",
    bins=30)

λ = 4.5
μ = 5.0
ρ = λ / μ
theoretical_wait = ρ / (μ * (1 - ρ))
vline!([theoretical_wait], label="Theoretical Mean", color=:red, linestyle=:dash)

plot(p1, p2, layout=(2,1), size=(800, 600))

savefig("queue_analysis.png")
println("Plot saved as 'queue_analysis.png'")

println("\n=== SUMMARY STATISTICS ===")
println("Max queue length: $(maximum(queue_lengths))")
println("Queue length variance: $(round(var(queue_lengths), digits=3))")
println("Max waiting time: $(round(DES.get_max_waiting_time(engine), digits=2))")
println("Waiting time std dev: $(round(DES.get_waiting_time_std(engine), digits=2))")
println("Waiting time CV: $(round(DES.get_waiting_time_std(engine) / DES.get_average_waiting_time(engine), digits=3))") 