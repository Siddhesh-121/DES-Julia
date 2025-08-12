#!/usr/bin/env julia

"""
COMPREHENSIVE DESLibrary Precompilation Workload
Exercises ALL core functions and logic for maximum sysimage benefits.
"""

println("🔥 COMPREHENSIVE DESLibrary Precompilation Starting...")
println("=" ^ 60)

# Load the complete library
include("../src/DESLibrary.jl")
using .DESLibrary

# Load all statistical packages that will be used
using Base.Threads, Statistics, HypothesisTests, Distributions, Random, Printf, SpecialFunctions

println("📦 Core packages loaded")

# =====================================================
# SECTION 1: Core Engine Functions
# =====================================================
println("🔧 Precompiling Core Engine Functions...")

# Engine creation with different parameters
engines = [
    DESEngine(1000, 2024),
    DESEngine(5000, 2025, 0.1),
    DESEngine(10000, 2026, 0.2)
]

# Test engine operations
for (i, engine) in enumerate(engines)
    # Test time operations
    current_time = DESLibrary.get_current_time(engine)
    
    # Test entity operations  
    DESLibrary.mark_entity_completed!(engine, 1)
    completed = engine.entities_processed
    
    # Test warmup operations
    warmup_status = is_warmup_complete(engine)
    
    println("  ✓ Engine $i operations precompiled")
end

# =====================================================
# SECTION 2: Event System
# =====================================================
println("🎯 Precompiling Event System...")

# Create different types of events
arrival_event = DESLibrary.create_arrival(1, :test, 1.0)
departure_event = DESLibrary.create_departure(2, :test, 2.0)
service_event = DESLibrary.create_service_completion(3, :test, 3.0)
generic_event = create_generic_event(4.0, 4, :test, :custom_event)

# Test event operations
events = [arrival_event, departure_event, service_event, generic_event]
for event in events
    event_time = DESLibrary.get_time(event)
    entity_id = DESLibrary.get_entity_id(event)
end

println("  ✓ Event system precompiled")

# =====================================================
# SECTION 3: MMC Model - Complete Workflow
# =====================================================
println("📊 Precompiling MMC Models (Complete Workflows)...")

# Test different MMC configurations
mmc_configs = [
    (λ=5.0, μ=2.0, c=3),
    (λ=10.0, μ=3.0, c=4), 
    (λ=15.0, μ=4.0, c=5),
    (λ=8.0, μ=2.5, c=6)
]

for (i, (λ, μ, c)) in enumerate(mmc_configs)
    println("  🔄 MMC Configuration $i: M/M/$c (λ=$λ, μ=$μ)")
    
    # Create engine and model
    engine = DESEngine(2000, 2024 + i, 0.1)
    model_id = Symbol("mmc_$i")
    model = MMCModel(λ, μ, c, model_id)
    
    # Full workflow
    add_model!(engine, model_id, model)
    initialize_model!(model, engine)
    simulate!(engine, verbose=false)
    
    # Results and statistics
    results = get_results(engine)[model_id]
    stats = get_statistics(model, engine)
    
    # Validation operations
    validation_result = validate_model(engine, model_id)
    
    # Reset operations
    reset_statistics!(model, engine)
    reset_model!(model)
    
    println("    ✓ M/M/$c workflow precompiled")
end

# =====================================================
# SECTION 4: CSMA/CD Model - Complete Workflow  
# =====================================================
println("📡 Precompiling CSMA/CD Models...")

# Test different CSMA/CD configurations
csmacd_configs = [
    (nodes=3, λ=5.0, tx_time=0.01, prop_delay=0.005, slot_time=0.005),
    (nodes=5, λ=8.0, tx_time=0.02, prop_delay=0.008, slot_time=0.008),
    (nodes=10, λ=10.0, tx_time=0.015, prop_delay=0.006, slot_time=0.006)
]

for (i, config) in enumerate(csmacd_configs)
    println("  🔄 CSMA/CD Configuration $i: $(config.nodes) nodes")
    
    # Create engine and model
    engine = DESEngine(1000, 3000 + i)
    model_id = Symbol("csmacd_$i") 
    model = CSMACDModel(config.nodes, config.λ, config.tx_time, 
                       config.prop_delay, config.slot_time, model_id)
    
    # Full workflow
    add_model!(engine, model_id, model)
    simulate!(engine, verbose=false)
    
    # Results and statistics
    results = get_results(engine)[model_id]
    
    println("    ✓ $(config.nodes)-node CSMA/CD precompiled")
end

# =====================================================
# SECTION 5: Statistical Analysis Functions
# =====================================================
println("📈 Precompiling Statistical Analysis...")

# Generate sample data from simulation results
sample_waiting_times = Float64[]
sample_utilizations = Float64[]
sample_queue_lengths = Float64[]

# Run multiple small simulations to get varied data
for i in 1:10
    engine = DESEngine(500, 4000 + i)
    model = MMCModel(5.0 + i, 2.0, 3, Symbol("stats_$i"))
    add_model!(engine, Symbol("stats_$i"), model)
    simulate!(engine, verbose=false)
    
    results = get_results(engine)[Symbol("stats_$i")]
    push!(sample_waiting_times, results.avg_waiting_time)
    push!(sample_utilizations, results.server_utilization)
    push!(sample_queue_lengths, results.avg_queue_length)
end

# Exercise statistical functions
println("  🧮 Basic statistics...")
waiting_mean = mean(sample_waiting_times)
waiting_std = std(sample_waiting_times)
waiting_median = median(sample_waiting_times)
waiting_cv = waiting_std / waiting_mean

util_mean = mean(sample_utilizations)
util_std = std(sample_utilizations)

# Exercise hypothesis testing
println("  🧪 Hypothesis testing...")
theoretical_waiting = 1.0 / (3 * 2.0 - 5.0)  # M/M/3 theoretical

# One-sample t-test
t_test = OneSampleTTest(sample_waiting_times, theoretical_waiting)
t_pvalue = pvalue(t_test)
t_statistic = t_test.t
# t_confidence = confint(t_test)  # Skip confint to avoid potential issues

# Two-sample tests
if length(sample_waiting_times) >= 4
    mid = div(length(sample_waiting_times), 2)
    group1 = sample_waiting_times[1:mid]
    group2 = sample_waiting_times[mid+1:end]
    two_sample_test = UnequalVarianceTTest(group1, group2)
    two_sample_p = pvalue(two_sample_test)
end

# Normality testing
if length(sample_waiting_times) >= 8
    normal_test = ExactOneSampleKSTest(sample_waiting_times, 
                                      Normal(waiting_mean, waiting_std))
    normal_p = pvalue(normal_test)
end

# Distribution fitting (basic)
# fitted_normal = fit(Normal, sample_waiting_times)  # Skip fit to avoid issues
# fitted_exp = fit(Exponential, sample_waiting_times)  # Skip fit to avoid issues
normal_dist = Normal(waiting_mean, waiting_std)
exp_dist = Exponential(waiting_mean)

println("  ✓ Statistical analysis precompiled")

# =====================================================
# SECTION 6: Validation Functions
# =====================================================
println("✅ Precompiling Validation Functions...")

# Test validation on different models
engine = DESEngine(1000, 5000)
model = MMCModel(8.0, 3.0, 4, :validation_test)
add_model!(engine, :validation_test, model)
simulate!(engine, verbose=false)

# Exercise validation functions
validation_basic = validate_model(engine, :validation_test)
validation_comprehensive = validate_model_comprehensive(engine, :validation_test)
conservation_check = validate_entity_conservation(engine, :validation_test, 1000)

println("  ✓ Validation functions precompiled")

# =====================================================
# SECTION 7: Batch Simulation Patterns
# =====================================================
println("🔄 Precompiling Batch Simulation Patterns...")

# Simulate batch processing patterns
batch_results = Vector{Float64}()
batch_configs = [
    (λ=5.0, μ=2.0, c=3),
    (λ=7.0, μ=2.5, c=3),
    (λ=9.0, μ=3.0, c=4)
]

for (i, (λ, μ, c)) in enumerate(batch_configs)
    local engine = DESEngine(800, 6000 + i, 0.1)
    local model = MMCModel(λ, μ, c, Symbol("batch_$i"))
    add_model!(engine, Symbol("batch_$i"), model)
    simulate!(engine, verbose=false)
    
    results = get_results(engine)[Symbol("batch_$i")]
    push!(batch_results, results.avg_waiting_time)
end

# Batch statistical analysis
batch_mean = mean(batch_results)
batch_std = std(batch_results)
batch_cv = batch_std / batch_mean

# Batch hypothesis testing
if length(batch_results) >= 3
    batch_theoretical = 1.0 / (3 * 2.0 - 5.0)
    batch_test = OneSampleTTest(batch_results, batch_theoretical)
    batch_p = pvalue(batch_test)
end

println("  ✓ Batch simulation patterns precompiled")

# =====================================================
# SECTION 8: Threading Patterns (if available)
# =====================================================
if nthreads() > 1
    println("🧵 Precompiling Threading Patterns...")
    
    # Simulate multi-threaded batch patterns
    thread_results = Vector{Float64}(undef, min(4, nthreads()))
    
    @threads for i in 1:min(4, nthreads())
        engine = DESEngine(300, 7000 + i * 100)
        model = MMCModel(3.0 + i, 1.5, 2, Symbol("thread_$i"))
        add_model!(engine, Symbol("thread_$i"), model)
        simulate!(engine, verbose=false)
        
        results = get_results(engine)[Symbol("thread_$i")]
        thread_results[i] = results.server_utilization
    end
    
    # Thread-safe statistical operations
    thread_mean = mean(thread_results)
    thread_std = std(thread_results)
    
    println("  ✓ Threading patterns precompiled")
else
    println("  ⚠️ Single-threaded environment - skipping thread patterns")
end

# =====================================================
# SECTION 9: Output and Formatting Functions
# =====================================================
println("📄 Precompiling Output Functions...")

# Test output formatting with sample results
local engine = DESEngine(500, 8000)
local model = MMCModel(6.0, 2.0, 3, :output_test)
add_model!(engine, :output_test, model)
simulate!(engine, verbose=false)
results = get_results(engine)[:output_test]

# Exercise printing functions
print_results(results, "Test Model")
print_results(results)

# Exercise validation reporting
validation_result = validate_model(engine, :output_test)

println("  ✓ Output functions precompiled")

# =====================================================
# FINAL SUMMARY
# =====================================================
println("\n✅ COMPREHENSIVE PRECOMPILATION COMPLETE!")
println("=" ^ 60)
println("Precompiled components:")
println("  ✓ Core Engine operations (3 configurations)")
println("  ✓ Event system (all event types)")
println("  ✓ MMC Models (4 complete workflows)")  
println("  ✓ CSMA/CD Models (3 configurations)")
println("  ✓ Statistical analysis (hypothesis tests, distributions)")
println("  ✓ Validation functions (basic + comprehensive)")
println("  ✓ Batch simulation patterns")
if nthreads() > 1
    println("  ✓ Multi-threading patterns")
end
println("  ✓ Output and formatting functions")
println("\n🚀 DESLibrary is now FULLY precompiled for maximum performance!")