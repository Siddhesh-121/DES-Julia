# Validation utilities for DES simulations

# Validate simulation results for correctness
function validate_results(results::SimulationResults; tolerance::Float64=0.1)
    validation_report = Dict{Symbol, Bool}()
    
    # Check for non-negative values
    validation_report[:non_negative_times] = (
        results.avg_waiting_time >= 0 &&
        results.avg_service_time >= 0 &&
        results.avg_time_in_system >= 0 &&
        results.max_waiting_time >= 0 &&
        results.max_service_time >= 0
    )
    
    # Check utilization bounds
    validation_report[:utilization_bounds] = (
        0.0 <= results.server_utilization <= 1.0
    )
    
    # Check time relationships
    validation_report[:time_consistency] = (
        results.avg_time_in_system >= results.avg_waiting_time &&
        results.avg_time_in_system >= results.avg_service_time &&
        results.max_waiting_time >= results.avg_waiting_time &&
        results.max_service_time >= results.avg_service_time
    )
    
    # Check queue length bounds
    validation_report[:queue_bounds] = (
        results.avg_queue_length >= 0 &&
        results.max_queue_length >= 0
    )
    
    # Check if entities were actually processed
    validation_report[:entities_processed] = results.entities_processed > 0
    
    # Check simulation time
    validation_report[:simulation_time] = results.simulation_time > 0
    
    # Overall validity
    all_valid = all(values(validation_report))
    
    return (
        is_valid = all_valid,
        checks = validation_report
    )
end

# Validate specific queue model parameters
function validate_mm1_parameters(λ::Float64, μ::Float64)
    checks = Dict{Symbol, Bool}()
    
    checks[:positive_arrival_rate] = λ > 0
    checks[:positive_service_rate] = μ > 0
    checks[:stable_system] = λ < μ
    
    return (
        is_valid = all(values(checks)),
        checks = checks,
        utilization = λ / μ
    )
end

function validate_mm2_parameters(λ::Float64, μ::Float64)
    checks = Dict{Symbol, Bool}()
    
    checks[:positive_arrival_rate] = λ > 0
    checks[:positive_service_rate] = μ > 0
    checks[:stable_system] = λ < 2 * μ  # Two servers
    
    return (
        is_valid = all(values(checks)),
        checks = checks,
        utilization_per_server = λ / (2 * μ)
    )
end

# Comprehensive model validation
function validate_model(engine::DESEngine, model_id::Symbol; 
                       expected_entities::Int=1000, 
                       tolerance::Float64=0.1)
    
    if !haskey(engine.models, model_id)
        return (is_valid = false, error = "Model $model_id not found")
    end
    
    model = engine.models[model_id]
    
    # Get simulation results
    results = get_statistics(model, engine)
    
    # Validate basic results
    basic_validation = validate_results(results, tolerance=tolerance)
    
    if !basic_validation.is_valid
        return (
            is_valid = false,
            error = "Basic validation failed",
            details = basic_validation.checks
        )
    end
    
    # FIXED: Check if expected number of entities were processed (accounting for warmup)
    # The total entities processed should equal expected_entities
    entity_check = abs(engine.entities_processed - expected_entities) <= 1
    
    # Model-specific validations
    model_validation = validate_model_specific(model, results, tolerance)
    
    return (
        is_valid = basic_validation.is_valid && entity_check && model_validation.is_valid,
        basic_checks = basic_validation.checks,
        entity_count_correct = entity_check,
        model_specific = model_validation
    )
end

# Model-specific validation

# MM2Model-specific validation removed - now using MMCModel

# Default validation for unknown model types
function validate_model_specific(model::QueueModel, results::SimulationResults, tolerance::Float64)
    # Basic validation only
    littles_law = validate_littles_law(results, 1.0)  # Dummy arrival rate
    
    return (
        is_valid = true,  # Assume valid for unknown models
        note = "Model-specific validation not implemented for $(typeof(model))"
    )
end

# FIXED: Add entity conservation validation
function validate_entity_conservation(engine::DESEngine, model_id::Symbol, expected_entities::Int)
    if !haskey(engine.models, model_id)
        return (is_valid = false, error = "Model $model_id not found")
    end
    
    model = engine.models[model_id]
    
    # Check that all entities that arrived also departed
    arrivals = engine.entities_arrived
    departures = engine.entities_processed
    
    # Check that no entities remain in the system
    entities_in_queue = length(model.queue)
    entities_being_served = count_entities_in_service(model)
    entities_in_system = entities_in_queue + entities_being_served
    
    # Entity conservation check
    conservation_valid = (arrivals == departures + entities_in_system)
    
    # Expected count check
    expected_count_valid = (arrivals == expected_entities)
    
    # Complete processing check (all entities should have left the system)
    complete_processing_valid = (entities_in_system == 0)
    
    return (
        is_valid = conservation_valid && expected_count_valid && complete_processing_valid,
        arrivals = arrivals,
        departures = departures,
        entities_in_queue = entities_in_queue,
        entities_being_served = entities_being_served,
        entities_in_system = entities_in_system,
        expected_entities = expected_entities,
        conservation_valid = conservation_valid,
        expected_count_valid = expected_count_valid,
        complete_processing_valid = complete_processing_valid
    )
end

# Comprehensive validation with entity conservation
function validate_model_comprehensive(engine::DESEngine, model_id::Symbol; 
                                   expected_entities::Int=1000, 
                                   tolerance::Float64=0.1)
    
    # Basic model validation
    basic_validation = validate_model(engine, model_id, expected_entities=expected_entities, tolerance=tolerance)
    
    # Entity conservation validation
    conservation_validation = validate_entity_conservation(engine, model_id, expected_entities)
    
    # Combined validation
    is_valid = basic_validation.is_valid && conservation_validation.is_valid
    
    return (
        is_valid = is_valid,
        basic_validation = basic_validation,
        conservation_validation = conservation_validation
    )
end

# Print validation report
function print_validation_report(validation_result)
    println("=== VALIDATION REPORT ===")
    
    if validation_result.is_valid
        println("✅ VALIDATION PASSED")
    else
        println("❌ VALIDATION FAILED")
        if haskey(validation_result, :error)
            println("Error: $(validation_result.error)")
        end
    end
    
    # Print detailed checks
    if haskey(validation_result, :basic_checks)
        println("\nBasic Checks:")
        for (check, passed) in validation_result.basic_checks
            status = passed ? "✅" : "❌"
            println("  $check: $status")
        end
    end
    
    if haskey(validation_result, :entity_count_correct)
        status = validation_result.entity_count_correct ? "✅" : "❌"
        println("Entity count correct: $status")
    end
    
    if haskey(validation_result, :model_specific)
        println("\nModel-specific validation:")
        model_val = validation_result.model_specific
        
        if haskey(model_val, :littles_law)
            ll = model_val.littles_law
            status = ll.is_valid ? "✅" : "❌"
            println("  Little's Law: $status (error: $(round(ll.relative_error * 100, digits=2))%)")
        end
        
        if haskey(model_val, :within_tolerance)
            status = model_val.within_tolerance ? "✅" : "❌"
            println("  Theoretical comparison: $status")
        end
        
        if haskey(model_val, :server_balance)
            status = model_val.server_balance ? "✅" : "❌"
            println("  Server balance: $status")
        end
    end
    
    println()
end

# FIXED: Print comprehensive validation report with entity conservation
function print_comprehensive_validation_report(validation_result)
    println("=== COMPREHENSIVE VALIDATION REPORT ===")
    
    if validation_result.is_valid
        println("✅ ALL VALIDATIONS PASSED")
    else
        println("❌ VALIDATION FAILED")
    end
    
    # Print basic validation
    if haskey(validation_result, :basic_validation)
        println("\n📊 Basic Validation:")
        basic = validation_result.basic_validation
        status = basic.is_valid ? "✅" : "❌"
        println("  Overall: $status")
        
        if haskey(basic, :basic_checks)
            for (check, passed) in basic.basic_checks
                status = passed ? "✅" : "❌"
                println("    $check: $status")
            end
        end
    end
    
    # Print entity conservation validation
    if haskey(validation_result, :conservation_validation)
        println("\n🔄 Entity Conservation Validation:")
        conserv = validation_result.conservation_validation
        status = conserv.is_valid ? "✅" : "❌"
        println("  Overall: $status")
        
        println("  Arrivals: $(conserv.arrivals)")
        println("  Departures: $(conserv.departures)")
        println("  Entities in queue: $(conserv.entities_in_queue)")
        println("  Entities being served: $(conserv.entities_being_served)")
        println("  Total entities in system: $(conserv.entities_in_system)")
        println("  Expected entities: $(conserv.expected_entities)")
        
        status = conserv.conservation_valid ? "✅" : "❌"
        println("  Conservation (arrivals = departures + in_system): $status")
        
        status = conserv.expected_count_valid ? "✅" : "❌"
        println("  Expected count (arrivals = expected): $status")
        
        status = conserv.complete_processing_valid ? "✅" : "❌"
        println("  Complete processing (in_system = 0): $status")
    end
    
    println()
end 