module DESLib

# High-performance imports
using DataStructures
using Random
using Distributions
using Statistics
using SpecialFunctions  # For numerical stability in MMC calculations

# Core DES components (order matters for dependencies)
include("Core/Events.jl")
include("Core/EventQueue.jl")
include("Core/RNG.jl")

# Queue model abstractions (needed before Engine)
include("Models/AbstractModel.jl")

# Core Engine (needs QueueModel)
include("Core/Engine.jl")

# Model helper functions (needs Engine and Events)
include("Models/ModelHelpers.jl")

# Queue model implementations
include("Models/MM1Model.jl")
include("Models/MMCModel.jl")

# Utilities
include("Utils/Statistics.jl")
include("Utils/Validation.jl")

# Export core types
export DESEngine, QueueModel, SimulationResults

# Export events
export Event, ArrivalEvent, DepartureEvent, ServiceCompletionEvent

# Export models
export MM1Model, MMCModel, MM2Model, MM3Model, MM4Model, MM5Model

# Export functions
export simulate!, add_model!, get_results, validate_results, validate_model
export increment_arrivals!  # For tracking arrivals in models
export is_warmup_complete, reset_statistics!  # For warmup period support

# FIXED: Export new validation functions
export validate_entity_conservation, validate_model_comprehensive
export print_comprehensive_validation_report

# Export utilities
export theoretical_mm1, theoretical_mm2, calculate_errors
export print_results, print_comparison, print_validation_report
export validate_littles_law, performance_summary
export get_theoretical_results, has_entities_in_service, count_entities_in_service

end 