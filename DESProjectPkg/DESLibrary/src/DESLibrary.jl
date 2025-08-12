module DESLibrary

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

# Backward compatibility for QueueModel
include("Models/QueueModelCompat.jl")

# Core Engine (needs QueueModel)
include("Core/Engine.jl")

# Model helper functions (needs Engine and Events)
include("Models/ModelHelpers.jl")

# Queue model implementations
include("Models/MMCModel.jl")
include("Models/CSMACDModel.jl")

# Utilities
include("Utils/Statistics.jl")
include("Utils/Validation.jl")

# =====================
# Public API Exports
# =====================
export DESEngine, SimulationModel, QueueModel, SimulationResults
export Event, ArrivalEvent, DepartureEvent, ServiceCompletionEvent, GenericEvent, create_generic_event
export MMCModel, MM2Model, MM3Model, MM4Model, MM5Model, CSMACDModel
export simulate!, add_model!, get_results, validate_results, validate_model
export increment_arrivals!
export is_warmup_complete, reset_statistics!
export set_end_time!
export mark_entity_completed!
export initialize_model!, process_event!, finalize_model!, get_statistics, reset_model!, reset_statistics!
export validate_entity_conservation, validate_model_comprehensive
export print_comprehensive_validation_report
export theoretical_mm1, theoretical_mm2, calculate_errors
export print_results, print_comparison, print_validation_report
export validate_littles_law, performance_summary
export get_theoretical_results, has_entities_in_service, count_entities_in_service

# =====================
# Advanced/INTERNAL functions (not exported)
# - Users can access via DESLibrary.function_name if needed
# - Not guaranteed stable, may change in future versions
# =====================
# (Do NOT export: is_empty, get_next_event!, get_current_time, get_model_id, complete_warmup!, all_entities_complete, stop_simulation!, etc.)

end