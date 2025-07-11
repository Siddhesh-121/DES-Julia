module DESProject

using DataStructures
using Random
using Distributions

include("DESProject/EventSystem.jl")      
include("DESProject/RNGManager.jl")       
include("DESProject/SimulationEngine.jl") 
include("DESProject/Handlers.jl")         

export Event, EventQueue, SimulationEngine
export ArrivalEvent, DepartureEvent
export schedule_event!, get_next_event!, get_current_time
export run!, stop!, process_event!
export RNGManager, set_arrival_distribution!, set_service_distribution!
export schedule_next_arrival!, schedule_departure!, initialize_simulation!
export next_arrival_time, next_service_time
export get_entities_processed
export get_max_waiting_time, get_waiting_time_std, get_queue_length_history, get_waiting_times
export initialize_queue_tracking!, get_time_weighted_average_queue_length, get_simulation_statistics
export record_departure_time!, get_average_time_in_system, get_max_time_in_system, get_time_in_system_std, get_time_in_system_history
export record_service_time!, get_average_service_time, get_max_service_time, get_service_time_std, get_service_times, get_average_customers_in_system

end 