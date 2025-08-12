abstract type SimulationModel end

struct SimulationResults
    avg_waiting_time::Float64
    avg_service_time::Float64
    avg_time_in_system::Float64
    server_utilization::Float64
    avg_queue_length::Float64
    max_queue_length::Int
    waiting_time_std::Float64
    service_time_std::Float64
    max_waiting_time::Float64
    max_service_time::Float64
    entities_processed::Int
    simulation_time::Float64
    extra_metrics::Dict{Symbol, Float64}
    function SimulationResults(
        avg_waiting_time::Float64,
        avg_service_time::Float64,
        avg_time_in_system::Float64,
        server_utilization::Float64,
        avg_queue_length::Float64,
        max_queue_length::Int,
        waiting_time_std::Float64,
        service_time_std::Float64,
        max_waiting_time::Float64,
        max_service_time::Float64,
        entities_processed::Int,
        simulation_time::Float64,
        extra_metrics::Dict{Symbol, Float64} = Dict{Symbol, Float64}()
    )
        new(
            avg_waiting_time, avg_service_time, avg_time_in_system,
            server_utilization, avg_queue_length, max_queue_length,
            waiting_time_std, service_time_std, max_waiting_time, max_service_time,
            entities_processed, simulation_time, extra_metrics
        )
    end
end

function initialize_model! end
function process_event! end
function finalize_model! end
function get_statistics end
function reset_model! end
function reset_statistics! end
function get_theoretical_results end
function validate_model end 