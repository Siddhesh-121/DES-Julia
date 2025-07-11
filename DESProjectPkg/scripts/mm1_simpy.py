import simpy
import random
import numpy as np
import time

def run_mm1_simulation(λ, μ, num_customers):
    env = simpy.Environment()
    server = simpy.Resource(env, capacity=1)
    
    waiting_times = []
    service_times = []
    times_in_system = []
    
    def customer(name):
        arrive = env.now
        with server.request() as req:
            yield req
            
            wait = env.now - arrive
            waiting_times.append(wait)
            
            service_time = random.expovariate(μ)
            service_times.append(service_time)
            yield env.timeout(service_time)
            
            total_time = env.now - arrive
            times_in_system.append(total_time)

    def arrival_process():
        for i in range(num_customers):
            env.process(customer(f"Customer {i+1}"))
            interarrival_time = random.expovariate(λ)
            yield env.timeout(interarrival_time)

    env.process(arrival_process())
    env.run()
    
    return waiting_times, service_times, times_in_system, env.now

λ = 4.5
μ = 5.0
num_customers = 100000
random.seed(2024)

# Add timing here
start_time = time.perf_counter()
waiting_times, service_times, times_in_system, actual_simulation_time = run_mm1_simulation(λ, μ, num_customers)
end_time = time.perf_counter()
execution_time = end_time - start_time

print(f"SimPy execution time: {execution_time:.4f} seconds")

if waiting_times and service_times and times_in_system:
    avg_wait_time = np.mean(waiting_times)
    avg_service_time = np.mean(service_times) 
    avg_time_in_system = np.mean(times_in_system)
    max_wait_time = max(waiting_times)
    max_time_in_system = max(times_in_system)
    wait_time_std = np.std(waiting_times, ddof=1)
    time_in_system_std = np.std(times_in_system, ddof=1)
    wait_time_cv = wait_time_std / avg_wait_time if avg_wait_time > 0 else 0
    time_in_system_cv = time_in_system_std / avg_time_in_system if avg_time_in_system > 0 else 0
    
    total_service_time = sum(service_times)
    server_utilization = total_service_time / actual_simulation_time if actual_simulation_time > 0 else 0
    
    approx_simulation_time = total_service_time / (λ/μ)
    approx_utilization = total_service_time / approx_simulation_time if approx_simulation_time > 0 else 0
    
    effective_arrival_rate = num_customers / actual_simulation_time if actual_simulation_time > 0 else 0
    
    arrival_rate = λ
    avg_queue_length = arrival_rate * avg_wait_time
    avg_customers_in_system = arrival_rate * avg_time_in_system
    
    ρ = λ / μ
    theoretical_wait_time = ρ / (μ * (1 - ρ)) if ρ < 1 else float('inf')
    theoretical_time_in_system = 1 / (μ - λ) if μ > λ else float('inf')
    theoretical_queue_length = (ρ**2) / (1 - ρ) if ρ < 1 else float('inf')
    theoretical_customers_in_system = ρ / (1 - ρ) if ρ < 1 else float('inf')
    
    print("--- SimPy M/M/1 Comprehensive Results ---")
    print(f"SIMULATION RESULTS:")
    print(f"Server Utilization: {server_utilization:.3f}")
    print(f"Average Queue Length: {avg_queue_length:.3f}")
    print(f"Average Customers in System: {avg_customers_in_system:.3f}")
    print(f"Average Waiting Time: {avg_wait_time:.3f}")
    print(f"Average Service Time: {avg_service_time:.3f}")
    print(f"Average Time in System: {avg_time_in_system:.3f}")
    print(f"Max Waiting Time: {max_wait_time:.3f}")
    print(f"Max Time in System: {max_time_in_system:.3f}")
    print(f"Waiting Time Std Dev: {wait_time_std:.3f}")
    print(f"Time in System Std Dev: {time_in_system_std:.3f}")
    print(f"Waiting Time CV: {wait_time_cv:.3f}")
    print(f"Time in System CV: {time_in_system_cv:.3f}")
    print(f"Actual Simulation Time: {actual_simulation_time:.3f}")
    print(f"Effective Arrival Rate: {effective_arrival_rate:.3f}")
    
    print(f"\nSIMULATION TIME COMPARISON:")
    print(f"Actual Simulation Time: {actual_simulation_time:.3f}")
    print(f"Approximated Time: {approx_simulation_time:.3f}")
    print(f"Time Estimation Error: {abs(actual_simulation_time - approx_simulation_time):.3f}")
    print(f"Actual Utilization: {server_utilization:.3f}")
    print(f"Approximated Utilization: {approx_utilization:.3f}")
    print(f"Utilization Estimation Error: {abs(server_utilization - approx_utilization):.3f}")
    
    print(f"\nTHEORETICAL M/M/1 VALUES:")
    print(f"Theoretical Utilization (rho): {ρ:.3f}")
    print(f"Theoretical Avg Queue Length: {theoretical_queue_length:.3f}")
    print(f"Theoretical Avg Customers in System: {theoretical_customers_in_system:.3f}")
    print(f"Theoretical Avg Waiting Time: {theoretical_wait_time:.3f}")
    print(f"Theoretical Avg Service Time: {1/μ:.3f}")
    print(f"Theoretical Avg Time in System: {theoretical_time_in_system:.3f}")
    
    print(f"\nVERIFICATION:")
    print(f"Utilization Error: {abs(server_utilization - ρ):.3f}")
    print(f"Queue Length Error: {abs(avg_queue_length - theoretical_queue_length):.3f}")
    print(f"Customers in System Error: {abs(avg_customers_in_system - theoretical_customers_in_system):.3f}")
    print(f"Waiting Time Error: {abs(avg_wait_time - theoretical_wait_time):.3f}")
    print(f"Service Time Error: {abs(avg_service_time - 1/μ):.3f}")
    print(f"Time in System Error: {abs(avg_time_in_system - theoretical_time_in_system):.3f}")
else:
    print("No data collected during simulation.")
