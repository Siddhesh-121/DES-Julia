#!/usr/bin/env python3

import simpy
import numpy as np
import time
import math
from statistics import mean, stdev
from collections import deque

LAMBDA = 20.0
MU = 2.0
C = 15
MAX_ENTITIES = 200000
WARMUP_FRACTION = 0.1
SEED = 2024

RHO = LAMBDA / MU
RHO_PER_SERVER = LAMBDA / (C * MU)

class MMCStatistics:
    def __init__(self, warmup_entities):
        self.warmup_entities = warmup_entities
        self.warmup_completed = False
        
        self.all_waiting_times = []
        self.all_service_times = []
        self.all_system_times = []
        self.all_arrival_times = []
        self.all_departure_times = []
        
        self.waiting_times = []
        self.service_times = []
        self.system_times = []
        self.arrival_times = []
        self.departure_times = []
        
        self.queue_length_samples = []
        self.queue_length_times = []
        self.queue_area = 0.0
        self.queue_area_stats = 0.0
        self.last_queue_time = 0.0
        self.last_queue_size = 0
        
        self.server_busy_times = [0.0] * C
        self.server_busy_times_stats = [0.0] * C
        self.server_start_times = [0.0] * C
        self.servers_busy = [False] * C
        
        self.total_entities = 0
        self.statistics_entities = 0
        self.max_queue_length = 0
        
    def record_arrival(self, env, entity_id):
        current_time = env.now
        self.all_arrival_times.append(current_time)
        self.total_entities += 1
        
        if self.warmup_completed:
            self.arrival_times.append(current_time)
            
        if not self.warmup_completed and self.total_entities >= self.warmup_entities:
            self.warmup_completed = True
            self.warmup_completion_time = current_time
            print(f"    Warmup completed at time {current_time:.2f} after {self.total_entities} entities")
    
    def record_service_start(self, env, entity_id, server_id):
        current_time = env.now
        self.servers_busy[server_id] = True
        self.server_start_times[server_id] = current_time
        
    def record_service_end(self, env, entity_id, server_id, arrival_time, service_time):
        current_time = env.now
        waiting_time = current_time - arrival_time - service_time
        system_time = current_time - arrival_time
        
        self.server_busy_times[server_id] += service_time
        if self.warmup_completed:
            self.server_busy_times_stats[server_id] += service_time
        self.servers_busy[server_id] = False
        
        self.all_waiting_times.append(waiting_time)
        self.all_service_times.append(service_time)
        self.all_system_times.append(system_time)
        self.all_departure_times.append(current_time)
        
        if self.warmup_completed:
            self.waiting_times.append(waiting_time)
            self.service_times.append(service_time)
            self.system_times.append(system_time)
            self.departure_times.append(current_time)
            self.statistics_entities += 1
    
    def record_queue_length(self, env, queue_length):
        current_time = env.now
        
        if hasattr(self, 'last_queue_time'):
            time_duration = current_time - self.last_queue_time
            self.queue_area += self.last_queue_size * time_duration
            if self.warmup_completed:
                self.queue_area_stats += self.last_queue_size * time_duration
        
        self.queue_length_samples.append(queue_length)
        self.queue_length_times.append(current_time)
        self.last_queue_time = current_time
        self.last_queue_size = queue_length
        self.max_queue_length = max(self.max_queue_length, queue_length)

def customer(env, customer_id, server, stats, arrival_time):
    stats.record_arrival(env, customer_id)
    
    with server.request() as request:
        queue_length = len(server.queue)
        stats.record_queue_length(env, queue_length)
        
        yield request
        
        queue_length = len(server.queue)
        stats.record_queue_length(env, queue_length)
        
        server_id = customer_id % C
        stats.record_service_start(env, customer_id, server_id)
        
        service_time = np.random.exponential(1.0 / MU)
        
        yield env.timeout(service_time)
        
        stats.record_service_end(env, customer_id, server_id, arrival_time, service_time)

def arrival_process(env, server, stats):
    customer_id = 0
    
    while customer_id < MAX_ENTITIES:
        inter_arrival = np.random.exponential(1.0 / LAMBDA)
        yield env.timeout(inter_arrival)
        
        arrival_time = env.now
        env.process(customer(env, customer_id, server, stats, arrival_time))
        customer_id += 1

def calculate_theoretical_values():
    if RHO_PER_SERVER >= 1.0:
        return {
            'waiting_time': float('inf'),
            'service_time': 1.0 / MU,
            'system_time': float('inf'),
            'queue_length': float('inf'),
            'utilization': 1.0,
            'erlang_c': 1.0
        }
    
    theoretical_service_time = 1.0 / MU
    theoretical_utilization = RHO_PER_SERVER
    
    if C == 1:
        erlang_c = RHO_PER_SERVER
    else:
        sum_term = sum((RHO ** k) / math.factorial(k) for k in range(C))
        erlang_c_numerator = (RHO ** C) / math.factorial(C)
        erlang_c_denominator = sum_term + erlang_c_numerator / (1 - RHO_PER_SERVER)
        erlang_c = erlang_c_numerator / (erlang_c_denominator * (1 - RHO_PER_SERVER))
    
    theoretical_waiting_time = erlang_c / (C * MU * (1 - RHO_PER_SERVER))
    theoretical_system_time = theoretical_waiting_time + theoretical_service_time
    theoretical_queue_length = erlang_c * RHO_PER_SERVER / (1 - RHO_PER_SERVER)
    
    return {
        'waiting_time': theoretical_waiting_time,
        'service_time': theoretical_service_time,
        'system_time': theoretical_system_time,
        'queue_length': theoretical_queue_length,
        'utilization': theoretical_utilization,
        'erlang_c': erlang_c
    }

def main():
    print("M/M/C Queue Simulation using SimPy")
    print("=" * 50)
    print("Configuration:")
    print(f"  Arrival rate (lambda): {LAMBDA} entities/time")
    print(f"  Service rate (mu): {MU} entities/time per server")
    print(f"  Number of servers (c): {C}")
    print(f"  Total entities: {MAX_ENTITIES}")
    print(f"  Warmup period: {WARMUP_FRACTION*100:.1f}%")
    print(f"  Traffic intensity (ρ): {RHO:.3f}")
    print(f"  Traffic intensity per server (ρ/c): {RHO_PER_SERVER:.3f}")
    print(f"  System status: {'Stable' if RHO_PER_SERVER < 1.0 else 'Unstable'}")
    
    if RHO_PER_SERVER >= 1.0:
        print("Warning: System is unstable (ρ/c ≥ 1.0). Results may be unreliable.")
    
    print("\nStarting simulation...")
    
    np.random.seed(SEED)
    
    env = simpy.Environment()
    
    server = simpy.Resource(env, capacity=C)
    
    warmup_entities = max(1, int(MAX_ENTITIES * WARMUP_FRACTION))
    stats = MMCStatistics(warmup_entities)
    
    env.process(arrival_process(env, server, stats))
    
    start_time = time.time()
    env.run()
    end_time = time.time()
    wall_clock_time = end_time - start_time
    
    stats.record_queue_length(env, len(server.queue))
    
    simulation_time = env.now
    stats_collection_time = simulation_time - (stats.warmup_completion_time if stats.warmup_completed else 0)
    
    total_throughput = stats.total_entities / wall_clock_time
    stats_throughput = stats.statistics_entities / wall_clock_time
    events_per_sec = stats.total_entities * 2 / wall_clock_time
    
    print("\nPERFORMANCE RESULTS")
    print("=" * 50)
    print(f"  Wall-clock time: {wall_clock_time * 1000:.2f} ms")
    print(f"  Simulation time: {simulation_time:.2f} time units")
    print(f"  Total entities processed: {stats.total_entities}")
    print(f"  Entities for statistics: {stats.statistics_entities} (after warmup)")
    print(f"  Warmup entities: {warmup_entities}")
    print(f"  Total throughput: {total_throughput:.0f} entities/sec")
    print(f"  Statistics throughput: {stats_throughput:.0f} entities/sec")
    print(f"  Event processing rate: {events_per_sec:.0f} events/sec")
    
    avg_waiting_time = mean(stats.waiting_times) if stats.waiting_times else 0.0
    avg_service_time = mean(stats.service_times) if stats.service_times else 0.0
    avg_system_time = mean(stats.system_times) if stats.system_times else 0.0
    avg_queue_length = stats.queue_area_stats / stats_collection_time if stats_collection_time > 0 else 0.0
    
    total_server_busy_time = sum(stats.server_busy_times_stats)
    server_utilization = total_server_busy_time / (C * stats_collection_time) if stats_collection_time > 0 else 0.0
    
    print("\nSIMULATION STATISTICS")
    print("=" * 50)
    print(f"  Average waiting time: {avg_waiting_time:.4f} time units")
    print(f"  Average service time: {avg_service_time:.4f} time units")
    print(f"  Average time in system: {avg_system_time:.4f} time units")
    print(f"  Average queue length: {avg_queue_length:.4f} entities")
    print(f"  Maximum queue length: {stats.max_queue_length} entities")
    print(f"  Server utilization: {server_utilization:.4f} ({server_utilization*100:.2f}%)")
    
    print("\nINDIVIDUAL SERVER UTILIZATIONS")
    print("=" * 50)
    server_utils = []
    for i in range(C):
        util = stats.server_busy_times_stats[i] / stats_collection_time if stats_collection_time > 0 else 0.0
        server_utils.append(util)
        print(f"  Server {i+1}: {util:.4f} ({util*100:.2f}%)")
    
    if len(server_utils) > 1:
        server_mean = mean(server_utils)
        server_std = stdev(server_utils) if len(server_utils) > 1 else 0.0
        server_cv = server_std / server_mean if server_mean > 0 else 0.0
        print(f"  Server balance (CV): {server_cv:.6f} (lower is better)")
        balance_status = "Excellent" if server_cv < 0.01 else "Good" if server_cv < 0.05 else "Fair"
        print(f"  Balance status: {balance_status}")
    
    theoretical = calculate_theoretical_values()
    
    print("\nTHEORETICAL CALCULATIONS")
    print("=" * 50)
    
    if RHO_PER_SERVER >= 1.0:
        print("  System is unstable - theoretical values are infinite")
    else:
        print(f"  Theoretical waiting time: {theoretical['waiting_time']:.4f} time units")
        print(f"  Theoretical service time: {theoretical['service_time']:.4f} time units")
        print(f"  Theoretical time in system: {theoretical['system_time']:.4f} time units")
        print(f"  Theoretical queue length: {theoretical['queue_length']:.4f} entities")
        print(f"  Theoretical utilization: {theoretical['utilization']:.4f} ({theoretical['utilization']*100:.2f}%)")
        print(f"  Erlang-C (P(wait)): {theoretical['erlang_c']:.6f}")
    
    print("\nACCURACY VALIDATION")
    print("=" * 50)
    
    if RHO_PER_SERVER < 1.0:
        waiting_error = abs(avg_waiting_time - theoretical['waiting_time'])
        service_error = abs(avg_service_time - theoretical['service_time'])
        system_error = abs(avg_system_time - theoretical['system_time'])
        queue_error = abs(avg_queue_length - theoretical['queue_length'])
        utilization_error = abs(server_utilization - theoretical['utilization'])
        
        waiting_rel_error = waiting_error / theoretical['waiting_time'] * 100
        service_rel_error = service_error / theoretical['service_time'] * 100
        system_rel_error = system_error / theoretical['system_time'] * 100
        queue_rel_error = queue_error / theoretical['queue_length'] * 100
        utilization_rel_error = utilization_error / theoretical['utilization'] * 100
        
        print(f"  Waiting time error: {waiting_error:.6f} ({waiting_rel_error:.2f}%)")
        print(f"  Service time error: {service_error:.6f} ({service_rel_error:.2f}%)")
        print(f"  System time error: {system_error:.6f} ({system_rel_error:.2f}%)")
        print(f"  Queue length error: {queue_error:.6f} ({queue_rel_error:.2f}%)")
        print(f"  Utilization error: {utilization_error:.6f} ({utilization_rel_error:.2f}%)")
        
        max_rel_error = max(waiting_rel_error, service_rel_error, system_rel_error, 
                           queue_rel_error, utilization_rel_error)
        
        print(f"\n  Maximum relative error: {max_rel_error:.2f}%")
        if max_rel_error < 1.0:
            print("  Accuracy status: Excellent (< 1% error)")
        elif max_rel_error < 5.0:
            print("  Accuracy status: Good (< 5% error)")
        elif max_rel_error < 10.0:
            print("  Accuracy status: Fair (< 10% error)")
        else:
            print("  Accuracy status: Poor (≥ 10% error)")
    else:
        print("  System is unstable - accuracy validation not applicable")
        max_rel_error = 0.0
    
    print("\nLITTLE'S LAW VALIDATION")
    print("=" * 50)
    
    effective_arrival_rate = stats.statistics_entities / stats_collection_time if stats_collection_time > 0 else 0.0
    L = avg_queue_length
    W = avg_waiting_time
    lambda_W = effective_arrival_rate * W
    
    littles_law_error = abs(L - lambda_W)
    littles_law_rel_error = (littles_law_error / L) * 100 if L > 0 else 0.0
    
    print(f"  L (average queue length): {L:.4f}")
    print(f"  lambda (effective arrival rate): {effective_arrival_rate:.4f}")
    print(f"  W (average waiting time): {W:.4f}")
    print(f"  lambda*W (arrival rate * waiting time): {lambda_W:.4f}")
    print(f"  |L - λW| (absolute error): {littles_law_error:.6f}")
    print(f"  Relative error: {littles_law_rel_error:.4f}%")
    
    if littles_law_rel_error < 1.0:
        print("  Little's Law: Validated (< 1% error)")
    elif littles_law_rel_error < 5.0:
        print("  Little's Law: Acceptable (< 5% error)")
    else:
        print("  Little's Law: Questionable (≥ 5% error)")
    
    print("\nSYSTEM PERFORMANCE SUMMARY")
    print("=" * 50)
    print(f"  Implementation: SimPy M/M/C")
    print(f"  Configuration: lambda={LAMBDA}, mu={MU}, c={C}")
    print(f"  Entities processed: {stats.total_entities}")
    print(f"  Runtime: {wall_clock_time * 1000:.2f} ms")
    print(f"  Throughput: {total_throughput:.0f} entities/sec")
    print(f"  Memory efficiency: No special optimization")
    print(f"  Server balance: {server_cv:.6f} CV" if len(server_utils) > 1 else "  Server balance: N/A")
    accuracy_text = f"{max_rel_error:.2f}% max error" if RHO_PER_SERVER < 1.0 else "N/A (unstable)"
    print(f"  Theoretical accuracy: {accuracy_text}")
    print(f"  Little's Law: {littles_law_rel_error:.4f}% error")
    
    print("\nSimPy M/M/C simulation completed successfully!")

if __name__ == "__main__":
    main() 