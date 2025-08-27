import simpy
import random
import math
import time

NUM_NODES = 10
LAMBDA = 5.0
TX_TIME = 0.01
PROP_DELAY = 0.005
SLOT_TIME = 0.005
MAX_SUCCESSES = 100000
SEED = 2024
WARMUP_FRACTION = 0.1

MAX_RETRY_BASE = 16
MAX_RETRY_SCALE = max(16, NUM_NODES * 2)
MAX_BACKOFF_EXPONENT = 8
FAIRNESS_THRESHOLD = 5
STARVATION_THRESHOLD = 10

PRECOMP_LEN = 10000

class Model:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.node_states = ["idle"] * num_nodes
        self.packet_queues = [[] for _ in range(num_nodes)]
        self.ongoing_transmissions = {}
        self.retry_counters = [0] * num_nodes
        self.node_success_counts = [0] * num_nodes
        self.node_drop_counts = [0] * num_nodes
        self.node_generated_counts = [0] * num_nodes
        self.successful_transmissions = 0
        self.total_collisions = 0
        self.transmission_delays = []
        self.max_queue_lengths = [0] * num_nodes
        self.total_generated = 0

def main():
    random.seed(SEED)
    env = simpy.Environment()
    model = Model(NUM_NODES)
    exp_samples = [random.expovariate(LAMBDA) for _ in range(PRECOMP_LEN)]
    backoff_samples = [random.randint(0, 255) for _ in range(PRECOMP_LEN)]
    exp_idx = [0]
    backoff_idx = [0]
    stop_event = env.event()

    def deterministic_exponential():
        i = exp_idx[0]
        v = exp_samples[i]
        exp_idx[0] = (i + 1) % PRECOMP_LEN
        return v

    def deterministic_backoff(max_slots):
        i = backoff_idx[0]
        v = backoff_samples[i] % max(1, max_slots)
        backoff_idx[0] = (i + 1) % PRECOMP_LEN
        return v

    def calculate_fairness_factor(node):
        queue_len = len(model.packet_queues[node])
        if queue_len >= STARVATION_THRESHOLD:
            return 0.1
        avg_queue_len = sum(len(q) for q in model.packet_queues) / model.num_nodes
        if avg_queue_len > 0:
            queue_ratio = queue_len / avg_queue_len
            return max(0.3, min(2.0, 2.0 - queue_ratio))
        return 1.0

    def calculate_enhanced_backoff(node):
        retry_count = model.retry_counters[node]
        k = min(retry_count, MAX_BACKOFF_EXPONENT)
        max_slots = 2 ** k
        backoff_slots = deterministic_backoff(max_slots)
        base_backoff = backoff_slots * SLOT_TIME
        fairness_factor = calculate_fairness_factor(node)
        node_jitter = node * SLOT_TIME * 0.01
        random_jitter = random.random() * SLOT_TIME * 0.1
        total_backoff = base_backoff * fairness_factor + node_jitter + random_jitter
        return max(SLOT_TIME * 0.1, total_backoff)

    def schedule_transmission_attempt(node, now):
        if model.node_states[node] == "idle" and model.packet_queues[node]:
            queue_len = len(model.packet_queues[node])
            if queue_len >= STARVATION_THRESHOLD:
                sense_delay = SLOT_TIME / 1000
            elif queue_len >= FAIRNESS_THRESHOLD:
                sense_delay = SLOT_TIME / 200
            else:
                sense_delay = SLOT_TIME / 100
            node_jitter = node * SLOT_TIME * 0.001
            random_jitter = random.random() * SLOT_TIME * 0.01
            total_delay = sense_delay + node_jitter + random_jitter
            env.process(_delay_and_call(total_delay, start_transmission, node))

    def handle_collision(collided_nodes, now):
        model.total_collisions += 1
        for node in collided_nodes:
            if node in model.ongoing_transmissions:
                del model.ongoing_transmissions[node]
            model.node_states[node] = "backoff"
            model.retry_counters[node] += 1
            queue_len = len(model.packet_queues[node])
            adaptive_retry_limit = MAX_RETRY_SCALE + 10 if queue_len >= STARVATION_THRESHOLD else MAX_RETRY_SCALE
            if model.retry_counters[node] > adaptive_retry_limit:
                if model.packet_queues[node]:
                    model.packet_queues[node].pop(0)
                    model.node_drop_counts[node] += 1
                    print(f" Node {node+1} drops frame after {adaptive_retry_limit} retries (queue: {queue_len})")
                model.retry_counters[node] = 0
                model.node_states[node] = "idle"
                if model.packet_queues[node]:
                    schedule_transmission_attempt(node, now)
            else:
                backoff_time = calculate_enhanced_backoff(node)
                env.process(_delay_and_call(backoff_time, backoff_expired, node))

    def start_transmission(node):
        now = env.now
        if model.node_states[node] != "idle" or not model.packet_queues[node]:
            return
        collided_nodes = []
        for other_node, start_time in list(model.ongoing_transmissions.items()):
            if other_node != node and abs(now - start_time) < 2 * PROP_DELAY:
                collided_nodes.append(other_node)
        if collided_nodes:
            collided_nodes.append(node)
            handle_collision(collided_nodes, now)
        else:
            model.node_states[node] = "transmitting"
            model.ongoing_transmissions[node] = now
            env.process(_delay_and_call(TX_TIME, complete_transmission, node))

    def complete_transmission(node):
        now = env.now
        if model.node_states[node] != "transmitting" or node not in model.ongoing_transmissions:
            return
        if model.packet_queues[node]:
            gen_time = model.packet_queues[node].pop(0)
            delay = now - gen_time
            model.transmission_delays.append(delay)
            model.successful_transmissions += 1
            model.node_success_counts[node] += 1
            model.retry_counters[node] = 0
            if model.successful_transmissions >= MAX_SUCCESSES:
                if not stop_event.triggered:
                    print(f" Target reached! {model.successful_transmissions} successful transmissions")
                    stop_event.succeed()
                del model.ongoing_transmissions[node]
                model.node_states[node] = "idle"
                return
        if node in model.ongoing_transmissions:
            del model.ongoing_transmissions[node]
        model.node_states[node] = "idle"
        if model.packet_queues[node]:
            schedule_transmission_attempt(node, now)

    def backoff_expired(node):
        now = env.now
        model.node_states[node] = "idle"
        if model.packet_queues[node]:
            schedule_transmission_attempt(node, now)

    def _delay_and_call(delay, fn, *args):
        yield env.timeout(max(0.0, delay))
        fn(*args)

    def frame_generator(node):
        interarrival = deterministic_exponential()
        yield env.timeout(interarrival)
        while model.total_generated < MAX_SUCCESSES:
            now = env.now
            model.packet_queues[node].append(now)
            model.total_generated += 1
            model.node_generated_counts[node] += 1
            qlen = len(model.packet_queues[node])
            if qlen > model.max_queue_lengths[node]:
                model.max_queue_lengths[node] = qlen
            if model.node_states[node] == "idle":
                schedule_transmission_attempt(node, now)
            if model.total_generated >= MAX_SUCCESSES:
                break
            interarrival = deterministic_exponential()
            yield env.timeout(interarrival)

    def monitor():
        while True:
            yield env.timeout(2.0)
            now = env.now
            all_idle = all(state == "idle" for state in model.node_states)
            total_queued = sum(len(q) for q in model.packet_queues)
            if total_queued > 0:
                queue_status = [len(q) for q in model.packet_queues]
                # print(f"Time {round(now, 1)}: Completed {model.successful_transmissions}, Queued {total_queued}, Queues: {queue_status}")
            if all_idle and total_queued == 0 and model.total_generated >= MAX_SUCCESSES:
                print("All nodes idle with empty queues - simulation complete")
                if not stop_event.triggered:
                    stop_event.succeed()
                return
            if now > 5000.0:
                print("Safety timeout reached")
                if not stop_event.triggered:
                    stop_event.succeed()
                return

    print("\nCSMA/CD Simulation using SimPy")
    print("=" * 50)
    print(f"Nodes: {NUM_NODES}, lambda: {LAMBDA}, Target: {MAX_SUCCESSES} frames")
    print(f"TX time: {TX_TIME}, Prop delay: {PROP_DELAY}, Slot time: {SLOT_TIME}")
    start_wall = time.time()
    for n in range(model.num_nodes):
        env.process(frame_generator(n))
    print(f"Initialized {model.num_nodes} nodes with frame generation")
    print("\nStarting simulation...")
    env.process(monitor())
    env.run(until=stop_event)
    end_wall = time.time()

    sim_time = env.now
    throughput = model.successful_transmissions / sim_time if sim_time > 0 else 0.0
    collision_rate = model.total_collisions / sim_time if sim_time > 0 else 0.0
    avg_delay = (sum(model.transmission_delays) / len(model.transmission_delays)) if model.transmission_delays else 0.0
    total_drops = sum(model.node_drop_counts)
    drop_rate = total_drops / sim_time if sim_time > 0 else 0.0
    if model.successful_transmissions > 0:
        sum_successes = sum(model.node_success_counts)
        sum_squares = sum(x * x for x in model.node_success_counts)
        fairness_index = (sum_successes * sum_successes) / (model.num_nodes * sum_squares) if sum_squares > 0 else 1.0
    else:
        fairness_index = 1.0
    queue_lengths = [len(q) for q in model.packet_queues]
    max_queue = max(queue_lengths) if queue_lengths else 0
    min_queue = min(queue_lengths) if queue_lengths else 0
    avg_queue = (sum(queue_lengths) / len(queue_lengths)) if queue_lengths else 0.0

    print("\nSIMULATION RESULTS")
    print("=" * 50)
    print(f"Wall-clock time: {round((end_wall - start_wall) * 1000, 2)} ms")
    print(f"Simulation time: {round(sim_time, 2)} time units")
    print(f"Successful transmissions: {model.successful_transmissions}")
    print(f"Total collisions: {model.total_collisions}")
    print(f"Total generated frames: {model.total_generated}")
    print(f"Total dropped frames: {total_drops}")
    print(f"Frames still queued: {sum(queue_lengths)}")
    flr = (total_drops / model.total_generated * 100) if model.total_generated > 0 else 0.0
    print(f"Frame loss rate: {round(flr, 3)}%")

    print("\nPERFORMANCE METRICS")
    print("=" * 50)
    print(f"Throughput: {round(throughput, 4)} frames/time unit")
    print(f"Collision rate: {round(collision_rate, 4)} collisions/time unit")
    print(f"Drop rate: {round(drop_rate, 4)} drops/time unit")
    print(f"Average delay: {round(avg_delay, 6)} time units")
    print(f"Fairness index: {round(fairness_index, 4)} (1.0 = perfect fairness)")

    print("\nQUEUE DISTRIBUTION")
    print("=" * 50)

    print("\nPER-NODE MAXIMUM QUEUE LENGTHS")
    print("-" * 40)
    for i in range(model.num_nodes):
        print(f"Node {i+1}: {model.max_queue_lengths[i]} frames")

    print("\nPER-NODE STATISTICS")
    print("=" * 80)
    print("Node | Generated | Successes | Drops | Max Queue | Success Rate")
    print("-" * 80)
    for i in range(model.num_nodes):
        generated = model.node_generated_counts[i]
        successes = model.node_success_counts[i]
        drops = model.node_drop_counts[i]
        maxq = model.max_queue_lengths[i]
        success_rate = (successes / generated) * 100 if generated > 0 else 0.0
        print(f"{str(i+1).rjust(4)} | {str(generated).rjust(9)} | {str(successes).rjust(9)} | {str(drops).rjust(5)} | {str(maxq).rjust(9)} | {str(round(success_rate, 3)).rjust(11)}%")

    total_generated = sum(model.node_generated_counts)
    total_successes = sum(model.node_success_counts)
    total_drops2 = sum(model.node_drop_counts)
    total_queued = sum(len(q) for q in model.packet_queues)

    print("\nFRAME ACCOUNTING SUMMARY")
    print("-" * 40)
    print(f"Total frames generated: {total_generated}")
    print(f"Total frames succeeded: {total_successes}")
    print(f"Total frames dropped:   {total_drops2}")
    print(f"Total frames queued:    {total_queued}")
    print(f"Frames processed:       {total_successes + total_drops2}")
    overall_success_rate = (total_successes / total_generated * 100) if total_generated > 0 else 0.0
    print(f"Overall success rate:   {round(overall_success_rate, 3)}%")
    if sum(queue_lengths) > 0:
        print("\nREMAINING FRAMES BY NODE")
        for i, q in enumerate(model.packet_queues, start=1):
            if q:
                print(f"Node {i}: {len(q)} frames")
    print("\nSimPy CSMA/CD simulation completed!")

if __name__ == "__main__":
    main()