using Random

if !isdefined(Main, :DESLibrary)
    include("../../src/DESLibrary.jl")
end



function run_csmacd_enhanced()
    NUM_NODES = 10
    LAMBDA = 5.0
    TX_TIME = 0.01
    PROP_DELAY = 0.005
    SLOT_TIME = 0.005
    MAX_FRAMES = 100000
    MAX_ENTITIES = 10000
    SEED = 2024
    WARMUP_FRACTION = 0.1

    println("\nCSMA/CD Simulation using DESLibrary (Enhanced)")
    println("=" ^ 50)
    println("Nodes: $NUM_NODES, λ: $LAMBDA, Max frames: $MAX_FRAMES")
    println("TX time: $TX_TIME, Prop delay: $PROP_DELAY, Slot time: $SLOT_TIME")

    Random.seed!(SEED)

    engine = DESLibrary.DESEngine(MAX_ENTITIES, SEED, WARMUP_FRACTION)
    model = DESLibrary.EnhancedCSMACDModel(
        NUM_NODES;
        model_id = :csmacd_enhanced,
        frame_generation_rate = LAMBDA,
        transmission_time = TX_TIME,
        propagation_delay = PROP_DELAY,
        slot_time = SLOT_TIME,
        max_frames = MAX_FRAMES,
        rng = MersenneTwister(SEED)
    )
    DESLibrary.add_model!(engine, :csmacd_enhanced, model)

    println("\nStarting simulation...")

    DESLibrary.initialize_model!(model, engine)
    engine.entities_processed = 0
    engine.entities_arrived = 0
    engine.stop_simulation = false

    last_monitor_time = 0.0
    start_time = time()

    while !engine.stop_simulation
        event = DESLibrary.get_next_event!(engine.event_queue)
        if event === nothing
            all_idle = all(state == DESLibrary.NODE_IDLE for state in model.node_states)
            total_queued = sum(length, model.packet_queues)
            if all_idle && total_queued == 0
                println("All nodes idle with empty queues - simulation complete")
                break
            else
                break
            end
        end
        if haskey(engine.models, event.model_id)
            DESLibrary.process_event!(engine.models[event.model_id], event, engine)
        end
        current_time = DESLibrary.get_current_time(engine)
        if current_time - last_monitor_time >= 2.0
            last_monitor_time = current_time
            total_queued = sum(length, model.packet_queues)
            if total_queued > 0
                queue_status = [length(q) for q in model.packet_queues]
                # println("Time $(round(current_time, digits=1)): Completed $(model.successful_transmissions), Queued $total_queued, Queues: $queue_status")
            end
            if current_time > 5000.0
                println("Safety timeout reached")
                break
            end
        end
    end

    end_time = time()
    stats = DESLibrary.get_statistics(model, engine)

    println("\nSIMULATION RESULTS")
    println("=" ^ 50)
    println("Wall-clock time: $(round((end_time - start_time) * 1000, digits=2)) ms")
    println("Simulation time: $(round(stats[:simulation_time], digits=2)) time units")
    println("Successful transmissions: $(stats[:successful_transmissions])")
    println("Total collisions: $(stats[:total_collisions])")
    println("Total generated frames: $(stats[:total_generated])")
    println("Total dropped frames: $(stats[:total_drops])")
    println("Frames still queued: $(stats[:frames_queued])")
    println("Frame loss rate: $(round(stats[:total_drops] / stats[:total_generated] * 100, digits=3))%")

    println("\nPERFORMANCE METRICS")
    println("=" ^ 50)
    println("Throughput: $(round(stats[:throughput], digits=4)) frames/time unit")
    println("Collision rate: $(round(stats[:collision_rate], digits=4)) collisions/time unit")
    println("Drop rate: $(round(stats[:drop_rate], digits=4)) drops/time unit")
    println("Average delay: $(round(stats[:average_delay], digits=6)) time units")
    println("Fairness index: $(round(stats[:fairness_index], digits=4)) (1.0 = perfect fairness)")

    println("\nQUEUE DISTRIBUTION")
    println("=" ^ 50)

    println("\nPER-NODE MAXIMUM QUEUE LENGTHS")
    println("-" ^ 40)
    for i in 1:model.num_nodes
        max_q = model.max_queue_lengths[i]
        println("Node $i: $max_q frames")
    end

    println("\nPER-NODE STATISTICS")
    println("=" ^ 80)
    println("Node | Generated | Successes | Drops | Max Queue | Success Rate")
    println("-" ^ 80)
    for i in 1:model.num_nodes
        generated = stats[:node_generated_counts][i]
        successes = stats[:node_success_counts][i]
        drops = stats[:node_drop_counts][i]
        max_queue = model.max_queue_lengths[i]
        success_rate = generated > 0 ? (successes / generated) * 100 : 0.0
        println("$(lpad(i, 4)) | $(lpad(generated, 9)) | $(lpad(successes, 9)) | $(lpad(drops, 5)) | $(lpad(max_queue, 9)) | $(lpad(round(success_rate, digits=3), 11))%")
    end

    total_generated = sum(stats[:node_generated_counts])
    total_successes = sum(stats[:node_success_counts])
    total_drops = sum(stats[:node_drop_counts])
    total_queued = sum(length(queue) for queue in model.packet_queues)

    println("\nFRAME ACCOUNTING SUMMARY")
    println("-" ^ 40)
    println("Total frames generated: $total_generated")
    println("Total frames succeeded: $total_successes")
    println("Total frames dropped:   $total_drops")
    println("Total frames queued:    $total_queued")
    println("Frames processed:       $(total_successes + total_drops)")
    println("Overall success rate:   $(round((total_successes/total_generated)*100, digits=3))%")

    if stats[:frames_queued] > 0
        println("\nREMAINING FRAMES BY NODE")
        for (i, queue) in enumerate(model.packet_queues)
            if !isempty(queue)
                println("Node $i: $(length(queue)) frames")
            end
        end
    end

    println("\nDES Julia CSMA/CD Enhanced simulation completed!")
end
run_csmacd_enhanced()