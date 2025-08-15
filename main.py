# --- START OF FILE main.py (REVISED LOOP LOGIC) ---

import pybullet as p
import time
import copy
import math


from physics_world import PhysicsWorld
from sensor import Sensor
from event_segmenter import EventSegmenter
from perception import Perception
from episodic_memory import EpisodicMemory
from schema_generator import SchemaGenerator
from schematic_memory import SchematicMemory
from prediction_module import PredictionModule  # Using the integrated PredictionModule with recent changes
from simulator import Simulator
from action_system import ActionSystem
from conflict_resolution_module import ConflictResolutionModule

# --- Constants ---
SIMULATION_TIME_STEP = 1 / 60.0
ROTATION_SPEED_DEG_PER_SEC = 45

ROTATION_SPEED_RAD_PER_SEC = math.radians(ROTATION_SPEED_DEG_PER_SEC)
TOTAL_SIMULATION_STEPS = 100000
PREDICTION_START_STEP_THRESHOLD = 20000
episodic_memory_IDLE_THRESHOLD_SEC = 10.0


class ConflictResolutionModulePlaceholder:  # (Placeholder as before)
    def handle_prediction_conflict(self, predicted_event_at_diff, actual_event_at_diff, 
                                   violated_rule_ids, reason_for_conflict):
        print("\n--- [Placeholder] ConflictResolutionModule ---")
        print(f"  Conflict Detected: {reason_for_conflict}")
        print(f"  Violated Rule(s)/Cause: {violated_rule_ids}")
        print(f"  Predicted Event: {predicted_event_at_diff}")
        print(f"  Actual Event: {actual_event_at_diff}")
        print("--- End [Placeholder] Conflict ---")




if __name__ == "__main__":

    physics_world = PhysicsWorld(gui=True)
    sensor = Sensor(physics_world)
    event_segmenter = EventSegmenter()
    perception = Perception(sensor)
    schematic_memory = SchematicMemory()
    prediction_module = PredictionModule(schematic_memory, conflict_resolution_module=None)


    simulator = Simulator(physics_world)


    action_system = ActionSystem(perception, simulator, physics_world)


    crm = ConflictResolutionModule(
        prediction_module=prediction_module,
        perception_module=perception,
        action_system=action_system,
        schematic_memory_module=schematic_memory
    )
    schematic_memory.conflict_resolver = crm




    episodic_memory = EpisodicMemory(
        perception=perception,
        prediction_module=prediction_module,
        sensor=sensor,
        conflict_resolution_module=crm,
        idle_threshold=episodic_memory_IDLE_THRESHOLD_SEC,
        prediction_start_step=PREDICTION_START_STEP_THRESHOLD
    )

    schema_generator = SchemaGenerator(episodic_memory, schematic_memory)
    episodic_memory.schema_generator = schema_generator



    sensor.get_data()  # Initial sensor read
    print("System started. Running automatically...")
    global_simulation_step_count = 0



    try:
        for step_count in range(TOTAL_SIMULATION_STEPS):
            global_simulation_step_count = step_count

            physics_world.step()

            sensor_data = sensor.get_data()
            sensor_events = event_segmenter.process(sensor_data)

            instance_stored_this_step = episodic_memory.check_and_store_episode(global_simulation_step_count)


            command_events_this_step = []

            current_sim_time_for_commands = perception.current_sim_time

            if instance_stored_this_step:
                if not action_system.is_instance_active():

                    print(
                        f"Main: AS is inactive, attempting to start a new episode (could be regular or reflexive). SimStep: {global_simulation_step_count}")
                    cmd_evts = action_system.start_new_instance(current_sim_time_for_commands)
                    command_events_this_step.extend(cmd_evts)
                    if cmd_evts and action_system.is_instance_active():
                        initial_velocity = ROTATION_SPEED_RAD_PER_SEC * action_system.current_rotation_direction
                        physics_world.set_arm_velocity(initial_velocity)


            elif action_system.is_instance_active():



                cmd_evts = action_system.check_and_generate_type2_command(current_sim_time_for_commands)

                command_events_this_step.extend(cmd_evts)


            all_step_events = sensor_events + command_events_this_step
            perception.process(all_step_events, time_step=SIMULATION_TIME_STEP)


            current_perception_seq_num = perception.current_seq_num
            newly_sequenced_events_this_step = []
            if perception.events:
                for p_event in reversed(perception.events):
                    if p_event.get('seq') == current_perception_seq_num:
                        newly_sequenced_events_this_step.append(p_event)
                    elif p_event.get('seq') < current_perception_seq_num:
                        break
                newly_sequenced_events_this_step.reverse()


            if prediction_module.is_active and not prediction_module.initial_events_processed:
                if perception.events:
                    min_seq_in_perception = min(
                        ev.get('seq') for ev in perception.events if isinstance(ev.get('seq'), int))

                    if prediction_module.last_processed_event_internal_idx == -1:
                        initial_events_for_pm = [ev for ev in perception.events if
                                                 ev.get('seq') == min_seq_in_perception]
                        if initial_events_for_pm:
                            print(
                                f"Main: Initial events for the episode detected (Perception Seq: {min_seq_in_perception}), passing to PredictionModule for processing...")
                            prediction_module.process_initial_events(initial_events_for_pm)



            for p_event in newly_sequenced_events_this_step:
                if p_event.get("category") == "command":
                    if prediction_module.is_active:
                        print(f"Main: New command perceived (Seq: {p_event.get('seq')}), PredictionModule is active, processing...")
                        cmd_event_for_pm = copy.deepcopy(p_event)
                        if 'current_instance_seq' not in cmd_event_for_pm:
                            cmd_event_for_pm['current_instance_seq'] = cmd_event_for_pm.get('seq')

                        prediction_module.process_new_command_event_from_perception(cmd_event_for_pm)



                if prediction_module.is_active and \
                        prediction_module.initial_events_processed and \
                        not prediction_module.paused_due_to_conflict and \
                        not prediction_module.paused_waiting_for_command:

                    can_continue_predicting = True
                    max_predict_in_step = 5
                    predict_count_this_step = 0
                    while can_continue_predicting and predict_count_this_step < max_predict_in_step:
                        if not prediction_module.is_active or \
                                not prediction_module.initial_events_processed or \
                                prediction_module.paused_due_to_conflict or \
                                prediction_module.paused_waiting_for_command:
                            can_continue_predicting = False
                            break

                        processed_something = prediction_module.check_and_predict_next_event()
                        if not processed_something:
                            can_continue_predicting = False


                        if prediction_module.paused_waiting_for_command:
                            print("Main: PM is paused waiting for a command, stopping this drive cycle.")
                            can_continue_predicting = False

                        predict_count_this_step += 1
                    if predict_count_this_step >= max_predict_in_step:
                        print(f"Main_Warning: Prediction count per step reached the limit of {max_predict_in_step}.")


            collision_detected_this_step = False
            for event_data in all_step_events:
                if (event_data.get('type') == 'collision_change' and
                        event_data.get('collision') is True):
                    collision_detected_this_step = True
                    break
            if collision_detected_this_step:
                action_system.update_collision_status(True)


            episodic_memory.update(sensor_events)


            time.sleep(SIMULATION_TIME_STEP)


            if (step_count + 1) % 3000 == 0:
                print(f"--- Step {step_count + 1} / {TOTAL_SIMULATION_STEPS} ---")
                if prediction_module.is_active and prediction_module.initial_events_processed:
                    prediction_module.print_current_prediction_sequence(num_events_to_show=10)


        print("\nSimulation loop finished.")
        print("\n--- Final Episodic Memory ---")
        num_basic = len(episodic_memory.get_episodes())
        if num_basic == 0:
            print("No Episodic Memory were stored.")
        else:
            for i in range(num_basic): episodic_memory.print_episode_table(i)

        print("\n--- Final Schematic Memory ---")
        num_advanced = len(schematic_memory.get_schemas())
        if num_advanced == 0:
            print("No Schematic Memory were generated.")
        else:
            for i in range(num_advanced): schematic_memory.print_schema_table(i)

        if prediction_module.is_active and prediction_module.initial_events_processed:
            print("\n--- Final Prediction Module State (if active and processed) ---")
            prediction_module.print_current_prediction_sequence()
        elif prediction_module.is_active:
            print("\n--- Prediction Module was active but did not process initial events for last instance ---")
        else:
            print("\n--- Prediction Module was not active at the end ---")




    except KeyboardInterrupt:
        print("User interrupted, exiting.")
    except Exception as e:
        print(f"FATAL: Uncaught exception in the main loop: {e}")
        import traceback

        traceback.print_exc()
    finally:
        if hasattr(physics_world, 'disconnect') and callable(physics_world.disconnect):
            physics_world.disconnect()
        elif 'p' in locals() and hasattr(p, 'isConnected') and p.isConnected():  # Fallback for raw pybullet
            p.disconnect()
        print("\nProgram finished.")

