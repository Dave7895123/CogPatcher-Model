
import time
import copy
# import pybullet as p # Not directly used in EpisodicMemory if sensor interaction is minimal here
from tabulate import tabulate

# from collections import defaultdict # Not used in current snippet
# import numpy as np # Not used in current snippet

FLOAT_EPSILON = 1e-6


class EpisodicMemory:
    def __init__(self, perception, prediction_module, sensor,
                 conflict_resolution_module,
                 reflexive_action_history_recorder=None,
                 idle_threshold=20.0, prediction_start_step=20000):
        self.perception = perception
        self.prediction_module = prediction_module
        self.sensor = sensor
        self.crm = conflict_resolution_module
        self.reflexive_action_history_recorder = reflexive_action_history_recorder
        self.idle_threshold = idle_threshold
        self.last_event_time = time.time()
        self.episodes = []
        self.schema_generator = None
        self.episode_counter = 0

        self.prediction_start_step = prediction_start_step
        self.prediction_enabled_for_current_instance = False

        self.reflexive_action_active = False
        self.current_instance_events_for_storage = None
        self.failed_instances = []

        if self.prediction_module and hasattr(self.prediction_module, 'set_active_status'):
            pass
        print(
            f"EpisodicMemory initialized. Prediction will be managed based on step count (threshold: {prediction_start_step}).")

    def update(self, events):
        non_null_events = [event for event in events if event.get("category") != "null" or event.get(
            "type") != "null_event"]
        if non_null_events:
            self.last_event_time = time.time()

    def check_and_store_episode(self, current_simulation_step_count):
        current_real_time = time.time()

        actual_events_for_current_instance = self.perception.format_events()

        time_is_idle = (current_real_time - self.last_event_time > self.idle_threshold)

        can_attempt_instance_processing = time_is_idle  # 主要基于时间是否空闲

        if not can_attempt_instance_processing:
            return False  # 如果时间还没到，直接返回

        print(f"\nEM_Debug: Idle time threshold reached (Idle: {self.idle_threshold:.1f}s). SimStep: {current_simulation_step_count}.")


        predicted_events_for_current_instance = []
        if self.prediction_module and self.prediction_enabled_for_current_instance:
            if hasattr(self.prediction_module, 'get_current_prediction_sequence'):
                predicted_events_for_current_instance = self.prediction_module.get_current_prediction_sequence()
            else:
                print("EM_Error: PredictionModule is missing 'get_current_prediction_sequence' method!")

        print(f"  EM_Debug: --- PRE-CRM CHECK ---")
        print(f"  EM_Debug: Predicted Sequence (len: {len(predicted_events_for_current_instance)}):")
        for i, ev in enumerate(predicted_events_for_current_instance):
            print(
                f"    Pred[{i}] CI_Seq: {ev.get('current_instance_seq')}, Cat: {ev.get('category')}, Cont: {ev.get('content')}, Skel: {ev.get('is_skeleton_event')}")

        print(f"  EM_Debug: Actual Sequence (len: {len(actual_events_for_current_instance)}):")
        for i, ev in enumerate(actual_events_for_current_instance):
            print(
                f"    Actual[{i}] CI_Seq: {ev.get('current_instance_seq')}, Cat: {ev.get('category')}, Cont: {ev.get('content')}")
        print(f"  EM_Debug: --- END PRE-CRM CHECK ---")


        crm_result = None
        store_this_instance_as_normal = False

        if self.prediction_module and self.prediction_enabled_for_current_instance:
            if self.crm and (predicted_events_for_current_instance or actual_events_for_current_instance):
                print(f"  EM_Debug: Invoking CRM for sequence comparison...")
                crm_result = self.crm.compare_sequences_at_instance_end(
                    predicted_events_for_current_instance,
                    actual_events_for_current_instance,
                    instance_number = self.episode_counter
                )
                print(f"  EM_Debug: CRM comparison result: {crm_result}")
            elif not self.crm:
                print("  EM_Warning: CRM not configured. Skipping comparison for active prediction.")

                if actual_events_for_current_instance:
                    store_this_instance_as_normal = True
            elif not predicted_events_for_current_instance and not actual_events_for_current_instance:
                print("  EM_Debug: Predicted and actual sequences are both empty (while prediction is active). Skipping CRM comparison, considered OK.")
                crm_result = {"status": "ok", "reason": "both_sequences_empty_at_EM_pred_active"}

        else:
            print("  EM_Debug: PredictionModule is not enabled. Storing actual events directly without CRM comparison.")
            if actual_events_for_current_instance:
                store_this_instance_as_normal = True



        if self.reflexive_action_active:
            print("  EM_Debug: Ending a reflexive action episode. Using existing CRM result.")
            instinctive_instance_had_conflict = (crm_result is not None and crm_result.get("status") == "error")

            if self.crm:
                print(f"  EM_Debug: Invoking CRM to process reflexive action outcome (Conflict: {instinctive_instance_had_conflict})...")
                outcome_result = self.crm.process_reflexive_action_outcome(instinctive_instance_had_conflict)
                print(f"  EM_Debug: CRM reflexive action processing result: {outcome_result}")

                if self.reflexive_action_history_recorder is not None:

                    record_data = copy.deepcopy(outcome_result)
                    record_data['simulation_step'] = current_simulation_step_count
                    self.reflexive_action_history_recorder.append(record_data)
                    print("  EM_Debug: Reflexive action outcome recorded to history.")


            else:
                print("  EM_Warning: CRM not configured, cannot process reflexive action outcome.")

            self.reflexive_action_active = False
            print("  EM_Debug: Reflexive action flag has been reset.")
            self.current_instance_events_for_storage = None

            if not instinctive_instance_had_conflict and actual_events_for_current_instance:
                print("  EM_Debug: Reflexive action episode has no conflict. Storing this episode.")
                store_this_instance_as_normal = True
            else:
                print("  EM_Debug: Reflexive action episode has a conflict or is empty. Not storing this episode.")
                store_this_instance_as_normal = False
                if instinctive_instance_had_conflict:
                    self.failed_instances.append({
                        "type": "instinctive_failed",
                        "predicted": copy.deepcopy(predicted_events_for_current_instance),
                        "actual": copy.deepcopy(actual_events_for_current_instance)
                    })


        elif crm_result:

            if crm_result.get("status") == "ok":



                if crm_result.get("used_special_rule_overall"):

                    print("  EM_Debug: Regular episode CRM comparison OK, but not storing due to a special comparison rule being used.")

                    store_this_instance_as_normal = False



                else:

                    if actual_events_for_current_instance:

                        print("  EM_Debug: Regular episode CRM comparison OK (no special rules used). Preparing to store.")

                        store_this_instance_as_normal = True

                    else:  # crm_result is ok, but no actual events, don't store

                        store_this_instance_as_normal = False


            elif crm_result.get("status") == "error":

                print("  EM_Debug: Regular episode CRM comparison found an error. Not storing.")

                store_this_instance_as_normal = False

                self.failed_instances.append({

                    "type": "regular_conflict",

                    "predicted": copy.deepcopy(predicted_events_for_current_instance),

                    "actual": copy.deepcopy(actual_events_for_current_instance),

                    "crm_reason": crm_result.get("reason")

                })


            elif crm_result.get("status") == "reflexive_action_triggered":

                print("  EM_Debug: CRM triggered a reflexive action. Setting reflexive mode for the next episode.")

                self.reflexive_action_active = True

                store_this_instance_as_normal = False


        if store_this_instance_as_normal and actual_events_for_current_instance:
            print(f"  EM_Debug: Storing current episode to EpisodicMemory (contains {len(actual_events_for_current_instance)} events)...")
            instance_data_to_store = {"events": copy.deepcopy(actual_events_for_current_instance)}
            self._process_episode(instance_data_to_store)
        elif store_this_instance_as_normal and not actual_events_for_current_instance:
            print("  EM_Debug: Marked as storable, but actual events are empty. Not storing.")


        print("  EM_Debug: Clearing PerceptualSystem state.")
        self.perception.clear_state()

        if self.prediction_module:
            if hasattr(self.prediction_module, 'clear_state_for_new_instance'):
                print("  EM_Debug: Clearing PredictionModule state.")
                self.prediction_module.clear_state_for_new_instance()


            if hasattr(self.prediction_module, 'set_active_status'):
                next_instance_will_have_prediction_enabled = (
                        current_simulation_step_count >= self.prediction_start_step)

                self.prediction_module.set_active_status(next_instance_will_have_prediction_enabled)
                self.prediction_enabled_for_current_instance = next_instance_will_have_prediction_enabled
                if next_instance_will_have_prediction_enabled:
                    print(
                        f"  EM_Debug: PredictionModule Enabled for the [next] episode. (SimStep: {current_simulation_step_count} >= {self.prediction_start_step}).")
                else:
                    print(
                        f"  EM_Debug: PredictionModule Disabled for the [next] episode. (SimStep: {current_simulation_step_count} < {self.prediction_start_step}).")

        self.last_event_time = current_real_time
        print(f"  EM_Debug: Idle timer reset. EpisodicMemory now has {len(self.episodes)} episodes. "
              f"Number of failed episodes: {len(self.failed_instances)}.")
        return True

        return False
    def _extract_key_events(self, sequenced_events):
        key_event_categories = ["arm_speed", "ball_speed"]
        key_events = []
        for event in sequenced_events:
            category = event.get("category")
            if category in key_event_categories:
                if isinstance(event.get("content"), dict):  # Ensure content is a dict for comparison
                    # Store a reference or a deepcopy if modification is a concern later
                    # For key event extraction, original event references are fine.
                    key_events.append(event)  # Store the event itself, not a copy
        return key_events

    def _compare_key_events(self, key_events1, key_events2):
        if len(key_events1) != len(key_events2):
            return False
        for e1, e2 in zip(key_events1, key_events2):
            # Compare category and content for equality
            if e1["category"] != e2["category"] or \
                    e1["content"] != e2["content"]:  # Assumes content dicts are comparable
                return False
        return True

    def _insert_padding_by_seq(self, events_list, insert_after_seq, num_padding):
        if num_padding <= 0:
            return
        insert_index = 0
        # Find the index where padding should be inserted.
        # Padding is inserted *before* the first event whose sequence number
        # is strictly greater than insert_after_seq.
        for i, event in enumerate(events_list):
            if event.get('seq', float('inf')) > insert_after_seq:
                insert_index = i
                break
        else:  # All events have seq <= insert_after_seq, or list is empty
            insert_index = len(events_list)  # Insert at the end

        padding_events = []
        # Padding events sequence numbers start from insert_after_seq + 1
        current_padding_seq = insert_after_seq + 1
        for _ in range(num_padding):
            padding_events.append({
                "seq": current_padding_seq,
                "category": "padding",
                "content": {"type": "alignment_gap"}
            })
            current_padding_seq += 1

        events_list[insert_index:insert_index] = padding_events

        # Adjust sequence numbers of subsequent events
        for i in range(insert_index + num_padding, len(events_list)):
            if 'seq' in events_list[i] and isinstance(events_list[i]['seq'], (int, float)):
                events_list[i]['seq'] += num_padding

    def _align_episodes_pairwise(self, instance_A_copy, instance_B_copy, key_events_A, key_events_B):
        events_A = instance_A_copy['events']
        events_B = instance_B_copy['events']

        if len(key_events_A) != len(key_events_B):  # Should not happen if _compare_key_events passed
            return events_A, events_B
        num_key_events = len(key_events_A)
        if num_key_events == 0:
            return events_A, events_B

        last_processed_key_seq_A = 0  # Tracks seq of last aligned key event in events_A (after its potential padding)
        last_processed_key_seq_B = 0  # Tracks seq of last aligned key event in events_B (after its potential padding)

        for i in range(num_key_events):
            # Original key events (from key_events_A/B) are used for their category and content.
            # Their sequence numbers are used as a reference but we need to find them
            # in the potentially modified events_A and events_B lists.

            original_key_A = key_events_A[i]
            original_key_B = key_events_B[i]  # Should be identical in category/content to original_key_A

            # Find the current key event in the (potentially modified by padding) lists
            current_key_event_A_found_in_list = None
            for evt_a in events_A:
                # Search from after the last processed key event's sequence
                if evt_a.get('seq', -1) > last_processed_key_seq_A and \
                        evt_a.get('category') == original_key_A['category'] and \
                        evt_a.get('content') == original_key_A['content']:
                    current_key_event_A_found_in_list = evt_a
                    break

            current_key_event_B_found_in_list = None
            for evt_b in events_B:
                if evt_b.get('seq', -1) > last_processed_key_seq_B and \
                        evt_b.get('category') == original_key_B['category'] and \
                        evt_b.get('content') == original_key_B['content']:
                    current_key_event_B_found_in_list = evt_b
                    break

            if not current_key_event_A_found_in_list or not current_key_event_B_found_in_list:
                print(f"WARN (EpisodicMemory._align_episodes_pairwise): Could not re-find key event pair #{i} in lists. "
                      f"Key A: {original_key_A}, Key B: {original_key_B}. Skipping this pair for alignment.")
                # If we can't find them, it's hard to proceed reliably for this pair.
                # We could try to use original_key_X['seq'] but that doesn't account for previous padding.
                # For now, we might just have to update last_processed based on an assumption or skip.
                # Let's assume they would have aligned at max of their original seq if found,
                # then update last_processed and continue to next key event.
                # This is a fallback, ideally this situation is rare.
                seq_A_for_this_key = original_key_A.get('seq', last_processed_key_seq_A)
                seq_B_for_this_key = original_key_B.get('seq', last_processed_key_seq_B)

                # Try to find them using their original sequence numbers if dynamic lookup failed
                if not current_key_event_A_found_in_list:
                    for evt_a_fallback in events_A:
                        if evt_a_fallback.get('seq') == original_key_A.get('seq') and \
                                evt_a_fallback.get('category') == original_key_A['category'] and \
                                evt_a_fallback.get('content') == original_key_A['content']:
                            current_key_event_A_found_in_list = evt_a_fallback  # Use this if found
                            seq_A_for_this_key = current_key_event_A_found_in_list.get('seq')
                            break
                if not current_key_event_B_found_in_list:
                    for evt_b_fallback in events_B:
                        if evt_b_fallback.get('seq') == original_key_B.get('seq') and \
                                evt_b_fallback.get('category') == original_key_B['category'] and \
                                evt_b_fallback.get('content') == original_key_B['content']:
                            current_key_event_B_found_in_list = evt_b_fallback
                            seq_B_for_this_key = current_key_event_B_found_in_list.get('seq')
                            break

                if not current_key_event_A_found_in_list or not current_key_event_B_found_in_list:
                    # Still not found, this is problematic.
                    # Update last_processed to effectively skip this alignment step for this key event
                    # by setting them to what they would be if this key event were aligned.
                    # This could lead to misalignment later if other key events depended on this one's position.
                    aligned_at_seq = max(seq_A_for_this_key, seq_B_for_this_key)
                    last_processed_key_seq_A = aligned_at_seq
                    last_processed_key_seq_B = aligned_at_seq
                    continue  # Move to the next key event pair
            else:
                seq_A_for_this_key = current_key_event_A_found_in_list['seq']
                seq_B_for_this_key = current_key_event_B_found_in_list['seq']

            key_event_category = original_key_A['category']  # Category is same for A and B

            # Determine padding and insertion point
            if seq_A_for_this_key < seq_B_for_this_key:
                diff = seq_B_for_this_key - seq_A_for_this_key
                insert_ref_seq_for_A = last_processed_key_seq_A  # Default: insert before current key A

                if key_event_category == "arm_speed":
                    # If current arm_speed event is NOT immediately after the last processed key event
                    if seq_A_for_this_key > last_processed_key_seq_A + 1:
                        # Insert padding after the event that is at seq last_processed_key_seq_A + 1
                        insert_ref_seq_for_A = last_processed_key_seq_A + 1
                        # print(f"DEBUG: Aligning ARM_SPEED A (seq {seq_A_for_this_key} vs B's {seq_B_for_this_key}). Insert {diff} pads after seq {insert_ref_seq_for_A} in A.")
                    # Else (arm_speed is immediately after prev key), insert_ref_seq_for_A remains last_processed_key_seq_A
                    # This means padding will be inserted just before this arm_speed event.
                    # else:
                    # print(f"DEBUG: Aligning ARM_SPEED A (seq {seq_A_for_this_key} vs B's {seq_B_for_this_key}). Arm_speed follows prev key. Insert {diff} pads after seq {insert_ref_seq_for_A} in A (std).")

                self._insert_padding_by_seq(events_A, insert_ref_seq_for_A, diff)

            elif seq_B_for_this_key < seq_A_for_this_key:
                diff = seq_A_for_this_key - seq_B_for_this_key
                insert_ref_seq_for_B = last_processed_key_seq_B  # Default: insert before current key B

                if key_event_category == "arm_speed":
                    if seq_B_for_this_key > last_processed_key_seq_B + 1:
                        insert_ref_seq_for_B = last_processed_key_seq_B + 1
                        # print(f"DEBUG: Aligning ARM_SPEED B (seq {seq_B_for_this_key} vs A's {seq_A_for_this_key}). Insert {diff} pads after seq {insert_ref_seq_for_B} in B.")
                    # else:
                    # print(f"DEBUG: Aligning ARM_SPEED B (seq {seq_B_for_this_key} vs A's {seq_A_for_this_key}). Arm_speed follows prev key. Insert {diff} pads after seq {insert_ref_seq_for_B} in B (std).")

                self._insert_padding_by_seq(events_B, insert_ref_seq_for_B, diff)

            # After padding, the current key events (A[i] and B[i]) are now effectively aligned.
            # Their new sequence number will be max of their original sequence numbers (before this step's padding).
            aligned_at_seq = max(seq_A_for_this_key, seq_B_for_this_key)
            last_processed_key_seq_A = aligned_at_seq
            last_processed_key_seq_B = aligned_at_seq
            # print(f"DEBUG: Key pair {i} ('{key_event_category}') aligned. Last processed seq for A & B is now {aligned_at_seq}.")

        return events_A, events_B

    def _regenerate_connections(self, sequenced_events):
        connections = []
        connected_index_pairs = set()
        non_padding_events = [e for e in sequenced_events if e.get("category") != "padding"]
        num_non_padding = len(non_padding_events)

        for i in range(num_non_padding):
            for j in range(i + 1, num_non_padding):
                e1, e2 = non_padding_events[i], non_padding_events[j]
                if not isinstance(e1.get('content'), dict) or not isinstance(e2.get('content'), dict):
                    continue

                common_elements = {}
                c1, c2 = e1['content'], e2['content']
                cat1, cat2 = e1.get("category"), e2.get("category")

                motion_categories = {"ball_motion", "arm_motion"}
                speed_categories = {"ball_speed", "arm_speed"}

                if cat1 in motion_categories and cat2 in motion_categories:
                    if c1.get('new_direction') is not None and c1.get('new_direction') == c2.get('new_direction'):
                        common_elements['new_direction'] = c1.get('new_direction')
                    if c1.get('current_pos') is not None and c1.get('current_pos') == c2.get('current_pos'):
                        common_elements['current_pos'] = c1.get('current_pos')
                elif cat1 == cat2 and cat1 in speed_categories:
                    if c1.get('change') is not None and c1.get('change') == c2.get('change'):
                        common_elements['change'] = c1.get('change')

                if common_elements:
                    seq1_val, seq2_val = e1.get('seq'), e2.get('seq')
                    if seq1_val is not None and seq2_val is not None:
                        pair_key = tuple(sorted((seq1_val, seq2_val)))
                        if pair_key not in connected_index_pairs:
                            connections.append(
                                {"event1_seq": seq1_val, "event2_seq": seq2_val,
                                 "event1_cat": cat1, "event2_cat": cat2,
                                 "common_elements": common_elements})
                            connected_index_pairs.add(pair_key)
        return connections

    def _process_episode(self, new_instance_with_sequenced_events):
        new_sequenced_events = new_instance_with_sequenced_events["events"]
        if not new_sequenced_events:
            print("WARN (EpisodicMemory): _process_episode received an empty event list.")
            return

        new_instance_to_store = {
            "events": new_sequenced_events,
            "connections": self._regenerate_connections(new_sequenced_events)
        }

        if self.episodes:
            # Extract key events from the new instance (these are references to original events)
            new_key_events_refs = self._extract_key_events(new_instance_to_store["events"])

            if new_key_events_refs:  # Only proceed if new instance has key events
                for existing_instance_ref in self.episodes:
                    # Extract key events from the existing instance (references)
                    existing_key_events_refs = self._extract_key_events(existing_instance_ref["events"])

                    # Compare based on category and content (not sequence numbers yet)
                    if self._compare_key_events(new_key_events_refs, existing_key_events_refs):
                        # print(f"与现有实例的关键事件匹配，准备对齐和生成进阶实例...")

                        # Deepcopy instances for alignment to avoid modifying stored instances directly
                        temp_new = copy.deepcopy(new_instance_to_store)
                        temp_existing = copy.deepcopy(existing_instance_ref)

                        # For alignment, we need key events that reflect the *current state* of temp_new/temp_existing
                        # if they were already modified. However, _extract_key_events should be run on the copies.
                        # More importantly, the key_events passed to _align_episodes_pairwise should be
                        # from the *original* unaligned structures to guide the alignment based on content.
                        # The _align_episodes_pairwise will then find these key events within temp_new and temp_existing.

                        # Pass the original key event lists (new_key_events_refs, existing_key_events_refs)
                        # which contain the sequence of key events to match.
                        aligned_new_events, aligned_existing_events = self._align_episodes_pairwise(
                            temp_new, temp_existing,
                            new_key_events_refs,  # Pass the list of key event dicts (references from original)
                            existing_key_events_refs
                        )
                        temp_new['events'] = aligned_new_events
                        temp_existing['events'] = aligned_existing_events

                        temp_new["connections"] = self._regenerate_connections(aligned_new_events)
                        temp_existing["connections"] = self._regenerate_connections(aligned_existing_events)

                        if self.schema_generator:
                            try:
                                self.schema_generator.generate_schema(temp_new, temp_existing)
                            except Exception as e:
                                print(f"ERROR (EpisodicMemory): Error while calling SchemaGenerator: {e}")
                        # break # Optional: if new instance should only be aligned with the first match

        self.episodes.append(new_instance_to_store)
        self.episode_counter += 1

    def get_episodes(self):
        return self.episodes

    def print_episode_table(self, instance_index):
        if not (0 <= instance_index < len(self.episodes)):
            print(f"Episode index {instance_index} is invalid (total episodes: {len(self.episodes)}).")
            return
        instance = self.episodes[instance_index]
        events = instance.get("events", [])
        instance_label = f"Episode {instance_index + 1}"
        event_table_data = []
        event_headers = ["Seq", "Category", "Content"]
        if not events:
            print(f"\n{instance_label} has no events.")
        else:
            for event in events:
                event_table_data.append([
                    event.get("seq", "N/A"),
                    event.get("category", "N/A"),
                    self._format_event_content(event)
                ])
            print(f"\n{instance_label} Event List:")
            print(tabulate(event_table_data, headers=event_headers, tablefmt="grid"))

    def _format_event_content(self, event):
        content = event.get("content", {})
        category = event.get("category", "unknown")
        if category == "padding":
            return f"填充 ({content.get('type', 'alignment_gap')})"
        if not isinstance(content, dict): return str(content)
        if category == "command":
            return f"Cmd: {content.get('original_command', 'N/A')} (Dir: {content.get('direction', 'N/A')}, Rot: {content.get('rotation', 'N/A')})"
        elif category in ["ball_speed", "arm_speed"]:
            return f"Speed: {content.get('change', 'N/A')}"
        elif category == "ball_motion":
            tags_str = f", Tags: {content.get('tags', [])}" if content.get('tags') else ""
            return f"Dir: {content.get('new_direction', 'N/A')} (From: {content.get('current_pos', 'N/A')}{tags_str})"
        elif category == "arm_motion":
            return f"Dir: {content.get('new_direction', 'N/A')} (From: {content.get('current_pos', 'N/A')})"
        elif category == "collision":
            return f"Collision: {content.get('collision', 'N/A')}"
        else:
            return str(content)
