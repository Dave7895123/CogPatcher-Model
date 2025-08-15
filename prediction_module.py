import copy


UNKNOWN_PLACEHOLDER = "ANY_PLACEHOLDER"
ABSTRACT_ID_PREFIX = "#"
ABSTRACT_ID_SUFFIX = "#"


class PredictionModule:
    def __init__(self, advanced_memory_module, conflict_resolution_module=None):
        self.advanced_memory = advanced_memory_module

        self.advanced_memory = advanced_memory_module
        self.paused_waiting_for_command = False


        self.last_checked_ball_motion_event = None
        self.last_checked_arm_motion_event = None
        self.last_perception_command_event = None
        self.abstract_identifier_library = {}


        self.current_prediction_sequence = []
        self.last_processed_event_internal_idx = -1
        self.initial_events_processed = False  #
        self.current_skeleton_source_instance_id = None

        self.paused_due_to_conflict = False
        self.event_causing_pause = None


        self.processed_perception_command_signatures = set()


        self.is_active = False
        self.prompted_inactive_this_instance = False
        print("PredictionModule initialized (Prediction initially DORMANT).")


    def set_active_status(self, active_status):
        if self.is_active != active_status:
            print(f"PM: Prediction status changed to: {'ACTIVE' if active_status else 'INACTIVE'}")
        self.is_active = active_status
        self.prompted_inactive_this_instance = False

    def handle_command_when_inactive(self, command_event):
        if not self.is_active and not self.prompted_inactive_this_instance:
            print(
                f"PM (First prompt this episode): PredictionModule is inactive, skipping command processing (Perception Seq: {command_event.get('seq')}).")
            self.prompted_inactive_this_instance = True

        self._update_key_information_from_event(command_event, is_from_perception=True)


    def _is_abstract_identifier(self, value):
        if not isinstance(value, str):
            return False
        if len(value) == 2:
            first_char = value[0]
            second_char = value[1]
            if (first_char == 'D' or first_char == 'P') and \
                    (second_char.isdigit() and '1' <= second_char <= '9'):
                return True
        return False

    def _get_command_signature(self, command_event):

        if not command_event or command_event.get("category") != "command":
            return None
        perception_seq = command_event.get('seq')
        content = command_event.get("content", {})
        direction = content.get("direction")
        rotation = content.get("rotation")
        return (perception_seq, direction, rotation)


    def _update_key_information_from_event(self, event, is_from_perception=True):
        if not event or not isinstance(event, dict): return
        category = event.get("category")
        if category == "command":
            if is_from_perception:
                event_to_store = copy.deepcopy(event)

                if 'current_instance_seq' not in event_to_store and 'seq' in event_to_store:
                    event_to_store['current_instance_seq'] = event_to_store['seq']
                self.last_perception_command_event = event_to_store

    def _update_last_checked_motion_event(self, processed_event):
        if not processed_event or not isinstance(processed_event, dict): return
        category = processed_event.get("category")
        if category == "ball_motion":
            self.last_checked_ball_motion_event = copy.deepcopy(processed_event)
        elif category == "arm_motion":
            self.last_checked_arm_motion_event = copy.deepcopy(processed_event)



    def process_initial_events(self, initial_actual_events_from_perception):

        self.initial_events_processed = True
        print("PM: Setting initial_events_processed = True (before calling SM to select a skeleton).")


        initial_events_for_pm_internal_use = []


        if self.current_prediction_sequence and self.last_processed_event_internal_idx > -1:
            print("PM_WARN: 'process_initial_events' called while prediction sequence is not in initial state. Clearing and restarting.")
            self.current_prediction_sequence = []
            self.last_processed_event_internal_idx = -1

        if not initial_actual_events_from_perception:
            print("PM_ERROR: The list of initial events is empty. Cannot proceed.")
            return

        num_initial_events = len(initial_actual_events_from_perception)

        for idx, p_event in enumerate(initial_actual_events_from_perception):
            event_copy = copy.deepcopy(p_event)

            event_ci_seq = p_event.get('current_instance_seq')
            if event_ci_seq is None:
                event_ci_seq = p_event.get('seq', self.last_processed_event_internal_idx + idx + 2)

            event_copy['current_instance_seq'] = event_ci_seq
            event_copy['is_skeleton_event'] = False
            initial_events_for_pm_internal_use.append(event_copy)


            self._update_key_information_from_event(event_copy, is_from_perception=True)
            self._update_last_checked_motion_event(event_copy)

            if event_copy.get("category") == "command":
                cmd_sig = self._get_command_signature(event_copy)
                if cmd_sig:
                    self.processed_perception_command_signatures.add(cmd_sig)
                    print(f"  PM: Initial command event (Sig: {cmd_sig}) marked as processed (from perception).")


        last_cmd_in_initial = None
        for ev in reversed(initial_events_for_pm_internal_use):
            if ev.get("category") == "command":
                last_cmd_in_initial = ev
                break
        if last_cmd_in_initial:
            last_initial_cmd_ci_seq = last_cmd_in_initial.get('current_instance_seq', 0)
        print(f"PM: Recorded {len(initial_events_for_pm_internal_use)} initial actual events for SM to select a skeleton.")


        if not self.is_active:
            print("PM: Prediction is disabled, skipping skeleton selection and loading.")

            self.current_prediction_sequence.extend(initial_events_for_pm_internal_use)
            self.last_processed_event_internal_idx = len(self.current_prediction_sequence) - 1
            print("PM (Prediction Inactive): Initial actual events have been added directly to the prediction sequence.")
            return

        print(f"PM (Prediction Active): Calling SchematicMemory to select a skeleton and construct the full sequence...")


        complete_skeleton_sequence, updated_library, skeleton_source_id = \
            self.advanced_memory.select_and_construct_skeleton_with_initial_events(
                initial_events_for_pm_internal_use,
                self.abstract_identifier_library
            )


        self.abstract_identifier_library = updated_library
        self.current_skeleton_source_instance_id = skeleton_source_id

        if complete_skeleton_sequence:
            print(
                f"PM: Received full skeleton sequence from SchematicMemory (Source: {skeleton_source_id}), with a total of {len(complete_skeleton_sequence)} events.")
            self.current_prediction_sequence = complete_skeleton_sequence


            if num_initial_events > 0 and len(self.current_prediction_sequence) >= num_initial_events:
                self.last_processed_event_internal_idx = num_initial_events - 1
                print(
                    f"PM: 'last_processed_event_internal_idx' set to {self.last_processed_event_internal_idx} (points to the last initial event in the full skeleton).")
            else:

                self.last_processed_event_internal_idx = -1
                if not complete_skeleton_sequence:
                    print("PM: SM failed to provide a skeleton, but prediction is active. Will use only the initial events (if not already added).")

                    if not self.current_prediction_sequence:
                        self.current_prediction_sequence.extend(initial_events_for_pm_internal_use)
                        self.last_processed_event_internal_idx = len(self.current_prediction_sequence) - 1

            print(f"PM: Full skeleton has been loaded into the prediction sequence. Total length: {len(self.current_prediction_sequence)}.")
        else:
            print(
                "PM: SchematicMemory failed to provide a full skeleton sequence. The prediction sequence will only contain the initial events.")

            if not self.current_prediction_sequence:
                self.current_prediction_sequence.extend(initial_events_for_pm_internal_use)
                self.last_processed_event_internal_idx = len(self.current_prediction_sequence) - 1
                print("PM (SM did not return a skeleton): Initial actual events have been added to the prediction sequence.")

        print(f"PM: Updated Abstract Identifier Library: {self.abstract_identifier_library}")
        self.paused_due_to_conflict = False


    def _is_event_content_uncertain(self, event_content):
        if not isinstance(event_content, dict): return False
        print(f"    PM_UncertainCheck: Checking content: {event_content}")
        for field, value in event_content.items():
            print(f"      PM_UncertainCheck: Field '{field}', Value '{value}'")
            if value == UNKNOWN_PLACEHOLDER or value == "UNKNOWN_VALUE":
                print(f"        PM_UncertainCheck: Found UNKNOWN/UNKNOWN_VALUE for field '{field}'. Returning True.")
                return True
            is_abs_id = self._is_abstract_identifier(value)
            print(f"        PM_UncertainCheck: Is abstract ID for field '{field}'? {is_abs_id}")
            if is_abs_id:
                print(f"        PM_UncertainCheck: Found abstract ID for field '{field}'. Returning True.")
                return True
        print(f"    PM_UncertainCheck: No uncertain elements found in content. Returning False.")
        return False

    def _get_next_event_to_process_in_sequence(self):
        if not self.current_prediction_sequence: return None, -1
        next_internal_idx = self.last_processed_event_internal_idx + 1
        for i in range(next_internal_idx, len(self.current_prediction_sequence)):
            event = self.current_prediction_sequence[i]
            if event.get('is_skeleton_event', False):
                return event, i
        return None, -1

    def check_and_predict_next_event(self):
        if not self.is_active: return False
        if not self.initial_events_processed: return False
        if self.paused_due_to_conflict:
            print("PM: Paused due to conflict, waiting for a new command event.")
            return False
        if self.paused_waiting_for_command:
            print("PM: Encountered a command in the skeleton. Pausing prediction and waiting for a new perceived command.")
            return False

        event_to_check_template, current_internal_idx = self._get_next_event_to_process_in_sequence()
        if event_to_check_template is None:
            return False

        print(
            f"\nPM: Checking prediction sequence event (Index: {current_internal_idx}, CI_Seq: {event_to_check_template.get('current_instance_seq')})")
        print(
            f"  PM_State: PausedConflict: {self.paused_due_to_conflict}, PausedCommand: {self.paused_waiting_for_command}")
        print(f"  PM_State: Event content: {event_to_check_template.get('content')}")
        print(f"  PM_State: Is Skeleton Event: {event_to_check_template.get('is_skeleton_event')}")

        if event_to_check_template.get("category") == "command":
            print(f"PM: Encountered a skeleton command event (CI_Seq: {event_to_check_template.get('current_instance_seq')}). Pausing prediction.")
            self.last_processed_event_internal_idx = current_internal_idx
            self.paused_waiting_for_command = True
            return True

        is_uncertain = self._is_event_content_uncertain(event_to_check_template.get("content", {}))
        if not is_uncertain:
            print(f"  PM: Event content is concrete, no prediction needed.")

            self._update_last_checked_motion_event(self.current_prediction_sequence[current_internal_idx])
            self.last_processed_event_internal_idx = current_internal_idx
            return True
        else:
            print(f"  PM: Event content is uncertain, calling SchematicMemory...")

            template_for_am_prediction = copy.deepcopy(event_to_check_template)

            predicted_event_details, updated_library, special_command = \
                self.advanced_memory.predict_event_details_iteratively(
                    self.last_checked_ball_motion_event, self.last_checked_arm_motion_event,
                    self.last_perception_command_event, template_for_am_prediction,
                    self.abstract_identifier_library
                )
            self.abstract_identifier_library = updated_library


            if special_command == "SHIFT_PREDICTION_SEQ":
                print(f"  PM: Received SHIFT_PREDICTION_SEQ. Inserting rule-based prediction at the original conflict site and shifting subsequent events.")


                original_event_in_sequence = self.current_prediction_sequence[current_internal_idx]
                original_ci_seq = original_event_in_sequence.get('current_instance_seq')

                print(f"    PM: The original conflicting event (Index: {current_internal_idx}, Original CI_Seq: {original_ci_seq}) will be shifted.")


                for i in range(current_internal_idx, len(self.current_prediction_sequence)):
                    event_to_shift = self.current_prediction_sequence[i]
                    if 'current_instance_seq' in event_to_shift and \
                            isinstance(event_to_shift['current_instance_seq'], (int, float)):
                        event_to_shift['current_instance_seq'] += 1
                        print(
                            f"      PM_Shift: Event (OrigCat:{event_to_shift.get('category')}, OrigIndex:{i}) CI_Seq updated to: {event_to_shift.get('current_instance_seq')}")


                new_event_from_rules = copy.deepcopy(
                    predicted_event_details)


                new_event_from_rules['is_skeleton_event'] = event_to_check_template.get('is_skeleton_event', True)
                new_event_from_rules['seq_in_skeleton'] = event_to_check_template.get('seq_in_skeleton')
                new_event_from_rules['category'] = event_to_check_template.get('category')
                new_event_from_rules['current_instance_seq'] = original_ci_seq


                self.current_prediction_sequence.insert(current_internal_idx, new_event_from_rules)
                print(
                    f"    PM: New rule-based event inserted at index {current_internal_idx}, CI_Seq: {new_event_from_rules.get('current_instance_seq')}..")


                self.paused_due_to_conflict = True

                self.event_causing_pause = copy.deepcopy(new_event_from_rules)
                print(
                    f"  PM: Paused due to SHIFT_PREDICTION_SEQ conflict. The pausing event is the newly inserted rule-based prediction (CI_Seq: {new_event_from_rules.get('current_instance_seq')})")


                self._update_last_checked_motion_event(new_event_from_rules)

                self.last_processed_event_internal_idx = current_internal_idx

                return True

            else:
                target_event_in_sequence = self.current_prediction_sequence[current_internal_idx]
                target_event_in_sequence["content"] = predicted_event_details.get("content", {})
                target_event_in_sequence["prediction_rules_applied"] = \
                    predicted_event_details.get("prediction_rules_applied", [])
                print(f"  PM: Event prediction complete and updated (CI_Seq: {target_event_in_sequence.get('current_instance_seq')})")

                self._update_last_checked_motion_event(target_event_in_sequence)
                self.last_processed_event_internal_idx = current_internal_idx
                return True

    def resume_prediction_after_command(self):
        if self.paused_due_to_conflict:
            print("PM: New command received, resuming prediction paused by conflict.")
            self.paused_due_to_conflict = False
            self.event_causing_pause = None
        if self.paused_waiting_for_command:
            print("PM: New command received, resuming prediction paused by skeleton command (via resume_prediction_after_command).")
            self.paused_waiting_for_command = False


    def _insert_event_into_sequence(self, event_to_insert_template, at_internal_idx):

        event_to_insert = copy.deepcopy(event_to_insert_template)

        base_seq_for_new_event = 0
        if at_internal_idx > 0 and at_internal_idx <= len(self.current_prediction_sequence):

            base_seq_for_new_event = self.current_prediction_sequence[at_internal_idx - 1].get('current_instance_seq',
                                                                                               0)
        elif not self.current_prediction_sequence:
            base_seq_for_new_event = 0
        elif at_internal_idx == 0:
            original_first_seq = self.current_prediction_sequence[0].get('current_instance_seq', 1)
            base_seq_for_new_event = original_first_seq - 1 if original_first_seq > 0 else 0

        else:
            base_seq_for_new_event = self.current_prediction_sequence[-1].get('current_instance_seq',
                                                                              0) if self.current_prediction_sequence else 0

        assigned_ci_seq = base_seq_for_new_event + 1
        event_to_insert['current_instance_seq'] = assigned_ci_seq

        num_shifted = 0
        for i in range(at_internal_idx, len(self.current_prediction_sequence)):
            event_ci_seq = self.current_prediction_sequence[i].get('current_instance_seq')
            if event_ci_seq is not None and event_ci_seq >= assigned_ci_seq:
                self.current_prediction_sequence[i]['current_instance_seq'] += 1
                num_shifted += 1

        self.current_prediction_sequence.insert(at_internal_idx, event_to_insert)

        return event_to_insert

    def process_new_command_event_from_perception(self, new_command_event_from_perception):

        cmd_for_info = copy.deepcopy(new_command_event_from_perception)
        if 'current_instance_seq' not in cmd_for_info:
            cmd_for_info['current_instance_seq'] = cmd_for_info.get('seq')
        self._update_key_information_from_event(cmd_for_info, is_from_perception=True)


        cmd_sig = self._get_command_signature(new_command_event_from_perception)
        if cmd_sig and cmd_sig in self.processed_perception_command_signatures:

            return


        if not self.is_active: return
        if not self.initial_events_processed:
            print("PM: Skeleton not initialized, cannot process new command logic (but the command has been recorded).")
            return

        if cmd_sig:
            self.processed_perception_command_signatures.add(cmd_sig)

        print(f"\nPM (Prediction Active): Processing new perceived command: PerceptionSeq {new_command_event_from_perception.get('seq')}, "
              f"Content: {new_command_event_from_perception.get('content', {}).get('direction')}")

        processed_new_cmd_template = copy.deepcopy(new_command_event_from_perception)
        processed_new_cmd_template['is_skeleton_event'] = False


        was_paused_waiting_for_sk_command = self.paused_waiting_for_command
        self.resume_prediction_after_command()

        if was_paused_waiting_for_sk_command:

            insert_at_internal_idx = self.last_processed_event_internal_idx
            print(f"  PM: Was paused for a skeleton command. The new command will be inserted at index {insert_at_internal_idx} (the original skeleton command's position).")
        else:
            insert_at_internal_idx = self.last_processed_event_internal_idx + 1
            print(f"  PM: Not in a paused-for-command state. The new command will be inserted at index {insert_at_internal_idx} (after the last processed event).")


        if insert_at_internal_idx < 0:
            insert_at_internal_idx = 0
        if insert_at_internal_idx > len(self.current_prediction_sequence):
            insert_at_internal_idx = len(self.current_prediction_sequence)


        print(f"  PM: Inserting new perceived command at index {insert_at_internal_idx}.")
        inserted_actual_cmd_event = self._insert_event_into_sequence(processed_new_cmd_template, insert_at_internal_idx)


        self.last_processed_event_internal_idx = insert_at_internal_idx
        print(
            f"  PM: 'last_processed_event_internal_idx' updated to: {self.last_processed_event_internal_idx} (points to the newly inserted actual command).")


        event_after_inserted_cmd_idx = self.last_processed_event_internal_idx + 1
        arbitrary_templates_to_insert = []

        if event_after_inserted_cmd_idx < len(self.current_prediction_sequence):
            event_after_inserted_cmd = self.current_prediction_sequence[event_after_inserted_cmd_idx]

            if event_after_inserted_cmd.get("category") == "command" and event_after_inserted_cmd.get(
                    "is_skeleton_event"):
                actual_content = inserted_actual_cmd_event.get("content", {})
                skeleton_content = event_after_inserted_cmd.get("content", {})

                if actual_content.get("direction") == skeleton_content.get("direction"):
                    print(
                        f"  PM: The newly inserted actual command (CI_Seq: {inserted_actual_cmd_event.get('current_instance_seq')}) compared to the subsequent skeleton command "
                        f"(CI_Seq: {event_after_inserted_cmd.get('current_instance_seq')}, Index: {event_after_inserted_cmd_idx}) has the same direction. Deleting skeleton command.")

                    deleted_skeleton_cmd_ci_seq = event_after_inserted_cmd.get('current_instance_seq')
                    del self.current_prediction_sequence[event_after_inserted_cmd_idx]

                    for i in range(event_after_inserted_cmd_idx, len(self.current_prediction_sequence)):
                        event_ci_seq_val = self.current_prediction_sequence[i].get('current_instance_seq')
                        if event_ci_seq_val is not None and event_ci_seq_val > deleted_skeleton_cmd_ci_seq:
                            self.current_prediction_sequence[i]['current_instance_seq'] -= 1
                    print(f"    PM: The identical skeleton command has been deleted from the sequence.")

                else:
                    print(
                        f"  PM: The newly inserted actual command (CI_Seq: {inserted_actual_cmd_event.get('current_instance_seq')}) compared to the subsequent skeleton command "
                        f"(CI_Seq: {event_after_inserted_cmd.get('current_instance_seq')}, Index: {event_after_inserted_cmd_idx}) has a different direction. Preparing to insert arbitrary motion events.")
                    arbitrary_templates_to_insert.extend([
                        {"category": "ball_motion",
                         "content": {"current_pos": UNKNOWN_PLACEHOLDER, "new_direction": UNKNOWN_PLACEHOLDER,
                                     "tags": []},
                         "is_skeleton_event": True, "seq_in_skeleton": None},
                        {"category": "arm_motion",
                         "content": {"current_pos": UNKNOWN_PLACEHOLDER, "new_direction": UNKNOWN_PLACEHOLDER},
                         "is_skeleton_event": True, "seq_in_skeleton": None}
                    ])

        current_insert_idx_for_arbitrary = self.last_processed_event_internal_idx + 1
        num_arbitrary_actually_inserted = 0
        for template_idx, template in enumerate(arbitrary_templates_to_insert):

            self._insert_event_into_sequence(template,
                                             current_insert_idx_for_arbitrary + num_arbitrary_actually_inserted)
            num_arbitrary_actually_inserted += 1

        if num_arbitrary_actually_inserted > 0:
            print(f"  PM: Inserted {num_arbitrary_actually_inserted} arbitrary motion events between the actual command and the subsequent (potential) skeleton command.")



        return


    def clear_state_for_new_instance(self):
        print("PM: Clearing PredictionModule state for a new episode...")
        self.paused_waiting_for_command = False
        self.last_checked_ball_motion_event = None
        self.last_checked_arm_motion_event = None
        self.last_perception_command_event = None
        self.abstract_identifier_library = {}
        self.current_prediction_sequence = []
        self.current_skeleton_source_instance_id = None
        self.initial_events_processed = False
        self.last_processed_event_internal_idx = -1
        self.paused_due_to_conflict = False
        self.event_causing_pause = None
        self.prompted_inactive_this_instance = False
        self.processed_perception_command_signatures = set()
        print("PM: PredictionModule state cleared.")

    def get_current_prediction_sequence(self):

        return copy.deepcopy(self.current_prediction_sequence)

    # --- Helper for debugging/display ---
    def print_current_prediction_sequence(self, num_events_to_show=None):
        print("\n--- PredictionModule: Current Prediction Sequence ---")
        if not self.current_prediction_sequence:
            print("  (empty)")
            print(
                f"  --- (Active: {self.is_active}, InitialProcessed: {self.initial_events_processed}, AbsLib: {self.abstract_identifier_library}) ---")
            return


        events_to_print = self.current_prediction_sequence
       # if num_events_to_show is not None and num_events_to_show < len(self.current_prediction_sequence):
           # events_to_print = self.current_prediction_sequence[:num_events_to_show]

        for idx, event in enumerate(events_to_print):
            is_skel_str = "SK" if event.get('is_skeleton_event') else "AC"
            sk_seq_str = f"(SkelSeq:{event.get('seq_in_skeleton')})" if event.get('is_skeleton_event') else ""
            proc_mark = "*" if idx <= self.last_processed_event_internal_idx else " "
            content_str = str(event.get('content', {}))
            rules_str = f" (Rules: {event['prediction_rules_applied']})" if event.get(
                "prediction_rules_applied") else ""
            print(f"  {proc_mark} [{idx}] CI_Seq:{event.get('current_instance_seq')} "
                  f"[{is_skel_str}{sk_seq_str}] {event.get('category')}: {content_str}{rules_str}")
       # if num_events_to_show is not None and len(self.current_prediction_sequence) > num_events_to_show:
           # print(f"  ... and {len(self.current_prediction_sequence) - num_events_to_show} more events.")
        print(
            f"  --- (LastProcIntIdx: {self.last_processed_event_internal_idx}, Paused: {self.paused_due_to_conflict}, "
            f"Active: {self.is_active}, InitialProcessed: {self.initial_events_processed}, AbsLib: {self.abstract_identifier_library}) ---")