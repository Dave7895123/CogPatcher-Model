# --- START OF FILE schematic_memory.py ---

from tabulate import tabulate
from copy import deepcopy
import itertools

UNKNOWN_PLACEHOLDER = "ANY_PLACEHOLDER"
ABSTRACT_ID_PREFIX = "#"
ABSTRACT_ID_SUFFIX = "#"
OCCLUSION_TAG = "IS_OCCLUDED"


class SchematicMemory:

    NEW_RULE2_CONSTRAINTS_DATA = {
        "LEFT_REAR QUADRANT": {
            "STATIONARY": {
                "main": {"CW": ("LEFT_FRONT", "LEFT_REAR QUADRANT"), "CCW": ("RIGHT_REAR", "LEFT_REAR QUADRANT")},
                "extra_arm": {"CW": ("RIGHT_FRONT", "LEFT_FRONT QUADRANT"), "CCW": ("RIGHT_FRONT", "RIGHT_REAR QUADRANT")}
            },
            "LEFT_FRONT": ("RIGHT_FRONT", "LEFT_FRONT QUADRANT"),
            "RIGHT_REAR": ("RIGHT_FRONT", "RIGHT_REAR QUADRANT")
        },
        "RIGHT_REAR QUADRANT": {
            "STATIONARY": {
                "main": {"CW": ("LEFT_REAR", "RIGHT_REAR QUADRANT"), "CCW": ("RIGHT_FRONT", "RIGHT_REAR QUADRANT")},
                "extra_arm": {"CW": ("LEFT_FRONT", "LEFT_REAR QUADRANT"), "CCW": ("LEFT_FRONT", "RIGHT_FRONT QUADRANT")}
            },
            "LEFT_REAR": ("LEFT_FRONT", "LEFT_REAR QUADRANT"),
            "RIGHT_FRONT": ("LEFT_FRONT", "RIGHT_FRONT QUADRANT")
        },
        "RIGHT_FRONT QUADRANT": {
            "STATIONARY": {
                "main": {"CW": ("RIGHT_REAR", "RIGHT_FRONT QUADRANT"), "CCW": ("LEFT_FRONT", "RIGHT_FRONT QUADRANT")},
                "extra_arm": {"CW": ("LEFT_REAR", "RIGHT_REAR QUADRANT"), "CCW": ("LEFT_REAR", "LEFT_FRONT QUADRANT")}
            },
            "LEFT_FRONT": ("LEFT_REAR", "LEFT_FRONT QUADRANT"),
            "RIGHT_REAR": ("LEFT_REAR", "RIGHT_REAR QUADRANT")
        },
        "LEFT_FRONT QUADRANT": {
            "STATIONARY": {
                "main": {"CW": ("RIGHT_FRONT", "LEFT_FRONT QUADRANT"), "CCW": ("LEFT_REAR", "LEFT_FRONT QUADRANT")},
                "extra_arm": {"CW": ("RIGHT_REAR", "RIGHT_FRONT QUADRANT"), "CCW": ("RIGHT_REAR", "LEFT_REAR QUADRANT")}
            },
            "LEFT_REAR": ("RIGHT_REAR", "LEFT_REAR QUADRANT"),
            "RIGHT_FRONT": ("RIGHT_REAR", "RIGHT_FRONT QUADRANT")
        }
    }

    def __init__(self, conflict_resolution_module=None):
        self.schemas = []
        self.rules = []
        self._define_special_rules()
        self.conflict_resolver = conflict_resolution_module
        print(f"SchematicMemory initialized {'with' if self.conflict_resolver else 'without'} ConflictResolutionModule.")

    def _is_initial_event_content_uncertain(self, event_content):
        if not isinstance(event_content, dict):
            return False
        for value in event_content.values():
            if value == UNKNOWN_PLACEHOLDER or value == "UNKNOWN_VALUE":
                return True
        return False

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

    def get_rule_by_id(self, rule_id):
        for rule in self.rules:
            if rule["rule_id"] == rule_id:
                return rule
        return None

    def _define_special_rules(self):
        rule1 = {
            "rule_id": "CMD_ARM_CONSISTENCY",
            "description": "Predicts the arm's motion direction immediately following a command.",
            "type": "causal_sequence_prediction",
            "trigger_event_category": "command",
            "predicted_event_category": "arm_motion",
            "condition_context": "next_non_null_event_batch_after_trigger",
            "action": {
                "copy_from_trigger_event": {
                    "source_field": "content.direction",
                    "target_field": "content.new_direction"
                }
            }
        }
        self.rules.append(rule1)


        rule2_definition = {
            "rule_id": "MOTION_CONTINUITY",
            "description": "Predicts the continuity of an object's position and direction of motion.",
            "type": "state_transition_prediction",
            "trigger_event_category": ["ball_motion", "arm_motion"],
            "predicted_event_category": ["ball_motion", "arm_motion"],
            # "constraints" 字段保留结构但不再由旧逻辑使用，新的约束在 NEW_RULE2_CONSTRAINTS_DATA
            "constraints": {},
            "action": {"custom_logic_in_prediction_pass": True}  # 标记为自定义逻辑
        }
        self.rules.append(rule2_definition)

    def add_schema(self, schema):
        self.schemas.append(schema)

    def get_schemas(self):
        return self.schemas

    def get_rules(self):
        return self.rules

    def _events_are_similar_for_skeleton_selection(self, actual_event, template_event, temp_bindings, current_library):
        if actual_event.get("category") != template_event.get("category"):
            return False

        actual_content = actual_event.get("content", {})
        template_content = template_event.get("content", {})

        if not isinstance(actual_content, dict) or not isinstance(template_content, dict):
            return str(actual_content) == str(template_content)

        fields_to_compare = []
        category = actual_event.get("category")

        if category in ["ball_motion", "arm_motion"]:
            fields_to_compare = ["current_pos", "new_direction"]
        elif category == "command":
            fields_to_compare = ["direction", "rotation"]

        for field in fields_to_compare:
            actual_value = actual_content.get(field)
            template_value = template_content.get(field)

            if category == "command" and field == "rotation":
                if template_value == "N/A" or template_value is None:
                    continue

            if template_value == UNKNOWN_PLACEHOLDER or template_value == "UNKNOWN_VALUE":
                continue

            if self._is_abstract_identifier(template_value):
                abstract_id = template_value
                if abstract_id in temp_bindings:
                    if temp_bindings[abstract_id] != actual_value:
                        return False
                elif abstract_id in current_library:
                    if current_library[abstract_id] != actual_value:
                        return False
                else:
                    if actual_value is not None and actual_value != UNKNOWN_PLACEHOLDER and actual_value != "UNKNOWN_VALUE":
                        temp_bindings[abstract_id] = actual_value
            elif actual_value != template_value:
                if actual_value == "UNKNOWN_VALUE":
                    continue
                return False
        return True

    def select_and_construct_skeleton_with_initial_events(self, processed_initial_events_for_skeleton_selection,
                                                          current_abstract_library):
        print(
            f"\nSM: Starting to select a skeleton and construct the full sequence with initial events (based on {len(processed_initial_events_for_skeleton_selection)} initial actual events)...")
        print(f"  SM: Received initial actual events (from PM):")
        for idx, ev in enumerate(processed_initial_events_for_skeleton_selection):
            is_skel_str = "SK" if ev.get('is_skeleton_event') else "AC"
            print(
                f"    [{idx}] CI_Seq:{ev.get('current_instance_seq')} [{is_skel_str}] {ev.get('category')}: {ev.get('content')}")

        print(f"  SM: Current abstract library received: {current_abstract_library}")

        if not processed_initial_events_for_skeleton_selection:
            print("  SM_ERROR: The list of initial actual events is empty.")
            return None, current_abstract_library, None


        initial_events_to_use_for_matching = []

        if self.conflict_resolver:
            print("  SM: Checking for uncertainty in initial events and attempting to resolve with ConflictResolver...")
            for original_event in processed_initial_events_for_skeleton_selection:
                event_copy_for_resolution = deepcopy(original_event)
                event_content = event_copy_for_resolution.get("content", {})

                if not event_copy_for_resolution.get('is_skeleton_event', False) and \
                        self._is_initial_event_content_uncertain(event_content) and \
                        OCCLUSION_TAG in event_content.get("tags", []):
                    print(
                        f"    SM: Found an uncertain initial actual event in the occlusion zone (Cat: {event_copy_for_resolution.get('category')}, CI_Seq: {event_copy_for_resolution.get('current_instance_seq')}): {event_content}")
                    try:
                        resolved_event_from_cr = self.conflict_resolver.resolve_uncertainty_in_initial_event(
                            event_copy_for_resolution)
                        if resolved_event_from_cr.get("content") != event_copy_for_resolution.get("content"):
                            print(f"      SM: Content after resolution by ConflictResolver: {resolved_event_from_cr.get('content')}")
                        initial_events_to_use_for_matching.append(resolved_event_from_cr)
                    except AttributeError:
                        print(
                            "    SM_WARN: ConflictResolutionModule is missing 'resolve_uncertainty_in_initial_event' method. Skipping uncertainty resolution.")
                        initial_events_to_use_for_matching.append(event_copy_for_resolution)
                    except Exception as e:
                        print(f"    SM_ERROR: Error calling ConflictResolver to resolve uncertainty: {e}. Using original event.")
                        initial_events_to_use_for_matching.append(event_copy_for_resolution)
                else:
                    initial_events_to_use_for_matching.append(event_copy_for_resolution)

            print(f"  SM: Initial event uncertainty check and resolution complete. Using the following list for matching:")
            for idx, ev in enumerate(initial_events_to_use_for_matching):
                is_skel_str = "SK" if ev.get('is_skeleton_event') else "AC"
                print(
                    f"    [{idx}] CI_Seq:{ev.get('current_instance_seq')} [{is_skel_str}] {ev.get('category')}: {ev.get('content')}")

        else:
            print("  SM: ConflictResolver not configured. Using initial events as is.")
            initial_events_to_use_for_matching = deepcopy(processed_initial_events_for_skeleton_selection)


        all_schemas = self.get_schemas()
        if not all_schemas:
            print("  SM: No schemas available in SchematicMemory to select from.")
            return None, current_abstract_library, None

        best_match_schema_obj = None
        min_event_count_for_best_match = float('inf')
        max_arbitrary_count_for_best_match = -1
        final_bindings_for_best_match = {}

        num_initial_to_match = len(initial_events_to_use_for_matching)
        print(
            f"  SM: Using {num_initial_to_match} processed actual events as the skeleton head. Comparing with the initial part of {len(all_schemas)} schemas to select the best remaining skeleton.")

        for adv_idx, adv_schema in enumerate(all_schemas):
            adv_events = adv_schema.get("events", [])
            adv_schema_id = adv_schema.get('schema_type', f"AdvInstance_{adv_idx}")

            EXPECTED_TOTAL_EVENTS = 16
            if len(adv_events) != EXPECTED_TOTAL_EVENTS:
                continue

            if len(adv_events) < num_initial_to_match:
                continue

            template_initial_events_from_adv_schema = adv_events[:num_initial_to_match]
            permutation_matched_this_candidate = False
            bindings_from_successful_permutation = {}


            for permuted_actual_events in itertools.permutations(initial_events_to_use_for_matching):
                temp_bindings_for_this_permutation = {}
                all_events_in_permutation_match = True
                for i in range(num_initial_to_match):
                    actual_event = permuted_actual_events[i]
                    template_event = template_initial_events_from_adv_schema[i]
                    if not self._events_are_similar_for_skeleton_selection(
                            actual_event,
                            template_event,
                            temp_bindings_for_this_permutation,
                            current_abstract_library
                    ):
                        all_events_in_permutation_match = False
                        break
                if all_events_in_permutation_match:
                    permutation_matched_this_candidate = True
                    bindings_from_successful_permutation = temp_bindings_for_this_permutation
                    break

            if not permutation_matched_this_candidate:
                continue

            print(
                f"  SM: Initial part of candidate {adv_schema_id} matches PM's initial events. Temporary bindings: {bindings_from_successful_permutation}")

            current_generalization_score = 0
            for skel_ev_idx in range(num_initial_to_match, len(adv_events)):
                skeleton_event_content = adv_events[skel_ev_idx].get("content", {})
                if not isinstance(skeleton_event_content, dict):
                    continue
                for value in skeleton_event_content.values():
                    if value == UNKNOWN_PLACEHOLDER:
                        current_generalization_score += 1
                    elif self._is_abstract_identifier(value):
                        current_generalization_score += 1

            is_new_best = False
            if best_match_schema_obj is None:
                is_new_best = True
            elif current_generalization_score > max_arbitrary_count_for_best_match:
                is_new_best = True

            if is_new_best:
                print(
                    f"    Found a new best candidate for the remaining skeleton: {adv_schema_id} (Generalization Score: {current_generalization_score})")
                best_match_schema_obj = adv_schema
                max_arbitrary_count_for_best_match = current_generalization_score
                final_bindings_for_best_match = bindings_from_successful_permutation

                best_match_schema_obj = adv_schema
                max_arbitrary_count_for_best_match = current_generalization_score
                final_bindings_for_best_match = bindings_from_successful_permutation


        if not best_match_schema_obj:
            print("  SM: Failed to find a schema whose initial part matches the PM's events to serve as a source for the remaining skeleton.")
            return None, current_abstract_library, None

        chosen_source_id = best_match_schema_obj.get('schema_type', 'UnknownChosenAdvInstance')
        print(
            f"  SM: Final skeleton source selected (for the remaining part): {chosen_source_id}. Applying initial bindings: {final_bindings_for_best_match}")

        updated_library = deepcopy(current_abstract_library)
        for abstract_id, concrete_value in final_bindings_for_best_match.items():
            if concrete_value == "UNKNOWN_VALUE" or concrete_value == UNKNOWN_PLACEHOLDER: continue
            if abstract_id not in updated_library:
                updated_library[abstract_id] = concrete_value
            elif updated_library[abstract_id] != concrete_value:
                print(
                    f"    SM_Bind_WARN: Attempting to update binding for '{abstract_id}' from '{updated_library[abstract_id]}' to '{concrete_value}'. The new value will be used.")
                updated_library[abstract_id] = concrete_value

        complete_skeleton_sequence = []


        for actual_initial_event in initial_events_to_use_for_matching:
            event_copy = deepcopy(actual_initial_event)
            complete_skeleton_sequence.append(event_copy)

        print(f"  SM: Added {len(initial_events_to_use_for_matching)} (potentially resolved) initial events to the full skeleton head.")


        full_skeleton_events_from_adv = best_match_schema_obj.get("events", [])
        next_ci_seq_for_skeleton_part = 0
        if complete_skeleton_sequence:
            last_initial_event_ci_seq = complete_skeleton_sequence[-1].get('current_instance_seq', 0)
            next_ci_seq_for_skeleton_part = last_initial_event_ci_seq
        else:
            next_ci_seq_for_skeleton_part = 0
        skeleton_event_templates_added_count = 0
        for i in range(num_initial_to_match, len(full_skeleton_events_from_adv)):
            event_template_copy = deepcopy(full_skeleton_events_from_adv[i])
            if event_template_copy.get("category") == "padding":
                print(
                    f"  SM: Filtering out padding event from skeleton: SeqInSchema={event_template_copy.get('seq')}, Content={event_template_copy.get('content')}")
                continue
            next_ci_seq_for_skeleton_part += 1
            event_template_copy['current_instance_seq'] = next_ci_seq_for_skeleton_part
            event_template_copy['is_skeleton_event'] = True
            event_template_copy['seq_in_skeleton'] = event_template_copy.get('seq')
            if 'seq' in event_template_copy: del event_template_copy['seq']
            complete_skeleton_sequence.append(event_template_copy)
            skeleton_event_templates_added_count += 1
        print(f"  SM: Added {skeleton_event_templates_added_count} skeleton event templates (padding events filtered).")
        print(f"  SM: Total length of the returned full skeleton sequence: {len(complete_skeleton_sequence)}")
        print(f"  SM: Updated abstract library: {updated_library}")
        print(f"  SM: The full skeleton sequence to be returned to PM:")
        for idx, ev in enumerate(complete_skeleton_sequence):
            is_skel_str = "SK" if ev.get('is_skeleton_event') else "AC"
            sk_seq_str = f"(SkelSeq:{ev.get('seq_in_skeleton')})" if ev.get('is_skeleton_event') else ""
            print(
                f"    [{idx}] CI_Seq:{ev.get('current_instance_seq')} [{is_skel_str}{sk_seq_str}] {ev.get('category')}: {ev.get('content')}")

        return complete_skeleton_sequence, updated_library, chosen_source_id

    def _apply_final_library_bindings_to_event_content(self, event_content, category_to_predict, library):
        if not isinstance(event_content, dict) or not library:
            return

        fields_to_check = []
        if category_to_predict in ["ball_motion", "arm_motion"]:
            fields_to_check = ["current_pos", "new_direction"]
        elif category_to_predict == "command":
            fields_to_check = ["direction", "rotation"]

        for field_name in fields_to_check:
            current_value = event_content.get(field_name)
            if self._is_abstract_identifier(current_value):
                abstract_id = current_value
                if abstract_id in library:
                    bound_value = library[abstract_id]
                    if event_content.get(field_name) != bound_value:
                        print(f"    SM_Final_Bind: Field '{field_name}'s abstract identifier '{abstract_id}' was resolved from the library to '{bound_value}'")
                        event_content[field_name] = bound_value

    def _format_advanced_event_content(self, event):
        content = event.get("content", {})
        category = event.get("category", "unknown")
        rules_applied_str = ""
        if event.get("prediction_rules_applied"):
            rules_applied_str = f" (Rules: {', '.join(event['prediction_rules_applied'])})"

        if category == "padding": return f"Padding ({content.get('type', 'unknown_padding')}){rules_applied_str}"
        if not isinstance(content, dict): return f"{str(content)}{rules_applied_str}"

        if category == "command":
            return f"Cmd: {content.get('original_command', 'N/A')} (Dir: {content.get('direction', 'N/A')}, Rot: {content.get('rotation', 'N/A')}){rules_applied_str}"
        elif category in ["ball_speed", "arm_speed"]:
            return f"Speed: {content.get('change', 'N/A')}{rules_applied_str}"
        elif category == "ball_motion":
            tags_str = f", Tags: {content.get('tags', [])}" if content.get('tags') else ""
            return f"Dir: {content.get('new_direction', 'N/A')} (From: {content.get('current_pos', 'N/A')}{tags_str}){rules_applied_str}"
        elif category == "arm_motion":
            return f"Dir: {content.get('new_direction', 'N/A')} (From: {content.get('current_pos', 'N/A')}){rules_applied_str}"
        elif category == "collision":
            return f"Collision: {content.get('collision', 'N/A')}{rules_applied_str}"
        else:
            main_content = "; ".join(f"{k}: {v}" for k, v in content.items()) if content else "N/A"
            return f"{main_content}{rules_applied_str}"

    def print_schema_table(self, schema_index):
        if schema_index < 0 or schema_index >= len(self.schemas):
            print(f"Schema index {schema_index} is invalid (total schemas: {len(self.schemas)}).")
            return
        schema = self.schemas[schema_index]
        schema_label = f"Schema {schema_index + 1}"
        print(f"\n{schema_label}")
        metadata_parts = []
        if schema.get('schema_type'): metadata_parts.append(f"Type: {schema.get('schema_type')}")
        if schema.get("common_key_events"): metadata_parts.append(
            f"Number of common key events: {len(schema.get('common_key_events'))}")
        if schema.get("generalization_rules_applied"): metadata_parts.append(
            f"Generalization rules applied: {schema.get('generalization_rules_applied')}")
        if metadata_parts: print(f"  Metadata: {'; '.join(metadata_parts)}")
        events = schema.get("events", [])
        if not events:
            print(f"{schema_label} has no events.")
        else:
            event_headers = ["Seq", "Category", "Content"]
            event_table_data = [
                [ev.get("seq", "N/A"), ev.get("category", "N/A"), self._format_advanced_event_content(ev)] for ev in
                events]
            print(f"{schema_label} Event List:")
            print(tabulate(event_table_data, headers=event_headers, tablefmt="grid"))

    def _is_event_content_fully_concrete(self, event):
        content = event.get("content", {})
        category = event.get("category")
        if not isinstance(content, dict): return True
        fields_to_check = []
        if category in ["ball_motion", "arm_motion"]:
            fields_to_check = ["current_pos", "new_direction"]
        elif category == "command":
            fields_to_check = ["direction", "rotation"]
        if not fields_to_check: return True
        for field_name in fields_to_check:
            value = content.get(field_name)
            if value is None: return False
            if value == UNKNOWN_PLACEHOLDER or value == "UNKNOWN_VALUE" or self._is_abstract_identifier(value): return False
        return True

    def predict_event_details_iteratively(self, last_checked_ball_motion_event,
                                          last_checked_arm_motion_event,
                                          last_perception_command_event,
                                          original_event_template, abstract_identifier_library_input, max_iterations=5):
        print(f"\nSM_Iterative: Starting iterative prediction of event details...")
        current_event_template = deepcopy(original_event_template)
        current_library = deepcopy(abstract_identifier_library_input)
        last_special_command = None

        if self.conflict_resolver:
            event_content_check = current_event_template.get("content", {})

            if self._is_initial_event_content_uncertain(event_content_check) and \
                    OCCLUSION_TAG in event_content_check.get("tags", []):
                print(
                    f"  SM_Iterative: Initial template (Cat: {current_event_template.get('category')}) is uncertain and in occlusion zone, attempting CRM resolution.")
                try:
                    resolved_template_from_cr = self.conflict_resolver.resolve_uncertainty_for_am(
                        deepcopy(current_event_template))
                    if resolved_template_from_cr.get("content") != current_event_template.get("content"):
                        print(f"    SM_Iterative: Template content after CRM resolution: {resolved_template_from_cr.get('content')}")
                        current_event_template = resolved_template_from_cr
                    else:
                        print(f"    SM_Iterative: CRM did not modify the template.")
                except AttributeError:
                    print("    SM_Iterative_WARN: ConflictResolutionModule is missing 'resolve_uncertainty_for_am' method.")
                except Exception as e:
                    print(f"    SM_Iterative_ERROR: Error calling CRM to resolve template: {e}.")


        for i in range(max_iterations):
            print(
                f"  SM_Iterative: Iteration {i + 1}. Current template: {current_event_template.get('content')}, Library: {current_library}")
            event_before_pass = deepcopy(current_event_template)

            predicted_event_pass, updated_library_pass, special_command_pass = \
                self._predict_event_details_single_pass(
                    last_checked_ball_motion_event,
                    last_checked_arm_motion_event,
                    last_perception_command_event,
                    current_event_template,
                    current_library
                )

            current_event_template = predicted_event_pass
            current_library = updated_library_pass
            if special_command_pass:
                last_special_command = special_command_pass


            print(
                f"  SM_Iterative: After iteration {i + 1}. Event: {current_event_template.get('content')}, Library: {current_library}, Special command: {last_special_command}")


            if self._is_event_content_fully_concrete(current_event_template):
                print(f"  SM_Iterative: Event is fully concrete at the rule level or has no more changes after iteration {i + 1}.")

            if event_before_pass == current_event_template and not special_command_pass:  # 确保没有特殊命令阻止break
                print(f"  SM_Iterative: Iteration {i + 1} did not change event content. Stopping rule iteration.")
                break
        else:
            if not self._is_event_content_fully_concrete(current_event_template):
                print(f"  SM_Iterative: Reached max iterations ({max_iterations}), event may still not be fully concrete at the rule level.")


        print(f"  SM_Iterative: Rule iteration finished. Applying final library bindings...")
        print(f"    Event content (before binding): {current_event_template.get('content')}")
        print(f"    Using library (before binding): {current_library}")

        print(
            f"  SM_Iterative: >>> DEBUG: FOR LOOP COMPLETED OR BROKEN. Current event BEFORE final bind: {current_event_template.get('content')}")  # 新增的调试语句
        print(f"  SM_Iterative: Rule iteration finished. Applying final library bindings...")

        if "content" in current_event_template and isinstance(current_event_template["content"], dict):
            self._apply_final_library_bindings_to_event_content(
                current_event_template["content"],
                current_event_template.get("category"),
                current_library
            )
            print(f"    Event content (after binding): {current_event_template.get('content')}")
        else:
            print(f"    SM_Iterative: Event template has no valid 'content' dictionary, skipping final library binding.")

        print(
            f"SM_Iterative: Iteration and final binding finished. Final event: {current_event_template.get('content')}, Final library: {current_library}")
        return current_event_template, current_library, last_special_command

    def _field_needs_filling(self, value):
        return value == UNKNOWN_PLACEHOLDER or \
            value == "UNKNOWN_VALUE" or \
            self._is_abstract_identifier(value) or \
            value is None

    def _field_needs_filling(self, value):
        return value == UNKNOWN_PLACEHOLDER or \
            value == "UNKNOWN_VALUE" or \
            self._is_abstract_identifier(value) or \
            value is None

    def _predict_event_details_single_pass(self, last_checked_ball_motion_event,
                                           last_checked_arm_motion_event,
                                           last_perception_command_event,
                                           original_event_template, abstract_identifier_library_input):
        event_resolved_by_rules = deepcopy(original_event_template)
        event_resolved_by_rules["prediction_rules_applied"] = \
            deepcopy(original_event_template.get("prediction_rules_applied", []))

        category_to_predict = event_resolved_by_rules.get("category")
        if "content" not in event_resolved_by_rules: event_resolved_by_rules["content"] = {}
        content_resolved_by_rules = event_resolved_by_rules["content"]

        print(f"    SM_TRACE: --- START _predict_event_details_single_pass ---")
        print(
            f"    SM_TRACE: Original Template Category: {category_to_predict}, Content: {original_event_template.get('content')}")
        print(
            f"    SM_TRACE: last_ball_motion: {last_checked_ball_motion_event.get('content') if last_checked_ball_motion_event else 'None'}")
        print(
            f"    SM_TRACE: last_arm_motion: {last_checked_arm_motion_event.get('content') if last_checked_arm_motion_event else 'None'}")
        print(
            f"    SM_TRACE: last_perception_cmd: {last_perception_command_event.get('content') if last_perception_command_event else 'None'}")
        print(f"    SM_TRACE: Initial abstract_lib: {abstract_identifier_library_input}")

        MAX_INNER_RULE_ITERATIONS = 1
        for rule_iter_count in range(MAX_INNER_RULE_ITERATIONS):
            print(f"    SM_TRACE: Starting inner rule iteration {rule_iter_count + 1}")
            made_change_in_this_pass = False

            rule1 = self.get_rule_by_id("CMD_ARM_CONSISTENCY")
            if rule1 and \
                    category_to_predict == rule1.get("predicted_event_category") and \
                    last_perception_command_event and \
                    last_perception_command_event.get("category") == rule1.get("trigger_event_category"):

                print(f"    SM_TRACE: Rule 1 condition MET.")
                target_field_value_rule1 = content_resolved_by_rules.get("new_direction")
                if self._field_needs_filling(target_field_value_rule1):
                    cmd_content = last_perception_command_event.get("content", {})
                    cmd_direction = cmd_content.get("direction")
                    if cmd_direction and cmd_direction != UNKNOWN_PLACEHOLDER and \
                            cmd_direction != "UNKNOWN_VALUE" and not self._is_abstract_identifier(cmd_direction):
                        if content_resolved_by_rules.get("new_direction") != cmd_direction:
                            content_resolved_by_rules["new_direction"] = cmd_direction
                            print(f"    Rule CMD_ARM_CONSISTENCY: Arm motion direction set to -> {cmd_direction}")
                            rule_id_to_add = rule1.get("rule_id")
                            if rule_id_to_add and rule_id_to_add not in event_resolved_by_rules[
                                "prediction_rules_applied"]:
                                event_resolved_by_rules["prediction_rules_applied"].append(rule_id_to_add)
                            made_change_in_this_pass = True
                            print(
                                f"    SM_TRACE: Rule 1 APPLIED. made_change_in_this_pass = {made_change_in_this_pass}")


            print(
                f"    SM_TRACE: AFTER Rule 1 logic processing, BEFORE getting rule2_def. Category: {category_to_predict}")


            rule2_def = self.get_rule_by_id("MOTION_CONTINUITY")
            rule2_applied_this_iteration = False  # Renamed from rule2_applied_this_iteration_flag for consistency

            print(
                f"    SM_TRACE: Considering Rule 2 (MOTION_CONTINUITY). Rule2_def object: {'Exists' if rule2_def else 'None'}")

            if rule2_def and category_to_predict in rule2_def.get("predicted_event_category", []):
                print(f"    SM_TRACE: Rule 2 category condition MET.")

                # 检查前提事件是否完整
                if not (last_checked_ball_motion_event and \
                        last_checked_arm_motion_event and \
                        last_perception_command_event):
                    print(f"    SM_TRACE: Rule 2 - SKIPPING due to incomplete context events.")
                else:
                    context_motion_event = None
                    if category_to_predict == "ball_motion":
                        context_motion_event = last_checked_ball_motion_event
                    elif category_to_predict == "arm_motion":
                        context_motion_event = last_checked_arm_motion_event

                    print(
                        f"    SM_TRACE: Rule 2 - context_motion_event chosen: {'Exists' if context_motion_event else 'None'}")
                    if context_motion_event:
                        print(
                            f"    SM_TRACE: Rule 2 - context_motion_event content: {context_motion_event.get('content')}")

                    if context_motion_event and context_motion_event.get("content"):
                        print(f"    SM_TRACE: Rule 2 - ENTERING CORE LOGIC")
                        context_content = context_motion_event.get("content", {})
                        context_pos = context_content.get("current_pos")
                        context_dir = context_content.get("new_direction")

                        command_content = last_perception_command_event.get("content", {})
                        command_rotation = command_content.get("rotation")

                        current_event_cp_val = content_resolved_by_rules.get("current_pos")
                        current_event_nd_val = content_resolved_by_rules.get("new_direction")

                        cp_needs_filling = self._field_needs_filling(current_event_cp_val)
                        nd_needs_filling = self._field_needs_filling(current_event_nd_val)

                        print(
                            f"    SM_TRACE: Rule 2 - Core: context_pos='{context_pos}', context_dir='{context_dir}', cmd_rotation='{command_rotation}'")
                        print(
                            f"    SM_TRACE: Rule 2 - Core: current_event_cp_val='{current_event_cp_val}', current_event_nd_val='{current_event_nd_val}'")
                        print(
                            f"    SM_TRACE: Rule 2 - Core: cp_needs_filling={cp_needs_filling}, nd_needs_filling={nd_needs_filling}")

                        predicted_direction_to_set = None
                        predicted_landing_pos_to_set = None
                        new_rule_applied_this_pass_flag = False
                        if current_event_nd_val == "STATIONARY" and not nd_needs_filling and \
                                cp_needs_filling and \
                                context_pos and not self._field_needs_filling(context_pos):
                            predicted_landing_pos_to_set = context_pos


                            print(f"    Rule MOTION_CONTINUITY (New-Stationary Inherits Position): "
                                  f"Current event direction is 'STATIONARY' (template value: {current_event_nd_val}), "
                                  f"current event position needs filling (template value: {current_event_cp_val}). "
                                  f"Inheriting previous event's position '{context_pos}' as current event's position.")
                            new_rule_applied_this_pass_flag = True



                        if context_pos and context_dir:
                            pos_rules_from_data = self.NEW_RULE2_CONSTRAINTS_DATA.get(context_pos)

                            if not pos_rules_from_data:
                                print(
                                    f"    SM_TRACE: Rule 2 - Core: No rules in NEW_RULE2_CONSTRAINTS_DATA for context_pos '{context_pos}'.")
                                pass
                            elif cp_needs_filling and nd_needs_filling:

                                print(f"    SM_TRACE: Rule 2 - Core: Case 1 (CP & ND need filling)")
                                if context_dir != "STATIONARY":
                                    if context_dir in pos_rules_from_data:
                                        predicted_direction_to_set, predicted_landing_pos_to_set = pos_rules_from_data[
                                            context_dir]
                                    else:
                                        print(
                                            f"    SM_TRACE: Rule 2 - Core: Case 1 - Context_dir '{context_dir}' NOT found in pos_rules_from_data for '{context_pos}'.")
                                else:  # context_dir == "STATIONARY"
                                    print(f"    SM_TRACE: Rule 2 - Core: Case 1 - Context_dir is 'STATIONARY'.")
                                    if "STATIONARY" in pos_rules_from_data and command_rotation:
                                        still_rules_map = pos_rules_from_data["STATIONARY"]
                                        target_rotation_map_data = None
                                        if category_to_predict == "arm_motion" and "extra_arm" in still_rules_map:
                                            if command_rotation in still_rules_map.get("extra_arm", {}):  # Safe get
                                                target_rotation_map_data = still_rules_map["extra_arm"][
                                                    command_rotation]
                                        if target_rotation_map_data is None and "main" in still_rules_map:
                                            if command_rotation in still_rules_map.get("main", {}):  # Safe get
                                                target_rotation_map_data = still_rules_map["main"][command_rotation]
                                        if target_rotation_map_data:
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 1 - Still context, rotation match found. Preds: {target_rotation_map_data}")
                                            predicted_direction_to_set, predicted_landing_pos_to_set = target_rotation_map_data
                                        else:
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 1 - Still context, no rotation match or data.")

                            elif context_dir == "STATIONARY" and (cp_needs_filling != nd_needs_filling):

                                print(
                                    f"    SM_TRACE: Rule 2 - Core: Case 2 (Context 'STATIONARY', one of CP/ND needs filling)")
                                if "STATIONARY" not in pos_rules_from_data:
                                    print(
                                        f"    SM_TRACE: Rule 2 - Core: Case 2 - 'STATIONARY' rules not found for context_pos '{context_pos}'.")
                                    pass
                                else:
                                    still_rules_group = pos_rules_from_data["STATIONARY"]
                                    candidate_predictions = []
                                    for lib_type in ["extra_arm", "main"]:
                                        if category_to_predict != "arm_motion" and lib_type == "extra_arm":
                                            continue
                                        if lib_type in still_rules_group:
                                            for rot_key, (pred_dir, pred_pos) in still_rules_group[lib_type].items():
                                                candidate_predictions.append((pred_dir, pred_pos, rot_key, lib_type))

                                    print(
                                        f"    SM_TRACE: Rule 2 - Core: Case 2 - Candidates from 'STATIONARY' group: {candidate_predictions}")
                                    filtered_candidates = []
                                    if not cp_needs_filling:
                                        for pred_dir, pred_pos, rot_key, lib_type in candidate_predictions:
                                            if pred_pos == current_event_cp_val:
                                                filtered_candidates.append((pred_dir, pred_pos, rot_key, lib_type))
                                        if filtered_candidates:
                                            predicted_landing_pos_to_set = current_event_cp_val
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 2 - CP known ('{current_event_cp_val}'). Filtered for ND: {filtered_candidates}")
                                    elif not nd_needs_filling:
                                        for pred_dir, pred_pos, rot_key, lib_type in candidate_predictions:
                                            if pred_dir == current_event_nd_val:
                                                filtered_candidates.append((pred_dir, pred_pos, rot_key, lib_type))
                                        if filtered_candidates:
                                            predicted_direction_to_set = current_event_nd_val
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 2 - ND known ('{current_event_nd_val}'). Filtered for CP: {filtered_candidates}")

                                    if len(filtered_candidates) > 1 and command_rotation:
                                        print(
                                            f"    SM_TRACE: Rule 2 - Core: Case 2 - Multiple filtered ({len(filtered_candidates)}), using command_rotation '{command_rotation}'.")
                                        final_choice = None
                                        for pred_dir, pred_pos, rot_key, lib_type in filtered_candidates:
                                            if lib_type == "extra_arm" and rot_key == command_rotation:
                                                final_choice = (pred_dir, pred_pos)
                                                break
                                        if final_choice is None:
                                            for pred_dir, pred_pos, rot_key, lib_type in filtered_candidates:
                                                if lib_type == "main" and rot_key == command_rotation:
                                                    final_choice = (pred_dir, pred_pos)
                                                    break
                                        if final_choice:
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 2 - Final choice by rotation: {final_choice}")
                                            if cp_needs_filling: predicted_landing_pos_to_set = final_choice[1]
                                            if nd_needs_filling: predicted_direction_to_set = final_choice[0]
                                        # else: No specific rotation match among multiple, might pick first or none

                                    if len(filtered_candidates) == 1:  # Only one candidate after filtering by known field
                                        print(
                                            f"    SM_TRACE: Rule 2 - Core: Case 2 - Single filtered candidate: {filtered_candidates[0]}")
                                        chosen_pred_dir, chosen_pred_pos, _, _ = filtered_candidates[0]
                                        if cp_needs_filling: predicted_landing_pos_to_set = chosen_pred_pos
                                        if nd_needs_filling: predicted_direction_to_set = chosen_pred_dir

                                    # Fallback if multiple candidates remain and rotation didn't select one, or no command_rotation
                                    # And if nothing has been set yet by the logic above for the unknown field
                                    if len(filtered_candidates) > 1 and \
                                            ((cp_needs_filling and predicted_landing_pos_to_set is None) or \
                                             (nd_needs_filling and predicted_direction_to_set is None)):
                                        if not command_rotation:
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 2 - Multiple candidates ({len(filtered_candidates)}) but no command_rotation. Taking first: {filtered_candidates[0]}")
                                            chosen_pred_dir, chosen_pred_pos, _, _ = filtered_candidates[0]
                                            if cp_needs_filling: predicted_landing_pos_to_set = chosen_pred_pos
                                            if nd_needs_filling: predicted_direction_to_set = chosen_pred_dir
                                        # else: command_rotation was present but didn't lead to a unique choice from 'final_choice' logic

                            elif context_dir != "STATIONARY" and (cp_needs_filling != nd_needs_filling):

                                print(
                                    f"    SM_TRACE: Rule 2 - Core: Case 3 (Context NOT 'STATIONARY' ('{context_dir}'), one of CP/ND needs filling.)")
                                print(
                                    f"    SM_TRACE: Rule 2 - Core: Case 3 - Event CP='{current_event_cp_val}' (needs_fill={cp_needs_filling}), Event ND='{current_event_nd_val}' (needs_fill={nd_needs_filling})")

                                if context_dir in pos_rules_from_data:
                                    rule_pred_dir, rule_pred_pos = pos_rules_from_data[context_dir]
                                    print(
                                        f"    SM_TRACE: Rule 2 - Core: Case 3 - Rulebook for non-still context '{context_dir}': dir='{rule_pred_dir}', pos='{rule_pred_pos}'")

                                    if not cp_needs_filling:
                                        if current_event_cp_val == rule_pred_pos:
                                            predicted_landing_pos_to_set = current_event_cp_val
                                            predicted_direction_to_set = rule_pred_dir
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 3 - Known CP matches rulebook. Setting ND to '{rule_pred_dir}'")
                                        else:
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 3 - Known CP '{current_event_cp_val}' MISMATCHES rulebook pos '{rule_pred_pos}'. No fill for ND.")

                                    elif not nd_needs_filling:
                                        if current_event_nd_val == rule_pred_dir:
                                            predicted_direction_to_set = current_event_nd_val
                                            predicted_landing_pos_to_set = rule_pred_pos
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 3 - Known ND matches rulebook. Setting CP to '{rule_pred_pos}'")
                                        else:
                                            print(
                                                f"    SM_TRACE: Rule 2 - Core: Case 3 - Known ND '{current_event_nd_val}' MISMATCHES rulebook dir '{rule_pred_dir}'. No fill for CP.")
                                else:
                                    print(
                                        f"    SM_TRACE: Rule 2 - Core: Case 3 - Context_dir '{context_dir}' NOT found in pos_rules_from_data for '{context_pos}'.")
                        else:  # context_pos or context_dir was None/empty
                            print(
                                f"    SM_TRACE: Rule 2 - Core: SKIPPING main prediction logic because context_pos or context_dir is missing/empty.")


                        temp_filled_something_by_rule2 = False
                        if cp_needs_filling and predicted_landing_pos_to_set is not None:
                            # Check if the value in the event is actually different before updating
                            if content_resolved_by_rules.get("current_pos") != predicted_landing_pos_to_set:
                                content_resolved_by_rules["current_pos"] = predicted_landing_pos_to_set
                                temp_filled_something_by_rule2 = True

                        if nd_needs_filling and predicted_direction_to_set is not None:
                            # Check if the value in the event is actually different before updating
                            if content_resolved_by_rules.get("new_direction") != predicted_direction_to_set:
                                content_resolved_by_rules["new_direction"] = predicted_direction_to_set
                                temp_filled_something_by_rule2 = True

                        if temp_filled_something_by_rule2:
                            rule2_applied_this_iteration = True
                            made_change_in_this_pass = True
                            print(f"    Rule MOTION_CONTINUITY: "
                                  f"Context ({context_pos}, {context_dir}), Command rotation ({command_rotation}), "
                                  f"Event template CP({original_event_template.get('content', {}).get('current_pos')}), "
                                  f"Event template ND({original_event_template.get('content', {}).get('new_direction')}) -> "
                                  f"Predicted CP({content_resolved_by_rules.get('current_pos')}), "
                                  f"Predicted ND({content_resolved_by_rules.get('new_direction')})")
                            print(
                                f"    SM_TRACE: Rule 2 APPLIED. made_change_in_this_pass = {made_change_in_this_pass}")


                        print(f"    SM_DEBUG_RULE2_FILL: cp_needs_filling={cp_needs_filling}")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: predicted_landing_pos_to_set='{predicted_landing_pos_to_set}'")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: current_event_cp_val_AFTER_FILL_ATTEMPT='{content_resolved_by_rules.get('current_pos')}' (Original template val: '{original_event_template.get('content', {}).get('current_pos')}')")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: Condition1 for fill (CP based on original vs predicted): {cp_needs_filling and predicted_landing_pos_to_set is not None and original_event_template.get('content', {}).get('current_pos') != predicted_landing_pos_to_set}")
                        print(f"    SM_DEBUG_RULE2_FILL: nd_needs_filling={nd_needs_filling}")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: predicted_direction_to_set='{predicted_direction_to_set}'")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: current_event_nd_val_AFTER_FILL_ATTEMPT='{content_resolved_by_rules.get('new_direction')}' (Original template val: '{original_event_template.get('content', {}).get('new_direction')}')")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: Condition2 for fill (ND based on original vs predicted): {nd_needs_filling and predicted_direction_to_set is not None and original_event_template.get('content', {}).get('new_direction') != predicted_direction_to_set}")
                        print(
                            f"    SM_DEBUG_RULE2_FILL: filled_something_by_rule2 (based on internal flag this pass)={temp_filled_something_by_rule2}")

                    else:  # context_motion_event was None or had no content
                        print(
                            f"    SM_TRACE: Rule 2 - SKIPPING main prediction logic because context_motion_event is missing or has no content.")


            if rule2_applied_this_iteration:
                rule_id_to_add_r2 = rule2_def.get("rule_id")  # Safe get
                if rule_id_to_add_r2 and rule_id_to_add_r2 not in event_resolved_by_rules["prediction_rules_applied"]:
                    event_resolved_by_rules["prediction_rules_applied"].append(rule_id_to_add_r2)


            print(
                f"    SM_TRACE: End of rule checks in iteration {rule_iter_count + 1}. made_change_in_this_pass = {made_change_in_this_pass}")
            if not made_change_in_this_pass:
                print(f"    SM_TRACE: No changes made in this iteration, breaking inner rule loop.")
                break



        updated_library = deepcopy(abstract_identifier_library_input)
        special_command = None
        shift_command_triggered_this_event = False

        fields_to_process_for_abstract_ids = []
        if category_to_predict in ["ball_motion", "arm_motion"]:
            fields_to_process_for_abstract_ids = ["current_pos", "new_direction"]
        elif category_to_predict == "command":
            fields_to_process_for_abstract_ids = ["direction", "rotation"]

        for field_name in fields_to_process_for_abstract_ids:
            original_value_in_template_for_pass = original_event_template.get("content", {}).get(field_name)
            value_after_rules_for_field = content_resolved_by_rules.get(field_name)

            if self._is_abstract_identifier(original_value_in_template_for_pass):
                abstract_id = original_value_in_template_for_pass
                is_concrete_from_rules = (
                        value_after_rules_for_field is not None and
                        value_after_rules_for_field != UNKNOWN_PLACEHOLDER and
                        value_after_rules_for_field != "UNKNOWN_VALUE" and
                        not self._is_abstract_identifier(value_after_rules_for_field)
                )

                if is_concrete_from_rules:
                    concrete_value_from_rule = value_after_rules_for_field
                    if abstract_id not in updated_library:
                        if not shift_command_triggered_this_event:
                            updated_library[abstract_id] = concrete_value_from_rule
                            print(
                                f"    Abstract Identifier Handling (Rule Priority): New binding added '{abstract_id}' -> '{concrete_value_from_rule}' (from rule)"
                            )

                    elif updated_library[abstract_id] != concrete_value_from_rule:
                        print(
                            f"    Abstract Identifier Handling: Conflict! ID '{abstract_id}' in template (bound to '{updated_library[abstract_id]}' in library) "
                            f"was resolved by a rule to '{concrete_value_from_rule}'. Issuing SHIFT command."
                        )

                        special_command = "SHIFT_PREDICTION_SEQ"
                        shift_command_triggered_this_event = True


        if "prediction_rules_applied" in event_resolved_by_rules and \
                not event_resolved_by_rules["prediction_rules_applied"]:
            del event_resolved_by_rules["prediction_rules_applied"]


        print(
            f"    SM_TRACE: Event after all rules/abstract_id processing (before final lib bind in iterative): {event_resolved_by_rules.get('content')}")
        print(f"    SM_TRACE: Final abstract_lib (from this pass): {updated_library}")
        print(f"    SM_TRACE: Special command: {special_command}")
        print(f"    SM_TRACE: --- END _predict_event_details_single_pass ---")

        return event_resolved_by_rules, updated_library, special_command
