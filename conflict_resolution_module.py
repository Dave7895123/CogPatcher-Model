import copy


UNKNOWN_PLACEHOLDER = "ANY_PLACEHOLDER"
OCCLUSION_TAG = "IS_OCCLUDED"
CMD_ARM_CONSISTENCY_RULE_ID = "CMD_ARM_CONSISTENCY"
MOTION_CONTINUITY_RULE_ID = "MOTION_CONTINUITY"


class ConflictResolutionModule:
    def __init__(self, prediction_module, perception_module, action_system, schematic_memory_module):

        self.prediction_module = prediction_module
        self.perception_module = perception_module
        self.action_system = action_system
        self.schematic_memory = schematic_memory_module

        self.special_comparison_rules = []

        self.learned_uncertainty_mappings = []

        self.pending_reflexive_action_data = None

        print("ConflictResolutionModule initialized.")

    def _match_event_to_pattern(self, event, pattern):

        if not event or not pattern:
            return False
        if event.get("category") != pattern.get("category"):
            return False

        event_content = event.get("content", {})
        pattern_content = pattern.get("content", {})

        for key, pattern_value in pattern_content.items():
            event_value = event_content.get(key)
            if key == "tags" and isinstance(pattern_value, list):
                if not isinstance(event_value, list) or not all(tag in event_value for tag in pattern_value):
                    return False
            elif event_value != pattern_value:

                if pattern_value != UNKNOWN_PLACEHOLDER and pattern_value != "UNKNOWN_VALUE":
                    return False
        return True

    def _evaluate_event_pair(self, predicted_event, actual_event):

        for rule in self.special_comparison_rules:
            if self._match_event_to_pattern(predicted_event, rule["trigger_predicted_event_pattern"]) and \
                    self._match_event_to_pattern(actual_event, rule["trigger_actual_event_pattern"]):
                print(
                    f"  CRM_Eval: Match by special rule.Pred: {predicted_event.get('content')}, Actual: {actual_event.get('content')}")

                return {"match": True, "used_special_rule": True}


        if predicted_event.get("category") != actual_event.get("category"):
            print(
                f"  CRM_Eval: Category mismatch. Pred: {predicted_event.get('category')}, Actual: {actual_event.get('category')}")

            return {"match": False, "used_special_rule": False}

        pred_content = predicted_event.get("content", {})
        actual_content = actual_event.get("content", {})
        category = predicted_event.get("category")


        if category == "command":
            if pred_content.get("direction") != actual_content.get("direction") or \
                    pred_content.get("rotation") != actual_content.get("rotation"):
                print(
                    f"  CRM_Eval: Command content mismatch. Pred: Dir={pred_content.get('direction')},Rot={pred_content.get('rotation')}; "
                    f"Actual: Dir={actual_content.get('direction')},Rot={actual_content.get('rotation')}")
                return {"match": False, "used_special_rule": False}
        elif category in ["ball_motion", "arm_motion"]:
            fields_to_compare = ["current_pos", "new_direction"]
            for field in fields_to_compare:
                pred_val = pred_content.get(field)
                actual_val = actual_content.get(field)
                if pred_val not in [UNKNOWN_PLACEHOLDER, "UNKNOWN_VALUE"] and pred_val != actual_val:
                    print(f"  CRM_Eval: Field '{field}' in category '{category}' does not match. Pred: {pred_val}, Actual: {actual_val}")
                    return {"match": False, "used_special_rule": False}
        else:
            if pred_content != actual_content:
                print(f"  CRM_Eval: '{category}' content mismatch. Pred: {pred_content}, Actual: {actual_content}")
                return {"match": False, "used_special_rule": False}


        return {"match": True, "used_special_rule": False}

    def compare_sequences_at_instance_end(self, predicted_sequence, actual_sequence, instance_number=None):

        print("\nCRM: Starting sequential comparison of predicted vs. actual sequences at the end of the episode...")

        if not predicted_sequence and not actual_sequence:
            print("  CRM: Both predicted and actual sequences are empty. Considered a match.")
            return {"status": "ok", "reason": "both_sequences_empty"}


        if len(predicted_sequence) != len(actual_sequence):
            print(
                f"  CRM_Error: Sequence length mismatch: Predicted ({len(predicted_sequence)}) vs. Actual ({len(actual_sequence)}). Sequential comparison failed.")

            first_pred_at_diff = None
            first_actual_at_diff = None
            min_len = min(len(predicted_sequence), len(actual_sequence))


            if len(predicted_sequence) > min_len:
                first_pred_at_diff = predicted_sequence[min_len]
            if len(actual_sequence) > min_len:
                first_actual_at_diff = actual_sequence[min_len]


            return {"status": "error",
                    "reason": "sequence_length_mismatch_for_sequential_comparison",
                    "predicted_at_diff": first_pred_at_diff,
                    "actual_at_diff": first_actual_at_diff,
                    "details": f"Pred len: {len(predicted_sequence)}, Actual len: {len(actual_sequence)}"}


        conflicting_pair = None
        conflicting_idx = -1

        special_rule_was_used_in_sequence = False

        for i in range(len(predicted_sequence)):
            predicted_event = predicted_sequence[i]
            actual_event = actual_sequence[i]


            evaluation_result = self._evaluate_event_pair(predicted_event, actual_event)

            if evaluation_result.get("used_special_rule"):
                special_rule_was_used_in_sequence = True


            if not evaluation_result.get("match"):

                print(f"  CRM: Mismatch found at index {i}:")
                print(
                    f"    Predicted (CI_Seq: {predicted_event.get('current_instance_seq')}): {predicted_event.get('category')} - {predicted_event.get('content')}")
                print(
                    f"    Actual    (Seq: {actual_event.get('seq')}): {actual_event.get('category')} - {actual_event.get('content')}")
                conflicting_pair = (predicted_event, actual_event)
                conflicting_idx = i
                break

        if conflicting_pair is None:
            print("  CRM: Sequential comparison passed. No conflicts found.")

            return {
                "status": "ok",
                "used_special_rule_overall": special_rule_was_used_in_sequence
            }
        else:

            conflicting_predicted_event, conflicting_actual_event = conflicting_pair
            actual_content = conflicting_actual_event.get("content", {})
            actual_tags = actual_content.get("tags", [])

            if OCCLUSION_TAG not in actual_tags:
                print(f"  CRM: Conflict at index {conflicting_idx}, but the actual event is not tagged with '{OCCLUSION_TAG}'.")
                return {
                    "status": "error",
                    "reason": "unhandled_conflict_no_occlusion_tag",
                    "predicted": conflicting_predicted_event,
                    "actual": conflicting_actual_event,
                    "index_of_conflict": conflicting_idx
                }
            else:
                print(f"  CRM: Conflict at index {conflicting_idx}, and the actual event is tagged with '{OCCLUSION_TAG}'.")
                pred_rules = conflicting_predicted_event.get("prediction_rules_applied", [])

                has_motion_continuity = any(
                    (isinstance(rule_id, str) and MOTION_CONTINUITY_RULE_ID in rule_id) or
                    (not isinstance(rule_id, str) and MOTION_CONTINUITY_RULE_ID == rule_id)
                    for rule_id in pred_rules
                )

                if not has_motion_continuity:
                    print(f"  CRM: Predicted event did not use '{MOTION_CONTINUITY_RULE_ID}' rule. Applied rules: {pred_rules}")
                    return {
                        "status": "error",
                        "reason": "conflict_with_occlusion_weak_rule",
                        "predicted": conflicting_predicted_event,
                        "actual": conflicting_actual_event,
                        "index_of_conflict": conflicting_idx,
                        "details": f"Applied rules: {pred_rules}"
                    }
                else:
                    print(f"  CRM: Predicted event used '{MOTION_CONTINUITY_RULE_ID}' rule. Triggering reflexive action...")
                    self.pending_reflexive_action_data = {
                        "original_conflicting_predicted": copy.deepcopy(conflicting_predicted_event),
                        "original_conflicting_actual_occluded": copy.deepcopy(conflicting_actual_event),
                        "triggering_instance_number": instance_number
                    }

                    last_command_rotation_from_conflict_instance = None
                    last_command_event_in_conflicting_instance = None
                    for event in reversed(predicted_sequence):  # Search the whole predicted sequence
                        if event.get("category") == "command":
                            last_command_event_in_conflicting_instance = event
                            break

                    if last_command_event_in_conflicting_instance:
                        content = last_command_event_in_conflicting_instance.get("content", {})
                        last_command_rotation_from_conflict_instance = content.get("rotation")  # This is "CW" or "CCW"
                        print(
                            f"  CRM: Extracted rotation from the last command in the conflicting episode: {last_command_rotation_from_conflict_instance}")
                    else:
                        print(f"  CRM_Warning: Could not find any command event in the conflicting episode's predicted sequence to extract rotation.")
                    # --- END MODIFICATION ---

                    try:
                        # --- MODIFICATION: Pass rotation to AS ---
                        self.action_system.execute_reflexive_action_sequence(
                            preferred_rotation=last_command_rotation_from_conflict_instance
                            # Pass the string "CW"/"CCW" or None
                        )
                        return {"status": "reflexive_action_triggered"}
                    except AttributeError:
                        print("  CRM_Error: ActionSystem is missing the 'execute_reflexive_action_sequence' method!")
                        pass
                    except TypeError as te:  # Catch if execute_reflexive_action_sequence signature changed
                        print(f"  CRM_Error: Argument mismatch when calling ActionSystem.execute_reflexive_action_sequence: {te}")
                        # Fallback or re-raise, for now, let's just print and proceed without triggering
                        self.pending_reflexive_action_data = None
                        return {
                            "status": "error",
                            "reason": "ach_missing_method_for_reflexive_action",
                            "predicted": conflicting_predicted_event,
                            "actual": conflicting_actual_event
                        }

    def process_reflexive_action_outcome(self, instinctive_instance_had_conflict):

        if not self.pending_reflexive_action_data:
            print("CRM_Error: 'process_reflexive_action_outcome' was called, but there is no pending reflexive action data.")
            return {"status": "error", "reason": "no_pending_reflexive_action"}

        original_data = self.pending_reflexive_action_data
        self.pending_reflexive_action_data = None  # 清理


        triggering_instance_num = original_data.get("triggering_instance_number")

        if instinctive_instance_had_conflict:
            print(
                f"CRM: Reflexive action episode failed to resolve the conflict. Original prediction: {original_data['original_conflicting_predicted']['content']}, "
                f"Original actual: {original_data['original_conflicting_actual_occluded']['content']}")
            return {
                "status": "error",
                "reason": "reflexive_action_failed",
                "original_predicted": original_data['original_conflicting_predicted'],
                "original_actual_occluded": original_data['original_conflicting_actual_occluded'],
                "triggering_instance_number": triggering_instance_num  # <--- 在返回字典中加入
            }
        else:
            print(f"CRM: Reflexive action episode was successful (no conflict). Learning new special comparison rule and uncertainty mapping.")

            new_special_rule = {
                "trigger_predicted_event_pattern": copy.deepcopy(original_data['original_conflicting_predicted']),
                "trigger_actual_event_pattern": copy.deepcopy(original_data['original_conflicting_actual_occluded'])
            }
            self.special_comparison_rules.append(new_special_rule)
            print(
                f"  CRM: New special comparison rule added: PredPattern={new_special_rule['trigger_predicted_event_pattern']['content']}, "
                f"ActualPattern={new_special_rule['trigger_actual_event_pattern']['content']}")


            actual_pattern_content = {}
            original_actual_content = original_data['original_conflicting_actual_occluded'].get('content', {})
            for key, value in original_actual_content.items():

                actual_pattern_content[key] = value
            if "tags" not in actual_pattern_content or OCCLUSION_TAG not in actual_pattern_content["tags"]:
                actual_pattern_content["tags"] = actual_pattern_content.get("tags", []) + [OCCLUSION_TAG]

            new_mapping = {
                "occluded_actual_pattern": {
                    "category": original_data['original_conflicting_actual_occluded'].get("category"),
                    "content": actual_pattern_content
                },
                "resolved_predicted_content": copy.deepcopy(
                    original_data['original_conflicting_predicted'].get("content", {}))
            }
            self.learned_uncertainty_mappings.append(new_mapping)
            print(f"  CRM: New uncertainty mapping added: ActualPattern={new_mapping['occluded_actual_pattern']['content']}, "
                  f"ResolvedContent={new_mapping['resolved_predicted_content']}")

            return {
                "status": "ok",
                "reason": "reflexive_action_learned_new_rule",
                "learned_rule": new_special_rule,
                "learned_mapping": new_mapping,
                "triggering_instance_number": triggering_instance_num
            }

    def _matches_occluded_pattern_for_am(self, event_from_am, pattern_data):

        if event_from_am.get("category") != pattern_data["category"]:
            return False

        event_content = event_from_am.get("content", {})
        pattern_content = pattern_data["content"]

        if OCCLUSION_TAG not in event_content.get("tags", []):
            return False


        for key, pattern_value in pattern_content.items():
            if key == "tags": continue  # 标签已检查

            event_value = event_content.get(key)
            if pattern_value not in [UNKNOWN_PLACEHOLDER, "UNKNOWN_VALUE"]:  # 模式中是具体值
                if event_value != pattern_value:
                    return False


        return True

    def resolve_uncertainty_for_am(self, event_from_am_with_unknowns):


        event_content = event_from_am_with_unknowns.get("content", {})
        if OCCLUSION_TAG not in event_content.get("tags", []):

            return event_from_am_with_unknowns


        has_unknown = False
        for field in ["current_pos", "new_direction"]:
            if event_content.get(field) in [UNKNOWN_PLACEHOLDER, "UNKNOWN_VALUE"]:
                has_unknown = True
                break
        if not has_unknown:

            return event_from_am_with_unknowns

        for mapping in self.learned_uncertainty_mappings:
            if self._matches_occluded_pattern_for_am(event_from_am_with_unknowns, mapping["occluded_actual_pattern"]):
                print(f"  CRM: Matching uncertainty map found. Resolving event.")
                resolved_event = copy.deepcopy(event_from_am_with_unknowns)
                resolved_content_from_mapping = mapping["resolved_predicted_content"]


                current_resolved_event_content = resolved_event.setdefault("content", {})


                for key, resolved_value in resolved_content_from_mapping.items():

                    if current_resolved_event_content.get(key) in [UNKNOWN_PLACEHOLDER, "UNKNOWN_VALUE"]:
                        current_resolved_event_content[key] = resolved_value
                        print(f"    CRM: Field '{key}' resolved from '{UNKNOWN_PLACEHOLDER}' to '{resolved_value}'.")


                print(f"  CRM: Resolved event content: {resolved_event['content']}")
                return resolved_event


        return event_from_am_with_unknowns

    def resolve_uncertainty_in_initial_event(self, initial_event_from_pm):

        if initial_event_from_pm.get('is_skeleton_event', True):
            return initial_event_from_pm
        return self.resolve_uncertainty_for_am(initial_event_from_pm)