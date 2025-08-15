import random
import time
import numpy as np
import pybullet as p


class ActionSystem:

    def __init__(self, perception, simulator, physics_world):
        self.perception = perception
        self.simulator = simulator
        self.physics_world = physics_world

        self.angle_specific_command_map = {
            0: {+1: "RIGHT_FRONT", -1: "LEFT_FRONT"},
            90: {+1: "LEFT_FRONT", -1: "LEFT_REAR"},
            180: {+1: "LEFT_REAR", -1: "RIGHT_REAR"},
            270: {+1: "RIGHT_REAR", -1: "RIGHT_FRONT"}
        }


        self.instance_active = False
        self.current_rotation_direction = 0
        self.type2_command_count = 0
        self.collision_occurred_in_instance = False
        self.target_angles_deg = set(self.angle_specific_command_map.keys())
        self.angle_tolerance_deg = 2.0

        self.currently_detected_target = None
        self.initial_trigger_acknowledged = False

        # New variable to track overall instance number for special behavior
        self.current_overall_instance_num = 0


        self.instinctive_mode_active = False
        self.instinctive_type2_issued_count = 0
        self.instinctive_initial_rotation_direction = 0
        self.instinctive_initial_rotation_user_str = None
        print("ActionSystem initialized.")

    def _get_current_angle_deg(self):
        try:
            joint_state = p.getJointState(self.physics_world.clock, 0)
            current_angle_rad = joint_state[0]
            current_angle_deg_raw = np.rad2deg(current_angle_rad)
            current_angle_deg = current_angle_deg_raw % 360
            if current_angle_deg < 0: current_angle_deg += 360
            return current_angle_deg, current_angle_deg_raw
        except Exception as e:
            print(f"ERROR: Failed to get joint state: {e}")
            return None, None

    def execute_reflexive_action_sequence(self, preferred_rotation=None):
        if self.instance_active and not self.instinctive_mode_active:
            print("AS_Warning: Attempted to start a new reflexive action within an active episode while reflexive mode is inactive. Stopping the current episode first.")
            self.physics_world.stop_arm_rotation()
            self.instance_active = False

        print(f"AS: Starting Reflexive Action Sequence (Overall Instance #{self.current_overall_instance_num + 1})")

        self.instinctive_mode_active = True
        self.instinctive_type2_issued_count = 0
        self.instinctive_initial_rotation_user_str = None
        if preferred_rotation == "CW":
            self.instinctive_initial_rotation_direction = -1
            self.instinctive_initial_rotation_user_str = "CW"
            print(f"  AS_Instinct: CRM suggested using rotation: 'CW'. Reflexive action initial rotation set to:CW (internal -1)")
        elif preferred_rotation == "CCW":
            self.instinctive_initial_rotation_direction = 1
            self.instinctive_initial_rotation_user_str = "CCW"
            print(f"  AS_Instinct: CRM suggested using rotation: 'CCW'. Reflexive action initial rotation set to:CCW (internal +1)")
        else:
            if preferred_rotation is not None:
                print(
                    f"  AS_Instinct_Warning: Invalid preferred_rotation ('{preferred_rotation}') received from CRM. A random rotation direction will be chosen.")
            self.instinctive_initial_rotation_direction = random.choice([-1, 1])
            self.instinctive_initial_rotation_user_str = "CW" if self.instinctive_initial_rotation_direction == -1 else "CCW"
            print(
                f"  AS_Instinct: No valid CRM suggestion or no suggestion provided. Randomly selecting initial rotation direction for reflexive action: {self.instinctive_initial_rotation_user_str} (internal {self.instinctive_initial_rotation_direction})")

    def _get_closest_target_angle(self, current_angle_deg):
        if current_angle_deg is None: return None
        closest_angle = None
        min_diff = float('inf')
        for target_deg in self.target_angles_deg:
            diff = abs(current_angle_deg - target_deg)
            if diff > 180: diff = 360 - diff
            if diff < min_diff:
                min_diff = diff
                closest_angle = target_deg
        return closest_angle

    def _is_in_target_zone(self, current_angle_deg, target_deg):
        if current_angle_deg is None or target_deg is None:
            return False
        angle_diff = abs(current_angle_deg - target_deg)
        if angle_diff > 180: angle_diff = 360 - angle_diff
        return angle_diff <= self.angle_tolerance_deg

    def start_new_instance(self, current_time):
        if self.instance_active: return []

        if not hasattr(self.__class__, '_global_episode_counter'):
            self.__class__._global_episode_counter = 0
        self.__class__._global_episode_counter += 1
        self.current_overall_instance_num = self.__class__._global_episode_counter

        print(f"AS: Starting new instance (Overall Instance #{self.current_overall_instance_num}).")
        self.instance_active = True
        self.collision_occurred_in_instance = False
        self.currently_detected_target = None
        self.initial_trigger_acknowledged = False

        if self.instinctive_mode_active:
            self.current_rotation_direction = self.instinctive_initial_rotation_direction
            self.type2_command_count = 0
            print(
                f"  AS_Instinct: (Instance Start) Reflexive Mode: Using preset rotation: "
                f"{self.instinctive_initial_rotation_user_str if self.instinctive_initial_rotation_user_str else ('CW' if self.current_rotation_direction == -1 else 'CCW')} "
                f"(internal {self.current_rotation_direction})")
        else:
            if self.current_overall_instance_num <= 3:
                self.current_rotation_direction = -1
                print(
                    f"  - Special Behavior: Forcing System CCW rotation for instance #{self.current_overall_instance_num}.")
            else:
                self.current_rotation_direction = random.choice([-1, 1])
            self.type2_command_count = 0

        initial_angle_deg, initial_angle_deg_raw = self._get_current_angle_deg()
        closest_target = self._get_closest_target_angle(initial_angle_deg)

        command_direction_name = "Unknown direction"
        if closest_target is not None and closest_target in self.angle_specific_command_map:
            angle_rules = self.angle_specific_command_map[closest_target]
            if self.current_rotation_direction in angle_rules:
                command_direction_name = angle_rules[self.current_rotation_direction]

        print(f"  - Type 1 Command: Initial Angle ≈ {initial_angle_deg:.1f} deg, Closest Target = {closest_target} deg")
        system_rotation_log_str = "UNDEFINED (internal)"
        if self.current_rotation_direction == 1:
            system_rotation_log_str = "CCW (internal +1)"
        elif self.current_rotation_direction == -1:
            system_rotation_log_str = "CW (internal -1)"
        print(
            f"  - System Rotation Direction = {system_rotation_log_str}, Generated Command = {command_direction_name}")
        if self._is_in_target_zone(initial_angle_deg, closest_target):
            print(f"  - Initial position IS within tolerance of target {closest_target} deg.")
            self.currently_detected_target = closest_target
        else:
            print(f"  - Initial position IS NOT within tolerance of any target.")
            self.initial_trigger_acknowledged = True
            self.currently_detected_target = None

        command_event = self._create_command_event(command_direction_name, current_time)
        return [command_event]

    def check_and_generate_type2_command(self, current_time):
        if not self.instance_active: return []


        current_angle_deg, current_angle_deg_raw = self._get_current_angle_deg()
        if current_angle_deg is None: return []

        is_currently_in_zone = False
        current_target_in_zone = None
        for target_deg in self.target_angles_deg:
            if self._is_in_target_zone(current_angle_deg, target_deg):
                is_currently_in_zone = True
                current_target_in_zone = target_deg
                break

        trigger_type2_check = False
        if not self.initial_trigger_acknowledged:
            if not is_currently_in_zone:
                self.initial_trigger_acknowledged = True
                self.currently_detected_target = None
        else:
            if is_currently_in_zone and current_target_in_zone != self.currently_detected_target:
                print(
                    f"AS: Entered target zone for {current_target_in_zone} deg (Angle: {current_angle_deg_raw:.1f} -> Norm: {current_angle_deg:.1f}). Triggering Type 2 Check.")
                trigger_type2_check = True
                self.currently_detected_target = current_target_in_zone
            elif not is_currently_in_zone and self.currently_detected_target is not None:
                self.currently_detected_target = None

        if not trigger_type2_check:
            return []



        command_content_name = None
        action_type = None
        detected_trigger_angle = current_target_in_zone

        if self.instinctive_mode_active:

            print(f"  AS_Instinct: (Type 2 Check) Reflexive Mode: Type 2 commands issued: {self.instinctive_type2_issued_count}")
            self.instinctive_type2_issued_count += 1


            if self.instinctive_type2_issued_count <= 4:
                action_type = "Continue"
                if detected_trigger_angle in self.angle_specific_command_map:
                    angle_rules = self.angle_specific_command_map[detected_trigger_angle]
                    command_content_name = angle_rules.get(self.current_rotation_direction, "Unknown direction")
                else:
                    command_content_name = "Unknown direction"
                print(
                    f"    AS_Instinct: Type 2 Command #{self.instinctive_type2_issued_count} -> 'Continue' ({command_content_name})")


            elif self.instinctive_type2_issued_count == 5:
                action_type = "STATIONARY"
                command_content_name = "STATIONARY"
                print(f"    AS_Instinct: Type 2 Command #{self.instinctive_type2_issued_count} -> 'STATIONARY'")

            else:
                print(f"  AS_Instinct_Error: ERROR: Reflexive Type 2 count exceeded expectation. ({self.instinctive_type2_issued_count})")
                self.instinctive_mode_active = False
                return []


        else:

            self.type2_command_count += 1
            print(
                f"  AS_Normal: Type 2 Check #{self.type2_command_count} triggered for Overall Instance #{self.current_overall_instance_num}. Collision: {self.collision_occurred_in_instance}")

            apply_original_logic = True
            if self.current_overall_instance_num == 1 and self.type2_command_count == 1:
                action_type, command_content_name = "STATIONARY", "STATIONARY"
                apply_original_logic = False
            elif self.current_overall_instance_num == 2 and self.type2_command_count == 1:
                action_type, command_content_name = "STATIONARY", "STATIONARY"
                apply_original_logic = False
            elif self.current_overall_instance_num == 3:
                if self.type2_command_count == 1:
                    action_type = "Continue"
                    if detected_trigger_angle in self.angle_specific_command_map:
                        angle_rules = self.angle_specific_command_map[detected_trigger_angle]
                        command_content_name = angle_rules.get(self.current_rotation_direction, "Unknown direction")
                    else:
                        command_content_name = "Unknown direction"
                    apply_original_logic = False
                elif self.type2_command_count == 2:
                    action_type, command_content_name = "STATIONARY", "STATIONARY"
                    apply_original_logic = False

            if apply_original_logic:
                if self.type2_command_count >= 4:
                    action_type, command_content_name = "STATIONARY", "STATIONARY"
                elif not self.collision_occurred_in_instance:
                    action_type = "Continue"
                    if detected_trigger_angle in self.angle_specific_command_map:
                        angle_rules = self.angle_specific_command_map[detected_trigger_angle]
                        command_content_name = angle_rules.get(self.current_rotation_direction, "Unknown direction")
                    else:
                        command_content_name = "Unknown direction"
                else:
                    decision = random.choice(["Continue", "STATIONARY"])
                    action_type = decision
                    if decision == "STATIONARY":
                        command_content_name = "STATIONARY"
                    else:
                        if detected_trigger_angle in self.angle_specific_command_map:
                            angle_rules = self.angle_specific_command_map[detected_trigger_angle]
                            command_content_name = angle_rules.get(self.current_rotation_direction, "Unknown direction")
                        else:
                            command_content_name = "Unknown direction"

        if command_content_name:
            print(f"  AS: Action: {action_type}, Generated Command Content: {command_content_name}")
            if action_type == "STATIONARY":
                self.physics_world.stop_arm_rotation()
                self.instance_active = False
                print("  AS: Instance marked as inactive due to 'STATIONARY' command.")


                if self.instinctive_mode_active and self.instinctive_type2_issued_count == 5:
                    self.instinctive_mode_active = False
                    print("  AS_Instinct: Reflexive mode deactivated.")

            command_event = self._create_command_event(command_content_name, current_time)
            return [command_event]
        else:
            print("AS_Warn: No command content determined for Type 2 check.")
            return []

    def update_collision_status(self, has_collided):
        if self.instance_active and has_collided:
            if not self.collision_occurred_in_instance:
                print(
                    f"ActionSystem: Collision detected for the first time in this instance (Overall #{self.current_overall_instance_num}).")
            self.collision_occurred_in_instance = True

    def _create_command_event(self, direction_name, event_time):
        print(f"AS_DEBUG _create_command_event: instinctive_mode={self.instinctive_mode_active}, "
              f"instinctive_user_str='{self.instinctive_initial_rotation_user_str}', "
              f"current_rotation_dir={self.current_rotation_direction}")
        rotation_representation_for_user = "UNDEFINED"
        if self.instinctive_mode_active and self.instinctive_initial_rotation_user_str:
            rotation_representation_for_user = self.instinctive_initial_rotation_user_str
        elif self.current_rotation_direction == 1:
            rotation_representation_for_user = "CCW"
        elif self.current_rotation_direction == -1:
            rotation_representation_for_user = "CW"
        return {
            "category": "command",
            "start_time": event_time,
            "end_time": event_time,
            "content": {
                "direction": direction_name,
                "rotation": rotation_representation_for_user,
                "original_command": f"auto_{direction_name}"
            }
        }

    def is_instance_active(self):
        return self.instance_active

    @classmethod
    def reset_overall_instance_counter(cls):
        cls._global_episode_counter = 0
        print("ActionSystem: Overall instance counter reset.")