# --- START OF FILE perception (12).py ---

import copy
import pybullet as p

FLOAT_EPSILON = 1e-9  # Adjusted for comparing simulation time steps


class Perception:
    def __init__(self, sensor):

        self.sensor = sensor
        self.events = []
        self.current_sim_time = 0.0
        self.current_seq_num = 0  # Current sequence number to assign
        self.last_event_sim_time = -1.0  # Simulation time of the last processed event batch
        # Initialized to -1 to ensure first events get seq 1

        self.time_step = 1 / 60  # Default, updated by process method

    def clear_state(self):

        print("DEBUG: Perception.clear_state() - Resetting event list, time, and sequence number.")
        self.events = []
        self.current_sim_time = 0.0
        self.current_seq_num = 0
        self.last_event_sim_time = -1.0

    def _get_initial_arm_pos(self):

        try:
            # Assuming sensor has clock_id and arm_link_idx readily available
            # and _get_angle, _categorize_position methods
            arm_state = p.getLinkState(self.sensor.clock_id, self.sensor.arm_link_idx)
            arm_pos = arm_state[0]  # worldPosition of the link
            angle = self.sensor._get_angle(arm_pos)  # Assuming sensor has _get_angle
            return self.sensor._categorize_position(angle)  # Assuming sensor has _categorize_position
        except Exception as e:
            print(f"WARN: Error while getting initial arm position: {e}")
            return "UNKNOWN"

    def process(self, incoming_events, time_step=1 / 60):

        self.time_step = time_step
        self.current_sim_time += time_step  # Increment simulation time based on external step


        if not self.events and self.current_seq_num == 0:
            # print("DEBUG: Perception processing initial events.") # Less verbose
            self.current_seq_num += 1  # Start with seq 1 for initial events

            sensor_data = self.sensor.get_data()  # Get fresh sensor data
            initial_perception_events = []
            if sensor_data is None:
                print("WARN: Sensor data unavailable. Using default initial event for the ball.")
                initial_perception_events.append({
                    "seq": self.current_seq_num,
                    "category": "ball_motion",
                    "content": {"type": "ball_direction_change", "new_direction": "UNKNOWN_VALUE", "current_pos": "UNKNOWN_VALUE"}
                })
            else:
                ball_motion = sensor_data.get("ball_motion", {})
                initial_perception_events.append({
                    "seq": self.current_seq_num,
                    "category": "ball_motion",
                    "content": {
                        "type": "ball_direction_change",
                        "new_direction": ball_motion.get("direction", "UNKNOWN_VALUE"),
                        "current_pos": ball_motion.get("prev_pos", "UNKNOWN_VALUE"),  # Sensor provides 'prev_pos'
                        "tags": copy.deepcopy(ball_motion.get("tags", []))
                    }
                })

            initial_perception_events.append({
                "seq": self.current_seq_num,
                "category": "arm_motion",
                "content": {"type": "arm_direction_change", "new_direction": "STATIONARY",
                            "current_pos": self._get_initial_arm_pos()}
            })

            for event in initial_perception_events:
                # Ensure 'current_pos' from 'prev_pos' if applicable (already done for ball_motion)
                if isinstance(event["content"], dict) and "prev_pos" in event["content"]:
                    event["content"]["current_pos"] = event["content"].pop("prev_pos")
                self.events.append(event)

            self.last_event_sim_time = self.current_sim_time  # Mark time of these initial events


        processed_this_step = False
        if incoming_events:
            # Determine if this batch of incoming_events constitutes a new sequence number
            # A new sequence number is assigned if current_sim_time has advanced
            # beyond last_event_sim_time by more than a very small epsilon.
            if self.current_sim_time > self.last_event_sim_time + FLOAT_EPSILON:
                self.current_seq_num += 1
                self.last_event_sim_time = self.current_sim_time

            current_batch_seq = self.current_seq_num  # All events in this batch get this seq

            for raw_event in incoming_events:
                if not raw_event: continue  # Skip if None or empty

                event_data = copy.deepcopy(raw_event)
                category = event_data.get("category")

                # Infer category if not present (copied from old logic)
                if not category:
                    event_type = event_data.get("type")
                    if event_type:
                        category = self._get_category(event_type)
                    elif isinstance(event_data.get("content"), dict) and event_data["content"].get("original_command"):
                        category = "command"
                    else:
                        category = "unknown"  # Or skip/log unknown events

                if category == "unknown":
                    # print(f"WARN: Perception skipping unknown event: {event_data}")
                    continue

                content = event_data.get("content", event_data)  # Use content if present, else full data
                # Standardize 'prev_pos' to 'current_pos'
                if isinstance(content, dict) and "prev_pos" in content:
                    content["current_pos"] = content.pop("prev_pos")

                # Create the sequenced event
                sequenced_event = {
                    "seq": current_batch_seq,
                    "category": category,
                    "content": content
                }



                self.events.append(sequenced_event)  # Simpler: add all, let downstream handle interpretation
                processed_this_step = True

        # --- Null event generation (if nothing else happened) ---
        # This logic might need adjustment. If incoming_events is empty, it implies no *new*
        # significant events. If the *last* event added was from a previous sim_time,
        # and this current_sim_time is new, then a null could indicate passage of time.
        if not processed_this_step and self.events:  # Only add null if events list isn't empty
            if self.current_sim_time > self.last_event_sim_time + FLOAT_EPSILON:
                pass

    def _get_category(self, event_type):
        mapping = {
            "ball_speed_change": "ball_speed",
            "ball_direction_change": "ball_motion",
            "arm_speed_change": "arm_speed",
            "arm_direction_change": "arm_motion",
            "collision_change": "collision",
            # Add other type -> category mappings if EventSegmenter produces more types
        }
        return mapping.get(event_type, "unknown")  # Default to unknown

    # _add_or_merge_event and _can_merge are removed as merging is now handled by seq num implicitly
    # or by preventing duplicates within the same seq num (if that logic is enabled).

    def get_events(self):
        return self.events  # Returns the list of sequenced events directly

    def format_events(self):
        return copy.deepcopy(self.events)  # Return a copy of the already processed list

# --- END OF FILE perception (12).py ---