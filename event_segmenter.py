# --- START OF FILE event_segmenter (9)_modified_for_occlusion_pos.py ---

class EventSegmenter:
    def __init__(self):

        self.prev_ball_speed = None
        self.prev_ball_direction = None
        self.prev_ball_pos_category = None
        self.prev_ball_tags = []

        self.prev_arm_speed = None
        self.prev_arm_direction = None
        self.prev_arm_pos_category = None

        self.prev_collision = None


        self.speed_threshold = 0.005


        self.last_ball_speed_event_change_type = None
        self.last_arm_speed_event_change_type = None

    def process(self, sensor_data):

        events = []
        if sensor_data is None:
            print("WARN: EventSegmenter received None sensor_data.")
            return events


        current_ball_speed_data = sensor_data.get("ball_speed", {})
        current_ball_motion_data = sensor_data.get("ball_motion", {})
        current_arm_speed_data = sensor_data.get("arm_speed", {})
        current_arm_motion_data = sensor_data.get("arm_motion", {})
        current_collision = sensor_data.get("collision", False)

        current_ball_speed_value = current_ball_speed_data.get("value", "UNKNOWN_VALUE")
        ball_speed_change_from_sensor = current_ball_speed_data.get("change")

        current_ball_direction = current_ball_motion_data.get("direction", "UNKNOWN_VALUE")
        current_ball_prev_pos_from_sensor = current_ball_motion_data.get("prev_pos", "UNKNOWN_VALUE")
        current_ball_tags = current_ball_motion_data.get("tags", [])

        current_arm_speed_value = current_arm_speed_data.get("value", "UNKNOWN_VALUE")
        arm_speed_change_from_sensor = current_arm_speed_data.get("change")

        current_arm_direction = current_arm_motion_data.get("direction", "UNKNOWN_VALUE")
        current_arm_prev_pos_from_sensor = current_arm_motion_data.get("prev_pos", "UNKNOWN_VALUE")


        if self.prev_collision is not None and current_collision != self.prev_collision:
            events.append({
                "type": "collision_change",
                "collision": current_collision
            })


        determined_ball_speed_change = None
        if ball_speed_change_from_sensor == "INCREASE" or ball_speed_change_from_sensor == "DECREASE":
            determined_ball_speed_change = ball_speed_change_from_sensor
        elif isinstance(current_ball_speed_value, (int, float)) and \
                isinstance(self.prev_ball_speed, (int, float)):
            speed_diff = current_ball_speed_value - self.prev_ball_speed
            if abs(speed_diff) > self.speed_threshold:
                determined_ball_speed_change = "INCREASE" if speed_diff > 0 else "DECREASE"

        if determined_ball_speed_change:
            if determined_ball_speed_change != self.last_ball_speed_event_change_type:
                events.append({"type": "ball_speed_change", "change": determined_ball_speed_change})
                self.last_ball_speed_event_change_type = determined_ball_speed_change


        current_ball_pos_for_event = self._get_pos_category(current_ball_prev_pos_from_sensor)

        if self.prev_ball_direction is not None:
            if self.prev_ball_direction != "UNKNOWN_VALUE" and current_ball_direction == "UNKNOWN_VALUE":
                events.append({
                    "type": "ball_direction_change", "new_direction": "UNKNOWN_VALUE",
                    "prev_pos": current_ball_pos_for_event, "tags": current_ball_tags
                })
            elif self.prev_ball_direction == "UNKNOWN_VALUE" and current_ball_direction != "UNKNOWN_VALUE":
                events.append({
                    "type": "ball_direction_change", "new_direction": current_ball_direction,
                    "prev_pos": current_ball_pos_for_event, "tags": current_ball_tags
                })
            elif (self.prev_ball_direction != "UNKNOWN_VALUE" and
                  current_ball_direction != "UNKNOWN_VALUE" and
                  current_ball_direction != self.prev_ball_direction):
                events.append({
                    "type": "ball_direction_change", "new_direction": current_ball_direction,
                    "prev_pos": current_ball_pos_for_event, "tags": current_ball_tags
                })


        determined_arm_speed_change = None
        if arm_speed_change_from_sensor == "INCREASE" or arm_speed_change_from_sensor == "DECREASE":
            determined_arm_speed_change = arm_speed_change_from_sensor
        elif isinstance(current_arm_speed_value, (int, float)) and \
                isinstance(self.prev_arm_speed, (int, float)):
            speed_diff = current_arm_speed_value - self.prev_arm_speed
            if abs(speed_diff) > self.speed_threshold:
                determined_arm_speed_change = "INCREASE" if speed_diff > 0 else "DECREASE"

        if determined_arm_speed_change:
            if determined_arm_speed_change != self.last_arm_speed_event_change_type:
                events.append({"type": "arm_speed_change", "change": determined_arm_speed_change})
                self.last_arm_speed_event_change_type = determined_arm_speed_change


        current_arm_pos_for_event = self._get_pos_category(current_arm_prev_pos_from_sensor)

        if self.prev_arm_direction is not None:
            if (self.prev_arm_direction != "UNKNOWN_VALUE" and
                    current_arm_direction != "UNKNOWN_VALUE" and
                    current_arm_direction != self.prev_arm_direction):
                events.append({
                    "type": "arm_direction_change", "new_direction": current_arm_direction,
                    "prev_pos": current_arm_pos_for_event
                })
            elif self.prev_arm_direction != "UNKNOWN_VALUE" and current_arm_direction == "UNKNOWN_VALUE":
                events.append({
                    "type": "arm_direction_change", "new_direction": "UNKNOWN_VALUE",
                    "prev_pos": current_arm_pos_for_event
                })
            elif self.prev_arm_direction == "UNKNOWN_VALUE" and current_arm_direction != "UNKNOWN_VALUE":
                events.append({
                    "type": "arm_direction_change", "new_direction": current_arm_direction,
                    "prev_pos": current_arm_pos_for_event
                })


        if isinstance(current_ball_speed_value, (int, float)):
            self.prev_ball_speed = current_ball_speed_value
        self.prev_ball_direction = current_ball_direction
        self.prev_ball_pos_category = current_ball_prev_pos_from_sensor
        self.prev_ball_tags = list(current_ball_tags)
        if isinstance(current_arm_speed_value, (int, float)):
            self.prev_arm_speed = current_arm_speed_value
        self.prev_arm_direction = current_arm_direction
        self.prev_arm_pos_category = current_arm_prev_pos_from_sensor
        self.prev_collision = current_collision

        return events

    def _get_pos_category(self, pos_from_sensor):
        return pos_from_sensor

# --- END OF FILE ---