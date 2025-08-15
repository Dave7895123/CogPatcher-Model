
import pybullet as p
import numpy as np


# 不再需要 import time

class Sensor:

    DIRECTION_CONFIRMATION_STEPS = 20
    SPEED_CHANGE_THRESHOLD = 5

    OCCLUSION_A_START_STEP = 40000
    OCCLUSION_A_END_STEP = 70000
    OCCLUSION_A_START_ANGLE = 110
    OCCLUSION_A_END_ANGLE = 160

    OCCLUSION_B_START_STEP = 70000
    OCCLUSION_B_START_ANGLE = 200
    OCCLUSION_B_END_ANGLE = 250

    COLLISION_DURATION_THRESHOLD_STEPS = 19

    def __init__(self, physics_world):
        self.world = physics_world
        self.ball_id = self.world.get_handle("ball")
        self.clock_id = self.world.get_handle("clock")
        self.arm_joint_idx = 0
        self.arm_link_idx = 0
        self.arm_tip_offset_from_center = 0.5


        self.confirmed_ball_direction = "STATIONARY"
        self.confirmed_ball_prev_pos = None
        self.potential_ball_direction = "STATIONARY"
        self.ball_direction_streak = 0
        self.ball_pos_at_potential_start = None
        self.confirmed_arm_direction = "STATIONARY"
        self.confirmed_arm_prev_pos = None
        self.potential_arm_direction = "STATIONARY"
        self.arm_direction_streak = 0
        self.arm_pos_at_potential_start = None


        self.last_reported_ball_speed = 0.0
        self.last_reported_arm_speed = 0.0


        self.simulation_steps = 0


        self.in_collision_with_arm = False
        self.collision_with_arm_start_step = None  # 从 time 改为 step

        self._initialize_state()

    def _initialize_state(self):

        try:
            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_angle = self._get_angle(ball_pos)
            self.confirmed_ball_prev_pos = self._categorize_position(ball_angle)
            self.ball_pos_at_potential_start = self.confirmed_ball_prev_pos
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            self.last_reported_ball_speed = np.linalg.norm(ball_vel)
            joint_state = p.getJointState(self.clock_id, self.arm_joint_idx)
            joint_angle_rad = joint_state[0]
            joint_angular_velocity = joint_state[1]
            arm_tip_x = self.arm_tip_offset_from_center * np.cos(joint_angle_rad)
            arm_tip_y = self.arm_tip_offset_from_center * np.sin(joint_angle_rad)
            base_pos, _ = p.getBasePositionAndOrientation(self.clock_id)
            arm_tip_z = base_pos[2]
            arm_tip_pos = [arm_tip_x, arm_tip_y, arm_tip_z]
            arm_angle = self._get_angle(arm_tip_pos)
            self.confirmed_arm_prev_pos = self._categorize_position(arm_angle)
            self.arm_pos_at_potential_start = self.confirmed_arm_prev_pos
            arm_tip_linear_speed = abs(joint_angular_velocity * self.arm_tip_offset_from_center)
            self.last_reported_arm_speed = arm_tip_linear_speed
        except p.error as e:
            print(f"WARN: Error during sensor state initialization: {e}")
            self.confirmed_ball_prev_pos = "UNKNOWN"
            self.ball_pos_at_potential_start = "UNKNOWN"
            self.confirmed_arm_prev_pos = "UNKNOWN"
            self.arm_pos_at_potential_start = "UNKNOWN"
            self.last_reported_ball_speed = 0.0
            self.last_reported_arm_speed = 0.0


    def _get_angle(self, position):
        x, y = position[0], position[1]
        angle = np.degrees(np.arctan2(y, x))
        return angle % 360

    def _categorize_position(self, angle):
        if 0 <= angle < 90:
            return "RIGHT_REAR QUADRANT"
        elif 90 <= angle < 180:
            return "RIGHT_FRONT QUADRANT"
        elif 180 <= angle < 270:
            return "LEFT_FRONT QUADRANT"
        else:
            return "LEFT_REAR QUADRANT"

    def _categorize_direction(self, velocity_vector):
        speed_threshold = 0.001
        speed = np.linalg.norm(velocity_vector)
        if speed < speed_threshold: return "STATIONARY"
        x, y = velocity_vector[0], velocity_vector[1]
        angle = np.degrees(np.arctan2(y, x)) % 360
        if 0 <= angle < 90:
            return "RIGHT_REAR"
        elif 90 <= angle < 180:
            return "RIGHT_FRONT"
        elif 180 <= angle < 270:
            return "LEFT_FRONT"
        else:
            return "LEFT_REAR"


    def _is_in_occlusion_zone(self, angle):

        if self.OCCLUSION_A_START_STEP <= self.simulation_steps < self.OCCLUSION_A_END_STEP:

            if self.OCCLUSION_A_START_ANGLE <= angle < self.OCCLUSION_A_END_ANGLE:
                return True


        elif self.simulation_steps >= self.OCCLUSION_B_START_STEP:

            if self.OCCLUSION_B_START_ANGLE <= angle < self.OCCLUSION_B_END_ANGLE:
                return True


        return False

    def _get_collision(self):
        try:
            contact_points = p.getContactPoints(bodyA=self.ball_id, bodyB=self.clock_id, linkIndexB=self.arm_link_idx)
            return len(contact_points) > 0
        except p.error:
            return False

    def get_data(self):
        self.simulation_steps += 1

        try:

            ball_pos, _ = p.getBasePositionAndOrientation(self.ball_id)
            ball_vel, _ = p.getBaseVelocity(self.ball_id)
            current_ball_speed = np.linalg.norm(ball_vel)
            joint_state = p.getJointState(self.clock_id, self.arm_joint_idx)
            joint_angle_rad, joint_angular_velocity = joint_state[0], joint_state[1]
            arm_tip_x = self.arm_tip_offset_from_center * np.cos(joint_angle_rad)
            arm_tip_y = self.arm_tip_offset_from_center * np.sin(joint_angle_rad)
            base_pos, _ = p.getBasePositionAndOrientation(self.clock_id)
            arm_tip_z = base_pos[2]
            arm_tip_pos = [arm_tip_x, arm_tip_y, arm_tip_z]
            arm_tip_vel = [-joint_angular_velocity * arm_tip_y, joint_angular_velocity * arm_tip_x, 0]
            current_arm_speed = np.linalg.norm(arm_tip_vel)
            current_ball_angle = self._get_angle(ball_pos)
            current_ball_pos_category = self._categorize_position(current_ball_angle)
            current_inst_ball_direction = self._categorize_direction(ball_vel)
            current_arm_angle = self._get_angle(arm_tip_pos)
            current_arm_pos_category = self._categorize_position(current_arm_angle)
            current_inst_arm_direction = self._categorize_direction(arm_tip_vel)
            if current_inst_ball_direction == self.potential_ball_direction:
                self.ball_direction_streak += 1
            else:
                self.potential_ball_direction = current_inst_ball_direction
                self.ball_direction_streak = 1
                self.ball_pos_at_potential_start = current_ball_pos_category
            if self.ball_direction_streak >= self.DIRECTION_CONFIRMATION_STEPS and self.potential_ball_direction != self.confirmed_ball_direction:
                self.confirmed_ball_prev_pos = self.ball_pos_at_potential_start
                self.confirmed_ball_direction = self.potential_ball_direction
            if current_inst_arm_direction == self.potential_arm_direction:
                self.arm_direction_streak += 1
            else:
                self.potential_arm_direction = current_inst_arm_direction
                self.arm_direction_streak = 1
                self.arm_pos_at_potential_start = current_arm_pos_category
            if self.arm_direction_streak >= self.DIRECTION_CONFIRMATION_STEPS and self.potential_arm_direction != self.confirmed_arm_direction:
                self.confirmed_arm_prev_pos = self.arm_pos_at_potential_start
                self.confirmed_arm_direction = self.potential_arm_direction
            ball_speed_change_desc, arm_speed_change_desc = None, None
            ball_speed_delta = current_ball_speed - self.last_reported_ball_speed
            if abs(ball_speed_delta) > self.SPEED_CHANGE_THRESHOLD:
                ball_speed_change_desc = "INCREASE" if ball_speed_delta > 0 else "DECREASE"
                self.last_reported_ball_speed = current_ball_speed
            arm_speed_delta = current_arm_speed - self.last_reported_arm_speed
            if abs(arm_speed_delta) > self.SPEED_CHANGE_THRESHOLD:
                arm_speed_change_desc = "INCREASE" if arm_speed_delta > 0 else "DECREASE"
                self.last_reported_arm_speed = current_arm_speed


            is_ball_in_occlusion_zone = self._is_in_occlusion_zone(current_ball_angle)
            current_raw_collision_state = self._get_collision()


            if current_raw_collision_state:
                if not self.in_collision_with_arm:
                    self.in_collision_with_arm = True
                    self.collision_with_arm_start_step = self.simulation_steps  # 记录开始步数
            else:
                if self.in_collision_with_arm:
                    self.in_collision_with_arm = False
                    self.collision_with_arm_start_step = None

            report_concrete_values_due_to_collision = False
            if self.in_collision_with_arm and self.collision_with_arm_start_step is not None:
                collision_duration_steps = self.simulation_steps - self.collision_with_arm_start_step
                if collision_duration_steps >= self.COLLISION_DURATION_THRESHOLD_STEPS:
                    report_concrete_values_due_to_collision = True


            arm_speed_data = {"value": current_arm_speed, "tags": ["NONE"]}
            arm_motion_data = {"direction": self.confirmed_arm_direction, "prev_pos": self.confirmed_arm_prev_pos,
                               "tags": ["NONE"]}


            if is_ball_in_occlusion_zone:

                ball_speed_data = {"value": "UNKNOWN_VALUE", "tags": []}
                ball_motion_data = {"direction": "UNKNOWN_VALUE", "prev_pos": "UNKNOWN_VALUE", "tags": ["IS_OCCLUDED"]}


                if report_concrete_values_due_to_collision:
                    ball_speed_data["value"] = current_ball_speed
                    ball_motion_data["direction"] = self.confirmed_ball_direction
                    ball_motion_data["prev_pos"] = self.confirmed_ball_prev_pos
            else:

                ball_speed_data = {"value": current_ball_speed, "tags": []}
                ball_motion_data = {"direction": self.confirmed_ball_direction,
                                    "prev_pos": self.confirmed_ball_prev_pos, "tags": []}


            if ball_speed_change_desc:

                if ball_speed_data["value"] == "UNKNOWN_VALUE":
                    ball_speed_data["change"] = "UNKNOWN_VALUE"
                else:
                    ball_speed_data["change"] = ball_speed_change_desc

            if arm_speed_change_desc:
                arm_speed_data["change"] = arm_speed_change_desc

            return {
                "ball_speed": ball_speed_data,
                "arm_speed": arm_speed_data,
                "ball_motion": ball_motion_data,
                "arm_motion": arm_motion_data,
                "collision": current_raw_collision_state
            }

        except p.error as e:
            print(f"Error in Sensor.get_data: {e}")
            return None
        except Exception as ex:
            print(f"Unexpected error in Sensor.get_data: {ex}")
            import traceback
            traceback.print_exc()
            return None