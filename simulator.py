import pybullet as p
import numpy as np

class Simulator:
    def __init__(self, physics_world):

        self.physics_world = physics_world

        self.arm_joint_index = 0

        self.positions = ["LEFT_FRONT QUADRANT", "RIGHT_FRONT QUADRANT", "RIGHT_REAR QUADRANT", "LEFT_REAR QUADRANT"]


    def get_arm_position(self):

        try:

            clock_id = self.physics_world.get_handle("clock")
            if clock_id is None:
                print("ERROR: Simulator could not get the 'clock' handle.")
                return "UNKNOWN"


            joint_state = p.getJointState(clock_id, self.arm_joint_index)
            current_angle_rad = joint_state[0]


            current_angle_deg = np.rad2deg(current_angle_rad)
            normalized_angle_deg = current_angle_deg % 360
            if normalized_angle_deg < 0:
                normalized_angle_deg += 360


            quadrant_index = -1
            if 0 <= normalized_angle_deg < 90:
                quadrant_index = 1 # "RIGHT_FRONT QUADRANT"
            elif 90 <= normalized_angle_deg < 180:
                quadrant_index = 0 # "LEFT_FRONT QUADRANT"
            elif 180 <= normalized_angle_deg < 270:
                quadrant_index = 3 # "LEFT_REAR QUADRANT"
            elif 270 <= normalized_angle_deg < 360:
                quadrant_index = 2 # "RIGHT_REAR QUADRANT"

            if 0 <= quadrant_index < len(self.positions):
                # print(f"DEBUG: Angle {normalized_angle_deg:.1f} deg -> Position: {self.positions[quadrant_index]}") # 可选调试
                return self.positions[quadrant_index]
            else:
                print(f"WARN: Could not map angle {normalized_angle_deg:.1f} to a known position.")
                return "UNKNOWN"

        except Exception as e:
            print(f"ERROR: Failed to get joint state in 'get_arm_position': {e}")
            return "UNKNOWN"