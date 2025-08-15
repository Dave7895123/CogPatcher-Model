

import pybullet as p
import pybullet_data
import numpy as np
import time

class PhysicsWorld:
    def __init__(self, gui=True):

        self.client = p.connect(p.GUI if gui else p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())

        self.arm_joint_index = 0

        self.motor_force = 50000000
        self._setup_scene()

    def _setup_scene(self):

        self.plane = p.loadURDF("plane.urdf")


        cylinder_visual = p.createVisualShape(
            shapeType=p.GEOM_MESH,
            fileName="cylinder_no_top_collision.stl", meshScale=[1, 1, 1],
            rgbaColor=[0.8, 0.2, 0.2, 1]
        )
        cylinder_collision = p.createCollisionShape(
            shapeType=p.GEOM_MESH,
            fileName="cylinder_no_top_collision.stl", meshScale=[1, 1, 1],
            flags=p.GEOM_FORCE_CONCAVE_TRIMESH
        )
        self.cylinder = p.createMultiBody(
            baseMass=0, baseCollisionShapeIndex=cylinder_collision,
            baseVisualShapeIndex=cylinder_visual, basePosition=[0, 0, 0],
            baseOrientation=p.getQuaternionFromEuler([0, 0, 0])
        )
        p.changeDynamics(self.cylinder, -1, linearDamping=100,
                         angularDamping=100, activationState=p.ACTIVATION_STATE_SLEEP)



        self.clock = p.loadURDF("clock.urdf", [0, 0, 0], useFixedBase=True)
        self.ball = p.loadURDF("sphere_small.urdf", [0.4, -0.2, 0.03], globalScaling=1)


        print(f"DEBUG: Initializing clock joint {self.arm_joint_index} with VELOCITY_CONTROL, target velocity 0.")
        p.setJointMotorControl2(
            bodyUniqueId=self.clock,
            jointIndex=self.arm_joint_index,
            controlMode=p.VELOCITY_CONTROL,
            targetVelocity=0,
            force=self.motor_force
        )


        p.changeDynamics(self.ball, -1, mass=0.05)
        p.setGravity(0, 0, -9.8)

    def get_handle(self, obj_name):

        return {"plane": self.plane, "clock": self.clock, "ball": self.ball}.get(obj_name)


    def set_arm_velocity(self, target_velocity):
        try:
            p.setJointMotorControl2(
                bodyUniqueId=self.clock,
                jointIndex=self.arm_joint_index,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=self.motor_force
            )
        except Exception as e:
            print(f"ERROR: Failed to set joint velocity: {e}")


    def stop_arm_rotation(self):

        print(f"DEBUG: Stopping rotation of clock joint {self.arm_joint_index} (setting target velocity to 0).")
        self.set_arm_velocity(0)


    def step(self):

        p.stepSimulation()

    def __enter__(self):

        return self

    def __exit__(self, exc_type, exc_val, exc_tb):

        print("DEBUG: Disconnecting from PyBullet...")
        p.disconnect(self.client)


if __name__ == "__main__":

    with PhysicsWorld(gui=True) as world:
        print("Test: Rotating clockwise for 3 seconds...")
        world.set_arm_velocity(1.0)
        for _ in range(int(3 * 240)):
            world.step()
            time.sleep(1/240)

        print("Test: Stopping rotation for 2 seconds...")
        world.stop_arm_rotation()
        for _ in range(int(2 * 240)):
            world.step()
            time.sleep(1/240)

        print("Test: Rotating counter-clockwise for 3 seconds...")
        world.set_arm_velocity(-0.5)
        for _ in range(int(3 * 240)):
            world.step()
            time.sleep(1/240)

        print("Test: Stopping again...")
        world.stop_arm_rotation()
        for _ in range(int(2 * 240)):
            world.step()
            time.sleep(1/240)

        print("Test complete.")