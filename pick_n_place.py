from isaacsim import SimulationApp
simulation_app = SimulationApp({"headless": False})  # headless=False to visualize the simulation

from isaacsim.core.api import World
from isaacsim.robot.manipulators.examples.franka import Franka
from isaacsim.core.api.objects import DynamicCuboid
from isaacsim.robot.manipulators.examples.franka.controllers import PickPlaceController

import numpy as np


class HelloWorld:
    def __init__(self):
        self.world = World()

    def setup_scene(self):
        self.world.scene.add_default_ground_plane()

        self.franka = self.world.scene.add(
            Franka(prim_path="/World/Fancy_Franka", name="fancy_franka")
        )

        self.cube = self.world.scene.add(
            DynamicCuboid(
                prim_path="/World/random_cube",
                name="fancy_cube",
                position=np.array([0.3, 0.3, 0.05]),
                scale=np.array([0.05, 0.05, 0.05]),
                color=np.array([0, 0, 1.0]),
            )
        )

    def setup_controller(self):
        self.controller = PickPlaceController(
            name="pick_place_controller",
            gripper=self.franka.gripper,
            robot_articulation=self.franka,
        )

        # gripper open
        self.franka.gripper.set_joint_positions(
            self.franka.gripper.joint_opened_positions
        )

    def run(self):
        self.world.reset()
        self.setup_controller()

        # start sim
        self.world.play()

        for _ in range(1000):   # constant step loop
            self.world.step(render=True)

            cube_position, _ = self.cube.get_world_pose()
            goal_position = np.array([-0.3, -0.3, 0.05])

            current_joint_positions = self.franka.get_joint_positions()

            actions = self.controller.forward(
                picking_position=cube_position,
                placing_position=goal_position,
                current_joint_positions=current_joint_positions,
            )

            self.franka.apply_action(actions)

            if self.controller.is_done():
                print("DONE")
                break


app = HelloWorld()
app.setup_scene()
app.run()

simulation_app.close()