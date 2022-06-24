#launch Isaac Sim before any other imports
#default first two lines in any standalone application
from omni.isaac.kit import SimulationApp
simulation_app = SimulationApp({"headless": False}) # we can also run as headless.

from omni.isaac.core import World
from omni.isaac.core.objects import DynamicSphere
import numpy as np

world = World()
world.scene.add_default_ground_plane()
ball = world.scene.add(
           DynamicSphere(
               prim_path="/ball",
               name="ball",
               position=np.array([0, 0, 0.12]),
               radius=0.12,  # mediciene ball diameter 24cm.
               color=np.array([1.0, 0, 0]),
               mass=4,
           )
       )
# Resetting the world needs to be called before querying anything related to an articulation specifically.
# Its recommended to always do a reset after adding your assets, for physics handles to be propagated properly
world.reset()
while True:
    # position, orientation = fancy_cube.get_world_pose()
    # linear_velocity = fancy_cube.get_linear_velocity()
    # # will be shown on terminal
    # print("Cube position is : " + str(position))
    # print("Cube's orientation is : " + str(orientation))
    # print("Cube's linear velocity is : " + str(linear_velocity))
    # # we have control over stepping physics and rendering in this workflow
    # # things run in sync
    world.step(render=True) # execute one physics step and one rendering step

simulation_app.close() # close Isaac Sim