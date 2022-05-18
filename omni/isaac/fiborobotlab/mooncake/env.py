import gym
from gym import spaces
import numpy as np
import math

import carb
from omni.isaac.imu_sensor import _imu_sensor


class MoonCakeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}

    def __init__(
        self,
        skip_frame=1,
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        max_episode_length=1000,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        from omni.isaac.core import World
        from mooncake import MoonCake
        from omni.isaac.core.objects import DynamicSphere
        # import omni.physx as _physx
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=0.01)
        self._my_world.scene.add_default_ground_plane()
        self.mooncake = self._my_world.scene.add(
            MoonCake(
                prim_path="/mooncake",
                name="mooncake_mk0",
                position=np.array([0, 0.0, 2.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.ball = self._my_world.scene.add(
            DynamicSphere(
                prim_path="/ball",
                name="ball",
                position=np.array([0, 0, 12]),
                radius=0.12,
                color=np.array([1.0, 0, 0]),
            )
        )
        self._is = _imu_sensor.acquire_imu_sensor_interface()

        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_imu()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        
        # self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)
        return

    def get_dt(self):
        return self._dt
    
    def step(self, action):
        previous_mooncake_position, _ = self.mooncake.get_world_pose()
        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction

            self.mooncake.apply_wheel_actions(ArticulationAction(joint_velocities=action * 10.0))
            self._my_world.step(render=False)
        observations = self.get_observations()
        info = {}
        done = False
        if self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length:
            done = True
        goal_world_position, _ = self.ball.get_world_pose()
        current_mooncake_position, _ = self.mooncake.get_world_pose()
        previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_mooncake_position)
        current_dist_to_goal = np.linalg.norm(goal_world_position - current_mooncake_position)
        reward = previous_dist_to_goal - current_dist_to_goal
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 100 * math.sqrt(np.random.rand()) + 20
        # self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 2.5]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        self._my_world.render()
        # wait_for_sensor_data is recommended when capturing multiple sensors, in this case we can set it to zero as we only need RGB
        # gt = self.sd_helper.get_groundtruth(
        #     ["rgb"], self.viewport_window, verify_sensor_init=False, wait_for_sensor_data=0
        # )
        reading = self._is.get_sensor_readings(self._sensor_handle)
        print(reading)
        # return gt["rgb"][:, :, :3]
        return reading

    def render(self, mode="human"):
        return

    def close(self):
        self._simulation_app.close()
        return

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]

    def _set_imu(self):
        import omni.kit
        from omni.isaac.synthetic_utils import SyntheticDataHelper

        props = _imu_sensor.SensorProperties()
        props.position = carb.Float3(0, 0, 0)
        props.orientation = carb.Float4(0, 0, 0, 1)
        props.sensorPeriod = 1 / 500  # 2ms
        self._sensor_handle = self._is.add_sensor_on_body("/mooncake/top_plate", props)