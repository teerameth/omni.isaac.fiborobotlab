import random

import gym
from gym import spaces
import numpy as np
import math
import time

import carb
from omni.isaac.imu_sensor import _imu_sensor

state_low = [-100, -100, -10]
state_high = [100, 100, 10]
action_low = [-1]
action_high = [1]


def q2falling(q):
    q[0] = 1 if q[0] > 1 else q[0]
    try:
        if q[1] == 0 and q[2] == 0 and q[3] == 0:
            return 0
        return 2*math.acos(q[0])*math.sqrt((q[1]**2 + q[2]**2)/(q[1]**2 + q[2]**2 + q[3]**2))
    except:
        print(q)
        return 0
class MoonCakeEnv(gym.Env):
    metadata = {"render.modes": ["human"]}
    def __init__(
            self,
            skip_frame=1,
            physics_dt=1.0 / 100.0,
            rendering_dt=1.0 / 60.0,
            max_episode_length=60,
            display_every_iter=20,
            seed=0,
            headless=True,
            observation_list=["lin_acc_y", "lin_acc_z", "ang_vel_x"],
    ) -> None:
        from omni.isaac.kit import SimulationApp
        ## Specify simulation parameters ##
        self._physics_dt = physics_dt
        self._rendering_dt = rendering_dt
        self._max_episode_length = max_episode_length / self._physics_dt  # 60 second after reset
        self._skip_frame = skip_frame
        self._iteration_count = 0
        self._display_every_iter = display_every_iter
        self._update_every = 1
        self._explore_every = 5
        self._headless = headless
        self._observation_list = observation_list
        self.simulation_app = SimulationApp({"headless": self._headless, "anti_aliasing": 0})

        self.previous_observations = {}

        ## Setup World ##
        from omni.isaac.core import World
        from mooncake import MoonCake
        from omni.isaac.core.objects import DynamicSphere
        self.world = World(physics_dt=self._physics_dt, rendering_dt=self._rendering_dt, stage_units_in_meters=0.01)
        self.world.scene.add_default_ground_plane()
        self.robot = self.world.scene.add(
            MoonCake(
                prim_path="/mooncake",
                name="mooncake_mk0",
                position=np.array([0, 0.0, 30.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.ball = self.world.scene.add(
            DynamicSphere(
                prim_path="/ball",
                name="ball",
                position=np.array([0, 0, 12]),
                radius=12,  # mediciene ball diameter 24cm.
                color=np.array([1.0, 0, 0]),
                mass=4,
            )
        )
        ## Setup IMU ##
        self.imu_interface = _imu_sensor.acquire_imu_sensor_interface()
        self.props = _imu_sensor.SensorProperties()
        self.props.position = carb.Float3(0, 0, 10)  # translate from /obike/chassic to above motor (cm.)
        self.props.orientation = carb.Float4(1, 0, 0, 0)  # (x, y, z, w)
        self.props.sensorPeriod = 1 / 500  # 2ms
        self._sensor_handle = self.imu_interface.add_sensor_on_body("/obike/chassic", self.props)

        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(len(self._observation_list),), dtype=np.float32)
    def step(self, action):
        ## EXECUTE ACTION ##
        from omni.isaac.core.utils.types import ArticulationAction
        # print(action)
        # action = (action-0.5)+random.random()*0.01
        joint_actions = ArticulationAction()
        joint_actions.joint_efforts = np.zeros(self.robot.num_dof)
        # joint_actions.joint_velocities = np.zeros(self.robot.num_dof)
        # joint_actions.joint_positions = np.zeros(self.robot.num_dof)
        # joint_actions.joint_efforts[self.robot._wheel_dof_indices[0]] = action[0] * 1000 * 2
        # joint_actions.joint_efforts[self.robot._wheel_dof_indices[1]] = action[1] * 1000 * 2
        # joint_actions.joint_efforts[self.robot._wheel_dof_indices[2]] = action[2] * 1000 * 2
        # joint_actions.joint_velocities[self.robot._wheel_dof_indices[1]] = -0.2* 1000
        # joint_actions.joint_positions[self.robot._wheel_dof_indices[2]] = 0
        # self.robot.apply_action(control_actions=joint_actions)
        self.robot.apply_wheel_actions(ArticulationAction(joint_efforts=[action[i] * 3000 for i in range(3)]))
        self.world.step(render=(not self._headless) and (self._iteration_count%self._display_every_iter==0))
        observations = self.get_observation()
        
        reward = self.previous_observations['fall_rotation'] - observations['fall_rotation']
        reward *= 100
        ## Check for stop event ##
        exceed_time_limit = self.world.current_time_step_index >= self._max_episode_length
        robot_fall = True if observations['fall_rotation'] > 50 / 180 * math.pi else False
        done = exceed_time_limit or robot_fall
        info = {}

        obs = [observations[name] for name in self._observation_list]
        scaled_observation = []
        for name in self._observation_list:
            if "lin_acc" in name: scaled_observation.append(observations[name]/1000)    # (-1000,1000)cm/s^2    -> (-1,1)
            if "ang_vel" in name: scaled_observation.append(observations[name]/10)      # (-10,10)rad/s         -> (-1,1)
            if "rotation" in name: scaled_observation.append(observations[name])        # quaternion already inrange(0,1)
        self.previous_observations = observations.copy()
        return obs, reward, done, info
    def reset(self):
        self._iteration_count += 1
        self.world.reset()
        self.robot.initialize()
        # self.world.scene.remove("/obike")
        # from obike import Obike
        # self.robot = self.world.scene.add(
        #     Obike(
        #         prim_path="/obike",
        #         name="obike_mk0",
        #         position=np.array([10 * random.random(), 10 * random.random(), 1.435]),
        #         orientation=np.array([1.0, 0.0, 0.0, 0.0]),
        #     )
        # )
        observations = self.get_observation()
        obs = [observations[name] for name in self._observation_list]
        self.previous_observations['fall_rotation'] = 0
        return obs
    def get_observation(self):
        observations = {"robot_position_x":None, "robot_position_y":None, "robot_position_z":None, "robot_rotation_x":None, "robot_rotation_y":None, "robot_rotation_z":None, "robot_rotation_w":None, "lin_acc_x":None, "lin_acc_y":None, "lin_acc_z":None, "ang_vel_x":None, "ang_vel_y":None, "ang_vel_z":None}
        [observations["robot_position_x"], observations["robot_position_y"], observations["robot_position_z"]], [observations["robot_rotation_x"], observations["robot_rotation_y"], observations["robot_rotation_z"], observations["robot_rotation_w"]] = self.robot.get_world_pose()
        reading = self.imu_interface.get_sensor_readings(self._sensor_handle)
        if reading.shape[0] == 0:  # no valid data in buffer -> init observation wih zeros
            observations["lin_acc_x"],  observations["lin_acc_y"],  observations["lin_acc_z"], observations["ang_vel_x"], observations["ang_vel_y"], observations["ang_vel_z"] = 0, 0, 0, 0, 0, 0
        else:
            observations["lin_acc_x"],  observations["lin_acc_y"],  observations["lin_acc_z"], observations["ang_vel_x"], observations["ang_vel_y"], observations["ang_vel_z"] = reading[-1]["lin_acc_x"], reading[-1]["lin_acc_y"], reading[-1]["lin_acc_z"], reading[-1]["ang_vel_x"], reading[-1]["ang_vel_y"], reading[-1]["ang_vel_z"]
        observations["fall_rotation"] = q2falling([observations["robot_rotation_x"], observations["robot_rotation_y"], observations["robot_rotation_z"], observations["robot_rotation_w"]])
        return observations
    def close(self):
        pass

    def seed(self, seed=None):
        self.np_random, seed = gym.utils.seeding.np_random(seed)
        np.random.seed(seed)
        return [seed]