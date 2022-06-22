import gym
from gym import spaces
import numpy as np
import math
import time

import carb
from omni.isaac.imu_sensor import _imu_sensor
from torch import dtype

sliding_window_width = 3
def velocity2omega(v_x, v_y, w_z=0, d=0.105, r=0.1):    # r: wheel radius, d=distance from center to wheel contact
    omega_0 = (v_x-d*w_z)/r
    omega_1 = -(v_x-math.sqrt(3)*v_y+2*d*w_z)/(2*r)
    omega_2 = -(v_x+math.sqrt(3)*v_y+2*d*w_z)/(2*r)
    return [omega_0, omega_1, omega_2]
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
        max_episode_length=1000,
        display_every_iter = 1,
        seed=0,
        headless=True,
    ) -> None:
        from omni.isaac.kit import SimulationApp

        self.headless = headless
        self._simulation_app = SimulationApp({"headless": self.headless, "anti_aliasing": 0})
        self._skip_frame = skip_frame
        self._dt = physics_dt * self._skip_frame
        self._physics_dt = physics_dt
        self._rendering_dt = rendering_dt
        self._max_episode_length = max_episode_length
        self._steps_after_reset = int(rendering_dt / physics_dt)
        self._iteration_count = 0
        self._display_every_iter = display_every_iter
        from omni.isaac.core import World
        from mooncake_old import MoonCake
        from omni.isaac.core.objects import DynamicSphere
        # import omni.physx as _physx
        self._my_world = World(physics_dt=physics_dt, rendering_dt=rendering_dt, stage_units_in_meters=0.01)
        self._my_world.scene.add_default_ground_plane()
        self.mooncake = self._my_world.scene.add(
            MoonCake(
                prim_path="/mooncake",
                name="mooncake_mk0",
                position=np.array([0, 0.0, 30.0]),
                orientation=np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )
        self.ball = self._my_world.scene.add(
            DynamicSphere(
                prim_path="/ball",
                name="ball",
                position=np.array([0, 0, 12]),
                radius=12,  # mediciene ball diameter 24cm.
                color=np.array([1.0, 0, 0]),
                mass = 4,
            )
        )

        self.seed(seed)
        self.sd_helper = None
        self.viewport_window = None
        self._set_imu()
        self.reward_range = (-float("inf"), float("inf"))
        gym.Env.__init__(self)
        # self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(2,), dtype=np.float32)
        # self.observation_space = spaces.Box(low=0, high=255, shape=(128, 128, 3), dtype=np.uint8)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-20, high=20, shape=(sliding_window_width, 9,), dtype=np.float32)
        
        # self.sub = _physx.get_physx_interface().subscribe_physics_step_events(self._on_update)
        return

    def get_dt(self):
        return self._dt
    
    def step(self, action):
        # self.mooncake.* -> ['__class__', '__delattr__', '__dict__', '__dir__', '__doc__', '__eq__', '__format__', '__ge__', '__getattribute__', '__gt__', '__hash__', '__init__', '__init_subclass__', '__le__', '__lt__', '__module__', '__ne__', '__new__', '__reduce__', '__reduce_ex__', '__repr__', '__setattr__', '__sizeof__', '__str__', '__subclasshook__', '__weakref__', '_applied_visual_material', '_articulation_controller', '_binding_api', '_dc_interface', '_default_joints_state', '_default_state', '_dofs_infos', '_handle', '_handles_initialized', '_name', '_num_dof', '_prim', '_prim_path', '_read_kinematic_hierarchy', '_root_handle', '_sensors', '_set_xform_properties', '_wheel_dof_indices', '_wheel_dof_names', 'apply_action', 'apply_visual_material', 'apply_wheel_actions', 'articulation_handle', 'change_prim_path', 'disable_gravity', 'dof_properties', 'get_angular_velocity', 'get_applied_action', 'get_applied_visual_material', 'get_articulation_body_count', 'get_articulation_controller', 'get_default_state', 'get_dof_index', 'get_enabled_self_collisions', 'get_joint_efforts', 'get_joint_positions', 'get_joint_velocities', 'get_joints_state', 'get_linear_velocity', 'get_local_pose', 'get_local_scale', 'get_sleep_threshold', 'get_solver_position_iteration_count', 'get_solver_velocity_iteration_count', 'get_stabilization_threshold', 'get_visibility', 'get_wheel_positions', 'get_wheel_velocities', 'get_world_pose', 'get_world_scale', 'handles_initialized', 'initialize', 'is_valid', 'is_visual_material_applied', 'name', 'num_dof', 'post_reset', 'prim', 'prim_path', 'read_kinematic_hierarchy', 'set_angular_velocity', 'set_default_state', 'set_enabled_self_collisions', 'set_joint_efforts', 'set_joint_positions', 'set_joint_velocities', 'set_joints_default_state', 'set_linear_velocity', 'set_local_pose', 'set_local_scale', 'set_sleep_threshold', 'set_solver_position_iteration_count', 'set_solver_velocity_iteration_count', 'set_stabilization_threshold', 'set_visibility', 'set_wheel_positions', 'set_wheel_velocities', 'set_world_pose', 'wheel_dof_indicies']
        previous_mooncake_position, previous_mooncake_rotation = self.mooncake.get_world_pose() # [x, y, z], [w, x, y, z]
        previous_ball_position, previous_ball_rotation = self.ball.get_world_pose()
        previous_fall_rotation = q2falling(previous_mooncake_rotation)
        print(previous_fall_rotation)
        time.sleep(0.1)
        previous_mooncake_wheel_velocities = self.mooncake.get_wheel_velocities()
        # print(action)
        ## Convert action(linear_velocity) -> angular_velocity
        # wheel_speed = velocity2omega(action[0], action[1])
        ## EXECUTE ACTION ##
        for i in range(self._skip_frame):
            from omni.isaac.core.utils.types import ArticulationAction
            # self.mooncake.apply_wheel_actions(ArticulationAction(joint_velocities=wheel_speed))
            # self.mooncake.apply_wheel_actions(ArticulationAction(joint_efforts=[wheel_speed[i]*500 for i in range(3)]))
            self.mooncake.apply_wheel_actions(ArticulationAction(joint_efforts=[action[i] * 10000 for i in range(3)]))
            self._my_world.step(render=False)
        observations = self.get_observations()
        # print(observations)
        info = {}
        done = False
        ## Check for stop event ##
        exceed_time_limit = self._my_world.current_time_step_index - self._steps_after_reset >= self._max_episode_length
        robot_fall = True if previous_fall_rotation > 50/180*math.pi else False    # if angle from normal line > 50deg mean it going to fall for sure
        if exceed_time_limit or robot_fall:
            done = True
            self._iteration_count += 1
            # print("Frame count after_reset: ", end="")
            # print(self._my_world.current_time_step_index - self._steps_after_reset) # Frames count after reset
            
        ## GET FEEDBACK & CALCULATE REWARD ##
        current_mooncake_position, current_mooncake_rotation = self.mooncake.get_world_pose() # [x, y, z], [w, x, y, z]
        current_ball_position, current_ball_rotation = self.ball.get_world_pose()

        current_fall_rotation = q2falling(current_mooncake_rotation)
        
        reward = previous_fall_rotation - current_fall_rotation
        reward *= 100
        # print("%.4f"%(reward))
        # previous_dist_to_goal = np.linalg.norm(goal_world_position - previous_mooncake_position)
        # current_dist_to_goal = np.linalg.norm(goal_world_position - current_mooncake_position)
        # reward = previous_dist_to_goal - current_dist_to_goal
        # print("observations, reward, done, info")
        # print(observations, reward, done, info)

        # print("Reward")
        # print(reward)
        # print(current_mooncake_rotation)
        return observations, reward, done, info

    def reset(self):
        self._my_world.reset()
        self._render_counter = 0
        self._observations_buffer = []  # clear observation buffer
        # randomize goal location in circle around robot
        alpha = 2 * math.pi * np.random.rand()
        r = 100 * math.sqrt(np.random.rand()) + 20
        # self.goal.set_world_pose(np.array([math.sin(alpha) * r, math.cos(alpha) * r, 2.5]))
        observations = self.get_observations()
        return observations

    def get_observations(self):
        if not self.headless and self._iteration_count%self._display_every_iter==0:
            if (self._my_world.current_time_step_index - self._steps_after_reset)*self._physics_dt/self._rendering_dt > self._render_counter:
                self._render_counter += 1
                self._my_world.render()
        reading = self._is.get_sensor_readings(self._sensor_handle)
        mooncake_wheel_velocities = self.mooncake.get_wheel_velocities()
        # print(reading)
        if reading.shape[0]:    # at least 1 data in buffer (use the most recent data)
            # IMU will  return [???, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            observations = np.array([reading[-1]["lin_acc_x"],
                                    reading[-1]["lin_acc_y"],
                                    reading[-1]["lin_acc_z"],
                                    reading[-1]["ang_vel_x"],
                                    reading[-1]["ang_vel_y"],
                                    reading[-1]["ang_vel_z"],
                                    mooncake_wheel_velocities[0],
                                    mooncake_wheel_velocities[1],
                                    mooncake_wheel_velocities[2]], dtype=np.float32)
            self._observations_buffer.append(observations)
        ## Fill NaN with ZEROs
        while len(self._observations_buffer) < sliding_window_width: self._observations_buffer = [np.zeros((9), dtype=np.float32)] + self._observations_buffer
        self._observations_buffer = self._observations_buffer[-sliding_window_width:]
        return np.array(self._observations_buffer, dtype=np.float32)

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
        self._is = _imu_sensor.acquire_imu_sensor_interface()
        props = _imu_sensor.SensorProperties()
        props.position = carb.Float3(0, 0, 17.15) # translate to surface of /mooncake/top_plate
        props.orientation = carb.Float4(0, 0, 0, 1) # (x, y, z, w)
        props.sensorPeriod = 1 / 500  # 2ms
        self._sensor_handle = self._is.add_sensor_on_body("/mooncake/base_plate", props)