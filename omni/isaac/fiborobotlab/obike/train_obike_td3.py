import tensorflow as tf
import numpy as np
import math
import time

def q2falling(q):
    q[0] = 1 if q[0] > 1 else q[0]
    try:
        if q[1] == 0 and q[2] == 0 and q[3] == 0:
            return 0
        return 2*math.acos(q[0])*math.sqrt((q[1]**2 + q[2]**2)/(q[1]**2 + q[2]**2 + q[3]**2))
    except:
        print(q)
        return 0

import carb
from omni.isaac.kit import SimulationApp
from omni.isaac.imu_sensor import _imu_sensor


print(tf.config.list_physical_devices('GPU'))

state_low = [-10, -10, -10]
state_high = [10, 10, 10]
action_low = [-1]
action_high = [1]

## Specify simulation parameters ##
_physics_dt = 1/1000
_rendering_dt = 1/30
_max_episode_length = 60/_physics_dt # 60 second after reset
_iteration_count = 0
_display_every_iter = 1
_update_every = 1
_explore_every = 5
_headless = False
simulation_app = SimulationApp({"headless": _headless, "anti_aliasing": 0})

## Setup World ##
from omni.isaac.core import World
from obike import Obike
# from omni.isaac.core.objects import DynamicSphere
world = World(physics_dt=_physics_dt, rendering_dt=_rendering_dt, stage_units_in_meters=0.01)
world.scene.add_default_ground_plane()
robot = world.scene.add(
    Obike(
            prim_path="/obike",
            name="obike_mk0",
            position=np.array([0, 0.0, 1.435]),
            orientation=np.array([1.0, 0.0, 0.0, 0.0]),
    )
)
## Setup IMU ##
imu_interface = _imu_sensor.acquire_imu_sensor_interface()
props = _imu_sensor.SensorProperties()
props.position = carb.Float3(0, 0, 10) # translate from /obike/chassic to above motor (cm.)
props.orientation = carb.Float4(1, 0, 0, 0) # (x, y, z, w)
props.sensorPeriod = 1 / 500  # 2ms
_sensor_handle = imu_interface.add_sensor_on_body("/obike/chassic", props)

from td3 import RBuffer, Critic, Actor, Agent
with tf.device('GPU:0'):
    tf.random.set_seed(336699)
    agent = Agent(n_action=1, n_state=3, action_low=action_low, action_high=action_high)
    episods = 20000
    ep_reward = []
    total_avgr = []
    target = False

    for s in range(episods):
        if target == True:
            break
        total_reward = 0
        state = world.reset()
        done = False
        previous = {'robot_position': None, 'robot_rotation': None, 'fall_rotation': None}
        present = {'robot_position': None, 'robot_rotation': None, 'fall_rotation': None}
        while not done:
            previous['robot_position'], previous['robot_rotation'] = robot.get_world_pose()
            previous['fall_rotation'] = q2falling(previous['robot_rotation'])
            reading = imu_interface.get_sensor_readings(_sensor_handle)
            if reading.shape[0] == 0:  # no valid data in buffer -> init observation wih zeros
                observations = np.array([0, 0, 0])
            else:  # IMU will  return [???, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
                observations = np.array([reading[-1]["lin_acc_y"],  # Use only lastest data in buffer
                                         reading[-1]["lin_acc_z"],
                                         reading[-1]["ang_vel_x"]])
            ## Scale accel from (-1000, 1000) -> (0, 1) for NN inputs
            ## Scale gyro from (-4.36, 4.36) -> (0, 1) for NN inputs (4.36 rad = 250 deg)
            # print(observations)
            # time.sleep(1)
            observations[0] = (observations[0] / 2000) + 0.5
            observations[1] = (observations[1] / 2000) + 0.5
            observations[2] = (observations[2] / 8.72) + 0.5
            # observations = np.array(observations, dtype=np.float32).reshape((batch_size, num_timesteps, input_dim))  # add extra dimension for batch_size=1
            observations = np.array(observations, dtype=np.float32)

            action = agent.act(observations)

            ## EXECUTE ACTION ##
            from omni.isaac.core.utils.types import ArticulationAction
            robot.apply_wheel_actions(ArticulationAction(joint_efforts=[action * 100 * 0.607, 0, 0]))
            world.step(render=True)
            present['robot_position'], present['robot_rotation'] = robot.get_world_pose()
            present['fall_rotation'] = q2falling(present['robot_rotation'])
            reward = previous['fall_rotation'] - present['fall_rotation']  # calculate reward from movement toward center
            ## Check for stop event ##
            exceed_time_limit = world.current_time_step_index >= _max_episode_length
            robot_fall = True if previous['fall_rotation'] > 50 / 180 * math.pi else False
            done = exceed_time_limit or robot_fall
            total_reward += reward
            if done:
                ep_reward.append(total_reward)
                avg_reward = np.mean(ep_reward[-100:])
                total_avgr.append(avg_reward)
                print("total reward after {} steps is {} and avg reward is {}".format(s, total_reward, avg_reward))
                if avg_reward == 200:
                    target = True