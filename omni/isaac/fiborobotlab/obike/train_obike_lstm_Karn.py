import gym
from gym import spaces
import numpy as np
import math
import time

import carb
from omni.isaac.kit import SimulationApp
from omni.isaac.imu_sensor import _imu_sensor

def discount_rewards(r, gamma=0.8):
    discounted_r = np.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size)):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

def q2falling(q):
    q[0] = 1 if q[0] > 1 else q[0]
    try:
        if q[1] == 0 and q[2] == 0 and q[3] == 0:
            return 0
        return 2*math.acos(q[0])*math.sqrt((q[1]**2 + q[2]**2)/(q[1]**2 + q[2]**2 + q[3]**2))
    except:
        print(q)
        return 0

def omni_unit2_sensor_unit(observations):
    observations[0] = observations[0] / 981.0
    observations[1] = observations[1] / 981.0
    observations[2] = ( observations[2] * 180.0 ) / math.pi
    return observations

def sensor_unit2_omni_unit(observations):
    observations[0] = observations[0] * 981.0
    observations[1] = observations[1] * 981.0
    observations[2] = ( observations[2] * math.pi ) / 180.0
    return observations

## Specify simulation parameters ##
_physics_dt = 1/100
_rendering_dt = 1/30
_max_episode_length = 60/_physics_dt # 60 second after reset
_iteration_count = 0
_display_every_iter = 1
_update_every = 1
_headless = False
simulation_app = SimulationApp({"headless": _headless, "anti_aliasing": 0})

## Setup World ##
from omni.isaac.core import World
from obike_old import Obike
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
props.orientation = carb.Float4(0, 0, 0, 1) # (x, y, z, w)
props.sensorPeriod = 1 / 500  # 2ms
_sensor_handle = imu_interface.add_sensor_on_body("/obike/chassic", props)

## Create LSTM Model ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

## Train parameter ##
n_episodes = 10000
input_dim =	3
output_dim = 1
num_timesteps =	1
batch_size = 1
lstm_nodes = 32

input_layer = tf.keras.Input(shape=(num_timesteps, input_dim), batch_size=batch_size)
lstm_cell = tf.keras.layers.LSTMCell(
    lstm_nodes,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='glorot_uniform',
    bias_initializer='zeros',
)
lstm_layer = tf.keras.layers.RNN(
    lstm_cell,
    return_state=True,
    return_sequences=True,
    stateful=True,
)
lstm_out, hidden_state, cell_state = lstm_layer(input_layer)
output = tf.keras.layers.Dense(output_dim)(lstm_out)
model = tf.keras.Model(
    inputs=input_layer,
    outputs=[hidden_state, cell_state, output]
)

# class SimpleLSTM(keras.Model):
#     def __init__(self, lstm_units, num_output):
#         super().__init__(self)
#         cell = layers.LSTMCell(lstm_units,
#                                kernel_initializer='glorot_uniform',
#                                recurrent_initializer='glorot_uniform',
#                                bias_initializer='zeros')
#         self.lstm = tf.keras.layers.RNN(cell,
#                                         return_state = True,
#                                         return_sequences=True,
#                                         stateful=False)
#         lstm_out, hidden_state, cell_state = self.lstm(input_layer)
#
#
#         self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
#         self.dense = layers.Dense(num_output)
#
#     def get_zero_initial_state(self, inputs):
#         return [tf.zeros((batch_size, lstm_nodes)), tf.zeros((batch_size, lstm_nodes))]
#     def __call__(self, inputs, states = None):
#         if states is None:
#             self.lstm.get_initial_state = self.get_zero_initial_state
#     def call(self, inputs, states=None, return_state = False, training=False):
#         x = inputs
#         if states is None: states = self.lstm1.get_initial_state(x) # state shape = (2, batch_size, lstm_units)
#         print(x.shape)
#         print(len(states))
#         print(states[0].shape)
#         x, sequence, states = self.lstm1(x, initial_state=states, training=training)
#         x = self.dense(x, training=training)
#
#         if return_state: return x, states
#         else: return x
#     @tf.function
#     def train_step(self, inputs):
#         inputs, labels = inputs
#         with tf.GradientTape() as tape:
#             predictions = self(inputs, training=True)
#             loss = self.loss(labels, predictions)
#         grads = tape.gradient(loss, model.trainable_variables)
#         self.optimizer.apply_gradients(zip(grads, model.trainable_variables))
#
#         return {'loss': loss}
# model = SimpleLSTM(lstm_units=8, num_output=1)
# model.build(input_shape=(None, 1, 3))
# model.summary()

optimizer = tf.optimizers.Adam(learning_rate=0.0025)
loss_fn = keras.losses.MeanSquaredError()  # Instantiate a loss function.
# train_mse_metric = keras.metrics.MeanSquaredError()

scores = []
gradBuffer = model.trainable_variables
for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0

for e in range(n_episodes):
    # print("\nStart of episodes %d" % (e,))
    # Reset the environment
    world.reset()
    lstm_layer.reset_states(states=[np.zeros((batch_size, lstm_nodes)), np.zeros((batch_size, lstm_nodes))])
    previous_states = None  # reset LSTM's internal state
    render_counter = 0

    ep_memory = []
    ep_score = 0
    done = False
    previous = {'robot_position': None, 'robot_rotation': None, 'fall_rotation': None}
    present = {'robot_position': None, 'robot_rotation': None, 'fall_rotation': None}

    while not done:
        previous['robot_position'], previous['robot_rotation'] = robot.get_world_pose()
        previous['fall_rotation'] = q2falling(previous['robot_rotation'])
        reading = imu_interface.get_sensor_readings(_sensor_handle)
        if reading.shape[0] == 0:# no valid data in buffer -> init observation wih zeros
            observations = np.array([0, 0, 0])
        else: # IMU will  return [???, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            observations = np.array([reading[-1]["lin_acc_y"],  # Use only lastest data in buffer
                                    reading[-1]["lin_acc_z"],
                                    reading[-1]["ang_vel_x"]])
            ## convert omniverse unit to sensor_read unit
            world_sensor = omni_unit2_sensor_unit(observations)
            key_gen = np.load('/home/teera/.local/share/ov/pkg/isaac_sim-2021.2.1/exts/omni.isaac.fiborobotlab/omni/isaac/fiborobotlab/obike/mu_cov_bike.npz')
            noi_accy,noi_accz,noi_ang_vel_x = np.random.default_rng().multivariate_normal(world_sensor, key_gen["cov"], 1).T
            ## convert sensor_read unit to omniverse unit
            noisy_sensor = np.array([noi_accy,noi_accz,noi_ang_vel_x])
            sim_observations = sensor_unit2_omni_unit(noisy_sensor)
        ## Scale accel from (-1000, 1000) -> (0, 1) for NN inputs
        ## Scale gyro from (-4.36, 4.36) -> (0, 1) for NN inputs (4.36 rad = 250 deg)
        # print(observations)
        # time.sleep(1)
        sim_observations[0] = (sim_observations[0] / 2000) + 0.5
        sim_observations[1] = (sim_observations[1] / 2000) + 0.5
        sim_observations[2] = (sim_observations[2] / 8.72) + 0.5
        sim_observations = np.array(sim_observations, dtype=np.float32).reshape((batch_size, num_timesteps, input_dim))  # add extra dimension for batch_size=1
        with tf.GradientTape() as tape:
            # forward pass
            h_state, c_state, logits = model(sim_observations)    # required input_shape=(None, 1, 3)
            # logits, previous_states = model.call(inputs=observations, states=previous_states, return_state=True, training=True)
            a_dist = logits.numpy()
            ## Choose random action with p = action dist
            # print("A_DIST")
            # print(a_dist)
            # a = np.random.choice(a_dist[0], p=a_dist[0])
            # a = np.argmax(a_dist == a)
            a = a_dist + 0.1*((np.random.rand(*a_dist.shape))-0.5)    # random with uniform distribution (.shape will return tuple so unpack with *)
            loss = loss_fn([a], logits)
            # loss = previous['fall_rotation']

        ## EXECUTE ACTION ##
        from omni.isaac.core.utils.types import ArticulationAction
        # print("LOGITS")
        # print(logits)
        robot.apply_wheel_actions(ArticulationAction(joint_efforts=[a*100*0.607, 0, 0]))
        world.step(render=True)
        present['robot_position'], present['robot_rotation'] = robot.get_world_pose()
        present['fall_rotation'] = q2falling(present['robot_rotation'])
        reward = previous['fall_rotation'] - present['fall_rotation']   # calculate reward from movement toward center
        ## Check for stop event ##
        exceed_time_limit = world.current_time_step_index >= _max_episode_length
        robot_fall = True if previous['fall_rotation'] > 25 / 180 * math.pi else False
        done = exceed_time_limit or robot_fall
        ep_score += reward
        # if done: reward-= 10 # small trick to make training faster
        grads = tape.gradient(loss, model.trainable_weights)
        ep_memory.append([grads, reward])
    scores.append(ep_score)
    # Discount the rewards
    ep_memory = np.array(ep_memory)
    ep_memory[:, 1] = discount_rewards(ep_memory[:, 1])

    for grads, reward in ep_memory:
        for ix, grad in enumerate(grads):
            gradBuffer[ix] += grad * reward

    if e % _update_every == 0:
        optimizer.apply_gradients(zip(gradBuffer, model.trainable_variables))
        for ix, grad in enumerate(gradBuffer): gradBuffer[ix] = grad * 0

    if e % 100 == 0:
        print("Episode {} Score {}".format(e, np.mean(scores[-100:])))