import gym
from gym import spaces
import numpy as np
import math
import time

import carb
from omni.isaac.kit import SimulationApp
from omni.isaac.imu_sensor import _imu_sensor



def q2falling(q):
    q[0] = 1 if q[0] > 1 else q[0]
    try:
        if q[1] == 0 and q[2] == 0 and q[3] == 0:
            return 0
        return 2*math.acos(q[0])*math.sqrt((q[1]**2 + q[2]**2)/(q[1]**2 + q[2]**2 + q[3]**2))
    except:
        print(q)
        return 0

## Specify simulation parameters ##
_physics_dt = 1/100
_rendering_dt = 1/30
_max_episode_length = 60/_physics_dt # 60 second after reset
_iteration_count = 0
_display_every_iter = 1
_headless = False
simulation_app = SimulationApp({"headless": _headless, "anti_aliasing": 0})

## Train parameter ##

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
            position=np.array([0, 0.0, 30.0]),
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

class SimpleLSTM(keras.Model):
    def __init__(self, num_input, embedding_dim, lstm_units, num_output):
        super().__init__(self)
        self.embedding = layers.Embedding(num_input, embedding_dim)
        self.lstm1 = layers.LSTM(lstm_units, return_sequences=True, return_state=True)
        self.dense = layers.Dense(num_output)
    def call(self, inputs, states=None, return_state = False, training=False):
        x = inputs
        x = self.embedding(x, training=training)
        if states is None: states = self.lstm1.get_initial_state(x) # state shape = (2, batch_size, lstm_units)
        print(x.shape)
        print(len(states))
        print(states[0].shape)
        x, sequence, states = self.lstm1(x, initial_state=states, training=training)
        x = self.dense(x, training=training)

        if return_state: return x, states
        else: return x
    @tf.function
    def train_step(self, inputs):
        inputs, labels = inputs
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.loss(labels, predictions)
        grads = tape.gradient(loss, model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return {'loss': loss}
model = SimpleLSTM(num_input=3, embedding_dim=4, lstm_units=8, num_output=1)
model.build(input_shape=(None, 3))
model.summary()

optimizer = tf.optimizers.Adam(learning_rate=0.0025)
loss_fn = keras.losses.MeanSquaredError()  # Instantiate a loss function.
# train_mse_metric = keras.metrics.MeanSquaredError()

for epoch in range(1000):
    print("\nStart of epoch %d" % (epoch,))
    train_loss = []
    ## loop until robot fall or exceed time limit (1 iteration = 1 batch)
    render_counter = 0
    previous = {'robot_position':None, 'robot_rotation':None, 'fall_rotation':None}
    previous_states = None  # internal state of LSTM
    while True:
        previous['robot_position'], previous['robot_rotation'] = robot.get_world_pose()
        previous['fall_rotation'] = q2falling(previous['robot_rotation'])

        if not _headless and _iteration_count % _display_every_iter == 0:  # limit rendering rate
            if world.current_time_step_index * _physics_dt / _rendering_dt > render_counter:
                render_counter += 1
                world.render()

        reading = imu_interface.get_sensor_readings(_sensor_handle)
        print(reading.shape)
        if reading.shape[0] == 0:# no valid data in buffer -> init observation wih zeros
            observations = np.array([0, 0, 0])
        else: # IMU will  return [???, acc_x, acc_y, acc_z, gyr_x, gyr_y, gyr_z]
            observations = np.array([reading[-1]["lin_acc_y"],  # Use only lastest data in buffer
                                    reading[-1]["lin_acc_z"],
                                    reading[-1]["ang_vel_x"]])
        ## Scale observations from (-10, 10) -> (0, 1) for NN inputs
        observations = observations/20 + 0.5
        observations = np.array(observations, dtype=np.float32).reshape((-1, 3))  # add extra dimension for batch_size=1
        ## EXECUTE ACTION in train step ##
        with tf.GradientTape() as tape:
            logits, previous_states = model.call(inputs=observations, states=previous_states, return_state=True, training=True)
            from omni.isaac.core.utils.types import ArticulationAction
            # robot.apply_wheel_actions(ArticulationAction(joint_efforts=[logits, 0, 0]))
            world.step(render=True)
            robot_position, robot_rotation = robot.get_world_pose()
            fall_rotation = q2falling(robot_rotation)
            delta_fall = fall_rotation - previous['fall_rotation']
            loss_value = tf.convert_to_tensor(-delta_fall, dtype=tf.float32)  # loss got higher when falling
            # tape.watch(loss_value)
        grads = tape.gradient(loss_value, model.trainable_weights)
        print("GRAD")
        print(grads)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        # train_mse_metric.update_state(y, logits)  # update training matric
        train_loss.append(loss_value)

        ## Check for stop event ##
        exceed_time_limit = world.current_time_step_index >= _max_episode_length
        robot_fall = True if previous['fall_rotation'] > 15 / 180 * math.pi else False  # if angle from normal line > 50deg mean it going to fall for sure
        if exceed_time_limit or robot_fall:
            ## RESET ENV ##
            world.reset()
            render_counter = 0
            _iteration_count += 1
            break   # Exti iteration