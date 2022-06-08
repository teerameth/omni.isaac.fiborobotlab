import carb
from omni.isaac.kit import SimulationApp
from omni.isaac.imu_sensor import _imu_sensor
import numpy as np
import math
import time
import tensorflow as tf

def q2falling(q):
    q[0] = 1 if q[0] > 1 else q[0]
    try:
        if q[1] == 0 and q[2] == 0 and q[3] == 0:
            return 0
        return 2*math.acos(q[0])*math.sqrt((q[1]**2 + q[2]**2)/(q[1]**2 + q[2]**2 + q[3]**2))
    except:
        print(q)
        return 0

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

## Create LSTM Model ##
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
