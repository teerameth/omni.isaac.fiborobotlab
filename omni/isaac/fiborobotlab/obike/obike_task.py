import time

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from obike import Obike

import omni
from pxr import Gf, UsdGeom
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path, get_all_matching_child_prims
from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.isaac_sensor import _isaac_sensor

import numpy as np
import torch
import math
import random

def euler_to_quaternion(r):
    (roll, pitch, yaw) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

def q2falling(q):
    fall_angle =  2*torch.acos(q[:,0])*torch.sqrt((q[:,1]*q[:,1] + q[:,2]*q[:,2])/(q[:,1]*q[:,1]) + q[:,2]*q[:,2] + q[:,3]*q[:,3])
    return fall_angle

class ObikeTask(RLTask):
    def __init__(
        self,
        name,
        sim_config,
        env,
        offset=None
    ) -> None:

        self._sim_config = sim_config
        self._cfg = sim_config.config
        self._task_cfg = sim_config.task_config

        self._num_envs = self._task_cfg["env"]["numEnvs"]
        self._env_spacing = self._task_cfg["env"]["envSpacing"]
        self._robot_positions = torch.tensor([0.0, 0.0, 0.0167])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        self._imu_buf = [{"lin_acc_x":0.0, "lin_acc_y":0.0, "lin_acc_z":0.0, "ang_vel_x":0.0, "ang_vel_y":0.0, "ang_vel_z":0.0}]*128  # default initial sensor buffer
        self._is = _isaac_sensor.acquire_imu_sensor_interface()     # Sensor reader
        RLTask.__init__(self, name, env)

        return

    def set_up_scene(self, scene) -> None:
        self.get_obike()    # mush be called before "super().set_up_scene(scene)"
        super().set_up_scene(scene)
        print(get_all_matching_child_prims("/"))
        self._robots = ArticulationView(prim_paths_expr="/World/envs/*/Obike/obike", name="obike_view")
        scene.add(self._robots)
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())
        return

    def get_obike(self):    # must be called at very first line of set_up_scene()
        obike = Obike(prim_path=self.default_zero_env_path + "/Obike", name="Obike", translation=self._robot_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Obike", get_prim_at_path(obike.prim_path), self._sim_config.parse_actor_config("Obike"))

    def get_robot(self):
        return self._robots


    def get_observations(self) -> dict:
        # dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)


        reaction_vel = dof_vel[:, self._reaction_wheel_dof_idx]

        imu_accel_y = torch.tensor([imu["lin_acc_y"] for imu in self._imu_buf])
        imu_accel_z = torch.tensor([imu["lin_acc_z"] for imu in self._imu_buf])
        imu_gyro_x = torch.tensor([imu["ang_vel_x"] for imu in self._imu_buf])

        self.obs_buf[:, 0] = reaction_vel
        self.obs_buf[:, 1] = imu_accel_y
        self.obs_buf[:, 2] = imu_accel_z
        self.obs_buf[:, 3] = imu_gyro_x

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        # print(observations)
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)
        actions = 2*(actions - 0.5)

        forces = torch.zeros((self._robots.count, self._robots.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._reaction_wheel_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        self._robots.set_joint_efforts(forces, indices=indices) # apply joints torque

        ## Read IMU & store in buffer ##
        buffer = []
        robots_prim_path = self._robots.prim_paths
        for robot_prim_path in robots_prim_path:
            reading = self._is.get_sensor_readings(robot_prim_path + "/chassic/sensor") # read from select sensor (by prim_path)
            if reading.shape[0]:
                buffer.append(reading[-1])  # get only lastest reading
            else: buffer.append({"lin_acc_x":0.0, "lin_acc_y":0.0, "lin_acc_z":0.0, "ang_vel_x":0.0, "ang_vel_y":0.0, "ang_vel_z":0.0})  # default initial sensor buffer
        self._imu_buf = buffer
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        # randomize DOF velocities
        dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, 1), device=self._device)

        # apply resets
        self._robots.set_joint_velocities(dof_vel, indices=env_ids)

        root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        root_vel = torch.zeros((num_resets, 6), device=self._device)

        self._robots.set_world_poses(root_pos, root_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):   # Run only once after simulation started
        # self._robots = self.get_robot()
        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()   # save initial position for reset
        self.initial_dof_pos = self._robots.get_joint_positions()
        # initialize some data used later on
        self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        self.basis_vec0 = self.heading_vec.clone()
        self.basis_vec1 = self.up_vec.clone()

        self._reaction_wheel_dof_idx = self._robots.get_dof_index("reaction_wheel_joint")
        self._rear_wheel_dof_idx = self._robots.get_dof_index("rear_wheel_joint")
        self._steering_arm_dof_idx = self._robots.get_dof_index("front_wheel_arm_joint")

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:    # calculate reward for each env
        reaction_vel = self.obs_buf[:, 0]
        robots_position, robots_orientation = self._robots.get_world_poses()
        fall_angles = q2falling(robots_orientation) # find fall angle of all robot (batched)
        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        reward = 1.0 - fall_angles * fall_angles - 0.01 * torch.abs(reaction_vel)
        reward = torch.where(torch.abs(fall_angles) > 25*(np.pi / 180), torch.ones_like(reward) * -2.0, reward)   # fall_angle must <= 25 degree

        self.rew_buf[:] = reward

    def is_done(self) -> None:  # check termination for each env
        robots_position, robots_orientation = self._robots.get_world_poses()
        fall_angles = q2falling(robots_orientation)  # find fall angle of all robot (batched)

        # cart_pos = self.obs_buf[:, 0]
        # pole_pos = self.obs_buf[:, 2]

        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        resets = torch.where(torch.abs(fall_angles) > 25*(np.pi / 180), 1, 0)            # reset by falling
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)  # reset by time
        self.reset_buf[:] = resets
