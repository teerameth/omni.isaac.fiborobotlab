import time

from omniisaacgymenvs.tasks.base.rl_task import RLTask
from mooncake import Mooncake, Ball

import omni
from scripts.common import set_drive_parameters
from pxr import UsdPhysics, Gf, UsdGeom
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path, get_all_matching_child_prims
import omni.isaac.core.utils.torch.rotations as torch_rot
# from omni.isaac.core.utils.torch.rotations import compute_heading_and_up, compute_rot, quat_conjugate, quat_from_angle_axis, quat_rotate
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale
from omni.isaac.isaac_sensor import _isaac_sensor

import numpy as np
import torch
import torch.nn.functional as f
import math
import random

def euler_to_quaternion(r):
    (roll, pitch, yaw) = (r[0], r[1], r[2])
    qx = np.sin(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) - np.cos(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    qy = np.cos(roll/2) * np.sin(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.cos(pitch/2) * np.sin(yaw/2)
    qz = np.cos(roll/2) * np.cos(pitch/2) * np.sin(yaw/2) - np.sin(roll/2) * np.sin(pitch/2) * np.cos(yaw/2)
    qw = np.cos(roll/2) * np.cos(pitch/2) * np.cos(yaw/2) + np.sin(roll/2) * np.sin(pitch/2) * np.sin(yaw/2)
    return [qx, qy, qz, qw]

# def q2falling(q):
#     fall_angle =  2*torch.acos(q[:,0])*torch.sqrt((q[:,1]*q[:,1] + q[:,2]*q[:,2])/(q[:,1]*q[:,1]) + q[:,2]*q[:,2] + q[:,3]*q[:,3])
#     return fall_angle

# def q2falling(robots_orientation):
#     up_vectors = torch.zeros_like(robots_orientation)
#     up_vectors[:, 3] = 1
#     return torch_rot.quat_diff_rad(robots_orientation, up_vectors)

def q2falling(q):
    norm_vec = f.normalize(q[:, 1:], p=1, dim=1)
    return 2 * torch.acos(q[:, 0]) * torch.sqrt((norm_vec[:, 0] * norm_vec[:, 0] + norm_vec[:, 1] * norm_vec[:, 1]))

class MooncakeTask(RLTask):
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
        self._ball_size = 0.12
        self._ball_positions = torch.tensor([0.0, 0.0, 0.12])   # ball diameter is 12 cm.
        self._robot_offset = 0.1962
        self._jump_offset = 0.01

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_wheel_velocity = self._task_cfg["env"]["maxWheelVelocity"]
        self.heading_weight = self._task_cfg["env"]["headingWeight"]
        self.up_weight = self._task_cfg["env"]["upWeight"]
        self.actions_cost_scale = self._task_cfg["env"]["actionsCost"]
        self.energy_cost_scale = self._task_cfg["env"]["energyCost"]
        self.joints_at_limit_cost_scale = self._task_cfg["env"]["jointsAtLimitCost"]
        self.death_cost = self._task_cfg["env"]["deathCost"]
        self.termination_height = self._task_cfg["env"]["terminationHeight"]
        self.alive_reward_scale = self._task_cfg["env"]["alive_reward_scale"]


        self._max_episode_length = 5000

        self._num_observations = 22
        self._num_actions = 3

        self._imu_buf = [{"lin_acc_x":0.0, "lin_acc_y":0.0, "lin_acc_z":0.0, "ang_vel_x":0.0, "ang_vel_y":0.0, "ang_vel_z":0.0}]*128  # default initial sensor buffer
        self._is = _isaac_sensor.acquire_imu_sensor_interface()     # Sensor reader
        self.previous_fall_angle = None
        RLTask.__init__(self, name, env)

        return

    def set_up_scene(self, scene) -> None:
        self.get_mooncake()    # mush be called before "super().set_up_scene(scene)"
        # self.get_ball()
        super().set_up_scene(scene)
        print(get_all_matching_child_prims("/"))
        self._robots = ArticulationView(prim_paths_expr="/World/envs/*/Mooncake/mooncake", name="mooncake_view")

        # Add ball for each robot
        stage = omni.usd.get_context().get_stage()
        for robot_path in self._robots.prim_paths:
            ball_path = robot_path[:-18] + "/ball"    # remove "/Mooncake/mooncake" and add "/ball" instead
            cubeGeom = UsdGeom.Sphere.Define(stage, ball_path)
            ballPrim = stage.GetPrimAtPath(ball_path)
            size = self._ball_size
            offset = Gf.Vec3f(0.0, 0.0, self._ball_size)
            cubeGeom.CreateRadiusAttr(size)
            cubeGeom.AddTranslateOp().Set(offset)
            # Attach Rigid Body and Collision Preset
            rigid_api = UsdPhysics.RigidBodyAPI.Apply(ballPrim)
            mass_api = UsdPhysics.MassAPI.Apply(ballPrim)
            mass_api.CreateMassAttr(4)
            rigid_api.CreateRigidBodyEnabledAttr(True)

            UsdPhysics.CollisionAPI.Apply(ballPrim)

            phys_api = UsdPhysics.MaterialAPI.Apply(ballPrim)
            phys_api.CreateStaticFrictionAttr().Set(1.0)
            phys_api.CreateDynamicFrictionAttr().Set(1.0)

        print(get_all_matching_child_prims("/"))
        self._ball = RigidPrimView(prim_paths_expr="/World/envs/*/ball", name="ball_view")
        scene.add(self._robots)
        scene.add(self._ball)
        # self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())

        return

    def get_mooncake(self):    # must be called at very first line of set_up_scene()
        robot_position = self._ball_positions
        robot_position[2] += self._robot_offset
        mooncake = Mooncake(prim_path=self.default_zero_env_path + "/Mooncake", name="Mooncake", translation=robot_position)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Mooncake", get_prim_at_path(mooncake.prim_path), self._sim_config.parse_actor_config("Mooncake"))
    def get_ball(self):
        ball = Ball(prim_path=self.default_zero_env_path + "/Ball", name="Ball", translation=self._ball_positions)
    def get_robot(self):
        return self._robots


    def get_observations(self) -> dict:
        # dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)


        wheel_vel_0 = dof_vel[:, self._wheel_0_dof_idx]
        wheel_vel_1 = dof_vel[:, self._wheel_1_dof_idx]
        wheel_vel_2 = dof_vel[:, self._wheel_2_dof_idx]

        imu_accel_x = torch.tensor([imu["lin_acc_x"] for imu in self._imu_buf])
        imu_accel_y = torch.tensor([imu["lin_acc_y"] for imu in self._imu_buf])
        imu_accel_z = torch.tensor([imu["lin_acc_z"] for imu in self._imu_buf])
        imu_gyro_x = torch.tensor([imu["ang_vel_x"] for imu in self._imu_buf])
        imu_gyro_y = torch.tensor([imu["ang_vel_y"] for imu in self._imu_buf])
        imu_gyro_z = torch.tensor([imu["ang_vel_z"] for imu in self._imu_buf])

        self.obs_buf[:, 0] = wheel_vel_0
        self.obs_buf[:, 1] = wheel_vel_1
        self.obs_buf[:, 2] = wheel_vel_2
        self.obs_buf[:, 3] = imu_accel_x
        self.obs_buf[:, 4] = imu_accel_y
        self.obs_buf[:, 5] = imu_accel_z
        self.obs_buf[:, 6] = imu_gyro_x
        self.obs_buf[:, 7] = imu_gyro_y
        self.obs_buf[:, 8] = imu_gyro_z

        robot_v = self._robots.get_linear_velocities()
        ball_v = self._robots.get_linear_velocities()
        robot_w = self._robots.get_angular_velocities()
        _, robot_orientation = self._robots.get_world_poses()
        self.obs_buf[:, 9] = robot_v[:, 0]
        self.obs_buf[:, 10] = robot_v[:, 1]
        self.obs_buf[:, 11] = robot_v[:, 2]
        self.obs_buf[:, 12] = ball_v[:, 0]
        self.obs_buf[:, 13] = ball_v[:, 1]
        self.obs_buf[:, 14] = ball_v[:, 2]
        self.obs_buf[:, 15] = robot_w[:, 0]
        self.obs_buf[:, 16] = robot_w[:, 1]
        self.obs_buf[:, 17] = robot_w[:, 2]
        self.obs_buf[:, 18] = robot_orientation[:, 0]
        self.obs_buf[:, 19] = robot_orientation[:, 1]
        self.obs_buf[:, 20] = robot_orientation[:, 2]
        self.obs_buf[:, 21] = robot_orientation[:, 3]



        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        # print(observations)
        return observations

    def pre_physics_step(self, actions) -> None:
        torch.nan_to_num(actions, nan=0.0)  # replace NaN with zero
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        self.actions = actions.clone().to(self._device)     # save for later energy calculation

        # actions[:, 0] = actions[:, 0] * self._max_wheel_velocity
        # actions[:, 1] = actions[:, 1] * self._max_wheel_velocity
        # actions[:, 2] = actions[:, 2] * 0.5   # omega_bz

        ## calculate wheel_velocities from wheel's kinematics
        # actions = actions.to(self._device) * self._max_wheel_velocity
        # wheel_velocities = torch.zeros((self._robots.count, 3), dtype=torch.float32, device=self._device)
        # wheel_velocities[:, 0] = -12.8558 * actions[:, 1] - 11.0172 * actions[:, 2]
        # wheel_velocities[:, 1] = 11.1334 * actions[:, 0] + 6.4279 * actions[:, 1] + 8.2664 * actions[:, 2]
        # wheel_velocities[:, 2] = 6.4279 * actions[:, 1] - 11.1334 * actions[:, 0] + 8.2664 * actions[:, 2]

        # wheel_velocities = torch.zeros((self._robots.count, self._robots.num_dof), dtype=torch.float32, device=self._device)
        # wheel_velocities[:, self._wheel_0_dof_idx] = -12.8558 * actions[:, 1] - 11.0172 * actions[:, 2]
        # wheel_velocities[:, self._wheel_1_dof_idx] = 11.1334 * actions[:, 0] + 6.4279 * actions[:, 1] + 8.2664 * actions[:, 2]
        # wheel_velocities[:, self._wheel_2_dof_idx] = 6.4279 * actions[:, 1] - 11.1334 * actions[:, 0] + 8.2664 * actions[:, 2]

        # self.wheel_velocities = wheel_velocities.clone().to(self._device)  # save for later energy calculation

        wheel_effort = torch.zeros((self._robots.count, self._robots.num_dof), dtype=torch.float32, device=self._device)
        wheel_effort[:, self._wheel_0_dof_idx] = actions[:, 0] * self._max_push_effort
        wheel_effort[:, self._wheel_1_dof_idx] = actions[:, 1] * self._max_push_effort
        wheel_effort[:, self._wheel_2_dof_idx] = actions[:, 2] * self._max_push_effort

        ## Apply joint velocities
        from omni.isaac.core.utils.types import ArticulationActions # batched version of ArticulationAction
        stage = omni.usd.get_context().get_stage()
        for env in range(self._num_envs):
            axle_0 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/envs/env_{}/Mooncake/mooncake/base_plate/wheel_0_joint".format(env)), "angular")
            axle_1 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/envs/env_{}/Mooncake/mooncake/base_plate/wheel_1_joint".format(env)), "angular")
            axle_2 = UsdPhysics.DriveAPI.Get(stage.GetPrimAtPath("/World/envs/env_{}/Mooncake/mooncake/base_plate/wheel_2_joint".format(env)), "angular")
            # set_drive_parameters(axle_0, "velocity", math.degrees(wheel_velocities[env, 0]), 0.05, math.radians(1e7))
            # set_drive_parameters(axle_1, "velocity", math.degrees(wheel_velocities[env, 1]), 0.05, math.radians(1e7))
            # set_drive_parameters(axle_2, "velocity", math.degrees(wheel_velocities[env, 2]), 0.05, math.radians(1e7))

            self._robots.apply_action(ArticulationActions(joint_efforts=wheel_effort))
            # self._robots.apply_action(ArticulationActions(joint_velocities=wheel_velocities))

            # self._robots[env].apply_wheel_actions(ArticulationAction(joint_efforts=wheel_effort[env]))

            # set_drive_parameters(axle_0, "effort", wheel_effort[env, 0], 0, math.radians(1e7))
            # set_drive_parameters(axle_1, "effort", wheel_effort[env, 1], 0, math.radians(1e7))
            # set_drive_parameters(axle_2, "effort", wheel_effort[env, 2], 0, math.radians(1e7))

        ## Read IMU & store in buffer ##
        buffer = []
        robots_prim_path = self._robots.prim_paths
        for robot_prim_path in robots_prim_path:
            reading = self._is.get_sensor_readings(robot_prim_path + "/base_plate/sensor") # read from select sensor (by prim_path)
            if reading.shape[0]:
                buffer.append(reading[-1])  # get only lastest reading
            else: buffer.append({"lin_acc_x":0.0, "lin_acc_y":0.0, "lin_acc_z":0.0, "ang_vel_x":0.0, "ang_vel_y":0.0, "ang_vel_z":0.0})  # default initial sensor buffer
        self._imu_buf = buffer
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        ## randomize DOF velocities ##
        # dof_vel = torch_rand_float(-0.1, 0.1, (num_resets, 57), device=self._device)
        # self._robots.set_joint_velocities(dof_vel, indices=env_ids)     # apply resets

        ## Reset Ball positions ##
        ball_pos, ball_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        ball_pos[:, 2] = 0.12   # force ball to touch floor prefectly
        root_vel = torch.zeros((num_resets, 6), device=self._device)
        # Apply Ball position
        self._ball.set_world_poses(ball_pos, ball_rot, indices=env_ids)

        ## Random Ball velocities ##
        ball_vel = torch_rand_float(-0.01, 0.01, (num_resets, 6), device=self._device)
        self._ball.set_velocities(ball_vel, indices=env_ids)

        ## Random Robot positions & orientations ##
        fall_direction = torch_rand_float(-np.pi, np.pi, (num_resets, 1), device=self._device).reshape(-1)
        fall_direction_axis = torch.Tensor([0, 0, 1]).repeat(num_resets, 1).to(self._device).reshape(-1, 3)
        fall_angle = torch_rand_float(0, np.pi/8, (num_resets, 1), device=self._device).reshape(-1)
        fall_angle_axis = torch.Tensor([0, 1, 0]).repeat(num_resets, 1).to(self._device).reshape(-1, 3)
        fall_direction_quat = torch_rot.quat_from_angle_axis(fall_direction, fall_direction_axis)
        fall_angle_quat = torch_rot.quat_from_angle_axis(fall_angle, fall_angle_axis)

        ## Apply Robot position ##
        robot_pos = ball_pos.clone()    # use ball position as reference
        robot_offset = torch.Tensor([0, 0, self._robot_offset]).repeat(num_resets).to(self._device).reshape(-1, 3)  # Distance from ball center to robot center is 18 cm.

        robot_pos = robot_pos + robot_offset
        # robot_pos = robot_pos + torch_rot.quat_rotate(fall_angle_quat, torch_rot.quat_rotate(fall_direction_quat, robot_offset))
        robot_rot = self.initial_root_rot[env_ids]
        # robot_rot = torch_rot.quat_apply(fall_direction_quat, robot_rot)
        # robot_rot = torch_rot.quat_apply(fall_angle_quat, robot_rot)

        # root_pos, root_rot = self.initial_root_pos[env_ids], self.initial_root_rot[env_ids]
        # root_vel = torch.zeros((num_resets, 6), device=self._device)
        #
        self._robots.set_world_poses(robot_pos, robot_rot, indices=env_ids)
        self._robots.set_velocities(root_vel, indices=env_ids)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):   # Run only once after simulation started
        # self._robots = self.get_robot()

        self.initial_root_pos, self.initial_root_rot = self._robots.get_world_poses()   # save initial position for reset
        self.initial_dof_pos = self._robots.get_joint_positions()
        # initialize some data used later on
        # self.start_rotation = torch.tensor([1, 0, 0, 0], device=self._device, dtype=torch.float32)
        # self.up_vec = torch.tensor([0, 0, 1], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        # self.heading_vec = torch.tensor([1, 0, 0], dtype=torch.float32, device=self._device).repeat((self.num_envs, 1))
        # self.inv_start_rot = quat_conjugate(self.start_rotation).repeat((self.num_envs, 1))

        # self.basis_vec0 = self.heading_vec.clone()
        # self.basis_vec1 = self.up_vec.clone()

        self._wheel_0_dof_idx = self._robots.get_dof_index("wheel_0_joint")
        self._wheel_1_dof_idx = self._robots.get_dof_index("wheel_1_joint")
        self._wheel_2_dof_idx = self._robots.get_dof_index("wheel_2_joint")

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:    # calculate reward for each env
        wheel_vel = self.obs_buf[:, :3]
        # if 'self.previous_wheel_vel' in locals():
        #     wheel_accel = wheel_vel - self.previous_wheel_vel
        # else: wheel_accel = torch.zeros_like(wheel_vel)
        # wheel_accel_cost = torch.sum(wheel_accel**2, dim=-1)
        # self.previous_wheel_vel = wheel_vel.clone()


        balls_position, balls_orientation = self._ball.get_world_poses()
        robots_position, robots_orientation = self._robots.get_world_poses()
        robots_omega = self._robots.get_angular_velocities()
        fall_angles = q2falling(robots_orientation) # find fall angle of all robot (batched)
        if self.previous_fall_angle is None: falling_velocity = torch.zeros_like(robots_position[:, 0])
        else: falling_velocity = fall_angles - self.previous_fall_angle

        ## aligning up axis of robot and environment
        # up_proj = torch.cos(fall_angles)
        # up_reward = torch.zeros_like(fall_angles)
        # up_reward = torch.where(up_proj > 0.93, up_reward + self.up_weight, up_reward)
        # falling_penalty = fall_angles
        q1 = self.initial_root_rot  # world frame
        q2 = robots_orientation   # robot orientation
        # find product of quaternions
        product_quaternion = torch.sum(q1*q2,dim=-1) # <q1, q2>
        quaternion_distance = 1 - (product_quaternion**2)
        # print(quaternion_distance)

        ## energy penalty for movement
        # actions_cost = torch.sum(self.wheel_velocities ** 2, dim=-1)
        # electricity_cost = torch.sum(torch.abs(self.actions * obs_buf[:, 12+num_dof:12+num_dof*2])* self.motor_effort_ratio.unsqueeze(0), dim=-1)

        ## rotation penality
        # rotation_cost = torch.sum(torch_rot.quat_diff_rad(robots_orientation, self.initial_root_rot)** 2, dim=-1)


        ## reward for duration of staying alive
        alive_reward = torch.ones_like(fall_angles) * self.alive_reward_scale
        # progress_reward = potentials - prev_potentials
        # print(robots_position - balls_position)
        # print(torch.sum((robots_position - balls_position)**2, dim=-1))
        total_reward = (
            - quaternion_distance
            # - wheel_accel_cost * 0.05
            # - falling_velocity
            + alive_reward
            # + up_reward
            # - math.e**(-0.01*fall_angles)
            # - actions_cost * self.actions_cost_scale
            # - torch.sum(robots_omega**2, dim=-1) * 10
            # - rotation_cost * 10
        )

        # adjust reward for fallen agents
        total_reward = torch.where(
            robots_position[:, 2] < self.termination_height,    # fall by height
            torch.ones_like(total_reward) * self.death_cost,
            total_reward
        )
        total_reward = torch.where(
            fall_angles > 50 / 180 * math.pi,   # fall by angle
            torch.ones_like(total_reward) * self.death_cost,
            total_reward
        )
        total_reward = torch.where(
            torch.sum((robots_position - balls_position)**2, dim=-1) > (self._robot_offset+self._jump_offset)**2,  # jump beyond jump_offset
            torch.ones_like(total_reward) * self.death_cost,
            total_reward
        )

        self.previous_fall_angle = fall_angles
        self.rew_buf[:] = total_reward

    def is_done(self) -> None:  # check termination for each env
        balls_position, balls_orientation = self._ball.get_world_poses()
        robots_position, robots_orientation = self._robots.get_world_poses()
        fall_angles = q2falling(robots_orientation)  # find fall angle of all robot (batched)
        robot_z_position = robots_position[:, 2]
        # print("Z position", robot_z_position)

        resets = torch.zeros(self._num_envs, dtype=torch.long, device=self._device)
        resets = torch.where(torch.sum((robots_position - balls_position)**2, dim=-1) > (self._robot_offset+self._jump_offset)**2, 1, resets) # jump beyond jump_offset
        resets = torch.where(robot_z_position < 0.25, 1, resets)          # reset by falling (Z-position)
        resets = torch.where(fall_angles > 50*(np.pi / 180), 1, resets)      # reset by falling (angle)
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)  # reset by time
        self.reset_buf[:] = resets
