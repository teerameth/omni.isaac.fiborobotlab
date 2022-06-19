from omniisaacgymenvs.tasks.base.rl_task import RLTask
from obike import Obike

import omni
from pxr import Gf, UsdGeom
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.utils.prims import get_prim_at_path, get_all_matching_child_prims
from omni.isaac.isaac_sensor import _isaac_sensor

import numpy as np
import torch
import math


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
        self._robot_positions = torch.tensor([0.0, 0.0, 2.0])

        self._reset_dist = self._task_cfg["env"]["resetDist"]
        self._max_push_effort = self._task_cfg["env"]["maxEffort"]
        self._max_episode_length = 500

        self._num_observations = 4
        self._num_actions = 1

        self._imu_buf = {"lin_acc_x":0.0, "lin_acc_y":0.0, "lin_acc_z":0.0, "ang_vel_x":0.0, "ang_vel_y":0.0, "ang_vel_z":0.0}  # default initial sensor buffer
        RLTask.__init__(self, name, env)
        return

    def set_up_scene(self, scene) -> None:
        self.get_robot()
        super().set_up_scene(scene)
        self._robots = ArticulationView(prim_paths_expr="/World/envs/.*/Obike", name="obike_view")
        scene.add(self._robots)
        self.meters_per_unit = UsdGeom.GetStageMetersPerUnit(omni.usd.get_context().get_stage())
        return

    def get_robot(self):
        robot = Obike(prim_path=self.default_zero_env_path + "/Obike", name="Obike", translation=self._robot_positions)
        # applies articulation settings from the task configuration yaml file
        self._sim_config.apply_articulation_settings("Obike", get_prim_at_path(robot.prim_path), self._sim_config.parse_actor_config("Obike"))
        ## Attact IMU sensor ##
        self._is = _isaac_sensor.acquire_imu_sensor_interface()
        self.body_path = "/World/envs/env_0/Obike/chassic"
        print("AAAAAAAAAAAAAAA")
        print(get_all_matching_child_prims(prim_path="/"))
        result, sensor = omni.kit.commands.execute(
            "IsaacSensorCreateImuSensor",
            path="/sensor",
            parent=self.body_path,
            sensor_period=1 / 500.0,
            offset=Gf.Vec3d(0, 0, 10),
            orientation=Gf.Quatd(1, 0, 0, 0),
            visualize=True,
        )

    def get_observations(self) -> dict:
        # dof_pos = self._robots.get_joint_positions(clone=False)
        dof_vel = self._robots.get_joint_velocities(clone=False)


        reaction_vel = dof_vel[:, self._reaction_wheel_dof_idx]
        imu_accel_y = self._imu_buf["lin_acc_y"]
        imu_accel_z = self._imu_buf["lin_acc_z"]
        imu_gyro_x = self._imu_buf["ang_vel_x"]

        self.obs_buf[:, 0] = reaction_vel
        self.obs_buf[:, 1] = imu_accel_y
        self.obs_buf[:, 2] = imu_accel_z
        self.obs_buf[:, 3] = imu_gyro_x

        observations = {
            self._robots.name: {
                "obs_buf": self.obs_buf
            }
        }
        return observations

    def pre_physics_step(self, actions) -> None:
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(reset_env_ids) > 0:
            self.reset_idx(reset_env_ids)

        actions = actions.to(self._device)

        forces = torch.zeros((self._robots.count, self._robots.num_dof), dtype=torch.float32, device=self._device)
        forces[:, self._reaction_wheel_dof_idx] = self._max_push_effort * actions[:, 0]

        indices = torch.arange(self._robots.count, dtype=torch.int32, device=self._device)
        self._robots.set_joint_efforts(forces, indices=indices)

        ## Read IMU & store in buffer ##
        reading = self._is.get_sensor_readings(self.body_path + "/sensor")
        if reading.shape[0]: self._imu_buf = reading[-1] # get only lastest reading
    def reset_idx(self, env_ids):
        num_resets = len(env_ids)

        ## RANDOM BY DOF ##
        # randomize DOF positions
        # dof_pos = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
        # dof_pos[:, self._cart_dof_idx] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_pos[:, self._pole_dof_idx] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # randomize DOF velocities
        # dof_vel = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
        # dof_vel[:, self._cart_dof_idx] = 0.5 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # dof_vel[:, self._pole_dof_idx] = 0.25 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        ## RANDOM BY POSE ##
        # random positions
        rand_translations = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
        rand_translations[:, 0] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        rand_translations[:, 1] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        rand_translations[:, 2] = 1.0 * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        # random orientations (x, y, z, w)
        rand_orientations = torch.zeros((num_resets, self._robots.num_dof), device=self._device)
        rand_orientations[:, 0] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # rand_orientations[:, 1] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # rand_orientations[:, 2] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))
        # rand_orientations[:, 3] = 0.125 * math.pi * (1.0 - 2.0 * torch.rand(num_resets, device=self._device))

        ## Apply resets ##
        indices = env_ids.to(dtype=torch.int32)
        ## BY DOF
        # self._robots.set_joint_positions(dof_pos, indices=indices)
        # self._robots.set_joint_velocities(dof_vel, indices=indices)
        ## BY POSE
        self._robots.set_local_poses(translations=rand_translations, orientations=rand_orientations, indices=indices)

        # bookkeeping
        self.reset_buf[env_ids] = 0
        self.progress_buf[env_ids] = 0


    def post_reset(self):
        self._reaction_wheel_dof_idx = self._robots.get_dof_index("reaction_wheel_joint")
        self._rear_wheel_dof_idx = self._robots.get_dof_index("rear_wheel_joint")
        self._steering_arm_dof_idx = self._robots.get_dof_index("front_wheel_arm_joint")

        # randomize all envs
        indices = torch.arange(self._robots.count, dtype=torch.int64, device=self._device)
        self.reset_idx(indices)

    def calculate_metrics(self) -> None:
        reaction_vel = self.obs_buf[:, 0]
        robots_position, robots_orientation = self._robots.get_world_poses()
        fall_angle = robots_orientation[0]

        # cart_pos = self.obs_buf[:, 0]
        # cart_vel = self.obs_buf[:, 1]
        # pole_angle = self.obs_buf[:, 2]
        # pole_vel = self.obs_buf[:, 3]

        # reward = 1.0 - pole_angle * pole_angle - 0.01 * torch.abs(cart_vel) - 0.005 * torch.abs(pole_vel)
        # reward = torch.where(torch.abs(cart_pos) > self._reset_dist, torch.ones_like(reward) * -2.0, reward)
        # reward = torch.where(torch.abs(pole_angle) > np.pi / 2, torch.ones_like(reward) * -2.0, reward)
        reward = 1.0 - fall_angle * fall_angle - 0.01 * torch.abs(reaction_vel)
        reward = torch.where(torch.abs(fall_angle) > 25*(np.pi / 180), torch.ones_like(reward) * -2.0, reward)   # fall_angle must <= 25 degree

        self.rew_buf[:] = reward

    def is_done(self) -> None:
        robots_position, robots_orientation = self._robots.get_world_poses()
        fall_angle = robots_orientation[0]
        # cart_pos = self.obs_buf[:, 0]
        # pole_pos = self.obs_buf[:, 2]

        # resets = torch.where(torch.abs(cart_pos) > self._reset_dist, 1, 0)
        # resets = torch.where(torch.abs(pole_pos) > math.pi / 2, 1, resets)
        # resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)

        resets = torch.where(torch.abs(fall_angle) > 25*(np.pi / 180), 1, 0)            # reset by falling
        resets = torch.where(self.progress_buf >= self._max_episode_length, 1, resets)  # reset by time

        self.reset_buf[:] = resets
