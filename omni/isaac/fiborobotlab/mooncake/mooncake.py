from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_server_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omni.isaac.core.utils.types import ArticulationAction

import carb

class Mooncake(Robot):
    def __init__(
            self,
            prim_path: str,
            name: Optional[str] = "Mooncake",
            usd_path: Optional[str] = None,
            translation: Optional[np.ndarray] = None,
            orientation: Optional[np.ndarray] = None,
    ) -> None:
        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            server_path = get_server_path()
            if server_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            self._usd_path = server_path + "/Library/mooncake.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        self._wheel_dof_indices = [self.get_dof_index("wheel_0_joint"),
                                   self.get_dof_index("wheel_1_joint"),
                                   self.get_dof_index("wheel_2_joint")]

    def apply_wheel_actions(self, actions: ArticulationAction) -> None:
        """[summary]

        Args:
            actions (ArticulationAction): [description]
        """
        actions_length = actions.get_length()
        if actions_length is not None and actions_length != 3:
            raise Exception("ArticulationAction passed should be equal to 3")
        joint_actions = ArticulationAction()
        if actions.joint_positions is not None:
            joint_actions.joint_positions = np.zeros(self.num_dof)
            joint_actions.joint_positions[self._wheel_dof_indices[0]] = actions.joint_positions[0]
            joint_actions.joint_positions[self._wheel_dof_indices[1]] = actions.joint_positions[1]
            joint_actions.joint_positions[self._wheel_dof_indices[2]] = actions.joint_positions[2]
        if actions.joint_velocities is not None:
            joint_actions.joint_velocities = np.zeros(self.num_dof)
            joint_actions.joint_velocities[self._wheel_dof_indices[0]] = actions.joint_velocities[0]
            joint_actions.joint_velocities[self._wheel_dof_indices[1]] = actions.joint_velocities[1]
            joint_actions.joint_velocities[self._wheel_dof_indices[2]] = actions.joint_velocities[2]
        if actions.joint_efforts is not None:
            joint_actions.joint_efforts = np.zeros(self.num_dof)
            joint_actions.joint_efforts[self._wheel_dof_indices[0]] = actions.joint_efforts[0]
            joint_actions.joint_efforts[self._wheel_dof_indices[1]] = actions.joint_efforts[1]
            joint_actions.joint_efforts[self._wheel_dof_indices[2]] = actions.joint_efforts[2]

        self.apply_action(control_actions=joint_actions)
        return
# class Ball(Robot):
#     def __init__(
#             self,
#             prim_path: str,
#             name: Optional[str] = "Ball",
#             usd_path: Optional[str] = None,
#             translation: Optional[np.ndarray] = None,
#             orientation: Optional[np.ndarray] = None,
#     ) -> None:
#         self._usd_path = usd_path
#         self._name = name
#
#         if self._usd_path is None:
#             server_path = get_server_path()
#             if server_path is None:
#                 carb.log_error("Could not find Isaac Sim assets folder")
#             self._usd_path = server_path + "/Library/ball.usd"
#
#         add_reference_to_stage(self._usd_path, prim_path)
#
#         super().__init__(
#             prim_path=prim_path,
#             name=name,
#             translation=translation,
#             orientation=orientation,
#             articulation_controller=None,
#         )