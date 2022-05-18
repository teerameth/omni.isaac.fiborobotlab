from typing import Optional, Tuple
import numpy as np
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import find_nucleus_server
from omni.isaac.core.utils.types import ArticulationAction
from omni.isaac.core.utils.prims import get_prim_at_path, define_prim
import carb

class MoonCake(Robot):
    """[summary]

    Args:
        stage (Usd.Stage): [description]
        prim_path (str): [description]
        name (str): [description]
        usd_path (str, optional): [description]
        position (Optional[np.ndarray], optional): [description]. Defaults to None.
        orientation (Optional[np.ndarray], optional): [description]. Defaults to None.
    """

    def __init__(
        self,
        prim_path: str,
        name: str = "mooncake",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        prim = get_prim_at_path(prim_path)
        if not prim.IsValid():
            prim = define_prim(prim_path, "Xform")
            if usd_path:
                prim.GetReferences().AddReference(usd_path)
            else:
                result, nucleus_server = find_nucleus_server()
                if result is False:
                    carb.log_error("Could not find nucleus server with /Isaac folder")
                    return
                asset_path = nucleus_server + "/Library/mooncake.usd" # load from nucleus server
                prim.GetReferences().AddReference(asset_path)
        super().__init__(
            prim_path=prim_path, name=name, position=position, orientation=orientation, articulation_controller=None
        )
        self._wheel_dof_names = ["wheel_0_joint", "wheel_1_joint", "wheel_2_joint"]
        self._wheel_dof_indices = None
        return
    @property
    def wheel_dof_indicies(self) -> Tuple[int, int, int]:
        """[summary]

        Returns:
            int: [description]
        """
        return self._wheel_dof_indices
    
    def get_wheel_positions(self) -> Tuple[float, float, float]:
        """[summary]

        Returns:
            Tuple[float, float, float]: [description]
        """
        joint_positions = self.get_joint_positions()
        return joint_positions[self._wheel_dof_indices[0]], joint_positions[self._wheel_dof_indices[1]], joint_positions[self._wheel_dof_indices[2]]
    
    def set_wheel_positions(self, positions: Tuple[float, float]) -> None:
        """[summary]

        Args:
            positions (Tuple[float, float, float]): [description]
        """
        joint_positions = [None, None]
        joint_positions[self._wheel_dof_indices[0]] = positions[0]
        joint_positions[self._wheel_dof_indices[1]] = positions[1]
        self.set_joint_positions(positions=np.array(joint_positions))
        return

    def get_wheel_velocities(self) -> Tuple[float, float, float]:
        """[summary]

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: [description]
        """
        joint_velocities = self.get_joint_velocities()
        return joint_velocities[self._wheel_dof_indices[0]], joint_velocities[self._wheel_dof_indices[1]], joint_velocities[self._wheel_dof_indices[2]]

    def set_wheel_velocities(self, velocities: Tuple[float, float, float]) -> None:
        """[summary]

        Args:
            velocities (Tuple[float, float, float]): [description]
        """
        joint_velocities = [None, None]
        joint_velocities[self._wheel_dof_indices[0]] = velocities[0]
        joint_velocities[self._wheel_dof_indices[1]] = velocities[1]
        joint_velocities[self._wheel_dof_indices[2]] = velocities[2]
        self.set_joint_velocities(velocities=np.array(joint_velocities))
        return
    
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

    def initialize(self) -> None:
        """[summary]
        """
        super().initialize()
        # print(self._dofs_infos)   # print Orderdict of all dof_name:dof_object
        self._wheel_dof_indices = (
            self.get_dof_index(self._wheel_dof_names[0]),
            self.get_dof_index(self._wheel_dof_names[1]),
            self.get_dof_index(self._wheel_dof_names[2]),
        )
        return
    
    def post_reset(self) -> None:
        """[summary]
        """
        super().post_reset()
        # print(len(self._articulation_controller._dof_controllers))
        # print(self._articulation_controller._dof_controllers)
        # Assign kd only for driven acticulation (3 wheels) and leave other as None
        kds = [None]*len(self._articulation_controller._dof_controllers)
        for i in self._wheel_dof_indices: kds[i] = 1e2
        self._articulation_controller.set_gains(kds=kds)
        self._articulation_controller.switch_control_mode(mode="velocity")
        return