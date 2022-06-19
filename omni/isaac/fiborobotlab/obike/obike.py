from typing import Optional
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_server_path
from omni.isaac.core.utils.stage import add_reference_to_stage

import carb

class Obike(Robot):
    def __init__(
            self,
            prim_path: str,
            name: Optional[str] = "Obike",
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
            self._usd_path = server_path + "/Library/obike.usd"

        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )