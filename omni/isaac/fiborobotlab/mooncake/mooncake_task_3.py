from omniisaacgymenvs.tasks.base.rl_task import RLTask
from mooncake import Mooncake

import omni
from pxr import UsdPhysics, Gf, UsdGeom
from omni.isaac.core.articulations import ArticulationView
from omni.isaac.core.prims import RigidPrimView
from omni.isaac.core.utils.prims import get_prim_at_path
import omni.isaac.core.utils.torch.rotations as torch_rot
from omni.isaac.isaac_sensor import _isaac_sensor
from omni.isaac.core.utils.torch.maths import torch_rand_float, tensor_clamp, unscale

import numpy as np
import torch
import torch.nn.functional as f
import math

class MooncakeTask(RLTask):