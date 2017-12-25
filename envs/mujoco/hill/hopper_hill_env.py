import numpy as np

from envs.mujoco.hill.hill_env import HillEnv
from envs.mujoco.hopper_env import HopperEnv
from misc.overrides import overrides
import envs.mujoco.hill.terrain as terrain
from spaces import Box

class HopperHillEnv(HillEnv):

    MODEL_CLASS = HopperEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(hfield, Box(np.array([-1.0, -1.0]), np.array([-0.5, -0.5])))
