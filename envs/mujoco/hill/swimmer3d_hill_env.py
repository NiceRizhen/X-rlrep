import numpy as np

from envs.mujoco.hill.hill_env import HillEnv
from envs.mujoco.swimmer3d_env import Swimmer3DEnv
from misc.overrides import overrides
import envs.mujoco.hill.terrain as terrain
from spaces import Box

class Swimmer3DHillEnv(HillEnv):

    MODEL_CLASS = Swimmer3DEnv

    @overrides
    def _mod_hfield(self, hfield):
        # clear a flat patch for the robot to start off from
        return terrain.clear_patch(hfield, Box(np.array([-3.0, -1.5]), np.array([0.0, -0.5])))
