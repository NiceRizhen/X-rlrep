from envs.mujoco.gather.gather_env import GatherEnv
from envs.mujoco.swimmer_env import SwimmerEnv


class SwimmerGatherEnv(GatherEnv):

    MODEL_CLASS = SwimmerEnv
    ORI_IND = 2
