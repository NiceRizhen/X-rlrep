from algos.policy_based.ddpg1 import DDPG
import gym
import gc
gc.enable()

def makeFilteredEnv(env):
  """ crate a new environment class with actions and states normalized to [-1,1] """
  acsp = env.action_space
  obsp = env.observation_space
  if not type(acsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous action space (i.e. Box) required.')
  if not type(obsp)==gym.spaces.box.Box:
    raise RuntimeError('Environment with continous observation space (i.e. Box) required.')

  env_type = type(env)

  class FilteredEnv(env_type):
    def __init__(self):
      self.__dict__.update(env.__dict__) # transfer properties

      # Observation space
      if np.any(obsp.high < 1e10):
        h = obsp.high
        l = obsp.low
        sc = h-l
        self.o_c = (h+l)/2.
        self.o_sc = sc / 2.
      else:
        self.o_c = np.zeros_like(obsp.high)
        self.o_sc = np.ones_like(obsp.high)

      # Action space
      h = acsp.high
      l = acsp.low
      sc = (h-l)
      self.a_c = (h+l)/2.
      self.a_sc = sc / 2.

      # Rewards
      self.r_sc = 0.1
      self.r_c = 0.

ENV_NAME = 'InvertedPendulum-v1'
env = makeFilteredEnv(gym.make(ENV_NAME))

algo = DDPG(
    env=env
)
algo.train()
