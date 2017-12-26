from algos.policy_based.tnpg_1 import TNPG
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

algo = TNPG(
    env=env,
    n_actions=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.02,
    reward_decay=0.99,
)
algo.train()
