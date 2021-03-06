from algos.policy_based.trpo import TRPO
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)
algo = TRPO(
    env=env,
    max_steps=100,
    episodes_per_roll=1000,
    gamma=0.95,
    cg_damping=0.1,
    max_kl=0.01
)

algo.train()
