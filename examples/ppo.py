from algos.policy_based.ppo import PPO
import gym

env_name = 'MountainCarContinuous-v0'
env = gym.make(env_name)

algo = PPO(
    env=env,
    ep_max=1000,
    ep_len=200,
    gamma=0.9,
    a_lr=0.0001,
    c_lr=0.0002,
    batch=32,
    a_update_steps=10,
    c_update_steps=10,
    s_dim=env.observation_space.shape[0],
    a_dim=env.action_space.shape[0],
    # a_dim=env.action_space.n
)

algo.train()
