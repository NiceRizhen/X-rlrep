from algos.policy_based.ppo_1 import PPO
import gym

env_name = 'Pendulum-v0'
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
    s_dim=3,
    a_dim=1,
)

algo.train()
