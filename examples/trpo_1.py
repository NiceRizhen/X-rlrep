from algos.policy_based.trpo_1 import TRPO, TRPOAgent
import gym

env_name = 'CartPole-v0'
env = gym.make(env_name)
'''
agent = TRPOAgent(
    env=env,
    max_steps=1000,
    episodes_per_roll=1000,
    gamma=0.95,
    cg_damping=0.1,
    max_kl=0.01,
)
agent.learn()
'''
algo = TRPO(
    env=env,
    max_steps=1000,
    episodes_per_roll=1000,
    gamma=0.95,
    cg_damping=0.1,
    max_kl=0.01
)

algo.train()
