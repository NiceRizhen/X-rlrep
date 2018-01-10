from algos.policy_based.ddpg import DDPG
import gym

env_name = 'BipedalWalkerHardcore-v2'
env = gym.make(env_name)
env = env.unwrapped
env.seed(1)

algo = DDPG(
    env=env,
    s_dim = env.observation_space.shape[0],
    a_dim = env.action_space.shape[0],
    a_bound = env.action_space.high,
    max_episodes=100,
    max_ep_steps=2000,
    lr_a=0.001,
    lr_c=0.002,
    gamma=0.9,
    tau=0.01,
    memory_capacity=10000,
    batch_size=32
)
algo.train()
