from algos.value_based.prioritized_dqn import PrioritizedReplayDQN
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)
env = env.unwrapped
algo = PrioritizedReplayDQN(
    env=env,
    memory_size=10000,
    learning_rate=0.005,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=500,
    memory_size_agent=10000,
    batch_size=32,
)
algo.train()
