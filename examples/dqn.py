from algos.value_based.dqn import DQN
import gym

env_name = 'MountainCar-v0'
env = gym.make(env_name)

algo = DQN(
    env=env,
    memory_size=10000,
    action_space=env.action_space.n,
    n_features=env.observation_space.shape[0],
    learning_rate=0.001,
    reward_decay=0.9,
    e_greedy=0.9,
    replace_target_iter=500,
    memory_size_2=10000,
    batch_size=32,
)
algo.train()
