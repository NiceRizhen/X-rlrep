from algos.policy_based.trpo import TRPO
from baselines.linear_feature_baseline import LinearFeatureBaseline
from envs.box2d.cartpole_env import CartpoleEnv
from envs.normalized_env import normalize
from policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(CartpoleEnv())

policy = GaussianMLPPolicy(
    env_spec=env.spec,
    # The neural network policy should have two hidden layers, each with 32 hidden units.
    hidden_sizes=(32, 32)
)

baseline = LinearFeatureBaseline(env_spec=env.spec)

algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
    batch_size=4000,
    max_path_length=100,
    n_itr=40,
    discount=0.99,
    step_size=0.01,
)
algo.train()
