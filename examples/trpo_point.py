import sys
sys.path.append("..")
from algos.policy_based.trpo import TRPO
from baselines.linear_feature_baseline import LinearFeatureBaseline
from point_env import PointEnv
from envs.normalized_env import normalize
from policies.gaussian_mlp_policy import GaussianMLPPolicy

env = normalize(PointEnv())
policy = GaussianMLPPolicy(
    env_spec=env.spec,
)
baseline = LinearFeatureBaseline(env_spec=env.spec)
algo = TRPO(
    env=env,
    policy=policy,
    baseline=baseline,
)
algo.train()
