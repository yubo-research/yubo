import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from kheperax.tasks.target import TargetKheperaxConfig
from qdax.custom_types import RNGKey

from optimizer.khepera_wrapper import KheperaxEnvConf
from optimizer.trajectories import collect_trajectory
from problems.env_conf import get_env_conf


def test_kheperax_wrapper():
    # Prepare config and random key
    kheperax_config = TargetKheperaxConfig.get_default_for_map("standard")
    random_key = RNGKey()

    # Create the wrapper
    env_conf = KheperaxEnvConf(kheperax_config=kheperax_config, random_key=random_key, max_steps=100, noise_seed_0=42)

    # Create a simple policy for testing
    # We need to get the action size from the env, so call make()
    env = env_conf.make()
    action_size = env.action_space.action_size

    class SimplePolicy:
        def __init__(self, action_size):
            self.action_size = action_size

        def __call__(self, state):
            # Return random actions in [-1, 1]
            return np.random.uniform(-1, 1, self.action_size)

    # Create policy
    policy = SimplePolicy(action_size)

    # Test collect_trajectory
    try:
        trajectory = collect_trajectory(env_conf, policy, noise_seed=123)
        print("SUCCESS: collect_trajectory worked!")
        print(f"Return: {trajectory.rreturn}")
        print(f"States shape: {trajectory.states.shape}")
        print(f"Actions shape: {trajectory.actions.shape}")
        return True
    except Exception as e:
        print(f"ERROR: collect_trajectory failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_kheperax_wrapper_with_env_conf():
    # Use the env_conf factory to get the KheperaxEnvConf
    env_conf = get_env_conf("khepera")
    env = env_conf.make()
    action_size = env.action_space.action_size

    class SimplePolicy:
        def __init__(self, action_size):
            self.action_size = action_size

        def __call__(self, state):
            return np.random.uniform(-1, 1, self.action_size)

    policy = SimplePolicy(action_size)
    try:
        trajectory = collect_trajectory(env_conf, policy, noise_seed=456)
        print("SUCCESS: collect_trajectory with get_env_conf worked!")
        print(f"Return: {trajectory.rreturn}")
        print(f"States shape: {trajectory.states.shape}")
        print(f"Actions shape: {trajectory.actions.shape}")
        return True
    except Exception as e:
        print(f"ERROR: collect_trajectory with get_env_conf failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    test_kheperax_wrapper()
    test_kheperax_wrapper_with_env_conf()
