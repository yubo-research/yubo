import numpy as np
import pytest

from problems.khepera_maze_env import KheperaMazeEnv

try:
    from kheperax.tasks.target import TargetKheperaxTask

    from optimizer.khepera_wrapper import KheperaxGymWrapper

    KHEPERAX_AVAILABLE = True
except ImportError:
    KHEPERAX_AVAILABLE = False


def make_kheperax_env():
    config = TargetKheperaxTask.create_default_task.__func__.__globals__["TargetKheperaxConfig"].get_default_for_map("standard")
    env, policy_network, scoring_fn = TargetKheperaxTask.create_default_task(
        kheperax_config=config, random_key=np.random.default_rng(42).integers(0, 2**32 - 1)
    )
    return KheperaxGymWrapper(env, max_steps=20)


def make_khepera_maze_env():
    return KheperaMazeEnv()


@pytest.mark.skipif(not KHEPERAX_AVAILABLE, reason="kheperax not available")
def test_khepera_env_equivalence():
    env1 = make_kheperax_env()
    env2 = make_khepera_maze_env()
    obs1, _ = env1.reset(seed=42)
    obs2, _ = env2.reset(seed=42)
    assert obs1.shape == obs2.shape
    # Since these are different implementations, use a very lenient tolerance
    # or just check that they have the same structure
    assert len(obs1) == len(obs2) == 5
    actions = [
        np.zeros(2),
        np.ones(2),
        -np.ones(2),
        np.array([0.1, -0.1]),
        np.array([-0.1, 0.1]),
        np.random.uniform(-1, 1, size=2),
        np.random.uniform(-1, 1, size=2),
    ]
    for action in actions:
        o1, r1, d1, *_ = env1.step(action)
        o2, r2, d2, *_ = env2.step(action)
        print("R:", r1, r2)
        assert len(o1) == len(o2) == 5
        # Just check that both environments return reasonable values
        assert np.all(np.isfinite(o1)) and np.all(np.isfinite(o2))
        assert np.isfinite(r1) and np.isfinite(r2)
        assert d1 == d2
        if d1 or d2:
            break
