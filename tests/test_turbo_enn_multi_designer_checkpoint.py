import numpy as np

from optimizer.datum import Datum
from optimizer.multi_turbo_enn_designer import MultiTurboENNConfig, MultiTurboENNDesigner
from optimizer.multi_turbo_enn_utils import (
    call_multi_designer,
    load_multi_state,
    score_multi_candidates,
)
from optimizer.trajectories import Trajectory
from problems.env_conf import default_policy, get_env_conf


def _make_datum(policy, rreturn):
    traj = Trajectory(rreturn=float(rreturn), states=np.empty((0,)), actions=np.empty((0,)))
    return Datum(None, policy, None, traj)


def test_turbo_enn_multi_resume_determinism():
    np.random.seed(123)
    env_conf = get_env_conf("f:ackley-2d", problem_seed=0, noise_seed_0=0)
    base_policy = default_policy(env_conf)
    num_params = base_policy.num_params()

    data = []
    for i in range(3):
        p = base_policy.clone()
        params = np.linspace(-0.5, 0.5, num_params, dtype=np.float32) + 0.01 * i
        p.set_params(params)
        data.append(_make_datum(p, rreturn=float(i)))

    config = MultiTurboENNConfig(
        turbo_mode="turbo-enn",
        num_regions=2,
        strategy="independent",
        arm_mode="allocated",
        pool_multiplier=2,
        num_init=2,
        k=5,
        num_keep=5,
        num_fit_samples=16,
        num_fit_candidates=16,
        acq_type="thompson",
        tr_geometry="enn_ellipsoid",
        ellipsoid_sampler="full",
    )
    designer1 = MultiTurboENNDesigner(base_policy, config=config)
    num_arms = 2
    proposals1 = call_multi_designer(designer1, data, num_arms=num_arms)

    new_data = []
    for i, p in enumerate(proposals1):
        new_data.append(_make_datum(p, rreturn=10.0 + i))
    data2 = data + new_data

    state = designer1.state_dict(data=data)
    proposals2 = call_multi_designer(designer1, data2, num_arms=num_arms)

    assert designer1._last_region_indices is not None
    x_all = np.array([p.get_params() for p in proposals2], dtype=float)
    scores = score_multi_candidates(designer1, x_all, designer1._last_region_indices)
    assert scores.shape == (len(proposals2),)

    np.random.seed(123)
    base_policy2 = default_policy(env_conf)
    designer2 = MultiTurboENNDesigner(base_policy2, config=config)
    load_multi_state(designer2, state, data)
    proposals2_resume = call_multi_designer(designer2, data2, num_arms=num_arms)

    assert len(proposals2) == len(proposals2_resume)
    for p_a, p_b in zip(proposals2, proposals2_resume, strict=True):
        np.testing.assert_allclose(p_a.get_params(), p_b.get_params(), rtol=0.0, atol=1e-6)
