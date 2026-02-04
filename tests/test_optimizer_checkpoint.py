import numpy as np
import torch
import torch.nn as nn

from common.collector import Collector
from optimizer.checkpointing import (
    apply_state_dict,
    build_state_dict,
    load_optimizer_checkpoint,
    save_optimizer_checkpoint,
)
from optimizer.datum import Datum
from optimizer.optimizer import Optimizer, _TraceEntry
from optimizer.trajectories import Trajectory
from problems.policy_mixin import PolicyParamsMixin


class DummyEnvConf:
    env_name = "dummy"
    problem_seed = 0
    noise_seed_0 = 0
    frozen_noise = True


class DummyPolicy(PolicyParamsMixin, nn.Module):
    def __init__(self):
        super().__init__()
        self._const_scale = 0.5
        self.linear = nn.Linear(2, 1, bias=False)
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])


def test_optimizer_checkpoint_roundtrip(tmp_path):
    env_conf = DummyEnvConf()
    policy = DummyPolicy()
    collector = Collector()
    opt = Optimizer(
        collector,
        env_conf=env_conf,
        env_tag="dummy",
        policy=policy,
        num_arms=1,
    )

    target_params = np.full((policy.num_params(),), 0.1, dtype=np.float32)
    policy.set_params(target_params)
    traj = Trajectory(rreturn=1.0, states=np.empty((0,)), actions=np.empty((0,)))
    datum = Datum(None, policy.clone(), None, traj)

    opt._data = [datum]
    opt.best_policy = policy.clone()
    opt.best_datum = datum
    opt.r_best_est = 1.0
    opt.y_best = 1.0
    opt._i_iter = 1
    opt._i_noise = 3
    opt._cum_dt_proposing = 0.5
    opt._trace = [_TraceEntry(1.0, 1.0, 0.1, 0.2)]

    ckpt_path = tmp_path / "opt_ckpt.npz"
    state = build_state_dict(opt, designer_name="random")
    assert isinstance(state, dict)
    save_optimizer_checkpoint(opt, str(ckpt_path), designer_name="random")

    opt2 = Optimizer(
        Collector(),
        env_conf=env_conf,
        env_tag="dummy",
        policy=DummyPolicy(),
        num_arms=1,
    )
    loaded = load_optimizer_checkpoint(str(ckpt_path))
    apply_state_dict(opt2, loaded, designer_name="random")

    assert opt2._i_iter == 1
    assert opt2._i_noise == 3
    assert len(opt2._data) == 1
    np.testing.assert_allclose(opt2._data[0].policy.get_params(), target_params, atol=1e-6)
    assert float(opt2.r_best_est) == 1.0
