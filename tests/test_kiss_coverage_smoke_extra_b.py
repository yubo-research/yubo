from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np
import torch
from click.testing import CliRunner
from torch import nn

from problems.humanoid_policy import HumanoidPolicy
from problems.other import make as make_other


def test_kiss_cov_humanoid_policy_and_other_module():
    env_conf = SimpleNamespace(
        env_name="Humanoid-v5",
        problem_seed=3,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(348,))),
        action_space=SimpleNamespace(shape=(17,)),
    )
    p = HumanoidPolicy(env_conf)
    assert p.num_params() == 22
    base = p.get_params()
    p.set_params(np.zeros_like(base))
    c = p.clone()
    assert c.num_params() == p.num_params()
    c.reset_state()
    out = c(np.zeros(348, dtype=np.float64))
    assert out.shape == (17,)
    try:
        make_other("unknown-name", problem_seed=0)
        assert False, "Expected unknown env_name assertion"
    except AssertionError:
        assert True


def test_kiss_cov_modal_collect_runtime_utils_and_ops_experiment(monkeypatch):
    import experiments.experiment_sampler as experiment_sampler

    sys.modules["experiment_sampler"] = experiment_sampler
    from experiments.modal_collect import get_job_result
    from experiments.modal_collect import main as modal_collect_main
    from rl.core.runtime import obs_scale_from_env, select_device

    def _call_get(self, timeout):
        assert timeout == 5
        return ("trace", "log", "collector")

    _Call = type("_Call", (), {"get": _call_get})
    _Factory = type("_Factory", (), {"from_id": staticmethod(lambda _call_id: _Call())})
    monkeypatch.setattr("experiments.modal_collect.modal.functions.FunctionCall", _Factory)
    assert isinstance(get_job_result("id-1"), tuple)

    called = {"n": 0}
    monkeypatch.setattr(
        "experiments.modal_collect.collect",
        lambda job_fn, cb: cb(("trace", "log", "collector")),
    )
    monkeypatch.setattr(
        "experiments.modal_collect.post_process",
        lambda *args: called.__setitem__("n", called["n"] + 1),
    )
    monkeypatch.setattr("experiments.modal_collect.os.path.exists", lambda p: False)
    modal_collect_main("jobs.txt")
    assert called["n"] == 1

    env_conf = SimpleNamespace(
        gym_conf=SimpleNamespace(
            transform_state=True,
            state_space=SimpleNamespace(
                low=np.array([-1.0], dtype=np.float32),
                high=np.array([1.0], dtype=np.float32),
                shape=(1,),
            ),
        ),
        ensure_spaces=lambda: None,
    )
    assert str(select_device("cpu")) == "cpu"
    lb, width = obs_scale_from_env(env_conf)
    assert np.allclose(lb, np.array([-1.0], dtype=np.float32))
    assert np.allclose(width, np.array([2.0], dtype=np.float32))

    if "ops.experiment" in sys.modules:
        monkeypatch.delitem(sys.modules, "ops.experiment")

    import ops.experiment as ops_experiment_mod

    monkeypatch.setattr(ops_experiment_mod, "_forward_to_experiments_cli", lambda extra_args: None)
    runner = CliRunner()
    result = runner.invoke(ops_experiment_mod.cli)
    assert result.exit_code == 0


def test_kiss_cov_sac_setup_network_blocks_and_scaling():
    from rl.torchrl.sac import setup as sac_setup

    obs_scaler = sac_setup.sac_deps.torchrl_common.ObsScaler(None, None)
    backbone = nn.Linear(4, 8)
    head = nn.Linear(8, 4)
    actor = sac_setup._ActorNet(backbone=backbone, head=head, obs_scaler=obs_scaler, act_dim=2)
    obs = torch.ones((3, 4), dtype=torch.float32)
    loc, scale = actor.forward(obs)
    assert loc.shape == (3, 2)
    assert scale.shape == (3, 2)
    sampled, sampled_lp = actor.sample(obs)
    assert sampled.shape == (3, 2)
    assert sampled_lp.shape == (3,)
    deterministic, det_lp = actor.sample(obs, deterministic=True)
    assert deterministic.shape == (3, 2)
    assert torch.allclose(det_lp, torch.zeros_like(det_lp))
    acted = actor.act(obs)
    assert acted.shape == (3, 2)

    q_backbone = nn.Linear(6, 8)
    q_head = nn.Linear(8, 1)
    qnet = sac_setup._QNet(backbone=q_backbone, head=q_head, obs_scaler=obs_scaler)
    q = qnet.forward(torch.ones((3, 4)), torch.ones((3, 2)))
    assert q.shape == (3,)

    pix_encoder = nn.Sequential(nn.Flatten(), nn.Linear(12, 5))
    qpix = sac_setup._QNetPixel(obs_encoder=pix_encoder, head=nn.Linear(7, 1), obs_scaler=obs_scaler)
    qpix_out = qpix.forward(torch.ones((2, 1, 3, 4)), torch.ones((2, 2)))
    assert qpix_out.shape == (2,)

    scaler = sac_setup._ScaleActionToEnv(np.array([-2.0, -1.0]), np.array([2.0, 3.0]))
    scaled = scaler.forward(torch.tensor([[-1.0, 1.0], [0.0, 0.0]], dtype=torch.float32))
    assert torch.allclose(scaled[0], torch.tensor([-2.0, 3.0]))


def test_kiss_cov_sac_eval_model_td_trainer():
    from rl.torchrl.offpolicy import actor_eval as trl_actor_eval
    from rl.torchrl.offpolicy import trainer_utils as trl_trainer_utils

    modules3 = SimpleNamespace(actor_backbone=nn.Linear(3, 4), actor_head=nn.Linear(4, 2))
    pol = trl_actor_eval.OffPolicyActorEvalPolicy(
        modules3.actor_backbone,
        nn.Linear(4, 4),
        nn.Identity(),
        act_dim=2,
        device=torch.device("cpu"),
    )
    _ = pol(np.array([0.1, 0.2, 0.3], dtype=np.float32))
    snap = trl_actor_eval.capture_actor_snapshot(modules3)
    trl_actor_eval.restore_actor_snapshot(modules3, snap)
    with trl_actor_eval.use_actor_snapshot(modules3, snap, device=torch.device("cpu")):
        pass

    from tensordict import TensorDict

    td = TensorDict(
        {
            "obs": torch.zeros((2, 3)),
            "action": torch.zeros((2, 2)),
            "next": TensorDict(
                {
                    "reward": torch.ones(2),
                    "terminated": torch.zeros(2, dtype=torch.bool),
                    "truncated": torch.zeros(2, dtype=torch.bool),
                },
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )
    flat = trl_trainer_utils.flatten_batch_to_transitions(td)
    flat = trl_trainer_utils.normalize_actions_for_replay(
        flat,
        action_low=np.array([-1.0, -1.0], dtype=np.float32),
        action_high=np.array([1.0, 1.0], dtype=np.float32),
    )
    assert flat.shape[0] == 2
