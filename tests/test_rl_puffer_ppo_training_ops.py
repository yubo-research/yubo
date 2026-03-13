from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rl.pufferlib.ppo import engine as ppo_engine
from rl.pufferlib.ppo import eval as ppo_eval
from rl.pufferlib.ppo import training_ops
from rl.pufferlib.ppo.config import PufferPPOConfig
from rl.pufferlib.ppo.specs import (
    _ActionSpec,
    _FlatBatch,
    _ObservationSpec,
    _RolloutBuffer,
    _RuntimeState,
    _TrainPlan,
)


class _CollectModel:
    def get_action_and_value(self, obs):
        b = int(obs.shape[0])
        action = torch.zeros((b,), dtype=torch.long)
        logprob = torch.zeros((b,), dtype=torch.float32)
        entropy = torch.zeros((b,), dtype=torch.float32)
        value = torch.zeros((b,), dtype=torch.float32)
        return action, logprob, entropy, value

    def get_value(self, obs):
        return torch.zeros((int(obs.shape[0]),), dtype=torch.float32, device=obs.device)


class _UpdateModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.1, dtype=torch.float32))

    def get_action_and_value(self, obs, action=None):
        b = int(obs.shape[0])
        if action is None:
            action = torch.zeros((b,), dtype=torch.long, device=obs.device)
        newlogprob = self.bias.expand(b)
        entropy = torch.ones((b,), dtype=torch.float32, device=obs.device) * 0.1
        newvalue = self.bias.expand(b)
        return action, newlogprob, entropy, newvalue


class _OneStepEnv:
    def step(self, _action):
        obs = np.zeros((1, 3), dtype=np.float32)
        rew = np.array([1.0], dtype=np.float32)
        term = np.array([False], dtype=bool)
        trunc = np.array([False], dtype=bool)
        infos = [{"episode_return": 7.0}]
        return obs, rew, term, trunc, infos


def test_engine_name_helpers_delegate(monkeypatch):
    import rl.core.ppo_envs as ppo_envs
    import rl.pufferlib.vector_env as vector_env

    captured = {}

    def _fake_make_vector_env(config, **kwargs):
        _ = config
        captured.update(kwargs)
        return "vec"

    monkeypatch.setattr(vector_env, "make_vector_env", _fake_make_vector_env)
    out = ppo_engine.make_vector_env(PufferPPOConfig(env_tag="atari:Pong"))
    assert out == "vec"
    assert captured["is_atari_env_tag_fn"] is ppo_envs.is_atari_env_tag
    assert captured["to_puffer_game_name_fn"] is ppo_envs.to_puffer_game_name
    assert captured["resolve_gym_env_name_fn"] is ppo_envs.resolve_gym_env_name


def test_engine_env_wrapper(monkeypatch):
    captured = {}

    def _fake(config, *, obs_spec):
        captured["env_tag"] = config.env_tag
        captured["mode"] = obs_spec.mode
        return "env-conf"

    monkeypatch.setattr(ppo_engine, "_env", _fake)
    cfg = PufferPPOConfig(env_tag="Pendulum-v1")
    out = ppo_engine._env(cfg, obs_spec=_ObservationSpec(mode="vector", raw_shape=(3,), vector_dim=3))
    assert out == "env-conf"
    assert captured == {"env_tag": "Pendulum-v1", "mode": "vector"}


def test_training_ops_collect_flatten_advantages_update():
    device = torch.device("cpu")
    plan = _TrainPlan(num_envs=1, num_steps=1, batch_size=1, minibatch_size=1, num_iterations=1)
    buffer = _RolloutBuffer(
        obs=torch.zeros((1, 1, 3), dtype=torch.float32, device=device),
        actions=torch.zeros((1, 1), dtype=torch.long, device=device),
        logprobs=torch.zeros((1, 1), dtype=torch.float32, device=device),
        rewards=torch.zeros((1, 1), dtype=torch.float32, device=device),
        dones=torch.zeros((1, 1), dtype=torch.float32, device=device),
        values=torch.zeros((1, 1), dtype=torch.float32, device=device),
    )
    state = _RuntimeState(
        next_obs=torch.zeros((1, 3), dtype=torch.float32, device=device),
        next_done=torch.zeros((1,), dtype=torch.float32, device=device),
        obs_spec=_ObservationSpec(mode="vector", raw_shape=(3,), vector_dim=3),
        action_spec=_ActionSpec(kind="discrete", dim=2),
        global_step=0,
        start_iteration=0,
        start_time=0.0,
        best_actor_state=None,
        best_return=-float("inf"),
        last_eval_return=float("nan"),
        last_heldout_return=None,
        last_episode_return=float("nan"),
        eval_env_conf=None,
    )

    model = _CollectModel()
    envs = _OneStepEnv()
    training_ops.collect_rollout(
        plan,
        model,
        envs,
        buffer,
        state,
        device,
        prepare_obs_fn=lambda obs_np, *, obs_spec, device: torch.as_tensor(obs_np, device=device, dtype=torch.float32),
    )
    assert state.global_step == 1
    assert state.last_episode_return == 7.0

    cfg = PufferPPOConfig(
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        total_timesteps=1,
        eval_interval=0,
    )
    advantages, returns = training_ops.compute_advantages(plan, cfg, model, state, buffer, device)
    assert advantages.shape == (1, 1)
    assert returns.shape == (1, 1)

    flat = training_ops.flatten_batch(plan, buffer, advantages, returns, (3,))
    assert flat.obs.shape == (1, 3)
    assert flat.actions.shape == (1,)

    update_model = _UpdateModel()
    optimizer = torch.optim.Adam(update_model.parameters(), lr=1e-2)
    batch = _FlatBatch(
        obs=torch.zeros((4, 3), dtype=torch.float32),
        actions=torch.zeros((4,), dtype=torch.long),
        logprobs=torch.zeros((4,), dtype=torch.float32),
        advantages=torch.ones((4,), dtype=torch.float32),
        returns=torch.ones((4,), dtype=torch.float32),
        values=torch.zeros((4,), dtype=torch.float32),
    )
    update_plan = _TrainPlan(num_envs=1, num_steps=4, batch_size=4, minibatch_size=2, num_iterations=1)
    b_inds = np.arange(4)
    update_stats = training_ops.ppo_update(cfg, update_plan, update_model, optimizer, batch, b_inds)
    assert isinstance(update_stats.approx_kl, float)
    assert isinstance(update_stats.clipfrac_mean, float)


def test_train_ppo_puffer_impl_calls_run_training(monkeypatch, tmp_path):
    cfg = PufferPPOConfig(
        exp_dir=str(tmp_path / "exp"),
        env_tag="atari:Pong",
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        eval_interval=0,
        device="cpu",
    )

    monkeypatch.setattr(ppo_engine, "_resolve_device", lambda _raw: torch.device("cpu"))
    monkeypatch.setattr(ppo_engine, "_seed_everything", lambda _seed: None)
    monkeypatch.setattr(ppo_engine, "_build_plan", lambda _cfg: _TrainPlan(2, 2, 4, 2, 2))
    monkeypatch.setattr(ppo_engine, "_prepare_outputs", lambda _cfg: tmp_path / "metrics.jsonl")
    monkeypatch.setattr(ppo_engine, "make_vector_env", lambda _cfg: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(ppo_eval, "validate_eval_config", lambda _cfg: None)
    monkeypatch.setattr(
        ppo_engine,
        "_run_training",
        lambda *args, **kwargs: ppo_engine.TrainResult(
            best_return=1.0,
            last_eval_return=1.0,
            last_heldout_return=None,
            num_iterations=2,
        ),
    )

    out = ppo_engine.train_ppo_puffer_impl(cfg)
    assert out.num_iterations == 2
