from __future__ import annotations

import importlib
import json
import time
from types import SimpleNamespace

import numpy as np
import torch
from torchrl_sac_loop_actor_stub import _ActorStub
from torchrl_sac_loop_test_stubs import _ReplayStub, _TrainEnvStub

torchrl_sac_loop = importlib.import_module("rl.torchrl.sac.loop")


def _sac_loop_config(**overrides):
    config = SimpleNamespace(
        env_tag="env",
        seed=1,
        log_interval_steps=4,
        collector=SimpleNamespace(init_random_frames=0, total_frames=20),
        optim=SimpleNamespace(update_every=1, optim_steps_per_batch=1),
        replay_buffer=SimpleNamespace(batch_size=32),
        eval=SimpleNamespace(interval_steps=5, num_denoise_passive=None, seed_base=None, noise_mode=None),
        checkpoint=SimpleNamespace(interval_steps=5),
    )
    for key, value in overrides.items():
        if "." in key:
            section, attr = key.split(".", 1)
            setattr(getattr(config, section), attr, value)
        else:
            setattr(config, key, value)
    return config


def test_as_float32_observation_casts_to_float32():
    observation = np.array([1.0, 2.0], dtype=np.float64)
    cast_observation = torchrl_sac_loop.as_float32_observation(observation)
    assert cast_observation.dtype == np.float32
    assert np.allclose(cast_observation, np.array([1.0, 2.0], dtype=np.float32))


def test_temporary_actor_state_restores_on_exception():
    modules = SimpleNamespace()
    restore_calls = []

    def _capture_actor_state(_modules):
        return {"current": 5}

    def _restore_actor_state(_modules, snapshot):
        restore_calls.append(snapshot)

    try:
        with torchrl_sac_loop.temporary_actor_state(
            modules,
            {"best": 99},
            capture_actor_state=_capture_actor_state,
            restore_actor_state=_restore_actor_state,
        ):
            raise RuntimeError("boom")
    except RuntimeError as exc:
        assert "boom" in str(exc)

    assert restore_calls[0] == {"best": 99}
    assert restore_calls[1] == {"current": 5}


def test_is_due():
    assert not torchrl_sac_loop.is_due(4, None)
    assert not torchrl_sac_loop.is_due(4, 0)
    assert not torchrl_sac_loop.is_due(4, 3)
    assert torchrl_sac_loop.is_due(6, 3)


def test_select_training_action_uses_random_before_learning_starts():
    config = _sac_loop_config(**{"collector.init_random_frames": 10})
    env_setup = SimpleNamespace(action_low=np.array([-1.0, -1.0]), action_high=np.array([1.0, 1.0]))
    modules = SimpleNamespace(actor=_ActorStub([0.1, -0.2]))
    train_env = _TrainEnvStub(
        sample_value=[0.25, 0.75],
        step_result=None,
        reset_state=[0.0, 0.0],
    )

    action_env, action_norm = torchrl_sac_loop.select_training_action(
        config,
        env_setup,
        modules,
        step=3,
        observation=np.array([0.0, 0.0], dtype=np.float32),
        train_env=train_env,
        device=torch.device("cpu"),
        unscale_action_from_env=lambda action, _low, _high: action * 2 - 1,
        scale_action_to_env=lambda action, _low, _high: action,
    )
    assert np.allclose(action_env, np.array([0.25, 0.75], dtype=np.float32))
    assert np.allclose(action_norm, np.array([-0.5, 0.5], dtype=np.float32))


def test_select_training_action_uses_actor_after_learning_starts():
    config = _sac_loop_config(**{"collector.init_random_frames": 2})
    env_setup = SimpleNamespace(action_low=np.array([-1.0, -1.0]), action_high=np.array([1.0, 1.0]))
    modules = SimpleNamespace(actor=_ActorStub([0.2, -0.1]))
    train_env = _TrainEnvStub(sample_value=[0.0, 0.0], step_result=None, reset_state=[0.0, 0.0])

    action_env, action_norm = torchrl_sac_loop.select_training_action(
        config,
        env_setup,
        modules,
        step=5,
        observation=np.array([1.0, 2.0], dtype=np.float32),
        train_env=train_env,
        device=torch.device("cpu"),
        unscale_action_from_env=lambda action, _low, _high: action,
        scale_action_to_env=lambda action, _low, _high: action + 0.5,
    )
    assert np.allclose(action_norm, np.array([0.2, -0.1], dtype=np.float32))
    assert np.allclose(action_env, np.array([0.7, 0.4], dtype=np.float32))


def test_advance_env_and_store_handles_done_and_continue():
    replay = _ReplayStub()
    training_setup = SimpleNamespace(replay=replay)

    done_env = _TrainEnvStub(
        sample_value=[0.0, 0.0],
        step_result=(np.array([9.0, 9.0], dtype=np.float32), 2.0, True, False, {}),
        reset_state=[7.0, 7.0],
    )
    next_observation_done = torchrl_sac_loop.advance_env_and_store(
        training_setup,
        train_env=done_env,
        observation=np.array([1.0, 1.0], dtype=np.float32),
        action_env=np.array([0.1, 0.2], dtype=np.float32),
        action_norm=np.array([0.1, 0.2], dtype=np.float32),
        make_transition=lambda *values: ("transition", values),
    )
    assert np.allclose(next_observation_done, np.array([7.0, 7.0], dtype=np.float32))
    assert len(replay.items) == 1

    continue_env = _TrainEnvStub(
        sample_value=[0.0, 0.0],
        step_result=(np.array([5.0, 6.0], dtype=np.float32), 1.0, False, False, {}),
        reset_state=[0.0, 0.0],
    )
    next_observation_continue = torchrl_sac_loop.advance_env_and_store(
        training_setup,
        train_env=continue_env,
        observation=np.array([1.0, 1.0], dtype=np.float32),
        action_env=np.array([0.1, 0.2], dtype=np.float32),
        action_norm=np.array([0.1, 0.2], dtype=np.float32),
        make_transition=lambda *values: ("transition", values),
    )
    assert np.allclose(next_observation_continue, np.array([5.0, 6.0], dtype=np.float32))
    assert len(replay.items) == 2


def test_run_updates_if_due():
    config = _sac_loop_config(
        **{
            "collector.init_random_frames": 3,
            "optim.update_every": 2,
            "optim.optim_steps_per_batch": 3,
            "replay_buffer.batch_size": 32,
        }
    )
    training_setup = SimpleNamespace()
    updated_losses, total_updates = torchrl_sac_loop.run_updates_if_due(
        config,
        training_setup,
        step=2,
        device=torch.device("cpu"),
        latest_losses={"loss_actor": float("nan")},
        total_updates=4,
        update_step=lambda *_args, **_kwargs: {"loss_actor": 1.0},
    )
    assert total_updates == 4
    assert np.isnan(updated_losses["loss_actor"])

    update_calls = {"count": 0}

    def _update_step(_setup, *, device, batch_size):
        _ = device
        assert batch_size == 32
        update_calls["count"] += 1
        return {"loss_actor": float(update_calls["count"])}

    updated_losses, total_updates = torchrl_sac_loop.run_updates_if_due(
        config,
        training_setup,
        step=6,
        device=torch.device("cpu"),
        latest_losses={"loss_actor": 0.0},
        total_updates=10,
        update_step=_update_step,
    )
    assert update_calls["count"] == 3
    assert total_updates == 13
    assert updated_losses["loss_actor"] == 3.0


def test_evaluate_heldout_if_enabled():
    config_disabled = _sac_loop_config(**{"eval.num_denoise_passive": None})
    env_setup = SimpleNamespace(problem_seed=1, noise_seed_0=2)
    modules = SimpleNamespace()
    train_state_disabled = SimpleNamespace(best_actor_state={"best": 1})
    result = torchrl_sac_loop.evaluate_heldout_if_enabled(
        config_disabled,
        env_setup,
        modules,
        train_state_disabled,
        device=torch.device("cpu"),
        capture_actor_state=lambda *_: {"current": 1},
        restore_actor_state=lambda *_: None,
        eval_policy_factory=lambda *_: object(),
        build_env_runtime=lambda *_args, **_kwargs: object(),
        evaluate_for_best=lambda *_args, **_kwargs: 1.0,
    )
    assert result is None

    restore_calls = []
    config_enabled = _sac_loop_config(**{"eval.num_denoise_passive": 3})
    train_state_enabled = SimpleNamespace(best_actor_state={"best": 99})
    result = torchrl_sac_loop.evaluate_heldout_if_enabled(
        config_enabled,
        env_setup,
        modules,
        train_state_enabled,
        device=torch.device("cpu"),
        capture_actor_state=lambda *_: {"current": 5},
        restore_actor_state=lambda _modules, snapshot: restore_calls.append(snapshot),
        eval_policy_factory=lambda *_: "policy",
        build_env_runtime=lambda *_args, **_kwargs: "env_conf",
        evaluate_for_best=lambda _env_conf, _policy, denoise, **_kwargs: (4.5 if denoise == 3 else -1.0),
    )
    assert result == 4.5
    assert restore_calls[0] == {"best": 99}
    assert restore_calls[1] == {"current": 5}


def test_evaluate_heldout_if_enabled_passes_pixel_flags():
    captured_kwargs = {}
    config = _sac_loop_config(env_tag="dm:cheetah-run", **{"eval.num_denoise_passive": 3})
    env_setup = SimpleNamespace(
        problem_seed=11,
        noise_seed_0=22,
        env_conf=SimpleNamespace(from_pixels=True, pixels_only=False),
    )
    modules = SimpleNamespace()
    train_state = SimpleNamespace(best_actor_state={"best": 99})

    result = torchrl_sac_loop.evaluate_heldout_if_enabled(
        config,
        env_setup,
        modules,
        train_state,
        device=torch.device("cpu"),
        capture_actor_state=lambda *_: {"current": 1},
        restore_actor_state=lambda *_: None,
        eval_policy_factory=lambda *_: "policy",
        build_env_runtime=lambda *_args, **kwargs: (captured_kwargs.update(kwargs) or "env_conf"),
        evaluate_for_best=lambda *_args, **_kwargs: 7.0,
    )
    assert result == 7.0
    assert captured_kwargs["from_pixels"] is True
    assert captured_kwargs["pixels_only"] is False
    assert captured_kwargs["problem_seed"] == 11
    assert captured_kwargs["noise_seed_0"] == 22


def test_evaluate_heldout_if_enabled_restores_actor_state_on_exception():
    restore_calls = []
    config = _sac_loop_config(**{"eval.num_denoise_passive": 3})
    env_setup = SimpleNamespace(problem_seed=1, noise_seed_0=2)
    modules = SimpleNamespace()
    train_state = SimpleNamespace(best_actor_state={"best": 99})

    def _raise_in_eval(*_args, **_kwargs):
        raise RuntimeError("heldout-failure")

    try:
        torchrl_sac_loop.evaluate_heldout_if_enabled(
            config,
            env_setup,
            modules,
            train_state,
            device=torch.device("cpu"),
            capture_actor_state=lambda *_: {"current": 5},
            restore_actor_state=lambda _modules, snapshot: restore_calls.append(snapshot),
            eval_policy_factory=lambda *_: "policy",
            build_env_runtime=lambda *_args, **_kwargs: "env_conf",
            evaluate_for_best=_raise_in_eval,
        )
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "heldout-failure" in str(exc)
    assert restore_calls[0] == {"best": 99}
    assert restore_calls[1] == {"current": 5}


def test_evaluate_if_due_updates_state_and_writes_metrics(monkeypatch):
    appended_records = []
    monkeypatch.setattr(
        "rl.logger.log_rl_iter",
        lambda payload, metrics_path=None: appended_records.append((metrics_path, payload)),
    )

    config = _sac_loop_config(**{"eval.interval_steps": 5, "eval.seed_base": 123})
    env_setup = SimpleNamespace()
    modules = SimpleNamespace()
    training_setup = SimpleNamespace(metrics_path="metrics.jsonl")
    train_state = SimpleNamespace(
        best_return=1.0,
        best_actor_state=None,
        last_eval_return=float("nan"),
        last_heldout_return=None,
    )
    torchrl_sac_loop.evaluate_if_due(
        config,
        env_setup,
        modules,
        training_setup,
        train_state,
        step=5,
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
        latest_losses={"loss_actor": 1.1, "loss_critic": 2.2, "loss_alpha": 3.3},
        total_updates=7,
        evaluate_actor=lambda *_args, **kwargs: (6.0 if kwargs["eval_seed"] == 123 else -1.0),
        capture_actor_state=lambda *_: {"snapshot": 42},
        evaluate_heldout=lambda *_args, **_kwargs: 5.5,
    )
    assert train_state.last_eval_return == 6.0
    assert train_state.best_return == 6.0
    assert train_state.best_actor_state == {"snapshot": 42}
    assert train_state.last_heldout_return == 5.5
    assert len(appended_records) == 1

    train_state_unchanged = SimpleNamespace(
        best_return=10.0,
        best_actor_state={"snapshot": 1},
        last_eval_return=float("nan"),
        last_heldout_return=None,
    )
    torchrl_sac_loop.evaluate_if_due(
        config,
        env_setup,
        modules,
        training_setup,
        train_state_unchanged,
        step=4,
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=0,
        evaluate_actor=lambda *_args, **_kwargs: 99.0,
        capture_actor_state=lambda *_: {"snapshot": 99},
        evaluate_heldout=lambda *_args, **_kwargs: 99.0,
    )
    assert train_state_unchanged.best_return == 10.0
    assert len(appended_records) == 1

    torchrl_sac_loop.evaluate_if_due(
        config,
        env_setup,
        modules,
        training_setup,
        train_state_unchanged,
        step=6,
        device=torch.device("cpu"),
        start_time=time.time() - 1.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=0,
        evaluate_actor=lambda *_args, **_kwargs: 11.0,
        capture_actor_state=lambda *_: {"snapshot": 11},
        evaluate_heldout=lambda *_args, **_kwargs: 10.5,
    )
    assert train_state_unchanged.best_return == 11.0
    assert len(appended_records) == 2


def test_log_and_checkpoint_helpers(tmp_path, capsys):
    config = _sac_loop_config(
        log_interval_steps=4,
        **{
            "collector.total_frames": 20,
            "checkpoint.interval_steps": 5,
        },
    )
    metrics_path = tmp_path / "metrics.jsonl"
    train_state = SimpleNamespace(
        last_eval_return=3.2,
        last_heldout_return=None,
        best_return=3.2,
        _last_log_mark=0,
    )
    training_setup = SimpleNamespace(metrics_path=metrics_path)
    torchrl_sac_loop.log_if_due(
        config,
        train_state,
        step=6,
        start_time=time.time() - 1.0,
        latest_losses={"loss_actor": 1.0, "loss_critic": 2.0, "loss_alpha": 3.0},
        total_updates=9,
        training_setup=training_setup,
    )
    rows = [line for line in metrics_path.read_text().splitlines() if line.strip()]
    assert len(rows) == 1
    record = json.loads(rows[0])
    assert record["actor"] == 1.0
    assert record["ret_eval"] == 3.2
    out = capsys.readouterr().out
    assert "ITER:" not in out
    assert "        6" in out
    assert "3.2" in out

    calls = []
    training_setup = SimpleNamespace(checkpoint_manager=SimpleNamespace(save_both=lambda payload, iteration: calls.append((payload, iteration))))
    modules = SimpleNamespace()
    torchrl_sac_loop.checkpoint_if_due(
        config,
        modules,
        training_setup,
        train_state,
        step=6,
        build_checkpoint_payload=lambda *_args, **kwargs: {"step": kwargs["step"]},
    )
    assert calls == [({"step": 6}, 6)]

    torchrl_sac_loop.save_final_checkpoint_if_enabled(
        config,
        modules,
        training_setup,
        train_state,
        build_checkpoint_payload=lambda *_args, **kwargs: {"step": kwargs["step"]},
    )
    assert calls[-1] == ({"step": 20}, 20)
