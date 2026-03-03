from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

from rl.core.config_utils import DataclassFromDictMixin, dataclass_config_from_dict
from rl.pufferlib.offpolicy.engine_utils import checkpoint_mark_if_due, init_run_artifacts, init_runtime
from rl.pufferlib.offpolicy.runtime_utils import obs_scale_from_env, select_device
from rl.torchrl.offpolicy.trainer_utils import flatten_batch_to_transitions, normalize_actions_for_replay, process_offpolicy_batch


@dataclass
class _Cfg(DataclassFromDictMixin):
    _tuple_int_keys = ("sizes",)
    _int_keys = ("workers",)

    sizes: tuple[int, ...] = (1,)
    workers: int = 1
    keep: str = "ok"


def test_dataclass_config_from_dict_and_mixin():
    cfg = dataclass_config_from_dict(_Cfg, {"sizes": ["2", 3], "workers": "4", "keep": "yes", "ignored": 9}, tuple_int_keys=("sizes",), int_keys=("workers",))
    assert cfg.sizes == (2, 3)
    assert cfg.workers == 4
    assert cfg.keep == "yes"

    cfg2 = _Cfg.from_dict({"sizes": [5, "6"], "workers": "7", "keep": "mix"})
    assert cfg2.sizes == (5, 6)
    assert cfg2.workers == 7
    assert cfg2.keep == "mix"


def test_engine_utils_init_runtime_and_checkpoint_mark(monkeypatch):
    monkeypatch.setattr("rl.core.env_conf.global_seed_for_run", lambda seed: int(seed) + 101)
    config = SimpleNamespace(device="cpu")
    env_setup = SimpleNamespace(problem_seed=7)
    seeded: list[int] = []
    env_out, device_out = init_runtime(
        config,
        build_env_setup_fn=lambda _cfg: env_setup,
        seed_everything_fn=lambda seed: seeded.append(int(seed)),
        resolve_device_fn=lambda raw: f"resolved:{raw}",
    )
    assert env_out is env_setup
    assert device_out == "resolved:cpu"
    assert seeded == [108]

    saved: list[int] = []
    mark = checkpoint_mark_if_due(
        global_step=15,
        checkpoint_interval_steps=10,
        previous_mark=0,
        due_mark_fn=lambda step, interval, prev: 1 if int(step) >= int(interval) and int(prev) == 0 else None,
        save_fn=lambda: saved.append(1),
    )
    assert mark == 1
    assert saved == [1]

    mark2 = checkpoint_mark_if_due(
        global_step=5,
        checkpoint_interval_steps=10,
        previous_mark=1,
        due_mark_fn=lambda *_args: None,
        save_fn=lambda: saved.append(2),
    )
    assert mark2 == 1
    assert saved == [1]


def test_engine_utils_init_run_artifacts(monkeypatch, tmp_path: Path):
    writes: list[tuple[str, dict]] = []

    data_io = ModuleType("analysis.data_io")
    data_io.write_config = lambda exp_dir, cfg: writes.append((str(exp_dir), dict(cfg)))
    checkpointing = ModuleType("rl.checkpointing")

    class _CheckpointManager:
        def __init__(self, *, exp_dir):
            self.exp_dir = exp_dir

    checkpointing.CheckpointManager = _CheckpointManager
    monkeypatch.setitem(sys.modules, "analysis.data_io", data_io)
    monkeypatch.setitem(sys.modules, "rl.checkpointing", checkpointing)

    exp_path, metrics_path, checkpoint_manager = init_run_artifacts(exp_dir=str(tmp_path / "exp"), config_dict={"a": 1})
    assert exp_path.exists()
    assert metrics_path == exp_path / "metrics.jsonl"
    assert writes == [(str(exp_path), {"a": 1})]
    assert checkpoint_manager.exp_dir == exp_path


def test_offpolicy_runtime_utils():
    assert select_device("cpu").type == "cpu"

    env_no_scale = SimpleNamespace(gym_conf=SimpleNamespace(transform_state=False), ensure_spaces=lambda: None)
    lb, width = obs_scale_from_env(env_no_scale)
    assert lb is None and width is None

    state_space = SimpleNamespace(low=np.asarray([-1.0, -2.0], dtype=np.float32), high=np.asarray([1.0, 2.0], dtype=np.float32), shape=(2,))
    env_scaled = SimpleNamespace(gym_conf=SimpleNamespace(transform_state=True, state_space=state_space), ensure_spaces=lambda: None)
    lb2, width2 = obs_scale_from_env(env_scaled)
    assert np.allclose(lb2, state_space.low)
    assert np.allclose(width2, np.asarray([2.0, 4.0], dtype=np.float32))


def test_offpolicy_trainer_utils_batch_pipeline():
    batch = TensorDict(
        {
            "observation": torch.zeros((2, 3), dtype=torch.float32),
            "action": torch.tensor([[0.0, 1.0], [1.0, -1.0]], dtype=torch.float32),
            "next": TensorDict(
                {
                    "observation": torch.ones((2, 3), dtype=torch.float32),
                    "reward": torch.tensor([1.0, 2.0], dtype=torch.float32),
                    "terminated": torch.tensor([False, True]),
                    "truncated": torch.tensor([False, False]),
                },
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )

    flat = flatten_batch_to_transitions(batch)
    assert flat["next", "reward"].shape == (2, 1)
    assert flat["next", "done"].shape == (2, 1)

    normalized = normalize_actions_for_replay(
        flat,
        action_low=np.asarray([-2.0, -2.0], dtype=np.float32),
        action_high=np.asarray([2.0, 2.0], dtype=np.float32),
    )
    assert normalized["action"].shape == (2, 2)
    assert float(normalized["action"].abs().max().item()) <= 1.0

    class _Replay:
        def __init__(self):
            self.write_count = 0
            self.rows: list[TensorDict] = []

        def add(self, td):
            self.rows.append(td)
            self.write_count += 1

    replay = _Replay()
    training = SimpleNamespace(replay=replay)
    config = SimpleNamespace(update_every=1, updates_per_step=2, learning_starts=1, batch_size=2)
    env_setup = SimpleNamespace(action_low=np.asarray([-2.0, -2.0], dtype=np.float32), action_high=np.asarray([2.0, 2.0], dtype=np.float32))
    updates: list[tuple[str, int]] = []

    latest, total_updates, n_frames = process_offpolicy_batch(
        batch,
        config=config,
        training=training,
        runtime_device=torch.device("cpu"),
        env_setup=env_setup,
        latest_losses={},
        total_updates=0,
        update_step_fn=lambda device, batch_size: (updates.append((str(device), int(batch_size))) or {"loss_actor": 1.0}),
    )
    assert n_frames == 2
    assert len(replay.rows) == 2
    assert total_updates == 4
    assert len(updates) == 4
    assert latest == {"loss_actor": 1.0}
