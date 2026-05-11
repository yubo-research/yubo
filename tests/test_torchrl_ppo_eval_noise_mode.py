from __future__ import annotations

import pytest

from tests.ppo_eval_noise_mode_helpers import run_ppo_eval_noise_mode_twice


@pytest.mark.parametrize(
    "eval_noise_mode,expected_eval_seeds,expected_heldout",
    [
        ("natural", [100, 101], [100, 101]),
        ("frozen", [100, 100], [99999, 99999]),
    ],
)
def test_ppo_eval_noise_mode_advances_eval_and_heldout(monkeypatch, tmp_path, eval_noise_mode, expected_eval_seeds, expected_heldout):
    eval_seeds, heldout = run_ppo_eval_noise_mode_twice(monkeypatch, tmp_path, eval_noise_mode=eval_noise_mode)
    assert eval_seeds == expected_eval_seeds
    assert heldout == expected_heldout


def test_ppo_eval_noise_mode_invalid_rejected_before_env_build(monkeypatch):
    from rl.torchrl.ppo import core as ppo_core

    monkeypatch.setattr(
        ppo_core,
        "build_env_setup",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("env build should not run")),
    )
    with pytest.raises(ValueError, match="eval_noise_mode must be one of"):
        ppo_core.train_ppo(ppo_core.PPOConfig(eval_noise_mode="invalid-mode"))
