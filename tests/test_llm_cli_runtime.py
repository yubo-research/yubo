from __future__ import annotations

from click.testing import CliRunner


def test_llm_envs_lists_supported_tags():
    from experiments.llm import cli, envs

    result = CliRunner().invoke(cli, ["envs"])

    assert result.exit_code == 0, result.output
    assert "llm:math:gsm8k" in result.output
    assert callable(envs)


def test_llm_template_direct_calls_cover_branches():
    from click.testing import CliRunner

    from experiments.llm import template

    runner = CliRunner()

    ok = runner.invoke(
        template,
        [
            "--env-tag",
            "llm:math:gsm8k",
            "--policy-tag",
            "qwen3-1p7b-lora-r1",
            "--optimizer",
            "eggroll",
            "--num-rounds",
            "2",
        ],
    )
    assert ok.exit_code == 0
    assert "[llm]" in ok.output
    assert "num_rounds = 2" in ok.output

    bad = runner.invoke(template, ["--optimizer", "not_a_real_optimizer", "--num-rounds", "1"])
    assert bad.exit_code != 0
    assert "Unknown direct LLM optimizer" in bad.output

    with runner.isolation():
        template(["--optimizer", "eggroll", "--num-rounds", "1"], standalone_mode=False)


def test_experiments_llm_main_delegates_to_cli(monkeypatch):
    from experiments.llm import cli, main

    called = []
    monkeypatch.setattr("experiments.llm.cli", lambda: called.append(True))
    main()
    assert called == [True]
    assert callable(cli)


def test_experiments_llm_local_dry_run_does_not_execute_runtime(tmp_path):
    from experiments.llm import cli, local

    exp_dir = tmp_path / "runs" / "llm"
    config = tmp_path / "llm.toml"
    config.write_text(
        f"""
[llm]
exp_dir = "{exp_dir.as_posix()}"
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
population_size = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["local", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "DRY_RUN: true" in result.output
    assert not exp_dir.exists()
    assert callable(local)


def test_eggroll_support_base_seed_prefers_noise_seed_over_problem_seed():
    from experiments import llm as llm_exp
    from llm.eggroll_support import base_seed

    cfg = llm_exp._parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "optimizer": "eggroll",
            "num_rounds": 1,
            "noise_seed_0": 7,
            "problem_seed": 3,
            "seed_offset": 2,
        }
    )

    assert base_seed(cfg) == 9


def test_eggroll_support_base_seed_falls_back_to_problem_seed():
    from experiments import llm as llm_exp
    from llm.eggroll_support import base_seed

    cfg = llm_exp._parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "optimizer": "eggroll",
            "num_rounds": 1,
            "problem_seed": 3,
            "seed_offset": 1,
        }
    )

    assert base_seed(cfg) == 4


def test_eggroll_support_base_seed_treats_zero_noise_seed_as_explicit():
    from experiments import llm as llm_exp
    from llm.eggroll_support import base_seed

    cfg = llm_exp._parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "optimizer": "eggroll",
            "num_rounds": 1,
            "noise_seed_0": 0,
            "problem_seed": 9,
            "seed_offset": 1,
        }
    )

    assert base_seed(cfg) == 1


def test_eggroll_support_base_seed_defaults_to_zero():
    from experiments import llm as llm_exp
    from llm.eggroll_support import base_seed

    cfg = llm_exp._parse_cfg(
        {
            "env_tag": "llm:math:gsm8k",
            "policy_tag": "qwen3-1p7b-lora-r1",
            "optimizer": "eggroll",
            "num_rounds": 1,
            "seed_offset": 5,
        }
    )

    assert base_seed(cfg) == 5


def test_init_worker_groups_sets_engine_ranks_after_collective_init():
    from unittest.mock import MagicMock

    from llm.eggroll_engine import EggrollArgs, init_worker_groups

    master_info = [{"tensor_rank": 0, "host": "10.0.0.1", "port": 10000}]
    engines = [MagicMock(name=f"engine_{i}") for i in range(2)]
    ray = MagicMock()
    ray.get.side_effect = [
        master_info,
        [[True, True], [True]],
        [None, None],
    ]
    args = EggrollArgs(
        base_seed=0,
        population_size=2,
        num_iterations=1,
        sigma=0.001,
        learning_rate=0.001,
        lora_r=1,
        lora_alpha=1,
        steps_per_adapter=1,
        max_tokens=8,
        temperature=0.0,
        samples_per_prompt=1,
        prompt_batch_size=1,
        pass_at_k=False,
        normalize_with_std=False,
        scale_lr_in_grad=False,
        num_gpus=1,
        num_engines=2,
        tensor_parallel_size=1,
        steps_per_eval=1,
        eval_batch_size=1,
        save_freq=None,
        checkpoint_dir=None,
        use_wandb=False,
        wandb_project="test",
        wandb_name=None,
    )

    init_worker_groups(ray, engines, args=args)

    engines[0].set_engine_rank.remote.assert_called_once_with(0)
    engines[1].set_engine_rank.remote.assert_called_once_with(1)


def test_init_worker_groups_raises_when_collective_init_fails():
    from unittest.mock import MagicMock

    import pytest

    from llm.eggroll_engine import EggrollArgs, init_worker_groups

    master_info = [{"tensor_rank": 0, "host": "10.0.0.1", "port": 10000}]
    engines = [MagicMock(name=f"engine_{i}") for i in range(2)]
    ray = MagicMock()
    ray.get.side_effect = [
        master_info,
        [[True, False], [True]],
    ]
    args = EggrollArgs(
        base_seed=0,
        population_size=2,
        num_iterations=1,
        sigma=0.001,
        learning_rate=0.001,
        lora_r=1,
        lora_alpha=1,
        steps_per_adapter=1,
        max_tokens=8,
        temperature=0.0,
        samples_per_prompt=1,
        prompt_batch_size=1,
        pass_at_k=False,
        normalize_with_std=False,
        scale_lr_in_grad=False,
        num_gpus=1,
        num_engines=2,
        tensor_parallel_size=1,
        steps_per_eval=1,
        eval_batch_size=1,
        save_freq=None,
        checkpoint_dir=None,
        use_wandb=False,
        wandb_project="test",
        wandb_name=None,
    )

    with pytest.raises(RuntimeError, match="Failed to initialize at least one vLLM worker group"):
        init_worker_groups(ray, engines, args=args)

    engines[0].set_engine_rank.remote.assert_not_called()
    engines[1].set_engine_rank.remote.assert_not_called()
