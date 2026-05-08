from __future__ import annotations

from click.testing import CliRunner


def test_llm_registry_resolves_env_and_policy_tags():
    from llm.registry import resolve_llm_env, resolve_llm_policy

    env = resolve_llm_env("llm:math:answer-tags:gsm8k")
    policy = resolve_llm_policy("qwen3-1p7b-lora-r4")

    assert env.task_name == "math:answer-tags:gsm8k"
    assert env.answer_format == "answer_tags"
    assert policy.model_name == "Qwen/Qwen3-1.7B"
    assert policy.lora_rank == 4
    assert policy.lora_alpha == 4


def test_llm_random_boxed_reward_pass_at_k():
    from llm.tasks import RandomTask

    task = RandomTask(batch_size=1, max_random_number=4, seed=0, answer_format="boxed")
    fitness, model_answers, sample_fitnesses = task.score(
        ["wrong 3", "final answer is boxed{2}"],
        [False, False],
        2,
        pass_at_k=True,
    )

    assert fitness == 1.0
    assert model_answers == (None, 2)
    assert sample_fitnesses.tolist() == [0.0, 1.0]


def test_llm_countdown_reward_uses_safe_arithmetic():
    from llm.tasks import countdown_answer_reward

    reward, answer = countdown_answer_reward("<answer>8*(3+2)</answer>", numbers=[8, 3, 2], target=40)

    assert reward == 1.0
    assert answer == "8*(3+2)"


def test_llm_config_parse_and_override(tmp_path):
    from experiments import llm

    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
population_size = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    raw = llm._load_toml_config(str(config))
    raw = {**raw, **llm._parse_overrides(("population_size=4", "pass_at_k=true", "samples_per_prompt=2"))}
    cfg = llm._parse_cfg(raw)

    assert cfg.env_tag == "llm:math:gsm8k"
    assert cfg.policy.model_name == "Qwen/Qwen3-1.7B"
    assert cfg.population_size == 4
    assert cfg.pass_at_k is True


def test_llm_sft_uses_num_epochs_budget(tmp_path):
    from experiments import llm

    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "sft"
num_epochs = 1
batch_size = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = llm._parse_cfg(llm._load_toml_config(str(config)))

    assert cfg.optimizer == "sft"
    assert cfg.num_epochs == 1
    assert cfg.num_rounds is None


def test_ops_llm_dry_run_does_not_write_metadata(tmp_path):
    from ops.llm import cli

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
    assert "Qwen/Qwen3-1.7B" in result.output
    assert not exp_dir.exists()


def test_llm_uhd_optimizer_points_to_uhd_text_schema(tmp_path):
    from experiments.llm import cli

    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "uhd"
num_rounds = 1
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["validate", str(config)])

    assert result.exit_code != 0
    assert "Use [uhd] with ./ops/exp_uhd.py" in result.output


def test_llm_es_summarize_centers_per_prompt():
    from llm.es import summarize_fitness

    summary = summarize_fitness([[1.0, 3.0], [3.0, 5.0]], normalize_with_std=False)

    assert summary.mean == 3.0
    assert summary.max == 5.0
    assert summary.normalized.tolist() == [-1.0, 1.0]


def test_llm_countdown_task_builds_owned_synthetic_batches():
    from llm.tasks import CountdownTask

    task = CountdownTask(batch_size=2, seed=0, dataset_size=4)
    prompts, answers = task.get_batch()

    assert len(prompts) == 2
    assert len(answers) == 2
    assert "<answer>" in prompts[0]


def test_llm_eggroll_lora_specs_repeat_per_prompt():
    from llm.eggroll import _engine_lora_specs

    specs = _engine_lora_specs([0, 1], ["/tmp/a", "/tmp/b"], es_step=3, num_prompts=2)

    assert specs == [
        ("adapter_0", 30001, "/tmp/a"),
        ("adapter_0", 30001, "/tmp/a"),
        ("adapter_1", 30002, "/tmp/b"),
        ("adapter_1", 30002, "/tmp/b"),
    ]


def test_ops_llm_dispatches_eggroll_runtime(tmp_path, monkeypatch):
    from experiments.llm import cli
    import llm.eggroll as eggroll

    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:zeros"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
population_size = 2
""".strip()
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(eggroll, "run_eggroll", lambda cfg: {"iterations": cfg.num_rounds, "best": 0.0})

    result = CliRunner().invoke(cli, ["local", str(config)])

    assert result.exit_code == 0, result.output
    assert 'RESULT: {"best": 0.0, "iterations": 1}' in result.output
