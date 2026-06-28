from __future__ import annotations

import pickle
import sys
import types

from click.testing import CliRunner

from tests.llm_test_mocks_env import FakeEnv
from tests.llm_test_mocks_state import FakeState
from tests.llm_test_support import AccidentalRollout, fake_ray_cls, make_fake_tokenizer


def test_llm_registry_resolves_env_and_policy_tags():
    from llm.registry import (
        policy_uses_chat_template,
        resolve_llm_env,
        resolve_llm_policy,
    )

    env = resolve_llm_env("llm:math:answer-tags:gsm8k")
    verifiers_env = resolve_llm_env("llm:verifiers:gsm8k")
    theorem_env = resolve_llm_env("llm:thm:lean4:cat-searcher/minif2f-lean4")
    policy = resolve_llm_policy("qwen3-1p7b-lora-r4")
    base_policy = resolve_llm_policy("qwen3-1p7b-base-lora-r4")

    assert env.task_name == "math:answer-tags:gsm8k"
    assert env.answer_format == "answer_tags"
    assert verifiers_env.task_kind == "verifiers"
    assert verifiers_env.dataset_name == "gsm8k"
    assert theorem_env.task_kind == "thm"
    assert theorem_env.dataset_name == "cat-searcher/minif2f-lean4"
    assert policy.model_name == "Qwen/Qwen3-1.7B"
    assert policy.lora_rank == 4
    assert policy.lora_alpha == 4
    assert policy_uses_chat_template(policy) is True
    assert policy_uses_chat_template(base_policy) is False

    kimina_policy = resolve_llm_policy("kimina-prover-1p5b-lora-r8")
    assert kimina_policy.model_name == "AI-MO/Kimina-Prover-Preview-Distill-1.5B"
    assert kimina_policy.lora_rank == 8
    assert kimina_policy.lora_alpha == 8

    gemma4_policy = resolve_llm_policy("gemma4-e2b-it-lora-r1")
    assert gemma4_policy.model_name == "google/gemma-4-E2B-it"
    assert gemma4_policy.lora_rank == 1
    assert gemma4_policy.tensor_parallel_size == 1
    assert policy_uses_chat_template(gemma4_policy) is True

    pythia_policy = resolve_llm_policy("pythia-14m-lora-r1")
    assert pythia_policy.model_name == "EleutherAI/pythia-14m"
    assert pythia_policy.lora_rank == 1
    assert pythia_policy.tensor_parallel_size == 1
    assert policy_uses_chat_template(pythia_policy) is False


def test_llm_task_execution_mode_is_explicit():
    from llm.tasks import RandomTask, TaskMode, VerifiersTask, task_mode

    assert task_mode(RandomTask(batch_size=1, max_random_number=4, seed=0)) is TaskMode.SCORE
    assert task_mode(VerifiersTask(batch_size=1, env_id="gsm8k")) is TaskMode.ROLLOUT

    try:
        task_mode(AccidentalRollout())
    except TypeError as exc:
        assert "execution_mode" in str(exc)
    else:
        raise AssertionError("task_mode should reject tasks without an explicit execution_mode")


def test_engine_pool_normalizes_transport_info_by_tensor_rank():
    from llm.engine_pool import transport_info_by_tensor_rank

    infos = [
        {"tensor_rank": 1, "host": "10.0.0.1", "port": 10001},
        {"tensor_rank": 0, "host": "10.0.0.1", "port": 10000},
    ]

    assert transport_info_by_tensor_rank(infos) == {
        0: ("10.0.0.1", 10000),
        1: ("10.0.0.1", 10001),
    }


def test_engine_pool_checks_all_collective_worker_results():
    from llm.engine_pool import collective_results_ok

    assert collective_results_ok([[True, True], [True]])
    assert not collective_results_ok([[True, False], [True]])
    assert not collective_results_ok([])


def test_vllm_placement_bundles_use_cpu_when_cluster_has_no_gpu():
    from llm.engine_pool import vllm_placement_bundles

    FakeRay = fake_ray_cls({"CPU": 8.0})

    assert vllm_placement_bundles(FakeRay, 1) == [{"CPU": 2}]
    assert vllm_placement_bundles(FakeRay, 2) == [{"CPU": 2}, {"CPU": 2}]


def test_vllm_placement_bundles_reserve_gpu_when_available():
    from llm.engine_pool import vllm_placement_bundles

    FakeRay = fake_ray_cls({"CPU": 16.0, "GPU": 2.0})

    assert vllm_placement_bundles(FakeRay, 1) == [{"GPU": 1, "CPU": 2}]


def test_ray_runtime_env_includes_theorem_image_overrides(monkeypatch):
    from llm.engine_pool import ray_env_vars

    monkeypatch.setenv("THM_LEAN4_DOCKER_IMAGE", "lean-img")
    monkeypatch.setenv("THM_COQ_DOCKER_IMAGE", "coq-img")
    monkeypatch.setenv("THM_ISABELLE_DOCKER_IMAGE", "isabelle-img")

    env = ray_env_vars()

    assert env["THM_LEAN4_DOCKER_IMAGE"] == "lean-img"
    assert env["THM_COQ_DOCKER_IMAGE"] == "coq-img"
    assert env["THM_ISABELLE_DOCKER_IMAGE"] == "isabelle-img"


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
    assert task.target_text(2) == " \\boxed{2}"


def test_llm_math_target_text_without_dataset_load():
    from llm.tasks_math import MathTask

    task = MathTask.__new__(MathTask)
    task.answer_format = "none"
    answer_tags_task = MathTask.__new__(MathTask)
    answer_tags_task.answer_format = "answer_tags"

    assert task.target_text("reasoning #### 42") == " \\boxed{42}"
    assert answer_tags_task.target_text("42") == " <answer>42</answer>"


def test_llm_math_nll_plain_prompt_mode_without_chat_template():
    from llm.tasks_math import MathTask

    task = MathTask.__new__(MathTask)
    task.apply_chat_template = False

    assert task.nll_user_contents(["prompt"], ["42"]) == [None]


def test_vllm_rlm_client_truncates_prompt_to_leave_generation_room():
    from llm.tasks_verifiers import _VLLMRLMClient

    llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=13)))
    client = _VLLMRLMClient(
        llm,
        None,
        {"max_tokens": 3},
        tokenizer=make_fake_tokenizer(),
    )

    assert client._truncate_prompt_to_context("0 1 2 3 4 5 6 7 8 9 10 11") == "10 11"


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
    raw = {
        **raw,
        **llm._parse_overrides(("population_size=4", "pass_at_k=true", "samples_per_prompt=2")),
    }
    cfg = llm._parse_cfg(raw)

    assert cfg.env_tag == "llm:math:gsm8k"
    assert cfg.policy.model_name == "Qwen/Qwen3-1.7B"
    assert cfg.population_size == 4
    assert cfg.pass_at_k is True


def test_llm_config_parses_semantic_update_program_fields(tmp_path):
    from experiments import llm

    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
llm_update_roles = "moe_router,moe_expert_down"
llm_update_layer_band = "late"
llm_update_expert_policy = "router"
llm_update_max_targets = 3
""".strip()
        + "\n",
        encoding="utf-8",
    )

    cfg = llm._parse_cfg(llm._load_toml_config(str(config)))

    assert cfg.llm_update_roles == ("moe_router", "moe_expert_down")
    assert cfg.llm_update_layer_band == "late"
    assert cfg.llm_update_expert_policy == "router"
    assert cfg.llm_update_max_targets == 3


def test_llm_config_rejects_unknown_update_role(tmp_path):
    from experiments import llm

    config = tmp_path / "llm.toml"
    config.write_text(
        """
[llm]
env_tag = "llm:math:gsm8k"
policy_tag = "qwen3-1p7b-lora-r1"
optimizer = "eggroll"
num_rounds = 1
llm_update_roles = "not_a_real_role"
""".strip()
        + "\n",
        encoding="utf-8",
    )

    try:
        llm._parse_cfg(llm._load_toml_config(str(config)))
    except ValueError as exc:
        assert "Unknown LLM update role" in str(exc)
    else:
        raise AssertionError("invalid llm_update_roles should fail during config parse")


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


def test_verifiers_task_uses_lazy_environment_adapter(monkeypatch):
    from llm.tasks_verifiers import _ENV_CACHE, VerifiersTask

    _ENV_CACHE.clear()
    calls = []

    def load_environment(env_id, **env_args):
        calls.append((env_id, env_args))
        return FakeEnv()

    verifiers_mod = types.ModuleType("verifiers")
    verifiers_utils_mod = types.ModuleType("verifiers.utils")
    verifiers_env_utils_mod = types.ModuleType("verifiers.utils.env_utils")
    verifiers_types_mod = types.ModuleType("verifiers.types")
    verifiers_utils_mod.__path__ = []
    verifiers_env_utils_mod.load_environment = load_environment
    verifiers_types_mod.State = FakeState
    monkeypatch.setitem(sys.modules, "verifiers", verifiers_mod)
    monkeypatch.setitem(sys.modules, "verifiers.utils", verifiers_utils_mod)
    monkeypatch.setitem(sys.modules, "verifiers.utils.env_utils", verifiers_env_utils_mod)
    monkeypatch.setitem(sys.modules, "verifiers.types", verifiers_types_mod)

    task = VerifiersTask(batch_size=2, env_id="gsm8k", seed=0, dataset_size=2)
    prompts, answers = task.get_batch()
    fitness, model_answers, sample_fitnesses = task.score(
        ["reasoning \\boxed{4}", "reasoning \\boxed{5}"],
        [False, False],
        answers[0],
        pass_at_k=True,
    )
    restored = pickle.loads(pickle.dumps(task))

    assert calls == [("gsm8k", {"num_train_examples": 2})]
    assert prompts[0] == "System: Use boxes.\nUser: 2+2?\nAssistant:"
    assert fitness == 1.0
    assert model_answers == ("4", "5")
    assert sample_fitnesses.tolist() == [1.0, 0.0]
    assert restored._env is None
    assert restored._dataset is None
    _ENV_CACHE.clear()
