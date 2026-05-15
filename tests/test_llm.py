from __future__ import annotations

import pickle
import sys
import types

from click.testing import CliRunner

from tests.llm_test_mocks_env import FakeEnv
from tests.llm_test_mocks_state import FakeState
from tests.llm_test_mocks_vllm import FakeAsyncEngine, FakeOutput


def test_llm_registry_resolves_env_and_policy_tags():
    from llm.registry import policy_uses_chat_template, resolve_llm_env, resolve_llm_policy

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


def test_llm_task_execution_mode_is_explicit():
    from llm.tasks import RandomTask, TaskMode, VerifiersTask, task_mode

    assert task_mode(RandomTask(batch_size=1, max_random_number=4, seed=0)) is TaskMode.SCORE
    assert task_mode(VerifiersTask(batch_size=1, env_id="gsm8k")) is TaskMode.ROLLOUT

    class AccidentalRollout:
        def generate_and_score(self):
            return None

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


def test_vllm_rlm_client_truncates_prompt_to_leave_generation_room():
    from llm.tasks_verifiers import _VLLMRLMClient

    class FakeTokenizer:
        def encode(self, text, add_special_tokens=False):
            return [int(x) for x in text.split()]

        def decode(self, token_ids, skip_special_tokens=False):
            return " ".join(str(x) for x in token_ids)

    llm = types.SimpleNamespace(llm_engine=types.SimpleNamespace(model_config=types.SimpleNamespace(max_model_len=13)))
    client = _VLLMRLMClient(
        llm,
        None,
        {"max_tokens": 3},
        tokenizer=FakeTokenizer(),
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
    import llm.eggroll as eggroll
    from experiments.llm import cli

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


def test_vllm_actor_defaults_disable_plugin_autoload(monkeypatch):
    from llm.vllm_actor_config import set_vllm_env_defaults

    monkeypatch.delenv("VLLM_PLUGINS", raising=False)

    set_vllm_env_defaults()

    assert "VLLM_PLUGINS" in __import__("os").environ
    assert __import__("os").environ["VLLM_PLUGINS"] == ""


def test_async_vllm_actor_universal_bridge(monkeypatch):
    import asyncio

    from llm.model_client import SampleCall
    from llm.tasks import RandomTask
    from llm.vllm_actor import AsyncTextVLLMActor, VLLMActorConfig

    # Mock dependencies to avoid real vLLM/Ray initialization
    vllm_mod = types.ModuleType("vllm")
    vllm_engine_mod = types.ModuleType("vllm.engine")
    vllm_async_mod = types.ModuleType("vllm.engine.async_llm_engine")
    vllm_arg_mod = types.ModuleType("vllm.engine.arg_utils")
    vllm_output_mod = types.ModuleType("vllm.outputs")

    vllm_async_mod.AsyncLLMEngine = FakeAsyncEngine
    vllm_arg_mod.AsyncEngineArgs = lambda **kwargs: kwargs
    vllm_output_mod.RequestOutput = FakeOutput
    vllm_mod.SamplingParams = lambda **kwargs: kwargs

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine", vllm_engine_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.async_llm_engine", vllm_async_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", vllm_arg_mod)
    monkeypatch.setitem(sys.modules, "vllm.outputs", vllm_output_mod)

    config = VLLMActorConfig(
        model_name="test",
        tensor_parallel_size=1,
        max_loras=1,
        lora_rank=8,
        max_tokens=10,
        prompt_batch_size=1,
        enforce_eager=True,
    )

    actor = AsyncTextVLLMActor(config=config)
    task = RandomTask(batch_size=1, max_random_number=4, seed=0)

    # Run the async bridge
    fitnesses, info, logs = asyncio.run(
        actor.generate_and_score_async(
            prompts=["2+2?"],
            sampling_params_kwargs={"temperature": 0.0},
            lora_request_specs=None,
            task_obj=task,
            answers=[4],
            args=types.SimpleNamespace(pass_at_k=False),
        )
    )

    assert fitnesses == [1.0]
    assert "final answer is 4" in logs[0]

    responses = asyncio.run(
        actor.sample(
            [
                SampleCall(
                    prompt="2+2?",
                    sampling={"temperature": 0.0, "max_tokens": 10},
                )
            ]
        )
    )

    assert responses[0].samples[0].text == "final answer is 4"


def test_async_vllm_actor_collective_rpc_is_async(monkeypatch):
    import asyncio

    from llm.vllm_actor import AsyncTextVLLMActor, VLLMActorConfig

    vllm_mod = types.ModuleType("vllm")
    vllm_engine_mod = types.ModuleType("vllm.engine")
    vllm_async_mod = types.ModuleType("vllm.engine.async_llm_engine")
    vllm_arg_mod = types.ModuleType("vllm.engine.arg_utils")

    vllm_async_mod.AsyncLLMEngine = FakeAsyncEngine
    vllm_arg_mod.AsyncEngineArgs = lambda **kwargs: kwargs

    monkeypatch.setitem(sys.modules, "vllm", vllm_mod)

    monkeypatch.setitem(sys.modules, "vllm.engine", vllm_engine_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.async_llm_engine", vllm_async_mod)
    monkeypatch.setitem(sys.modules, "vllm.engine.arg_utils", vllm_arg_mod)

    config = VLLMActorConfig(
        model_name="test",
        tensor_parallel_size=1,
        max_loras=1,
        lora_rank=8,
        max_tokens=10,
        prompt_batch_size=1,
        enforce_eager=True,
    )

    actor = AsyncTextVLLMActor(config=config)

    # Check that it is a coroutine and returns the right value
    res = actor.collective_rpc("test_method")
    assert asyncio.iscoroutine(res)

    final_res = asyncio.run(res)
    assert final_res == "rpc_res_test_method"
