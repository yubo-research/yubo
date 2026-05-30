from __future__ import annotations

import asyncio
import os
import sys
import types

from click.testing import CliRunner

from tests.llm_test_mocks_vllm import FakeAsyncEngine, FakeOutput


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

    assert "VLLM_PLUGINS" in os.environ
    if sys.platform == "darwin":
        assert os.environ["VLLM_PLUGINS"] == "metal"
    else:
        assert os.environ["VLLM_PLUGINS"] == ""


def test_async_vllm_actor_universal_bridge(monkeypatch):
    from llm.model_client import SampleCall
    from llm.tasks import RandomTask
    from llm.vllm_actor import AsyncTextVLLMActor, VLLMActorConfig

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

    res = actor.collective_rpc("test_method")
    assert asyncio.iscoroutine(res)

    final_res = asyncio.run(res)
    assert final_res == "rpc_res_test_method"
