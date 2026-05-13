from __future__ import annotations

import types


class FakeRLM:
    def __init__(self, client, model=None, rlm_max_turns=None, **kwargs):
        self.client = client

    async def run(self, task_payload):
        from tests.llm_test_mocks_state import FakeState

        res1 = await self.client.create([{"role": "user", "content": "2+2?"}])
        return FakeState({"reward": 1.0, "completion": [res1.choices[0].message]})

    async def rollout(self, task_payload):
        from tests.llm_test_mocks_state import FakeState

        return FakeState(
            {
                "reward": 1.0,
                "trajectory": [
                    types.SimpleNamespace(role="user", content="prompt"),
                    types.SimpleNamespace(role="assistant", content="thought"),
                    types.SimpleNamespace(role="assistant", content="QED"),
                ],
            }
        )


class FakeOutput:
    def __init__(self, text):
        self.outputs = [types.SimpleNamespace(text=text, finish_reason="stop", token_ids=[1, 2, 3])]


class FakeAsyncEngine:
    async def generate(self, prompt, sampling_params, request_id, lora_request=None):
        yield FakeOutput("final answer is 4")

    async def collective_rpc(self, method, args=()):
        return f"rpc_res_{method}"

    @classmethod
    def from_engine_args(cls, args):
        return cls()
