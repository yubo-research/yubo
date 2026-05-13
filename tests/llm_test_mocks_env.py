from __future__ import annotations


class FakeParser:
    def parse_answer(self, completion):
        text = completion[-1]["content"]
        marker = "\\boxed{"
        start = text.rfind(marker)
        if start < 0:
            return None
        end = text.find("}", start + len(marker))
        return None if end < 0 else text[start + len(marker) : end]


class FakeRubric:
    def __init__(self):
        self.parser = FakeParser()

    async def score_rollout(self, state):
        parsed = self.parser.parse_answer(state["completion"])
        state["reward"] = 1.0 if parsed == state["input"]["answer"] else 0.0
        state["metrics"] = {"correct_answer": state["reward"]}


class FakeEnv:
    requires_group_rollouts = False
    provides_advantages = False
    max_turns = 5

    def __init__(self):
        self.rubric = FakeRubric()

    def get_dataset(self, n=-1, seed=None):
        rows = [
            {
                "example_id": 0,
                "prompt": [
                    {"role": "system", "content": "Use boxes."},
                    {"role": "user", "content": "2+2?"},
                ],
                "answer": "4",
            },
            {
                "example_id": 1,
                "prompt": [{"role": "user", "content": "3+3?"}],
                "answer": "6",
            },
        ]
        return rows if n < 0 else rows[:n]
