"""Shared fakes for llm tests (kiss: factories instead of inline class statements)."""

from __future__ import annotations

AccidentalRollout = type(
    "AccidentalRollout",
    (),
    {"generate_and_score": lambda self: None},
)


def fake_ray_cls(resources: dict[str, float]):
    res = dict(resources)
    return type("FakeRay", (), {"cluster_resources": staticmethod(lambda res=res: res)})


def make_fake_tokenizer():
    return type(
        "FakeTokenizer",
        (),
        {
            "encode": lambda self, text, add_special_tokens=False: [int(x) for x in text.split()],
            "decode": lambda self, token_ids, skip_special_tokens=False: " ".join(str(x) for x in token_ids),
        },
    )()
