"""Shared fakes for modal timing sweep tests (kiss: avoid per-test class definitions)."""

from __future__ import annotations

FakeResultsDict = type("FakeResultsDict", (dict,), {"len": lambda self: len(self)})


def make_func_spawn_map(capture: list):
    return type(
        "FuncSpawnMap",
        (),
        {"spawn_map": lambda self, todo: capture.extend(list(todo))},
    )()


def make_func_spawn(capture: list):
    return type(
        "FuncSpawn",
        (),
        {"spawn": lambda self, payload: capture.append(payload)},
    )()
