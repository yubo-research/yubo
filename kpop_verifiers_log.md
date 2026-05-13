# KPop Experiment Log: Verifiers `uvloop` Patching Error

## Problem Statement
The `verifiers` integration fails with `ValueError: Can't patch loop of type <class 'uvloop.Loop'>`.
This occurs because `llm/tasks_verifiers.py` uses `nest_asyncio.apply()` to support running async code (like scoring rubrics) from synchronous contexts.
However, Ray/vLLM uses `uvloop`, which `nest_asyncio` does not support.

---

## Cycle 1: Eliminating Unnecessary `nest_asyncio` Calls

### Hypothesize
The `ValueError` is triggered by calling `_run_async` (which applies `nest_asyncio`) from within an already asynchronous context (`run_single` in `generate_and_score_async`).
In this context, we can simply `await` the coroutine directly, avoiding `nest_asyncio` and the `uvloop` incompatibility.

### Predict
If we replace `_run_async(env.rubric.score_rollout(state))` with `await env.rubric.score_rollout(state)` inside the `async` agentic loop, the `ValueError` will be bypassed during the primary generation/scoring phase.

### Falsify
- Update `llm/tasks_verifiers.py` to `await` the rubric in `run_single`.
- Run the experiment.
- If it still crashes with the same error during scoring, the hypothesis is rejected.

---
