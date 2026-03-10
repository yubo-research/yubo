from contextlib import contextmanager

from rl.core.ppo_eval import evaluate_heldout_with_best_actor, update_best_actor_if_improved


def test_evaluate_heldout_with_best_actor_calls_eval_inside_context():
    calls: list[tuple[str, int | None]] = []

    @contextmanager
    def _with_actor_state(snapshot):
        calls.append(("enter", snapshot.get("id")))
        try:
            yield
        finally:
            calls.append(("exit", snapshot.get("id")))

    def _evaluate_for_best(env_conf, policy, num_denoise, *, i_noise):
        calls.append(("eval", int(i_noise)))
        _ = env_conf, policy
        return float(num_denoise) + float(i_noise)

    out = evaluate_heldout_with_best_actor(
        best_actor_state={"id": 7},
        num_denoise_passive=3,
        heldout_i_noise=11,
        with_actor_state=_with_actor_state,
        evaluate_for_best=_evaluate_for_best,
        eval_env_conf=object(),
        eval_policy=object(),
    )
    assert out == 14.0
    assert calls == [("enter", 7), ("eval", 11), ("exit", 7)]


def test_evaluate_heldout_with_best_actor_returns_none_if_disabled():
    called = {"eval": False}

    @contextmanager
    def _with_actor_state(_snapshot):
        yield

    def _evaluate_for_best(*_args, **_kwargs):
        called["eval"] = True
        return 1.0

    out = evaluate_heldout_with_best_actor(
        best_actor_state={"id": 1},
        num_denoise_passive=None,
        heldout_i_noise=5,
        with_actor_state=_with_actor_state,
        evaluate_for_best=_evaluate_for_best,
        eval_env_conf=object(),
        eval_policy=object(),
    )
    assert out is None
    assert called["eval"] is False


def test_update_best_actor_if_improved_updates_snapshot():
    calls = {"n": 0}

    def _capture():
        calls["n"] += 1
        return {"snap": calls["n"]}

    best_return, best_actor_state, improved = update_best_actor_if_improved(
        eval_return=10.0,
        best_return=8.0,
        best_actor_state={"snap": 0},
        capture_actor_state=_capture,
    )
    assert improved is True
    assert best_return == 10.0
    assert best_actor_state == {"snap": 1}
    assert calls["n"] == 1


def test_update_best_actor_if_improved_keeps_previous_state():
    calls = {"n": 0}

    def _capture():
        calls["n"] += 1
        return {"snap": calls["n"]}

    prior = {"snap": 7}
    best_return, best_actor_state, improved = update_best_actor_if_improved(
        eval_return=4.0,
        best_return=8.0,
        best_actor_state=prior,
        capture_actor_state=_capture,
    )
    assert improved is False
    assert best_return == 8.0
    assert best_actor_state is prior
    assert calls["n"] == 0
