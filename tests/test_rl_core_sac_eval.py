from contextlib import contextmanager

import pytest

from rl.core.sac_eval import evaluate_heldout_with_best_actor, update_best_actor_if_improved


def test_sac_evaluate_heldout_with_best_actor_calls_eval_inside_context():
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
        return float(num_denoise + i_noise)

    result = evaluate_heldout_with_best_actor(
        best_actor_state={"id": 5},
        num_denoise_passive_eval=3,
        heldout_i_noise=7,
        with_actor_state=_with_actor_state,
        evaluate_for_best=_evaluate_for_best,
        eval_env_conf=object(),
        eval_policy=object(),
    )
    assert result == 10.0
    assert calls == [("enter", 5), ("eval", 7), ("exit", 5)]


@pytest.mark.parametrize(
    ("eval_return", "best_return", "expected_improved"),
    [(4.0, 3.0, True), (2.0, 3.0, False)],
)
def test_sac_update_best_actor_if_improved_variants(eval_return, best_return, expected_improved):
    calls = {"n": 0}

    def _capture():
        calls["n"] += 1
        return {"snap": calls["n"]}

    prior = {"snap": 9}
    best_return, best_actor_state, improved = update_best_actor_if_improved(
        eval_return=eval_return,
        best_return=best_return,
        best_actor_state=prior,
        capture_actor_state=_capture,
    )
    assert improved is expected_improved
    if expected_improved:
        assert best_return == eval_return
        assert best_actor_state == {"snap": 1}
        assert calls["n"] == 1
    else:
        assert best_return == 3.0
        assert best_actor_state is prior
        assert calls["n"] == 0
