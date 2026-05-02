from contextlib import contextmanager

from rl.core.ppo_eval import (
    evaluate_heldout_with_best_actor,
    update_best_actor_if_improved,
)
from tests.rl_core_eval_heldout_helpers import run_evaluate_heldout_with_context


def test_evaluate_heldout_with_best_actor_calls_eval_inside_context():
    run_evaluate_heldout_with_context(
        evaluate_heldout_with_best_actor,
        best_id=7,
        num_denoise_passive=3,
        heldout_i_noise=11,
        combine_fn=lambda nd, ino: float(nd) + float(ino),
        expected_out=14.0,
    )


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
