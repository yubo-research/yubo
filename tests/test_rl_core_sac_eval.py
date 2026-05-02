import pytest

from rl.core.sac_eval import (
    evaluate_heldout_with_best_actor,
    update_best_actor_if_improved,
)
from tests.rl_core_eval_heldout_helpers import run_evaluate_heldout_with_context


def test_sac_evaluate_heldout_with_best_actor_calls_eval_inside_context():
    run_evaluate_heldout_with_context(
        evaluate_heldout_with_best_actor,
        best_id=5,
        num_denoise_passive=3,
        heldout_i_noise=7,
        combine_fn=lambda nd, ino: float(nd + ino),
        expected_out=10.0,
    )


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
