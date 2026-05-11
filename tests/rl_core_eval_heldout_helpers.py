from __future__ import annotations

from contextlib import contextmanager


def run_evaluate_heldout_with_context(
    evaluate_heldout_with_best_actor,
    *,
    best_id: int,
    num_denoise_passive: int,
    heldout_i_noise: int,
    combine_fn,
    expected_out: float,
):
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
        return combine_fn(num_denoise, i_noise)

    out = evaluate_heldout_with_best_actor(
        best_actor_state={"id": best_id},
        num_denoise_passive=num_denoise_passive,
        heldout_i_noise=heldout_i_noise,
        with_actor_state=_with_actor_state,
        evaluate_for_best=_evaluate_for_best,
        eval_env_conf=object(),
        eval_policy=object(),
    )
    assert out == expected_out
    assert calls == [("enter", best_id), ("eval", heldout_i_noise), ("exit", best_id)]
