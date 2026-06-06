from __future__ import annotations


def _run_bszo_iterations(
    bszo,
    *,
    evaluate_fn,
    accuracy_fn,
    num_steps: int,
    log_interval: int,
    accuracy_interval: int,
    target_accuracy: float | None,
) -> None:
    import time

    from ops.uhd_setup_simple_common import _should_log_simple

    k = bszo.k
    t0 = time.perf_counter()
    acc = None
    step = 0
    for step in range(num_steps):
        for _ in range(k + 1):
            bszo.ask()
            mu, se = evaluate_fn(bszo.eval_seed)
            bszo.tell(mu, se)

        if not _should_log_simple(step, num_steps, log_interval):
            continue
        y_best = bszo.y_best
        y_str = f"{y_best:.4f}" if y_best is not None else "N/A"
        if accuracy_fn is not None and (acc is None or step == num_steps - 1 or (accuracy_interval > 0 and step % accuracy_interval == 0)):
            acc = accuracy_fn()
        from optimizer.uhd_loop_support import format_uhd_eval_line

        line = format_uhd_eval_line(
            i_iter=step,
            proposal_dt=0.0,
            eval_dt=0.0,
            sigma=bszo.epsilon,
            mu=mu,
            se=se,
            y_best_str=y_str,
        )
        if acc is not None:
            line += f" test_acc = {acc:.4f}"
        print(line)
        if target_accuracy is not None and acc is not None and acc >= target_accuracy:
            elapsed = time.perf_counter() - t0
            print(f"BSZO: target reached {acc:.4f} >= {target_accuracy:.4f} at step={step} ({elapsed:.2f}s)")
            break

    elapsed = time.perf_counter() - t0
    print(f"BSZO: elapsed = {elapsed:.2f}s ({min(step + 1, num_steps)} steps)")
