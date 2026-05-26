from __future__ import annotations


def test_RunFields() -> None:
    from ops.modal_uhd_runner_fields import RunFields

    fields = RunFields(
        env_tag="mnist",
        num_rounds=1,
        policy_tag=None,
        lr=0.1,
        sigma=0.01,
        ndt=None,
        nmt=None,
        problem_seed=None,
        noise_seed_0=None,
        log_interval=10,
        accuracy_interval=100,
        target_accuracy=None,
    )
    assert fields.env_tag == "mnist"
    assert fields.num_rounds == 1


def test_EarlyRejectFields() -> None:
    from ops.modal_uhd_runner_fields import EarlyRejectFields

    fields = EarlyRejectFields(
        tau=None,
        mode=None,
        ema_beta=None,
        warmup_pos=None,
        quantile=None,
        window=None,
    )
    assert fields.tau is None
    assert fields.mode is None
