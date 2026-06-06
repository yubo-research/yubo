from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch.nn as nn


def test_kiss_tidy_b_uhd_setups(monkeypatch):
    import ops.uhd_setup as uhd_setup
    import ops.uhd_setup_bszo as ub
    import ops.uhd_setup_bszo_core as bszoc
    import ops.uhd_setup_make_loop as um
    import ops.uhd_setup_monolith_bszo as mbzo
    import ops.uhd_setup_simple_gym as us
    from ops.uhd_setup_bszo_evaluate import make_bszo_gym_evaluate_fn, make_bszo_mnist_evaluate_fn
    from ops.uhd_setup_mnist_loop_eval import make_uhd_mnist_torch_evaluate_fn
    from ops.uhd_setup_monolith_bszo import run_bszo_loop as monolith_bszo_run_bszo_loop
    from ops.uhd_setup_monolith_make_loop import make_loop as monolith_make_loop

    assert callable(make_bszo_mnist_evaluate_fn) and callable(make_bszo_gym_evaluate_fn)
    assert callable(make_uhd_mnist_torch_evaluate_fn)

    with patch("ops.uhd_setup_monolith_make_loop.UHDDriver", lambda *a, **k: SimpleNamespace(run=lambda: None)):
        um.make_loop(
            "mnist",
            1,
            lr=0.01,
            sigma=0.01,
            num_dim_target=2,
            num_module_target=1,
            policy_tag="pure-function",
            problem_seed=0,
            noise_seed_0=0,
        )
        monolith_make_loop(
            "mnist",
            1,
            lr=0.01,
            sigma=0.01,
            num_dim_target=2,
            num_module_target=1,
            policy_tag="pure-function",
            problem_seed=0,
            noise_seed_0=0,
        )

    monkeypatch.setattr(mbzo, "_run_bszo_iterations", lambda *a, **k: None)
    ub.run_bszo_loop("mnist", 1, lr=0.01, policy_tag="pure-function", problem_seed=0, noise_seed_0=0)
    monolith_bszo_run_bszo_loop("mnist", 1, lr=0.01, policy_tag="pure-function", problem_seed=0, noise_seed_0=0)

    _lin = nn.Linear(1, 1, bias=False)

    def _gconf(*a, **k):
        return SimpleNamespace(
            noise_seed_0=0,
            problem_seed=0,
            frozen_noise=False,
            make_torch_env=lambda: SimpleNamespace(torch_env=lambda: SimpleNamespace(module=_lin)),
        )

    monkeypatch.setattr("problems.env_conf.get_env_conf", _gconf)
    monkeypatch.setattr(
        "optimizer.trajectories.collect_trajectory",
        lambda **k: SimpleNamespace(rreturn=0.1),
    )
    monkeypatch.setattr(bszoc, "_run_bszo_iterations", lambda *a, **k: None)

    def _fake_build_problem(env_tag, policy_tag, **kwargs):
        return SimpleNamespace(
            env=SimpleNamespace(problem_seed=None, make=lambda: object()),
        )

    monkeypatch.setattr("problems.problem.build_problem", _fake_build_problem)
    monkeypatch.setattr(uhd_setup, "_run_simple_gym", lambda *a, **k: None)
    us.run_simple_loop(
        "x",
        1,
        sigma=0.01,
        optimizer="simple",
        policy_tag="pure-function",
    )


def test_kiss_tidy_b_uhd_setups_import_wrappers():
    import ops.uhd_setup_bszo as ub
    import ops.uhd_setup_make_loop as um
    import ops.uhd_setup_simple_gym as us

    _bsz = Mock(k=0, eval_seed=0, y_best=None, ask=Mock(), tell=Mock())

    assert ub.run_bszo_loop.__name__ == "run_bszo_loop"
    assert um.make_loop.__name__ == "make_loop"
    assert us.run_simple_loop.__name__ == "run_simple_loop"
    ub._run_bszo_iterations(
        _bsz,
        evaluate_fn=lambda s: (0.0, 0.0),
        accuracy_fn=None,
        num_steps=0,
        log_interval=1,
        accuracy_interval=1,
        target_accuracy=None,
    )
