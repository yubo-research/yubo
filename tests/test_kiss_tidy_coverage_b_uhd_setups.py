from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import Mock, patch

import torch.nn as nn


def test_kiss_tidy_b_uhd_setups(monkeypatch):
    import ops.uhd_setup_bszo_core as bszoc
    import ops.uhd_setup_make_loop_impl as mli
    import ops.uhd_setup_monolith_bszo as mbzo
    import ops.uhd_setup_monolith_make_loop as mml
    import ops.uhd_setup_simple_gym_impl as sgi

    with patch("optimizer.uhd_loop.UHDLoop", lambda *a, **k: SimpleNamespace(run=lambda: None)):
        mli.make_loop(
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
        mml.make_loop(
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
    mbzo.run_bszo_loop("mnist", 1, lr=0.01, policy_tag="pure-function", problem_seed=0, noise_seed_0=0)

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
    bszoc.run_bszo_loop("x", 1, lr=0.01, problem_seed=0, noise_seed_0=0)

    def _sgc(*a, **k):
        return SimpleNamespace(problem_seed=None, make=lambda: object())

    monkeypatch.setattr("problems.env_conf.get_env_conf", _sgc)
    monkeypatch.setattr(sgi, "_run_simple_gym", lambda *a, **k: None)
    sgi.run_simple_loop("x", 1, sigma=0.01, optimizer="simple")


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
