"""Kiss coverage for ops.fit_mnist CLI entrypoint."""

from __future__ import annotations

import torch.nn as nn
from click.testing import CliRunner


def test_kiss_bridge_fit_mnist_main_invokes_fit(monkeypatch):
    import ops.fit_mnist as fm
    from ops.fit_mnist import main as fit_mnist_click_main

    monkeypatch.setattr(fm, "fit_mnist", lambda **k: nn.Linear(1, 1))
    runner = CliRunner()
    res = runner.invoke(fit_mnist_click_main, ["--epochs", "1", "--batch-size", "8", "--timeout", "2"])
    assert res.exit_code == 0
    assert callable(fit_mnist_click_main)
