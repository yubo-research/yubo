from rl import builtins
from rl.registry import available_algos, available_backend_bindings


def test_register_all():
    builtins.register_all()
    algos = available_algos()
    assert "ppo" in algos
    assert "sac" in algos
    assert "r2d2" in algos
    bindings = available_backend_bindings()
    assert bindings[("ppo", "torchrl")] == "ppo"
    assert bindings[("ppo", "pufferlib")] == "ppo"
    assert bindings[("sac", "torchrl")] == "sac"
    assert bindings[("sac", "pufferlib")] == "sac"
