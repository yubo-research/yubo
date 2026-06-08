from rl import builtins
from rl.registry import available_algos


def test_register_all():
    builtins.register_all()
    algos = available_algos()
    assert "ppo" in algos
    assert "sac" in algos
