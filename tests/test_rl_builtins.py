from rl import builtins
from rl.registry import available_algos, get_algo


def test_register_all():
    builtins.register_all()
    algos = available_algos()
    assert "ppo" in algos
    assert "sac" in algos
    assert get_algo("ppo").config_cls is not None
    assert get_algo("sac").config_cls is not None
