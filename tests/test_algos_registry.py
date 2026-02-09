from rl.algos import builtins
from rl.algos.registry import AlgoSpec, available_algos, get_algo, register_algo


def test_algos_registry_registers_builtin_algos():
    builtins.register_all()
    ppo_algo = get_algo("ppo")
    sac_algo = get_algo("sac")
    assert ppo_algo.name == "ppo"
    assert sac_algo.name == "sac"
    assert "ppo" in available_algos()
    assert "sac" in available_algos()


def test_algos_registry_custom_register():
    class DummyConfig:
        @classmethod
        def from_dict(cls, _raw: dict):
            return cls()

    def train(_cfg: DummyConfig):
        return None

    spec = AlgoSpec(name="dummy", config_cls=DummyConfig, train_fn=train)
    assert spec.name == "dummy"

    name = "__pytest_dummy_algo__"
    register_algo(name, DummyConfig, train)
    assert get_algo(name).name == name
