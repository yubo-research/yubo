from rl.algos import builtins
from rl.algos.registry import AlgoSpec, available_algos, get_algo, register_algo


def test_algos_registry_registers_ppo():
    builtins.register_all()
    algo = get_algo("ppo")
    assert algo.name == "ppo"
    assert "ppo" in available_algos()


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
