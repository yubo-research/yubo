from types import SimpleNamespace

import pytest

from rl.registry import (
    AlgoSpec,
    available_algos,
    get_algo,
    register_algo,
    register_algo_backend,
    register_algo_lazy,
)


def test_algo_spec():
    spec = AlgoSpec(name="test", config_cls=SimpleNamespace, train_fn=lambda: None)
    assert spec.name == "test"
    assert spec.config_cls is SimpleNamespace


def test_register_and_get_algo():
    def _train():
        pass

    register_algo("_test_reg", SimpleNamespace, _train)
    spec = get_algo("_test_reg")
    assert spec.name == "_test_reg"
    assert spec.train_fn is _train
    assert "_test_reg" in available_algos()


def test_register_duplicate_raises():
    def _train():
        pass

    register_algo("_test_dup", type(None), _train)
    with pytest.raises(ValueError, match="already registered"):
        register_algo("_test_dup", type(None), _train)


def test_get_unknown_algo_raises():
    with pytest.raises(ValueError, match="Unknown algorithm"):
        get_algo("_nonexistent_algo_xyz")


def test_register_algo_lazy_registers_on_first_get(monkeypatch):
    import rl.registry as registry

    lazy_name = "_lazy_reg_test"
    lazy_module = "rl._fake_lazy_for_test"
    called = {"register": False}

    class _LazyModule:
        @staticmethod
        def register():
            called["register"] = True
            register_algo(lazy_name, SimpleNamespace, lambda: None)

    monkeypatch.setattr(
        registry.importlib,
        "import_module",
        lambda name: _LazyModule if name == lazy_module else None,
    )
    register_algo_lazy(lazy_name, lazy_module)
    spec = get_algo(lazy_name)
    assert called["register"] is True
    assert spec.name == lazy_name


def test_register_algo_lazy_duplicate_raises():
    name = "_lazy_dup_test"
    register_algo_lazy(name, "rl._fake_lazy_dup_1")
    with pytest.raises(ValueError, match="already registered"):
        register_algo_lazy(name, "rl._fake_lazy_dup_2")


def test_register_algo_backend_and_resolve_get_algo():
    def _train():
        pass

    register_algo("_impl_backend_reg", SimpleNamespace, _train)
    register_algo_backend("_canonical_backend_reg", "pufferlib", "_impl_backend_reg")

    spec = get_algo("_canonical_backend_reg", backend="pufferlib")
    assert spec.name == "_impl_backend_reg"
    assert spec.train_fn is _train


def test_register_algo_backend_duplicate_same_target_is_ok():
    register_algo_backend("_canonical_backend_idem", "torchrl", "_impl_backend_idem")
    register_algo_backend("_canonical_backend_idem", "torchrl", "_impl_backend_idem")


def test_register_algo_backend_conflicting_target_raises():
    register_algo_backend("_canonical_backend_conflict", "torchrl", "_impl_backend_a")
    with pytest.raises(ValueError, match="already points to"):
        register_algo_backend("_canonical_backend_conflict", "torchrl", "_impl_backend_b")


def test_get_algo_unknown_backend_for_bound_algo_raises():
    def _train():
        pass

    register_algo("_impl_backend_unknown", SimpleNamespace, _train)
    register_algo_backend("_canonical_backend_unknown", "torchrl", "_impl_backend_unknown")
    with pytest.raises(ValueError, match="Unknown backend"):
        get_algo("_canonical_backend_unknown", backend="pufferlib")
