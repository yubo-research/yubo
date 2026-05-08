from types import SimpleNamespace


def test_eggroll_solver_accepts_muon_when_optax_exposes_it():
    from optimizer.eggroll_designer import _solver, _solver_kwargs

    def dummy_muon(*args, **kwargs):
        return args, kwargs

    optax_mod = SimpleNamespace(
        adam=object(),
        adamw=object(),
        sgd=object(),
        contrib=SimpleNamespace(muon=dummy_muon),
    )

    assert _solver(optax_mod, "muon", b1=0.9, b2=0.999, weight_decay=0.1) is dummy_muon
    assert _solver_kwargs("muon", b1=0.9, b2=0.999, weight_decay=0.1) == {"weight_decay": 0.1}


def test_eggroll_solver_rejects_muon_when_optax_does_not_expose_it():
    import pytest

    from optimizer.eggroll_designer import _solver
    from optimizer.designer_errors import NoSuchDesignerError

    optax_mod = SimpleNamespace(adam=object(), adamw=object(), sgd=object())

    with pytest.raises(NoSuchDesignerError, match="adam, adamw, sgd"):
        _solver(optax_mod, "muon", b1=0.9, b2=0.999, weight_decay=0.1)
