import numpy as np
import pytest


class MockPolicy:
    def __init__(self, num_params=5):
        self._num_params = num_params
        self.problem_seed = 0

    def num_params(self):
        return self._num_params

    def set_params(self, x):
        pass

    def get_params(self):
        return np.zeros(self._num_params)

    def clone(self):
        return MockPolicy(self._num_params)


def test_designers_init():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    assert designers is not None


def test_designers_create_random():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("random")
    assert designer is not None


def test_designers_create_sobol():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("sobol")
    assert designer is not None


def test_designers_create_lhd():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("lhd")
    assert designer is not None


def test_designers_create_center():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    designer = designers.create("center")
    assert designer is not None


def test_designers_no_such_designer():
    from optimizer.designers import Designers, NoSuchDesignerError

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    with pytest.raises(NoSuchDesignerError):
        designers.create("nonexistent_designer")


def test_designers_is_valid():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    assert designers.is_valid("random") is True
    assert designers.is_valid("nonexistent_designer") is False


def test_designers_catalog():
    from optimizer.designers import Designers

    policy = MockPolicy()
    designers = Designers(policy, num_arms=1)
    catalog = designers.catalog()
    assert isinstance(catalog, list)
    names = {entry.base_name for entry in catalog}
    assert "sobol" in names


def test_designers_catalog_dataclasses_are_instantiable():
    # This is intentionally direct: it ensures the catalog-related dataclasses
    # are covered by tests (for `kiss check` coverage gating).
    from optimizer.designer_types import DesignerDef
    from optimizer.designers import (
        DesignerCatalogEntry,
        DesignerOptionSpec,
        DesignerSpec,
    )

    opt = DesignerOptionSpec(
        name="k",
        required=True,
        value_type="int",
        description="dummy",
        example_suffix="k=1",
        allowed_values=None,
    )
    assert opt.example("x") == "x/k=1"
    entry = DesignerCatalogEntry(base_name="x", options=[opt], dispatch=lambda *_: None)
    spec = DesignerSpec(base="x", general={}, specific={})

    assert entry.base_name == "x"
    assert entry.options[0].name == "k"
    assert spec.base == "x"

    def dummy_builder(_ctx, _opts):
        pass

    def_with_opts = DesignerDef(name="test", builder=dummy_builder, option_specs=(opt,))
    def_no_opts = DesignerDef(name="test2", builder=dummy_builder, option_specs=())

    assert def_with_opts.name == "test"
    assert def_with_opts.has_options() is True
    assert def_no_opts.has_options() is False
