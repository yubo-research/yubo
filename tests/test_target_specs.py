def test_target_spec_dataclass():
    from uhd.target_specs import TargetSpec

    spec = TargetSpec(
        requires_dims=True,
        controller_type_factory=lambda: None,
        make_adamw_metric=lambda: None,
        make_bo_metric=lambda: None,
    )
    assert spec.requires_dims is True


def test_target_specs_dict():
    from uhd.target_specs import TARGET_SPECS

    assert "tm_sphere" in TARGET_SPECS
    assert "tm_ackley" in TARGET_SPECS
    assert "tm_mnist" in TARGET_SPECS


def test_sphere_spec():
    from uhd.target_specs import TARGET_SPECS

    spec = TARGET_SPECS["tm_sphere"]
    assert spec.requires_dims is True
    factory = spec.controller_type_factory
    assert callable(factory)


def test_ackley_spec():
    from uhd.target_specs import TARGET_SPECS

    spec = TARGET_SPECS["tm_ackley"]
    assert spec.requires_dims is True
