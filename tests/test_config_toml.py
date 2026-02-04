from common.config_toml import apply_overrides, load_toml, parse_value


def test_load_toml_roundtrip(tmp_path):
    path = tmp_path / "cfg.toml"
    path.write_text("[a]\nb=1\n", encoding="utf-8")
    cfg = load_toml(str(path))
    assert cfg["a"]["b"] == 1


def test_apply_overrides_nested_keys():
    cfg = {"a": {"b": 1}}
    out = apply_overrides(cfg, {"a.c": 2, "d": 3})
    assert out["a"]["b"] == 1
    assert out["a"]["c"] == 2
    assert out["d"] == 3


def test_parse_value_typed_literals():
    assert parse_value("none") is None
    assert parse_value("TRUE") is True
    assert parse_value("false") is False
    assert parse_value("3") == 3
    assert parse_value("3.25") == 3.25
    assert parse_value("hello") == "hello"
