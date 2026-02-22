"""Tests for common.config_toml."""

import pytest

from common.config_toml import apply_overrides, load_toml, parse_set_args, parse_value


def test_load_toml(tmp_path):
    toml_path = tmp_path / "cfg.toml"
    toml_path.write_text("[a]\nb = 1\n", encoding="utf-8")
    cfg = load_toml(str(toml_path))
    assert cfg == {"a": {"b": 1}}


def test_load_toml_missing_raises():
    with pytest.raises(FileNotFoundError, match="nonexistent"):
        load_toml("nonexistent.toml")


def test_apply_overrides_shallow():
    cfg = {"a": 1, "b": 2}
    out = apply_overrides(cfg, {"a": 10})
    assert out["a"] == 10
    assert out["b"] == 2


def test_apply_overrides_nested():
    cfg = {"x": {"y": 1}}
    out = apply_overrides(cfg, {"x.z": 2})
    assert out["x"]["y"] == 1
    assert out["x"]["z"] == 2


def test_apply_overrides_non_dict_path_raises():
    cfg = {"a": "not_a_dict"}
    with pytest.raises(ValueError, match="expected table"):
        apply_overrides(cfg, {"a.b": 1})


def test_parse_value_none():
    assert parse_value("none") is None
    assert parse_value("null") is None


def test_parse_value_bool():
    assert parse_value("true") is True
    assert parse_value("false") is False


def test_parse_value_int_float_str():
    assert parse_value("42") == 42
    assert parse_value("3.14") == 3.14
    assert parse_value("hello") == "hello"


def test_parse_set_args():
    out = parse_set_args(["--set", "a=1", "--set", "b=2.0"])
    assert out == {"a": 1, "b": 2.0}


def test_parse_set_args_equals_form():
    out = parse_set_args(["--set=a=true", "--set=b=none"])
    assert out == {"a": True, "b": None}


def test_parse_set_args_missing_value_raises():
    with pytest.raises(ValueError, match="Expected KEY=VALUE"):
        parse_set_args(["--set"])


def test_parse_set_args_unknown_arg_raises():
    with pytest.raises(ValueError, match="Unknown argument"):
        parse_set_args(["--other"])
