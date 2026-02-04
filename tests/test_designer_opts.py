import pytest

from optimizer.designer_errors import NoSuchDesignerError
from optimizer.designer_opts import OptParser


def test_opt_parser_require_int_and_optional_int():
    opts = {"k": 3}
    assert OptParser.require_int(opts, "k", example="x/k=3") == 3
    assert OptParser.optional_int(opts, "k") == 3
    assert OptParser.optional_int(opts, "missing") is None
    with pytest.raises(NoSuchDesignerError):
        OptParser.require_int({}, "k", example="x/k=3")
    with pytest.raises(NoSuchDesignerError):
        OptParser.optional_int({"k": "3"}, "k")


def test_opt_parser_require_str_in_and_optional_str_in():
    allowed = {"a", "b"}
    opts = {"mode": "a"}
    assert OptParser.require_str_in(opts, "mode", allowed, example="x/mode=a") == "a"
    assert OptParser.optional_str_in(opts, "mode", allowed) == "a"
    assert OptParser.optional_str_in(opts, "missing", allowed) is None
    with pytest.raises(NoSuchDesignerError):
        OptParser.require_str_in({}, "mode", allowed, example="x/mode=a")
    with pytest.raises(NoSuchDesignerError):
        OptParser.require_str_in({"mode": 1}, "mode", allowed, example="x/mode=a")
    with pytest.raises(NoSuchDesignerError):
        OptParser.optional_str_in({"mode": "c"}, "mode", allowed)


def test_opt_parser_optional_float():
    opts = {"len": 1.6}
    assert OptParser.optional_float(opts, "len") == 1.6
    assert OptParser.optional_float({}, "len") is None
    with pytest.raises(NoSuchDesignerError):
        OptParser.optional_float({"len": "1.6"}, "len")
