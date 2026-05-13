from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np


if TYPE_CHECKING:
    from experiments.enn_turbo_scale_check import (
        ScaleRow,
        build_rows,
        check_coordinate_sampling,
        check_failure_clock,
        print_table,
        raasp_mean_theory,
        sample_raasp_support,
    )
    from experiments.enn_turbo_scale_check import (
        main as enn_turbo_main,
    )

    _ENN_TURBO_REFS = (
        ScaleRow,
        raasp_mean_theory,
        sample_raasp_support,
        check_coordinate_sampling,
        check_failure_clock,
        build_rows,
        print_table,
        enn_turbo_main,
    )


def test_added_mapping_key_helpers():
    from common.mapping_keys import coerce_mapping_keys, normalize_toml_key

    assert normalize_toml_key("foo-bar") == "foo_bar"
    assert coerce_mapping_keys(
        {"foo-bar": 1},
        source="test",
        valid_keys={"foo_bar"},
        not_mapping_msg="not mapping",
    ) == {"foo_bar": 1}


def test_added_external_run_utils_helpers(tmp_path):
    from experiments.external_run_utils import (
        abs_path,
        append_cli_arg,
        deep_update,
        log_path,
        normalize_mapping,
        optional_abs_path,
        parse_section_overrides,
        parse_toml_value,
        quote_command,
        string_env_vars,
        write_metadata,
    )

    assert normalize_mapping({"foo-bar": 1}, source="cfg", valid_keys={"foo_bar"}) == {"foo_bar": 1}
    assert deep_update({"a": {"b": 1}, "c": 0}, {"a": {"d": 2}}) == {
        "a": {"b": 1, "d": 2},
        "c": 0,
    }
    assert parse_toml_value("true") is True
    assert parse_section_overrides(("llm.population-size=4",), valid_by_section={"llm": {"population_size"}}) == {"llm": {"population_size": 4}}

    base = tmp_path / "base"
    rel = abs_path("child", base=base)
    assert rel == (base / "child").resolve()
    assert optional_abs_path("", base=base) is None
    assert optional_abs_path("child", base=base) == str(rel)
    assert log_path(base, None, default="logs/out.txt") == base / "logs" / "out.txt"
    assert string_env_vars({"A": 1}) == {"A": "1"}
    assert quote_command(["echo", "hello world"]) == "echo 'hello world'"

    cmd = ["run"]
    append_cli_arg(cmd, "dry_run", True)
    append_cli_arg(cmd, "name", "x")
    append_cli_arg(cmd, "item", [1, 2])
    assert cmd == ["run", "--dry-run", "--name", "x", "--item", "1", "--item", "2"]

    exp_dir = tmp_path / "exp"
    write_metadata(exp_dir, {"b": 2}, ["echo", "ok"])
    assert (exp_dir / "config.json").exists()
    assert (exp_dir / "command.txt").read_text(encoding="utf-8").strip() == "echo ok"


def test_added_uhd_enn_shims_are_connected(monkeypatch):
    import optimizer.uhd_enn_imputers as imputers
    import optimizer.uhd_enn_regression as regression

    monkeypatch.setitem(
        __import__("sys").modules,
        "optimizer.uhd_simple_be",
        type(
            "_SimpleBE",
            (),
            {
                "_fit_enn": staticmethod(lambda zs, ys, enn_k: ("fit", len(zs), len(ys), int(enn_k))),
                "_predict_enn": staticmethod(lambda _model, _params, x: (np.ones((len(x),)), np.zeros((len(x),)))),
            },
        ),
    )

    assert regression.fit_enn([np.array([1.0])], [1.0], 3) == ("fit", 1, 1, 3)
    assert regression.predict_enn("m", "p", np.zeros((2, 1)))[0].tolist() == [1.0, 1.0]
    assert callable(imputers.JAXMinusImputer)
    assert callable(imputers.JAXPointImputer)


def test_added_llm_task_modules_are_connected():
    from llm.tasks_base import LLMTask
    from llm.tasks_countdown import CountdownTask
    from llm.tasks_factory import build_task
    from llm.tasks_math import MathTask
    from llm.tasks_static import RandomTask, ZerosTask

    assert all(callable(obj) for obj in (LLMTask, CountdownTask, MathTask, RandomTask, ZerosTask, build_task))


def test_added_text_and_pretrain_objective_modules_are_connected():
    from problems import (
        pre_obj_hyperscalees,
        pre_obj_nanoegg,
        text_obj_cache,
        text_obj_lora,
        text_obj_objective,
    )

    assert callable(pre_obj_hyperscalees.HyperscaleESLLMVectorObjective)
    assert callable(pre_obj_nanoegg.NanoEggPretrainVectorObjective)
    assert callable(text_obj_cache._PromptBatchCache)
    assert callable(text_obj_lora._LoraSubspaceCodec)
    assert callable(text_obj_objective.TextObjective)
