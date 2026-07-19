from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np
import pytest

from analysis.data_io import TraceRecord
from analysis.uhd_paper_cheetah import (
    BO_UHD_POLICIES,
    CHEETAH_POLICY_NUM_PARAMS,
    CheetahExperimentPaths,
    bo_coverage_report,
    final_best_summary,
    load_cheetah_bo_uhd_comparison,
    load_uhd_batch_cheetah,
    load_uhd_cheetah_simple,
    loaded_traces_summary,
    normalize_bo_label,
    plot_cheetah_bo_uhd_comparison,
    plot_uhd_batch_cheetah,
)
from analysis.uhd_paper_trace import (
    best_so_far_array,
    sem_1d,
    summarize_curve,
    traces_to_array,
)


def _trace(returns: list[float]) -> list[TraceRecord]:
    return [TraceRecord(i_iter=i, dt_prop=0.0, dt_eval=0.0, rreturn=float(r)) for i, r in enumerate(returns)]


def test_normalize_bo_label_eggroll_sigma():
    assert normalize_bo_label("random") == "random"
    assert normalize_bo_label("turbo-enn-fit-ucb") == "turbo-enn-fit-ucb"
    opt05 = "eggroll/noiser=eggroll/sigma=0.05/rank=1/lr=0.01/optax=adamw/batch_size=10/num_envs=1/steps=1000"
    opt10 = "eggroll/noiser=eggroll/sigma=0.1/rank=1/lr=0.01/optax=adamw/batch_size=10/num_envs=1/steps=1000"
    opt_adapt = "eggroll/noiser=eggroll/sigma=0.1/adapt_sigma=true/rank=1/lr=0.01/optax=adamw/batch_size=10/num_envs=1/steps=1000"
    assert normalize_bo_label(opt05) == "eggroll (sigma=0.05)"
    assert normalize_bo_label(opt10) == "eggroll (sigma=0.1)"
    assert normalize_bo_label(opt_adapt) == "eggroll (adapt sigma=0.1)"


def test_label_uhd_config():
    from analysis.uhd_paper_cheetah import label_uhd_config, legend_label

    assert label_uhd_config({"optimizer": "simple"}) == "simple"
    assert (
        label_uhd_config(
            {
                "optimizer": "simple_be",
                "be_enn_index_driver": "flat",
                "be_acquisition": "ucb",
            }
        )
        == "simple_be/flat/ucb"
    )
    assert legend_label("eggroll (sigma=0.05)") == r"eggroll ($\sigma$=0.05)"
    assert legend_label("eggroll (sigma=0.1)") == r"eggroll ($\sigma$=0.1)"
    assert legend_label("eggroll (sigma=0.2)") == r"eggroll ($\sigma$=0.2)"
    assert legend_label("eggroll (adapt sigma=0.1)") == r"eggroll (adapt $\sigma$=0.1)"
    assert legend_label("random") == "random"


def test_summarize_curve_and_best_so_far():
    traces = [_trace([1.0, 3.0, 2.0]), _trace([0.0, 2.0, 4.0])]
    arr = best_so_far_array(traces)
    assert arr.shape == (2, 3)
    assert list(arr[0]) == [1.0, 3.0, 3.0]
    assert list(arr[1]) == [0.0, 2.0, 4.0]
    curve = summarize_curve(traces)
    assert curve["n"] == 2
    np.testing.assert_allclose(curve["mean"], [0.5, 2.5, 3.5])


def test_sem_1d():
    assert np.isnan(sem_1d(np.array([1.0])))
    assert sem_1d(np.array([1.0, 3.0])) == pytest.approx(1.0)


def test_traces_to_array_empty():
    assert traces_to_array([]).shape == (0, 0)


def test_load_cheetah_bo_uhd_comparison_smoke():
    root = Path(__file__).resolve().parents[1]
    paths = CheetahExperimentPaths(yubo_root=root)
    bo, uhd, all_exp = load_cheetah_bo_uhd_comparison(paths)
    assert isinstance(bo, list)
    assert isinstance(uhd, list)
    assert len(all_exp) == len(bo) + len(uhd)


def test_bo_coverage_report_includes_eggroll_sigma():
    root = Path(__file__).resolve().parents[1]
    paths = CheetahExperimentPaths(yubo_root=root)
    bo, _, _ = load_cheetah_bo_uhd_comparison(paths)
    report = bo_coverage_report(bo)
    methods = set(report["method"])
    assert "eggroll (sigma=0.1)" in methods
    assert "eggroll (sigma=0.2)" in methods
    assert "eggroll (sigma=0.3)" in methods
    assert "eggroll (adapt sigma=0.1)" in methods


def test_plot_cheetah_bo_uhd_comparison_runs_with_tex_legend():
    root = Path(__file__).resolve().parents[1]
    paths = CheetahExperimentPaths(yubo_root=root)
    _, _, all_exp = load_cheetah_bo_uhd_comparison(paths)
    if not all_exp:
        pytest.skip("no cheetah traces on disk")
    import matplotlib.pyplot as plt

    plt.rcParams["text.usetex"] = True
    fig, summary = plot_cheetah_bo_uhd_comparison(all_exp)
    assert fig is not None
    assert not summary.empty
    plt.close(fig)


def test_final_best_summary_includes_policy():
    traces = [_trace([1.0, 2.0])]
    df = final_best_summary([{"label": "simple", "policy_tag": "linear", "traces": traces, "hash": "x"}])
    assert df.loc[0, "policy"] == "linear"
    assert df.loc[0, "best_mean"] == 2.0


def test_loaded_traces_summary():
    exps = [
        {
            "label": "simple",
            "policy_tag": "linear",
            "traces": [_trace([1.0])],
            "hash": "a",
        },
        {"label": "eggroll", "policy_tag": "linear", "traces": [], "hash": "b"},
    ]
    df = loaded_traces_summary(exps)
    assert len(df) == 2
    assert set(df["reps_done"]) == {0, 1}


def test_load_uhd_batch_and_simple_smoke(tmp_path: Path):
    root = Path(__file__).resolve().parents[1]
    paths = CheetahExperimentPaths(yubo_root=root)
    assert paths.bo_policy_optimizer.name == "exp_cheetah_policy_optimizer"
    assert paths.bo_eggroll_s01.name == "exp_cheetah_eggroll_s01"
    assert paths.bo_eggroll_s02.name == "exp_cheetah_eggroll_s02"
    assert paths.bo_eggroll_s03.name == "exp_cheetah_eggroll_s03"
    assert paths.bo_eggroll_adapt.name == "exp_cheetah_eggroll_adapt"
    assert len(paths.bo_extra_eggroll_dirs) == 4
    assert BO_UHD_POLICIES == ("linear", "mlp-32-16", "mlp-128-64")
    assert CHEETAH_POLICY_NUM_PARAMS["mlp-128-64"] == 10984
    _, batch_summary = load_uhd_batch_cheetah(paths.uhd)
    assert "label" in batch_summary.columns
    simple = load_uhd_cheetah_simple(paths.uhd)
    assert isinstance(simple, list)

    from analysis.data_io import TraceRecord, write_trace_jsonl
    from analysis.uhd_paper_cheetah import load_bo_cheetah_fn, load_traces_from_dir

    traces_dir = tmp_path / "traces"
    traces_dir.mkdir()
    write_trace_jsonl(
        str(traces_dir / "00000.jsonl"),
        [TraceRecord(i_iter=0, dt_prop=0.0, dt_eval=0.0, rreturn=1.0)],
    )
    loaded = load_traces_from_dir(traces_dir)
    assert len(loaded) == 1
    assert loaded[0][0].rreturn == 1.0

    exp_dir = tmp_path / "bo_exp"
    exp_dir.mkdir()
    (exp_dir / "config.json").write_text(
        json.dumps(
            {
                "env_tag": "cheetah:fn",
                "opt_name": "random",
                "policy_tag": "linear",
            }
        )
    )
    (exp_dir / "traces").mkdir()
    write_trace_jsonl(
        str(exp_dir / "traces" / "00000.jsonl"),
        [TraceRecord(i_iter=0, dt_prop=0.0, dt_eval=0.0, rreturn=2.0)],
    )
    bo = load_bo_cheetah_fn(tmp_path)
    assert len(bo) == 1
    assert bo[0]["label"] == "random"


def test_plot_uhd_batch_cheetah_runs():
    import matplotlib.pyplot as plt

    experiments = [
        {
            "label": "simple",
            "hash": "x",
            "config": {},
            "num_reps_expected": 1,
            "traces": [_trace([1.0, 2.0, 3.0])],
        }
    ]
    fig, summary = plot_uhd_batch_cheetah(experiments)
    assert fig is not None
    assert summary.loc[0, "best_mean"] == 3.0
    plt.close(fig)
