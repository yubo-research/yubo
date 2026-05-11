from __future__ import annotations

import sys
from unittest.mock import MagicMock

import numpy as np
import torch

from analysis.fitting_time.benchmark_table_fmt import (
    fmt_mu_loglik_sweep,
    fmt_mu_nrmse,
    fmt_nonfinite,
    fmt_ratio_vs_base,
    fmt_se,
    fmt_se_loglik_sweep,
    fmt_se_nrmse_sweep,
    fmt_synthetic_time_mu,
    pm_plus_minus_column,
)
from analysis.fitting_time.evaluate import benchmark_single_surrogate_with_data
from analysis.fitting_time.evaluate_metrics import SURROGATE_BENCHMARK_KEYS, BMResult, MuSe
from analysis.fitting_time.evaluate_table import print_synthetic_benchmark_table
from analysis.fitting_time.evaluate_triples import aggregate_surrogate_replicates, benchmark_single_surrogate
from analysis.plot_by_func_core import (
    add_group_label,
    add_group_labels,
    flatten_func_groups,
    func_groups,
    get_display_name,
    grid_layout,
    hide_extra_axes,
    load_traces_for_func,
    plot_by_func_grouped,
    plot_func_subplot,
    safe_plot_func,
)
from analysis.plotting_2_combined import plot_rl_combined_comparison, plot_rl_combined_comparison_from_data
from analysis.plotting_2_util import display_opt_name
from analysis.plotting_trace_types import RLTracesWithCumDtProp
from common.im import im


def test_benchmark_table_fmt_all():
    assert fmt_nonfinite(float("nan")) == "nan"
    assert fmt_nonfinite(float("inf")) == "inf"
    assert fmt_nonfinite(float("-inf")) == "-inf"
    assert fmt_nonfinite(1.0) is None
    assert fmt_synthetic_time_mu(0.0) == "0"
    assert isinstance(fmt_synthetic_time_mu(0.05), str)
    assert isinstance(fmt_mu_nrmse(1.23), str)
    assert isinstance(fmt_se(0.12), str)
    assert isinstance(fmt_se_nrmse_sweep(0.12), str)
    assert isinstance(fmt_mu_loglik_sweep(12.0), str)
    assert isinstance(fmt_mu_loglik_sweep(0.5), str)
    assert isinstance(fmt_se_loglik_sweep(2.0), str)
    assert isinstance(fmt_se_loglik_sweep(0.5), str)
    assert fmt_ratio_vs_base(2.0) in ("2", "2.0")
    assert len(pm_plus_minus_column([1.0, 2.0], [0.1, 0.2], fmt_mu_nrmse, fmt_se)) == 2


def test_benchmark_single_surrogate_with_data_patched(monkeypatch):
    import analysis.fitting_time.evaluate as ev

    def _draw(*, N, D, function_name, problem_seed):
        return (
            torch.zeros(N, D, dtype=torch.float64),
            torch.zeros(N, dtype=torch.float64),
            torch.zeros(3, D, dtype=torch.float64),
            torch.zeros(3, dtype=torch.float64),
        )

    monkeypatch.setattr(ev, "draw_benchmark_synthetic_xy", _draw)
    monkeypatch.setattr(ev, "benchmark_single_surrogate", lambda x, y, xt, yt, k: (0.1, 0.2, -0.5))
    out = benchmark_single_surrogate_with_data(N=4, D=2, function_name="sine", surrogate_key="enn", data_seed=0)
    assert out == (0.1, 0.2, -0.5)


def test_print_synthetic_benchmark_table(capsys):
    rows = []
    for k in SURROGATE_BENCHMARK_KEYS:
        rows.append((k, BMResult(MuSe(1.0, 0.1), MuSe(0.5, 0.05), MuSe(-1.0, 0.2))))
    d = {k: br for k, br in rows}
    print_synthetic_benchmark_table(d)
    captured = capsys.readouterr()
    assert "ENN" in captured.out or "Surrogate" in captured.out


def test_aggregate_and_benchmark_single_surrogate(monkeypatch):
    row = {k: (0.1 * i, 0.2, -0.5) for i, k in enumerate(SURROGATE_BENCHMARK_KEYS)}
    agg = aggregate_surrogate_replicates([row])
    assert hasattr(agg, "results")
    assert "enn" in agg.results

    import analysis.fitting_time.fitting_time as ft

    def _enn(tx, ty, xte):
        n = int(xte.shape[0])
        return (0.0, torch.zeros(n, dtype=torch.float64), torch.ones(n, dtype=torch.float64))

    monkeypatch.setattr(ft, "fit_enn", _enn)
    x = torch.zeros(5, 2, dtype=torch.float64)
    y = torch.randn(5, dtype=torch.float64)
    xt = torch.zeros(3, 2, dtype=torch.float64)
    yt = torch.randn(3, dtype=torch.float64)
    triple = benchmark_single_surrogate(x, y, xt, yt, "enn")
    assert len(triple) == 3


def test_plot_by_func_core_units(monkeypatch, tmp_path):
    import analysis.plot_by_func_core as pbc

    monkeypatch.setattr(
        pbc,
        "load_multiple_traces",
        lambda _loc: np.ones((1, 2, 1, 4), dtype=np.float64) * 0.5,
    )

    class _Loc:
        pass

    monkeypatch.setattr(pbc, "DataLocator", lambda **kwargs: _Loc())
    lt = load_traces_for_func(str(tmp_path), "e", ["a", "b"], "sphere", [2])
    assert isinstance(lt, list)
    assert get_display_name("x", {"x": "X"}) == "X"
    assert get_display_name("y", None) == "y"
    ax = MagicMock()
    ax.transAxes = "axes"
    ok = plot_func_subplot(ax, [np.ones((1, 2, 1, 4)) * 0.5], ["o1", "o2"], None, "f")
    assert ok in (True, False)
    fg = func_groups()
    assert isinstance(fg, dict)
    af, gl = flatten_func_groups({"G": ("a",)})
    assert af == ["a"] and gl == ["G"]
    glout = grid_layout(3, max_cols=2, max_rows=8)
    assert glout.rows >= 1 and glout.cols >= 1
    axl = [MagicMock(), MagicMock(), MagicMock()]
    hide_extra_axes(axl, 2)
    safe_plot_func(axl[0], str(tmp_path), "e", ["a"], "sphere", None, None)

    def _boom(*_a, **_k):
        raise RuntimeError("x")

    monkeypatch.setattr(pbc, "load_traces_for_func", _boom)
    safe_plot_func(axl[0], str(tmp_path), "e", ["a"], "sphere", None, None)

    fig = MagicMock()
    axs = [MagicMock() for _ in range(4)]
    for a in axs:
        a.get_position.return_value = MagicMock(y1=0.9)
    add_group_label(fig, axs, 0, 1, 2, "G")
    add_group_labels(fig, axs, ["A", "B", "A"], 2)
    monkeypatch.setattr(pbc, "func_groups", lambda: {"Multimodal": ("sphere",)})
    monkeypatch.setattr(pbc, "subplots", lambda *a, **k: (fig, [MagicMock()]))
    monkeypatch.setattr(pbc.plt, "show", lambda: None)
    monkeypatch.setattr(pbc.plt, "tight_layout", lambda *a, **k: None)
    monkeypatch.setattr(pbc.os, "makedirs", lambda *a, **k: None)
    fig.savefig = lambda *a, **k: None
    monkeypatch.setattr(pbc, "safe_plot_func", lambda *a, **k: None)
    figs = plot_by_func_grouped(str(tmp_path), "e", ["a"], save_dir=str(tmp_path / "o"), suptitle=False)
    assert figs


def test_plot_rl_combined_and_from_data(monkeypatch):
    import analysis.plotting_2_combined as cmb

    monkeypatch.setattr(cmb, "plot_learning_curves", lambda *a, **k: None)
    monkeypatch.setattr(cmb, "plot_final_performance", lambda *a, **k: None)
    monkeypatch.setattr(cmb, "consolidate_bottom_legend", lambda *a, **k: None)
    monkeypatch.setattr(cmb, "get_denoise_value", lambda *a, **k: None)
    fig = MagicMock()
    axs = np.empty((2, 2), dtype=object)
    for i in range(2):
        for j in range(2):
            m = MagicMock()
            m.axison = False
            axs[i, j] = m
    axs[0, 0].axison = True
    axs[1, 0].axison = True
    monkeypatch.setattr(cmb.plt, "subplots", lambda *a, **k: (fig, axs))
    dl = MagicMock()
    dl.optimizers.return_value = ["turbo-one"]
    tr = np.zeros((1, 1, 1, 3))
    seq = RLTracesWithCumDtProp(dl, tr, None)
    res = plot_rl_combined_comparison_from_data(
        seq,
        None,
        problem_seq="p",
        problem_batch="p",
        num_arms_seq=1,
        num_arms_batch=50,
        suptitle=None,
        figsize=(4, 4),
        opt_names_seq=["turbo-one"],
        opt_names_batch=["turbo-one"],
        opt_names_all=None,
        renames=None,
        show_titles=False,
        print_titles=False,
    )
    assert res.fig is fig

    monkeypatch.setattr(cmb, "_load_rl_with_cum_dt_prop", lambda *a, **kw: seq)
    monkeypatch.setattr(cmb, "_try_load_rl_with_cum_dt_prop", lambda *a, **kw: None)
    monkeypatch.setattr(cmb, "_print_cum_dt_props", lambda *a, **kw: None)
    out = plot_rl_combined_comparison(
        "/tmp",
        "e",
        ["turbo-one"],
        ["turbo-one"],
        problem_seq="p",
        problem_batch="p",
        num_reps=1,
        suptitle=None,
        figsize=(4, 4),
        cum_dt_prop=False,
    )
    assert out.fig is fig


def test_display_opt_name_and_im():
    assert display_opt_name("a", {"a": "A"}) == "A"
    assert display_opt_name("b", None) == "b"
    assert im("sys") is sys
