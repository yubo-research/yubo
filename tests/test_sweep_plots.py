import json
from pathlib import Path

import matplotlib
import matplotlib.colors as mcolors


matplotlib.use("Agg")

import numpy as np
import pytest


class _FakeAx:
    def errorbar(self, *args, **kwargs):
        return None

    def set_xscale(self, *args, **kwargs):
        return None

    def set_xlabel(self, *args, **kwargs):
        return None

    def set_ylabel(self, *args, **kwargs):
        return None

    def set_title(self, *args, **kwargs):
        return None

    def set_xticks(self, *args, **kwargs):
        return None

    def set_xticklabels(self, *args, **kwargs):
        return None

    def set_ylim(self, *args, **kwargs):
        return None

    def grid(self, *args, **kwargs):
        return None

    def plot(self, *args, **kwargs):
        return None

    def fill_between(self, *args, **kwargs):
        return None

    def legend(self, *args, **kwargs):
        return None

    def bar(self, *args, **kwargs):
        return None

    def axhline(self, *args, **kwargs):
        return None

    def set_axis_off(self, *args, **kwargs):
        return None

    def text(self, *args, **kwargs):
        return None


def _patch_plt(monkeypatch, sweep_plots_mod, *, two_rows: bool = False):
    def fake_subplots(*args, **kwargs):
        if two_rows or (len(args) >= 2 and args[0] == 2 and args[1] == 1):
            return None, (_FakeAx(), _FakeAx())
        return None, _FakeAx()

    monkeypatch.setattr(sweep_plots_mod.plt, "subplots", fake_subplots)
    monkeypatch.setattr(sweep_plots_mod.plt, "tight_layout", lambda *a, **k: None)
    monkeypatch.setattr(sweep_plots_mod.plt, "show", lambda *a, **k: None)


def _write_config_run(root: Path, sub: str, opt_name: str, env_tag: str) -> None:
    d = root / sub
    d.mkdir(parents=True)
    (d / "config.json").write_text(
        json.dumps({"opt_name": opt_name, "env_tag": env_tag}),
    )


def test_plot_param_sweep_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis import sweep_plots

    exp = "exp_test_sweep"
    base = tmp_path / exp
    base.mkdir()
    _write_config_run(base, "run_a", "turbo-enn-fit-ucb/k=3", "f:ackley-10d")
    _write_config_run(base, "run_b", "turbo-enn-fit-ucb/k=30", "f:ackley-10d")

    monkeypatch.setattr(
        sweep_plots,
        "load_traces",
        lambda path, key="rreturn": np.ones((2, 5)),
    )
    _patch_plt(monkeypatch, sweep_plots)

    sweep_plots.plot_param_sweep(exp_dir=exp, results_dir=tmp_path)


def test_plot_curves_respects_env_tag(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis import sweep_plots

    exp = "exp_test_curves"
    base = tmp_path / exp
    base.mkdir()
    _write_config_run(base, "r1", "turbo-enn-fit-ucb/k=3", "f:sphere-10d")
    _write_config_run(base, "r2", "turbo-enn-fit-ucb/k=10", "f:ackley-10d")

    seen: list[str] = []

    def capture_load(path, key="rreturn"):
        seen.append(path)
        return np.ones((2, 4))

    monkeypatch.setattr(sweep_plots, "load_traces", capture_load)
    _patch_plt(monkeypatch, sweep_plots, two_rows=True)

    sweep_plots.plot_curves(
        exp_dir=exp,
        results_dir=tmp_path,
        env_tag="f:ackley-10d",
    )

    assert len(seen) == 1
    assert "r2" in seen[0].replace("\\", "/")


def test_draw_plot_curves_panels_matches_bar_colors_and_adds_markers() -> None:
    import analysis.plotting as ap
    from analysis.sweep_plots_panels import _draw_plot_curves_panels

    _, (ax_curve, ax_bar) = matplotlib.pyplot.subplots(
        2,
        1,
        figsize=(8, 6),
        gridspec_kw={"height_ratios": [2.0, 1.0]},
    )
    param_values = [3, 10]
    all_curves = [
        np.array([[1.0, 2.0, 3.0], [1.5, 2.5, 3.5]]),
        np.array([[0.5, 1.0, 1.5], [0.75, 1.25, 1.75]]),
    ]

    _draw_plot_curves_panels(
        ax_curve,
        ax_bar,
        param_values,
        all_curves,
        param_name_for_print="K",
        xlabel="Round",
        ylabel=r"$y_{\mathrm{best}}$",
        title="",
        show_legend=True,
        show_curve_ylabel=True,
        show_bar_ylabel=True,
        show_title=False,
        panel_label=None,
        curve_ylabel_fontsize=None,
    )

    assert [line.get_marker() for line in ax_curve.lines] == ap.markers[:2]
    expected_colors = [mcolors.to_rgba(c, alpha=0.85) for c in ap.colors[:2]]
    actual_colors = [patch.get_facecolor() for patch in ax_bar.patches]
    assert actual_colors == expected_colors


def test_plot_curves_four_envs_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis import sweep_plots

    exp = "exp_test_four"
    base = tmp_path / exp
    base.mkdir()
    tags = list(sweep_plots.DEFAULT_SYNTH_10D_ENV_TAGS)
    for i, tag in enumerate(tags):
        _write_config_run(
            base,
            f"run_{i}",
            "turbo-enn-fit-ucb/k=10",
            tag,
        )

    monkeypatch.setattr(
        sweep_plots,
        "load_traces",
        lambda path, key="rreturn": np.ones((2, 4)),
    )
    monkeypatch.setattr(sweep_plots.plt, "show", lambda *a, **k: None)

    sweep_plots.plot_curves_four_envs(exp_dir=exp, results_dir=tmp_path)


def test_plot_curves_four_sources_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis import sweep_plots

    exp_a = "exp_test_four_a"
    exp_b = "exp_test_four_b"
    base_a = tmp_path / exp_a
    base_b = tmp_path / exp_b
    base_a.mkdir()
    base_b.mkdir()
    _write_config_run(base_a, "run_a0", "turbo-enn-fit-ucb/k=3", "env:a")
    _write_config_run(base_a, "run_a1", "turbo-enn-fit-ucb/k=10", "env:b")
    _write_config_run(base_b, "run_b0", "turbo-enn-fit-ucb/k=3", "env:c")
    _write_config_run(base_b, "run_b1", "turbo-enn-fit-ucb/k=10", "env:d")

    monkeypatch.setattr(
        sweep_plots,
        "load_traces",
        lambda path, key="rreturn": np.ones((2, 4)),
    )
    monkeypatch.setattr(sweep_plots.plt, "show", lambda *a, **k: None)

    sweep_plots.plot_curves_four_sources(
        panel_sources=(
            (exp_a, "env:a"),
            (exp_a, "env:b"),
            (exp_b, "env:c"),
            (exp_b, "env:d"),
        ),
        results_dir=tmp_path,
    )


def test_plot_curves_four_sources_saves_when_requested(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis import sweep_plots

    exp = "exp_test_four_save"
    base = tmp_path / exp
    base.mkdir()
    for i, tag in enumerate(("env:a", "env:b", "env:c", "env:d")):
        _write_config_run(
            base,
            f"run_{i}",
            "turbo-enn-fit-ucb/k=10",
            tag,
        )

    saved = []
    monkeypatch.setattr(
        sweep_plots,
        "load_traces",
        lambda path, key="rreturn": np.ones((2, 4)),
    )
    monkeypatch.setattr(sweep_plots.plt, "show", lambda *a, **k: None)
    monkeypatch.setattr(
        sweep_plots.plt.Figure,
        "savefig",
        lambda self, path, **kwargs: saved.append((path, kwargs)),
    )

    out_path = tmp_path / "four_panel.pdf"
    sweep_plots.plot_curves_four_sources(
        panel_sources=(
            (exp, "env:a"),
            (exp, "env:b"),
            (exp, "env:c"),
            (exp, "env:d"),
        ),
        results_dir=tmp_path,
        save_path=out_path,
    )

    assert saved == [(out_path, {"bbox_inches": "tight"})]


def test_plot_curves_four_sources_mixed_param_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from analysis import sweep_plots

    exp_p = "exp_test_four_p"
    exp_k = "exp_test_four_k"
    base_p = tmp_path / exp_p
    base_k = tmp_path / exp_k
    base_p.mkdir()
    base_k.mkdir()
    _write_config_run(base_p, "run_p0", "turbo-enn-fit-ucb/nfs=10", "env:a")
    _write_config_run(base_p, "run_p1", "turbo-enn-fit-ucb/nfs=30", "env:c")
    _write_config_run(base_k, "run_k0", "turbo-enn-fit-ucb/k=3", "env:b")
    _write_config_run(base_k, "run_k1", "turbo-enn-fit-ucb/k=10", "env:d")

    monkeypatch.setattr(
        sweep_plots,
        "load_traces",
        lambda path, key="rreturn": np.ones((2, 4)),
    )
    monkeypatch.setattr(sweep_plots.plt, "show", lambda *a, **k: None)

    sweep_plots.plot_curves_four_sources(
        panel_sources=(
            (exp_p, "env:a", r"nfs=(\d+)", "P"),
            (exp_k, "env:b", r"k=(\d+)", "K"),
            (exp_p, "env:c", r"nfs=(\d+)", "P"),
            (exp_k, "env:d", r"k=(\d+)", "K"),
        ),
        results_dir=tmp_path,
    )


def test_plot_curves_four_envs_panel_labels_length_mismatch(
    tmp_path: Path,
) -> None:
    from analysis import sweep_plots

    with pytest.raises(ValueError, match="same length"):
        sweep_plots.plot_curves_four_envs(
            exp_dir="missing",
            results_dir=tmp_path,
            env_tags=("a", "b"),
            panel_labels=("x",),
        )


def test_plot_curves_four_sources_panel_labels_length_mismatch(
    tmp_path: Path,
) -> None:
    from analysis import sweep_plots

    with pytest.raises(ValueError, match="same length"):
        sweep_plots.plot_curves_four_sources(
            panel_sources=(("exp_a", "env:a"), ("exp_b", "env:b")),
            results_dir=tmp_path,
            panel_labels=("x",),
        )


def test_mean_ybest_mean_sem_per_curve() -> None:
    from analysis.sweep_plots import _mean_ybest_mean_sem_per_curve

    c1 = np.array([[1.0, 2.0, 3.0], [1.0, 3.0, 5.0]])
    means, sems = _mean_ybest_mean_sem_per_curve([c1])
    assert means[0] == 2.5
    assert len(sems) == 1
    assert sems[0] > 0


def test_bar_heights_shift_if_negative() -> None:
    from analysis.sweep_plots import _bar_heights_shift_if_negative

    h, s = _bar_heights_shift_if_negative([1.0, 2.0])
    assert s == 0.0
    assert h == [1.0, 2.0]

    h, s = _bar_heights_shift_if_negative([-2.0, -1.0])
    assert s == -2.0
    assert h == [0.0, 1.0]


def test_plot_curves_missing_dir_prints(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    from analysis import sweep_plots

    sweep_plots.plot_curves(
        exp_dir="nonexistent_exp",
        results_dir=tmp_path,
    )
    err = capsys.readouterr().out
    assert "Directory not found" in err


def test_load_traces_or_skip_skips_incomplete_1d_trace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from analysis import sweep_plots

    monkeypatch.setattr(
        sweep_plots,
        "load_traces",
        lambda path, key="rreturn": np.array([np.nan]),
    )

    result = sweep_plots._load_traces_or_skip(
        Path("/tmp/fake"),
        val=10,
        trace_key="rreturn",
        param_name_for_print="P",
    )

    assert result is None
