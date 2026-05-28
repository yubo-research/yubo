import click
import pytest
from click.testing import CliRunner

from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
from experiments.modal_synthetic_sine_benchmark_batches_reps import (
    surrogate_rep_json_dest,
)
from ops.synthetic_sine_benchmark_batches import _resolve_surrogate_key, cli


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("enn", "enn"),
        ("ENN+HNSW", "enn_hnsw"),
        ("enn-hnsw", "enn_hnsw"),
        ("SMAC RF", "smac_rf"),
        ("smac", "smac_rf"),
        ("svgp_linear", "svgp_linear"),
        ("Vecchia", "vecchia"),
    ],
)
def test_resolve_surrogate_key_ok(raw: str, expected: str) -> None:
    assert _resolve_surrogate_key(raw) == expected


def test_resolve_surrogate_key_bad() -> None:
    with pytest.raises(click.BadParameter):
        _resolve_surrogate_key("not_a_surrogate")


def _assert_local_single_skips_existing(tmp_path, *, dest_kwargs, cli_args) -> None:
    fn = normalize_benchmark_function_name("sphere")
    dest = surrogate_rep_json_dest(tmp_path, function_name=fn, **dest_kwargs)
    dest.parent.mkdir(parents=True)
    dest.write_text("{}")

    runner = CliRunner()
    result = runner.invoke(cli, cli_args + ["--output-dir", str(tmp_path), "--no-aggregate"])
    assert result.exit_code == 0
    assert "skip existing" in result.output


def test_local_single_skips_existing(tmp_path) -> None:
    _assert_local_single_skips_existing(
        tmp_path,
        dest_kwargs=dict(n=1, d=2, problem_seed=3, rep_index=0, surrogate_key="enn"),
        cli_args=[
            "local-single",
            "1",
            "sphere",
            "0",
            "enn",
            "-d",
            "2",
            "--problem-seed",
            "3",
        ],
    )


def test_local_single_skips_existing_default_d10(tmp_path) -> None:
    _assert_local_single_skips_existing(
        tmp_path,
        dest_kwargs=dict(n=5, d=10, problem_seed=17, rep_index=2, surrogate_key="vecchia"),
        cli_args=[
            "local-single",
            "5",
            "sphere",
            "2",
            "vecchia",
            "--problem-seed",
            "17",
        ],
    )
