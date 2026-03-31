import pytest

from analysis.fitting_time.sweeps import sweep


def test_sweep_smoke_sine(capsys):
    sweep(
        N=14,
        D=2,
        function_name="sine",
        kp_pairs=[(3, 8), (5, 10)],
        problem_seed=0,
        num_reps=2,
        num_fit_candidates=20,
    )
    out = capsys.readouterr().out
    assert "±" in out
    assert "NRMSE" in out and "LogLik" in out


def test_sweep_rejects_empty_pairs():
    with pytest.raises(ValueError, match="kp_pairs"):
        sweep(N=5, D=2, function_name="sine", kp_pairs=[], num_reps=1)
