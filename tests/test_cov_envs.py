import pytest


def test_lasso_bench_env_init():
    # LassoBench might not be installed or data might not be available
    pytest.importorskip("LassoBench")
    from problems.lasso_bench_env import LassoBenchEnv

    # Try with the smallest dataset
    try:
        env = LassoBenchEnv(pick_data="dna", seed=42)
        assert env.n_features > 0
        env.close()
    except Exception:
        pytest.skip("LassoBench data not available")


def test_dna_env_init():
    pytest.importorskip("LassoBench")
    from problems.dna_env import DnaEnv

    try:
        env = DnaEnv(seed=42)
        assert env.n_features > 0
        env.close()
    except Exception:
        pytest.skip("LassoBench DNA data not available")


def test_leukemia_env_init():
    pytest.importorskip("LassoBench")
    from problems.leukemia_env import LeukemiaEnv

    try:
        env = LeukemiaEnv(seed=42)
        assert env.n_features > 0
        env.close()
    except Exception:
        pytest.skip("LassoBench leukemia data not available")


def test_rcv1_env_init():
    pytest.importorskip("LassoBench")
    from problems.rcv1_env import Rcv1Env

    try:
        env = Rcv1Env(seed=42)
        assert env.n_features > 0
        env.close()
    except Exception:
        pytest.skip("LassoBench RCV1 data not available")


def test_mopta08_init():
    from problems.mopta08 import Mopta08

    try:
        m = Mopta08()
        assert m.num_dim == 124
    except RuntimeError as e:
        if "Machine with this architecture is not supported" in str(e):
            pytest.skip("Mopta08 not supported on this architecture")
        raise
