import tempfile


def test_data_locator_init():
    from analysis.data_locator import DataLocator

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DataLocator(
            results_path=tmpdir,
            exp_dir="test_exp",
            num_arms=2,
            num_rounds=10,
            num_reps=3,
            opt_names=["random", "sobol"],
        )
        assert dl.num_arms == 2
        assert dl.num_rounds == 10
        assert dl.num_reps == 3


def test_data_locator_str():
    from analysis.data_locator import DataLocator

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DataLocator(
            results_path=tmpdir,
            exp_dir="test_exp",
            num_arms=2,
            num_rounds=10,
            num_reps=3,
            opt_names=["random"],
        )
        s = str(dl)
        assert "test_exp" in s
