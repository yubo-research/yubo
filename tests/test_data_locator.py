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


def test_organize_data():
    import numpy as np

    from analysis.data_locator import DataLocator

    with tempfile.TemporaryDirectory() as tmpdir:
        dl = DataLocator(
            results_path=tmpdir,
            exp_dir="test_exp",
            num_arms=2,
            num_rounds=10,
            num_reps=3,
            opt_names=["opt1", "opt2"],
        )

        # Mock the optimizers method for testing
        original_optimizers = dl.optimizers
        dl.optimizers = lambda: ["opt1", "opt2"]

        mu = np.array([1.0, 2.0])
        se = np.array([0.1, 0.2])
        mu_out, se_out = dl.organize_data(["opt1", "opt2"], mu, se)
        assert mu_out.shape == (2,)
        assert se_out.shape == (2,)

        # Restore
        dl.optimizers = original_optimizers
