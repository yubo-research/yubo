from unittest.mock import MagicMock

import numpy as np
import pytest


class TestCenterDesigner:
    def test_init(self):
        from optimizer.center_designer import CenterDesigner

        mock_policy = MagicMock()
        cd = CenterDesigner(mock_policy)
        assert cd._policy == mock_policy

    def test_call(self):
        from optimizer.center_designer import CenterDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
        policy = default_policy(env_conf)

        cd = CenterDesigner(policy)
        policies = cd(None, num_arms=1)

        assert len(policies) == 1
        params = policies[0].get_params()
        assert len(params) == 2
        np.testing.assert_array_almost_equal(params, [0.0, 0.0])

    def test_call_asserts_num_arms(self):
        from optimizer.center_designer import CenterDesigner

        cd = CenterDesigner(MagicMock())
        with pytest.raises(AssertionError):
            cd(None, num_arms=2)


class TestRandomDesigner:
    def test_init(self):
        from optimizer.random_designer import RandomDesigner

        mock_policy = MagicMock()
        rd = RandomDesigner(mock_policy)
        assert rd._policy == mock_policy

    def test_call(self):
        import common.all_bounds as all_bounds
        from optimizer.random_designer import RandomDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
        policy = default_policy(env_conf)

        rd = RandomDesigner(policy)
        policies = rd(None, num_arms=3)

        assert len(policies) == 3
        for p in policies:
            params = p.get_params()
            assert len(params) == 2
            assert all(all_bounds.p_low <= x <= all_bounds.p_high for x in params)

    def test_call_with_telemetry(self):
        from optimizer.random_designer import RandomDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
        policy = default_policy(env_conf)

        mock_telemetry = MagicMock()

        rd = RandomDesigner(policy)
        rd(None, num_arms=1, telemetry=mock_telemetry)

        mock_telemetry.set_dt_fit.assert_called_once_with(0)
        mock_telemetry.set_dt_select.assert_called_once()


class TestSobolDesigner:
    def test_init(self):
        from optimizer.sobol_designer import SobolDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-3d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        sd = SobolDesigner(policy)
        assert sd._policy == policy
        assert sd.seed == 42 + 12345

    def test_call(self):
        from optimizer.sobol_designer import SobolDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        sd = SobolDesigner(policy)
        policies = sd(None, num_arms=5)

        assert len(policies) == 5
        assert len(sd.fig_last_arms) == 5

    def test_estimate(self):
        from optimizer.sobol_designer import SobolDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        sd = SobolDesigner(policy)
        result = sd.estimate([], [[1, 2], [3, 4]])
        assert result == [None, None]


class TestLHDDesigner:
    def test_call(self):
        from optimizer.lhd_designer import LHDDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        lhd = LHDDesigner(policy)
        policies = lhd(None, num_arms=2)

        assert len(policies) == 2


class TestOptunaDesigner:
    def test_init(self):
        from optimizer.optuna_designer import OptunaDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        od = OptunaDesigner(policy)
        assert od._policy == policy

    def test_call(self):
        from optimizer.optuna_designer import OptunaDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        od = OptunaDesigner(policy)
        policies = od([], num_arms=2)

        assert len(policies) == 2


class TestAxDesigner:
    def test_init(self):
        from optimizer.ax_designer import AxDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        ad = AxDesigner(policy)
        assert ad._policy == policy
        assert ad._ax_client is None

    def test_call_asserts_num_arms(self):
        from optimizer.ax_designer import AxDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        ad = AxDesigner(policy)
        with pytest.raises(AssertionError):
            ad([], num_arms=2)
