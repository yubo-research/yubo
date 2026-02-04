from unittest.mock import MagicMock

import pytest


class TestBTDesigner:
    def test_init(self):
        from optimizer.bt_designer import BTDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        bd = BTDesigner(
            policy,
            lambda m: MagicMock(),
            num_restarts=10,
            raw_samples=10,
            start_at_max=False,
        )
        assert bd._policy == policy
        assert bd._num_restarts == 10

    def test_call_sobol_init(self):
        from optimizer.bt_designer import BTDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        bd = BTDesigner(
            policy,
            lambda m: MagicMock(),
            num_restarts=10,
            raw_samples=10,
            start_at_max=False,
            init_sobol=5,
        )

        policies = bd([], num_arms=1)
        assert len(policies) == 1


class TestTurboRefDesigner:
    def test_init(self):
        from optimizer.turbo_ref_designer import TuRBORefDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        trd = TuRBORefDesigner(policy, num_trust_regions=1, num_init=10)
        assert trd._policy == policy
        assert trd._num_init == 10


class TestTurboENNDesigner:
    def test_init(self):
        from optimizer.turbo_enn_designer import TurboENNDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        ted = TurboENNDesigner(policy, turbo_mode="turbo-enn", k=10)
        assert ted._policy == policy
        assert ted._k == 10


class TestMultiTurboENNDesigner:
    def test_init(self):
        from optimizer.multi_turbo_enn_designer import MultiTurboENNConfig, MultiTurboENNDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-5d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        config = MultiTurboENNConfig(turbo_mode="turbo-enn", num_regions=2, acq_type="thompson")
        mtd = MultiTurboENNDesigner(policy, config=config)
        assert mtd._policy == policy
        assert mtd._num_regions == 2


class TestMTSDesigner:
    def test_init(self):
        from optimizer.mts_designer import MTSDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        md = MTSDesigner(policy, init_style="find")
        assert md._policy == policy
        assert md._init_style == "find"


class TestCMADesigner:
    def test_init(self):
        from optimizer.cma_designer import CMAESDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        cd = CMAESDesigner(policy)
        assert cd._policy == policy

    def test_call_asserts_num_arms(self):
        from optimizer.cma_designer import CMAESDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        cd = CMAESDesigner(policy)
        with pytest.raises(AssertionError):
            cd([], num_arms=1)


class TestMCMCBODesigner:
    def test_init(self):
        from optimizer.mcmc_bo_designer import MCMCBODesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        md = MCMCBODesigner(policy, num_init=10)
        assert md._policy == policy
        assert md._num_init == 10


class TestVecchiaDesigner:
    def test_init(self):
        from optimizer.vecchia_designer import VecchiaDesigner
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        vd = VecchiaDesigner(policy, num_candidates_per_arm=100)
        assert vd._policy == policy
        assert vd._num_candidates_per_arm == 100


class TestDesigners:
    def test_init(self):
        from optimizer.designers import Designers
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        assert d._policy == policy
        assert d._num_arms == 5

    def test_create_random(self):
        from optimizer.designers import Designers
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        designer = d.create("random")
        assert designer is not None

    def test_create_sobol(self):
        from optimizer.designers import Designers
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        designer = d.create("sobol")
        assert designer is not None

    def test_create_invalid(self):
        from optimizer.designers import Designers, NoSuchDesignerError
        from problems.env_conf import default_policy, get_env_conf

        env_conf = get_env_conf("f:sphere-2d", problem_seed=42, noise_seed_0=18)
        policy = default_policy(env_conf)

        d = Designers(policy, num_arms=5)
        with pytest.raises(NoSuchDesignerError):
            d.create("invalid_designer_name")
