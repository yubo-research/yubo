def test_smac_designer_basic():
    """Test SMACDesigner can be created and returns policies."""
    from optimizer.smac_designer import SMACDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    designer = SMACDesigner(policy)
    # First call with no data
    policies = designer(None, num_arms=1)
    assert len(policies) == 1

    # Call with num_arms > 1
    policies = designer(None, num_arms=3)
    assert len(policies) == 3


def test_smac_designer_is_valid():
    """Test SMACDesigner produces valid policies."""
    import numpy as np

    import common.all_bounds as all_bounds
    from optimizer.smac_designer import SMACDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    designer = SMACDesigner(policy)
    policies = designer(None, num_arms=5)

    for p in policies:
        params = p.get_params()
        # Check params are within bounds
        assert np.all(params >= all_bounds.p_low)
        assert np.all(params <= all_bounds.p_high)
        # Check correct dimensionality
        assert len(params) == policy.num_params()


def test_smac_designer_with_tell():
    """Test SMACDesigner can receive feedback via data."""
    from dataclasses import dataclass

    from optimizer.smac_designer import SMACDesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    designer = SMACDesigner(policy)

    # First round - ask for 2 policies
    policies = designer(None, num_arms=2)
    assert len(policies) == 2

    # Create mock data for tell
    @dataclass
    class MockTrajectory:
        rreturn: float

    @dataclass
    class MockDatum:
        trajectory: MockTrajectory

    # Simulate feedback
    data = [
        MockDatum(trajectory=MockTrajectory(rreturn=0.5)),
        MockDatum(trajectory=MockTrajectory(rreturn=0.7)),
    ]

    # Second round - tell results and ask for more
    policies = designer(data, num_arms=2)
    assert len(policies) == 2
