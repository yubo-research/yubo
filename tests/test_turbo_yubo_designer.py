def test_turbo_yubo_designer():
    from unittest.mock import Mock

    import torch

    from optimizer.turbo_yubo_designer import TurboYUBODesigner

    policy = Mock()
    policy.num_params.return_value = 2
    policy.clone.return_value = policy

    designer = TurboYUBODesigner(policy)

    assert designer._policy == policy
    assert designer._num_keep is None
    assert designer._keep_style is None
    assert designer._turbo_yubo_state is None
    assert designer._X_train.shape == (0, 2)
    assert designer._Y_train.shape == (0, 1)

    num_arms = 3
    data = []

    policies = designer(data, num_arms)
    assert len(policies) == num_arms
    assert all(p == policy for p in policies)

    X = torch.rand(5, 2, dtype=torch.float64)
    Y = torch.rand(5, 1, dtype=torch.float64)

    mock_data = []
    for i in range(5):
        mock_datum = Mock()
        mock_datum.trajectory.rreturn = Y[i].item()
        mock_datum.policy.get_params.return_value = X[i].numpy()
        mock_data.append(mock_datum)

    policies = designer(mock_data, num_arms)
    assert len(policies) == num_arms
    assert designer._turbo_yubo_state is not None
    assert designer._turbo_yubo_state.num_dim == 2
    assert designer._turbo_yubo_state.batch_size == num_arms

    print("All TurboYUBODesigner tests passed! ✓")
