def test_acq_turbo():
    import numpy as np
    import torch

    from acq.acq_turbo_yubo import AcqTurboYUBO, TurboYUBOState
    from acq.fit_gp import fit_gp_XY

    dim = 2
    num_arms = 3
    num_rounds = 3

    def objective_function(x):
        x = np.array(x)
        a = 20
        b = 0.2
        c = 2 * np.pi
        n = len(x)
        sum1 = np.sum(x**2)
        sum2 = np.sum(np.cos(c * x))
        term1 = -a * np.exp(-b * np.sqrt(sum1 / n))
        term2 = -np.exp(sum2 / n)
        return term1 + term2 + a + np.exp(1)

    X_history = []
    Y_history = []

    print("Starting TuRBO-1 optimization...")
    print(f"Problem dimension: {dim}")
    print(f"Batch size: {num_arms}")
    print(f"Number of rounds: {num_rounds}")
    print("-" * 50)

    for round_idx in range(num_rounds):
        print(f"\nRound {round_idx + 1}:")
        if round_idx == 0:
            X_next = torch.rand(num_arms, dim, dtype=torch.float64)
        else:
            X_all = torch.cat(X_history, dim=0)
            Y_all = torch.cat(Y_history, dim=0)
            assert X_all.shape[1] == dim, f"X_all shape: {X_all.shape}"
            assert Y_all.shape[1] == 1, f"Y_all shape: {Y_all.shape}"
            model = fit_gp_XY(X_all, Y_all)
            state = TurboYUBOState(num_dim=dim, batch_size=num_arms)
            acq_turbo = AcqTurboYUBO(model=model, state=state)
            X_next = acq_turbo.draw(num_arms)
        if X_next.dim() == 3:
            X_next = X_next.squeeze(1)
        X_next = X_next.reshape(num_arms, dim)
        assert X_next.shape == (num_arms, dim), f"X_next shape: {X_next.shape}"
        Y_next = []
        for x in X_next:
            x_scaled = -5 + 10 * x
            y = objective_function(x_scaled.numpy())
            Y_next.append(y)
        Y_next = torch.tensor(Y_next, dtype=torch.float64).reshape(num_arms, 1)
        assert Y_next.shape == (num_arms, 1), f"Y_next shape: {Y_next.shape}"
        X_history.append(X_next)
        Y_history.append(Y_next)
        best_idx = torch.argmax(Y_next).item()
        best_x = X_next[best_idx]
        best_y = Y_next[best_idx].item()
        best_x_scaled = -5 + 10 * best_x
        print(f"  Best point: {best_x_scaled}")
        print(f"  Best value: {best_y:.6f}")
        if round_idx > 0:
            print(f"  Trust region length: {acq_turbo.state.length:.4f}")
            print(f"  Success counter: {acq_turbo.state.success_counter}")
            print(f"  Failure counter: {acq_turbo.state.failure_counter}")
            if acq_turbo.state.restart_triggered:
                print("  *** Trust region restart triggered ***")
    print("\n" + "=" * 50)
    print("OPTIMIZATION COMPLETE")
    print("=" * 50)
    all_X = torch.cat(X_history, dim=0)
    all_Y = torch.cat(Y_history, dim=0)
    assert all_X.shape == (num_rounds * num_arms, dim), f"all_X shape: {all_X.shape}"
    assert all_Y.shape == (num_rounds * num_arms, 1), f"all_Y shape: {all_Y.shape}"
    best_overall_idx = torch.argmax(all_Y).item()
    best_overall_x = all_X[best_overall_idx]
    best_overall_y = all_Y[best_overall_idx].item()
    best_overall_x_scaled = -5 + 10 * best_overall_x
    print(f"Best overall point: {best_overall_x_scaled}")
    print(f"Best overall value: {best_overall_y:.6f}")
    print(f"Total evaluations: {len(all_Y)}")
    assert all_X.shape[0] == num_rounds * num_arms, f"Expected {num_rounds * num_arms} evaluations, got {all_X.shape[0]}"
    assert all_X.shape[1] == dim, f"Expected dimension {dim}, got {all_X.shape[1]}"
    assert all_Y.shape[1] == 1, f"Expected single output, got {all_Y.shape[1]}"
    try:
        acq_turbo = AcqTurboYUBO(model=None)
        assert False, "AcqTurbo should require a model"
    except AssertionError:
        pass
    print("\nAll tests passed! ✓")


if __name__ == "__main__":
    test_acq_turbo()
