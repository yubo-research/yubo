def test_acq_turbo_yubo_draw():
    import torch

    from acq.turbo_yubo.acq_turbo_yubo import AcqTurboYUBO
    from acq.turbo_yubo.turbo_yubo_config import TurboYUBOConfig
    from acq.turbo_yubo.turbo_yubo_state import TYDefaultTR

    class _DummyModel:
        def __init__(self, X, Y):
            self.train_inputs = (X,)
            self.train_targets = Y
            self.covar_module = None

        def posterior(self, X):
            class _P:
                def __init__(self, X):
                    self._X = X

                def sample(self, sample_shape):
                    return torch.rand(sample_shape + (self._X.shape[0], 1), dtype=self._X.dtype, device=self._X.device)

            return _P(X)

    X = torch.rand(size=(4, 2), dtype=torch.double)
    Y = torch.rand(size=(4,), dtype=torch.double)
    model = _DummyModel(X, Y)

    state = TYDefaultTR(num_dim=2, _num_arms=2)

    def _fake_raasp(x_center, lb, ub, num_candidates, device, dtype):
        lb_t = torch.as_tensor(lb, dtype=dtype, device=device)
        ub_t = torch.as_tensor(ub, dtype=dtype, device=device)
        u = torch.rand((num_candidates, x_center.shape[-1]), dtype=dtype, device=device)
        return lb_t + (ub_t - lb_t) * u

    acq = AcqTurboYUBO(model=model, state=state, config=TurboYUBOConfig(candidate_sampler=_fake_raasp))
    out = acq.draw(num_arms=2)

    assert out.shape == (2, 2)
    assert torch.all(out >= 0) and torch.all(out <= 1)
