def test_acq_turbo_yubo_draw():
    import torch

    from acq.acq_turbo_yubo import AcqTurboYUBO
    from acq.turbo_yubo_config import TurboYUBOConfig
    from acq.turbo_yubo_state import TurboYUBOState

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

    state = TurboYUBOState(num_dim=2, batch_size=2)
    acq = AcqTurboYUBO(model=model, state=state, config=TurboYUBOConfig(raasp=False, tr=True))
    out = acq.draw(num_arms=2)

    assert out.shape == (2, 2)
    assert torch.all(out >= 0) and torch.all(out <= 1)
