import numpy as np
import torch


def test_dngo_designer_sobol_phase():
    from optimizer.dngo_designer import DNGODesigner
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    designer = DNGODesigner(policy, init_sobol=1)
    policies = designer([], num_arms=3)
    assert len(policies) == 3


def test_dngo_designer_with_data():
    from optimizer.datum import Datum
    from optimizer.dngo_designer import DNGODesigner
    from optimizer.trajectories import Trajectory
    from problems.env_conf import default_policy, get_env_conf

    env_conf = get_env_conf("f:sphere-2d", problem_seed=17, noise_seed_0=18)
    policy = default_policy(env_conf)

    data = []
    for i in range(5):
        p = policy.clone()
        params = np.random.default_rng(i).uniform(0, 1, p.num_params())
        p.set_params(params)
        trajectory = Trajectory(
            rreturn=float(i * 0.1),
            states=np.array([[]]),
            actions=np.array([[]]),
        )
        datum = Datum(designer=None, policy=p, expected_acqf=0.0, trajectory=trajectory)
        data.append(datum)

    designer = DNGODesigner(policy, init_sobol=1)
    policies = designer(data, num_arms=2)
    assert len(policies) == 2


def test_dngo_model_posterior():
    from analysis.fitting_time.dngo import DNGOConfig, DNGOSurrogate
    from optimizer.dngo_designer import DNGOModel

    rng = np.random.default_rng(42)
    X_train = rng.uniform(0, 1, (10, 2))
    Y_train = np.sum(X_train**2, axis=1)

    dngo = DNGOSurrogate(DNGOConfig(num_epochs=50))
    dngo.fit(X_train, Y_train)

    train_X = torch.tensor(X_train, dtype=torch.double)
    train_Y = torch.tensor(Y_train, dtype=torch.double).unsqueeze(-1)
    model = DNGOModel(dngo, train_X, train_Y)

    assert model.num_outputs == 1
    assert model.batch_shape == torch.Size([])
    assert model._input_batch_shape == torch.Size([])

    X_test = torch.rand(5, 2, dtype=torch.double)
    posterior = model.posterior(X_test)
    assert posterior is not None
    assert hasattr(posterior, "distribution")

    X_batch = torch.rand(3, 4, 2, dtype=torch.double)
    posterior_batch = model.posterior(X_batch)
    mean = posterior_batch.distribution.mean
    assert mean.shape == torch.Size([3, 4, 1])


def test_iid_normal_sampler():
    from botorch.posteriors.torch import TorchPosterior
    from torch.distributions import Normal

    from optimizer.dngo_designer import IIDNormalSampler

    mean = torch.zeros(5, 1)
    std = torch.ones(5, 1)
    dist = Normal(mean, std)
    posterior = TorchPosterior(distribution=dist)

    sampler = IIDNormalSampler(sample_shape=torch.Size([10]))
    samples = sampler.forward(posterior)
    assert samples.shape == torch.Size([10, 5, 1])
