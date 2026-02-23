from types import SimpleNamespace

import numpy as np

from rl.backends.torchrl.common.env_contract import (
    ActionContract,
    EnvIOContract,
    ObservationContract,
    resolve_action_contract,
    resolve_backbone_name,
    resolve_env_io_contract,
    resolve_observation_contract,
)


def test_resolve_env_io_contract_vector_continuous():
    env_conf = SimpleNamespace(
        from_pixels=False,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(17,))),
        action_space=SimpleNamespace(shape=(3,), low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0])),
    )
    contract = resolve_env_io_contract(env_conf)
    assert contract.observation.mode == "vector"
    assert contract.observation.vector_dim == 17
    assert contract.action.kind == "continuous"
    assert contract.action.dim == 3


def test_resolve_env_io_contract_pixels_discrete():
    env_conf = SimpleNamespace(
        from_pixels=True,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(4, 84, 84, 1))),
        action_space=SimpleNamespace(n=6, shape=()),
    )
    contract = resolve_env_io_contract(env_conf)
    assert contract.observation.mode == "pixels"
    assert contract.observation.model_channels == 4
    assert contract.observation.image_size == 84
    assert contract.action.kind == "discrete"
    assert contract.action.dim == 6


def test_resolve_backbone_name_from_observation_contract():
    pixel_contract = SimpleNamespace(mode="pixels", model_channels=3)
    atari_contract = SimpleNamespace(mode="pixels", model_channels=4)
    vector_contract = SimpleNamespace(mode="vector", model_channels=None)
    assert resolve_backbone_name("mlp", pixel_contract) == "nature_cnn"
    assert resolve_backbone_name("mlp", atari_contract) == "nature_cnn_atari"
    assert resolve_backbone_name("mlp", vector_contract) == "mlp"


def test_contract_dataclasses_construct():
    obs = ObservationContract(mode="vector", raw_shape=(3,), vector_dim=3)
    action = ActionContract(
        kind="continuous",
        dim=2,
        low=np.array([-1.0, -1.0], dtype=np.float32),
        high=np.array([1.0, 1.0], dtype=np.float32),
    )
    contract = EnvIOContract(observation=obs, action=action)
    assert contract.observation.vector_dim == 3
    assert contract.action.dim == 2


def test_resolve_observation_contract_direct_vector_and_pixels():
    vec_env = SimpleNamespace(
        from_pixels=False,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(2, 3))),
    )
    vec_obs = resolve_observation_contract(vec_env)
    assert vec_obs.mode == "vector"
    assert vec_obs.vector_dim == 6

    pix_env = SimpleNamespace(
        from_pixels=True,
        gym_conf=SimpleNamespace(state_space=SimpleNamespace(shape=(84, 84, 3))),
    )
    pix_obs = resolve_observation_contract(pix_env)
    assert pix_obs.mode == "pixels"
    assert pix_obs.model_channels == 3
    assert pix_obs.image_size == 84


def test_resolve_action_contract_direct_discrete_and_continuous():
    discrete_space = SimpleNamespace(n=4, shape=())
    discrete = resolve_action_contract(discrete_space)
    assert discrete.kind == "discrete"
    assert discrete.dim == 4
    assert np.allclose(discrete.low, np.array([0.0], dtype=np.float32))
    assert np.allclose(discrete.high, np.array([3.0], dtype=np.float32))

    continuous_space = SimpleNamespace(
        shape=(2,),
        low=np.array([-2.0, -1.0], dtype=np.float32),
        high=np.array([2.0, 1.0], dtype=np.float32),
    )
    continuous = resolve_action_contract(continuous_space)
    assert continuous.kind == "continuous"
    assert continuous.dim == 2
    assert np.allclose(continuous.low, continuous_space.low)
    assert np.allclose(continuous.high, continuous_space.high)
