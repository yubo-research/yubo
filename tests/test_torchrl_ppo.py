from types import SimpleNamespace

import numpy as np
import pytest

from rl.core.env_contract import ActionContract, EnvIOContract, ObservationContract
from rl.torchrl.ppo import core as torchrl_on_policy_core
from rl.torchrl.ppo import deps as op_deps
from rl.torchrl.ppo.core import _TanhNormal as _CoreTanhNormal


@pytest.mark.parametrize("dist_cls", [torchrl_on_policy_core._TanhNormal, _CoreTanhNormal])
def test_tanhnormal_support_property(dist_cls):
    dist = dist_cls(
        loc=torchrl_on_policy_core.torch.zeros(1),
        scale=torchrl_on_policy_core.torch.ones(1),
    )
    assert dist.support is torchrl_on_policy_core.torch.distributions.constraints.real


def test_on_policy_core_tanhnormal_support_fget_direct_call():
    dist = torchrl_on_policy_core._TanhNormal(
        loc=torchrl_on_policy_core.torch.zeros(1),
        scale=torchrl_on_policy_core.torch.ones(1),
    )
    support_value = torchrl_on_policy_core._TanhNormal.support.fget(dist)
    assert support_value is torchrl_on_policy_core.torch.distributions.constraints.real


def test_continuous_ppo_actor_uses_unsquashed_independent_normal():
    ppo = torchrl_on_policy_core
    env = SimpleNamespace(
        io_contract=EnvIOContract(
            observation=ObservationContract(mode="vector", raw_shape=(3,), vector_dim=3),
            action=ActionContract(
                kind="continuous",
                dim=2,
                low=np.asarray([-2.0, -1.0], dtype=np.float32),
                high=np.asarray([2.0, 1.0], dtype=np.float32),
            ),
        ),
        obs_dim=3,
        act_dim=2,
        action_low=np.asarray([-2.0, -1.0], dtype=np.float32),
        action_high=np.asarray([2.0, 1.0], dtype=np.float32),
        obs_lb=None,
        obs_width=None,
        is_discrete=False,
        env_conf=SimpleNamespace(env_name="fake"),
    )
    modules = ppo.build_modules(
        ppo.PPOConfig(policy_tag="actor-critic-mlp-32-32"),
        env,
        device=ppo.torch.device("cpu"),
    )

    probabilistic = modules.actor.module[1]
    assert probabilistic.distribution_class.__name__ == "IndependentNormal"
    assert probabilistic.distribution_kwargs == {}


def test_ppo_training_uses_adam_optimizer(tmp_path):
    ppo = torchrl_on_policy_core
    env = SimpleNamespace(
        io_contract=EnvIOContract(
            observation=ObservationContract(mode="vector", raw_shape=(3,), vector_dim=3),
            action=ActionContract(
                kind="continuous",
                dim=2,
                low=np.asarray([-2.0, -1.0], dtype=np.float32),
                high=np.asarray([2.0, 1.0], dtype=np.float32),
            ),
        ),
        obs_dim=3,
        act_dim=2,
        action_low=np.asarray([-2.0, -1.0], dtype=np.float32),
        action_high=np.asarray([2.0, 1.0], dtype=np.float32),
        obs_lb=None,
        obs_width=None,
        is_discrete=False,
        env_conf=SimpleNamespace(env_name="fake"),
    )
    config = ppo.PPOConfig(
        exp_dir=str(tmp_path),
        policy_tag="actor-critic-mlp-32-32",
    )
    modules = ppo.build_modules(config, env, device=ppo.torch.device("cpu"))
    runtime = op_deps.torchrl_runtime.TorchRLRuntime(
        device=ppo.torch.device("cpu"),
        collector_backend="multi_sync",
        single_env_backend="n/a",
        collector_workers=1,
    )

    training = ppo.build_training(config, env, modules, runtime=runtime)

    assert isinstance(training.optimizer, ppo.torch.optim.Adam)
    assert not isinstance(training.optimizer, ppo.torch.optim.AdamW)


def test_discrete_actor_net_handles_unbatched_atari_obs():
    ppo = torchrl_on_policy_core
    backbone = op_deps.backbone.build_backbone(
        op_deps.backbone.BackboneSpec(
            name="nature_cnn_atari",
            hidden_sizes=(),
            activation="relu",
            layer_norm=False,
        ),
        input_dim=64,
    )[0]
    head = op_deps.backbone.build_mlp_head(
        op_deps.backbone.HeadSpec(hidden_sizes=(), activation="relu"),
        input_dim=64,
        output_dim=6,
    )
    net = ppo._DiscreteActorNet(
        backbone,
        head,
        op_deps.torchrl_common.ObsScaler(None, None),
        obs_contract=ObservationContract(mode="pixels", raw_shape=(4, 84, 84), model_channels=4, image_size=84),
    )
    obs = ppo.torch.randint(0, 256, (4, 84, 84), dtype=ppo.torch.uint8).float() / 255.0
    logits = net(obs)
    assert logits.shape == (6,)


def test_resolve_backbone_name_for_pixel_contract():
    assert (
        op_deps.torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="pixels", raw_shape=(4, 84, 84, 1), model_channels=4, image_size=84),
        )
        == "nature_cnn_atari"
    )
    assert (
        op_deps.torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="pixels", raw_shape=(84, 84, 3), model_channels=3, image_size=84),
        )
        == "nature_cnn"
    )
    assert (
        op_deps.torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="vector", raw_shape=(17,), vector_dim=17),
        )
        == "mlp"
    )
