import pytest

import rl.backbone as backbone
from rl.core import env_contract as torchrl_env_contract
from rl.core import runtime as torchrl_common
from rl.core.env_contract import ActionContract, EnvIOContract, ObservationContract
from rl.torchrl.ppo import core as torchrl_on_policy_core
from rl.torchrl.ppo.config import PPOConfig
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


def test_discrete_actor_net_handles_unbatched_atari_obs():
    ppo = torchrl_on_policy_core
    trunk = backbone.build_backbone(
        backbone.BackboneSpec(
            name="nature_cnn_atari",
            hidden_sizes=(),
            activation="relu",
            layer_norm=False,
        ),
        input_dim=64,
    )[0]
    head = backbone.build_mlp_head(
        backbone.HeadSpec(hidden_sizes=(), activation="relu"),
        input_dim=64,
        output_dim=6,
    )
    net = ppo._DiscreteActorNet(
        trunk,
        head,
        torchrl_common.ObsScaler(None, None),
        obs_contract=ObservationContract(mode="pixels", raw_shape=(4, 84, 84), model_channels=4, image_size=84),
    )
    obs = ppo.torch.randint(0, 256, (4, 84, 84), dtype=ppo.torch.uint8).float() / 255.0
    logits = net(obs)
    assert logits.shape == (6,)


def test_resolve_backbone_name_for_pixel_contract():
    assert (
        torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="pixels", raw_shape=(4, 84, 84, 1), model_channels=4, image_size=84),
        )
        == "nature_cnn_atari"
    )
    assert (
        torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="pixels", raw_shape=(84, 84, 3), model_channels=3, image_size=84),
        )
        == "nature_cnn"
    )
    assert (
        torchrl_env_contract.resolve_backbone_name(
            "mlp",
            ObservationContract(mode="vector", raw_shape=(17,), vector_dim=17),
        )
        == "mlp"
    )


def test_ppo_build_pipeline(monkeypatch, tmp_path):
    class _EnvConf:
        obs_mode = "vector"

        def ensure_spaces(self):
            return None

    monkeypatch.setattr(
        torchrl_on_policy_core.envs,
        "conf_for_run",
        lambda **_kwargs: type("R", (), {"env_conf": _EnvConf(), "problem_seed": 3, "noise_seed_0": 4})(),
    )
    monkeypatch.setattr(
        torchrl_on_policy_core.env_contract,
        "resolve_env_io_contract",
        lambda *_args, **_kwargs: EnvIOContract(
            observation=ObservationContract(mode="vector", raw_shape=(3,), vector_dim=3),
            action=ActionContract(
                kind="continuous",
                dim=2,
                low=torchrl_on_policy_core.np.array([-1.0, -1.0], dtype=torchrl_on_policy_core.np.float32),
                high=torchrl_on_policy_core.np.array([1.0, 1.0], dtype=torchrl_on_policy_core.np.float32),
            ),
        ),
    )
    monkeypatch.setattr(
        torchrl_on_policy_core.runtime,
        "obs_scale_from_env",
        lambda _env_conf: (None, None),
    )

    cfg = PPOConfig(
        exp_dir=str(tmp_path),
        env_tag="pend",
        total_timesteps=8,
        num_envs=2,
        num_steps=4,
        num_minibatches=1,
        epochs=1,
    )
    env = torchrl_on_policy_core.build_env_setup(cfg)
    modules = torchrl_on_policy_core.build_modules(cfg, env, device=torchrl_on_policy_core.torch.device("cpu"))
    training = torchrl_on_policy_core.build_training(
        cfg,
        env,
        modules,
        runtime=type("RT", (), {"collector_backend": "none", "single_env_backend": "serial"})(),
    )
    assert env.problem_seed == 3
    assert training.frames_per_batch == 8
    assert training.num_iterations == 1
    assert training.env is None
