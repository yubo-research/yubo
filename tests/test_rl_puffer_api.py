import json
import os
import tempfile
from types import SimpleNamespace

from testing_support.vector_fakes import FakePufferVecEnv, FakePufferVecEnvContinuous


def test_puffer_config_from_dict_converts_hidden_sizes():
    import rl.pufferlib.ppo as pufferlib_ppo

    cfg = pufferlib_ppo.PufferPPOConfig.from_dict(
        {
            "env_tag": "cheetah",
            "policy_tag": "mlp-32-16",
            "actor_head_hidden_sizes": [64, 32],
            "critic_head_hidden_sizes": [64],
            "backbone_hidden_sizes": [],
        }
    )
    assert cfg.actor_head_hidden_sizes == (64, 32)
    assert cfg.critic_head_hidden_sizes == (64,)
    assert cfg.backbone_hidden_sizes == ()


def test_puffer_config_from_dict_uses_env_defaults():
    import rl.pufferlib.ppo as pufferlib_ppo

    cfg = pufferlib_ppo.PufferPPOConfig.from_dict({"env_tag": "cheetah", "policy_tag": "mlp-32-16"})
    assert cfg.backbone_hidden_sizes == (64, 64)


def test_puffer_register_delegates_to_registry():
    import rl.pufferlib.ppo as pufferlib_ppo
    import rl.registry as registry

    calls = []

    def fake_register_algo(name, config_cls, train_fn, *, backend=None):
        calls.append((name, config_cls, train_fn, backend))

    prev = registry.register_algo
    registry.register_algo = fake_register_algo
    try:
        pufferlib_ppo.register()
    finally:
        registry.register_algo = prev

    assert len(calls) == 1
    name, config_cls, train_fn, backend = calls[0]
    assert name == "ppo"
    assert backend == "pufferlib"
    assert config_cls is pufferlib_ppo.PufferPPOConfig
    assert train_fn is pufferlib_ppo.train_ppo_puffer


def _train_ppo_with_fake_vec(fake_vec_cls, *, exp_dir: str, extra_cfg: dict):
    import rl.pufferlib.ppo as pufferlib_ppo
    import rl.pufferlib.ppo.engine_helpers as engine_helpers

    prev = engine_helpers.make_vector_env
    engine_helpers.make_vector_env = lambda _cfg: fake_vec_cls(num_envs=2)
    try:
        base = dict(
            exp_dir=exp_dir,
            total_timesteps=8,
            num_envs=2,
            num_steps=2,
            num_minibatches=2,
            update_epochs=1,
            learning_rate=1e-3,
            actor_head_hidden_sizes=(32,),
            critic_head_hidden_sizes=(32,),
            share_backbone=True,
            vector_backend="serial",
            eval_interval=0,
            log_interval=1,
            device="cpu",
        )
        base.update(extra_cfg)
        cfg = pufferlib_ppo.PufferPPOConfig(**base)
        return pufferlib_ppo.train_ppo_puffer(cfg)
    finally:
        engine_helpers.make_vector_env = prev


def test_train_ppo_puffer_fake_vector_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = os.path.join(tmp, "exp")
        result = _train_ppo_with_fake_vec(
            FakePufferVecEnv,
            exp_dir=exp_dir,
            extra_cfg=dict(
                env_tag="atari:Pong",
                seed=7,
            ),
        )
        assert result.num_iterations == 2
        assert os.path.exists(os.path.join(exp_dir, "config.json"))
        assert os.path.exists(os.path.join(exp_dir, "metrics.jsonl"))


def test_train_ppo_puffer_continuous_fake_vector_smoke():
    with tempfile.TemporaryDirectory() as tmp:
        exp_dir = os.path.join(tmp, "exp_cont")
        result = _train_ppo_with_fake_vec(
            FakePufferVecEnvContinuous,
            exp_dir=exp_dir,
            extra_cfg=dict(
                env_tag="bw-heur",
                seed=9,
                backbone_name="mlp",
                backbone_hidden_sizes=(32,),
                log_std_init=-0.5,
            ),
        )
        assert result.num_iterations == 2
        assert os.path.exists(os.path.join(exp_dir, "config.json"))
        assert os.path.exists(os.path.join(exp_dir, "metrics.jsonl"))


def test_train_ppo_puffer_rejects_invalid_eval_noise_mode():
    import rl.pufferlib.ppo as pufferlib_ppo

    cfg = pufferlib_ppo.PufferPPOConfig(eval_noise_mode="invalid-mode")
    try:
        pufferlib_ppo.train_ppo_puffer(cfg)
    except ValueError as e:
        assert "eval_noise_mode must be one of" in str(e)
    else:
        raise AssertionError("expected ValueError")


def test_train_ppo_puffer_resume_from_checkpoint():
    import rl.pufferlib.ppo as pufferlib_ppo
    import rl.pufferlib.ppo.engine_helpers as engine_helpers

    prev = engine_helpers.make_vector_env
    engine_helpers.make_vector_env = lambda _cfg: FakePufferVecEnv(num_envs=2)
    try:
        with tempfile.TemporaryDirectory() as tmp:
            first_dir = os.path.join(tmp, "exp_first")
            first_cfg = pufferlib_ppo.PufferPPOConfig(
                exp_dir=first_dir,
                env_tag="atari:Pong",
                seed=7,
                total_timesteps=8,
                num_envs=2,
                num_steps=2,
                num_minibatches=2,
                update_epochs=1,
                learning_rate=1e-3,
                actor_head_hidden_sizes=(32,),
                critic_head_hidden_sizes=(32,),
                share_backbone=True,
                vector_backend="serial",
                eval_interval=0,
                checkpoint_interval=1,
                log_interval=0,
                device="cpu",
            )
            first_result = pufferlib_ppo.train_ppo_puffer(first_cfg)
            assert first_result.num_iterations == 2
            resume_path = os.path.join(first_dir, "checkpoints", "checkpoint_last.pt")
            assert os.path.exists(resume_path)

            second_dir = os.path.join(tmp, "exp_resume")
            second_cfg = pufferlib_ppo.PufferPPOConfig(
                exp_dir=second_dir,
                env_tag="atari:Pong",
                seed=7,
                total_timesteps=12,
                num_envs=2,
                num_steps=2,
                num_minibatches=2,
                update_epochs=1,
                learning_rate=1e-3,
                actor_head_hidden_sizes=(32,),
                critic_head_hidden_sizes=(32,),
                share_backbone=True,
                vector_backend="serial",
                eval_interval=0,
                checkpoint_interval=1,
                resume_from=str(resume_path),
                log_interval=0,
                device="cpu",
            )
            second_result = pufferlib_ppo.train_ppo_puffer(second_cfg)
            assert second_result.num_iterations == 3
            metrics_path = os.path.join(second_dir, "metrics.jsonl")
            metrics_lines = open(metrics_path, encoding="utf-8").read().strip().splitlines()
            assert len(metrics_lines) == 1
            metric = json.loads(metrics_lines[0])
            assert metric["iteration"] == 3
            assert os.path.exists(os.path.join(second_dir, "checkpoints", "checkpoint_last.pt"))
    finally:
        engine_helpers.make_vector_env = prev


def test_train_ppo_puffer_renders_video_when_enabled():
    import common.video as video_mod
    import rl.pufferlib.ppo as pufferlib_ppo
    import rl.pufferlib.ppo.engine_helpers as engine_helpers

    video_calls: list[dict] = []

    def _fake_render_policy_videos(
        env_conf,
        policy,
        *,
        video_dir,
        video_prefix,
        num_episodes,
        num_video_episodes,
        episode_selection,
        seed_base,
    ):
        _ = env_conf, policy
        video_calls.append(
            {
                "video_dir": str(video_dir),
                "video_prefix": video_prefix,
                "num_episodes": int(num_episodes),
                "num_video_episodes": int(num_video_episodes),
                "episode_selection": str(episode_selection),
                "seed_base": int(seed_base),
            }
        )

    prev_make = engine_helpers.make_vector_env
    prev_build = engine_helpers.build_eval_env_conf
    prev_render = video_mod.render_policy_videos

    engine_helpers.make_vector_env = lambda _cfg: FakePufferVecEnv(num_envs=2)

    def _stub_build_eval_env_conf(_config, *, obs_spec):
        _ = obs_spec
        return SimpleNamespace(gym_conf=SimpleNamespace(max_steps=1))

    engine_helpers.build_eval_env_conf = _stub_build_eval_env_conf
    video_mod.render_policy_videos = _fake_render_policy_videos
    try:
        with tempfile.TemporaryDirectory() as tmp:
            exp_dir = os.path.join(tmp, "exp_video")
            cfg = pufferlib_ppo.PufferPPOConfig(
                exp_dir=exp_dir,
                env_tag="atari:Pong",
                seed=11,
                total_timesteps=8,
                num_envs=2,
                num_steps=2,
                num_minibatches=2,
                update_epochs=1,
                learning_rate=1e-3,
                actor_head_hidden_sizes=(32,),
                critic_head_hidden_sizes=(32,),
                share_backbone=True,
                vector_backend="serial",
                eval_interval=0,
                log_interval=0,
                device="cpu",
                video_enable=True,
                video_num_episodes=3,
                video_num_video_episodes=1,
            )
            _ = pufferlib_ppo.train_ppo_puffer(cfg)

        assert len(video_calls) == 1
        call = video_calls[0]
        assert call["video_prefix"] == "policy"
        assert call["num_episodes"] == 3
        assert call["num_video_episodes"] == 1
        assert call["episode_selection"] == "best"
        assert call["seed_base"] == 29
        assert str(call["video_dir"]).endswith(os.path.join("exp_video", "videos"))
    finally:
        engine_helpers.make_vector_env = prev_make
        engine_helpers.build_eval_env_conf = prev_build
        video_mod.render_policy_videos = prev_render
