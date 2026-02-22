import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rl.algos.backends.pufferlib.ppo import api as pufferlib_ppo


class _FakeVecEnv:
    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.single_action_space = SimpleNamespace(n=6)
        self.single_observation_space = SimpleNamespace(shape=(80, 4, 105))
        self._t = 0

    def reset(self, seed=None):
        _ = seed
        self._t = 0
        obs = np.zeros((self.num_envs, 80, 4, 105), dtype=np.uint8)
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, action):
        _ = action
        self._t += 1
        obs = np.zeros((self.num_envs, 80, 4, 105), dtype=np.uint8)
        rew = np.ones((self.num_envs,), dtype=np.float32)
        term = np.zeros((self.num_envs,), dtype=bool)
        trunc = np.zeros((self.num_envs,), dtype=bool)
        infos = []
        if self._t % 2 == 0:
            infos = [{"episode_return": float(self._t), "episode_length": int(self._t)}]
        return obs, rew, term, trunc, infos

    def close(self):
        return None


class _FakeVecEnvContinuous:
    def __init__(self, num_envs: int):
        self.num_envs = int(num_envs)
        self.single_action_space = SimpleNamespace(
            shape=(4,),
            low=-np.ones((4,), dtype=np.float32),
            high=np.ones((4,), dtype=np.float32),
        )
        self.single_observation_space = SimpleNamespace(shape=(24,))
        self._t = 0

    def reset(self, seed=None):
        _ = seed
        self._t = 0
        obs = np.zeros((self.num_envs, 24), dtype=np.float32)
        infos = [{} for _ in range(self.num_envs)]
        return obs, infos

    def step(self, action):
        action = np.asarray(action)
        assert action.shape == (self.num_envs, 4)
        self._t += 1
        obs = np.zeros((self.num_envs, 24), dtype=np.float32)
        rew = np.ones((self.num_envs,), dtype=np.float32)
        term = np.zeros((self.num_envs,), dtype=bool)
        trunc = np.zeros((self.num_envs,), dtype=bool)
        infos = []
        if self._t % 2 == 0:
            infos = [{"episode_return": float(self._t), "episode_length": int(self._t)}]
        return obs, rew, term, trunc, infos

    def close(self):
        return None


class _FakeVectorModule:
    class Serial:
        pass

    class Multiprocessing:
        pass

    @staticmethod
    def make(env_creator, env_kwargs=None, backend=None, num_envs=1, seed=0, **kwargs):
        _ = env_creator, env_kwargs, backend, seed, kwargs
        return _FakeVecEnv(num_envs=int(num_envs))


class _FakeAtariModule:
    @staticmethod
    def env_creator(game_name):
        _ = game_name

        def _thunk():
            raise RuntimeError("not used in fake vector backend")

        return _thunk


def test_puffer_config_from_dict_converts_hidden_sizes():
    cfg = pufferlib_ppo.PufferPPOConfig.from_dict(
        {
            "actor_head_hidden_sizes": [64, 32],
            "critic_head_hidden_sizes": [64],
            "backbone_hidden_sizes": [],
        }
    )
    assert cfg.actor_head_hidden_sizes == (64, 32)
    assert cfg.critic_head_hidden_sizes == (64,)
    assert cfg.backbone_hidden_sizes == ()


def test_puffer_register_delegates_to_registry(monkeypatch):
    calls = []

    def fake_register_algo(name, config_cls, train_fn):
        calls.append((name, config_cls, train_fn))

    monkeypatch.setattr("rl.algos.registry.register_algo", fake_register_algo)
    pufferlib_ppo.register()

    assert len(calls) == 1
    name, config_cls, train_fn = calls[0]
    assert name == "ppo_puffer"
    assert config_cls is pufferlib_ppo.PufferPPOConfig
    assert train_fn is pufferlib_ppo.train_ppo_puffer


def test_to_puffer_game_name_from_atari_tag():
    assert pufferlib_ppo._to_puffer_game_name("atari:Pong") == "pong"
    assert pufferlib_ppo._to_puffer_game_name("ALE/Breakout-v5") == "breakout"


def test_resolve_gym_env_name_from_tag():
    env_name, env_kwargs = pufferlib_ppo._resolve_gym_env_name("bw-heur")
    assert env_name == "BipedalWalker-v3"
    assert isinstance(env_kwargs, dict)


def test_train_ppo_puffer_fake_vector_smoke(monkeypatch, tmp_path: Path):
    def _fake_import_modules():
        return object(), _FakeVectorModule, _FakeAtariModule

    monkeypatch.setattr(pufferlib_ppo, "import_pufferlib_modules", _fake_import_modules)

    cfg = pufferlib_ppo.PufferPPOConfig(
        exp_dir=str(tmp_path / "exp"),
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
        log_interval=1,
        device="cpu",
    )
    result = pufferlib_ppo.train_ppo_puffer(cfg)

    assert result.num_iterations == 2
    assert (tmp_path / "exp" / "config.json").exists()
    assert (tmp_path / "exp" / "metrics.jsonl").exists()


def test_train_ppo_puffer_continuous_fake_vector_smoke(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(
        pufferlib_ppo,
        "_make_vector_env",
        lambda _cfg: _FakeVecEnvContinuous(num_envs=2),
    )

    cfg = pufferlib_ppo.PufferPPOConfig(
        exp_dir=str(tmp_path / "exp_cont"),
        env_tag="bw-heur",
        seed=9,
        total_timesteps=8,
        num_envs=2,
        num_steps=2,
        num_minibatches=2,
        update_epochs=1,
        learning_rate=1e-3,
        backbone_name="mlp",
        backbone_hidden_sizes=(32,),
        actor_head_hidden_sizes=(32,),
        critic_head_hidden_sizes=(32,),
        share_backbone=True,
        vector_backend="serial",
        eval_interval=0,
        log_interval=1,
        device="cpu",
        log_std_init=-0.5,
    )
    result = pufferlib_ppo.train_ppo_puffer(cfg)

    assert result.num_iterations == 2
    assert (tmp_path / "exp_cont" / "config.json").exists()
    assert (tmp_path / "exp_cont" / "metrics.jsonl").exists()


def test_train_ppo_puffer_rejects_invalid_eval_noise_mode():
    cfg = pufferlib_ppo.PufferPPOConfig(eval_noise_mode="invalid-mode")
    with pytest.raises(ValueError, match="eval_noise_mode must be one of"):
        pufferlib_ppo.train_ppo_puffer(cfg)


def test_train_ppo_puffer_resume_from_checkpoint(monkeypatch, tmp_path: Path):
    def _fake_import_modules():
        return object(), _FakeVectorModule, _FakeAtariModule

    monkeypatch.setattr(pufferlib_ppo, "import_pufferlib_modules", _fake_import_modules)

    first_dir = tmp_path / "exp_first"
    first_cfg = pufferlib_ppo.PufferPPOConfig(
        exp_dir=str(first_dir),
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
    resume_path = first_dir / "checkpoints" / "checkpoint_last.pt"
    assert resume_path.exists()

    second_dir = tmp_path / "exp_resume"
    second_cfg = pufferlib_ppo.PufferPPOConfig(
        exp_dir=str(second_dir),
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
    metrics_lines = (second_dir / "metrics.jsonl").read_text(encoding="utf-8").strip().splitlines()
    assert len(metrics_lines) == 1
    metric = json.loads(metrics_lines[0])
    assert metric["iteration"] == 3
    assert (second_dir / "checkpoints" / "checkpoint_last.pt").exists()


def test_train_ppo_puffer_renders_video_when_enabled(monkeypatch, tmp_path: Path):
    def _fake_import_modules():
        return object(), _FakeVectorModule, _FakeAtariModule

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

    monkeypatch.setattr(pufferlib_ppo, "import_pufferlib_modules", _fake_import_modules)
    monkeypatch.setattr(
        pufferlib_ppo,
        "_build_eval_env_conf",
        lambda *_args, **_kwargs: SimpleNamespace(gym_conf=SimpleNamespace(max_steps=1)),
    )
    monkeypatch.setattr("common.video.render_policy_videos", _fake_render_policy_videos)

    cfg = pufferlib_ppo.PufferPPOConfig(
        exp_dir=str(tmp_path / "exp_video"),
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
    assert call["seed_base"] == 11
    assert call["video_dir"].endswith("exp_video/videos")
