from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def patch_puffer_sac_engine_for_kiss(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._init_run_artifacts",
        lambda config: (Path(tmp_path), Path(tmp_path) / "metrics.jsonl", SimpleNamespace(save_both=lambda payload, iteration: None)),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._init_runtime",
        lambda config: (
            SimpleNamespace(
                problem_seed=1,
                noise_seed_0=2,
                act_dim=2,
                action_low=np.array([-1.0, -1.0]),
                action_high=np.array([1.0, 1.0]),
            ),
            torch.device("cpu"),
        ),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine.make_vector_env",
        lambda config: SimpleNamespace(reset=lambda seed=None: (np.zeros((1, 3), dtype=np.float32), {}), close=lambda: None),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine.infer_observation_spec",
        lambda config, obs_np: SimpleNamespace(mode="vector", vector_dim=3, raw_shape=(3,)),
    )
    monkeypatch.setattr("rl.pufferlib.sac.engine.prepare_obs_np", lambda obs_np, obs_spec: np.asarray(obs_np, dtype=np.float32))
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._build_training_components",
        lambda config, env_setup, obs_spec, obs_batch, device: (
            SimpleNamespace(
                actor=SimpleNamespace(sample=lambda obs_t, deterministic=False: (torch.zeros((obs_t.shape[0], 2)), None)),
                actor_backbone=torch.nn.Linear(3, 4),
                actor_head=torch.nn.Linear(4, 2),
                q1=torch.nn.Linear(5, 1),
                q2=torch.nn.Linear(5, 1),
                q1_target=torch.nn.Linear(5, 1),
                q2_target=torch.nn.Linear(5, 1),
                obs_scaler=torch.nn.Identity(),
                log_alpha=torch.nn.Parameter(torch.tensor(0.0)),
            ),
            SimpleNamespace(
                actor_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3),
                critic_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3),
                alpha_optimizer=torch.optim.Adam([torch.nn.Parameter(torch.tensor(0.0))], lr=1e-3),
            ),
            SimpleNamespace(state_dict=lambda: {}, load_state_dict=lambda state: None, size=0),
            SimpleNamespace(
                global_step=0,
                total_updates=0,
                best_return=float("nan"),
                best_actor_state=None,
                last_eval_return=0.0,
                last_heldout_return=None,
                start_time=0.0,
                eval_mark=0,
                log_mark=0,
                ckpt_mark=0,
            ),
        ),
    )
    monkeypatch.setattr("rl.pufferlib.sac.engine._log_header", lambda *args, **kwargs: None)
    monkeypatch.setattr("rl.pufferlib.sac.engine._train_loop", lambda *args, **kwargs: None)
    monkeypatch.setattr("rl.pufferlib.sac.engine.render_videos_if_enabled", lambda *args, **kwargs: None)
    monkeypatch.setattr(
        "importlib.import_module",
        lambda name: SimpleNamespace(normalize_eval_noise_mode=lambda mode: None, log_run_footer=lambda **kwargs: None),
    )


def patch_torchrl_ppo_core_for_kiss(monkeypatch):
    monkeypatch.setattr(
        "rl.torchrl.ppo.core.core_env_conf.build_seeded_env_conf_from_run",
        lambda **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(ensure_spaces=lambda: None),
            problem_seed=1,
            noise_seed_0=2,
        ),
    )
    monkeypatch.setattr(
        "rl.torchrl.ppo.core.torchrl_env_contract.resolve_env_io_contract",
        lambda env_conf, default_image_size=84: SimpleNamespace(
            observation=SimpleNamespace(mode="vector", vector_dim=3),
            action=SimpleNamespace(
                kind="continuous",
                dim=2,
                low=np.array([-1.0, -1.0]),
                high=np.array([1.0, 1.0]),
            ),
        ),
    )
    monkeypatch.setattr("rl.torchrl.ppo.core.torchrl_common.obs_scale_from_env", lambda env_conf: (None, None))


def patch_torchrl_sac_setup_for_kiss(monkeypatch):
    monkeypatch.setattr(
        "rl.torchrl.sac.setup.build_continuous_gym_env_setup",
        lambda **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                make=lambda: SimpleNamespace(reset=lambda seed=None: (None, {}), action_space=SimpleNamespace(seed=lambda seed: None)),
                state_space=SimpleNamespace(shape=(3,)),
            ),
            problem_seed=1,
            noise_seed_0=2,
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
            obs_lb=None,
            obs_width=None,
        ),
    )
    monkeypatch.setattr("rl.torchrl.sac.setup.sac_deps.torchrl_common.obs_scale_from_env", lambda env_conf: (None, None))
