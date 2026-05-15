from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def patch_puffer_sac_engine_for_kiss(monkeypatch, tmp_path):
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._init_run_artifacts",
        lambda config: (
            Path(tmp_path),
            Path(tmp_path) / "metrics.jsonl",
            SimpleNamespace(save_both=lambda payload, iteration: None),
        ),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._init_runtime",
        lambda config: (
            SimpleNamespace(
                problem_seed=1,
                noise_seed_0=2,
                obs_dim=3,
                act_dim=2,
                action_low=np.array([-1.0, -1.0]),
                action_high=np.array([1.0, 1.0]),
            ),
            torch.device("cpu"),
        ),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine.make_vector_env",
        lambda config: SimpleNamespace(
            reset=lambda seed=None: (np.zeros((1, 3), dtype=np.float32), {}),
            close=lambda: None,
        ),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine.infer_observation_spec",
        lambda config, obs_np: SimpleNamespace(mode="vector", vector_dim=3, raw_shape=(3,)),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine.prepare_obs_np",
        lambda obs_np, obs_spec: np.asarray(obs_np, dtype=np.float32),
    )
    monkeypatch.setattr(
        "rl.pufferlib.sac.engine._build_training_components",
        lambda config, env_setup, obs_spec, obs_batch, device: (
            SimpleNamespace(
                actor=SimpleNamespace(
                    sample=lambda obs_t, deterministic=False: (
                        torch.zeros((obs_t.shape[0], 2)),
                        None,
                    )
                ),
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


def patch_torchrl_ppo_core_for_kiss(monkeypatch):
    from rl.core.env_contract import ActionContract, ObservationContract

    mock_env_setup = SimpleNamespace(
        env_conf=SimpleNamespace(
            ensure_spaces=lambda: None,
            gym_conf=None,
            problem_seed=1,
            make_gym_env=lambda seed=0, **k: SimpleNamespace(
                reset=lambda seed=None: (None, {}),
                observation_space=SimpleNamespace(shape=(3,), dtype=np.float32),
                action_space=SimpleNamespace(shape=(2,), dtype=np.float32),
            ),
        ),
        problem_seed=1,
        noise_seed_0=2,
        obs_dim=3,
        act_dim=2,
        action_low=np.array([-1.0, -1.0]),
        action_high=np.array([1.0, 1.0]),
        obs_lb=np.array([-1.0, -1.0, -1.0]),
        obs_width=np.array([2.0, 2.0, 2.0]),
        is_discrete=False,
        io_contract=SimpleNamespace(
            observation=ObservationContract(mode="vector", raw_shape=(3,), vector_dim=3),
            action=ActionContract(kind="continuous", dim=2, low=np.array([-1.0, -1.0]), high=np.array([1.0, 1.0])),
        ),
    )
    monkeypatch.setattr("rl.torchrl.ppo.core_env_setup.build_env_setup", lambda **kwargs: mock_env_setup)
    monkeypatch.setattr("rl.torchrl.ppo.core.build_env_setup", lambda **kwargs: mock_env_setup)
    monkeypatch.setattr("rl.torchrl.ppo.core_build.env_contract.resolve_env_io_contract", lambda env_conf, **k: mock_env_setup.io_contract)
    monkeypatch.setattr("rl.torchrl.ppo.core_build.ObsScaler", lambda *a, **k: torch.nn.Identity())

    # Mock collect_env to avoid real construction
    monkeypatch.setattr(
        "rl.torchrl.ppo.core_build.tr_envs.SerialEnv",
        lambda *a, **k: SimpleNamespace(
            set_seed=lambda *a: None,
            num_envs=1,
            specs=SimpleNamespace(
                observation_spec=SimpleNamespace(get=lambda k: SimpleNamespace(shape=(3,))),
                action_spec=SimpleNamespace(shape=(2,), low=torch.ones(2), high=torch.ones(2)),
            ),
        ),
    )


def patch_torchrl_sac_setup_for_kiss(monkeypatch):
    monkeypatch.setattr(
        "rl.torchrl.sac.sac_setup_build.build_env_setup",
        lambda config, **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                make=lambda: SimpleNamespace(
                    reset=lambda seed=None: (None, {}),
                    action_space=SimpleNamespace(seed=lambda seed: None),
                ),
                state_space=SimpleNamespace(shape=(3,)),
                gym_conf=None,
            ),
            problem_seed=1,
            noise_seed_0=2,
            obs_dim=3,
            act_dim=2,
            action_low=np.array([-1.0, -1.0], dtype=np.float32),
            action_high=np.array([1.0, 1.0], dtype=np.float32),
            obs_lb=np.array([-1.0, -1.0, -1.0], dtype=np.float32),
            obs_width=np.array([2.0, 2.0, 2.0], dtype=np.float32),
        ),
    )
