from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import torch


def test_kiss_tidy_e_torchrl_sac_sampling_vector(monkeypatch, tmp_path):
    monkeypatch.setattr("rl.core.runtime.select_device", lambda d: torch.device("cpu"))
    from rl.torchrl.sac import sac_train_loop
    from rl.torchrl.sac import sac_trainer_phase_a as pha
    from rl.torchrl.sac import sac_trainer_phase_b_impl as phb
    from rl.torchrl.sac.config import SACConfig
    from rl.torchrl.sac.sac_setup_build import (
        build_env_setup,
        build_modules,
        build_training,
    )
    from rl.torchrl.sac.sac_setup_types import _TrainState

    assert callable(sac_train_loop.train_sac)
    monkeypatch.setattr(
        "rl.torchrl.sac.sac_setup_build.build_env_setup",
        lambda _config, **kwargs: SimpleNamespace(
            env_conf=SimpleNamespace(
                from_pixels=False,
                ensure_spaces=lambda: None,
                make=lambda: SimpleNamespace(
                    reset=lambda seed=None: (None, {}),
                    action_space=SimpleNamespace(seed=lambda seed: None),
                ),
                state_space=SimpleNamespace(shape=(3,)),
            ),
            problem_seed=1,
            noise_seed_0=2,
            act_dim=1,
            action_low=np.array([-1.0], dtype=np.float32),
            action_high=np.array([1.0], dtype=np.float32),
            obs_lb=None,
            obs_width=None,
        ),
    )
    monkeypatch.setattr(
        "rl.torchrl.sac.sac_setup_build.runtime.obs_scale_from_env",
        lambda env_conf: (None, None),
    )
    sc = SACConfig(
        env_tag="pend",
        exp_dir=str(tmp_path),
        device="cpu",
        replay_size=64,
        batch_size=4,
        total_timesteps=1,
    )
    env = build_env_setup(sc)
    mods = build_modules(sc, env, device=torch.device("cpu"))
    tr = build_training(sc, mods)
    assert tr.replay is not None
    monkeypatch.setattr("rl.registry.register_algo", lambda *a, **k: None)
    sac_train_loop.register()
    payload = pha.checkpoint_payload(mods, tr, _TrainState(), step=1)
    assert "step" in payload
    st = pha.resume_if_requested(sc, mods, tr, device=torch.device("cpu"))
    assert st.start_step == 0
    _ = pha.build_eval_policy(mods, env, torch.device("cpu"))
    monkeypatch.setattr(
        "rl.core.episode_rollout.collect_denoised_trajectory",
        lambda *a, **k: (SimpleNamespace(rreturn=0.3), None),
    )
    assert pha.evaluate_actor(sc, env, mods, device=torch.device("cpu"), eval_seed=0) == 0.3
    td = __import__("tensordict").TensorDict(
        {
            "observation": torch.zeros(2, 3),
            "action": torch.zeros(2, 1),
            "next": __import__("tensordict").TensorDict(
                {
                    "observation": torch.zeros(2, 3),
                    "reward": torch.zeros(2, 1),
                    "done": torch.zeros(2, 1, dtype=torch.bool),
                },
                batch_size=[2],
            ),
        },
        batch_size=[2],
    )
    flat = phb.flatten_batch_to_transitions(td)
    phb.normalize_actions_for_replay(flat, action_low=env.action_low, action_high=env.action_high)
    monkeypatch.setattr("rl.torchrl.sac.loop.evaluate_if_due", lambda *a, **k: None)
    monkeypatch.setattr("rl.torchrl.sac.loop.log_if_due", lambda *a, **k: None)
    monkeypatch.setattr("rl.torchrl.sac.loop.checkpoint_if_due", lambda *a, **k: None)
    tr_state = SimpleNamespace(
        best_return=0.0,
        last_eval_return=0.0,
        last_heldout_return=None,
        start_step=0,
        best_actor_state=None,
    )
    rt = sc.resolve_runtime(capabilities=None)
    phb.run_sac_eval_log_checkpoint(
        sc,
        env,
        mods,
        tr,
        tr_state,
        step=1,
        runtime=rt,
        start_time=0.0,
        latest_losses={"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        total_updates=0,
        evaluate_for_best=lambda *a, **k: 0.0,
    )
    sc2 = SACConfig(
        env_tag="pend",
        exp_dir=str(tmp_path),
        device="cpu",
        replay_size=128,
        batch_size=4,
        learning_starts=0,
        update_every=1,
        updates_per_step=1,
        learner_update_chunk_size=1,
        total_timesteps=4,
    )
    out = phb.process_sac_batch(
        td,
        sc2,
        mods,
        tr,
        rt,
        env,
        {"loss_actor": 0.0, "loss_critic": 0.0, "loss_alpha": 0.0},
        0,
    )
    assert len(out) == 3
    losses = phb.update_step(sc2, mods, tr, device=torch.device("cpu"), batch_size=4)
    assert "loss_actor" in losses
    col = MagicMock()
    col.set_seed = MagicMock()
    col.shutdown = MagicMock()
    col.__iter__ = MagicMock(return_value=iter(()))
    monkeypatch.setattr("torchrl.collectors.Collector", lambda *a, **kw: col)
    stub_env = MagicMock()
    stub_env.reset = MagicMock(return_value=(td["observation"][0:1].numpy(), {}))
    stub_env.step = MagicMock(return_value=(td["observation"][0:1].numpy(), 0.0, True, False, {}))
    monkeypatch.setattr("rl.torchrl.sac.setup._make_collect_env_sac", lambda *a, **k: stub_env)
    assert phb.build_sac_collector(sc, env, mods, runtime=rt, total_frames=4) is not None


def test_kiss_tidy_e_sparse_jl_and_vector_fakes():
    from sampling.sparse_jl_t_accum import (
        accumulate_into,
        accumulate_into_wr,
        accumulate_noise_into,
    )
    from sampling.sparse_jl_t_hash import (
        CHUNK_SIZE,
        compute_rows_and_signs,
        compute_rows_and_signs_wr,
        vmix64,
        vmix64_inplace,
        wrap_int64,
    )
    from sampling.sparse_jl_t_transforms import block_sparse_hash_scatter_from_nz_t
    from testing_support.vector_fakes import (
        FakePufferVecEnv,
        FakePufferVecEnvContinuous,
    )

    dvc = torch.device("cpu")
    y = torch.zeros(8, dtype=torch.float32)
    accumulate_into(y, torch.tensor([1.0, -1.0]), 0, 8, 2, 1)
    accumulate_into_wr(y, torch.ones(1), 0, 8, 1, 2)
    accumulate_noise_into(y, 4, d=8, s=2, seed_jl=3, seed_noise=4, sigma=0.1, prob=0.5, chunk_size=2)
    assert wrap_int64(-1) < 0
    z = torch.tensor([1, 2, 3], dtype=torch.int64)
    t2 = torch.empty_like(z)
    vmix64_inplace(z.clone(), t2)
    _ = vmix64(torch.tensor([4], dtype=torch.int64))
    r, _s = compute_rows_and_signs(torch.tensor([0, 1], dtype=torch.int64), 8, 2, 7, dvc)
    assert r.shape[0] == 2
    r2, _s2 = compute_rows_and_signs_wr(torch.tensor([0], dtype=torch.int64), 8, 2, 7, dvc)
    assert r2.shape[0] == 1
    _ = CHUNK_SIZE > 0
    out2 = block_sparse_hash_scatter_from_nz_t(
        torch.tensor([0, 1], dtype=torch.int64),
        torch.ones(2),
        16,
        2,
        9,
        torch.float32,
        dvc,
    )
    assert out2.shape[0] == 16
    fv = FakePufferVecEnv(2)
    o, _ = fv.reset(seed=0)
    fv.step(np.zeros(2, dtype=np.int64))
    fv.close()
    fc = FakePufferVecEnvContinuous(2)
    o2, _ = fc.reset()
    fc.step(np.zeros((2, 4), dtype=np.float32))
    fc.close()
    assert o.shape[0] == 2 and o2.shape[0] == 2
