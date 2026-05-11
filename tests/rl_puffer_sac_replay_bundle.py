from types import SimpleNamespace

import numpy as np
import torch
import torch.nn as nn

from testing_support.dyn_import import import_dotted


def build_replay_sac_bundle():
    puffer_sac = import_dotted("rl", "pufferlib", "sac")
    env_utils = import_dotted("rl", "pufferlib", "sac", "env_utils")
    model_utils = import_dotted("rl", "pufferlib", "sac", "model_utils")
    replay_mod = import_dotted("rl", "pufferlib", "sac", "replay")
    ReplayBuffer = replay_mod.ReplayBuffer
    runtime_utils = import_dotted("rl", "pufferlib", "sac", "runtime_utils")
    ObsScaler = runtime_utils.ObsScaler
    core_replay = import_dotted("rl", "core", "replay")
    make_replay_buffer = core_replay.make_replay_buffer

    env_setup = env_utils.EnvSetup(
        env_conf=SimpleNamespace(from_pixels=False, pixels_only=True),
        problem_seed=0,
        noise_seed_0=0,
        obs_lb=None,
        obs_width=None,
        act_dim=2,
        action_low=np.asarray([-1.0, -1.0], dtype=np.float32),
        action_high=np.asarray([1.0, 1.0], dtype=np.float32),
    )
    obs_spec = env_utils.ObservationSpec(mode="vector", raw_shape=(3,), vector_dim=3)
    cfg = puffer_sac.SACConfig(
        backbone_hidden_sizes=(8,),
        actor_head_hidden_sizes=(),
        critic_head_hidden_sizes=(),
        batch_size=4,
        target_entropy=-2.0,
    )
    modules = model_utils.build_modules(cfg, env_setup, obs_spec, device=torch.device("cpu"))
    optimizers = model_utils.build_optimizers(cfg, modules)
    replay = ReplayBuffer(obs_shape=(3,), act_dim=2, capacity=32)

    return SimpleNamespace(
        np=np,
        torch=torch,
        nn=nn,
        env_utils=env_utils,
        model_utils=model_utils,
        ReplayBuffer=ReplayBuffer,
        ObsScaler=ObsScaler,
        make_replay_buffer=make_replay_buffer,
        env_setup=env_setup,
        obs_spec=obs_spec,
        cfg=cfg,
        modules=modules,
        optimizers=optimizers,
        replay=replay,
    )
