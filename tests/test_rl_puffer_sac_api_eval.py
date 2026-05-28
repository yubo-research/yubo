def test_eval_utils_paths(monkeypatch, tmp_path):
    import numpy as np
    import torch
    from rl_puffer_sac_eval_paths_lib import (
        run_eval_utils_heldout_and_due,
        run_eval_utils_metrics_and_maybe_eval,
        run_eval_utils_policy_and_evaluate,
        run_eval_utils_video_if_enabled,
    )

    from testing_support.dyn_import import import_dotted

    puffer_sac = import_dotted("rl", "pufferlib", "sac")
    eval_utils = import_dotted("rl", "pufferlib", "sac", "eval_utils")
    env_utils = import_dotted("rl", "pufferlib", "sac", "env_utils")
    model_utils = import_dotted("rl", "pufferlib", "sac", "model_utils")

    env_setup = env_utils.EnvSetup(
        env_conf=__import__("types").SimpleNamespace(from_pixels=False, pixels_only=True),
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
    _ = model_utils.build_optimizers(cfg, modules)

    actor = run_eval_utils_policy_and_evaluate(monkeypatch, eval_utils, puffer_sac, env_setup, obs_spec, cfg)
    run_eval_utils_heldout_and_due(monkeypatch, eval_utils, puffer_sac, env_setup, obs_spec, actor)
    run_eval_utils_metrics_and_maybe_eval(monkeypatch, eval_utils, puffer_sac, env_setup, obs_spec, modules, tmp_path)
    run_eval_utils_video_if_enabled(monkeypatch, eval_utils, puffer_sac, env_setup, modules, obs_spec, tmp_path)
