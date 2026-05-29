from __future__ import annotations

import importlib

from .sac_train_loop_run import _finalize_sac_run, _resolve_sac_run_seeds, _run_sac_training_loop
from .sac_train_refs import _sac_training_refs


def train_sac(config):
    t = _sac_training_refs()
    _SAC_RUNTIME_CAPABILITIES = t.cfg_mod._SAC_RUNTIME_CAPABILITIES
    TrainResult = t.cfg_mod.TrainResult
    _checkpoint_payload = t.phase_a.checkpoint_payload
    _resume_if_requested = t.phase_a.resume_if_requested
    _build_sac_collector = t.phase_b.build_sac_collector
    _process_sac_batch = t.phase_b.process_sac_batch
    _run_sac_eval_log_checkpoint = t.phase_b.run_sac_eval_log_checkpoint
    build_env_setup = t.setup.build_env_setup
    build_modules = t.setup.build_modules
    build_training = t.setup.build_training

    with t.torchrl_common.temporary_distribution_validate_args(False):
        _resolve_sac_run_seeds(config, t)
        env = build_env_setup(config)
        runtime = config.resolve_runtime(capabilities=_SAC_RUNTIME_CAPABILITIES)
        modules = build_modules(config, env, device=runtime.device)
        training = build_training(config, modules)
        state = _resume_if_requested(config, modules, training, device=runtime.device)

        t.rl_logger.log_run_header("sac", config, env, training, runtime)
        t.rl_logger.log_rl_status(f"metrics={training.metrics_path} checkpoints={training.exp_dir / 'checkpoints'}")
        start_time, total_updates, latest_losses = _run_sac_training_loop(
            config,
            env,
            modules,
            training,
            state,
            runtime,
            t,
            _build_sac_collector=_build_sac_collector,
            _process_sac_batch=_process_sac_batch,
            _run_sac_eval_log_checkpoint=_run_sac_eval_log_checkpoint,
        )
        _finalize_sac_run(
            config,
            env,
            modules,
            training,
            state,
            runtime,
            start_time,
            t,
            _checkpoint_payload=_checkpoint_payload,
        )
        return TrainResult(
            best_return=float(state.best_return),
            last_eval_return=float(state.last_eval_return),
            last_heldout_return=state.last_heldout_return,
            num_steps=int(config.collector.total_frames),
        )


def register() -> None:
    registry_module = importlib.import_module("rl.registry")
    SACConfig = importlib.import_module("rl.torchrl.sac.config").SACConfig
    registry_module.register_algo("sac", SACConfig, train_sac)
