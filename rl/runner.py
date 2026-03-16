from __future__ import annotations

import dataclasses

import rl.config_model_defaults as model_defaults
from common.config_toml import apply_overrides, load_toml, parse_set_args
from rl.env_provider import register_get_env_conf
from rl.registry import get_algo
from rl.runner_helpers import parse_runtime_args, seeded_exp_dir, split_config_and_args

_EXPERIMENT_META_KEYS = {"opt_name", "backend", "num_reps", "local_workers"}
_KEY_ALIASES = {"runtime_device": "device", "num_arms": "num_envs"}


def _apply_rl_defaults(opt_name: str, data: dict) -> dict:
    key = str(opt_name).strip().lower()
    apply_defaults = getattr(model_defaults, f"apply_{key}_env_model_defaults", None)
    if apply_defaults is None:
        raise ValueError(f"Unsupported opt_name '{opt_name}'.")
    return apply_defaults(data)


def _extract_experiment_cfg(cfg: dict) -> dict:
    if "rl" in cfg:
        raise ValueError("Legacy [rl] schema is removed. Use [experiment] with opt_name/env_tag.")
    section = cfg.get("experiment", cfg)
    if not isinstance(section, dict):
        raise ValueError("Config must be a table at root or under [experiment].")
    opt_name = str(section.get("opt_name", "")).strip().lower()
    get_algo(opt_name)
    env_tag = section.get("env_tag")
    if env_tag is None or str(env_tag).strip() == "":
        raise ValueError("experiment.env_tag must be a non-empty string.")
    return section


def _apply_runner_overrides(cfg: dict, overrides: dict) -> dict:
    if not overrides:
        return cfg
    if "experiment" not in cfg or not isinstance(cfg.get("experiment"), dict):
        return apply_overrides(cfg, overrides)
    routed = {}
    for key, value in overrides.items():
        k = str(key)
        routed[k if "." in k else f"experiment.{k}"] = value
    return apply_overrides(cfg, routed)


def _algo_input_dict(*, exp_cfg: dict, config_cls: type) -> dict:
    allowed = {f.name for f in dataclasses.fields(config_cls)}
    out: dict = {}
    for key, value in exp_cfg.items():
        mapped = _KEY_ALIASES.get(str(key), str(key))
        if mapped in _EXPERIMENT_META_KEYS:
            continue
        if mapped in allowed:
            out[mapped] = value
    return out


def _run_from_cfg(cfg: dict, seed: int | None = None):
    exp_cfg = _extract_experiment_cfg(cfg)
    opt_name = str(exp_cfg["opt_name"]).strip().lower()
    algo = get_algo(opt_name)
    algo_input = _algo_input_dict(exp_cfg=exp_cfg, config_cls=algo.config_cls)
    algo_input = _apply_rl_defaults(opt_name, algo_input)
    config = algo.config_cls.from_dict(algo_input)
    if seed is not None and hasattr(config, "seed"):
        config.seed = int(seed)
    if seed is not None and hasattr(config, "exp_dir"):
        config.exp_dir = seeded_exp_dir(str(config.exp_dir), int(seed))
    return algo.train_fn(config)


def _run_single(path: str, overrides: dict | None, seed: int | None):
    from problems.env_conf import get_env_conf
    from rl import builtins

    builtins.register_all()
    register_get_env_conf(get_env_conf)
    overrides_keys = sorted(overrides) if overrides else []
    print(f"[rl] start config={path} seed={seed} overrides={overrides_keys}", flush=True)
    cfg = load_toml(path)
    _apply_runner_overrides(cfg, overrides or {})
    return _run_from_cfg(cfg, seed=seed)


def main(argv: list[str] | None = None):
    import sys

    if not sys.stdout.isatty() and hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if not sys.stderr.isatty() and hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)
    if argv is None:
        argv = sys.argv[1:]
    config_path, rest = split_config_and_args(argv)
    from problems.env_conf import get_env_conf
    from rl import builtins

    builtins.register_all()
    register_get_env_conf(get_env_conf)
    runtime = parse_runtime_args(rest)
    overrides = parse_set_args(runtime.cleaned)
    cfg = load_toml(config_path)
    _apply_runner_overrides(cfg, overrides)
    exp_cfg = _extract_experiment_cfg(cfg)
    num_reps = int(exp_cfg.get("num_reps", 1))
    if num_reps < 1:
        raise ValueError("experiment.num_reps must be >= 1.")
    seeds = list(range(num_reps))
    cfg_workers = int(exp_cfg.get("local_workers", 1))
    workers = runtime.workers if runtime.workers_cli_set else cfg_workers
    overrides_keys = sorted(overrides) if overrides else []
    print(
        f"[rl] config={config_path} seeds={seeds} workers={workers} overrides={overrides_keys}",
        flush=True,
    )
    if workers < 1:
        raise SystemExit("--workers must be >= 1")
    if workers == 1 or len(seeds) == 1:
        for seed in seeds:
            _run_single(config_path, overrides, seed)
        return
    from concurrent.futures import ProcessPoolExecutor, as_completed

    results = {}
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(_run_single, config_path, overrides, seed): seed for seed in seeds}
        for fut in as_completed(futures):
            seed = futures[fut]
            results[seed] = fut.result()
    print(f"[rl] completed seeds={sorted(results)}", flush=True)


if __name__ == "__main__":
    import sys

    print(
        "RL runs via experiments.experiment. Use: python -m experiments.experiment local <config>",
        file=sys.stderr,
    )
    sys.exit(1)
