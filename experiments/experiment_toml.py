#!/usr/bin/env python

import sys
from dataclasses import fields

from common.config_toml import apply_overrides, load_toml, parse_value
from experiments.experiment_sampler import ExperimentConfig, sampler, scan_local, scan_parallel


def _parse_set_args(argv):
    overrides = {}
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--set":
            if i + 1 >= len(argv):
                raise ValueError("Expected KEY=VALUE after --set")
            kv = argv[i + 1]
            if "=" not in kv:
                raise ValueError(f"Invalid --set value: {kv}")
            k, v = kv.split("=", 1)
            overrides[k] = parse_value(v)
            i += 2
            continue
        if arg.startswith("--set="):
            kv = arg[len("--set=") :]
            if "=" not in kv:
                raise ValueError(f"Invalid --set value: {kv}")
            k, v = kv.split("=", 1)
            overrides[k] = parse_value(v)
            i += 1
            continue
        raise ValueError(f"Unknown argument: {arg}")
    return overrides


def _extract_experiment_cfg(cfg):
    if "experiment" in cfg and isinstance(cfg["experiment"], dict):
        return cfg["experiment"]
    return cfg


def _filter_experiment_keys(cfg):
    valid = {f.name for f in fields(ExperimentConfig)}
    return {k: v for k, v in cfg.items() if k in valid}


def _extract_optimizer_cfg(cfg):
    opt_cfg = cfg.get("optimizer")
    if isinstance(opt_cfg, dict):
        return opt_cfg
    return None


def _stringify_opt_value(value):
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, str):
        return value
    raise TypeError(f"Unsupported optimizer option type: {type(value).__name__}")


def _flatten_opt_params(params: dict) -> list[tuple[str, str]]:
    flattened = []
    for k, v in params.items():
        if not isinstance(k, str):
            raise TypeError(f"Optimizer option key must be str, got {type(k).__name__}")
        flattened.append((k, _stringify_opt_value(v)))
    return flattened


def _build_opt_name(base: str, params: dict, general: dict) -> str:
    if not base:
        raise ValueError("optimizer.name/opt_name is required when [optimizer] is provided")
    parts = [base]
    opts = _flatten_opt_params(general) + _flatten_opt_params(params)
    for k, v in opts:
        parts.append(f"{k}={v}")
    if len(parts) == 1:
        return base
    return "/".join(parts)


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]
    if "--config" not in argv:
        raise SystemExit("Usage: experiment_toml.py --config path/to/config.toml [--set key=val ...]")
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        raise SystemExit("Missing path after --config")
    config_path = argv[idx + 1]
    rest = argv[idx + 2 :]

    cfg = load_toml(config_path)
    overrides = _parse_set_args(rest)
    if overrides:
        apply_overrides(cfg, overrides)

    exp_cfg = _extract_experiment_cfg(cfg)
    opt_cfg = _extract_optimizer_cfg(cfg)
    if opt_cfg is not None:
        opt_name = opt_cfg.get("opt_name") or opt_cfg.get("name") or exp_cfg.get("opt_name")
        params = opt_cfg.get("params", {}) or {}
        general = opt_cfg.get("general", {}) or {}
        if not isinstance(params, dict):
            raise TypeError("optimizer.params must be a table/dict")
        if not isinstance(general, dict):
            raise TypeError("optimizer.general must be a table/dict")
        exp_cfg = dict(exp_cfg)
        exp_cfg["opt_name"] = _build_opt_name(opt_name, params, general)
    exp_cfg = _filter_experiment_keys(exp_cfg)
    config = ExperimentConfig.from_dict(exp_cfg)
    if config.run_workers and int(config.run_workers) > 1:

        def _scan(run_configs, max_total_seconds=None):
            scan_parallel(
                run_configs,
                max_total_seconds=max_total_seconds,
                max_workers=int(config.run_workers),
            )

        sampler(config, distributor_fn=_scan)
    else:
        sampler(config, distributor_fn=scan_local)


if __name__ == "__main__":
    main()
