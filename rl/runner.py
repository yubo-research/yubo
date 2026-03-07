from common.config_toml import apply_overrides, load_toml, parse_set_args
from rl.registry import get_algo, resolve_algo_name
from rl.runner_helpers import parse_runtime_args, seeded_exp_dir, split_config_and_args


def _extract_algo_cfg(cfg: dict) -> tuple[str, str | None, dict]:
    if "rl" not in cfg or not isinstance(cfg["rl"], dict):
        raise ValueError("Config must contain [rl] table.")
    root = cfg["rl"]
    algo = root.get("algo")
    if not algo:
        raise ValueError('[rl] must set algo (for example "ppo" or "sac").')
    backend_raw = root.get("backend")
    backend = None if backend_raw is None else str(backend_raw).strip()
    if backend == "":
        raise ValueError("[rl].backend cannot be empty when provided.")
    _ = resolve_algo_name(str(algo), backend=backend) if backend is not None else str(algo)
    algo_cfg = root.get(str(algo))
    if algo_cfg is None:
        algo_cfg = {}
    if not isinstance(algo_cfg, dict):
        raise ValueError(f"Algorithm config for '{algo}' must be a table.")
    return (str(algo), backend, algo_cfg)


def _extract_run_cfg(cfg: dict) -> tuple[list[int], int]:
    if "rl" not in cfg or not isinstance(cfg["rl"], dict):
        return ([], 1)
    run_cfg = cfg["rl"].get("run", {})
    if run_cfg is None:
        return ([], 1)
    if not isinstance(run_cfg, dict):
        raise ValueError("[rl.run] must be a table.")
    if "seeds" in run_cfg:
        raise ValueError("[rl.run].seeds is removed. Use [rl.run].num_reps (BO-style replicate indexing).")
    num_reps_raw = run_cfg.get("num_reps")
    workers = int(run_cfg.get("workers", 1))
    if num_reps_raw is None:
        return ([], workers)
    num_reps = int(num_reps_raw)
    if num_reps < 1:
        raise ValueError("[rl.run].num_reps must be >= 1 when provided.")
    return (list(range(num_reps)), workers)


def _run_from_cfg(cfg: dict, seed: int | None = None):
    algo_name, backend, algo_cfg = _extract_algo_cfg(cfg)
    algo = get_algo(algo_name, backend=backend)
    config = algo.config_cls.from_dict(algo_cfg)
    if seed is not None and hasattr(config, "seed"):
        config.seed = int(seed)
    if hasattr(config, "seed") and (hasattr(config, "problem_seed") or hasattr(config, "noise_seed_0")):
        core_env_conf = __import__("rl.core.env_conf", fromlist=["resolve_run_seeds"])
        resolved = core_env_conf.resolve_run_seeds(
            seed=int(getattr(config, "seed")),
            problem_seed=getattr(config, "problem_seed", None),
            noise_seed_0=getattr(config, "noise_seed_0", None),
        )
        if hasattr(config, "problem_seed"):
            config.problem_seed = int(resolved.problem_seed)
        if hasattr(config, "noise_seed_0"):
            config.noise_seed_0 = int(resolved.noise_seed_0)
    if seed is not None and hasattr(config, "exp_dir"):
        config.exp_dir = seeded_exp_dir(str(config.exp_dir), int(seed))
    return algo.train_fn(config)


def _run_single(path: str, overrides: dict | None, seed: int | None):
    from rl import builtins

    builtins.register_all()
    overrides_keys = sorted(overrides) if overrides else []
    print(f"[rl] start config={path} seed={seed} overrides={overrides_keys}", flush=True)
    cfg = load_toml(path)
    if overrides:
        apply_overrides(cfg, overrides)
    return _run_from_cfg(cfg, seed=seed)


def main(argv: list[str] | None = None):
    import sys

    if argv is None:
        argv = sys.argv[1:]
    config_path, rest = split_config_and_args(argv)
    from rl import builtins

    builtins.register_all()
    runtime = parse_runtime_args(rest)
    overrides = parse_set_args(runtime.cleaned)
    cfg = load_toml(config_path)
    if overrides:
        apply_overrides(cfg, overrides)
    seeds, cfg_workers = _extract_run_cfg(cfg)
    workers = runtime.workers if runtime.workers_cli_set else cfg_workers
    overrides_keys = sorted(overrides) if overrides else []
    print(f"[rl] config={config_path} seeds={(seeds if seeds else ['default'])} workers={workers} overrides={overrides_keys}", flush=True)
    if not seeds:
        _run_from_cfg(cfg, seed=None)
        return
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
    main()
