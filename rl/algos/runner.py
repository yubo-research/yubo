from common.config_toml import apply_overrides, load_toml, parse_set_args
from rl.algos.registry import get_algo, resolve_algo_name
from rl.algos.runner_helpers import parse_runtime_args, parse_seeds, seeded_exp_dir, split_config_and_args


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

    resolved_algo = resolve_algo_name(str(algo), backend=backend) if backend is not None else str(algo)
    algo_cfg = root.get(str(algo))
    if algo_cfg is None and resolved_algo != str(algo):
        algo_cfg = root.get(str(resolved_algo))
    if algo_cfg is None:
        algo_cfg = {}
    if not isinstance(algo_cfg, dict):
        raise ValueError(f"Algorithm config for '{algo}' must be a table.")
    return str(algo), backend, algo_cfg


def _extract_run_cfg(cfg: dict) -> tuple[list[int], int]:
    if "rl" not in cfg or not isinstance(cfg["rl"], dict):
        return [], 1
    run_cfg = cfg["rl"].get("run", {})
    if run_cfg is None:
        return [], 1
    if not isinstance(run_cfg, dict):
        raise ValueError("[rl.run] must be a table.")
    seeds_raw = run_cfg.get("seeds")
    workers = int(run_cfg.get("workers", 1))
    if seeds_raw is None:
        return [], workers
    if isinstance(seeds_raw, str):
        return parse_seeds(seeds_raw), workers
    if isinstance(seeds_raw, int):
        return [int(seeds_raw)], workers
    if isinstance(seeds_raw, list):
        return [int(x) for x in seeds_raw], workers
    raise ValueError("[rl.run].seeds must be a list, int, or comma/range string.")


def _run_from_cfg(cfg: dict, seed: int | None = None):
    algo_name, backend, algo_cfg = _extract_algo_cfg(cfg)
    algo = get_algo(algo_name, backend=backend)
    config = algo.config_cls.from_dict(algo_cfg)
    if seed is not None and hasattr(config, "seed"):
        config.seed = int(seed)
    if seed is not None and hasattr(config, "exp_dir"):
        config.exp_dir = seeded_exp_dir(str(config.exp_dir), int(seed))
    return algo.train_fn(config)


def _run_single(path: str, overrides: dict | None, seed: int | None):
    from rl.algos import builtins

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

    from rl.algos import builtins

    builtins.register_all()
    runtime = parse_runtime_args(rest)

    overrides = parse_set_args(runtime.cleaned)
    cfg = load_toml(config_path)
    if overrides:
        apply_overrides(cfg, overrides)
    seeds, cfg_workers = _extract_run_cfg(cfg)
    if runtime.seeds_raw is not None:
        seeds = parse_seeds(runtime.seeds_raw)
    workers = runtime.workers if runtime.workers_cli_set else cfg_workers
    overrides_keys = sorted(overrides) if overrides else []
    print(
        f"[rl] config={config_path} seeds={seeds if seeds else ['default']} workers={workers} overrides={overrides_keys}",
        flush=True,
    )
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
