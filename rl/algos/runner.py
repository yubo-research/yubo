from common.config_toml import apply_overrides, load_toml, parse_set_args
from rl.algos.registry import get_algo


class _RuntimeArgs:
    def __init__(self, seeds_raw: str | None, workers: int, workers_cli_set: bool, cleaned: list[str]):
        self.seeds_raw = seeds_raw
        self.workers = workers
        self.workers_cli_set = workers_cli_set
        self.cleaned = cleaned


def _split_config_and_args(argv: list[str]) -> tuple[str, list[str]]:
    if "--config" not in argv:
        raise SystemExit("Usage: runner.py --config path/to/config.toml [--set key=val ...]")
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        raise SystemExit("Missing path after --config")
    return argv[idx + 1], argv[idx + 2 :]


def _parse_runtime_args(argv: list[str]) -> _RuntimeArgs:
    seeds_raw = None
    workers = 1
    workers_cli_set = False
    cleaned = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--seeds":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value after --seeds")
            seeds_raw = argv[i + 1]
            i += 2
            continue
        if arg.startswith("--seeds="):
            seeds_raw = arg.split("=", 1)[1]
            i += 1
            continue
        if arg == "--workers":
            if i + 1 >= len(argv):
                raise SystemExit("Missing value after --workers")
            workers = int(argv[i + 1])
            workers_cli_set = True
            i += 2
            continue
        if arg.startswith("--workers="):
            workers = int(arg.split("=", 1)[1])
            workers_cli_set = True
            i += 1
            continue
        cleaned.append(arg)
        i += 1
    return _RuntimeArgs(seeds_raw=seeds_raw, workers=workers, workers_cli_set=workers_cli_set, cleaned=cleaned)


def _extract_algo_cfg(cfg: dict) -> tuple[str, dict]:
    if "rl" not in cfg or not isinstance(cfg["rl"], dict):
        raise ValueError("Config must contain [rl] table.")
    root = cfg["rl"]
    algo = root.get("algo")
    if not algo:
        raise ValueError('[rl] must set algo (for example "ppo" or "sac").')
    algo_cfg = root.get(str(algo), {})
    if algo_cfg is None:
        algo_cfg = {}
    if not isinstance(algo_cfg, dict):
        raise ValueError(f"[rl.{algo}] must be a table.")
    return str(algo), algo_cfg


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
        return _parse_seeds(seeds_raw), workers
    if isinstance(seeds_raw, int):
        return [int(seeds_raw)], workers
    if isinstance(seeds_raw, list):
        return [int(x) for x in seeds_raw], workers
    raise ValueError("[rl.run].seeds must be a list, int, or comma/range string.")


def _run_from_cfg(cfg: dict, seed: int | None = None):
    algo_name, algo_cfg = _extract_algo_cfg(cfg)
    algo = get_algo(algo_name)
    config = algo.config_cls.from_dict(algo_cfg)
    if seed is not None and hasattr(config, "seed"):
        config.seed = int(seed)
    if seed is not None and hasattr(config, "exp_dir"):
        config.exp_dir = _seeded_exp_dir(str(config.exp_dir), int(seed))
    return algo.train_fn(config)


def _parse_seeds(raw: str | None) -> list[int]:
    if raw is None:
        return []
    raw = raw.strip()
    if not raw:
        return []
    seeds = []
    for token in raw.split(","):
        token = token.strip()
        if not token:
            continue
        if "-" in token:
            lo, hi = token.split("-", 1)
            lo_i = int(lo)
            hi_i = int(hi)
            if hi_i < lo_i:
                raise ValueError(f"Invalid seed range: {token}")
            seeds.extend(range(lo_i, hi_i + 1))
        else:
            seeds.append(int(token))
    return seeds


def _seeded_exp_dir(exp_dir: str, seed: int) -> str:
    from pathlib import Path

    suffix = f"seed_{seed}"
    p = Path(exp_dir)
    if p.name == suffix:
        return str(p)
    return str(p / suffix)


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
    config_path, rest = _split_config_and_args(argv)

    from rl.algos import builtins

    builtins.register_all()
    runtime = _parse_runtime_args(rest)

    overrides = parse_set_args(runtime.cleaned)
    cfg = load_toml(config_path)
    if overrides:
        apply_overrides(cfg, overrides)
    seeds, cfg_workers = _extract_run_cfg(cfg)
    if runtime.seeds_raw is not None:
        seeds = _parse_seeds(runtime.seeds_raw)
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
