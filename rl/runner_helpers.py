from pathlib import Path


class _RuntimeArgs:
    def __init__(self, seeds_raw: str | None, workers: int, workers_cli_set: bool, cleaned: list):
        self.seeds_raw = seeds_raw
        self.workers = workers
        self.workers_cli_set = workers_cli_set
        self.cleaned = cleaned


def parse_runtime_args(argv: list[str]) -> _RuntimeArgs:
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
    return _RuntimeArgs(
        seeds_raw=seeds_raw,
        workers=workers,
        workers_cli_set=workers_cli_set,
        cleaned=cleaned,
    )


def split_config_and_args(argv: list[str]) -> tuple[str, list[str]]:
    if argv and argv[0] == "local":
        argv = argv[1:]
    if "--config" not in argv:
        raise SystemExit("Usage: runner.py [local] --config path/to/config.toml [--set key=val ...]")
    idx = argv.index("--config")
    if idx + 1 >= len(argv):
        raise SystemExit("Missing path after --config")
    return argv[idx + 1], argv[idx + 2 :]


def parse_seeds(raw: str | None) -> list[int]:
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


def seeded_exp_dir(exp_dir: str, seed: int) -> str:
    suffix = f"seed_{seed}"
    p = Path(exp_dir)
    if p.name == suffix:
        return str(p)
    return str(p / suffix)
