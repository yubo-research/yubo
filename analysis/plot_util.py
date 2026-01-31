import json
from pathlib import Path


def normalize_results_and_exp_dir(results_path: str, exp_dir: str) -> tuple[str, str]:
    """
    Normalize (results_path, exp_dir) to what DataLocator expects.

    DataLocator assumes root = Path(results_path) / exp_dir.
    This helper makes common caller variants work:
    - results_path is repo root (contains a "results/" subdir)
    - exp_dir is "results/<exp>" while results_path already points at ".../results"
    - exp_dir is an absolute/relative path to the experiment directory
    - results_path itself points at the experiment directory
    """
    rp = Path(results_path).expanduser()
    ed = Path(exp_dir).expanduser()

    # Case: results_path is repo root and has results/<exp_dir>
    if (rp / "results" / exp_dir).exists():
        return str(rp / "results"), exp_dir

    # Case: exp_dir includes "results/<exp>" while results_path already ends with ".../results"
    if rp.name == "results" and exp_dir.startswith("results/"):
        exp_dir = exp_dir[len("results/") :]
        return str(rp), exp_dir

    # Case: exp_dir is a path to the experiment directory
    if ed.exists() and ed.is_dir():
        return str(ed.parent), ed.name

    # Case: results_path points directly at the experiment directory
    if rp.exists() and rp.is_dir() and rp.name == exp_dir:
        return str(rp.parent), rp.name

    return str(rp), exp_dir


def uniq_int(vals: list) -> int | None:
    """Extract unique integer from a list, return None if zero or more than one."""
    xs = {v for v in vals if isinstance(v, int)}
    if len(xs) == 1:
        return next(iter(xs))
    return None


def collect_config_rows(
    root: Path,
    opt_names: list[str],
    *,
    include_opt_name: bool = True,
) -> list[dict]:
    """Collect configuration rows from experiment subdirectories."""
    rows: list[dict] = []
    for child in root.iterdir() if root.exists() else []:
        if not child.is_dir():
            continue
        cfg = child / "config.json"
        if not cfg.exists():
            continue
        try:
            with open(cfg) as f:
                c = json.load(f)
        except Exception:
            continue

        env_tag = c.get("env_tag") or c.get("env")
        opt = c.get("opt_name")
        if not isinstance(env_tag, str) or not isinstance(opt, str):
            continue
        if opt not in opt_names:
            continue

        row = {
            "env_tag": env_tag,
            "num_arms": c.get("num_arms"),
            "num_rounds": c.get("num_rounds"),
            "num_reps": c.get("num_reps"),
        }
        if include_opt_name:
            row["opt_name"] = opt
        rows.append(row)

    return rows
