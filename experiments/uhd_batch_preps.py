from typing import Any


def prep_uhd_batch_tlunar(
    results_dir: str = "results/uhd",
    num_reps: int = 30,
) -> list[tuple[dict[str, Any], int]]:
    _ = results_dir

    optimizers = [
        "simple",
        "simple_be",
        "mezo",
        "mezo_be",
    ]

    configs: list[tuple[dict[str, Any], int]] = []

    for opt in optimizers:
        cfg: dict[str, Any] = {
            "env_tag": "tlunar:fn",
            "num_rounds": 1000,
            "problem_seed": 0,
            "noise_seed_0": 0,
            "optimizer": opt,
            "perturb": "dense",
            "log_interval": 1,
            "accuracy_interval": 100,
            "batch_size": 4096,
        }

        if opt == "simple":
            cfg["lr"] = 0.001
        elif opt == "simple_be":
            cfg["lr"] = 0.001
            cfg["be_num_probes"] = 10
            cfg["be_num_candidates"] = 10
            cfg["be_warmup"] = 20
            cfg["be_fit_interval"] = 10
            cfg["be_enn_k"] = 25
        elif opt == "mezo":
            cfg["lr"] = 0.001
        elif opt == "mezo_be":
            cfg["lr"] = 0.001
            cfg["be_num_probes"] = 10
            cfg["be_num_candidates"] = 10
            cfg["be_warmup"] = 20
            cfg["be_fit_interval"] = 10
            cfg["be_enn_k"] = 25

        configs.append((cfg, num_reps))

    return configs
