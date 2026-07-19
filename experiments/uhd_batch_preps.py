from typing import Any

from experiments.batch_preps_rl_sweeps import CHEETAH_BO_POLICY_ROUNDS


def prep_uhd_batch_cheetah(
    results_dir: str = "results/uhd",
    num_reps: int = 10,
) -> list[tuple[dict[str, Any], int]]:
    _ = results_dir

    common: dict[str, Any] = {
        "env_tag": "cheetah",
        "policy_tag": "mlp-32-16",
        "num_rounds": 10000,
        "problem_seed": 0,
        "noise_seed_0": 0,
        "perturb": "dense",
        "log_interval": 1,
        "sigma": 0.1,
    }

    be_common: dict[str, Any] = {
        "be_num_probes": 10,
        "be_num_candidates": 10,
        "be_warmup": 10,
        "be_fit_interval": 1,
        "be_enn_k": 15,
    }

    drivers = ["flat", "bpann_disk"]
    configs: list[tuple[dict[str, Any], int]] = []

    configs.append(({**common, "optimizer": "simple"}, num_reps))

    for be_enn_index_driver in drivers:
        for be_acquisition in ["ucb", "mu"]:
            configs.append(
                (
                    {
                        **common,
                        "optimizer": "simple_be",
                        **be_common,
                        "be_enn_index_driver": be_enn_index_driver,
                        "be_acquisition": be_acquisition,
                    },
                    num_reps,
                )
            )

    return configs


def prep_uhd_batch_cheetah_simple(
    results_dir: str = "results/uhd",
    num_reps: int = 10,
) -> list[tuple[dict[str, Any], int]]:
    _ = results_dir

    common: dict[str, Any] = {
        "env_tag": "cheetah:fn",
        "problem_seed": 0,
        "noise_seed_0": 0,
        "perturb": "dense",
        "log_interval": 1,
        "sigma": 0.1,
        "optimizer": "simple",
    }

    configs: list[tuple[dict[str, Any], int]] = []
    for policy_tag, num_rounds in CHEETAH_BO_POLICY_ROUNDS.items():
        configs.append(({**common, "policy_tag": policy_tag, "num_rounds": num_rounds}, num_reps))
    return configs
