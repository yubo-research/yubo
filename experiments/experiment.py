#!/usr/bin/env python


from common.util import parse_kv
from experiments.experiment_sampler import ExperimentConfig, sampler, scan_local

if __name__ == "__main__":
    import sys

    d_args = parse_kv(sys.argv[1:])

    reqd_keys = ["exp-dir", "env-tag", "opt-name", "num-arms", "num-rounds", "num-reps"]
    opt_keys = [
        "num-denoise",
        "num-denoise-passive",
        "max-proposal-seconds",
        "max-total-seconds",
        "b-trace",
    ]
    valid_keys = set("--" + k for k in reqd_keys + opt_keys)

    for k in d_args:
        assert k in valid_keys, f"Unknown argument {k}. Valid: {sorted(valid_keys)}"

    for k in reqd_keys:
        k = "--" + k
        assert k in d_args, (
            f"Missing {k} in {list(d_args.keys())}. Required: {reqd_keys}"
        )

    d_args_cleaned = {}
    for k, v in d_args.items():
        kk = k.replace("--", "")
        kk = kk.replace("-", "_")
        d_args_cleaned[kk] = v

    config = ExperimentConfig.from_dict(d_args_cleaned)
    sampler(config, distributor_fn=scan_local)
