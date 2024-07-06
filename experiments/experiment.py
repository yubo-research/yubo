#!/usr/bin/env python


from experiment_sampler import sampler, scan_local

from common.util import parse_kv

if __name__ == "__main__":
    import sys

    d_args = parse_kv(sys.argv[1:])
    reqd_keys = ["exp-dir", "env-tag", "opt-name", "num-arms", "num-rounds", "num-reps"]
    for k in reqd_keys:
        k = "--" + k
        assert k in d_args, f"Missing {k} in {list(d_args.keys())}. Required: {reqd_keys}"

    d_args_cleaned = {}
    for k, v in d_args.items():
        kk = k.replace("--", "")
        kk = kk.replace("-", "_")
        d_args_cleaned[kk] = v

    sampler(d_args_cleaned, distributor_fn=scan_local)
