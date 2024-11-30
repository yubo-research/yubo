import fnmatch
import os

import numpy as np

from common.util import parse_kv


class DataLocator:
    def __init__(self, results_path, exp_dir, num_arms=None, num_rounds=None, num_reps=None, num_dim=None, opt_names=None):
        self.results = results_path
        self.exp_dir = exp_dir
        self.num_arms = num_arms
        self.num_rounds = num_rounds
        self.num_reps = num_reps
        self.num_dim = num_dim
        self._opt_names = opt_names

    def __str__(self):
        return self._root_path()

    def _root_path(self):
        return f"{self.results}/{self.exp_dir}"

    def problems(self):
        # env=f:sphere-1d--opt_name=mtv-pss--num_arms=3--num_rounds=3--num_reps=100
        return sorted({d[0]["env"] for d in self._load()})

    def _load(self, problem=None, opt_name=None):
        if opt_name is not None:
            opt_names = set([opt_name])
        elif self._opt_names is not None:
            opt_names = set(self._opt_names)
        else:
            opt_names = None
        root_path = self._root_path()
        data_sets = []
        for fn in os.listdir(root_path):
            if "--" not in fn:
                continue
            d = parse_kv(fn.split("--"))
            if problem is not None and not fnmatch.fnmatch(d["env"], problem):
                continue
            if opt_names is not None and d["opt_name"] not in opt_names:
                continue
            if self.num_arms is not None and int(d["num_arms"]) != self.num_arms:
                continue
            if self.num_rounds is not None and int(d["num_rounds"]) != self.num_rounds:
                continue
            if self.num_reps is not None and int(d["num_reps"]) != self.num_reps:
                continue
            if self.num_dim is not None:
                x = d["env"].split("-")
                if len(x) == 2 and x[1][-1] == "d":
                    num_dim = int(x[1][:-1])
                    if num_dim != self.num_dim:
                        continue

            data_sets.append((d, f"{root_path}/{fn}"))

        return data_sets

    def optimizers(self):
        return sorted({d[0]["opt_name"] for d in self._load()})

    def optimizers_in(self, problem):
        return sorted({d[0]["opt_name"] for d in self._load(problem=problem)})

    def organize_data(self, opt_names, mu, se):
        # mu and se are ordered by self.optimizers()
        # You want them ordered by opt_names.
        data = dict(zip(self.optimizers(), list(zip(mu, se))))
        data = [data[k] for k in opt_names]
        mu, se = list(zip(*data))
        return np.array(mu), np.array(se)

    def __call__(self, problem, opt_name):
        return sorted({d[1] for d in self._load(problem, opt_name)})
