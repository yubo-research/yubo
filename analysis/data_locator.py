import fnmatch
import json
import os
from pathlib import Path

import numpy as np

from common.util import parse_kv


class DataLocator:
    def __init__(
        self,
        results_path,
        exp_dir,
        num_arms=None,
        num_rounds=None,
        num_reps=None,
        num_dim=None,
        opt_names=None,
        problems: set = None,
        problems_exact: bool = False,
        key="return",
        grep_for="TRACE",
    ):
        assert len(opt_names) == len(set(opt_names)), opt_names

        self.results = results_path
        self.exp_dir = exp_dir
        self.num_arms = num_arms
        self.num_rounds = num_rounds
        self.num_reps = num_reps
        self.num_dim = num_dim
        self._opt_names = opt_names
        self._problems = problems
        self._problems_exact = problems_exact
        self.key = key
        self.grep_for = grep_for

    def __str__(self):
        return f"{self._root_path()} {self.exp_dir} num_dim = {self.num_dim} num_arms = {self.num_arms} num_rounds = {self.num_rounds} num_reps = {self.num_reps}"

    def _root_path(self):
        return f"{self.results}/{self.exp_dir}"

    def problems(self):
        problems = set()
        for p in [d[0].get("env") or d[0].get("env_tag") for d in self._load()]:
            if p is None:
                continue
            if self._problems is not None:
                for pp in self._problems:
                    if self._problems_exact:
                        if pp == p:
                            problems.add(p)
                    else:
                        if pp in p:
                            problems.add(p)
            else:
                problems.add(p)
        problems = sorted(problems)

        return problems

    def _load_config_json(self, dir_path):
        config_path = Path(dir_path) / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                return json.load(f)
        return None

    def _parse_entry(self, fn, full_path):
        config = self._load_config_json(full_path)
        if config is not None:
            return {
                "env": config.get("env_tag"),
                "opt_name": config.get("opt_name"),
                "num_arms": config.get("num_arms"),
                "num_rounds": config.get("num_rounds"),
                "num_reps": config.get("num_reps"),
            }
        if "--" in fn:
            return parse_kv(fn.split("--"))
        return None

    def _passes_filters(self, d, problem, opt_names):
        env_key = d.get("env") or d.get("env_tag")
        checks = [
            (problem is not None and env_key, lambda: fnmatch.fnmatch(env_key, problem)),
            (opt_names is not None, lambda: d.get("opt_name") in opt_names),
            (self.num_arms is not None and d.get("num_arms") is not None, lambda: int(d["num_arms"]) == self.num_arms),
            (self.num_rounds is not None and d.get("num_rounds") is not None, lambda: int(d["num_rounds"]) == self.num_rounds),
            (self.num_reps is not None and d.get("num_reps") is not None, lambda: int(d["num_reps"]) == self.num_reps),
            (self.num_dim is not None and env_key, lambda: self._check_dim(env_key)),
        ]
        return all(check_fn() for condition, check_fn in checks if condition)

    def _check_dim(self, env_key):
        x = env_key.split("-")
        if len(x) == 2 and x[1][-1] == "d":
            return int(x[1][:-1]) == self.num_dim
        return True

    def _load(self, problem=None, opt_name=None):
        opt_names = {opt_name} if opt_name is not None else (list(self._opt_names) if self._opt_names is not None else None)
        root_path = self._root_path()
        data_sets = []
        for fn in os.listdir(root_path):
            full_path = f"{root_path}/{fn}"
            d = self._parse_entry(fn, full_path)
            if d is None:
                continue
            if self._passes_filters(d, problem, opt_names):
                data_sets.append((d, full_path))
        return data_sets

    def optimizers(self):
        opt_names = {d[0].get("opt_name") for d in self._load()}
        opt_names.discard(None)
        return [n for n in self._opt_names if n in opt_names]

    def optimizers_in(self, problem):
        return sorted(
            {
                d[0].get("opt_name")
                for d in self._load(problem=problem)
                if d[0].get("opt_name")
            }
        )

    def organize_data(self, opt_names, mu, se):
        data = dict(zip(self.optimizers(), list(zip(mu, se))))
        data = [data[k] for k in opt_names]
        mu, se = list(zip(*data))
        return np.array(mu), np.array(se)

    def __call__(self, problem, opt_name):
        return sorted({d[1] for d in self._load(problem, opt_name)})
