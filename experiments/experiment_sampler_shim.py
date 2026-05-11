"""Forwarders to :mod:`experiments.experiment_sampler` for patch-friendly indirection (no import cycle)."""

from __future__ import annotations

import sys
from typing import Any


def _m() -> Any:
    return sys.modules["experiments.experiment_sampler"]


def data_is_done(*args: Any, **kwargs: Any) -> Any:
    return _m().data_is_done(*args, **kwargs)


def data_writer(*args: Any, **kwargs: Any) -> Any:
    return _m().data_writer(*args, **kwargs)


def ensure_parent(*args: Any, **kwargs: Any) -> Any:
    return _m().ensure_parent(*args, **kwargs)


def build_problem(*args: Any, **kwargs: Any) -> Any:
    return _m().build_problem(*args, **kwargs)


def mk_replicates(*args: Any, **kwargs: Any) -> Any:
    return _m().mk_replicates(*args, **kwargs)


def post_process(*args: Any, **kwargs: Any) -> Any:
    return _m().post_process(*args, **kwargs)


def sample_1(*args: Any, **kwargs: Any) -> Any:
    return _m().sample_1(*args, **kwargs)


def seed_all(*args: Any, **kwargs: Any) -> Any:
    return _m().seed_all(*args, **kwargs)


def torch_module() -> Any:
    return _m().torch


def mp_module() -> Any:
    return _m().mp
