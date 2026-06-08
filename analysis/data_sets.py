import os

from .data_sets_config import CACHE_DEBUG
from .data_sets_kv import _load_kv_cached, extract_kv, load, load_kv
from .data_sets_multi import (
    _ensure_float_traces,
    load_multiple_traces,
    range_summarize,
    rank_summarize,
)
from .data_sets_traces import (
    _load_traces_cached,
    _load_traces_jsonl_cached,
    load_traces,
    load_traces_jsonl,
)


def clear_cache():
    _load_kv_cached.cache_clear()
    _load_traces_jsonl_cached.cache_clear()
    _load_traces_cached.cache_clear()


def cache_stats():
    return {
        "load_kv": _load_kv_cached.cache_info(),
        "load_traces_jsonl": _load_traces_jsonl_cached.cache_info(),
        "load_traces": _load_traces_cached.cache_info(),
    }


def problems_in(results_path, exp_tag):
    return sorted(os.listdir(f"{results_path}/{exp_tag}"))


def _optimizers_in(results_path, exp_tag, problem):
    return sorted(os.listdir(f"{results_path}/{exp_tag}/{problem}"))


optimizers_in = _optimizers_in


def all_in(results_path, exp_tag):
    optimizers = set()
    problems = problems_in(results_path, exp_tag)
    for problem in problems:
        optimizers.update(optimizers_in(results_path, exp_tag, problem))
    return problems, sorted(optimizers)


__all__ = [
    "CACHE_DEBUG",
    "_ensure_float_traces",
    "all_in",
    "cache_stats",
    "clear_cache",
    "extract_kv",
    "load",
    "load_kv",
    "load_multiple_traces",
    "load_traces",
    "load_traces_jsonl",
    "optimizers_in",
    "problems_in",
    "rank_summarize",
    "range_summarize",
]
