from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass


@dataclass
class ModalResultParts:
    trace_fn: object
    collector_log: object
    collector_trace: object
    trace_records: object | None
    wall_seconds: float | None
    stop_reason: str | None


def unpack_modal_result_value(value) -> ModalResultParts:
    wall_seconds = None
    stop_reason = None
    if len(value) == 6:
        (trace_fn, collector_log, collector_trace, trace_records, wall_seconds, stop_reason) = value
    elif len(value) == 4:
        (trace_fn, collector_log, collector_trace, trace_records) = value
    else:
        (trace_fn, collector_log, collector_trace) = value
        trace_records = None
    return ModalResultParts(
        trace_fn=trace_fn,
        collector_log=collector_log,
        collector_trace=collector_trace,
        trace_records=trace_records,
        wall_seconds=wall_seconds,
        stop_reason=stop_reason,
    )


def iter_modal_results_for_collect(
    res_dict,
    *,
    post_process,
    data_is_done,
    gotitem_log: Callable[[str, float, float | None, str | None], None],
) -> set[str]:
    print("DICT_SIZE:", res_dict.len())
    collected_keys: set[str] = set()
    print("ITEMS")
    for key, value in res_dict.items():
        if key.endswith("key_max"):
            print("SKIP", key)
            continue
        print("GETITEM", key)
        t_0 = time.time()
        parts = unpack_modal_result_value(value)
        t_f = time.time()
        gotitem_log(key, t_f - t_0, parts.wall_seconds, parts.stop_reason)
        if not data_is_done(parts.trace_fn):
            post_process(
                parts.collector_log,
                parts.collector_trace,
                parts.trace_fn,
                parts.trace_records,
                wall_seconds=parts.wall_seconds,
                stop_reason=parts.stop_reason,
            )
        collected_keys.add(key)
    return collected_keys


def gen_jobs_from_configs(batch_tag: str, configs, mk_replicates, data_is_done, job_key_fn) -> Iterator[tuple[str, object]]:
    for config in configs:
        for run_config in mk_replicates(config):
            if not data_is_done(run_config.trace_fn):
                key = job_key_fn(batch_tag, run_config.trace_fn)
                yield key, run_config
