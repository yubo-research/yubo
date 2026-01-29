import resource
import sys
import time

import torch

from common.collector import Collector
from uhd.mk_turbo_config import mk_turbo_config
from uhd.opt_turbo import optimize_turbo
from uhd.tm_sphere import TMSphere
from uhd.uhd_collector import UHDCollector


def test_optimize_uhd_bo_large_problem_timings():
    num_rounds = 1000
    num_dim = 300000
    num_active = 10
    torch.manual_seed(0)
    s = TMSphere(num_dim, num_active, seed=123)
    base_c = Collector()
    c = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c)
    conf = mk_turbo_config(use_tr=False, num_raasp=20)
    r_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    y = optimize_turbo(s, c, num_rounds=num_rounds, config=conf)
    t1 = time.perf_counter()
    r_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lines = list(base_c)
    traces = [ln for ln in lines if ln.startswith("TRACE: ")]
    assert len(traces) == num_rounds
    assert lines[-1] == "DONE"
    assert isinstance(y, float)
    elapsed = t1 - t0
    # Ensure it runs within a reasonable bound on typical CI/CPU
    print(f"ELAPSED_OPT_UHD_BO: elapsed_sec = {elapsed:.2f}")
    assert elapsed < 20.0
    denom = 1024 * 1024 if sys.platform == "darwin" else 1024
    before_mb = r_before / denom
    after_mb = r_after / denom
    delta_mb = max(0.0, after_mb - before_mb)
    print(
        f"PEAK_MEM_OPT_UHD_BO: ru_maxrss_before_MB = {before_mb:.2f} ru_maxrss_after_MB = {after_mb:.2f} delta_MB = {delta_mb:.2f} MB"
    )


def test_optimize_uhd_bo_tr_raasp_timing():
    num_dim = 100000
    num_active = 100
    num_rounds = 1000
    num_trail = 100
    torch.manual_seed(0)
    s = TMSphere(num_dim, num_active, seed=17)
    base_c = Collector()
    c = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c)
    k = int(0.01 * num_dim)
    conf = mk_turbo_config(use_tr=True, num_raasp=k, num_trail=num_trail)
    r_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    y = optimize_turbo(s, c, num_rounds=num_rounds, config=conf)
    t1 = time.perf_counter()
    r_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lines = list(base_c)
    traces = [ln for ln in lines if ln.startswith("TRACE: ")]
    assert len(traces) == num_rounds
    assert lines[-1] == "DONE"
    assert isinstance(y, float)
    elapsed = t1 - t0
    print(f"ELAPSED_OPT_UHD_BO: elapsed_sec = {elapsed:.2f}")
    assert elapsed < 60.0
    denom = 1024 * 1024 if sys.platform == "darwin" else 1024
    before_mb = r_before / denom
    after_mb = r_after / denom
    delta_mb = max(0.0, after_mb - before_mb)
    print(
        f"PEAK_MEM_OPT_UHD_BO: ru_maxrss_before_MB = {before_mb:.2f} ru_maxrss_after_MB = {after_mb:.2f} delta_MB = {delta_mb:.2f} MB"
    )


def test_optimize_uhd_bo_raasp_no_tr_timing():
    num_dim = 100000
    num_active = 100
    num_rounds = 1000
    num_trail = 100
    torch.manual_seed(0)
    s = TMSphere(num_dim, num_active, seed=17)
    base_c = Collector()
    c = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c)
    k = int(0.01 * num_dim)
    conf = mk_turbo_config(use_tr=False, num_raasp=k, num_trail=num_trail)
    r_before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    t0 = time.perf_counter()
    y = optimize_turbo(s, c, num_rounds=num_rounds, config=conf)
    t1 = time.perf_counter()
    r_after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    lines = list(base_c)
    traces = [ln for ln in lines if ln.startswith("TRACE: ")]
    assert len(traces) == num_rounds
    assert lines[-1] == "DONE"
    assert isinstance(y, float)
    elapsed = t1 - t0
    print(f"ELAPSED_OPT_UHD_BO: elapsed_sec = {elapsed:.2f}")
    assert elapsed < 30.0
    denom = 1024 * 1024 if sys.platform == "darwin" else 1024
    before_mb = r_before / denom
    after_mb = r_after / denom
    delta_mb = max(0.0, after_mb - before_mb)
    print(
        f"PEAK_MEM_OPT_UHD_BO: ru_maxrss_before_MB = {before_mb:.2f} ru_maxrss_after_MB = {after_mb:.2f} delta_MB = {delta_mb:.2f} MB"
    )
