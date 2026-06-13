from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock

import torch
from click.testing import CliRunner

from analysis.data_io import TraceRecord
from analysis.fitting_time.evaluate import (
    SURROGATE_BENCHMARK_KEYS,
    BMResult,
    MuSe,
    SyntheticSineSurrogateBenchmark,
)


def _runner():
    return CliRunner()


def test_experiment_sampler_shim_lazy_imports_missing_sampler(monkeypatch):
    import experiments.experiment_sampler_shim as sh

    original = sys.modules.pop("experiments.experiment_sampler", None)
    fake_sampler = SimpleNamespace(torch=object())
    calls = []

    def fake_import_module(name):
        calls.append(name)
        return fake_sampler

    monkeypatch.setattr(sh, "import_module", fake_import_module)
    try:
        assert sh._m() is fake_sampler
    finally:
        if original is not None:
            sys.modules["experiments.experiment_sampler"] = original

    assert calls == ["experiments.experiment_sampler"]


def test_kiss_tidy_b_batch_preps_and_timing(tmp_path):
    from experiments.batch_preps_rl_sweeps import prep_sweep_k_bw, prep_sweep_p_bw
    from experiments.batch_preps_timing import prep_timing_sweep

    rd = str(tmp_path / "r")
    assert prep_sweep_k_bw(rd)
    assert prep_sweep_p_bw(rd)
    cfgs = prep_timing_sweep(rd)
    assert cfgs and all(getattr(c, "max_proposal_seconds", None) is not None for c in cfgs)


def test_kiss_tidy_b_dispatch_post_mk_shim_sample_util(monkeypatch, tmp_path):
    import experiments.experiment_sampler_jobs as jobs
    import experiments.experiment_sampler_shim as ess
    from experiments.experiment_sampler_dispatch import post_process
    from experiments.experiment_sampler_jobs import mk_replicates
    from experiments.experiment_sampler_sampling import sample_1
    from experiments.experiment_sampler_shim import (
        build_problem,
        data_is_done,
        data_writer,
    )
    from experiments.experiment_sampler_shim import (
        ensure_parent as shim_ensure_parent,
    )
    from experiments.experiment_sampler_shim import (
        mk_replicates as shim_mk_replicates,
    )
    from experiments.experiment_sampler_shim import (
        post_process as shim_post_process,
    )
    from experiments.experiment_sampler_shim import (
        sample_1 as shim_sample_1,
    )
    from experiments.experiment_util import ensure_parent

    logs = []

    @contextmanager
    def _dw(path):
        p = tmp_path / "out.txt"
        p.parent.mkdir(parents=True, exist_ok=True)
        f = open(p, "w")
        try:
            yield f
        finally:
            f.close()

    def _shim_seed_all(*_a, **_k):
        logs.append("seed")

    fake_m = SimpleNamespace(
        data_is_done=lambda *a, **k: False,
        data_writer=_dw,
        ensure_parent=ensure_parent,
        build_problem=lambda *a, **k: SimpleNamespace(
            env=SimpleNamespace(env_name="f:sphere-2d", problem_seed=0),
            build_policy=lambda: object(),
            policy_tag="pure-function",
        ),
        mk_replicates=lambda *a, **k: [],
        torch=torch,
        mp=__import__("multiprocessing"),
    )
    setattr(fake_m, "post_process", post_process)
    setattr(fake_m, "sample_1", lambda *a, **k: None)
    setattr(fake_m, "seed_all", _shim_seed_all)
    monkeypatch.setitem(sys.modules, "experiments.experiment_sampler", fake_m)
    assert data_is_done("x") is False
    with data_writer(str(tmp_path / "z.txt")) as wf:
        wf.write("a")
    shim_ensure_parent(str(tmp_path / "d/e.txt"))
    build_problem("f:sphere-2d", "pure-function")
    shim_mk_replicates(object())
    shim_post_process([], [], str(tmp_path / "trace"))
    shim_sample_1(object())

    assert ess.torch_module() is torch
    ess.mp_module()

    cfg = jobs.prep_args_1(str(tmp_path), "exp", "f:sphere-2d", "random", 1, 1, 1, num_denoise=1)
    rcs = mk_replicates(cfg)
    assert isinstance(rcs, list)

    te = SimpleNamespace(dt_prop=0.1, dt_eval=0.1, rreturn=0.0, env_steps_iter=0, env_steps_total=0)

    def _gen():
        yield te

    def _mk_opt():
        o = Mock(
            _i_iter=1,
            _cum_dt_proposing=0.0,
            _t_0=0.0,
            r_best_est=0.0,
            best_policy=None,
        )
        o.collect_trace = lambda **kwargs: _gen()
        return o

    def _fake_load(parts, name):
        if parts == ("optimizer", "optimizer") and name == "Optimizer":
            return lambda *a, **k: _mk_opt()
        raise AssertionError((parts, name))

    import experiments.experiment_sampler_sampling as samp

    monkeypatch.setattr(samp, "_load_attr", _fake_load)
    from experiments.experiment_sampler_types import RunConfig

    env = SimpleNamespace(env_name="f:sphere-2d", problem_seed=0)
    prob = SimpleNamespace(env=env, build_policy=lambda: object(), policy_tag="pure-function")
    tr = str(tmp_path / "traces" / "00000")
    rc = RunConfig(
        opt_name="random",
        num_rounds=1,
        num_arms=1,
        num_denoise=1,
        num_denoise_passive=None,
        max_proposal_seconds=None,
        b_trace=False,
        trace_fn=tr,
        problem=prob,
        runtime_device="cpu",
        bo_console=False,
    )
    sr = sample_1(rc)
    assert sr.stop_reason == "completed"
    ensure_parent(str(tmp_path / "nested" / "f.txt"))
    rec = [TraceRecord(0, 0.1, 0.1, 0.5, env_name="e", opt_name="o")]
    post_process(
        ["l1"],
        ["t1"],
        str(tmp_path / "trace2"),
        rec,
        wall_seconds=1.0,
        stop_reason="completed",
    )


def test_kiss_tidy_b_modal_batches_pkg_and_impl(monkeypatch):
    import experiments.modal_batches as mb
    import experiments.modal_batches_impl as mbi
    from experiments.modal_batches import batches

    st = SimpleNamespace(last=None)

    def _set_stop(tag):
        st.last = tag

    monkeypatch.setattr(mb, "status", lambda: None)
    monkeypatch.setattr(mb, "collect", lambda: None)
    monkeypatch.setattr(mb, "clean_up", lambda *a, **k: None)
    monkeypatch.setattr(mb, "stop", _set_stop)
    monkeypatch.setattr(mbi, "clean_up", lambda tag: None)
    monkeypatch.setattr(mbi, "stop", _set_stop)
    monkeypatch.setattr(
        mb,
        "modal",
        SimpleNamespace(Function=SimpleNamespace(lookup=lambda *a, **k: None)),
    )
    monkeypatch.setattr(mb, "batches_submitter", lambda *a, **k: None)
    batches("work", tag="t", num=0)
    batches("submit-missing", batch_tag="prep_timing_sweep", tag="t")
    batches("status", tag="t")
    batches("collect", tag="t")
    batches("clean_up", tag="t")
    batches("stop", tag="t")
    mbi.stop("x")
    assert st.last == "x"


def test_kiss_tidy_b_synthetic_modal_impl_and_reps(tmp_path, monkeypatch):
    from modal_timing_sweep_test_support import FakeResultsDict, make_func_spawn_map

    import experiments.modal_synthetic_sine_benchmark_batches_impl as ssb
    import experiments.modal_synthetic_sine_benchmark_batches_reps as reps
    from experiments.modal_synthetic_sine_benchmark_batches_impl import (
        batches as ssb_batches,
    )
    from experiments.modal_synthetic_sine_benchmark_batches_impl import (
        clean_up as ssb_clean_up,
    )
    from experiments.modal_synthetic_sine_benchmark_batches_impl import (
        status as ssb_status,
    )
    from experiments.modal_synthetic_sine_benchmark_batches_impl import (
        stop as ssb_stop,
    )

    monkeypatch.setattr(ssb, "_results_dict", lambda t: FakeResultsDict())
    monkeypatch.setattr(ssb, "_submitted_dict", lambda t: FakeResultsDict())
    monkeypatch.setattr(
        ssb.modal,
        "Function",
        SimpleNamespace(from_name=lambda *a, **k: make_func_spawn_map([])),
    )
    monkeypatch.setattr(ssb.modal.Dict.objects, "delete", lambda *a, **k: None)
    cap = []
    monkeypatch.setattr(ssb, "_submit_missing", lambda *a, **k: cap.append(a))
    monkeypatch.setattr(ssb, "_collect", lambda *a, **k: cap.append(a))
    ssb.synthetic_sine_benchmark_batch_resubmitter.get_raw_f()([("k1", (10, 2, "sine", 0, 0, 1, list(SURROGATE_BENCHMARK_KEYS)[0]))], "tg")
    ssb.synthetic_sine_benchmark_batch_deleter.get_raw_f()(["k0"], "tg")
    ssb_status("tg")
    ssb_clean_up("tg")
    ssb_stop("tg")
    ssb_batches("tg", "status")
    ssb.batches(
        "tg",
        "submit",
        jobs_fn="example_sphere_n12_d2_seed0",
        output_dir=str(tmp_path),
        num_reps=1,
    )
    ssb.batches("tg", "collect", output_dir=str(tmp_path))
    assert cap

    od = tmp_path / "o"
    od.mkdir()
    k = reps.job_key(n=12, d=2, function_name="sphere", problem_seed=0, num_reps=1)
    assert "sphere" in k or "N12" in k
    p1 = reps.rep_json_dest(od, n=12, d=2, function_name="sphere", problem_seed=0, rep_index=0)
    p1.parent.mkdir(parents=True, exist_ok=True)
    p1.write_text("{}")
    assert reps.existing_rep_payload_path(od, n=12, d=2, function_name="sphere", problem_seed=0, rep_index=0)
    assert reps.legacy_single_rep_dest(od, n=12, d=2, function_name="sphere", problem_seed=0) == reps.benchmark_json_dest(
        od, n=12, d=2, function_name="sphere", problem_seed=0, num_reps=1
    )
    assert reps.aggregate_surrogate_results_to_rep(od, n=12, d=2, function_name="sphere", problem_seed=0, rep_index=0) is None
    assert reps.aggregate_reps_to_dest(od, n=12, d=2, function_name="sphere", problem_seed=0, num_reps=2) is None
    od_empty = tmp_path / "ej"
    od_empty.mkdir()
    jobs = list(reps.iter_missing_jobs("example_sphere_n12_d2_seed0", od_empty, 1))
    assert jobs


def test_kiss_tidy_b_wide_row_core():
    from experiments.synthetic_sine_benchmark_payload_core import (
        synthetic_surrogate_benchmark_to_wide_row,
    )

    bench = SyntheticSineSurrogateBenchmark(results={k: BMResult(MuSe(1.0, 0.1), MuSe(2.0, 0.1), MuSe(3.0, 0.1)) for k in SURROGATE_BENCHMARK_KEYS})
    row = synthetic_surrogate_benchmark_to_wide_row(bench)
    assert any(k.endswith("_mu") for k in row)
