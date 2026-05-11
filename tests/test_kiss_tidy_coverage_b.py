from __future__ import annotations

import sys
from contextlib import contextmanager
from types import SimpleNamespace
from unittest.mock import Mock, patch

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


def test_kiss_tidy_b_batch_preps_and_timing(tmp_path):
    from experiments.batch_preps_rl_sweeps import prep_sweep_k_bw, prep_sweep_p_bw
    from experiments.batch_preps_timing import prep_timing_sweep

    rd = str(tmp_path / "r")
    assert prep_sweep_k_bw(rd)
    assert prep_sweep_p_bw(rd)
    cfgs = prep_timing_sweep(rd)
    assert cfgs and all(getattr(c, "max_proposal_seconds", None) is not None for c in cfgs)


def test_kiss_tidy_b_dispatch_post_mk_shim_sample_util(monkeypatch, tmp_path):
    import experiments.experiment_sampler_dispatch as disp
    import experiments.experiment_sampler_jobs as jobs
    import experiments.experiment_sampler_sampling as samp
    import experiments.experiment_sampler_shim as sh
    import experiments.experiment_util as eu

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

    fake_m = SimpleNamespace(
        data_is_done=lambda *a, **k: False,
        data_writer=_dw,
        ensure_parent=lambda p: eu.ensure_parent(p),
        build_problem=lambda *a, **k: SimpleNamespace(
            env=SimpleNamespace(env_name="f:sphere-2d", problem_seed=0),
            build_policy=lambda: object(),
            policy_tag="pure-function",
        ),
        mk_replicates=lambda *a, **k: [],
        post_process=disp.post_process,
        sample_1=lambda *a, **k: None,
        seed_all=lambda *a, **k: logs.append("seed"),
        torch=torch,
        mp=__import__("multiprocessing"),
    )
    monkeypatch.setitem(sys.modules, "experiments.experiment_sampler", fake_m)
    assert sh.data_is_done("x") is False
    with sh.data_writer(str(tmp_path / "z.txt")) as wf:
        wf.write("a")
    sh.ensure_parent(str(tmp_path / "d/e.txt"))
    sh.build_problem("f:sphere-2d", "pure-function")
    sh.mk_replicates(object())
    sh.post_process([], [], str(tmp_path / "trace"))
    sh.sample_1(object())
    sh.seed_all(0)
    assert sh.torch_module() is torch
    sh.mp_module()

    cfg = jobs.prep_args_1(str(tmp_path), "exp", "f:sphere-2d", "random", 1, 1, 1, num_denoise=1)
    rcs = jobs.mk_replicates(cfg)
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
    sr = samp.sample_1(rc)
    assert sr.stop_reason == "completed"
    eu.ensure_parent(str(tmp_path / "nested" / "f.txt"))
    rec = [TraceRecord(0, 0.1, 0.1, 0.5, env_name="e", opt_name="o")]
    disp.post_process(
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
    mb.batches("work", tag="t", num=0)
    mb.batches("submit-missing", batch_tag="prep_timing_sweep", tag="t")
    mb.batches("status", tag="t")
    mb.batches("collect", tag="t")
    mb.batches("clean_up", tag="t")
    mb.batches("stop", tag="t")
    mbi.stop("x")
    assert st.last == "x"


def test_kiss_tidy_b_synthetic_modal_impl_and_reps(tmp_path, monkeypatch):
    from modal_timing_sweep_test_support import FakeResultsDict, make_func_spawn_map

    import experiments.modal_synthetic_sine_benchmark_batches_impl as ssb
    import experiments.modal_synthetic_sine_benchmark_batches_reps as reps

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
    ssb.status("tg")
    ssb.clean_up("tg")
    ssb.stop("tg")
    ssb.batches("tg", "status")
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


def test_kiss_tidy_b_ops_cli_batches_uhd(monkeypatch, tmp_path):
    import subprocess

    import ops.exp_uhd_cli as ecli
    import ops.exp_uhd_full as efull
    import ops.exp_uhd_run as erun
    import ops.modal_batches as omb
    import ops.modal_uhd as muhd
    import ops.modal_uhd_runner_impl as mrun_impl
    import ops.synthetic_sine_benchmark_batches as ssbo
    import ops.uhd_batch_cli as ubc

    assert callable(mrun_impl._build_image)

    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(returncode=0))
    for mod in (omb, ssbo):
        monkeypatch.setattr(mod.sys, "exit", lambda *a, **k: None)
    r = _runner()
    for cmd, args in (
        (omb.cli, ["deploy", "t"]),
        (omb.cli, ["submit", "t", "prep_timing_sweep"]),
        (omb.cli, ["submit-force", "t", "prep_timing_sweep"]),
        (omb.cli, ["collect", "t"]),
        (omb.cli, ["status", "t"]),
        (omb.cli, ["stop", "t"]),
        (omb.cli, ["clean-up", "t"]),
        (ssbo.cli, ["deploy", "t"]),
        (ssbo.cli, ["submit", "t", "example_sphere_n12_d2_seed0"]),
        (ssbo.cli, ["collect", "t"]),
        (ssbo.cli, ["status", "t"]),
        (ssbo.cli, ["stop", "t"]),
    ):
        assert r.invoke(cmd, args).exit_code == 0

    import analysis.fitting_time.evaluate as evm

    monkeypatch.setattr(evm, "benchmark_single_surrogate_with_data", lambda **k: (1.0, 2.0, 3.0))
    outd = tmp_path / "ssb"
    assert (
        r.invoke(
            ssbo.cli,
            [
                "local-single",
                "8",
                "sphere",
                "0",
                "gp",
                "--output-dir",
                str(outd),
                "--num-reps",
                "1",
            ],
        ).exit_code
        == 0
    )

    tom = tmp_path / "u.toml"
    tom.write_text(
        '[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\noptimizer = "mezo"\n'
        'lr = 0.01\nnum_dim_target = 2\nnum_module_target = 1\npolicy_tag = "pure-function"\n'
        "problem_seed = 0\nnoise_seed_0 = 0\nbatch_size = 4\nlog_interval = 1\n"
        "accuracy_interval = 1000\n"
    )

    with patch("optimizer.uhd_loop.UHDLoop", lambda *a, **k: SimpleNamespace(run=lambda: None)):
        erun.run_parsed_uhd_local(
            SimpleNamespace(
                optimizer="mezo",
                env_tag="mnist",
                num_rounds=1,
                lr=0.01,
                num_dim_target=2,
                num_module_target=1,
                policy_tag="pure-function",
                problem_seed=0,
                noise_seed_0=0,
                batch_size=4,
                log_interval=1,
                accuracy_interval=1000,
                target_accuracy=None,
                early_reject=None,
                enn=None,
            )
        )

    monkeypatch.setattr("ops.uhd_setup_simple_gym.run_simple_loop", lambda *a, **k: None)
    monkeypatch.setattr("ops.uhd_setup_bszo.run_bszo_loop", lambda *a, **k: None)
    with patch("optimizer.uhd_loop.UHDLoop", lambda *a, **k: SimpleNamespace(run=lambda: None)):
        erun.run_parsed_uhd_local(
            SimpleNamespace(
                optimizer="simple",
                env_tag="mnist",
                num_rounds=1,
                num_dim_target=2,
                policy_tag="pure-function",
                problem_seed=0,
                noise_seed_0=0,
                batch_size=4,
                log_interval=1,
                accuracy_interval=1000,
                target_accuracy=None,
                be=None,
            )
        )
    erun.run_parsed_uhd_local(
        SimpleNamespace(
            optimizer="bszo",
            env_tag="mnist",
            num_rounds=1,
            lr=0.01,
            policy_tag="pure-function",
            problem_seed=0,
            noise_seed_0=0,
            batch_size=4,
            log_interval=1,
            accuracy_interval=1000,
            target_accuracy=None,
            bszo_k=2,
            bszo_epsilon=1e-4,
            bszo_sigma_p_sq=1.0,
            bszo_sigma_e_sq=1.0,
            bszo_alpha=0.1,
        )
    )

    def _fake_im(name):
        if name == "ops.exp_uhd_parse":
            return SimpleNamespace(
                _load_toml_config=lambda p: {
                    "uhd": {
                        "env_tag": "mnist",
                        "num_rounds": 1,
                        "optimizer": "mezo",
                        "lr": 0.01,
                        "num_dim_target": 2,
                        "num_module_target": 1,
                        "policy_tag": "pure-function",
                        "problem_seed": 0,
                        "noise_seed_0": 0,
                        "batch_size": 4,
                        "log_interval": 1,
                        "accuracy_interval": 1000,
                    }
                },
                _validate_required=lambda c: None,
                _parse_cfg=lambda c: SimpleNamespace(
                    env_tag="mnist",
                    num_rounds=1,
                    lr=0.01,
                    num_dim_target=2,
                    num_module_target=1,
                    policy_tag="pure-function",
                    problem_seed=0,
                    noise_seed_0=0,
                    log_interval=1,
                    accuracy_interval=1000,
                    target_accuracy=None,
                    early_reject=None,
                    enn=None,
                ),
            )
        if name == "tomllib":
            import tomllib as tl

            return tl
        if name == "ops.modal_uhd":
            return SimpleNamespace(run=lambda *a, **k: "modal-log")
        if name == "ops.exp_uhd_run":
            return erun
        raise AssertionError(name)

    monkeypatch.setattr("common.im.im", _fake_im)
    assert r.invoke(ecli.cli, ["modal", str(tom)]).exit_code == 0
    assert "ok" in erun.uhd_config_toml_to_modal_log(
        str(tom),
        "A100",
        exp_uhd_parse=_fake_im("ops.exp_uhd_parse"),
        tomllib=__import__("tomllib"),
        modal_run=lambda *a, **k: "ok",
    )

    monkeypatch.setattr(mrun_impl, "run", lambda *a, **k: "MR")
    assert muhd.run("mnist", 1, 0.01, 2, 1, gpu="cpu", problem_seed=0, noise_seed_0=0) == "MR"

    monkeypatch.setattr("ops.modal_uhd.run", lambda *a, **k: "full-ok")
    efull.modal_cmd(str(tom), None, "A100")

    monkeypatch.setattr(
        "ops.uhd_batch_cli._load_toml",
        lambda p: {"env_tag": "mnist", "num_rounds": 1, "optimizer": "mezo"},
    )
    monkeypatch.setattr("ops.uhd_batch_cli._batch_modal", lambda *a, **k: None)
    monkeypatch.setattr("ops.uhd_batch_cli._collect", lambda *a, **k: None)
    t2 = tmp_path / "b.toml"
    t2.write_text('[uhd]\nenv_tag = "mnist"\nnum_rounds = 1\n')
    assert r.invoke(ubc.cli, ["modal", str(t2), "--num-reps", "1"]).exit_code == 0
    assert r.invoke(ubc.cli, ["collect"]).exit_code == 0
