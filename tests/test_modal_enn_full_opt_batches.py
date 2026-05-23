from __future__ import annotations

from pathlib import Path

from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
from common.experiment_seeds import problem_seed_from_rep_index
from experiments import modal_enn_full_opt_batches as full_opt_batches
from experiments import modal_enn_full_opt_batches_json as full_opt_json


def test_full_opt_job_key_and_dest():
    dest = full_opt_batches.full_opt_result_json_dest(
        "results/enn_incremental",
        env_tag="f:ackley-10d",
        problem_seed=18,
        rep_index=0,
        num_reps=10,
        index_driver=EnnIncrementalIndexDriver.FLAT,
    )
    assert dest.name == "enn_full_opt_f_ackley-10d_pseed18_nrep10_rep0_flat.json"


def test_iter_full_opt_jobs_yields_expected_job(tmp_path: Path):
    import experiments.modal_enn_incremental_batches_common as common

    jobs = list(
        common.iter_full_opt_jobs(
            tmp_path,
            "flat",
            num_reps=1,
            d=10,
            problem_seed=17,
        )
    )
    assert len(jobs) == 4
    key, job = jobs[0]
    env_tag, ps, ri, nr, drv = job
    assert env_tag == "f:sphere-10d"
    assert ps == problem_seed_from_rep_index(0)
    assert ri == 0
    assert nr == 1
    assert drv == "flat"
    assert key.startswith("enn_full_opt_")


def test_full_opt_result_json_complete_roundtrip(tmp_path: Path):
    dest = full_opt_batches.full_opt_result_json_dest(
        tmp_path,
        env_tag="f:ackley-10d",
        problem_seed=19,
        rep_index=1,
        num_reps=10,
        index_driver="hnsw",
    )
    payload = {
        "N": [1, 3],
        "proposal_elapsed_seconds": [0.01, 0.02],
        "_meta": {
            "env_tag": "f:ackley-10d",
            "opt_name": "turbo-enn-fit-ucb/idx=hnsw",
            "index_driver": "hnsw",
            "policy_tag": "pure-function",
            "problem_seed": 19,
            "rep_index": 1,
            "num_reps": 10,
            "num_arms": 1,
            "num_denoise": 1,
            "num_rounds": 100_000,
            "stop_reason": "completed",
        },
    }
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(__import__("json").dumps(payload))
    assert full_opt_json.full_opt_result_json_complete(
        dest,
        (1, 3),
        env_tag="f:ackley-10d",
        problem_seed=19,
        rep_index=1,
        num_reps=10,
        index_driver="hnsw",
        opt_name="turbo-enn-fit-ucb/idx=hnsw",
    )
