from __future__ import annotations

import json
from pathlib import Path

import experiments.modal_enn_incremental_batches_json as add_json


def test_add_result_json_complete_rejects_wrong_meta(tmp_path: Path):
    from analysis.fitting_time.fitting_time_enn_incremental import (
        enn_incremental_checkpoint_ns,
    )

    chk = enn_incremental_checkpoint_ns()
    dest = tmp_path / "enn_incremental_D2_sphere_pseed17_nrep10_rep0_flat.json"
    dest.write_text(
        json.dumps(
            {
                "N": list(chk),
                "add_seconds": [0.01] * len(chk),
                "log_likelihood": [-1.0] * len(chk),
                "_meta": {
                    "D": 2,
                    "function_name": "sphere",
                    "problem_seed": 999,
                    "rep_index": 0,
                    "num_reps": 10,
                    "index_driver": "flat",
                },
            }
        )
    )

    assert not add_json.result_json_complete(
        dest,
        chk,
        d=2,
        function_name="sphere",
        problem_seed=17,
        rep_index=0,
        num_reps=10,
        index_driver="flat",
    )
