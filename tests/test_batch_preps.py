import tempfile
from unittest.mock import patch

import experiments.batch_preps as _bp


def _assert_policy_tag_present(cmds):
    assert isinstance(cmds, list)
    assert cmds, "Expected non-empty prep output."
    assert all(cmd.policy_tag is not None for cmd in cmds)


def test_prep_mtv_repro():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_mtv_repro(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_ts_hd():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_ts_hd(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_sweep_q():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_q(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_cum_time_obs():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_cum_time_obs(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_seq():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_seq(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_sweep_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_k(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_push():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_push(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_tlunar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_tlunar(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_hop():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_hop(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_bw():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_bw(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_leukemia():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_leukemia(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_dna():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_dna(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_rl_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_rl_one(tmpdir, name="test_env")
        _assert_policy_tag_present(cmds)


def test_prep_ant():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_ant(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_human():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_human(tmpdir)
        _assert_policy_tag_present(cmds)


@patch("experiments.experiment_sampler.build_problem")
@patch("experiments.experiment_sampler.data_is_done")
def test_prep_mtv_repro_configs_build_with_explicit_policy_tag(mock_data_is_done, mock_build_problem):
    from experiments.experiment_sampler import mk_replicates

    mock_data_is_done.return_value = False
    mock_build_problem.return_value = object()
    with tempfile.TemporaryDirectory() as tmpdir:
        cfg = _bp.prep_mtv_repro(tmpdir)[0]
        mk_replicates(cfg)
    assert mock_build_problem.called
    for call in mock_build_problem.call_args_list:
        assert call.args[1] is not None
