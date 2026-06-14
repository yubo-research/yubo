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


def test_prep_sweep_p():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_p(tmpdir)
        assert isinstance(cmds, list)


def test_prep_sweep_k_tlunar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_k_tlunar(tmpdir)
        assert isinstance(cmds, list)


def test_prep_sweep_p_tlunar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_p_tlunar(tmpdir)
        assert isinstance(cmds, list)


def test_prep_push():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_push(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_tlunar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_tlunar(tmpdir)
        _assert_policy_tag_present(cmds)


def test_prep_validate():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_validate(tmpdir)
        _assert_policy_tag_present(cmds)
        assert len(cmds) == 3
        assert {c.opt_name for c in cmds} == {"random", "turbo-one", "turbo-1"}
        assert all(c.env_tag == "tlunar:fn" for c in cmds)
        assert all(c.policy_tag == "turbo-lunar" for c in cmds)
        assert all(c.num_arms == 50 and c.num_rounds == 30 and c.num_reps == 10 for c in cmds)
        assert all(c.num_denoise == 50 for c in cmds)
        assert all(f"{tmpdir}/validate_turbo" in c.exp_dir for c in cmds)


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


def test_prep_cheetah_policy_optimizer():
    from experiments.batch_preps_rl_sweeps import EGGROLL_OPT

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_cheetah_policy_optimizer(tmpdir)
        assert len(cmds) == 6
        assert {c.policy_tag for c in cmds} == {"linear", "mlp-32-16"}
        assert {c.opt_name for c in cmds} == {
            "random",
            "turbo-enn-fit-ucb",
            EGGROLL_OPT,
        }
        assert all(c.env_tag == "cheetah:fn" for c in cmds)
        assert all(c.num_arms == 10 and c.num_rounds == 1000 and c.num_reps == 10 for c in cmds)
        assert all(c.num_denoise == 1 for c in cmds)
        assert all(c.num_denoise_passive is None for c in cmds)
        assert all("exp_cheetah_policy_optimizer" in c.exp_dir for c in cmds)


def test_prep_cheetah_eggroll_s01():
    from experiments.batch_preps_rl_sweeps import EGGROLL_OPT_S01

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_cheetah_eggroll_s01(tmpdir)
        assert len(cmds) == 2
        assert {c.policy_tag for c in cmds} == {"linear", "mlp-32-16"}
        assert all(c.opt_name == EGGROLL_OPT_S01 for c in cmds)
        assert "sigma=0.1" in EGGROLL_OPT_S01
        assert all(c.env_tag == "cheetah:fn" for c in cmds)
        assert all(c.num_arms == 10 and c.num_rounds == 1000 and c.num_reps == 10 for c in cmds)
        assert all(c.num_denoise == 1 for c in cmds)
        assert all(c.num_denoise_passive is None for c in cmds)
        assert all("exp_cheetah_eggroll_s01" in c.exp_dir for c in cmds)
        assert cmds[0].to_dir_name() != cmds[1].to_dir_name()


def test_prep_cheetah_eggroll_s02():
    from experiments.batch_preps_rl_sweeps import EGGROLL_OPT_S02

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_cheetah_eggroll_s02(tmpdir)
        assert len(cmds) == 2
        assert all(c.opt_name == EGGROLL_OPT_S02 for c in cmds)
        assert "sigma=0.2" in EGGROLL_OPT_S02
        assert all("exp_cheetah_eggroll_s02" in c.exp_dir for c in cmds)


def test_prep_cheetah_eggroll_s03():
    from experiments.batch_preps_rl_sweeps import EGGROLL_OPT_S03

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_cheetah_eggroll_s03(tmpdir)
        assert len(cmds) == 2
        assert all(c.opt_name == EGGROLL_OPT_S03 for c in cmds)
        assert "sigma=0.3" in EGGROLL_OPT_S03
        assert all("exp_cheetah_eggroll_s03" in c.exp_dir for c in cmds)


def test_prep_run_others():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_run_others(tmpdir)
        assert len(cmds) == 22
        pairs = {(c.opt_name, c.env_tag) for c in cmds}
        assert pairs == _bp._RUN_OTHERS_NONFAIL_CELLS
        assert all("exp_ennbo_run_others" in c.exp_dir for c in cmds)


def test_prep_turbo_abl():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_turbo_abl(tmpdir)
        assert len(cmds) == 20  # 2 opts × 10 envs
        opts = {c.opt_name for c in cmds}
        assert opts == {"turbo-one-nds", "turbo-one-ucb"}
        assert all("exp_ennbo_turbo_abl" in c.exp_dir for c in cmds)


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
