import tempfile

import experiments.batch_preps as _bp


def test_prep_mtv_repro():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_mtv_repro(tmpdir)
        assert isinstance(cmds, list)


def test_prep_ts_hd():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_ts_hd(tmpdir)
        assert isinstance(cmds, list)


def test_prep_sweep_q():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_q(tmpdir)
        assert isinstance(cmds, list)


def test_prep_cum_time_obs():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_cum_time_obs(tmpdir)
        assert isinstance(cmds, list)


def test_prep_seq():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_seq(tmpdir)
        assert isinstance(cmds, list)


def test_prep_sweep_k():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_sweep_k(tmpdir)
        assert isinstance(cmds, list)


def test_prep_push():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_push(tmpdir)
        assert isinstance(cmds, list)


def test_prep_tlunar():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_tlunar(tmpdir)
        assert isinstance(cmds, list)


def test_prep_hop():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_hop(tmpdir)
        assert isinstance(cmds, list)


def test_prep_bw():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_bw(tmpdir)
        assert isinstance(cmds, list)


def test_prep_leukemia():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_leukemia(tmpdir)
        assert isinstance(cmds, list)


def test_prep_dna():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_dna(tmpdir)
        assert isinstance(cmds, list)


def test_prep_rl_one():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_rl_one(tmpdir, name="test_env")
        assert isinstance(cmds, list)


def test_prep_ant():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_ant(tmpdir)
        assert isinstance(cmds, list)


def test_prep_human():
    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = _bp.prep_human(tmpdir)
        assert isinstance(cmds, list)
