import tempfile


def test_prep_mtv_repro():
    from experiments.batch_preps import prep_mtv_repro

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_mtv_repro(tmpdir)
        assert isinstance(cmds, list)


def test_prep_ts_hd():
    from experiments.batch_preps import prep_ts_hd

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_ts_hd(tmpdir)
        assert isinstance(cmds, list)


def test_prep_sweep_q():
    from experiments.batch_preps import prep_sweep_q

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_sweep_q(tmpdir)
        assert isinstance(cmds, list)


def test_prep_cum_time_obs():
    from experiments.batch_preps import prep_cum_time_obs

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_cum_time_obs(tmpdir)
        assert isinstance(cmds, list)


def test_prep_seq():
    from experiments.batch_preps import prep_seq

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_seq(tmpdir)
        assert isinstance(cmds, list)


def test_prep_sweep_k():
    from experiments.batch_preps import prep_sweep_k

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_sweep_k(tmpdir)
        assert isinstance(cmds, list)


def test_prep_push():
    from experiments.batch_preps import prep_push

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_push(tmpdir)
        assert isinstance(cmds, list)


def test_prep_tlunar():
    from experiments.batch_preps import prep_tlunar

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_tlunar(tmpdir)
        assert isinstance(cmds, list)


def test_prep_hop():
    from experiments.batch_preps import prep_hop

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_hop(tmpdir)
        assert isinstance(cmds, list)


def test_prep_bw():
    from experiments.batch_preps import prep_bw

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_bw(tmpdir)
        assert isinstance(cmds, list)


def test_prep_leukemia():
    from experiments.batch_preps import prep_leukemia

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_leukemia(tmpdir)
        assert isinstance(cmds, list)


def test_prep_dna():
    from experiments.batch_preps import prep_dna

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_dna(tmpdir)
        assert isinstance(cmds, list)


def test_prep_rl_one():
    from experiments.batch_preps import prep_rl_one

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_rl_one(tmpdir, name="test_env")
        assert isinstance(cmds, list)


def test_prep_ant():
    from experiments.batch_preps import prep_ant

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_ant(tmpdir)
        assert isinstance(cmds, list)


def test_prep_human():
    from experiments.batch_preps import prep_human

    with tempfile.TemporaryDirectory() as tmpdir:
        cmds = prep_human(tmpdir)
        assert isinstance(cmds, list)
