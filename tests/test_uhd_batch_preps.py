import tempfile


def test_prep_uhd_batch_tlunar():
    from experiments.uhd_batch_preps import prep_uhd_batch_tlunar

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_tlunar(tmpdir, num_reps=5)
        assert isinstance(configs, list)
        assert len(configs) == 4  # simple, simple_be, mezo, mezo_be

        for cfg, num_reps in configs:
            assert isinstance(cfg, dict)
            assert num_reps == 5
            assert cfg["env_tag"] == "tlunar:fn"
            assert cfg["num_rounds"] == 30
            assert cfg["optimizer"] in ["simple", "simple_be", "mezo", "mezo_be"]


def test_prep_uhd_batch_tlunar_default_reps():
    from experiments.uhd_batch_preps import prep_uhd_batch_tlunar

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_tlunar(tmpdir)
        assert all(num_reps == 30 for _, num_reps in configs)


def test_prep_uhd_batch_tlunar_simple_be_params():
    from experiments.uhd_batch_preps import prep_uhd_batch_tlunar

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_tlunar(tmpdir)

        # Find simple_be config
        simple_be_cfg = None
        for cfg, _ in configs:
            if cfg["optimizer"] == "simple_be":
                simple_be_cfg = cfg
                break

        assert simple_be_cfg is not None
        assert simple_be_cfg["be_num_probes"] == 10
        assert simple_be_cfg["be_num_candidates"] == 10
        assert simple_be_cfg["be_warmup"] == 20
        assert simple_be_cfg["be_fit_interval"] == 10
        assert simple_be_cfg["be_enn_k"] == 25


def test_prep_uhd_batch_tlunar_mezo_params():
    from experiments.uhd_batch_preps import prep_uhd_batch_tlunar

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_tlunar(tmpdir)

        # Find mezo config
        mezo_cfg = None
        for cfg, _ in configs:
            if cfg["optimizer"] == "mezo":
                mezo_cfg = cfg
                break

        assert mezo_cfg is not None
        assert mezo_cfg["lr"] == 0.001
        assert mezo_cfg["batch_size"] == 4096
