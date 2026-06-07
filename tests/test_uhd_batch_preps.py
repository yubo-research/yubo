import tempfile


def test_uhd_batch_cheetah():
    from experiments.uhd_batch_preps import prep_uhd_batch_cheetah

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_cheetah(tmpdir, num_reps=5)
        assert isinstance(configs, list)
        assert len(configs) == 7  # simple + 3 drivers × 2 acquisitions

        for cfg, num_reps in configs:
            assert isinstance(cfg, dict)
            assert num_reps == 5
            assert cfg["env_tag"] == "cheetah"
            assert cfg["policy_tag"] == "mlp-32-16"
            assert cfg["num_rounds"] == 10000
            assert cfg["optimizer"] in ["simple", "simple_be"]


def test_prep_uhd_batch_cheetah_default_reps():
    from experiments.uhd_batch_preps import prep_uhd_batch_cheetah

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_cheetah(tmpdir)
        assert all(num_reps == 10 for _, num_reps in configs)


def test_prep_uhd_batch_cheetah_simple_be_params():
    from experiments.uhd_batch_preps import prep_uhd_batch_cheetah

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_cheetah(tmpdir)

        be_cfgs = [cfg for cfg, _ in configs if cfg["optimizer"] == "simple_be"]
        assert len(be_cfgs) == 6
        assert {cfg["be_enn_index_driver"] for cfg in be_cfgs} == {
            "flat",
            "hnsw",
            "hnsw_disk",
        }
        assert {cfg["be_acquisition"] for cfg in be_cfgs} == {"ucb", "mu"}

        for cfg in be_cfgs:
            assert cfg["be_num_probes"] == 10
            assert cfg["be_num_candidates"] == 10
            assert cfg["be_warmup"] == 10
            assert cfg["be_fit_interval"] == 1
            assert cfg["be_enn_k"] == 15


def test_prep_uhd_batch_cheetah_simple_has_no_be_keys():
    from experiments.uhd_batch_preps import prep_uhd_batch_cheetah

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_cheetah(tmpdir)

        simple_cfg = next(cfg for cfg, _ in configs if cfg["optimizer"] == "simple")
        assert not any(k.startswith("be_") for k in simple_cfg)
        assert simple_cfg["sigma"] == 0.1
