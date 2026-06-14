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


def test_prep_uhd_batch_cheetah_simple():
    from experiments.uhd_batch_preps import prep_uhd_batch_cheetah_simple

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_cheetah_simple(tmpdir, num_reps=5)
        assert len(configs) == 2
        assert {cfg["policy_tag"] for cfg, _ in configs} == {"linear", "mlp-32-16"}
        for cfg, num_reps in configs:
            assert num_reps == 5
            assert cfg["env_tag"] == "cheetah:fn"
            assert cfg["num_rounds"] == 1000
            assert cfg["optimizer"] == "simple"
            assert cfg["perturb"] == "dense"
            assert cfg["sigma"] == 0.1
            assert not any(k.startswith("be_") for k in cfg)


def test_prep_uhd_batch_cheetah_simple_parses():
    from experiments.uhd_batch_preps import prep_uhd_batch_cheetah_simple
    from ops.exp_uhd_parse import _parse_cfg

    with tempfile.TemporaryDirectory() as tmpdir:
        configs = prep_uhd_batch_cheetah_simple(tmpdir)
        for cfg, _ in configs:
            parsed = _parse_cfg(cfg)
            assert parsed.env_tag == "cheetah:fn"
            assert parsed.optimizer == "simple"


def test_cheetah_fn_is_not_uhd_vector_objective():
    from problems.uhd_obj import supports_uhd_vector_objective

    assert supports_uhd_vector_objective("cheetah:fn") is False
