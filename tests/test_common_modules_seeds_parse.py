class TestExperimentSeeds:
    def test_problem_seed_from_rep_index(self):
        from common.experiment_seeds import problem_seed_from_rep_index

        assert problem_seed_from_rep_index(0) == 18
        assert problem_seed_from_rep_index(1) == 19
        assert problem_seed_from_rep_index(2) == 20

    def test_noise_seed_0_from_problem_seed(self):
        from common.experiment_seeds import noise_seed_0_from_problem_seed

        assert noise_seed_0_from_problem_seed(18) == 180
        assert noise_seed_0_from_problem_seed(19) == 190

    def test_global_seed_for_run(self):
        from common.experiment_seeds import global_seed_for_run

        assert global_seed_for_run(18) == 45
        assert global_seed_for_run(42) == 69

    def test_rl_seed_alignment(self):
        from common.experiment_seeds import problem_seed_from_rep_index, resolve_problem_seed

        for rl_seed, bo_i_rep in [(0, 0), (1, 1), (2, 2)]:
            assert resolve_problem_seed(seed=rl_seed, problem_seed=None) == problem_seed_from_rep_index(bo_i_rep)

    def test_resolve_noise_seed_0(self):
        from common.experiment_seeds import resolve_noise_seed_0

        assert resolve_noise_seed_0(problem_seed=18, noise_seed_0=None) == 180
        assert resolve_noise_seed_0(problem_seed=18, noise_seed_0=42) == 42


class TestParseKv:
    def test_parse_kv_simple(self):
        from common.util import parse_kv

        result = parse_kv(["a=1", "b=2"])
        assert result == {"a": "1", "b": "2"}

    def test_parse_kv_empty(self):
        from common.util import parse_kv

        result = parse_kv([])
        assert result == {}

    def test_parse_kv_with_dashes(self):
        from common.util import parse_kv

        result = parse_kv(["--opt-name=random", "--num-arms=5"])
        assert result == {"--opt-name": "random", "--num-arms": "5"}
