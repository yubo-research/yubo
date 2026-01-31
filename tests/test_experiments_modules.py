class TestFuncNames:
    def test_funcs_nd_is_list(self):
        from experiments.func_names import funcs_nd

        assert isinstance(funcs_nd, list)
        assert len(funcs_nd) > 0
        assert "ackley" in funcs_nd
        assert "sphere" in funcs_nd

    def test_funcs_1d_is_list(self):
        from experiments.func_names import funcs_1d

        assert isinstance(funcs_1d, list)
        assert len(funcs_1d) > 0

    def test_funcs_all_is_list(self):
        from experiments.func_names import funcs_all

        assert isinstance(funcs_all, list)
        assert len(funcs_all) > 0

    def test_func_brief_is_list(self):
        from experiments.func_names import func_brief

        assert isinstance(func_brief, list)
        assert len(func_brief) > 0

    def test_funcs_multimodal_contains_known_functions(self):
        from experiments.func_names import funcs_multimodal

        assert "ackley" in funcs_multimodal
        assert "griewank" in funcs_multimodal
        assert "rastrigin" in funcs_multimodal

    def test_funcs_bowl_contains_known_functions(self):
        from experiments.func_names import funcs_bowl

        assert "sphere" in funcs_bowl

    def test_funcs_valley_contains_known_functions(self):
        from experiments.func_names import funcs_valley

        assert "rosenbrock" in funcs_valley
