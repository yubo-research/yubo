def test_load_all_funcs():
    from experiments.func_names import funcs_bowl, funcs_multimodal, funcs_other, funcs_plate, funcs_ridges, funcs_valley

    for func_list in [
        funcs_multimodal,
        funcs_bowl,
        funcs_plate,
        funcs_valley,
        funcs_ridges,
        funcs_other,
    ]:
        for func in func_list:
            try:
                from problems.benchmark_functions import all_benchmarks

                assert func in all_benchmarks()
            except ImportError:
                raise ImportError(f"Function {func} not found in all_benchmarks()") from None
