def all_benchmarks():
    import problems.benchmark_functions_1 as benchmark_functions_1
    import problems.benchmark_functions_2 as benchmark_functions_2
    import problems.sig_opt as sig_opt

    all_bf = {}

    for mod in [sig_opt]:
        if hasattr(mod, "all_functions"):
            all_bf |= mod.all_functions()
        else:
            all_bf |= _collect(mod)
    return all_bf


def _collect(mod):
    import inspect

    all_bf = {}
    for name in dir(mod):
        obj = getattr(mod, name)

        if inspect.isclass(obj) and Exception not in obj.mro():
            name = name.lower()
            assert name not in all_bf, (name, mod)
            all_bf[name] = obj
    return all_bf
