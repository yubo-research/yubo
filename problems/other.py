def make(env_name, problem_seed):
    if env_name == "xgb":
        from problems.xgb_hp import XGBHP

        return XGBHP(problem_seed)
    assert False, ("Unknown env_name", env_name)
