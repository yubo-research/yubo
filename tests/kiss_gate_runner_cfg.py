def runner_dummy_config_cls():
    def from_dict(cls, _d):
        return cls()

    return type(
        "_Cfg",
        (),
        {
            "seed": 7,
            "exp_dir": "tmp",
            "problem_seed": None,
            "noise_seed_0": None,
            "from_dict": classmethod(from_dict),
        },
    )
