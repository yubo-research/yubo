def test_prep_cmd():
    from experiments.bat_optimal_init_figures import prep_cmd

    cmd = prep_cmd(
        exp_dir="test_exp",
        problem="f:ackley-3d",
        opt="random",
        num_arms=5,
        num_replications=10,
        num_rounds=3,
    )
    assert isinstance(cmd, str)
    assert "f:ackley-3d" in cmd
    assert "random" in cmd


def test_prep_cmds():
    from experiments.bat_optimal_init_figures import prep_cmds

    cmds = prep_cmds(
        exp_dir="test_exp",
        funcs=["ackley", "sphere"],
        dims=[3],
        num_arms=5,
        num_replications=10,
        opts=["random", "sobol"],
        noises=[None],
    )
    assert isinstance(cmds, list)
    assert len(cmds) > 0
