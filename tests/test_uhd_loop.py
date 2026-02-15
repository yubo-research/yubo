import torch.nn as nn

from optimizer.uhd_loop import UHDLoop


def test_run_prints_eval(capsys):
    module = nn.Linear(4, 2)
    returns = [(1.0, 0.1), (0.5, 0.2), (2.0, 0.05)]
    call_count = [0]

    def evaluate_fn(eval_seed):
        r = returns[call_count[0]]
        call_count[0] += 1
        return r

    loop = UHDLoop(module, evaluate_fn, num_iterations=3)
    loop.run()

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert len(lines) == 4
    assert "num_params = 10" in lines[0]
    assert "i_iter = 0" in lines[1]
    assert "mu =" in lines[1]
    assert "se =" in lines[1]
    assert "y_best =" in lines[1]
    assert "i_iter = 1" in lines[2]
    assert "i_iter = 2" in lines[3]


def test_run_num_iterations():
    module = nn.Linear(4, 2)
    call_count = [0]

    def evaluate_fn(eval_seed):
        call_count[0] += 1
        return float(call_count[0]), 0.0

    loop = UHDLoop(module, evaluate_fn, num_iterations=5)
    loop.run()

    assert call_count[0] == 5
