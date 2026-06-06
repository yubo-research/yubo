import inspect

import torch
import torch.nn as nn

from ops.uhd_setup_monolith_make_loop import make_loop
from optimizer.uhd_driver import UHDDriver


def test_run_prints_eval(capsys):
    module = nn.Linear(4, 2)
    returns = [(1.0, 0.1), (0.5, 0.2), (2.0, 0.05)]
    call_count = [0]

    def evaluate_fn(eval_seed):
        r = returns[call_count[0]]
        call_count[0] += 1
        return r

    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_mezo import UHDMeZO

    perturbator = GaussianPerturbator(module)
    uhd = UHDMeZO(perturbator, dim=10, lr_scheduler=ConstantLR(0.001), sigma=0.001)
    loop = UHDDriver(
        module,
        uhd,
        perturbator,
        evaluate_fn,
        optimizer="mezo",
        num_iterations=3,
    )
    loop.run()

    captured = capsys.readouterr()
    lines = captured.out.strip().split("\n")
    assert len(lines) == 4
    assert "num_params = 10" in lines[0]
    assert "optimizer = mezo" in lines[0]
    assert "i_iter = 0" in lines[1]
    assert "proposal_dt =" in lines[1]
    assert "eval_dt =" in lines[1]
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

    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_mezo import UHDMeZO

    perturbator = GaussianPerturbator(module)
    uhd = UHDMeZO(perturbator, dim=10, lr_scheduler=ConstantLR(0.001), sigma=0.001)
    loop = UHDDriver(
        module,
        uhd,
        perturbator,
        evaluate_fn,
        optimizer="mezo",
        num_iterations=5,
    )
    loop.run()

    assert call_count[0] == 5


def test_log_param_stats_zero_numel_does_not_divide_by_zero(capsys):
    module = nn.Module()
    module.p = nn.Parameter(torch.empty(0))

    def evaluate_fn(eval_seed):
        return 0.0, 0.0

    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_mezo import UHDMeZO

    perturbator = GaussianPerturbator(module)
    uhd = UHDMeZO(perturbator, dim=0, lr_scheduler=ConstantLR(0.001), sigma=0.001)
    loop = UHDDriver(
        module,
        uhd,
        perturbator,
        evaluate_fn,
        optimizer="mezo",
        num_iterations=1,
        log_param_stats=True,
    )
    loop.run()

    out = capsys.readouterr().out
    assert "num_params = 0" in out
    assert "mean_param" not in out


def test_mezo_and_mezo_be_share_make_loop_signature():
    """Configs differing only in optimizer should hit the same driver entrypoint."""
    mezo_sig = inspect.signature(make_loop)
    assert "optimizer" in mezo_sig.parameters
    assert "lr" in mezo_sig.parameters
    assert "sigma" in mezo_sig.parameters
    assert "num_module_target" in mezo_sig.parameters
    assert "be" in mezo_sig.parameters


def test_format_uhd_eval_line_includes_optional_fields():
    from optimizer.uhd_loop_support import format_uhd_eval_line

    line = format_uhd_eval_line(
        i_iter=3,
        proposal_dt=0.1,
        eval_dt=0.2,
        sigma=0.001,
        mu=1.5,
        se=0.05,
        y_best_str="1.5000",
        acc=0.9,
        mean_param=0.01,
        std_param=0.02,
    )
    assert "i_iter = 3" in line
    assert "test_acc = 0.9000" in line
    assert "mean_param" in line


def test_skip_mezo_negative_pair():
    from optimizer.gaussian_perturbator import GaussianPerturbator
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_mezo import UHDMeZO
    from optimizer.uhd_mezo_phase_util import skip_mezo_negative_pair

    module = nn.Linear(2, 1)
    perturbator = GaussianPerturbator(module)
    mezo = UHDMeZO(perturbator, dim=3, lr_scheduler=ConstantLR(0.001))
    mezo.ask()
    mezo.tell(1.0, 0.1)
    seed_before = mezo.eval_seed
    skip_mezo_negative_pair(mezo)
    assert mezo.positive_phase
    assert mezo.eval_seed == seed_before + 1


def test_mezo_be_receives_lr_from_factory():
    from ops.uhd_config import BEConfig
    from ops.uhd_setup_simple_common import _make_simple_optimizer
    from optimizer.gaussian_perturbator import GaussianPerturbator

    module = nn.Linear(2, 1)
    perturbator = GaussianPerturbator(module)
    uhd = _make_simple_optimizer(
        module,
        perturbator,
        optimizer="mezo_be",
        sigma=0.002,
        dim=3,
        lr=0.05,
        embed_module=module,
        be=BEConfig(10, 10, 20, 10, 25, None),
    )
    assert uhd._mezo._lr_scheduler.lr == 0.05
    assert uhd.sigma == 0.002
