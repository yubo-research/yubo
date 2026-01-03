import torch
from torch import Tensor, nn

from common.collector import Collector
from uhd.tm_sphere import TMSphere
from uhd.uhd_experiment import ExpParams, run_experiment


def test_run_experiment_new_controller_and_collector_and_seeding():
    num_dim = 8
    num_active = 3
    num_rounds = 0
    num_reps = 5
    base_seed = 123

    seen = []
    collector_ids = []

    def dummy_optimizer(controller: nn.Module, collector: Collector, num_rounds: int) -> Tensor:
        assert isinstance(controller, TMSphere)
        seen.append((float(controller.x_0.item()), controller.active_idx.tolist()))
        collector_ids.append(id(collector))
        return torch.tensor(0.0)

    params = ExpParams(
        num_dim=num_dim,
        num_active=num_active,
        num_rounds=num_rounds,
        num_reps=num_reps,
        seed=base_seed,
        optimizer=dummy_optimizer,
        controller=lambda s: TMSphere(num_dim, num_active, s),
    )

    run_experiment(params, opt_name="adamw")

    assert len(seen) == num_reps
    assert len(set(collector_ids)) == num_reps

    expected = []
    for i in range(num_reps):
        s = base_seed + i
        m = TMSphere(num_dim, num_active, s)
        expected.append((float(m.x_0.item()), m.active_idx.tolist()))
    assert seen == expected

    r1 = torch.rand(())
    torch.manual_seed(base_seed + num_reps - 1)
    r2 = torch.rand(())
    assert torch.allclose(r1, r2)
