import re

import pytest
import torch

from common.collector import Collector
from uhd.opt_turbo import TurboObservation, UHDBOConfig, _trim_history, optimize_turbo
from uhd.tm_sphere import TMSphere
from uhd.uhd_collector import UHDCollector


def make_config(s: TMSphere, eps: float, num_candidates: int) -> UHDBOConfig:
    def embedder_fn(params: torch.Tensor) -> torch.Tensor:
        x = params.index_select(0, s.active_idx) - s.x_0
        return -torch.sum(x * x)

    class _Metric:
        def measure(self, controller: TMSphere) -> float:
            return float(controller().detach().cpu().item())

    class _Embedder:
        def embed(self, params: torch.Tensor) -> torch.Tensor:
            return embedder_fn(params)

    class _Perturber:
        def __init__(self, eps: float) -> None:
            self.eps = float(eps)
            self._backup = None

        def perturb(self, current_params: torch.Tensor, ys) -> torch.Tensor:
            self._backup = current_params.clone()
            current_params.add_(self.eps * torch.randn_like(current_params))
            current_params.clamp_(0.0, 1.0)
            return current_params

        def unperturb(self, target: torch.Tensor) -> None:
            assert self._backup is not None
            target.copy_(self._backup)
            self._backup = None

        def incorporate(self, target: torch.Tensor) -> None:
            self._backup = None

    class _Selector:
        def select(self, embeddings):
            vals = torch.tensor([float(e.detach().cpu().item()) for e in embeddings])
            return int(torch.argmax(vals).item())

    return UHDBOConfig(
        num_candidates=num_candidates,
        perturber=_Perturber(eps),
        embedder=_Embedder(),
        selector=_Selector(),
        metric=_Metric(),
    )


def test_uhd_bo_improves_or_equal_and_clips_and_traces():
    s0 = TMSphere(10, 3, seed=1234)
    c0 = Collector()
    y0 = float(s0().detach().cpu().item())
    lines0 = list(c0)
    assert lines0 == []

    s1 = TMSphere(10, 3, seed=1234)
    base_c1 = Collector()
    c1 = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c1)
    torch.manual_seed(1234)
    conf = make_config(s1, eps=0.1, num_candidates=7)
    y1 = optimize_turbo(s1, c1, num_rounds=50, config=conf)
    lines1 = list(base_c1)
    traces = [ln for ln in lines1 if ln.startswith("TRACE: ")]
    assert len(traces) == 50
    assert lines1[-1] == "DONE"
    assert "name = tm_sphere" in traces[0] and "opt_name = uhd_bo" in traces[0]
    assert isinstance(y1, float)
    assert y1 >= y0
    assert torch.all(s1.parameters_.data >= 0) and torch.all(s1.parameters_.data <= 1)
    assert s1.parameters_.numel() == 10


def test_uhd_bo_deterministic_with_manual_seed():
    s1 = TMSphere(12, 4, seed=999)
    base_c1 = Collector()
    c1 = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c1)
    conf1 = make_config(s1, eps=0.05, num_candidates=5)
    torch.manual_seed(777)
    y1 = optimize_turbo(s1, c1, num_rounds=25, config=conf1)
    lines1 = list(base_c1)

    s2 = TMSphere(12, 4, seed=999)
    base_c2 = Collector()
    c2 = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c2)
    conf2 = make_config(s2, eps=0.05, num_candidates=5)
    torch.manual_seed(777)
    y2 = optimize_turbo(s2, c2, num_rounds=25, config=conf2)
    lines2 = list(base_c2)

    assert isinstance(y1, float) and isinstance(y2, float)
    assert y1 == y2
    assert torch.allclose(s1.parameters_.data, s2.parameters_.data)

    def extract_iters_and_returns(lines):
        out = []
        for ln in lines:
            m = re.search(r"i_iter = (\d+).*y_best = ([-0-9.e+]+)$", ln)
            assert m is not None, ln
            out.append((int(m.group(1)), float(m.group(2))))
        return out

    t1 = [ln for ln in lines1 if ln.startswith("TRACE: ")]
    t2 = [ln for ln in lines2 if ln.startswith("TRACE: ")]
    assert extract_iters_and_returns(t1) == extract_iters_and_returns(t2)


def test_trim_history_trims_oldest_non_best_and_preserves_best():
    o0 = TurboObservation(y=1.0, embedding=torch.tensor([1.0]))
    best = TurboObservation(y=5.0, embedding=torch.tensor([5.0]))
    o2 = TurboObservation(y=2.0, embedding=torch.tensor([2.0]))
    o3 = TurboObservation(y=3.0, embedding=torch.tensor([3.0]))
    o4 = TurboObservation(y=4.0, embedding=torch.tensor([4.0]))
    history = [o0, best, o2, o3, o4]
    _trim_history(history, best, 3)
    assert len(history) == 3
    assert history[0] is best
    assert history[1:] == [o3, o4]


def test_trim_history_limit_one_keeps_best_only():
    o0 = TurboObservation(y=1.0, embedding=torch.tensor([1.0]))
    o1 = TurboObservation(y=2.0, embedding=torch.tensor([2.0]))
    best = TurboObservation(y=3.0, embedding=torch.tensor([3.0]))
    history = [o0, o1, best]
    _trim_history(history, best, 1)
    assert history == [best]


def test_trim_history_no_change_when_within_limit():
    o0 = TurboObservation(y=1.0, embedding=torch.tensor([1.0]))
    best = TurboObservation(y=2.0, embedding=torch.tensor([2.0]))
    o2 = TurboObservation(y=0.5, embedding=torch.tensor([0.5]))
    history = [o0, best, o2]
    _trim_history(history, best, 4)
    assert history == [o0, best, o2]


@pytest.mark.parametrize("num_rounds,num_candidates", [(-1, 3), (10, 0)])
def test_uhd_bo_invalid_args(num_rounds, num_candidates):
    s = TMSphere(5, 1, seed=0)
    base_c = Collector()
    c = UHDCollector(name="tm_sphere", opt_name="uhd_bo", collector=base_c)
    conf = make_config(s, eps=0.1, num_candidates=max(1, num_candidates))
    if num_candidates <= 0:

        class _Metric2:
            def measure(self, controller: TMSphere) -> float:
                return float(controller().detach().cpu().item())

        class _Selector2:
            def select(self, embeddings):
                vals = torch.tensor([float(e.detach().cpu().item()) for e in embeddings])
                return int(torch.argmax(vals).item())

        conf = UHDBOConfig(
            num_candidates=num_candidates,
            perturber=conf.perturber,
            embedder=conf.embedder,
            selector=_Selector2(),
            metric=_Metric2(),
        )
    with pytest.raises(AssertionError):
        optimize_turbo(s, c, num_rounds=num_rounds, config=conf)
