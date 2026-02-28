import pytest
import torch
from torch import nn

from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.lr_scheduler import ConstantLR
from optimizer.uhd_bszo import UHDBSZO


def _make_bszo(*, k=2, sigma_p_sq=1.0, sigma_e_sq=1.0, alpha=0.1, epsilon=1e-3, lr=0.001):
    module = nn.Linear(3, 2, bias=False)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bszo = UHDBSZO(
        gp,
        dim,
        lr_scheduler=ConstantLR(lr),
        epsilon=epsilon,
        k=k,
        sigma_p_sq=sigma_p_sq,
        sigma_e_sq=sigma_e_sq,
        alpha=alpha,
    )
    return module, gp, bszo


def _run_step(bszo, evaluate_fn):
    """Run one complete BSZO gradient step (k+1 ask/tell calls)."""
    for _ in range(bszo.k + 1):
        bszo.ask()
        mu, se = evaluate_fn()
        bszo.tell(mu, se)


# ── Test 1: Kalman pre-adaptive math (Corollary 3.6) ────────────────


def test_kalman_pre_adaptive_mu_equals_gamma_Y():
    sigma_p_sq, sigma_e_sq = 2.0, 3.0
    gamma = sigma_p_sq / (sigma_p_sq + sigma_e_sq)
    k = 3

    _, _, bszo = _make_bszo(k=k, sigma_p_sq=sigma_p_sq, sigma_e_sq=sigma_e_sq)
    kf = bszo._kf
    kf.init_step()

    Y = [1.5, -2.0, 0.7]
    for i in range(k):
        kf.update_coord(i, Y[i])

    for i in range(k):
        assert abs(kf.mu_post[i] - gamma * Y[i]) < 1e-12


def test_kalman_pre_adaptive_Sigma_equals_gamma_sigma_e_I():
    sigma_p_sq, sigma_e_sq = 2.0, 3.0
    gamma = sigma_p_sq / (sigma_p_sq + sigma_e_sq)
    k = 3

    _, _, bszo = _make_bszo(k=k, sigma_p_sq=sigma_p_sq, sigma_e_sq=sigma_e_sq)
    kf = bszo._kf
    kf.init_step()

    Y = [1.5, -2.0, 0.7]
    for i in range(k):
        kf.update_coord(i, Y[i])

    expected_diag = gamma * sigma_e_sq
    for i in range(k):
        for j in range(k):
            expected = expected_diag if i == j else 0.0
            assert abs(kf.Sigma[i][j] - expected) < 1e-12


# ── Test 2: Kalman post-adaptive ─────────────────────────────────────


def test_adaptive_step_modifies_posterior():
    k = 2
    _, _, bszo = _make_bszo(k=k, sigma_p_sq=1.0, sigma_e_sq=1.0)
    kf = bszo._kf
    kf.init_step()

    Y = [5.0, -3.0]
    for i in range(k):
        kf.update_coord(i, Y[i])
        kf.last_d_idx = i
        kf.last_y = Y[i]
    kf.Y = list(Y)

    mu_pre = list(kf.mu_post)
    kf.adaptive_step()

    changed = any(abs(kf.mu_post[i] - mu_pre[i]) > 1e-12 for i in range(k))
    assert changed


def test_adaptive_step_post_values_numerically():
    """Verify exact post-adaptive μ and Σ values."""
    sigma_p_sq, sigma_e_sq, alpha = 1.0, 1.0, 0.1
    gamma = sigma_p_sq / (sigma_p_sq + sigma_e_sq)
    k = 2

    _, _, bszo = _make_bszo(k=k, sigma_p_sq=sigma_p_sq, sigma_e_sq=sigma_e_sq, alpha=alpha)
    kf = bszo._kf
    kf.init_step()

    Y = [5.0, -3.0]
    for i in range(k):
        kf.update_coord(i, Y[i])
        kf.last_d_idx = i
        kf.last_y = Y[i]
    kf.Y = list(Y)

    r = Y[k - 1] - gamma * Y[k - 1]
    new_sigma_e_sq = (1 - alpha) * sigma_e_sq + alpha * r * r
    diag_pre = gamma * sigma_e_sq

    # Both diags equal → argmax picks j=0
    innov = Y[0] - gamma * Y[0]
    denom = diag_pre + new_sigma_e_sq
    K_j = diag_pre / denom

    expected_mu = [gamma * Y[0] + K_j * innov, gamma * Y[1]]
    expected_S00 = diag_pre - K_j * diag_pre

    kf.adaptive_step()

    assert abs(kf.mu_post[0] - expected_mu[0]) < 1e-12
    assert abs(kf.mu_post[1] - expected_mu[1]) < 1e-12
    assert abs(kf.Sigma[0][0] - expected_S00) < 1e-12
    assert abs(kf.Sigma[1][1] - diag_pre) < 1e-12


# ── Test 3: Gradient direction on quadratic ──────────────────────────


def test_gradient_direction_quadratic():
    """On f(θ) = -||θ||² (reward), BSZO moves θ toward the origin."""
    torch.manual_seed(42)
    module = nn.Linear(4, 1, bias=False)
    nn.init.constant_(module.weight, 5.0)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bszo = UHDBSZO(
        gp,
        dim,
        lr_scheduler=ConstantLR(0.005),
        epsilon=0.01,
        k=3,
    )

    def evaluate_fn():
        return -float((module.weight**2).sum()), 0.0

    norm_before = float((module.weight**2).sum())

    for _ in range(100):
        _run_step(bszo, evaluate_fn)

    norm_after = float((module.weight**2).sum())
    assert norm_after < norm_before


# ── Test 4: Adaptive noise σ²_e ──────────────────────────────────────


def test_adaptive_noise_residual_formula():
    """σ²_e ← (1−α)σ²_e + αr² where r = Y_last − μ_post[last]."""
    sigma_p_sq, sigma_e_sq, alpha = 1.0, 1.0, 0.2
    gamma = sigma_p_sq / (sigma_p_sq + sigma_e_sq)
    k = 2

    _, _, bszo = _make_bszo(k=k, sigma_p_sq=sigma_p_sq, sigma_e_sq=sigma_e_sq, alpha=alpha)
    kf = bszo._kf
    kf.init_step()

    Y = [3.0, -1.0]
    for i in range(k):
        kf.update_coord(i, Y[i])
        kf.last_d_idx = i
        kf.last_y = Y[i]
    kf.Y = list(Y)

    residual = Y[k - 1] - gamma * Y[k - 1]
    expected = (1.0 - alpha) * sigma_e_sq + alpha * residual**2

    kf.adaptive_step()

    assert abs(kf.sigma_e_sq - expected) < 1e-12


def test_sigma_e_changes_after_full_step():
    _, _, bszo = _make_bszo(k=2, sigma_e_sq=1.0)
    sigma_e_before = bszo._kf.sigma_e_sq

    for _ in range(bszo.k + 1):
        bszo.ask()
        bszo.tell(float(torch.randn(1).item()), 0.0)

    assert bszo._kf.sigma_e_sq != sigma_e_before


# ── Test 5: Phase state machine ──────────────────────────────────────


def test_phase_starts_at_zero():
    _, _, bszo = _make_bszo(k=2)
    assert bszo.phase == 0


def test_phase_transitions_k2():
    _, _, bszo = _make_bszo(k=2)

    bszo.ask()
    bszo.tell(1.0, 0.0)
    assert bszo.phase == 1

    bszo.ask()
    bszo.tell(1.1, 0.0)
    assert bszo.phase == 2

    bszo.ask()
    bszo.tell(0.9, 0.0)
    assert bszo.phase == 0


def test_eval_seed_constant_within_step():
    _, _, bszo = _make_bszo(k=2)

    seeds = []
    for _ in range(bszo.k + 1):
        seeds.append(bszo.eval_seed)
        bszo.ask()
        bszo.tell(1.0, 0.0)

    assert all(s == 0 for s in seeds)


def test_eval_seed_advances_after_step():
    _, _, bszo = _make_bszo(k=2)
    assert bszo.eval_seed == 0

    _run_step(bszo, lambda: (1.0, 0.0))
    assert bszo.eval_seed == 1

    _run_step(bszo, lambda: (1.0, 0.0))
    assert bszo.eval_seed == 2


def test_set_perturb_base_rejects_non_baseline_phase():
    _, _, bszo = _make_bszo(k=2)
    bszo.ask()
    bszo.tell(1.0, 0.0)

    with pytest.raises(RuntimeError, match="baseline"):
        bszo.set_perturb_base(42)


def test_set_perturb_base_works_at_phase_0():
    _, _, bszo = _make_bszo(k=2)
    bszo.set_perturb_base(100)
    assert bszo.perturb_seed(0) == 100
    assert bszo.perturb_seed(1) == 101


def test_perturb_base_advances_by_k():
    _, _, bszo = _make_bszo(k=3)
    assert bszo.perturb_seed(0) == 0

    _run_step(bszo, lambda: (1.0, 0.0))
    assert bszo.perturb_seed(0) == 3


# ── Test 6: Integration ──────────────────────────────────────────────


def test_y_best_starts_none():
    _, _, bszo = _make_bszo()
    assert bszo.y_best is None


def test_y_best_tracks_maximum():
    _, _, bszo = _make_bszo(k=2)
    values = [1.0, 1.1, 0.9, 2.0, 1.5, 1.8]
    for v in values:
        bszo.ask()
        bszo.tell(v, 0.0)
    assert bszo.y_best == 2.0


def test_mu_se_avg_reflect_last_tell():
    _, _, bszo = _make_bszo(k=2)
    bszo.ask()
    bszo.tell(1.5, 0.3)
    assert bszo.mu_avg == 1.5
    assert bszo.se_avg == 0.3


def test_y_best_improves_on_quadratic():
    torch.manual_seed(7)
    module = nn.Linear(3, 1, bias=False)
    nn.init.constant_(module.weight, 3.0)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bszo = UHDBSZO(
        gp,
        dim,
        lr_scheduler=ConstantLR(0.005),
        epsilon=0.01,
        k=2,
    )

    def evaluate_fn():
        return -float((module.weight**2).sum()), 0.0

    for _ in range(100):
        _run_step(bszo, evaluate_fn)

    assert bszo.y_best is not None
    assert bszo.y_best > -27.0


def test_params_stay_finite():
    torch.manual_seed(0)
    module, _, bszo = _make_bszo(k=2, lr=1e-4, epsilon=1e-3)

    for _ in range(20):
        _run_step(bszo, lambda: (float(torch.randn(1).item()), 0.0))

    assert all(torch.isfinite(p.data).all() for p in module.parameters())


def test_epsilon_property():
    _, _, bszo = _make_bszo(epsilon=0.05)
    assert bszo.epsilon == 0.05


# ── _apply_gradient magnitude (4.2) ──────────────────────────────────


def _get_perturbation_direction(module, seed):
    """Replay perturbator noise to extract the z vector for a given seed."""
    ref = nn.Linear(module.in_features, module.out_features, bias=module.bias is not None)
    ref.weight.data.zero_()
    if ref.bias is not None:
        ref.bias.data.zero_()
    gp = GaussianPerturbator(ref)
    gp.perturb(seed, 1.0)
    z = ref.weight.data.clone()
    gp.unperturb()
    return z


def test_apply_gradient_magnitude():
    """Verify Δθ = lr · Σ_i μ_i · z_i (no /k)."""
    torch.manual_seed(99)
    module = nn.Linear(3, 1, bias=False)
    nn.init.constant_(module.weight, 2.0)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    lr = 0.1
    k = 2
    bszo = UHDBSZO(gp, dim, lr_scheduler=ConstantLR(lr), epsilon=0.01, k=k)

    theta_before = module.weight.data.clone()

    def evaluate_fn():
        return -float((module.weight**2).sum()), 0.0

    _run_step(bszo, evaluate_fn)

    theta_after = module.weight.data.clone()
    actual_delta = theta_after - theta_before

    z0 = _get_perturbation_direction(module, 0)
    z1 = _get_perturbation_direction(module, 1)

    mu_post = bszo._kf.mu_post
    expected_delta = lr * (mu_post[0] * z0 + mu_post[1] * z1)
    assert torch.allclose(actual_delta, expected_delta, atol=1e-5)


# ── weight_decay (4.3) ───────────────────────────────────────────────


def test_weight_decay_shrinks_params():
    """With constant f, gradient is zero; only weight_decay acts: θ *= (1 - lr·wd)."""
    module = nn.Linear(3, 1, bias=False)
    nn.init.constant_(module.weight, 4.0)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    lr, wd = 0.01, 0.1
    bszo = UHDBSZO(gp, dim, lr_scheduler=ConstantLR(lr), epsilon=0.01, k=2, weight_decay=wd)

    _run_step(bszo, lambda: (1.0, 0.0))

    decay = 1.0 - lr * wd
    expected = 4.0 * decay
    assert torch.allclose(module.weight.data, torch.full_like(module.weight, expected), atol=1e-4)


# ── Integration: _run_bszo_iterations (4.4) ──────────────────────────


def test_run_bszo_iterations_loop():
    """Test the loop runner directly with a synthetic evaluate_fn."""
    from ops.uhd_setup import _run_bszo_iterations

    torch.manual_seed(0)
    module = nn.Linear(3, 1, bias=False)
    nn.init.constant_(module.weight, 3.0)
    dim = sum(p.numel() for p in module.parameters())
    gp = GaussianPerturbator(module)
    bszo = UHDBSZO(gp, dim, lr_scheduler=ConstantLR(0.005), epsilon=0.01, k=2)

    def evaluate_fn(eval_seed):
        return -float((module.weight**2).sum()), 0.0

    _run_bszo_iterations(
        bszo,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_steps=10,
        log_interval=5,
        accuracy_interval=0,
        target_accuracy=None,
    )

    assert bszo.eval_seed == 10
    assert bszo.y_best is not None
    assert bszo.y_best > -27.0


def test_run_bszo_wiring(monkeypatch):
    """Verify _run_bszo passes all BSZO config fields to run_bszo_loop."""
    import ops.exp_uhd as exp_uhd
    from ops.uhd_config import BEConfig, EarlyRejectConfig, ENNConfig, UHDConfig

    captured = {}

    def fake_run_bszo_loop(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs

    monkeypatch.setattr("ops.uhd_setup.run_bszo_loop", fake_run_bszo_loop)

    early_reject = EarlyRejectConfig(tau=None, mode=None, ema_beta=None, warmup_pos=None, quantile=None, window=None)
    be = BEConfig(10, 10, 20, 10, 25, None)
    enn = ENNConfig(
        minus_impute=False,
        d=100,
        s=4,
        jl_seed=123,
        k=25,
        fit_interval=50,
        warmup_real_obs=200,
        refresh_interval=50,
        se_threshold=0.25,
        target="mu_minus",
        num_candidates=1,
        select_interval=1,
        embedder="direction",
        gather_t=64,
    )
    cfg = UHDConfig(
        env_tag="mnist",
        num_rounds=5,
        problem_seed=42,
        noise_seed_0=7,
        lr=1e-4,
        num_dim_target=None,
        num_module_target=None,
        log_interval=1,
        accuracy_interval=100,
        target_accuracy=0.9,
        optimizer="bszo",
        batch_size=256,
        early_reject=early_reject,
        be=be,
        enn=enn,
        bszo_k=4,
        bszo_epsilon=1e-3,
        bszo_sigma_p_sq=2.0,
        bszo_sigma_e_sq=0.5,
        bszo_alpha=0.2,
    )

    exp_uhd._run_bszo(cfg)

    assert captured["args"] == ("mnist", 5)
    kw = captured["kwargs"]
    assert kw["lr"] == 1e-4
    assert kw["problem_seed"] == 42
    assert kw["noise_seed_0"] == 7
    assert kw["batch_size"] == 256
    assert kw["bszo_k"] == 4
    assert kw["bszo_epsilon"] == 1e-3
    assert kw["bszo_sigma_p_sq"] == 2.0
    assert kw["bszo_sigma_e_sq"] == 0.5
    assert kw["bszo_alpha"] == 0.2
    assert kw["target_accuracy"] == 0.9
