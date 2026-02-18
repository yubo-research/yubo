import math
import warnings

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.uhd_enn_imputer import ENNImputerConfig, ENNMinusImputer, ENNMuPlusSeedSelector, ENNSeedSelectConfig
from optimizer.uhd_loop import UHDLoop


def _maybe_attach_enn(loop: UHDLoop, *, module, env, enabled: bool, cfg: ENNImputerConfig) -> None:
    if not enabled:
        return
    if not hasattr(env, "torch_env"):
        return
    if not isinstance(loop.perturbator, SparseGaussianPerturbator):
        return

    k = getattr(loop.perturbator, "_k", None)
    if k is not None:
        imputer = ENNMinusImputer(
            module=module,
            cfg=cfg,
            noise_nz_fn=lambda s, sig: loop.perturbator.sample_global_nz(seed=s, sigma=sig),
        )
        loop.set_enn(minus_imputer=imputer, seed_selector=None)
        return

    if str(cfg.target) != "mu_plus" or int(cfg.num_candidates) <= 1:
        return

    prob = float(getattr(loop.perturbator, "_prob", 1.0))
    selector = ENNMuPlusSeedSelector(
        module=module,
        perturbator=loop.perturbator,
        cfg=ENNSeedSelectConfig(
            d=int(cfg.d),
            s=int(cfg.s),
            jl_seed=int(cfg.jl_seed),
            k=int(cfg.k),
            fit_interval=int(cfg.fit_interval),
            warmup_real_obs=int(cfg.warmup_real_obs),
            num_candidates=int(cfg.num_candidates),
            noise_prob=prob,
            embedder=str(cfg.embedder),
            gather_t=int(cfg.gather_t),
        ),
    )
    loop.set_enn(minus_imputer=None, seed_selector=selector)


def _get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _make_accuracy_fn(module, device):
    from torchvision import datasets

    from problems.mnist_env import _MNIST_ROOT, _mnist_transform

    test_dataset = datasets.MNIST(
        root=_MNIST_ROOT,
        train=False,
        download=True,
        transform=_mnist_transform(),
    )
    images = torch.stack([test_dataset[i][0] for i in range(len(test_dataset))]).to(device)
    labels = torch.tensor([test_dataset[i][1] for i in range(len(test_dataset))]).to(device)

    def accuracy_fn():
        module.eval()
        with torch.no_grad():
            preds = module(images).argmax(dim=1)
        module.train()
        return float((preds == labels).float().mean())

    return accuracy_fn


def _preload_mnist_train_to_device(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load full MNIST train set as tensors and move once to device.

    This removes per-eval DataLoader overhead and repeated host→device copies.
    """
    from torchvision import datasets

    from problems.mnist_env import _MNIST_ROOT, _mnist_transform

    train_dataset = datasets.MNIST(
        root=_MNIST_ROOT,
        train=True,
        download=True,
        transform=_mnist_transform(),
    )
    loader = DataLoader(train_dataset, batch_size=8192, shuffle=False, drop_last=False)
    xs: list[torch.Tensor] = []
    ys: list[torch.Tensor] = []
    for x, y in loader:
        xs.append(x)
        ys.append(y)
    images = torch.cat(xs, dim=0).to(device)
    labels = torch.cat(ys, dim=0).to(device)
    return images, labels


def _parse_enn_cfg(enn: dict[str, object] | None) -> tuple[bool, ENNImputerConfig]:
    enn = {} if enn is None else dict(enn)
    enabled = bool(enn.get("enn_minus_impute", False))
    cfg = ENNImputerConfig(
        d=int(enn.get("enn_d", 100)),
        s=int(enn.get("enn_s", 4)),
        jl_seed=int(enn.get("enn_jl_seed", 123)),
        k=int(enn.get("enn_k", 25)),
        fit_interval=int(enn.get("enn_fit_interval", 50)),
        warmup_real_obs=int(enn.get("enn_warmup_real_obs", 200)),
        refresh_interval=int(enn.get("enn_refresh_interval", 50)),
        se_threshold=float(enn.get("enn_se_threshold", 0.25)),
        target=str(enn.get("enn_target", "mu_minus")),
        num_candidates=int(enn.get("enn_num_candidates", 1)),
        select_interval=int(enn.get("enn_select_interval", 1)),
        embedder=str(enn.get("enn_embedder", "direction")),
        gather_t=int(enn.get("enn_gather_t", 64)),
    )
    return enabled, cfg


def make_loop(
    env_tag,
    num_rounds,
    lr=0.001,
    sigma=0.001,
    num_dim_target=None,
    num_module_target=None,
    *,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    enn: dict[str, object] | None = None,
    early_reject_tau: float | None = None,
    early_reject_mode: str | None = None,
    early_reject_ema_beta: float | None = None,
    early_reject_warmup_pos: int | None = None,
    early_reject_quantile: float | None = None,
    early_reject_window: int | None = None,
):
    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf
    from problems.torch_policy import TorchPolicy

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    env = env_conf.make()

    if hasattr(env, "torch_env"):
        device = _get_device()
        torch_env = env.torch_env()
        module = torch_env.module.to(device)
        module.train()

        # MNIST-specific fast path: preload full train tensors to device and sample batches by seed.
        # This preserves antithetic pairing (same eval_seed → same sampled batch).
        train_images, train_labels = _preload_mnist_train_to_device(device)
        batch_size = int(getattr(torch_env, "_batch_size", 4096))
        use_full_train_loss = str(env_tag) == "mnist_fulltrain"
        use_full_train_acc = str(env_tag) == "mnist_fulltrain_acc"
        full_train_chunk = 8192

        def evaluate_fn(eval_seed):
            if use_full_train_loss:
                was_training = module.training
                module.eval()
                try:
                    with torch.inference_mode():
                        n = int(train_images.shape[0])
                        s = 0.0
                        ss = 0.0
                        count = 0
                        for start in range(0, n, full_train_chunk):
                            end = min(start + full_train_chunk, n)
                            logits = module(train_images[start:end])
                            per_sample = F.cross_entropy(logits, train_labels[start:end], reduction="none")
                            s += float(per_sample.sum())
                            ss += float((per_sample * per_sample).sum())
                            count += int(per_sample.numel())
                    mean_loss = s / count
                    var = max(ss / count - mean_loss * mean_loss, 0.0)
                    std = var**0.5
                    mu = -float(mean_loss)
                    se = float(std / math.sqrt(count))
                    return mu, se
                finally:
                    module.train(was_training)

            if use_full_train_acc:
                was_training = module.training
                module.eval()
                try:
                    with torch.inference_mode():
                        n = int(train_images.shape[0])
                        correct = 0
                        for start in range(0, n, full_train_chunk):
                            end = min(start + full_train_chunk, n)
                            logits = module(train_images[start:end])
                            pred = logits.argmax(dim=1)
                            correct += int((pred == train_labels[start:end]).sum())
                    acc = correct / n
                    # Report a binomial standard error (for display; objective is full-train).
                    se = float((acc * (1.0 - acc) / n) ** 0.5)
                    return float(acc), se
                finally:
                    module.train(was_training)

            noise_seed = eval_seed + noise_seed_0
            g = torch.Generator()
            g.manual_seed(int(noise_seed))
            idx = torch.randint(train_images.shape[0], (batch_size,), generator=g)
            idx_d = idx.to(device)
            images = train_images.index_select(0, idx_d)
            labels = train_labels.index_select(0, idx_d)
            with torch.inference_mode():
                logits = module(images)
                per_sample = F.cross_entropy(logits, labels, reduction="none")
            mu = -float(per_sample.mean())
            se = float(per_sample.std() / math.sqrt(len(per_sample)))
            return mu, se

        accuracy_fn = _make_accuracy_fn(module, device)
    else:
        env.close()

        from optimizer.trajectories import collect_trajectory
        from problems.mlp_torch_policy import MLPPolicyModule

        # Gym rollouts are CPU-bound (MuJoCo sim). Keep the policy on CPU too:
        # calling an MPS/CUDA model every env step adds huge device-transfer overhead.
        device = torch.device("cpu")

        # Match experiments/experiment_sampler.py seeding scheme (for reproducibility/comparison).
        if env_conf.problem_seed is not None:
            seed_all(int(env_conf.problem_seed) + 27)

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]

        # Prefer the env's configured policy_class if it is an nn.Module.
        # This keeps behavior aligned with `ops/experiment.py` for tags like `stand-mlp` / `stand-mlp:fn`.
        policy = None
        if env_conf.policy_class is not None:
            cand = env_conf.policy_class(env_conf)
            if isinstance(cand, torch.nn.Module):
                module = cand.to(device)
                policy = cand
            else:
                warnings.warn(
                    f"Ignoring non-module policy_class {env_conf.policy_class}; using MLPPolicyModule((32,16)).",
                    stacklevel=2,
                )

        if policy is None:
            module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
            policy = TorchPolicy(module, env_conf)

        def evaluate_fn(eval_seed):
            if bool(getattr(env_conf, "frozen_noise", False)):
                noise_seed = noise_seed_0
            else:
                noise_seed = eval_seed + noise_seed_0
            return float(collect_trajectory(env_conf, policy, noise_seed=noise_seed).rreturn), 0.0

    acc_fn = accuracy_fn if hasattr(env, "torch_env") else None
    loop = UHDLoop(
        module,
        evaluate_fn,
        num_iterations=num_rounds,
        lr=lr,
        sigma=sigma,
        accuracy_fn=acc_fn,
        num_dim_target=num_dim_target,
        num_module_target=num_module_target,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
        print_summary=True,
    )
    if early_reject_tau is not None or early_reject_mode is not None:
        loop.set_early_reject_advanced(
            tau=None if early_reject_tau is None else float(early_reject_tau),
            mode="y_best" if early_reject_mode is None else str(early_reject_mode),
            ema_beta=0.99 if early_reject_ema_beta is None else float(early_reject_ema_beta),
            warmup_pos=200 if early_reject_warmup_pos is None else int(early_reject_warmup_pos),
            quantile=0.5 if early_reject_quantile is None else float(early_reject_quantile),
            window=200 if early_reject_window is None else int(early_reject_window),
        )

    enn_minus_impute, enn_cfg = _parse_enn_cfg(enn)
    _maybe_attach_enn(loop, module=module, env=env, enabled=enn_minus_impute, cfg=enn_cfg)
    return loop
