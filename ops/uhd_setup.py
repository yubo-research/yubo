import math

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ops.uhd_config import BEConfig, EarlyRejectConfig
from optimizer.gaussian_perturbator import GaussianPerturbator
from optimizer.sparse_gaussian_perturbator import SparseGaussianPerturbator
from optimizer.uhd_enn_imputer import ENNImputerConfig, ENNMinusImputer
from optimizer.uhd_enn_seed_selector import ENNMuPlusSeedSelector, ENNSeedSelectConfig
from optimizer.uhd_loop import UHDLoop


def _action_dim(space):
    if hasattr(space, "shape") and space.shape:
        return int(space.shape[0])
    if hasattr(space, "n"):
        return int(space.n)
    raise ValueError(f"Cannot get action dim from {type(space).__name__}: {space}")


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

    def _accuracy_fn():
        module.eval()
        with torch.no_grad():
            preds = module(images).argmax(dim=1)
        module.train()
        return float((preds == labels).float().mean())

    return _accuracy_fn


def _preload_mnist_train_to_device(
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
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


def _evaluate_gym_with_denoise(
    env_conf,
    policy,
    *,
    eval_seed: int,
    noise_seed_0: int,
    frozen: bool,
    num_denoise: int | None,
) -> tuple[float, float]:
    import numpy as np

    from optimizer.trajectories import collect_trajectory

    if num_denoise is None or int(num_denoise) <= 1:
        ns = noise_seed_0 if frozen else int(eval_seed) + noise_seed_0
        return float(collect_trajectory(env_conf, policy, noise_seed=ns).rreturn), 0.0

    base_seed = noise_seed_0 if frozen else int(eval_seed) + noise_seed_0
    rets: list[float] = []
    for j in range(int(num_denoise)):
        ns = int(base_seed + j)
        rets.append(float(collect_trajectory(env_conf, policy, noise_seed=ns).rreturn))
    mu = float(np.mean(rets))
    se = float(np.std(rets) / math.sqrt(len(rets)))
    return mu, se


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
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    enn: dict[str, object] | None = None,
    early_reject: EarlyRejectConfig | None = None,
):
    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf
    from problems.torch_policy import TorchPolicy

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed))

    env = env_conf.make()

    if hasattr(env, "torch_env"):
        device = _get_device()
        torch_env = env.torch_env()
        module = torch_env.module.to(device)
        module.train()

        train_images, train_labels = _preload_mnist_train_to_device(device)
        use_full_train_loss = str(env_tag) == "mnist_fulltrain"
        use_full_train_acc = str(env_tag) == "mnist_fulltrain_acc"
        full_train_chunk = 8192

        def _evaluate_fn(eval_seed):
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
                    se = float((acc * (1.0 - acc) / n) ** 0.5)
                    return float(acc), se
                finally:
                    module.train(was_training)

            if num_denoise is None or int(num_denoise) <= 1:
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

            vals = []
            for j in range(int(num_denoise)):
                noise_seed = eval_seed + noise_seed_0 + j
                g = torch.Generator()
                g.manual_seed(int(noise_seed))
                idx = torch.randint(train_images.shape[0], (batch_size,), generator=g)
                idx_d = idx.to(device)
                images = train_images.index_select(0, idx_d)
                labels = train_labels.index_select(0, idx_d)
                with torch.inference_mode():
                    logits = module(images)
                    per_sample = F.cross_entropy(logits, labels, reduction="none")
                vals.append(-float(per_sample.mean()))
            mu = float(sum(vals) / len(vals))
            se = float((torch.tensor(vals, dtype=torch.float64).std(unbiased=False) / math.sqrt(len(vals))).item())
            return mu, se

        accuracy_fn = _make_accuracy_fn(module, device)
    else:
        env.close()

        from problems.mlp_torch_policy import MLPPolicyModule

        device = torch.device("cpu")

        if env_conf.problem_seed is not None:
            seed_all(int(env_conf.problem_seed) + 27)

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = _action_dim(env_conf.action_space)

        np_policy = _try_make_np_policy(env_conf)
        if np_policy is not None:
            return _make_simple_loop_for_np_policy(
                env_conf,
                np_policy,
                optimizer="mezo",
                num_rounds=num_rounds,
                sigma=sigma,
                lr=lr,
                log_interval=log_interval,
                target_accuracy=target_accuracy,
                num_denoise=num_denoise,
                be_num_probes=10,
                be_num_candidates=10,
                be_warmup=20,
                be_fit_interval=10,
                be_enn_k=25,
            )

        # Torch-based policy for continuous action spaces
        policy = None
        module = None
        if env_conf.policy_class is not None:
            cand = env_conf.policy_class(env_conf)
            if isinstance(cand, torch.nn.Module):
                module = cand.to(device)
                policy = cand

        if policy is None:
            module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
            from problems.torch_policy import TorchPolicy

            policy = TorchPolicy(module, env_conf)

        def _evaluate_fn(eval_seed):
            return _evaluate_gym_with_denoise(
                env_conf,
                policy,
                eval_seed=eval_seed,
                noise_seed_0=noise_seed_0,
                frozen=bool(getattr(env_conf, "frozen_noise", False)),
                num_denoise=num_denoise,
            )

        accuracy_fn = None

    acc_fn = accuracy_fn if hasattr(env, "torch_env") else None
    loop = UHDLoop(
        module,
        _evaluate_fn,
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
    er = early_reject if early_reject is not None else EarlyRejectConfig()
    if er.tau is not None or er.mode is not None:
        loop.set_early_reject_advanced(
            tau=er.tau,
            mode="y_best" if er.mode is None else er.mode,
            ema_beta=0.99 if er.ema_beta is None else er.ema_beta,
            warmup_pos=200 if er.warmup_pos is None else er.warmup_pos,
            quantile=0.5 if er.quantile is None else er.quantile,
            window=200 if er.window is None else er.window,
        )
    enn_minus_impute, enn_cfg = _parse_enn_cfg(enn)
    _maybe_attach_enn(loop, module=module, env=env, enabled=enn_minus_impute, cfg=enn_cfg)
    return loop


def _make_simple_loop_for_np_policy(
    env_conf,
    np_policy,
    optimizer: str,
    num_rounds: int,
    sigma: float,
    lr: float,
    log_interval: int,
    target_accuracy: float | None,
    num_denoise: int | None = None,
    be: BEConfig | None = None,
):
    from common.seed_all import seed_all
    from embedding.behavioral_embedder import BehavioralEmbedder
    from optimizer.uhd_mezo_np import UHDMeZOBENp, UHDMeZONp
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed) + 27)

    dim = np_policy.num_params()
    noise_seed_0 = env_conf.noise_seed_0 or 0
    frozen = bool(getattr(env_conf, "frozen_noise", False))

    param_clip = (-1.0, 1.0)
    if optimizer == "simple":
        uhd = UHDSimpleNp(np_policy, sigma_0=sigma, param_clip=param_clip)
    elif optimizer == "simple_be":
        num_state = env_conf.gym_conf.state_space.shape[0]
        cfg = be if be is not None else BEConfig()
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        uhd = UHDSimpleBENp(
            np_policy,
            embedder,
            sigma_0=sigma,
            param_clip=param_clip,
            num_candidates=cfg.num_candidates,
            warmup=cfg.warmup,
            fit_interval=cfg.fit_interval,
            enn_k=cfg.enn_k,
        )
    elif optimizer == "mezo":
        uhd = UHDMeZONp(np_policy, sigma=sigma, lr=lr, param_clip=param_clip)
    elif optimizer == "mezo_be":
        num_state = env_conf.gym_conf.state_space.shape[0]
        cfg = be if be is not None else BEConfig()
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        uhd = UHDMeZOBENp(
            np_policy,
            embedder,
            sigma=sigma,
            lr=lr,
            param_clip=param_clip,
            num_candidates=cfg.num_candidates,
            warmup=cfg.warmup,
            fit_interval=cfg.fit_interval,
            enn_k=cfg.enn_k,
        )
    else:
        raise ValueError(f"Unknown optimizer for numpy policy: {optimizer}")

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_conf,
            np_policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Np: num_params = {dim}, optimizer = {optimizer}")
    _run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=None,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=0,
        target_accuracy=target_accuracy,
    )

    # Return a dummy object since caller expects something with run() method
    class _DummyLoop:
        def run(self):
            pass

    return _DummyLoop()


def _make_simple_optimizer(
    module,
    perturbator,
    *,
    optimizer: str,
    sigma: float,
    dim: int,
    embed_module=None,
    embed_bounds=None,
    be: BEConfig | None = None,
):
    cfg = be if be is not None else BEConfig()
    if optimizer in {"simple_be", "mezo_be"}:
        from embedding.behavioral_embedder import BehavioralEmbedder

        if embed_bounds is None:
            embed_bounds = _mnist_embed_bounds()
        if embed_module is None:
            embed_module = module
        embedder = BehavioralEmbedder(embed_bounds, num_probes=cfg.num_probes, seed=0)

        if optimizer == "mezo_be":
            from optimizer.uhd_simple_be import UHDMeZOBE

            return UHDMeZOBE(
                perturbator,
                dim,
                embed_module,
                embedder,
                sigma=sigma,
                num_candidates=cfg.num_candidates,
                warmup=cfg.warmup,
                fit_interval=cfg.fit_interval,
                enn_k=cfg.enn_k,
            )

        from optimizer.uhd_simple_be import UHDSimpleBE

        return UHDSimpleBE(
            perturbator,
            sigma_0=sigma,
            dim=dim,
            module=embed_module,
            embedder=embedder,
            num_candidates=cfg.num_candidates,
            warmup=cfg.warmup,
            fit_interval=cfg.fit_interval,
            enn_k=cfg.enn_k,
            sigma_range=cfg.sigma_range,
        )
    from optimizer.uhd_simple import UHDSimple

    return UHDSimple(perturbator, sigma_0=sigma, dim=dim, sigma_range=cfg.sigma_range)


def _mnist_embed_bounds() -> torch.Tensor:
    lb = (0.0 - 0.1307) / 0.3081
    ub = (1.0 - 0.1307) / 0.3081
    bounds = torch.zeros(2, 1, 28, 28)
    bounds[0] = lb
    bounds[1] = ub
    return bounds


def _gym_embed_bounds(num_state: int) -> torch.Tensor:
    bounds = torch.zeros(2, num_state)
    bounds[0] = -1.0
    bounds[1] = 1.0
    return bounds


def run_simple_loop(
    env_tag: str,
    num_rounds: int,
    sigma: float = 0.001,
    optimizer: str = "simple",
    *,
    num_dim_target: float | None = None,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    num_denoise: int | None = None,
    be: BEConfig | None = None,
) -> None:
    from common.seed_all import seed_all
    from problems.env_conf import get_env_conf

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed))
    env = env_conf.make()
    if hasattr(env, "torch_env"):
        _run_simple_torch(
            env,
            env_conf,
            env_tag,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            num_dim_target=num_dim_target,
            batch_size=batch_size,
            log_interval=log_interval,
            accuracy_interval=accuracy_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )
    else:
        _run_simple_gym(
            env,
            env_conf,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            num_dim_target=num_dim_target,
            log_interval=log_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )


def _run_simple_torch(
    env,
    env_conf,
    env_tag,
    num_rounds,
    *,
    optimizer,
    sigma,
    num_dim_target,
    batch_size,
    log_interval,
    accuracy_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    device = _get_device()
    torch_env = env.torch_env()
    module = torch_env.module.to(device)
    module.train()
    train_images, train_labels = _preload_mnist_train_to_device(device)

    dim = sum(p.numel() for p in module.parameters())
    perturbator = _make_perturbator(module, num_dim_target)
    uhd = _make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        be=be,
    )
    accuracy_fn = _make_accuracy_fn(module, device)

    if str(env_tag) == "mnist_fulltrain_acc":

        def evaluate_fn():
            return _eval_full_train_acc(module, train_images, train_labels, 8192)
    else:

        def evaluate_fn():
            if num_denoise is None or int(num_denoise) <= 1:
                g = torch.Generator()
                g.manual_seed(int(uhd.eval_seed))
                idx = torch.randint(train_images.shape[0], (batch_size,), generator=g).to(device)
                with torch.inference_mode():
                    logits = module(train_images.index_select(0, idx))
                    per_sample = F.cross_entropy(logits, train_labels.index_select(0, idx), reduction="none")
                mu = -float(per_sample.mean())
                se = float(per_sample.std() / math.sqrt(len(per_sample)))
                return mu, se

            vals = []
            for j in range(int(num_denoise)):
                g = torch.Generator()
                g.manual_seed(int(uhd.eval_seed + j))
                idx = torch.randint(train_images.shape[0], (batch_size,), generator=g).to(device)
                with torch.inference_mode():
                    logits = module(train_images.index_select(0, idx))
                    per_sample = F.cross_entropy(logits, train_labels.index_select(0, idx), reduction="none")
                vals.append(-float(per_sample.mean()))
            mu = float(sum(vals) / len(vals))
            se = float((torch.tensor(vals, dtype=torch.float64).std(unbiased=False) / math.sqrt(len(vals))).item())
            return mu, se

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}")
    _run_simple_iterations(
        uhd,
        evaluate_fn=evaluate_fn,
        accuracy_fn=accuracy_fn,
        num_rounds=num_rounds,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
    )


def _run_simple_gym(
    env,
    env_conf,
    num_rounds,
    *,
    optimizer,
    sigma,
    num_dim_target,
    log_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    from common.seed_all import seed_all

    env.close()
    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed) + 27)

    np_policy = _try_make_np_policy(env_conf)
    if np_policy is not None:
        _run_simple_gym_np(
            np_policy,
            env_conf,
            num_rounds,
            optimizer=optimizer,
            sigma=sigma,
            log_interval=log_interval,
            target_accuracy=target_accuracy,
            num_denoise=num_denoise,
            be=be,
        )
        return

    device = torch.device("cpu")
    num_state = env_conf.gym_conf.state_space.shape[0]
    num_action = _action_dim(env_conf.action_space)
    noise_seed_0 = env_conf.noise_seed_0 or 0

    module, policy = _make_gym_policy(env_conf, device, num_state, num_action)
    dim = sum(p.numel() for p in module.parameters())
    perturbator = _make_perturbator(module, num_dim_target)

    embed_module = getattr(module, "model", module)
    embed_bounds = _gym_embed_bounds(num_state)
    uhd = _make_simple_optimizer(
        module,
        perturbator,
        optimizer=optimizer,
        sigma=sigma,
        dim=dim,
        embed_module=embed_module,
        embed_bounds=embed_bounds,
        be=be,
    )

    frozen = bool(getattr(env_conf, "frozen_noise", False))

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_conf,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Simple: num_params = {dim}, optimizer = {optimizer}, state={num_state}, action={num_action}")
    _run_simple_iterations(
        uhd, evaluate_fn=evaluate_fn, accuracy_fn=None, num_rounds=num_rounds, log_interval=log_interval, accuracy_interval=0, target_accuracy=target_accuracy
    )


def _try_make_np_policy(env_conf):
    if env_conf.policy_class is None:
        return None
    cand = env_conf.policy_class(env_conf)
    if isinstance(cand, torch.nn.Module):
        return None
    if not hasattr(cand, "get_params"):
        return None
    return cand


def _run_simple_gym_np(
    policy,
    env_conf,
    num_rounds,
    *,
    optimizer,
    sigma,
    log_interval,
    target_accuracy,
    num_denoise,
    be: BEConfig | None = None,
) -> None:
    from embedding.behavioral_embedder import BehavioralEmbedder
    from optimizer.uhd_mezo_np import UHDMeZOBENp, UHDMeZONp
    from optimizer.uhd_simple_be_np import UHDSimpleBENp
    from optimizer.uhd_simple_np import UHDSimpleNp

    noise_seed_0 = env_conf.noise_seed_0 or 0
    frozen = bool(getattr(env_conf, "frozen_noise", False))
    dim = policy.num_params()
    param_clip = (-1.0, 1.0)

    cfg = be if be is not None else BEConfig()
    if optimizer in {"simple_be", "mezo_be"}:
        num_state = env_conf.gym_conf.state_space.shape[0]
        embedder = BehavioralEmbedder(_gym_embed_bounds(num_state), num_probes=cfg.num_probes, seed=0)
        if optimizer == "simple_be":
            uhd = UHDSimpleBENp(
                policy,
                embedder,
                sigma_0=sigma,
                param_clip=param_clip,
                num_candidates=cfg.num_candidates,
                warmup=cfg.warmup,
                fit_interval=cfg.fit_interval,
                enn_k=cfg.enn_k,
            )
        else:
            uhd = UHDMeZOBENp(
                policy,
                embedder,
                sigma=sigma,
                lr=0.001,
                param_clip=param_clip,
                num_candidates=cfg.num_candidates,
                warmup=cfg.warmup,
                fit_interval=cfg.fit_interval,
                enn_k=cfg.enn_k,
            )
    elif optimizer == "mezo":
        uhd = UHDMeZONp(policy, sigma=sigma, lr=0.001, param_clip=param_clip)
    else:
        uhd = UHDSimpleNp(policy, sigma_0=sigma, param_clip=param_clip)

    def evaluate_fn():
        return _evaluate_gym_with_denoise(
            env_conf,
            policy,
            eval_seed=uhd.eval_seed,
            noise_seed_0=noise_seed_0,
            frozen=frozen,
            num_denoise=num_denoise,
        )

    print(f"UHD-Np: num_params = {dim}, optimizer = {optimizer}")
    _run_simple_iterations(
        uhd, evaluate_fn=evaluate_fn, accuracy_fn=None, num_rounds=num_rounds, log_interval=log_interval, accuracy_interval=0, target_accuracy=target_accuracy
    )


def _make_gym_policy(env_conf, device, num_state, num_action):
    import warnings

    from problems.mlp_torch_policy import MLPPolicyModule
    from problems.torch_policy import TorchPolicy

    if env_conf.policy_class is not None:
        cand = env_conf.policy_class(env_conf)
        if isinstance(cand, torch.nn.Module):
            return cand.to(device), cand
        warnings.warn(f"Non-module policy_class {env_conf.policy_class}; using MLPPolicyModule.", stacklevel=2)
    module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16)).to(device)
    return module, TorchPolicy(module, env_conf)


def _make_perturbator(module, num_dim_target):
    if num_dim_target is not None:
        return SparseGaussianPerturbator(module, num_dim_target=num_dim_target)
    return GaussianPerturbator(module)


def _run_simple_iterations(
    uhd,
    *,
    evaluate_fn,
    accuracy_fn,
    num_rounds: int,
    log_interval: int,
    accuracy_interval: int,
    target_accuracy: float | None,
) -> None:
    import time

    t0 = time.perf_counter()
    acc = None
    for i in range(num_rounds):
        uhd.ask()
        mu, se = evaluate_fn()
        uhd.tell(mu, se)

        if not _should_log_simple(i, num_rounds, log_interval):
            continue
        y_best = uhd.y_best
        y_str = f"{y_best:.4f}" if y_best is not None else "N/A"
        if accuracy_fn is not None and (acc is None or i == num_rounds - 1 or (accuracy_interval > 0 and i % accuracy_interval == 0)):
            acc = accuracy_fn()
        line = f"EVAL: i_iter = {i} sigma = {uhd.sigma:.6f} mu = {mu:.4f} se = {se:.4f} y_best = {y_str}"
        if acc is not None:
            line += f" test_acc = {acc:.4f}"
        print(line)
        if target_accuracy is not None and mu >= target_accuracy:
            elapsed = time.perf_counter() - t0
            print(f"UHD-Simple: target reached {mu:.4f} >= {target_accuracy:.4f} at i_iter={i} ({elapsed:.2f}s)")
            break

    elapsed = time.perf_counter() - t0
    print(f"UHD-Simple: elapsed = {elapsed:.2f}s ({min(i + 1, num_rounds)} iterations)")


def _eval_full_train_acc(module, images, labels, chunk: int) -> tuple[float, float]:
    was_training = module.training
    module.eval()
    correct = _count_correct(module, images, labels, chunk)
    if was_training:
        module.train()
    n = int(images.shape[0])
    acc = correct / n
    se = float((acc * (1.0 - acc) / n) ** 0.5)
    return float(acc), se


def _count_correct(module, images, labels, chunk: int) -> int:
    n = int(images.shape[0])
    correct = 0
    with torch.inference_mode():
        for start in range(0, n, chunk):
            end = min(start + chunk, n)
            pred = module(images[start:end]).argmax(dim=1)
            correct += int((pred == labels[start:end]).sum())
    return correct


def _should_log_simple(i: int, num_rounds: int, log_interval: int) -> bool:
    return i == 0 or i == num_rounds - 1 or (log_interval > 0 and i % log_interval == 0)


def run_bszo_loop(
    env_tag: str,
    num_steps: int,
    lr: float = 0.001,
    *,
    problem_seed: int | None = None,
    noise_seed_0: int | None = None,
    batch_size: int = 4096,
    log_interval: int = 1,
    accuracy_interval: int = 1000,
    target_accuracy: float | None = None,
    bszo_k: int = 2,
    bszo_epsilon: float = 1e-4,
    bszo_sigma_p_sq: float = 1.0,
    bszo_sigma_e_sq: float = 1.0,
    bszo_alpha: float = 0.1,
) -> None:
    from common.seed_all import seed_all
    from optimizer.lr_scheduler import ConstantLR
    from optimizer.uhd_bszo import UHDBSZO
    from problems.env_conf import get_env_conf

    env_conf = get_env_conf(env_tag, problem_seed=problem_seed, noise_seed_0=noise_seed_0)
    ns0 = env_conf.noise_seed_0 or 0
    if env_conf.problem_seed is not None:
        seed_all(int(env_conf.problem_seed))

    # Try to make torch env (for MLP policies or MNIST-style envs)
    env = env_conf.make_torch_env()
    if not hasattr(env, "torch_env"):
        raise ValueError(f"BSZO requires an environment with torch_env(), got: {env_tag}")

    device = _get_device()
    module = env.torch_env().module.to(device)
    module.train()

    dim = sum(p.numel() for p in module.parameters())
    perturbator = GaussianPerturbator(module)
    bszo = UHDBSZO(
        perturbator,
        dim,
        lr_scheduler=ConstantLR(lr),
        epsilon=bszo_epsilon,
        k=bszo_k,
        sigma_p_sq=bszo_sigma_p_sq,
        sigma_e_sq=bszo_sigma_e_sq,
        alpha=bszo_alpha,
    )

    # Check if this is an MNIST environment (has MNIST data)
    is_mnist = env_tag.startswith("mnist")

    if is_mnist:
        train_images, train_labels = _preload_mnist_train_to_device(device)
        accuracy_fn = _make_accuracy_fn(module, device)

        def evaluate_fn(eval_seed: int) -> tuple[float, float]:
            g = torch.Generator()
            g.manual_seed(int(eval_seed + ns0))
            idx = torch.randint(train_images.shape[0], (batch_size,), generator=g).to(device)
            with torch.inference_mode():
                logits = module(train_images.index_select(0, idx))
                per_sample = F.cross_entropy(logits, train_labels.index_select(0, idx), reduction="none")
            mu = -float(per_sample.mean())
            se = float(per_sample.std() / math.sqrt(len(per_sample)))
            return mu, se
    else:
        # Gym environment: evaluate by running episodes
        from optimizer.trajectories import collect_trajectory

        accuracy_fn = None  # No accuracy metric for gym envs

        def evaluate_fn(eval_seed: int) -> tuple[float, float]:
            ns = noise_seed_0 if getattr(env_conf, "frozen_noise", False) else int(eval_seed) + ns0
            result = collect_trajectory(env_conf, module, noise_seed=ns)
            return float(result.rreturn), 0.0

    print(f"BSZO: num_params = {dim}, k = {bszo_k}, epsilon = {bszo_epsilon}, lr = {lr}")
    _run_bszo_iterations(
        bszo,
        evaluate_fn=evaluate_fn,
        accuracy_fn=accuracy_fn,
        num_steps=num_steps,
        log_interval=log_interval,
        accuracy_interval=accuracy_interval,
        target_accuracy=target_accuracy,
    )


def _run_bszo_iterations(
    bszo,
    *,
    evaluate_fn,
    accuracy_fn,
    num_steps: int,
    log_interval: int,
    accuracy_interval: int,
    target_accuracy: float | None,
) -> None:
    import time

    k = bszo.k
    t0 = time.perf_counter()
    acc = None
    step = 0
    for step in range(num_steps):
        for _ in range(k + 1):
            bszo.ask()
            mu, se = evaluate_fn(bszo.eval_seed)
            bszo.tell(mu, se)

        if not _should_log_simple(step, num_steps, log_interval):
            continue
        y_best = bszo.y_best
        y_str = f"{y_best:.4f}" if y_best is not None else "N/A"
        if accuracy_fn is not None and (acc is None or step == num_steps - 1 or (accuracy_interval > 0 and step % accuracy_interval == 0)):
            acc = accuracy_fn()
        line = f"EVAL: step = {step} mu = {mu:.4f} se = {se:.4f} y_best = {y_str}"
        if acc is not None:
            line += f" test_acc = {acc:.4f}"
        print(line)
        if target_accuracy is not None and acc is not None and acc >= target_accuracy:
            elapsed = time.perf_counter() - t0
            print(f"BSZO: target reached {acc:.4f} >= {target_accuracy:.4f} at step={step} ({elapsed:.2f}s)")
            break

    elapsed = time.perf_counter() - t0
    print(f"BSZO: elapsed = {elapsed:.2f}s ({min(step + 1, num_steps)} steps)")
