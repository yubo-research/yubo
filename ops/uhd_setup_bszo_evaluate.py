from __future__ import annotations


def make_bszo_mnist_evaluate_fn(module, train_images, train_labels, batch_size, device, ns0: int):
    import math

    import torch
    import torch.nn.functional as F

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

    return evaluate_fn


def make_bszo_gym_evaluate_fn(env, module, *, noise_seed_0_arg: int | None):
    ns0 = env.noise_seed_0 or 0

    def evaluate_fn(eval_seed: int) -> tuple[float, float]:
        from optimizer.trajectories import collect_trajectory

        ns = noise_seed_0_arg if getattr(env, "frozen_noise", False) else int(eval_seed) + ns0
        result = collect_trajectory(env, module, noise_seed=ns)
        return float(result.rreturn), 0.0

    return evaluate_fn
