from __future__ import annotations


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
    be=None,
) -> None:
    import math

    import torch

    from common.im import im

    c = im("ops.uhd_setup_simple_common")
    u = im("ops.uhd_setup_util")
    _eval_full_train_acc = c._eval_full_train_acc
    _make_perturbator = c._make_perturbator
    _make_simple_optimizer = c._make_simple_optimizer
    _run_simple_iterations = c._run_simple_iterations
    _get_device = u._get_device
    _make_accuracy_fn = u._make_accuracy_fn
    _preload_mnist_train_to_device = u._preload_mnist_train_to_device

    F = torch.nn.functional
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
