from __future__ import annotations


def make_uhd_mnist_torch_evaluate_fn(
    env_tag: str,
    noise_seed_0: int,
    num_denoise: int | None,
    batch_size: int,
    module,
    device,
    train_images,
    train_labels,
    *,
    full_train_chunk: int = 8192,
):
    import math

    import torch
    import torch.nn.functional as F

    use_full_train_loss = str(env_tag) == "mnist_fulltrain"
    use_full_train_acc = str(env_tag) == "mnist_fulltrain_acc"

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

    return _evaluate_fn
