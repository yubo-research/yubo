import numpy as np
import torch
from nds import ndomsort


def ty_thompson(train_x, model, x_cand, num_arms):
    if len(train_x) == 0:
        indices = torch.randperm(len(x_cand))[:num_arms]
        return x_cand[indices]
    with torch.no_grad():
        posterior = model.posterior(x_cand)
        samples = posterior.sample(sample_shape=torch.Size([num_arms])).squeeze(-1)
        if samples.dim() == 1:
            samples = samples.unsqueeze(0)
        y_cand = samples.t().contiguous()
        chosen = []
        for i in range(num_arms):
            indbest = torch.argmax(y_cand[:, i]).item()
            chosen.append(indbest)
            y_cand[indbest, :] = -float("inf")
        chosen = torch.tensor(chosen, device=x_cand.device)

        return x_cand[chosen]


def ty_pareto(train_x, model, x_cand, num_arms):
    mvn = model.posterior(x_cand)

    mets = [mvn.mu, mvn.se]

    return x_cand[_i_pareto_front_selection(num_arms, *mets), :]


def _i_pareto_front_selection(num_select, *metrics):
    combined_data = np.concatenate(metrics, axis=1)
    idx_front = np.array(
        ndomsort.non_domin_sort(-combined_data, only_front_indices=True)
    )

    i_keep = []
    for n_front in range(1 + max(idx_front)):
        front_indices = np.where(idx_front == n_front)[0]
        if num_select is None:
            i_keep.extend(front_indices)
            break

        if len(i_keep) + len(front_indices) <= num_select:
            i_keep.extend(front_indices)
        else:
            remaining = num_select - len(i_keep)
            i_keep.extend(
                np.random.choice(front_indices, size=remaining, replace=False)
            )
            break

    return np.array(i_keep)
