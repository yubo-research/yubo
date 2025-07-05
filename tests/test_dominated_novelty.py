import numpy as np

from acq.acq_enn import AcqENN, ENNConfig


def test_dominated_novelty():
    num_dim = 2
    config = ENNConfig(
        k=3,
        num_candidates_per_arm=10,
        region_type="sobol",
        acq="dominated_novelty",
        k_novelty=1,
    )

    acq = AcqENN(num_dim, config)

    x_train = np.random.uniform(0, 1, (20, num_dim))
    y_train = np.random.uniform(0, 1, (20, 1))
    d_train = np.random.uniform(0, 1, (20, 7))

    acq.add(x_train, y_train, d_train)

    num_arms = 5
    x_selected = acq.draw(num_arms)

    print(f"Selected {len(x_selected)} arms with shape {x_selected.shape}")
    print(f"All points in [0,1]: {np.all((x_selected >= 0) & (x_selected <= 1))}")
    print(f"Unique points: {len(np.unique(x_selected, axis=0)) == len(x_selected)}")
    return x_selected
