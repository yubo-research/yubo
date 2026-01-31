import numpy as np


def _grid_hash(x, num_bins_per_dim):
    N, d = x.shape
    x_min = x.min(axis=0)
    x_max = x.max(axis=0)
    x_range = x_max - x_min
    x_range = np.where(x_range > 0, x_range, 1.0)
    grid_coords = ((x - x_min) / x_range * num_bins_per_dim).astype(np.int32)
    grid_coords = np.clip(grid_coords, 0, num_bins_per_dim - 1)
    grid_hash = np.zeros(N, dtype=np.int64)
    for i in range(d):
        grid_hash = grid_hash * num_bins_per_dim + grid_coords[:, i]
    return grid_hash


def _cell_to_points(grid_hash):
    cell_to_points = {}
    for idx, cell_hash in enumerate(grid_hash):
        cell_to_points.setdefault(cell_hash, []).append(idx)
    return cell_to_points


def _select_from_cells(cell_to_points, M):
    selected_indices = []
    cell_keys = list(cell_to_points.keys())
    np.random.shuffle(cell_keys)
    num_cells = len(cell_keys)
    points_per_cell = M // num_cells
    remaining = M % num_cells
    for i, cell_hash in enumerate(cell_keys):
        points_in_cell = cell_to_points[cell_hash]
        num_to_take = points_per_cell + (1 if i < remaining else 0)
        num_to_take = min(num_to_take, len(points_in_cell))
        if num_to_take > 0:
            selected = np.random.choice(points_in_cell, size=num_to_take, replace=False)
            selected_indices.extend(selected.tolist())
    return selected_indices


def _fill_remaining(selected_indices, N, M):
    if len(selected_indices) >= M:
        return selected_indices[:M]
    remaining_needed = M - len(selected_indices)
    unselected = sorted(set(range(N)) - set(selected_indices))
    if len(unselected) >= remaining_needed:
        additional = np.random.choice(unselected, size=remaining_needed, replace=False)
        selected_indices.extend(additional.tolist())
    else:
        selected_indices.extend(unselected)
    return selected_indices[:M]


def diversity_subsample(x, M, num_bins_per_dim=None, seed=None):
    assert x.ndim == 2, f"x must be 2D, got shape {x.shape}"
    N, d = x.shape
    assert M < N, f"M ({M}) must be less than N ({N})"
    assert M > 0, f"M ({M}) must be positive"

    if seed is not None:
        np.random.seed(seed)

    if num_bins_per_dim is None:
        num_bins_per_dim = max(2, int(np.ceil(M ** (1.0 / d))))

    grid_hash = _grid_hash(x, num_bins_per_dim)
    cell_to_points = _cell_to_points(grid_hash)
    selected_indices = _select_from_cells(cell_to_points, M)
    selected_indices = _fill_remaining(selected_indices, N, M)
    return np.array(selected_indices, dtype=np.int64)
