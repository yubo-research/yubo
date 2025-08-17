import faiss
import numpy as np


class EpistemicNovelty:
    def __init__(self, k_novelty):
        self.k_novelty = k_novelty
        self.descriptors = None
        self.descriptor_ses = None
        self.index = None
        self._num_dim = None

    def add(self, desc: np.ndarray, desc_se: np.ndarray):
        assert desc.ndim == 2, desc.ndim
        assert desc_se.ndim == 2, desc_se.ndim
        assert desc.shape == desc_se.shape, (desc.shape, desc_se.shape)

        if self._num_dim is None:
            self._num_dim = desc.shape[1]
            self.descriptors = desc.copy()
            self.descriptor_ses = desc_se.copy()
        else:
            assert desc.shape[1] == self._num_dim, (desc.shape[1], self._num_dim)
            self.descriptors = np.vstack([self.descriptors, desc])
            self.descriptor_ses = np.vstack([self.descriptor_ses, desc_se])

        if self.index is None:
            self.index = faiss.IndexFlatL2(self._num_dim)

        desc_array = desc.astype(np.double)
        self.index.add(desc_array)

    def dominated_novelty_of_last_addition(self, *, k_novelty=None):
        if self.descriptors is None or len(self.descriptors) < 2:
            return None, None
        desc = self.descriptors[-1].astype(np.double).reshape(1, -1)
        desc_se = self.descriptor_ses[-1].astype(np.double).reshape(1, -1)
        return self.novelty(desc, desc_se, k_novelty=k_novelty)

    def novelty(self, desc: np.ndarray, desc_se: np.ndarray, *, k_novelty=None, exclude_nearest=False):
        if self.descriptors is None or len(self.descriptors) < 2:
            return None, None

        if k_novelty is None:
            k_novelty = self.k_novelty

        if desc.ndim == 1:
            desc = desc.reshape(1, -1)
            desc_se = desc_se.reshape(1, -1)

        num_samples = desc.shape[0]
        k_search = min(k_novelty + (1 if exclude_nearest else 0), len(self.descriptors))
        _, neighbor_idx = self.index.search(desc.astype(np.double), k=k_search)

        if exclude_nearest:
            neighbor_idx = neighbor_idx[:, 1:]
        else:
            neighbor_idx = neighbor_idx[:, :k_novelty]

        desc_others = self.descriptors[neighbor_idx]
        desc_se_others = self.descriptor_ses[neighbor_idx]

        distances = np.linalg.norm(desc[:, None, :] - desc_others, axis=2)
        total_var = np.sum(desc_se[:, None, :] ** 2 + desc_se_others**2, axis=2)
        precisions = 1.0 / (total_var + 1e-8)

        novelty = np.sum(distances * precisions, axis=1) / np.sum(precisions, axis=1)
        novelty_variance = np.sum(precisions * (distances - novelty[:, None]) ** 2, axis=1) / np.sum(precisions, axis=1)
        novelty_se = np.sqrt(novelty_variance)

        mask = np.sum(precisions, axis=1) == 0
        novelty[mask] = 0.0
        novelty_se[mask] = 0.0

        if num_samples == 1:
            return novelty[0], novelty_se[0]
        else:
            return novelty, novelty_se
