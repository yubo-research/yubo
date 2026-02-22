import numpy as np
import torch
from torch import nn

from .normalizer import Normalizer


class TorchPolicy:
    def __init__(self, module: nn.Module, env_conf):
        self.module = module
        self.problem_seed = env_conf.problem_seed

        if env_conf.gym_conf is not None and env_conf.gym_conf.transform_state:
            num_state = env_conf.gym_conf.state_space.shape[0]
            self._normalizer = Normalizer(shape=(num_state,))
        else:
            self._normalizer = None

        self._clamp = env_conf.gym_conf is not None

    def __call__(self, state):
        if self._normalizer is not None:
            state = np.asarray(state, dtype=np.float32)
            self._normalizer.update(state)
            mean, std = self._normalizer.mean_and_std()
            std = np.where(std == 0, 1.0, std)
            state = (state - mean) / std

        device = next(self.module.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            action_t = self.module(state_t)
        if self._clamp:
            action_t = torch.clamp(action_t, -1, 1)
        return action_t.detach().cpu().numpy()


class GaussianTorchPolicy:
    """Torch policy wrapper for diagonal Gaussian actor modules.

    Expected module API:
    - `sample_action(state_t, deterministic: bool) -> (action_t, log_prob_t, entropy_t)`
    """

    def __init__(
        self,
        module: nn.Module,
        env_conf,
        *,
        deterministic_eval: bool = True,
        squash_mode: str = "clip",
    ):
        if not hasattr(module, "sample_action"):
            raise TypeError("GaussianTorchPolicy requires module.sample_action(state_t, deterministic=...).")

        self.module = module
        self.problem_seed = env_conf.problem_seed
        self._deterministic_eval = bool(deterministic_eval)
        mode = str(squash_mode).strip().lower()
        if mode not in {"clip", "tanh_clip"}:
            raise ValueError(f"Unsupported squash_mode '{squash_mode}'. Expected one of: clip, tanh_clip.")
        self._squash_mode = mode

        if env_conf.gym_conf is not None and env_conf.gym_conf.transform_state:
            num_state = env_conf.gym_conf.state_space.shape[0]
            self._normalizer = Normalizer(shape=(num_state,))
        else:
            self._normalizer = None

        self._clamp = env_conf.gym_conf is not None

    def _normalize_state(self, state):
        if self._normalizer is None:
            return state
        state = np.asarray(state, dtype=np.float32)
        self._normalizer.update(state)
        mean, std = self._normalizer.mean_and_std()
        std = np.where(std == 0, 1.0, std)
        return (state - mean) / std

    def _postprocess_action(self, action_t: torch.Tensor) -> torch.Tensor:
        if self._squash_mode == "tanh_clip":
            action_t = torch.tanh(action_t)
        if self._clamp:
            action_t = torch.clamp(action_t, -1, 1)
        return action_t

    def __call__(self, state, *, deterministic: bool | None = None):
        if deterministic is None:
            deterministic = self._deterministic_eval

        state = self._normalize_state(state)
        device = next(self.module.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            raw_action_t, _, _ = self.module.sample_action(state_t, deterministic=bool(deterministic))
            action_t = self._postprocess_action(raw_action_t)
        return action_t.detach().cpu().numpy()

    def sample_with_log_prob(self, state, *, deterministic: bool = False):
        state = self._normalize_state(state)
        device = next(self.module.parameters()).device
        state_t = torch.as_tensor(state, dtype=torch.float32).to(device)
        with torch.no_grad():
            raw_action_t, log_prob_t, _ = self.module.sample_action(state_t, deterministic=bool(deterministic))
            action_t = self._postprocess_action(raw_action_t)
        return action_t.detach().cpu().numpy(), log_prob_t.detach().cpu().numpy()
