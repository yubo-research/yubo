import numpy as np
import torch
import torch.nn as nn

from policies.policy_mixin import PolicyParamsMixin
from problems.normalizer import Normalizer, normalize_running_state_array
from rl.shared_gaussian_actor import SharedGaussianActorModule, get_gaussian_actor_spec

from .common import _ensure_env_spaces, _obs_space_from_env_conf


class GaussianActorBackbonePolicy(PolicyParamsMixin, nn.Module):
    def __init__(
        self,
        env_conf,
        *,
        variant: str = "rl-gauss-tanh",
        deterministic_eval: bool = True,
        squash_mode: str = "clip",
        init_log_std: float = -0.5,
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._deterministic_eval = bool(deterministic_eval)
        mode = str(squash_mode).strip().lower()
        if mode not in {"clip", "tanh_clip"}:
            raise ValueError(f"Unsupported squash_mode '{squash_mode}'. Expected: clip, tanh_clip.")
        self._squash_mode = mode
        _ensure_env_spaces(env_conf)
        num_state = int(_obs_space_from_env_conf(env_conf).shape[0])
        num_action = int(env_conf.action_space.shape[0])
        self._normalizer = Normalizer(shape=(num_state,))
        self._clamp = env_conf.gym_conf is not None
        self.actor = SharedGaussianActorModule(
            num_state,
            num_action,
            *get_gaussian_actor_spec(variant),
            init_log_std=init_log_std,
        )
        self._const_scale = 0.5
        self._cache_flat_params_init()

    def _cache_flat_params_init(self):
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _normalize(self, state):
        return normalize_running_state_array(state, self._normalizer)

    def _postprocess_action(self, action_t: torch.Tensor) -> torch.Tensor:
        out = torch.tanh(action_t) if self._squash_mode == "tanh_clip" else action_t
        if self._clamp:
            return out.clamp(-1, 1)
        return out

    def __call__(self, state):
        state = self._normalize(state)
        state_t = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            raw_action_t, _, _ = self.actor.sample_action(state_t, deterministic=self._deterministic_eval)
            action_t = self._postprocess_action(raw_action_t)
        return action_t.detach().cpu().numpy()


class GaussianActorBackbonePolicyFactory:
    def __init__(
        self,
        variant: str = "rl-gauss-tanh",
        *,
        deterministic_eval: bool = True,
        squash_mode: str = "clip",
        init_log_std: float = -0.5,
    ):
        self._variant = str(variant)
        self._deterministic_eval = bool(deterministic_eval)
        self._squash_mode = str(squash_mode)
        self._init_log_std = float(init_log_std)

    def __call__(self, env_conf):
        return GaussianActorBackbonePolicy(
            env_conf,
            variant=self._variant,
            deterministic_eval=self._deterministic_eval,
            squash_mode=self._squash_mode,
            init_log_std=self._init_log_std,
        )
