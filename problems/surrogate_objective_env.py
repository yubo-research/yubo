from __future__ import annotations

from problems.jax_env_core import _gymnax_box_from_shape, _stable_scale


class PaperObjectiveAdapter:
    _SPECS = {
        "llm:deepscaler:passk4": (64, 64, 4.0),
        "llm:deepscaler:rlvr": (64, 64, 3.0),
        "hft:goog-2023-sell-q30-t10": (128, 32, 2.0),
        "rwkv:distill:gsm8k": (64, 64, 1.5),
    }

    def __init__(self, env_name: str, *, jax, jnp) -> None:
        from gymnax.environments import spaces

        _ = jax
        self._jnp = jnp
        self.env_name = str(env_name)
        obs_dim, action_dim, reward_scale = self._resolve_spec(self.env_name)
        self._obs_dim = int(obs_dim)
        self._action_dim = int(action_dim)
        self._reward_scale = float(reward_scale)
        self._tag_scale = _stable_scale(self.env_name)
        self.observation_space = _gymnax_box_from_shape(spaces, jnp, (self._obs_dim,), low=-1.0, high=1.0)
        self.action_space = _gymnax_box_from_shape(spaces, jnp, (self._action_dim,), low=-1.0, high=1.0)

    @classmethod
    def _resolve_spec(cls, env_name: str) -> tuple[int, int, float]:
        for prefix, spec in cls._SPECS.items():
            if env_name.startswith(prefix):
                return spec
        raise ValueError(env_name)

    def _obs(self):
        idx = self._jnp.arange(self._obs_dim, dtype=self._jnp.float32)
        return self._jnp.sin((idx + 1.0) * self._tag_scale)

    def _target(self):
        idx = self._jnp.arange(self._action_dim, dtype=self._jnp.float32)
        return self._jnp.tanh(self._jnp.cos((idx + 1.0) * (self._tag_scale + 0.25)))

    def reset(self, _key):
        state = {"t": self._jnp.array(0, dtype=self._jnp.int32)}
        return self._obs(), state

    def step(self, _key, state, action):
        action = self.clip_action(action)
        target = self._target()
        mse = self._jnp.mean((action - target) ** 2)
        alignment = self._jnp.mean(action * target)
        reward = self._reward_scale * (alignment - mse)
        next_state = {"t": state["t"] + self._jnp.array(1, dtype=self._jnp.int32)}
        done = self._jnp.array(True)
        result = (self._obs(), next_state, reward.astype(self._jnp.float32), done, {})
        return result

    def clip_action(self, action):
        action = self._jnp.ravel(self._jnp.asarray(action, dtype=self._jnp.float32))
        if action.shape[0] != self._action_dim:
            action = self._jnp.resize(action, (self._action_dim,))
        return self._jnp.clip(action, -1.0, 1.0)


SurrogateObjectiveAdapter = PaperObjectiveAdapter
