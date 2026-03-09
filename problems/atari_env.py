from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any

import gymnasium as gym
import numpy as np
from ale_py import ALEInterface, roms
from scipy.ndimage import zoom

ATARI_FRAME_SIZE = 84
ATARI_FRAME_STACK = 4
ATARI_MAX_EPISODE_STEPS = 108000  # 30 min at 60fps, 4 frame skip -> ~27k steps


@dataclass(frozen=True)
class AtariPreprocessOptions:
    terminal_on_life_loss: bool = False
    grayscale_obs: bool = True
    grayscale_newaxis: bool = True
    scale_obs: bool = False
    repeat_action_probability: float = 0.0
    use_minimal_action_set: bool = True
    color_averaging: bool = False


def _parse_atari_tag(tag: str) -> str:
    """Parse atari:Pong or atari:Pong:agent57 -> ALE/Pong-v5."""
    if tag.startswith("atari:"):
        parts = tag.split(":", 1)[1].strip().split(":")
        game = parts[0].split("-")[0]
    elif tag.startswith("ALE/"):
        return tag if "-v" in tag else f"{tag}-v5"
    else:
        raise ValueError(f"Expected atari:Game or ALE/Game-v5, got: {tag}")
    return f"ALE/{game}-v5"


def _to_rom_id(env_id: str) -> str:
    if not env_id.startswith("ALE/"):
        raise ValueError(f"Expected ALE/<Game>-v5, got: {env_id}")
    core = env_id.split("/", 1)[1]
    game = core.split("-v", 1)[0]
    return game.lower().replace("-", "_")


def _resize_gray(frame: np.ndarray, size: int) -> np.ndarray:
    if frame.shape[0] == size and frame.shape[1] == size:
        return np.asarray(frame, dtype=np.uint8)
    scale_h = float(size) / float(frame.shape[0])
    scale_w = float(size) / float(frame.shape[1])
    out = zoom(frame, (scale_h, scale_w), order=1)
    return np.asarray(np.clip(out, 0, 255), dtype=np.uint8)


class ALEAtariEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"]}

    def __init__(
        self,
        env_id: str,
        *,
        render_mode: str | None = None,
        frameskip: int = 1,
        noop_max: int = 30,
        frame_skip: int = 4,
        screen_size: int = ATARI_FRAME_SIZE,
        preprocess: AtariPreprocessOptions | None = None,
        max_episode_steps: int = ATARI_MAX_EPISODE_STEPS,
    ):
        super().__init__()
        if int(frameskip) != 1:
            raise ValueError(f"Direct ALE adapter requires frameskip=1; got {frameskip}.")
        self.env_id = str(env_id)
        self.render_mode = render_mode
        self.noop_max = int(noop_max)
        self.frame_skip = int(frame_skip)
        self.screen_size = int(screen_size)
        self.max_episode_steps = int(max_episode_steps)
        self.preprocess = preprocess if preprocess is not None else AtariPreprocessOptions()

        if not self.preprocess.grayscale_obs:
            raise ValueError("Direct ALE adapter currently requires grayscale_obs=True.")
        repeat_action_probability = float(self.preprocess.repeat_action_probability)
        if not np.isfinite(repeat_action_probability) or repeat_action_probability < 0.0 or repeat_action_probability > 1.0:
            raise ValueError("repeat_action_probability must be finite and in [0, 1].")

        rom_id = _to_rom_id(self.env_id)
        rom_path = roms.get_rom_path(rom_id)

        ale = ALEInterface()
        ale.setInt("frame_skip", 1)
        ale.setFloat("repeat_action_probability", repeat_action_probability)
        ale.setBool("color_averaging", bool(self.preprocess.color_averaging))
        ale.loadROM(rom_path)
        self.ale = ale

        action_set = self.ale.getMinimalActionSet() if bool(self.preprocess.use_minimal_action_set) else self.ale.getLegalActionSet()
        self._action_set = [int(a) for a in action_set]
        if not self._action_set:
            raise ValueError("ALE action set is empty.")
        self._noop_action = 0 if 0 in self._action_set else int(self._action_set[0])
        self.action_space = gym.spaces.Discrete(len(self._action_set))

        obs_dtype = np.float32 if self.preprocess.scale_obs else np.uint8
        frame_shape = (self.screen_size, self.screen_size, 1) if bool(self.preprocess.grayscale_newaxis) else (self.screen_size, self.screen_size)
        obs_space = gym.spaces.Box(
            low=0,
            high=1 if self.preprocess.scale_obs else 255,
            shape=(ATARI_FRAME_STACK, *frame_shape),
            dtype=obs_dtype,
        )
        self.observation_space = obs_space

        self._frames: deque[np.ndarray] = deque(maxlen=ATARI_FRAME_STACK)
        self._episode_step = 0
        self._lives = 0
        self._rng = np.random.default_rng(0)

    def _get_gray(self) -> np.ndarray:
        frame = self.ale.getScreenGrayscale()
        if frame.ndim != 2:
            frame = np.asarray(frame).squeeze()
        frame = _resize_gray(np.asarray(frame, dtype=np.uint8), self.screen_size)
        if self.preprocess.scale_obs:
            out: np.ndarray = frame.astype(np.float32) / 255.0
        else:
            out = frame
        if self.preprocess.grayscale_newaxis:
            return out[..., None]
        return out

    def _build_obs(self) -> np.ndarray:
        if len(self._frames) < ATARI_FRAME_STACK:
            last = self._frames[-1]
            while len(self._frames) < ATARI_FRAME_STACK:
                self._frames.append(last.copy())
        stacked = np.stack(list(self._frames), axis=0)
        return stacked

    def _advance_noops(self) -> None:
        noops = int(self._rng.integers(1, self.noop_max + 1))
        for _ in range(noops):
            self.ale.act(int(self._noop_action))
            if self.ale.game_over():
                self.ale.reset_game()

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        if seed is not None:
            self._rng = np.random.default_rng(int(seed))
            self.ale.setInt("random_seed", int(seed))

        self.ale.reset_game()
        self._episode_step = 0
        self._advance_noops()

        self._frames.clear()
        frame = self._get_gray()
        for _ in range(ATARI_FRAME_STACK):
            self._frames.append(frame.copy())

        self._lives = int(self.ale.lives())
        return self._build_obs(), {}

    def step(self, action: int):
        a = int(action)
        if a < 0 or a >= len(self._action_set):
            raise ValueError(f"Invalid action index {action}; expected [0, {len(self._action_set) - 1}].")

        ale_action = int(self._action_set[a])
        reward = 0.0
        prev = self._get_gray()
        curr = prev

        terminated = False
        for _ in range(self.frame_skip):
            reward += float(self.ale.act(ale_action))
            curr = self._get_gray()
            if self.ale.game_over():
                terminated = True
                break
            prev = curr

        pooled = np.maximum(prev, curr)
        self._frames.append(pooled)
        self._episode_step += 1

        if self.preprocess.terminal_on_life_loss and not terminated:
            lives = int(self.ale.lives())
            if 0 < lives < self._lives:
                terminated = True
            self._lives = lives

        truncated = self._episode_step >= self.max_episode_steps
        obs = self._build_obs()
        return obs, reward, bool(terminated), bool(truncated), {}

    def render(self):
        if self.render_mode != "rgb_array":
            return None
        return np.asarray(self.ale.getScreenRGB(), dtype=np.uint8)

    def close(self):
        return


def make_atari_env(
    env_id: str,
    *,
    render_mode: str | None = None,
    frameskip: int = 1,
    noop_max: int = 30,
    frame_skip: int = 4,
    screen_size: int = ATARI_FRAME_SIZE,
    preprocess: AtariPreprocessOptions | None = None,
    max_episode_steps: int = ATARI_MAX_EPISODE_STEPS,
    **kwargs: Any,
) -> gym.Env:
    """Create Atari env with DQN-style preprocessing using ALE."""
    if kwargs:
        unknown = ", ".join(sorted(kwargs.keys()))
        raise ValueError(f"Unsupported Atari kwargs for direct ALE adapter: {unknown}")
    return ALEAtariEnv(
        env_id,
        render_mode=render_mode,
        frameskip=frameskip,
        noop_max=noop_max,
        frame_skip=frame_skip,
        screen_size=screen_size,
        preprocess=preprocess,
        max_episode_steps=max_episode_steps,
    )


def make(env_id: str, *, render_mode: str | None = None, **kwargs: Any) -> gym.Env:
    """Create Atari env. env_id can be atari:Pong or ALE/Pong-v5."""
    resolved = _parse_atari_tag(env_id) if "atari:" in env_id or env_id.startswith("ALE/") else env_id
    return make_atari_env(resolved, render_mode=render_mode, **kwargs)
