"""type()-built doubles for torchrl video helper tests (kiss)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np


def env_conf_no_transform_instance():
    def ensure_spaces(self):
        self.gym_conf = None

    return type("_EnvConfNoTransform", (), {"ensure_spaces": ensure_spaces})()


def env_conf_transform_with_space(low, high):
    Space = type("_SpaceStub", (), {"low": low, "high": high})

    def ensure_spaces(self):
        self.gym_conf = SimpleNamespace(transform_state=True, state_space=Space())

    return type("_EnvConfTransform", (), {"ensure_spaces": ensure_spaces})()


def patch_rollout_video_writer(monkeypatch, frames: list, video_path_holder: dict):
    def _fake_writer(video_path, *, fps=30):
        _ = fps
        path = Path(video_path)
        video_path_holder["path"] = path

        def append_data(self, frame):
            frames.append(np.asarray(frame))

        def close(self):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()

        return type("_Writer", (), {"append_data": append_data, "close": close})()

    monkeypatch.setattr("common.video_rollout._open_frame_video_writer", _fake_writer)
