from __future__ import annotations

import os
import sys

from problems.dm_control_env_core import (
    DEFAULT_RENDER_HEIGHT,
    DEFAULT_RENDER_WIDTH,
    PIXEL_HEIGHT,
    PIXEL_WIDTH,
    DMControlEnv,
    configure_headless_render_backend,
    parse_env_name,
    suite,
)
from problems.dm_control_pixel_wrapper import PixelObsWrapper, make_dm_control
from problems.dm_control_spaces import BoxSpace, DictSpace

make = make_dm_control
_PixelObsWrapper = PixelObsWrapper
_configure_headless_render_backend = configure_headless_render_backend
_parse_env_name = parse_env_name

__all__ = [
    "BoxSpace",
    "DEFAULT_RENDER_HEIGHT",
    "DEFAULT_RENDER_WIDTH",
    "DMControlEnv",
    "DictSpace",
    "PIXEL_HEIGHT",
    "PIXEL_WIDTH",
    "PixelObsWrapper",
    "_PixelObsWrapper",
    "_configure_headless_render_backend",
    "_parse_env_name",
    "configure_headless_render_backend",
    "make",
    "make_dm_control",
    "os",
    "parse_env_name",
    "suite",
    "sys",
]
