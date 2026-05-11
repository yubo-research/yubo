from types import SimpleNamespace


class DMControlWrapper:
    def __init__(self, dm_env, from_pixels, pixels_only):
        self.dm_env = dm_env
        self.from_pixels = from_pixels
        self.pixels_only = pixels_only
        self.observation_spec = SimpleNamespace(keys=lambda *args: ["state"])


class TransformedEnv:
    def __init__(self, base, transforms):
        self.base = base
        self.transforms = transforms


class TrEnvModule:
    DMControlWrapper = DMControlWrapper
    TransformedEnv = TransformedEnv
