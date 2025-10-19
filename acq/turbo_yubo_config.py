from dataclasses import dataclass

from acq.turbo_yubo_model_factory import build_turbo_yubo_model


@dataclass
class TurboYUBOConfig:
    raasp: bool = True
    lhd: bool = True
    tr: bool = True
    model_factory = staticmethod(build_turbo_yubo_model)
