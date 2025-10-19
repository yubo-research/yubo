from dataclasses import dataclass

from acq.turbo_yubo.turbo_yubo_model_factory import build_default_turbo_yubo_model
from sampling.lhd import latin_hypercube_design


@dataclass
class TurboYUBOConfig:
    raasp: bool = True
    initializer: staticmethod = staticmethod(latin_hypercube_design)
    model_factory: staticmethod = staticmethod(build_default_turbo_yubo_model)
