from dataclasses import dataclass

from acq.turbo_yubo.turbo_yubo_model_factory import build_default_turbo_yubo_model
from sampling.lhd import latin_hypercube_design
from sampling.sampling_util import raasp_turbo_np


@dataclass
class TurboYUBOConfig:
    raasp: staticmethod = staticmethod(raasp_turbo_np)
    initializer: staticmethod = staticmethod(latin_hypercube_design)
    model_factory: staticmethod = staticmethod(build_default_turbo_yubo_model)
