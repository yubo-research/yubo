from dataclasses import dataclass
from typing import Callable

from acq.turbo_yubo.ty_default_tr import TYDefaultTR
from acq.turbo_yubo.ty_model_factory import (
    build_default_turbo_yubo_model,
    default_targeter,
)
from acq.turbo_yubo.ty_selectors import ty_thompson
from sampling.lhd import latin_hypercube_design
from sampling.sampling_util import raasp_turbo_np


@dataclass
class TurboYUBOConfig:
    candidate_sampler: Callable = raasp_turbo_np
    candidate_initializer: Callable = latin_hypercube_design
    model_factory: Callable = build_default_turbo_yubo_model
    trust_region_manager: Callable = TYDefaultTR
    targeter: Callable = default_targeter
    candidate_selector: Callable = ty_thompson
