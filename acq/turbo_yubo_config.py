from dataclasses import dataclass

from acq.turbo_yubo_model_factory import build_turbo_yubo_model


@dataclass
class TurboYUBOConfig:
    raasp: bool = True
    lhd: bool = True
    tr: bool = True
    model_factory = staticmethod(build_turbo_yubo_model)


# TODO: Add field that contains a function that returns a fitted GP model with the _TurboPosteriorModel interface.
# TODO: Put both _TurboPosteriorModel and turbo_train_gp inside a new function and make that the default for the new field
