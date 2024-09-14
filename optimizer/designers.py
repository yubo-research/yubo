from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)

from acq.acq_dpp import AcqDPP
from acq.acq_min_dist import AcqMinDist
from acq.acq_mtv import AcqMTV
from acq.acq_ts import AcqTS
from acq.acq_var import AcqVar

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .cma_designer import CMAESDesigner
from .optuna_designer import OptunaDesigner
from .random_designer import RandomDesigner
from .sobol_designer import SobolDesigner
from .turbo_designer import TuRBODesigner


class NoSuchDesignerError(Exception):
    pass


class Designers:
    def __init__(self, policy, num_arms):
        self._policy = policy
        self._num_arms = num_arms

    def is_valid(self, designer_name):
        return designer_name in self._designers

    def create(self, designer_name):
        init_ax_default = max(5, 2 * self._policy.num_params())
        default_num_X_samples = max(64, 10 * self._num_arms)
        # default_num_Y_samples = 512

        # mtv:k100
        if ":" in designer_name:
            designer_name, options = designer_name.split(":")
            options = options.split("-")
        else:
            options = []

        num_keep = None
        use_vanilla = False
        for option in options:
            if option[0] == "k":
                # Select a subset of num_keep observations
                #  to use for modeling and acquisition function
                #  optimization. The goal is to limit the
                #  O(num_observations^3) query complexity to O(num_keep^3)
                num_keep = int(option[1:])
                print(f"OPTION: num_keep = {num_keep}")
            elif option == "van":
                use_vanilla = True
                print(f"OPTION use_vanilla = {use_vanilla}")
            else:
                assert False, ("Unknown option", option)

        def bt_designer(acq_factory, acq_kwargs=None, init_sobol=1, opt_sequential=False):
            return BTDesigner(
                self._policy,
                acq_factory,
                acq_kwargs=acq_kwargs,
                num_keep=num_keep,
                use_vanilla=use_vanilla,
                init_sobol=init_sobol,
                opt_sequential=opt_sequential,
            )

        if designer_name == "cma":
            return CMAESDesigner(self._policy)
        elif designer_name == "optuna":
            return OptunaDesigner(self._policy)
        elif designer_name == "ax":
            return AxDesigner(self._policy)
        elif designer_name == "maximin":
            return bt_designer(lambda m: AcqMinDist(m, toroidal=False))
        elif designer_name == "maximin-toroidal":
            return bt_designer(lambda m: AcqMinDist(m, toroidal=True))
        elif designer_name == "variance":
            return bt_designer(AcqVar)

        # Init only, no surrogate, all exploration
        elif designer_name == "random":
            return RandomDesigner(self._policy)
        elif designer_name == "sobol":
            return SobolDesigner(self._policy)

        # All exploitation
        elif designer_name == "sr":
            return bt_designer(qSimpleRegret)

        # Various methods, first batch is Sobol
        elif designer_name == "ts":
            return bt_designer(
                AcqTS,
                acq_kwargs={
                    "sampler": "cholesky",
                    "num_candidates": 1024,
                },
            )
        elif designer_name == "ts-ciq":
            return bt_designer(AcqTS, acq_kwargs={"sampler": "ciq"})
        elif designer_name == "ts-lanczos":
            return bt_designer(AcqTS, acq_kwargs={"sampler": "lanczos"})
        elif designer_name.startswith("ts_sweep"):
            num_candidates = int(designer_name.split("-")[1])
            return bt_designer(
                AcqTS,
                acq_kwargs={
                    "sampler": "cholesky",
                    "num_candidates": num_candidates,
                },
            )
        elif designer_name == "ucb":
            return bt_designer(qUpperConfidenceBound, acq_kwargs={"beta": 1})
        elif designer_name == "ei":
            return bt_designer(qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None})
        elif designer_name == "lei":
            return bt_designer(qLogNoisyExpectedImprovement, acq_kwargs={"X_baseline": None})
        elif designer_name == "gibbon":
            return bt_designer(qLowerBoundMaxValueEntropy, opt_sequential=True, acq_kwargs={"candidate_set": None})
        elif designer_name == "turbo-1":
            return TuRBODesigner(self._policy, num_init=init_ax_default)
        elif designer_name == "turbo-5":
            return TuRBODesigner(self._policy, num_init=init_ax_default, num_trust_regions=5)
        elif designer_name == "dpp":
            return bt_designer(AcqDPP, init_sobol=1, acq_kwargs={"num_X_samples": default_num_X_samples})

        # MTV
        elif designer_name == "mtv":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={"num_X_samples": default_num_X_samples, "sample_type": "pss"},
            )
        elif designer_name == "pss":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "num_X_samples": default_num_X_samples,
                    "sample_type": "pss",
                },
            )
        elif designer_name == "mtv-sts":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "num_X_samples": default_num_X_samples,
                    "sample_type": "sts",
                    "num_refinements": 30,
                },
            )
        elif designer_name == "mtv-sts-t":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "num_X_samples": default_num_X_samples,
                    "sample_type": "sts",
                    "num_refinements": 30,
                    "x_max_type": "ts_meas",
                },
            )
        elif designer_name == "mtv-sts-m":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "num_X_samples": default_num_X_samples,
                    "sample_type": "sts",
                    "num_refinements": 30,
                    "x_max_type": "meas",
                },
            )
        elif designer_name == "sts":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                },
            )
        elif designer_name == "sts-t":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                    "x_max_type": "ts_meas",
                },
            )
        elif designer_name == "sts-m":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                    "x_max_type": "meas",
                },
            )

        # Long sobol init, sequential opt
        elif designer_name == "sobol_ucb":
            return bt_designer(qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1})
        elif designer_name == "sobol_ei":
            return bt_designer(qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None})
        elif designer_name == "sobol_gibbon":
            return bt_designer(qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None})

        raise NoSuchDesignerError(designer_name)
