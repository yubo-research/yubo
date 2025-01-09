from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.acquisition.max_value_entropy_search import (
    qLowerBoundMaxValueEntropy,
)
from botorch.acquisition.monte_carlo import (
    qNoisyExpectedImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

from acq.acq_dpp import AcqDPP
from acq.acq_min_dist import AcqMinDist
from acq.acq_mtv import AcqMTV
from acq.acq_sobol import AcqSobol
from acq.acq_ts import AcqTS
from acq.acq_tsroots import AcqTSRoots
from acq.acq_var import AcqVar

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .cma_designer import CMAESDesigner
from .mcmc_bo_designer import MCMCBODesigner
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
        init_yubo_default = self._num_arms
        default_num_X_samples = max(64, 10 * self._num_arms)
        # default_num_Y_samples = 512

        # mtv:k100
        if ":" in designer_name:
            designer_name, options = designer_name.split(":")
            options = options.split("-")
        else:
            options = []

        num_keep = None
        keep_style = None
        model_type = None
        for option in options:
            if option[0] == "k":
                if option[1] == "s":
                    keep_style = "some"
                elif option[1] == "b":
                    keep_style = "best"
                elif option[1] == "r":
                    keep_style = "random"
                else:
                    assert False, option
                num_keep = int(option[2:])
                print(f"OPTION: num_keep = {num_keep} keep_style = {keep_style}")
            elif option == "vanilla":
                model_type = "vanilla"
                print("OPTION model_type = vanilla")
            elif option == "dumbo":
                model_type = "dumbo"
                print("OPTION model_type = dumbo")
            elif option == "rdumbo":
                model_type = "rdumbo"
                print("OPTION model_type = rdumbo")
            else:
                assert False, ("Unknown option", option)

        def bt_designer(acq_factory, acq_kwargs=None, init_sobol=1, opt_sequential=False):
            return BTDesigner(
                self._policy,
                acq_factory,
                acq_kwargs=acq_kwargs,
                num_keep=num_keep,
                keep_style=keep_style,
                model_type=model_type,
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
        elif designer_name == "btsobol":
            return bt_designer(AcqSobol)

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
        elif designer_name.startswith("pss_sweep_kmcmc"):
            k_mcmc = int(designer_name.split("-")[1])
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={"ts_only": True, "num_X_samples": default_num_X_samples, "sample_type": "pss", "k_mcmc": k_mcmc},
            )
        elif designer_name.startswith("pss_sweep_num_mcmc"):
            num_mcmc = int(designer_name.split("-")[1])
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={"ts_only": True, "num_X_samples": default_num_X_samples, "sample_type": "pss", "k_mcmc": None, "num_mcmc": num_mcmc},
            )
        elif designer_name.startswith("sts_sweep"):
            num_refinements = int(designer_name.split("-")[1])
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": num_refinements,
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
            return TuRBODesigner(self._policy, num_init=init_yubo_default)
        elif designer_name == "turbo-5":
            return TuRBODesigner(self._policy, num_init=init_yubo_default, num_trust_regions=5)
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
        elif designer_name == "mtv-sts2":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "num_X_samples": default_num_X_samples,
                    "sample_type": "sts2",
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
        elif designer_name == "sts-ns":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                    "no_stagger": True,
                },
            )
        elif designer_name == "sts-ui":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                    "no_stagger": False,
                    "x_max_type": "rand",
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
        elif designer_name == "sts2":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts2",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                },
            )
        elif designer_name == "path":
            return bt_designer(
                PathwiseThompsonSampling,
                init_sobol=init_yubo_default,
            )
        elif designer_name == "tsroots":
            return bt_designer(
                AcqTSRoots,
                init_sobol=init_yubo_default,
            )
        elif designer_name == "mcmcbo":
            return MCMCBODesigner(
                self._policy,
                num_init=init_yubo_default,
            )

        # Long sobol init, sequential opt
        elif designer_name == "sobol_ucb":
            return bt_designer(qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1})
        elif designer_name == "sobol_ei":
            return bt_designer(qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None})
        elif designer_name == "sobol_gibbon":
            return bt_designer(qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None})

        raise NoSuchDesignerError(designer_name)
