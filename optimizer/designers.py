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

# from acq.acq_tsroots import AcqTSRoots
from acq.acq_var import AcqVar
from acq.turbo_yubo.turbo_yubo_config import TurboYUBOConfig
from acq.turbo_yubo.ty_model_factory import TurboYUBONOOPModel
from acq.turbo_yubo.ty_signal_tr import ty_signal_tr_factory_factory

from .ax_designer import AxDesigner
from .bt_designer import BTDesigner
from .center_designer import CenterDesigner
from .cma_designer import CMAESDesigner
from .lhd_designer import LHDDesigner
from .mcmc_bo_designer import MCMCBODesigner
from .mts_designer import MTSDesigner
from .optuna_designer import OptunaDesigner
from .random_designer import RandomDesigner
from .sobol_designer import SobolDesigner
from .turbo_enn_designer import TurboENNDesigner
from .turbo_ref_designer import TuRBORefDesigner
from .turbo_yubo_designer import TurboYUBODesigner
from .vecchia_designer import VecchiaDesigner


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
        model_spec = None
        sample_around_best = False
        for option in options:
            if option[0] == "K":
                if option[1] == "s":
                    keep_style = "some"
                elif option[1] == "b":
                    keep_style = "best"
                elif option[1] == "r":
                    keep_style = "random"
                elif option[1] == "t":
                    keep_style = "trailing"
                elif option[1] == "p":
                    keep_style = "lap"
                else:
                    assert False, option
                num_keep = int(option[2:])
                print(f"OPTION: num_keep = {num_keep} keep_style = {keep_style}")
            elif option[0] == "M":
                model_spec = option[1:]
                print(f"OPTION model_spec = {option}")
            elif option[0] == "O":
                if option[1:] == "sab":
                    sample_around_best = True
            else:
                assert False, ("Unknown option", option)

        def bt_designer(acq_factory, acq_kwargs=None, init_sobol=1, opt_sequential=False, num_restarts=10, raw_samples=10, start_at_max=False):
            return BTDesigner(
                self._policy,
                acq_factory,
                acq_kwargs=acq_kwargs,
                num_keep=num_keep,
                keep_style=keep_style,
                model_spec=model_spec,
                init_sobol=init_sobol,
                opt_sequential=opt_sequential,
                optimizer_options={"batch_limit": 10, "maxiter": 1000, "sample_around_best": sample_around_best},
                num_restarts=10,
                raw_samples=10,
                start_at_max=start_at_max,
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
        elif designer_name == "lhd":
            return LHDDesigner(self._policy)
        elif designer_name == "btsobol":
            return bt_designer(AcqSobol)
        elif designer_name == "center":
            return CenterDesigner(self._policy)

        # All exploitation
        elif designer_name == "sr":
            return bt_designer(qSimpleRegret)

        # Various methods, first batch is Sobol
        elif designer_name == "ts":
            return bt_designer(
                AcqTS,
                acq_kwargs={
                    "sampler": "cholesky",
                    "num_candidates": 1000,
                },
            )
        elif designer_name == "ts-10000":
            return bt_designer(
                AcqTS,
                acq_kwargs={
                    "sampler": "lanczos",
                    "num_candidates": 10000,
                },
            )
        elif designer_name.startswith("ts_sweep"):
            num_candidates = int(designer_name.split("-")[1])
            return bt_designer(
                AcqTS,
                acq_kwargs={
                    "sampler": "lanczos",
                    "num_candidates": num_candidates,
                },
            )
        elif designer_name.startswith("rff-"):
            num_candidates = int(designer_name.split("-")[1])
            return bt_designer(
                AcqTS,
                acq_kwargs={
                    "sampler": "rff",
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
        elif designer_name == "lei-m":
            return bt_designer(qLogNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}, start_at_max=True)
        elif designer_name == "gibbon":
            return bt_designer(qLowerBoundMaxValueEntropy, opt_sequential=True, acq_kwargs={"candidate_set": None})
        elif designer_name == "turbo-1":
            return TuRBORefDesigner(self._policy, num_init=init_yubo_default, ard=True)
        elif designer_name == "turbo-1-iso":
            return TuRBORefDesigner(self._policy, num_init=init_yubo_default, ard=False)
        elif designer_name == "turbo-0":
            return TuRBORefDesigner(self._policy, num_init=init_yubo_default, surrogate_type="none", ard=True)
        elif designer_name == "turbo-enn":
            num_keep_val = num_keep if keep_style == "trailing" else None
            return TurboENNDesigner(self._policy, turbo_mode="turbo-enn", k=10, num_keep=num_keep_val)
        elif designer_name == "turbo-enn-p":
            num_keep_val = num_keep if keep_style == "trailing" else None
            return TurboENNDesigner(
                self._policy,
                turbo_mode="turbo-enn",
                k=10,
                num_keep=num_keep_val,
                num_fit_samples=None,
                acq_type="pareto",
            )
        elif designer_name.startswith("turbo-enn-fit-"):
            num_keep_val = num_keep if keep_style == "trailing" else None
            suffix = designer_name[len("turbo-enn-fit-") :]
            parts = suffix.split("-")
            kind = parts[0]
            variant = parts[1] if len(parts) > 1 else None

            if kind == "p":
                acq_type = "pareto"
            elif kind == "ts":
                acq_type = "thompson"
            elif kind == "ucb":
                acq_type = "ucb"
            else:
                raise NoSuchDesignerError(designer_name)

            if variant is None:
                tr_type = None
            else:
                raise NoSuchDesignerError(designer_name)

            return TurboENNDesigner(
                self._policy,
                turbo_mode="turbo-enn",
                k=10,
                num_keep=num_keep_val,
                num_fit_samples=100,
                num_fit_candidates=100,
                acq_type=acq_type,
                tr_type=tr_type,
            )
        elif designer_name == "turbo-enn-f":
            num_keep_val = num_keep if keep_style == "trailing" else None

            def num_candidates(num_dim, num_arms):
                return min(5000, 100 * num_arms)

            return TurboENNDesigner(
                self._policy,
                turbo_mode="turbo-enn",
                k=10,
                num_keep=num_keep_val,
                num_fit_samples=100,
                num_fit_candidates=100,
                acq_type="ucb",
                num_candidates=num_candidates,
                candidate_rv="uniform",
            )

        elif designer_name == "turbo-zero":
            return TurboENNDesigner(self._policy, turbo_mode="turbo-zero")
        elif designer_name == "turbo-zero-f":
            return TurboENNDesigner(
                self._policy,
                turbo_mode="turbo-zero",
                num_candidates=min(5000, 100 * self._num_arms),
                candidate_rv="uniform",
            )
        elif designer_name == "turbo-one":
            return TurboENNDesigner(self._policy, turbo_mode="turbo-one", num_init=init_yubo_default)
        elif designer_name == "turbo-one-f":
            return TurboENNDesigner(
                self._policy,
                turbo_mode="turbo-one",
                num_init=init_yubo_default,
                num_candidates=min(5000, 100 * self._num_arms),
                candidate_rv="uniform",
            )

        # MORBO variants (multi-objective)
        elif designer_name == "morbo-zero":
            return TurboENNDesigner(self._policy, turbo_mode="turbo-zero", tr_type="morbo")
        elif designer_name == "morbo-one":
            return TurboENNDesigner(self._policy, turbo_mode="turbo-one", num_init=init_yubo_default, tr_type="morbo")
        elif designer_name == "morbo-enn":
            num_keep_val = num_keep if keep_style == "trailing" else None
            return TurboENNDesigner(self._policy, turbo_mode="turbo-enn", k=10, num_keep=num_keep_val, tr_type="morbo")
        elif designer_name.startswith("morbo-enn-fit-"):
            num_keep_val = num_keep if keep_style == "trailing" else None
            suffix = designer_name[len("morbo-enn-fit-") :]
            parts = suffix.split("-")
            kind = parts[0]

            if kind == "p":
                acq_type = "pareto"
            elif kind == "ts":
                acq_type = "thompson"
            elif kind == "ucb":
                acq_type = "ucb"
            else:
                raise NoSuchDesignerError(designer_name)

            return TurboENNDesigner(
                self._policy,
                turbo_mode="turbo-enn",
                k=10,
                num_keep=num_keep_val,
                num_fit_samples=100,
                num_fit_candidates=100 * self._num_arms,
                acq_type=acq_type,
                tr_type="morbo",
            )
        # elif designer_name.startswith("turbo-enn-"):
        # k = int(designer_name.split("-")[-1])
        # return TuRBORefDesigner(self._policy, num_init=init_yubo_default, surrogate_type=designer_name[6:], ard=True)

        elif designer_name == "dpp":
            return bt_designer(AcqDPP, init_sobol=1, acq_kwargs={"num_X_samples": default_num_X_samples})
        elif designer_name == "vecchia":
            return VecchiaDesigner(self._policy, num_candidates_per_arm=default_num_X_samples)

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
        elif designer_name == "mtv-mts":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "num_X_samples": default_num_X_samples,
                    "sample_type": "mts",
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
        elif designer_name == "sts-ch":
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "ts_chain": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 30,
                },
            )
        elif designer_name.startswith("sts-ar-"):
            num_acc_rej = int(designer_name.split("-")[-1])
            return bt_designer(
                AcqMTV,
                init_sobol=0,
                acq_kwargs={
                    "ts_only": True,
                    "sample_type": "sts",
                    "num_X_samples": default_num_X_samples,
                    "num_refinements": 0,
                    "num_acc_rej": num_acc_rej,
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
        elif designer_name == "path-b":
            return bt_designer(
                PathwiseThompsonSampling,
                init_sobol=init_yubo_default,
                num_restarts=20,
                raw_samples=100,
            )
        elif designer_name == "path-m":
            return bt_designer(
                PathwiseThompsonSampling,
                init_sobol=init_yubo_default,
                start_at_max=True,
            )

        # elif designer_name == "tsroots":
        #     return bt_designer(
        #         AcqTSRoots,
        #         init_sobol=init_yubo_default,
        #     )
        elif designer_name == "mcmcbo":
            return MCMCBODesigner(
                self._policy,
                num_init=init_yubo_default,
            )

        elif designer_name == "mts":
            return MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="find")
        elif designer_name == "mts-stagger":
            return MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="find", use_stagger=True)
        elif designer_name == "mts-ts":
            return MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="ts")
        elif designer_name == "mts-meas":
            return MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="meas")

        elif designer_name == "turbo-yubo":
            return TurboYUBODesigner(self._policy, num_keep=num_keep, keep_style=keep_style, config=TurboYUBOConfig())

        elif designer_name == "turbo-yubo-gumbel":
            return TurboYUBODesigner(
                self._policy,
                num_keep=num_keep,
                keep_style=keep_style,
                config=TurboYUBOConfig(
                    trust_region_manager=ty_signal_tr_factory_factory(use_gumbel=True),
                ),
            )
        elif designer_name == "tyg-0":
            return TurboYUBODesigner(
                self._policy,
                num_keep=num_keep,
                keep_style=keep_style,
                config=TurboYUBOConfig(
                    trust_region_manager=ty_signal_tr_factory_factory(use_gumbel=True),
                    model_factory=TurboYUBONOOPModel,
                ),
            )

        # Long sobol init, sequential opt
        elif designer_name == "sobol_ucb":
            return bt_designer(qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1})
        elif designer_name == "sobol_ei":
            return bt_designer(qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None})
        elif designer_name == "sobol_gibbon":
            return bt_designer(qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None})

        raise NoSuchDesignerError(designer_name)
