# Lazy imports to reduce dependency depth - imported when needed in methods


class NoSuchDesignerError(Exception):
    pass


class _PrefixContext:
    def __init__(self, policy, num_arms, bt, num_keep_val, default_num_X_samples, init_yubo_default):
        self.policy = policy
        self.num_arms = num_arms
        self.bt = bt
        self.num_keep_val = num_keep_val
        self.default_num_X_samples = default_num_X_samples
        self.init_yubo_default = init_yubo_default


def _dispatch_prefix(name, ctx):
    handlers = {
        "ts_sweep-": _h_ts_sweep,
        "rff-": _h_rff,
        "pss_sweep_kmcmc-": _h_pss_sweep_kmcmc,
        "pss_sweep_num_mcmc-": _h_pss_sweep_num_mcmc,
        "sts_sweep-": _h_sts_sweep,
        "turbo-enn-sweep-": _h_turbo_enn_sweep,
        "turbo-enn-fit-": _h_turbo_enn_fit,
        "turbo-enn-f": _h_turbo_enn_f,
        "morbo-enn-fit-": _h_morbo_enn_fit,
        "sts-ar-": _h_sts_ar,
    }
    for prefix, handler in handlers.items():
        if name.startswith(prefix):
            return handler(name, ctx)
    return None


def _h_ts_sweep(name, ctx):
    from acq.acq_ts import AcqTS
    num_candidates = int(name.split("-")[1])
    return ctx.bt(AcqTS, acq_kwargs={"sampler": "lanczos", "num_candidates": num_candidates})


def _h_rff(name, ctx):
    from acq.acq_ts import AcqTS
    num_candidates = int(name.split("-")[1])
    return ctx.bt(AcqTS, acq_kwargs={"sampler": "rff", "num_candidates": num_candidates})


def _h_pss_sweep_kmcmc(name, ctx):
    from acq.acq_mtv import AcqMTV
    k_mcmc = int(name.split("-")[1])
    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "num_X_samples": ctx.default_num_X_samples, "sample_type": "pss", "k_mcmc": k_mcmc})


def _h_pss_sweep_num_mcmc(name, ctx):
    from acq.acq_mtv import AcqMTV
    num_mcmc = int(name.split("-")[1])
    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "num_X_samples": ctx.default_num_X_samples, "sample_type": "pss", "k_mcmc": None, "num_mcmc": num_mcmc})


def _h_sts_sweep(name, ctx):
    from acq.acq_mtv import AcqMTV
    num_refinements = int(name.split("-")[1])
    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": ctx.default_num_X_samples, "num_refinements": num_refinements})


def _h_turbo_enn_sweep(name, ctx):
    from .turbo_enn_designer import TurboENNDesigner
    k = int(name.split("-")[-1])
    return TurboENNDesigner(ctx.policy, turbo_mode="turbo-enn", k=k, num_keep=None, num_fit_samples=None, acq_type="pareto")


def _h_turbo_enn_fit(name, ctx):
    from .turbo_enn_designer import TurboENNDesigner
    suffix = name[len("turbo-enn-fit-"):]
    parts = suffix.split("-")
    kind = parts[0]
    acq_type_map = {"p": "pareto", "ts": "thompson", "ucb": "ucb"}
    acq_type = acq_type_map.get(kind)
    if acq_type is None or len(parts) > 1:
        raise NoSuchDesignerError(name)
    return TurboENNDesigner(ctx.policy, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, num_fit_samples=100, num_fit_candidates=100, acq_type=acq_type, tr_type=None)


def _h_turbo_enn_f(name, ctx):
    from .turbo_enn_designer import TurboENNDesigner
    def num_candidates(num_dim, num_arms):
        return 100 * num_arms
    return TurboENNDesigner(ctx.policy, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, num_fit_samples=100, num_fit_candidates=100, acq_type="ucb", num_candidates=num_candidates, candidate_rv="uniform")


def _h_morbo_enn_fit(name, ctx):
    from .turbo_enn_designer import TurboENNDesigner
    suffix = name[len("morbo-enn-fit-"):]
    parts = suffix.split("-")
    kind = parts[0]
    acq_type_map = {"p": "pareto", "ts": "thompson", "ucb": "ucb"}
    acq_type = acq_type_map.get(kind)
    if acq_type is None:
        raise NoSuchDesignerError(name)
    return TurboENNDesigner(ctx.policy, turbo_mode="turbo-enn", k=10, num_keep=ctx.num_keep_val, num_fit_samples=100, num_fit_candidates=100 * ctx.num_arms, acq_type=acq_type, tr_type="morbo")


def _h_sts_ar(name, ctx):
    from acq.acq_mtv import AcqMTV
    num_acc_rej = int(name.split("-")[-1])
    return ctx.bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": ctx.default_num_X_samples, "num_refinements": 0, "num_acc_rej": num_acc_rej})


def _parse_options(designer_name):
    if ":" in designer_name:
        designer_name, options_str = designer_name.split(":")
        options = options_str.split("-")
    else:
        options = []

    num_keep = None
    keep_style = None
    model_spec = None
    sample_around_best = False

    keep_style_map = {"s": "some", "b": "best", "r": "random", "t": "trailing", "p": "lap"}

    for option in options:
        if option[0] == "K":
            keep_style = keep_style_map.get(option[1])
            assert keep_style is not None, option
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

    return designer_name, num_keep, keep_style, model_spec, sample_around_best


class Designers:
    def __init__(self, policy, num_arms):
        self._policy = policy
        self._num_arms = num_arms

    def is_valid(self, designer_name):
        return designer_name in self._designers

    def _bt_designer(
        self,
        acq_factory,
        acq_kwargs=None,
        init_sobol=1,
        opt_sequential=False,
        num_restarts=10,
        raw_samples=10,
        start_at_max=False,
        num_keep=None,
        keep_style=None,
        model_spec=None,
        sample_around_best=False,
    ):
        from .bt_designer import BTDesigner
        return BTDesigner(
            self._policy,
            acq_factory,
            acq_kwargs=acq_kwargs,
            num_keep=num_keep,
            keep_style=keep_style,
            model_spec=model_spec,
            init_sobol=init_sobol,
            opt_sequential=opt_sequential,
            optimizer_options={
                "batch_limit": 10,
                "maxiter": 1000,
                "sample_around_best": sample_around_best,
            },
            num_restarts=num_restarts,
            raw_samples=raw_samples,
            start_at_max=start_at_max,
        )

    def _get_simple_designers(self, opts):
        # Lazy imports to reduce module dependency depth
        from botorch.acquisition.logei import qLogNoisyExpectedImprovement
        from botorch.acquisition.max_value_entropy_search import qLowerBoundMaxValueEntropy
        from botorch.acquisition.monte_carlo import qNoisyExpectedImprovement, qSimpleRegret, qUpperConfidenceBound
        from botorch.acquisition.thompson_sampling import PathwiseThompsonSampling

        from acq.acq_dpp import AcqDPP
        from acq.acq_min_dist import AcqMinDist
        from acq.acq_mtv import AcqMTV
        from acq.acq_sobol import AcqSobol
        from acq.acq_ts import AcqTS
        from acq.acq_var import AcqVar

        from .ax_designer import AxDesigner
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
        from .vecchia_designer import VecchiaDesigner

        num_keep, keep_style, model_spec, sample_around_best = opts
        init_ax_default = max(5, 2 * self._policy.num_params())
        init_yubo_default = self._num_arms
        default_num_X_samples = max(64, 10 * self._num_arms)

        def bt(*args, **kw):
            return self._bt_designer(
                *args, num_keep=num_keep, keep_style=keep_style, model_spec=model_spec,
                sample_around_best=sample_around_best, **kw
            )

        num_keep_val = num_keep if keep_style == "trailing" else None

        return {
            "cma": lambda: CMAESDesigner(self._policy),
            "optuna": lambda: OptunaDesigner(self._policy),
            "ax": lambda: AxDesigner(self._policy),
            "maximin": lambda: bt(lambda m: AcqMinDist(m, toroidal=False)),
            "maximin-toroidal": lambda: bt(lambda m: AcqMinDist(m, toroidal=True)),
            "variance": lambda: bt(AcqVar),
            "random": lambda: RandomDesigner(self._policy),
            "sobol": lambda: SobolDesigner(self._policy),
            "lhd": lambda: LHDDesigner(self._policy),
            "btsobol": lambda: bt(AcqSobol),
            "center": lambda: CenterDesigner(self._policy),
            "sr": lambda: bt(qSimpleRegret),
            "ts": lambda: bt(AcqTS, acq_kwargs={"sampler": "cholesky", "num_candidates": 1000}),
            "ts-10000": lambda: bt(AcqTS, acq_kwargs={"sampler": "lanczos", "num_candidates": 10000}),
            "ucb": lambda: bt(qUpperConfidenceBound, acq_kwargs={"beta": 1}),
            "ei": lambda: bt(qNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "lei": lambda: bt(qLogNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}),
            "lei-m": lambda: bt(qLogNoisyExpectedImprovement, acq_kwargs={"X_baseline": None}, start_at_max=True),
            "gibbon": lambda: bt(qLowerBoundMaxValueEntropy, opt_sequential=True, acq_kwargs={"candidate_set": None}),
            "turbo-1": lambda: TuRBORefDesigner(self._policy, num_init=init_yubo_default, ard=True),
            "turbo-1-iso": lambda: TuRBORefDesigner(self._policy, num_init=init_yubo_default, ard=False),
            "turbo-0": lambda: TuRBORefDesigner(self._policy, num_init=init_yubo_default, surrogate_type="none", ard=True),
            "turbo-enn": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-enn", k=10, num_keep=num_keep_val),
            "turbo-enn-p": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-enn", k=10, num_keep=num_keep_val, num_fit_samples=None, acq_type="pareto"),
            "turbo-zero": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-zero"),
            "turbo-one": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-one", num_init=init_yubo_default),
            "lhd_only": lambda: TurboENNDesigner(self._policy, turbo_mode="lhd-only"),
            "morbo-zero": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-zero", tr_type="morbo"),
            "morbo-one": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-one", num_init=init_yubo_default, tr_type="morbo"),
            "morbo-enn": lambda: TurboENNDesigner(self._policy, turbo_mode="turbo-enn", k=10, num_keep=num_keep_val, tr_type="morbo"),
            "dpp": lambda: bt(AcqDPP, init_sobol=1, acq_kwargs={"num_X_samples": default_num_X_samples}),
            "vecchia": lambda: VecchiaDesigner(self._policy, num_candidates_per_arm=default_num_X_samples),
            "mtv": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"num_X_samples": default_num_X_samples, "sample_type": "pss"}),
            "pss": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "num_X_samples": default_num_X_samples, "sample_type": "pss"}),
            "mtv-sts": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"num_X_samples": default_num_X_samples, "sample_type": "sts", "num_refinements": 30}),
            "mtv-mts": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"num_X_samples": default_num_X_samples, "sample_type": "mts", "num_refinements": 30}),
            "mtv-sts2": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"num_X_samples": default_num_X_samples, "sample_type": "sts2", "num_refinements": 30}),
            "mtv-sts-t": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"num_X_samples": default_num_X_samples, "sample_type": "sts", "num_refinements": 30, "x_max_type": "ts_meas"}),
            "sts": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": default_num_X_samples, "num_refinements": 30}),
            "sts-ch": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "ts_chain": True, "sample_type": "sts", "num_X_samples": default_num_X_samples, "num_refinements": 30}),
            "sts-ns": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": default_num_X_samples, "num_refinements": 30, "no_stagger": True}),
            "sts-ui": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": default_num_X_samples, "num_refinements": 30, "no_stagger": False, "x_max_type": "rand"}),
            "sts-t": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": default_num_X_samples, "num_refinements": 30, "x_max_type": "ts_meas"}),
            "sts-m": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts", "num_X_samples": default_num_X_samples, "num_refinements": 30, "x_max_type": "meas"}),
            "sts2": lambda: bt(AcqMTV, init_sobol=0, acq_kwargs={"ts_only": True, "sample_type": "sts2", "num_X_samples": default_num_X_samples, "num_refinements": 30}),
            "path": lambda: bt(PathwiseThompsonSampling, init_sobol=init_yubo_default),
            "path-b": lambda: bt(PathwiseThompsonSampling, init_sobol=init_yubo_default, num_restarts=20, raw_samples=100),
            "path-m": lambda: bt(PathwiseThompsonSampling, init_sobol=init_yubo_default, start_at_max=True),
            "mcmcbo": lambda: MCMCBODesigner(self._policy, num_init=init_yubo_default),
            "mts": lambda: MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="find"),
            "mts-stagger": lambda: MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="find", use_stagger=True),
            "mts-ts": lambda: MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="ts"),
            "mts-meas": lambda: MTSDesigner(self._policy, keep_style=keep_style, num_keep=num_keep, init_style="meas"),
            "sobol_ucb": lambda: bt(qUpperConfidenceBound, init_sobol=init_ax_default, acq_kwargs={"beta": 1}),
            "sobol_ei": lambda: bt(qNoisyExpectedImprovement, init_sobol=init_ax_default, acq_kwargs={"X_baseline": None}),
            "sobol_gibbon": lambda: bt(qLowerBoundMaxValueEntropy, init_sobol=init_ax_default, acq_kwargs={"candidate_set": None}),
        }

    def _create_prefix_designer(self, designer_name, opts):
        num_keep, keep_style, model_spec, sample_around_best = opts
        init_yubo_default = self._num_arms
        default_num_X_samples = max(64, 10 * self._num_arms)

        def bt(*args, **kw):
            return self._bt_designer(
                *args, num_keep=num_keep, keep_style=keep_style, model_spec=model_spec,
                sample_around_best=sample_around_best, **kw
            )

        num_keep_val = num_keep if keep_style == "trailing" else None
        ctx = _PrefixContext(self._policy, self._num_arms, bt, num_keep_val, default_num_X_samples, init_yubo_default)
        return _dispatch_prefix(designer_name, ctx)

    def create(self, designer_name):
        designer_name, num_keep, keep_style, model_spec, sample_around_best = _parse_options(designer_name)
        opts = (num_keep, keep_style, model_spec, sample_around_best)

        simple_designers = self._get_simple_designers(opts)
        if designer_name in simple_designers:
            return simple_designers[designer_name]()

        prefix_result = self._create_prefix_designer(designer_name, opts)
        if prefix_result is not None:
            return prefix_result

        raise NoSuchDesignerError(designer_name)
