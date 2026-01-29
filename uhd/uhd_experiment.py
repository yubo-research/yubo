#!/usr/bin/env python

import sys
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from uhd.exp_params import ExpParams


def run_experiment(params: "ExpParams", opt_name: str) -> None:
    from uhd.exp_params import ExpParams as _ExpParams

    assert isinstance(params, _ExpParams)
    import torch
    import torch.nn as nn

    from common.collector import Collector
    from uhd.uhd_collector import UHDCollector

    controllers = []
    collectors = []
    for i_rep in range(params.num_reps):
        this_seed = params.seed + i_rep
        torch.manual_seed(this_seed)
        controller = params.controller(this_seed)
        assert isinstance(controller, nn.Module)
        controller_name = controller.name
        base_collector = Collector()
        mapped_opt_name = "uhd_bo" if opt_name in ("random", "turbo") else opt_name
        collector = UHDCollector(
            name=controller_name, opt_name=mapped_opt_name, collector=base_collector
        )
        params.optimizer(controller, collector, params.num_rounds)
        controllers.append(controller)
        collectors.append(collector)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(
            f"Usage: {sys.argv[0]} target=tm_sphere|tm_ackley|tm_mnist opt_name=random|turbo|adamw num_rounds=<int> num_reps=<int> num_trail=<int>\n"
            f"  If target in (tm_sphere, tm_ackley): also require num_dim=<int> num_active=<int>\n"
            f"  If opt_name=random: require eps=<float>\n"
            f"  If opt_name=turbo: require use_tr=<true|false> num_raasp=<int or -1<=x<0 for proportion> num_candidates=<int> alpha=<float>"
        )
        sys.exit(1)
    from common.util import parse_kv

    args = parse_kv(sys.argv[1:])
    base_required = ["target", "opt_name", "num_rounds", "num_reps", "num_trail"]
    for k in base_required:
        if k not in args:
            print(f"Error: Missing {k}. Got keys: {list(args.keys())}")
            sys.exit(1)
    num_rounds = int(args["num_rounds"])
    num_reps = int(args["num_reps"])

    target = args["target"]
    assert target in ("tm_sphere", "tm_ackley", "tm_mnist"), target
    num_trail = int(args["num_trail"])
    assert num_trail > 0

    opt_name = args["opt_name"]
    assert opt_name in ("random", "turbo", "adamw"), opt_name

    from uhd.target_specs import TARGET_SPECS

    spec = TARGET_SPECS[target]

    if spec.requires_dims:
        if "num_dim" not in args or "num_active" not in args:
            print(
                "Error: num_dim and num_active are required when target is tm_sphere or tm_ackley"
            )
            sys.exit(1)
        num_dim = int(args["num_dim"])
        num_active = int(args["num_active"])

    if opt_name == "random":
        assert "eps" in args, "eps required when opt_name=random"
        eps = float(args["eps"])

        def get_conf():
            from uhd.mk_random_perturbations_config import (
                mk_random_perturbations_config,
            )

            conf = mk_random_perturbations_config(eps=eps, num_trail=num_trail)
            conf.metric = spec.make_bo_metric()
            return conf
    elif opt_name == "turbo":
        assert "use_tr" in args and "num_raasp" in args, (
            "use_tr and num_raasp required when opt_name=turbo"
        )
        assert "num_candidates" in args, "num_candidates required when opt_name=turbo"
        assert "alpha" in args, "alpha required when opt_name=turbo"
        use_tr_raw = str(args["use_tr"]).lower()
        if use_tr_raw in ("true", "1", "yes"):
            use_tr = True
        elif use_tr_raw in ("false", "0", "no"):
            use_tr = False
        else:
            raise AssertionError(f"Invalid use_tr={args['use_tr']}")
        num_raasp = float(args["num_raasp"])
        assert num_raasp >= -1.0
        num_candidates = int(args["num_candidates"])
        assert num_candidates > 0
        alpha = float(args["alpha"])

        def get_conf():
            from uhd.mk_turbo_config import mk_turbo_config

            conf = mk_turbo_config(
                use_tr=use_tr,
                num_raasp=num_raasp,
                num_trail=num_trail,
                num_candidates=num_candidates,
                alpha=alpha,
            )
            conf.metric = spec.make_bo_metric()
            return conf
    else:

        def get_conf():
            from uhd.opt_adamw import AdamWConfig

            metric = spec.make_adamw_metric()

            return AdamWConfig(
                lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-2, metric=metric
            )

    _Controller = spec.controller_type_factory()

    if spec.requires_dims:
        assert "num_dim" in locals() and "num_active" in locals()

        def _controller_factory(seed: int) -> Any:
            return _Controller(num_dim, num_active, seed)

        controller_factory = _controller_factory
        cd = num_dim
        ca = num_active
    else:

        def _controller_factory(seed: int) -> Any:
            return _Controller(seed)

        controller_factory = _controller_factory
        cd = 0
        ca = 0

    def optimizer_wrapper(controller: Any, collector: Any, num_rounds: int) -> Any:
        import torch
        import torch.nn as nn

        assert isinstance(controller, nn.Module)
        conf = get_conf()
        if opt_name in ("random", "turbo"):
            from uhd.opt_turbo import optimize_turbo

            y = optimize_turbo(controller, collector, num_rounds, conf)
            return torch.tensor(y)
        elif opt_name == "adamw":
            from uhd.opt_adamw import optimize_adamw

            loss = optimize_adamw(controller, collector, num_rounds, conf)
            return torch.tensor(loss)
        else:
            raise AssertionError(f"Invalid opt_name={opt_name}")

    from uhd.exp_params import ExpParams

    params = ExpParams(
        num_dim=cd,
        num_active=ca,
        num_rounds=num_rounds,
        num_reps=num_reps,
        seed=17,
        optimizer=optimizer_wrapper,
        controller=controller_factory,
    )
    run_experiment(params, opt_name)


def __getattr__(name: str) -> Any:
    if name == "ExpParams":
        from uhd.exp_params import ExpParams

        return ExpParams
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
