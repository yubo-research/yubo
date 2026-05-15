from experiments.batch_preps_core import (
    TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS,
    prep_args_1,
)

_TIMING_SWEEP_ENV_TAGS = (
    ("tlunar:fn", 50, 30, 1, 50, None, True),
    ("tlunar", 1, 10000, 1, 1, 30, False),
    ("push:fn", 50, 300, 1, 50, None, True),
    ("push", 1, 10000, 1, 1, 30, False),
    ("hop:fn", 50, 1000, 30, 10, None, True),
    ("hop", 1, 10000, 30, None, 30, False),
    ("bw-heur:fn", 50, 100, 30, 30, None, True),
    ("bw-heur", 1, 10000, 30, None, 30, False),
    ("dna:fn", 1, 1000, 30, 10, None, True),
    ("dna", 1, 1000, 30, None, 10, False),
)

# Full-matrix cells in timing_sweep.md with numeric cumulative proposal (not FAIL, not blank).
# Rows turbo-enn-p, turbo-one, cma, optuna are omitted entirely. Update when regenerating the doc.
_RUN_OTHERS_NONFAIL_CELLS = frozenset(
    {
        ("ucb", "tlunar:fn"),
        ("ucb", "bw-heur:fn"),
        ("smac", "tlunar:fn"),
        ("smac", "tlunar"),
        ("smac", "push:fn"),
        ("smac", "push"),
        ("smac", "hop:fn"),
        ("smac", "bw-heur:fn"),
        ("smac", "bw-heur"),
        ("smac", "dna:fn"),
        ("smac", "dna"),
        ("dngo", "tlunar:fn"),
        ("dngo", "push:fn"),
        ("dngo", "bw-heur:fn"),
        ("dngo", "dna:fn"),
        ("dngo", "dna"),
        ("vecchia", "tlunar:fn"),
        ("vecchia", "push:fn"),
        ("vecchia", "hop:fn"),
        ("vecchia", "bw-heur:fn"),
        ("vecchia", "dna:fn"),
        ("vecchia", "dna"),
    }
)


def prep_timing_sweep(results_dir):
    exp_dir = "exp_ennbo_timing_sweep"

    opts = [
        # "optuna",
        # "cma",
        # "turbo-zero",
        # "turbo-one",
        # "turbo-enn-fit-ucb",
        # "turbo-enn-p",
        "ucb",
        "lei",
        "smac",
        "dngo",
        "vecchia",
        "ucb:Msparse",
        # "turbo-one-ucb",
        # "turbo-one-nds",
    ]
    env_tags = _TIMING_SWEEP_ENV_TAGS

    for x in env_tags:
        assert len(x) == 7, (len(x), x)

    cmds = []
    for opt in opts:
        for (
            env_tag,
            num_arms,
            num_rounds,
            _,
            num_denoise,
            num_denoise_passive,
            fn,
        ) in env_tags:
            if fn:
                assert env_tag.endswith(":fn"), (env_tag, fn)
            else:
                assert not env_tag.endswith(":fn"), (env_tag, fn)
            cfg = prep_args_1(
                results_dir,
                exp_dir=exp_dir,
                problem=env_tag,
                opt=opt,
                num_arms=num_arms,
                num_replications=1,
                num_rounds=num_rounds,
                noise=None,
                num_denoise=num_denoise,
                num_denoise_passive=num_denoise_passive,
            )
            cfg.max_total_seconds = None
            cfg.max_proposal_seconds = float(TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS)
            cmds.append(cfg)

    return cmds


def prep_run_others(results_dir):
    """Configs for timing-sweep cells that are not FAIL (see timing_sweep.md), new exp_dir.

    Submit with modal_batches: batch_tag ``run_others`` (or ``prep_run_others``).
    """
    exp_dir = "exp_ennbo_run_others"
    opts = [
        "ucb",
        "lei",
        "smac",
        "dngo",
        "vecchia",
        "ucb:Msparse",
    ]
    cmds = []
    for opt in opts:
        for (
            env_tag,
            num_arms,
            num_rounds,
            _,
            num_denoise,
            num_denoise_passive,
            fn,
        ) in _TIMING_SWEEP_ENV_TAGS:
            if (opt, env_tag) not in _RUN_OTHERS_NONFAIL_CELLS:
                continue
            if fn:
                assert env_tag.endswith(":fn"), (env_tag, fn)
            else:
                assert not env_tag.endswith(":fn"), (env_tag, fn)
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem=env_tag,
                    opt=opt,
                    num_arms=num_arms,
                    num_replications=1,
                    num_rounds=num_rounds,
                    noise=None,
                    num_denoise=num_denoise,
                    num_denoise_passive=num_denoise_passive,
                )
            )
    return cmds


def prep_turbo_abl(results_dir):
    """TuRBO ablation: turbo-one-nds & turbo-one-ucb across all 10 env_tags."""
    exp_dir = "exp_ennbo_turbo_abl"
    opts = [
        "turbo-one-nds",
        "turbo-one-ucb",
    ]
    cmds = []
    for opt in opts:
        for (
            env_tag,
            num_arms,
            num_rounds,
            _,
            num_denoise,
            num_denoise_passive,
            fn,
        ) in _TIMING_SWEEP_ENV_TAGS:
            if fn:
                assert env_tag.endswith(":fn"), (env_tag, fn)
            else:
                assert not env_tag.endswith(":fn"), (env_tag, fn)
            cmds.append(
                prep_args_1(
                    results_dir,
                    exp_dir=exp_dir,
                    problem=env_tag,
                    opt=opt,
                    num_arms=num_arms,
                    num_replications=1,
                    num_rounds=num_rounds,
                    noise=None,
                    num_denoise=num_denoise,
                    num_denoise_passive=num_denoise_passive,
                )
            )
    return cmds
