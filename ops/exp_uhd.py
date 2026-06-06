#!/usr/bin/env python

from common.im import im

_PARSE_EXPORTS = frozenset(
    {
        "_ALL_TOML_KEYS",
        "_BE_DEFAULTS",
        "_ENN_DEFAULTS",
        "_ER_DEFAULTS",
        "_OPTIONAL_TOML_KEYS",
        "_REQUIRED_TOML_KEYS",
        "_coerce_mapping_keys",
        "_normalize_key",
        "_parse_be_fields",
        "_parse_early_reject_fields",
        "_parse_enn_fields",
        "_parse_perturb",
    }
)

_EXP_UHD_CLI_EXPORTS = frozenset(
    {
        "cli",
        "local",
        "modal_cmd",
        "_cli",
        "_local",
    }
)

_EXP_UHD_RUN_EXPORTS = frozenset(
    {
        "_run_bszo",
        "_run_simple",
        "_run_mezo",
        "_run_parsed",
    }
)


def __getattr__(name: str):
    if name in _PARSE_EXPORTS:
        p = im("ops.exp_uhd_parse")
        return getattr(p, name)
    if name in _EXP_UHD_CLI_EXPORTS:
        return getattr(im("ops.exp_uhd_cli"), name)
    if name in _EXP_UHD_RUN_EXPORTS:
        run_mod = im("ops.exp_uhd_run")
        if name == "_run_parsed":
            return run_mod.run_parsed_uhd_local
        return getattr(run_mod, name)
    msg = f"module {__name__!r} has no attribute {name!r}"
    raise AttributeError(msg)


if __name__ == "__main__":
    im("ops.exp_uhd_cli").cli()
