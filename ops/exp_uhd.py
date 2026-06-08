#!/usr/bin/env python3

import sys
from pathlib import Path


def _configure_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if callable(reconfigure):
            reconfigure(line_buffering=True)


def _ensure_repo_root_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(repo_root))


_configure_stdio()
_ensure_repo_root_on_path()

from common.im import im  # noqa: E402

_PARSE_EXPORTS = frozenset(
    {
        "_ALL_TOML_KEYS",
        "_BE_DEFAULTS",
        "_ENN_DEFAULTS",
        "_ER_DEFAULTS",
        "_OPTIONAL_TOML_KEYS",
        "_REQUIRED_TOML_KEYS",
        "_coerce_mapping_keys",
        "_load_toml_config",
        "_normalize_key",
        "_parse_be_fields",
        "_parse_early_reject_fields",
        "_parse_enn_fields",
        "_parse_perturb",
        "_parse_perturb_spec",
        "_parse_override_value",
        "_parse_overrides",
        "_parse_cfg",
        "_parse_budget_fields",
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
        "_run_unified",
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
