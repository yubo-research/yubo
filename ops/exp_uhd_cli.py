#!/usr/bin/env python

import click

from common.im import im
from ops import exp_uhd_run


@click.group()
def _cli():
    pass


cli = _cli


@_cli.command(name="local", help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
def _local(config_toml: str) -> None:
    tomllib = im("tomllib")
    p = im("ops.exp_uhd_parse")
    try:
        cfg = p._load_toml_config(config_toml)
        p._validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = p._parse_cfg(cfg)
    exp_uhd_run.run_parsed_uhd_local(parsed)


_run_parsed = exp_uhd_run.run_parsed_uhd_local
_run_bszo = exp_uhd_run._run_bszo
_run_mezo = exp_uhd_run._run_mezo
_run_simple = exp_uhd_run._run_simple

local = _local


@_cli.command(
    name="modal",
    help="Run on Modal. Streams to stdout; optionally saves to --log-file.",
)
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.option(
    "--log-file",
    type=click.Path(dir_okay=False),
    default=None,
    help="Also save log to this local file.",
)
@click.option("--gpu", type=str, default="A100", help="Modal GPU type (e.g. T4, A10, A100, H100).")
def modal_cmd(config_toml: str, log_file: str | None, gpu: str) -> None:
    log_text = exp_uhd_run.uhd_config_toml_to_modal_log(
        config_toml,
        gpu,
        exp_uhd_parse=im("ops.exp_uhd_parse"),
        tomllib=im("tomllib"),
        modal_run=im("ops.modal_uhd").run,
    )

    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(log_text)
        click.echo(f"Log saved to {log_file}")


if __name__ == "__main__":
    cli()
