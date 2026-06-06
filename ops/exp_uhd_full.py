#!/usr/bin/env python

import click

from common.im import im


@click.group()
def _cli():
    pass


cli = _cli


@_cli.command(name="local", help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
def _local(config_toml: str) -> None:
    im("ops.exp_uhd_run").run_local_from_toml(config_toml)


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
    tomllib = im("tomllib")
    p = im("ops.exp_uhd_parse")
    modal_run = im("ops.modal_uhd").run
    log_text = im("ops.exp_uhd_run").uhd_config_toml_to_modal_log(
        config_toml,
        gpu,
        exp_uhd_parse=p,
        tomllib=tomllib,
        modal_run=modal_run,
    )

    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(log_text)
        click.echo(f"Log saved to {log_file}")


if __name__ == "__main__":
    cli()
