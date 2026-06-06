#!/usr/bin/env python

import click

from common.im import im


class _CallableCommand(click.Command):
    def __call__(self, *args, **kwargs):
        if args or kwargs:
            config_toml = args[0] if args else kwargs.get("config_toml")
            log_file = kwargs.get("log_file", None)
            gpu = kwargs.get("gpu", "A100")
            if len(args) >= 3:
                log_file = args[2]
            if len(args) >= 4:
                gpu = args[3]
            return self.callback(config_toml, log_file, gpu)
        return super().__call__(*args, **kwargs)


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
    cls=_CallableCommand,
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
    log_text = im("ops.exp_uhd_run").uhd_config_toml_to_modal_log(
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
