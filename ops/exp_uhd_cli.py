#!/usr/bin/env python

import click

import common.im as common_im


def _im(name: str):
    return common_im.im(name)


def _run_mod():
    return _im("ops.exp_uhd_run")


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


def _local_options(fn):
    fn = click.option("--workers", type=int, default=1, help="Parallel workers when [uhd].num_reps > 1.")(fn)
    fn = click.option(
        "--results-dir",
        type=str,
        default="results/uhd",
        help="Results directory when [uhd].num_reps > 1.",
    )(fn)
    fn = click.option(
        "-o",
        "--opt",
        "overrides",
        multiple=True,
        help="Override config key: --opt key=value (e.g. --opt total_timesteps=1000).",
    )(fn)
    fn = click.option(
        "--dashboard",
        is_flag=True,
        help="Use the LLM dashboard. Only valid for llm:* envs.",
    )(fn)
    fn = click.option(
        "--child-process",
        is_flag=True,
        hidden=True,
        help="Internal flag used by the dashboard parent process to run the experiment in a clean child process.",
    )(fn)
    return fn


@_cli.command(name="local", help="Run locally (single process) from a config TOML.")
@click.argument("config_toml", type=click.Path(exists=True, dir_okay=False, path_type=str))
@_local_options
def _local(
    config_toml: str,
    workers: int = 1,
    results_dir: str = "results/uhd",
    overrides: tuple[str, ...] = (),
    dashboard: bool = False,
    child_process: bool = False,
) -> None:
    tomllib = _im("tomllib")
    p = _im("ops.exp_uhd_parse")
    try:
        cfg = p._load_toml_config(config_toml)
        if overrides:
            cfg = {**cfg, **p._parse_overrides(overrides)}
        p._validate_required(cfg)
    except (OSError, tomllib.TOMLDecodeError, TypeError, ValueError) as e:
        raise click.ClickException(str(e)) from e

    parsed = p._parse_cfg(cfg)
    _run_mod().run_parsed_uhd_local(
        parsed,
        cfg=cfg,
        results_dir=results_dir,
        workers=workers,
        config_toml=config_toml,
        overrides=tuple(overrides),
        dashboard=bool(dashboard),
        child_process=bool(child_process),
    )


def _run_parsed(*args, **kwargs):
    return _run_mod().run_parsed_uhd_local(*args, **kwargs)


def _run_bszo(*args, **kwargs):
    return _run_mod()._run_bszo(*args, **kwargs)


def _run_mezo(*args, **kwargs):
    return _run_mod()._run_mezo(*args, **kwargs)


def _run_simple(*args, **kwargs):
    return _run_mod()._run_simple(*args, **kwargs)


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
def modal_cmd(config_toml: str, log_file: str | None = None, gpu: str = "A100") -> None:
    log_text = _run_mod().uhd_config_toml_to_modal_log(
        config_toml,
        gpu,
        exp_uhd_parse=_im("ops.exp_uhd_parse"),
        tomllib=_im("tomllib"),
        modal_run=_im("ops.modal_uhd").run,
    )

    if log_file is not None:
        with open(log_file, "w") as f:
            f.write(log_text)
        click.echo(f"Log saved to {log_file}")


if __name__ == "__main__":
    cli()
