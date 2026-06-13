"""Local ENN batch runners (add, fit, fit-ind) for enn_incremental_batches CLI."""

from __future__ import annotations

import json
from typing import Callable

import click


def register_local_commands(
    cli,
    *,
    resolve_checkpoints: Callable[[str | None], tuple[int, ...]],
    ensure_repo_imports: Callable[[], None],
) -> None:
    @cli.command(
        params=[
            click.Argument(["function_name"]),
            click.Argument(["rep_index"], type=int),
            click.Argument(
                ["index_driver"],
                type=click.Choice(["flat", "hnsw", "hnsw_disk"], case_sensitive=False),
            ),
            click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
            click.Option(
                ("--problem-seed", "problem_seed"),
                type=int,
                default=17,
                show_default=True,
            ),
            click.Option(
                ("--num-reps", "num_reps"),
                type=int,
                default=10,
                show_default=True,
            ),
            click.Option(
                ("--output-dir", "output_dir"),
                type=click.Path(),
                default="results/enn_incremental",
                show_default=True,
            ),
            click.Option(
                ("--checkpoints", "checkpoint_csv"),
                default="",
                show_default=True,
                help="Comma-separated checkpoint Ns; default uses the batch checkpoint grid.",
            ),
            click.Option(("--force/--no-force", "force"), default=False, show_default=True),
        ],
    )
    def local(
        function_name: str,
        rep_index: int,
        index_driver: str,
        d_dims: int,
        problem_seed: int,
        num_reps: int,
        output_dir: str,
        checkpoint_csv: str,
        force: bool,
    ):
        """Run one incremental ENN job locally and write the result JSON."""
        if d_dims < 1:
            raise click.BadParameter("D must be positive")
        if rep_index < 0:
            raise click.BadParameter("REP_INDEX must be >= 0")
        if num_reps < 1:
            raise click.BadParameter("num-reps must be >= 1")

        ensure_repo_imports()
        from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
        from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
        from analysis.fitting_time.fitting_time_enn_incremental import (
            EnnIncrementalIndexDriver,
            benchmark_enn_incremental_add_timing,
        )
        from experiments.modal_enn_incremental_batches_impl import (
            result_json_dest,
            result_to_payload,
        )

        fn = normalize_benchmark_function_name(function_name)
        driver = EnnIncrementalIndexDriver(index_driver.lower())
        checkpoints = resolve_checkpoints(checkpoint_csv or None)
        dest = result_json_dest(
            output_dir,
            d=d_dims,
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
            num_reps=num_reps,
            index_driver=driver,
        )
        if dest.exists() and not force:
            click.echo(f"skip existing {dest.resolve()}")
            return

        data_seed = synthetic_benchmark_data_seed(
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
        )
        click.echo(
            f"running incremental ENN D={d_dims} fn={fn} problem_seed={problem_seed} "
            f"rep_index={rep_index} data_seed={data_seed} index_driver={driver.value} "
            f"checkpoints={checkpoints or 'default'}",
            err=True,
        )
        result = benchmark_enn_incremental_add_timing(
            D=d_dims,
            function_name=fn,
            problem_seed=data_seed,
            index_driver=driver,
            checkpoints=checkpoints,
        )
        payload = result_to_payload(
            result,
            problem_seed=problem_seed,
            data_seed=data_seed,
            rep_index=rep_index,
            num_reps=num_reps,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        click.echo(f"wrote {dest.resolve()}")

    @cli.command(
        "local-fit",
        params=[
            click.Argument(["function_name"]),
            click.Argument(["n"], type=int),
            click.Argument(["rep_index"], type=int),
            click.Argument(
                ["index_driver"],
                type=click.Choice(["flat", "hnsw", "hnsw_disk"], case_sensitive=False),
            ),
            click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
            click.Option(
                ("--problem-seed", "problem_seed"),
                type=int,
                default=17,
                show_default=True,
            ),
            click.Option(
                ("--num-reps", "num_reps"),
                type=int,
                default=10,
                show_default=True,
            ),
            click.Option(
                ("--output-dir", "output_dir"),
                type=click.Path(),
                default="results/enn_incremental",
                show_default=True,
            ),
            click.Option(("--force/--no-force", "force"), default=False, show_default=True),
        ],
    )
    def local_fit(
        function_name: str,
        n: int,
        rep_index: int,
        index_driver: str,
        d_dims: int,
        problem_seed: int,
        num_reps: int,
        output_dir: str,
        force: bool,
    ):
        """Run one ENN fit-timing job locally and write the fit-only result JSON."""
        if d_dims < 1:
            raise click.BadParameter("D must be positive")
        if n < 1:
            raise click.BadParameter("N must be >= 1")
        if rep_index < 0:
            raise click.BadParameter("REP_INDEX must be >= 0")
        if num_reps < 1:
            raise click.BadParameter("num-reps must be >= 1")

        ensure_repo_imports()
        from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
        from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
        from analysis.fitting_time.fitting_time_enn_fit import benchmark_enn_fit_timing
        from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
        from experiments import modal_enn_fit_batches as fit_batches

        fn = normalize_benchmark_function_name(function_name)
        driver = EnnIncrementalIndexDriver(index_driver.lower())
        dest = fit_batches.fit_result_json_dest(
            output_dir,
            d=d_dims,
            function_name=fn,
            n=n,
            problem_seed=problem_seed,
            rep_index=rep_index,
            num_reps=num_reps,
            index_driver=driver,
            normalize_function_name=normalize_benchmark_function_name,
        )
        if (
            dest.exists()
            and not force
            and fit_batches.fit_result_json_complete(
                dest,
                n,
                d=d_dims,
                function_name=fn,
                problem_seed=problem_seed,
                rep_index=rep_index,
                num_reps=num_reps,
                index_driver=driver,
                normalize_function_name=normalize_benchmark_function_name,
            )
        ):
            click.echo(f"skip existing {dest.resolve()}")
            return

        data_seed = synthetic_benchmark_data_seed(
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
        )
        result = benchmark_enn_fit_timing(
            D=d_dims,
            function_name=fn,
            data_seed=data_seed,
            problem_seed=problem_seed,
            n=n,
            index_driver=driver,
        )
        payload = fit_batches.fit_result_to_payload(
            result,
            problem_seed=problem_seed,
            data_seed=data_seed,
            rep_index=rep_index,
            num_reps=num_reps,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        click.echo(f"wrote {dest.resolve()}")

    @cli.command(
        "local-fit-ind",
        params=[
            click.Argument(["function_name"]),
            click.Argument(["rep_index"], type=int),
            click.Argument(
                ["index_driver"],
                type=click.Choice(["flat", "hnsw", "hnsw_disk"], case_sensitive=False),
            ),
            click.Option(("-d", "--d", "d_dims"), type=int, default=10, show_default=True),
            click.Option(
                ("--problem-seed", "problem_seed"),
                type=int,
                default=17,
                show_default=True,
            ),
            click.Option(
                ("--num-reps", "num_reps"),
                type=int,
                default=10,
                show_default=True,
            ),
            click.Option(
                ("--output-dir", "output_dir"),
                type=click.Path(),
                default="results/enn_incremental",
                show_default=True,
            ),
            click.Option(
                ("--checkpoints", "checkpoint_csv"),
                default="",
                show_default=True,
                help="Comma-separated checkpoint Ns; default uses the batch checkpoint grid.",
            ),
            click.Option(("--force/--no-force", "force"), default=False, show_default=True),
        ],
    )
    def local_fit_ind(
        function_name: str,
        rep_index: int,
        index_driver: str,
        d_dims: int,
        problem_seed: int,
        num_reps: int,
        output_dir: str,
        checkpoint_csv: str,
        force: bool,
    ):
        """Run one fit-ind ENN job locally and write the per-checkpoint fit-timing JSON."""
        if d_dims < 1:
            raise click.BadParameter("D must be positive")
        if rep_index < 0:
            raise click.BadParameter("REP_INDEX must be >= 0")
        if num_reps < 1:
            raise click.BadParameter("num-reps must be >= 1")

        ensure_repo_imports()
        from analysis.fitting_time.evaluate import synthetic_benchmark_data_seed
        from analysis.fitting_time.evaluate_metrics import normalize_benchmark_function_name
        from analysis.fitting_time.fitting_time_enn_fit_ind import benchmark_enn_fit_ind_timing
        from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
        from experiments import modal_enn_fit_ind_batches as fit_ind_batches
        from experiments import modal_enn_fit_ind_batches_json as fit_ind_json

        fn = normalize_benchmark_function_name(function_name)
        driver = EnnIncrementalIndexDriver(index_driver.lower())
        checkpoints = resolve_checkpoints(checkpoint_csv or None)
        dest = fit_ind_batches.fit_ind_result_json_dest(
            output_dir,
            d=d_dims,
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
            num_reps=num_reps,
            index_driver=driver,
            normalize_function_name=normalize_benchmark_function_name,
        )
        if (
            dest.exists()
            and not force
            and fit_ind_json.fit_ind_result_json_complete(
                dest,
                checkpoints,
                d=d_dims,
                function_name=fn,
                problem_seed=problem_seed,
                rep_index=rep_index,
                num_reps=num_reps,
                index_driver=driver,
            )
        ):
            click.echo(f"skip existing {dest.resolve()}")
            return

        data_seed = synthetic_benchmark_data_seed(
            function_name=fn,
            problem_seed=problem_seed,
            rep_index=rep_index,
        )
        click.echo(
            f"running fit-ind ENN D={d_dims} fn={fn} problem_seed={problem_seed} "
            f"rep_index={rep_index} data_seed={data_seed} index_driver={driver.value} "
            f"checkpoints={checkpoints or 'default'}",
            err=True,
        )
        result = benchmark_enn_fit_ind_timing(
            D=d_dims,
            function_name=fn,
            problem_seed=data_seed,
            index_driver=driver,
            checkpoints=checkpoints,
        )
        payload = fit_ind_batches.fit_ind_result_to_payload(
            result,
            problem_seed=problem_seed,
            data_seed=data_seed,
            rep_index=rep_index,
            num_reps=num_reps,
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        click.echo(f"wrote {dest.resolve()}")

    @cli.command(
        "local-full-opt",
        params=[
            click.Argument(["env_tag"]),
            click.Argument(["rep_index"], type=int),
            click.Argument(
                ["index_driver"],
                type=click.Choice(["flat", "hnsw", "hnsw_disk"], case_sensitive=False),
            ),
            click.Option(
                ("--num-reps", "num_reps"),
                type=int,
                default=10,
                show_default=True,
            ),
            click.Option(
                ("--output-dir", "output_dir"),
                type=click.Path(),
                default="results/enn_incremental",
                show_default=True,
            ),
            click.Option(
                ("--checkpoints", "checkpoint_csv"),
                default="",
                show_default=True,
                help="Comma-separated checkpoint Ns; default uses the full-opt grid (through 100k).",
            ),
            click.Option(
                ("--num-rounds", "num_rounds"),
                type=int,
                default=None,
                help="BO iteration cap; default is 100_000.",
            ),
            click.Option(("--force/--no-force", "force"), default=False, show_default=True),
        ],
    )
    def local_full_opt(
        env_tag: str,
        rep_index: int,
        index_driver: str,
        num_reps: int,
        output_dir: str,
        checkpoint_csv: str,
        num_rounds: int | None,
        force: bool,
    ):
        """Run one full-optimization ENN job locally and write checkpoint proposal-time JSON."""
        if rep_index < 0:
            raise click.BadParameter("REP_INDEX must be >= 0")
        if num_reps < 1:
            raise click.BadParameter("num-reps must be >= 1")

        ensure_repo_imports()
        from analysis.fitting_time.fitting_time_enn_full_opt import (
            FULL_OPT_NUM_ROUNDS,
            benchmark_enn_full_optimization_proposal_timing,
            opt_name_for_index_driver,
            resolve_full_opt_checkpoints,
        )
        from analysis.fitting_time.fitting_time_enn_incremental import EnnIncrementalIndexDriver
        from common.experiment_seeds import problem_seed_from_rep_index
        from experiments import modal_enn_full_opt_batches as full_opt_batches
        from experiments import modal_enn_full_opt_batches_json as full_opt_json

        driver = EnnIncrementalIndexDriver(index_driver.lower())
        rounds = FULL_OPT_NUM_ROUNDS if num_rounds is None else int(num_rounds)
        try:
            checkpoints = resolve_full_opt_checkpoints(checkpoint_csv or None)
        except ValueError as exc:
            raise click.BadParameter(str(exc)) from exc
        ps = problem_seed_from_rep_index(rep_index)
        opt_name = opt_name_for_index_driver(driver)
        dest = full_opt_batches.full_opt_result_json_dest(
            output_dir,
            env_tag=env_tag,
            problem_seed=ps,
            rep_index=rep_index,
            num_reps=num_reps,
            index_driver=driver,
        )
        if (
            dest.exists()
            and not force
            and full_opt_json.full_opt_result_json_complete(
                dest,
                checkpoints,
                env_tag=env_tag,
                problem_seed=ps,
                rep_index=rep_index,
                num_reps=num_reps,
                index_driver=driver,
                opt_name=opt_name,
            )
        ):
            click.echo(f"skip existing {dest.resolve()}")
            return

        click.echo(
            f"running full-opt ENN env_tag={env_tag} problem_seed={ps} rep_index={rep_index} "
            f"index_driver={driver.value} checkpoints={checkpoints or 'default'} num_rounds={rounds}",
            err=True,
        )
        result = benchmark_enn_full_optimization_proposal_timing(
            env_tag=env_tag,
            problem_seed=ps,
            rep_index=rep_index,
            index_driver=driver,
            checkpoints=checkpoints,
            num_rounds=rounds,
        )
        payload = full_opt_batches.full_opt_result_to_payload(result, num_reps=num_reps)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        click.echo(f"wrote {dest.resolve()}")
