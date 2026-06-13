from click.testing import CliRunner


def test_hyperscalees_llm_dry_run_does_not_write_metadata(tmp_path):
    from experiments.hyperscalees_llm import cli, local

    exp_dir = tmp_path / "runs" / "dry_run"
    config = tmp_path / "config.toml"
    config.write_text(
        f"""
[experiment]
exp_dir = "{exp_dir.as_posix()}"
repo_dir = ".external/HyperscaleES"
dry_run = false

[hyperscalees]
script = "general_do_evolution"

[args]
model_choice = "7w0.1B"
task = "fastzero"
num_epochs = 1
track = false
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["local", str(config), "--dry-run"])

    assert result.exit_code == 0, result.output
    assert "DRY_RUN: true" in result.output
    assert not exp_dir.exists()
    assert callable(local)


def test_hyperscalees_llm_lists_all_dynamic_bandit_tasks():
    from experiments.hyperscalees_llm import cli

    result = CliRunner().invoke(cli, ["tasks"])

    assert result.exit_code == 0, result.output
    assert "fastzero\n" in result.output
    assert "basic_arithmetic\n" in result.output
    assert "zebra_puzzles\n" in result.output


def test_hyperscalees_llm_template_direct_calls_cover_branches():
    from click.testing import CliRunner

    from experiments.hyperscalees_llm import template

    runner = CliRunner()

    ok = runner.invoke(
        template,
        ["--task", "fastzero", "--model-choice", "7w1.5B", "--noiser", "gaussian"],
    )
    assert ok.exit_code == 0
    assert "runs/yubo_hyperscalees/llm_bandits/fastzero_7w1p5b_gaussian" in ok.output

    bad_script = runner.invoke(
        template,
        ["--task", "fastzero", "--script", "not_a_real_script"],
    )
    assert bad_script.exit_code != 0
    assert "Unknown HyperscaleES script" in bad_script.output

    bad_task = runner.invoke(template, ["--task", "not_a_real_task"])
    assert bad_task.exit_code != 0
    assert "Unsupported HyperscaleES LLM bandit task" in bad_task.output

    with runner.isolation():
        template(
            ["--task", "basic_arithmetic", "--exp-dir", "runs/custom"],
            standalone_mode=False,
        )


def test_hyperscalees_llm_template_roundtrips_for_dynamic_bandit_task(tmp_path):
    from experiments import hyperscalees_llm
    from experiments.hyperscalees_llm import cli, template

    result = CliRunner().invoke(
        cli,
        [
            "template",
            "--task",
            "basic_arithmetic",
            "--model-choice",
            "7w1.5B",
            "--num-epochs",
            "3",
            "--parallel-generations-per-gpu",
            "2",
        ],
    )

    assert result.exit_code == 0, result.output
    config = tmp_path / "basic_arithmetic.toml"
    config.write_text(result.output, encoding="utf-8")
    cfg = hyperscalees_llm._finalize_config(hyperscalees_llm._load_toml_config(str(config)))
    cmd = hyperscalees_llm._make_command(cfg["hyperscalees"], cfg["args"])
    assert "--task" in cmd
    assert "basic_arithmetic" in cmd
    assert "--model-choice" in cmd
    assert "7w1.5B" in cmd
    assert callable(template)


def test_ops_hyperscalees_llm_forwards_to_experiments_cli():
    from ops.hyperscalees_llm import cli

    result = CliRunner().invoke(cli, ["scripts"])

    assert result.exit_code == 0, result.output
    assert "general_do_evolution\n" in result.output
    assert "do_grpo\n" in result.output


def test_ops_nanoegg_pretrain_forwards_to_experiments_cli(tmp_path):
    from ops.nanoegg_pretrain import cli

    exp_dir = tmp_path / "nanoegg"
    config = tmp_path / "nanoegg.toml"
    config.write_text(
        f"""
[experiment]
exp_dir = "{exp_dir.as_posix()}"
repo_dir = ".external/nano-egg"
dry_run = true

[nanoegg]
num_epochs = 1
population_size = 2
batch_size = 1
regenerate_model = true
track = false

[data]
synthetic = true
""".strip()
        + "\n",
        encoding="utf-8",
    )

    result = CliRunner().invoke(cli, ["local", str(config)])

    assert result.exit_code == 0, result.output
    assert "NANOEGG_REPO:" in result.output
    assert "DRY_RUN: true" in result.output
