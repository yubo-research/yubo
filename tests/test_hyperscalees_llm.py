from click.testing import CliRunner


def test_hyperscalees_llm_dry_run_does_not_write_metadata(tmp_path):
    from experiments.hyperscalees_llm import cli

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


def test_hyperscalees_llm_lists_all_dynamic_bandit_tasks():
    from experiments.hyperscalees_llm import cli

    result = CliRunner().invoke(cli, ["tasks"])

    assert result.exit_code == 0, result.output
    assert "fastzero\n" in result.output
    assert "basic_arithmetic\n" in result.output
    assert "zebra_puzzles\n" in result.output


def test_hyperscalees_llm_template_roundtrips_for_dynamic_bandit_task(tmp_path):
    from experiments import hyperscalees_llm
    from experiments.hyperscalees_llm import cli

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


def test_ops_hyperscalees_llm_forwards_to_experiments_cli():
    from ops.hyperscalees_llm import cli

    result = CliRunner().invoke(cli, ["scripts"])

    assert result.exit_code == 0, result.output
    assert "general_do_evolution\n" in result.output
    assert "do_grpo\n" in result.output

