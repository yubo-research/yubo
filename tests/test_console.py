"""Tests for common.console."""

from types import SimpleNamespace

from common.console import (
    BOConsoleCollector,
    print_bo_footer,
    print_iteration_log,
    print_iteration_simple,
    print_run_footer,
    print_run_header,
    register_algo_metrics,
    register_opt_metrics,
)


def test_register_algo_metrics():
    register_algo_metrics("test_algo", [("m1", 6, ".2f"), ("m2", 8, ".4f")])


def test_register_opt_metrics():
    register_opt_metrics("test_opt", [("k", 6, ".2f")])


def test_print_bo_footer():
    print_bo_footer(best_return=10.5, total_time=12.3)


def test_print_run_header():
    config = SimpleNamespace(env_tag="dm:cheetah-run", seed=42, backbone_name="mlp")
    env = SimpleNamespace(env_conf=SimpleNamespace(from_pixels=False), obs_dim=17, act_dim=6)
    training = SimpleNamespace(frames_per_batch=64, num_iterations=100)
    runtime = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    print_run_header("ppo", config, env, training, runtime)


def test_print_run_header_atari_backbone(capsys):
    from rl.core.env_contract import (
        EnvIOContract,
        ObservationContract,
    )

    config = SimpleNamespace(env_tag="atari:Pong", seed=1, backbone_name="mlp")
    env = SimpleNamespace(
        env_conf=SimpleNamespace(from_pixels=True, env_name="ALE/Pong-v5"),
        io_contract=EnvIOContract(
            observation=ObservationContract(mode="pixels", raw_shape=(4, 84, 84, 1), model_channels=4, image_size=84),
            action=SimpleNamespace(),  # unused by print_run_header
        ),
        obs_dim=64,
        act_dim=6,
    )
    training = SimpleNamespace(frames_per_batch=64, num_iterations=2)
    runtime = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    print_run_header("ppo", config, env, training, runtime)
    out = capsys.readouterr().out
    assert "backbone=nature_cnn_atari" in out
    assert "obs_mode=pixels" in out


def test_print_iteration_log():
    print_iteration_log(
        iteration=1,
        num_iterations=10,
        frames_per_batch=64,
        eval_return=10.0,
        best_return=10.0,
        elapsed=1.0,
    )


def test_print_iteration_simple():
    print_iteration_simple(iteration=1, num_iterations=10, frames_per_batch=64, elapsed=1.0)


def test_prefixed_rl_table_alignment(capsys):
    prefix = "[rl/ppo/puffer] "
    config = SimpleNamespace(env_tag="bw-heur", seed=1, backbone_name="mlp")
    env = SimpleNamespace(env_conf=SimpleNamespace(from_pixels=False), obs_dim=24, act_dim=4)
    training = SimpleNamespace(frames_per_batch=4096, num_iterations=10)
    runtime = SimpleNamespace(device=SimpleNamespace(type="mps"))

    print_run_header("ppo", config, env, training, runtime, prefix=prefix)
    print_iteration_log(
        iteration=1,
        num_iterations=10,
        frames_per_batch=4096,
        eval_return=-91.5,
        heldout_return=-91.5,
        best_return=-91.5,
        algo_metrics={"kl": 0.0022, "clipfrac": 0.0013},
        algo_name="ppo",
        elapsed=6.6,
        prefix=prefix,
    )
    print_iteration_simple(
        iteration=2,
        num_iterations=10,
        frames_per_batch=4096,
        elapsed=11.1,
        algo_name="ppo",
        prefix=prefix,
    )

    lines = [line for line in capsys.readouterr().out.splitlines() if line]
    assert any(line.startswith(prefix) and "iter" in line for line in lines)
    assert any(line.startswith(prefix) and "4,096" in line for line in lines)
    assert any(line.startswith(prefix) and "8,192" in line for line in lines)
    assert not any(line.startswith(prefix + "(") for line in lines)


def test_print_run_footer():
    print_run_footer(best_return=10.0, total_iters_or_steps=100, total_time=12.5)


def test_bo_console_collector_echoes_iter_lines(capsys):
    """BOConsoleCollector echoes ITER: lines to stdout (same format as data file)."""
    collector = BOConsoleCollector()
    line = "ITER: iter = 1 elapsed = 1.0s eval_dt = 0.1s proposal_dt = 0.05s fit_dt = 0.000 select_dt = 0.001 tr_length = 0.800 proposal_elapsed = 0.05s y_best = 5.0 ret_best = 5.0 ret_eval = 4.0"
    collector(line)
    out = capsys.readouterr().out
    assert line in out


def test_rl_logger_facade(tmp_path):
    from rl import logger as rl_logger

    rl_logger.append_metrics(tmp_path / "metrics.jsonl", {"x": 1})
    rl_logger.log_run_header_basic(
        algo_name="ppo",
        env_tag="pend",
        seed=1,
        backbone_name="mlp",
        obs_mode="state",
        obs_dim=3,
        act_dim=1,
        frames_per_batch=8,
        num_iterations=2,
        device_type="cpu",
    )
    rl_logger.log_eval_iteration(1, 2, 8, eval_return=1.0, best_return=1.0, elapsed=0.1)
    rl_logger.log_progress_iteration(1, 2, 8, elapsed=0.1)
    rl_logger.log_run_footer(1.0, 2, 0.2)
