"""Tests for common.console."""

from types import SimpleNamespace

from common.console import (
    BOConsoleCollector,
    print_bo_footer,
    print_bo_header_top,
    print_bo_round,
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


def test_print_bo_header_top():
    print_bo_header_top(
        env_tag="dm:cheetah-run:gauss",
        opt_name="turbo-enn-p",
        num_rounds=10,
        num_arms=4,
    )


def test_print_bo_round(capsys):
    print_bo_round(
        parsed={"iter": 1, "ret_best": 10.5, "ret_eval": 9.0, "elapsed": 1.2},
        opt_name="turbo-enn-p",
    )
    out = capsys.readouterr().out
    assert "ret_best=10.5" in out
    assert "ret_eval=9.0" in out
    assert "elapsed=1.2s" in out


def test_print_bo_round_includes_proposal_timing(capsys):
    print_bo_round(
        parsed={
            "iter": 2,
            "ret_best": 12.0,
            "ret_eval": 11.0,
            "proposal_dt": 0.1234,
            "proposal_elapsed": 1.2499,
            "elapsed": 2.0,
        },
        opt_name="turbo-enn-p",
    )
    out = capsys.readouterr().out
    assert "proposal_dt=0.123s" in out
    assert "proposal_elapsed=1.250s" in out


def test_print_bo_footer():
    print_bo_footer(best_return=10.5, total_time=12.3)


def test_print_run_header():
    config = SimpleNamespace(env_tag="dm:cheetah-run", seed=42, backbone_name="mlp")
    env = SimpleNamespace(env_conf=SimpleNamespace(from_pixels=False), obs_dim=17, act_dim=6)
    training = SimpleNamespace(frames_per_batch=64, num_iterations=100)
    runtime = SimpleNamespace(device=SimpleNamespace(type="cpu"))
    print_run_header("ppo", config, env, training, runtime)


def test_print_run_header_atari_backbone(capsys):
    from rl.backends.torchrl.common.env_contract import (
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
    assert "from_pixels=True" in out


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


def test_bo_console_collector():
    collector = BOConsoleCollector(
        env_tag="dm:cheetah-run:gauss",
        opt_name="turbo-enn-p",
        num_rounds=2,
        num_arms=4,
    )
    collector("ITER  iter=1  ret_best=5.0  ret_eval=4.0  elapsed=1.0")
