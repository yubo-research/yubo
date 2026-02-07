#!/usr/bin/env python

import warnings

import click

from optimizer.uhd_loop import UHDLoop


@click.group()
def cli():
    pass


def _make_loop(env_tag, num_rounds):
    from problems.env_conf import get_env_conf
    from problems.torch_policy import TorchPolicy

    env_conf = get_env_conf(env_tag)
    assert not env_conf.frozen_noise, "frozen_noise not supported for UHD"

    env = env_conf.make()

    if hasattr(env, "torch_env"):
        # Env provides a TorchEnv (e.g. MNIST).
        # TorchPolicy runs module.forward() on data from the env;
        # GaussianPerturbator perturbs the same module in-place.
        torch_env = env.torch_env()
        module = torch_env.module
        policy = TorchPolicy(module, env_conf)

        def evaluate_fn():
            state, _ = torch_env.reset()
            logits = policy(state)
            _, reward, _, _ = torch_env.step(logits)
            return float(reward)
    else:
        # Gym env — build policy network, evaluate via collect_trajectory.
        env.close()

        from optimizer.trajectories import collect_trajectory
        from problems.mlp_torch_policy import MLPPolicyModule

        if env_conf.policy_class is not None:
            warnings.warn(
                f"Replacing policy_class {env_conf.policy_class} with MLPPolicyModule",
                stacklevel=2,
            )

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]

        module = MLPPolicyModule(num_state, num_action, hidden_sizes=(32, 16))
        policy = TorchPolicy(module, env_conf)

        def evaluate_fn():
            return float(collect_trajectory(env_conf, policy).rreturn)

    return UHDLoop(module, evaluate_fn, sigma_0=0.1, num_iterations=num_rounds)


@cli.command()
@click.option(
    "--env-tag",
    required=True,
    help="Environment tag (e.g. lunar, ant, bw, mnist)",
)
@click.option("--num-rounds", required=True, type=int, help="Number of UHD iterations")
def local(env_tag, num_rounds):
    loop = _make_loop(env_tag, num_rounds)
    loop.run()


if __name__ == "__main__":
    cli()
