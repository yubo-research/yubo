import torch

from problems.environment_spec import get_environment_spec, materialize_env
from problems.unified_mj_env import save_unified_checkpoint


def run_warp_example():
    # 1. Materialize a Warp-accelerated MuJoCo environment
    # The 'warp:' prefix triggers our new UnifiedMJXWarpAdapter
    env_tag = "warp:gymnasium:Ant-v5"
    print(f"Creating environment: {env_tag}")

    spec = get_environment_spec(env_tag)
    runtime = materialize_env(spec)
    env = runtime.make()

    # 2. Reset the environment (returns Torch views by default)
    print("Resetting environment...")
    obs_dict, data = env.reset()
    torch_obs = obs_dict["obs"]
    print(f"Initial observation (Torch): shape={torch_obs.shape}, device={torch_obs.device}")

    # 3. Step the environment
    # We provide a Torch action; the adapter handles the DLPack conversion to Warp
    action = torch.randn(env.action_space.shape, device=torch_obs.device)
    print("Stepping environment with Torch action...")
    state = env.step(data, action)

    # 4. Access framework-native views (Zero-Copy)
    # The state object acts as a bridge between frameworks
    torch_views = state.torch()
    print(f"Next observation (Torch): {torch_views['obs'].shape}")

    try:
        import jax  # noqa: F401

        jax_views = state.jax()
        print(f"Next observation (JAX):   {jax_views['obs'].shape}, type={type(jax_views['obs'])}")
    except ImportError:
        print("JAX not available for view demonstration.")

    # 5. Unified Checkpointing with Orbax
    # We save a Pytree containing both Torch and Warp data
    print("Saving unified checkpoint...")
    import os

    abs_path = os.path.abspath("_tmp/example_checkpoint")
    checkpoint_state = {
        "step": 1,
        "obs": state.obs,  # Warp array
        "policy_output": torch_views["obs"],  # Torch tensor
    }
    save_unified_checkpoint(abs_path, checkpoint_state)
    print(f"Checkpoint saved to {abs_path}/0")


if __name__ == "__main__":
    run_warp_example()
