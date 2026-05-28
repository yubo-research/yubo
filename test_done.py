from rl.torchrl.collect_utils import make_collect_env
from rl.torchrl.ppo.config import PPOConfig
from rl.torchrl.ppo.core_env_setup import build_env_setup


def main():
    cfg = PPOConfig.from_dict({"env_tag": "cheetah", "policy_tag": "actor-critic-mlp-32-32-tanh", "seed": 0, "from_pixels": False})
    env_setup = build_env_setup(cfg)
    env = make_collect_env(env_setup.env_conf)

    td = env.reset()
    done_count = 0
    print(f"Action spec: {env.action_spec}")
    for i in range(10):
        # TorchRL envs take tensordict and add "next"
        action = env.action_spec.rand()
        td.set("action", action)
        td = env.step(td)
        done_flag = td["next", "done"]
        print(f"Step {i}: done = {done_flag}")
        if done_flag.any():
            done_count += 1
            td = env.reset()
        else:
            from tensordict.nn import step

            td = step(td)

    print(f"Total dones in 10 steps: {done_count}")


if __name__ == "__main__":
    main()
