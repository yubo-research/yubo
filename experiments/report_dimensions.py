from rl_gym.env_conf import _env_confs, default_policy, get_env_conf

if __name__ == "__main__":
    for env_tag in _env_confs:
        env_conf = get_env_conf(env_tag, seed=1)
        policy = default_policy(env_conf)
        print(env_tag, policy.num_params())
