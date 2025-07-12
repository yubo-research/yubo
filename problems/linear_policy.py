from problems.linear_policty_calculator import LinearPolicyCalculator


# Simple random search of static linear policies is competitive for reinforcement learning
# https://proceedings.neurips.cc/paper_files/paper/2018/hash/7634ea65a4e6d9041cfd3f7de18e334a-Abstract.html
# Ant: 4000 after ~50,000 episodes
class LinearPolicy:
    def __init__(self, env_conf):
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]
        self._calculator = LinearPolicyCalculator(id_int=env_conf.problem_seed, num_state=num_state, num_action=num_action)

    def num_params(self):
        return self._calculator.num_params()

    def set_params(self, x):
        self._calculator.set_params(x)

    def get_params(self):
        return self._calculator.get_params()

    def clone(self):
        lp = LinearPolicy(self._env_conf)
        lp._calculator = self._calculator.clone()
        return lp

    def __call__(self, state):
        return self._calculator.calculate(state)
