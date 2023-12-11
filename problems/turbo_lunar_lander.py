import gymnasium as gym
import numpy as np
from gymnasium.utils.step_api_compatibility import step_api_compatibility


class TurboLunarLander:
    def __init__(self, noise_seed):
        assert False, "Use env_tag=tlunar instead"
        self._noise_seed = noise_seed
        self.n_dim = 12
        self.bounds = np.array([(0, 2)] * self.n_dim)

    def _heuristic(self, env, s, w):
        angle_targ = s[0] * w[0] + s[2] * w[1]
        if angle_targ > w[2]:
            angle_targ = w[2]
        if angle_targ < -w[2]:
            angle_targ = -w[2]
        hover_targ = w[3] * np.abs(s[0])

        angle_todo = (angle_targ - s[4]) * w[4] - (s[5]) * w[5]
        hover_todo = (hover_targ - s[1]) * w[6] - (s[3]) * w[7]

        if s[6] or s[7]:
            angle_todo = w[8]
            hover_todo = -(s[3]) * w[9]

        a = 0
        if hover_todo > np.abs(angle_todo) and hover_todo > w[10]:
            a = 2
        elif angle_todo < -w[11]:
            a = 3
        elif angle_todo > +w[11]:
            a = 1
        return a

    def __call__(self, w):
        env = gym.make("LunarLander-v2", continuous=False)
        total_reward = 0
        # print("NOISE_SEED:", self._noise_seed_)
        s, info = env.reset(seed=self._noise_seed)
        while True:
            a = self._heuristic(env, s, w)
            s, r, terminated, truncated, info = step_api_compatibility(env.step(a), True)
            total_reward += r
            if terminated or truncated:
                break
        return -total_reward
