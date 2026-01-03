import numpy as np
import torch


class PufferLibBreakoutPolicy:
    def __init__(self, env_conf, use_lstm=False):
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        self._use_lstm = use_lstm

        from pufferlib.ocean.breakout.breakout import Breakout
        from pufferlib.ocean.torch import Policy, Recurrent

        self._puffer_env = Breakout(num_envs=1)
        self._base_policy = Policy(self._puffer_env, hidden_size=128)
        if use_lstm:
            self._nn_policy = Recurrent(
                self._puffer_env, self._base_policy, input_size=128, hidden_size=128
            )
        else:
            self._nn_policy = self._base_policy
        self._nn_policy.eval()

        self._num_params = sum(p.numel() for p in self._nn_policy.parameters())
        self._flat_params = np.zeros(self._num_params, dtype=np.float32)

        if use_lstm:
            self._lstm_h = None
            self._lstm_c = None

    def num_params(self):
        return self._num_params

    def get_params(self):
        return self._flat_params.copy()

    def set_params(self, x):
        assert x.min() >= -1 and x.max() <= 1, (x.min(), x.max())
        self._flat_params = np.asarray(x, dtype=np.float32).copy()
        scaled = self._flat_params * 2.0
        idx = 0
        with torch.no_grad():
            for p in self._nn_policy.parameters():
                numel = p.numel()
                p.data.copy_(
                    torch.from_numpy(scaled[idx : idx + numel].reshape(p.shape)).float()
                )
                idx += numel

    def clone(self):
        p = PufferLibBreakoutPolicy(self._env_conf, use_lstm=self._use_lstm)
        p.set_params(self._flat_params)
        return p

    def reset_state(self):
        if self._use_lstm:
            self._lstm_h = None
            self._lstm_c = None

    def __call__(self, state):
        with torch.no_grad():
            obs_t = torch.from_numpy(state).float().unsqueeze(0)
            if self._use_lstm:
                state_dict = {"lstm_h": self._lstm_h, "lstm_c": self._lstm_c}
                logits, _ = self._nn_policy(obs_t, state_dict)
                self._lstm_h = state_dict.get("lstm_h")
                self._lstm_c = state_dict.get("lstm_c")
            else:
                logits, _ = self._nn_policy(obs_t)
            action = int(logits.argmax(dim=-1).item())
        return action

    def close(self):
        if hasattr(self, "_puffer_env") and self._puffer_env is not None:
            self._puffer_env.close()
            self._puffer_env = None
