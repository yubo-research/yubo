import numpy as np
import torch
import torch.nn as nn

from problems.policy_mixin import PolicyParamsMixin


class MLPPolicyFactory:
    def __init__(
        self,
        hidden_sizes,
        *,
        use_layer_norm=True,
        rnn_hidden_size=None,
        use_prev_action=False,
        use_phase_features=False,
        num_phase_harmonics=1,
    ):
        self._hidden_sizes = hidden_sizes
        self._use_layer_norm = bool(use_layer_norm)
        self._rnn_hidden_size = rnn_hidden_size
        self._use_prev_action = bool(use_prev_action)
        self._use_phase_features = bool(use_phase_features)
        self._num_phase_harmonics = int(num_phase_harmonics)

    def __call__(self, env_conf):
        return MLPPolicy(
            env_conf,
            self._hidden_sizes,
            use_layer_norm=self._use_layer_norm,
            rnn_hidden_size=self._rnn_hidden_size,
            use_prev_action=self._use_prev_action,
            use_phase_features=self._use_phase_features,
            num_phase_harmonics=self._num_phase_harmonics,
        )


class MLPPolicy(PolicyParamsMixin, nn.Module):
    def __init__(
        self,
        env_conf,
        hidden_sizes,
        *,
        use_layer_norm=True,
        rnn_hidden_size=None,
        use_prev_action=False,
        use_phase_features=False,
        num_phase_harmonics=1,
    ):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf
        num_state, num_action = self._init_flags(
            env_conf,
            use_layer_norm=use_layer_norm,
            rnn_hidden_size=rnn_hidden_size,
            use_prev_action=use_prev_action,
            use_phase_features=use_phase_features,
            num_phase_harmonics=num_phase_harmonics,
        )
        self._build_network(num_state, num_action, hidden_sizes)
        self._init_params()
        self._cache_flat_params_init()

    def _init_flags(
        self,
        env_conf,
        *,
        use_layer_norm,
        rnn_hidden_size,
        use_prev_action,
        use_phase_features,
        num_phase_harmonics,
    ):
        num_state = int(env_conf.gym_conf.state_space.shape[0])
        num_action = int(env_conf.action_space.shape[0])
        self._const_scale = 0.5
        self._use_layer_norm = bool(use_layer_norm)
        self._rnn_hidden_size = None if rnn_hidden_size is None else int(rnn_hidden_size)
        if self._rnn_hidden_size is not None:
            assert self._rnn_hidden_size >= 1
        self._use_prev_action = bool(use_prev_action)
        self._use_phase_features = bool(use_phase_features)
        self._num_phase_harmonics = int(num_phase_harmonics)
        assert self._num_phase_harmonics >= 1
        self.in_norm = nn.LayerNorm(num_state, elementwise_affine=True) if self._use_layer_norm else None
        return num_state, num_action

    def _build_network(self, num_state, num_action, hidden_sizes):
        dims = [num_state] + list(hidden_sizes) + [num_action]
        if self._rnn_hidden_size is None:
            layers = []
            for i in range(len(dims) - 2):
                layers.append(nn.Linear(dims[i], dims[i + 1]))
                layers.append(nn.SiLU())
            layers.append(nn.Linear(dims[-2], dims[-1]))
            layers.append(nn.Tanh())
            self.model = nn.Sequential(*layers)
            self.embed = None
            self.rnn = None
            self.head = None
            self._use_prev_action = False
            self._prev_action = None
            return

        self.model = None
        phase_dim = 2 * self._num_phase_harmonics if self._use_phase_features else 0
        in_dim = num_state + phase_dim + (num_action if self._use_prev_action else 0)
        feat_layers = []
        d_in = in_dim
        for hs in list(hidden_sizes):
            feat_layers.append(nn.Linear(d_in, int(hs)))
            feat_layers.append(nn.SiLU())
            d_in = int(hs)
        feat_layers.append(nn.Linear(d_in, self._rnn_hidden_size))
        feat_layers.append(nn.SiLU())
        self.embed = nn.Sequential(*feat_layers)
        self.rnn = nn.GRUCell(self._rnn_hidden_size, self._rnn_hidden_size)
        self.head = nn.Sequential(nn.Linear(self._rnn_hidden_size, num_action), nn.Tanh())
        self.reset_state()

    def _cache_flat_params_init(self):
        with torch.inference_mode():
            self._flat_params_init = np.concatenate([p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()])

    def _init_params(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            if isinstance(m, nn.RNNCell):
                nn.init.orthogonal_(m.weight_ih, gain=0.5)
                nn.init.orthogonal_(m.weight_hh, gain=0.5)
                if m.bias_ih is not None:
                    nn.init.zeros_(m.bias_ih)
                if m.bias_hh is not None:
                    nn.init.zeros_(m.bias_hh)
            if isinstance(m, nn.GRUCell):
                nn.init.orthogonal_(m.weight_ih, gain=0.5)
                nn.init.orthogonal_(m.weight_hh, gain=0.5)
                if m.bias_ih is not None:
                    nn.init.zeros_(m.bias_ih)
                if m.bias_hh is not None:
                    nn.init.zeros_(m.bias_hh)

    def forward(self, x):
        if self.model is not None:
            return self.model(x)

        if self._use_phase_features:
            self._phase += float(self._phase_omega)
            if self._phase >= 2.0 * np.pi:
                self._phase -= 2.0 * np.pi
            feats = []
            for k in range(1, self._num_phase_harmonics + 1):
                feats.append(np.sin(k * self._phase))
                feats.append(np.cos(k * self._phase))
            phase_feat = torch.as_tensor(feats, dtype=torch.float32)
            x = torch.cat([x, phase_feat], dim=0)

        if self._use_prev_action:
            x = torch.cat([x, self._prev_action], dim=0)
        x = self.embed(x)
        self._h = self.rnn(x, self._h)
        return self.head(self._h)

    def reset_state(self):
        if self._rnn_hidden_size is None:
            return
        self._h = torch.zeros((self._rnn_hidden_size,), dtype=torch.float32)
        if self._use_prev_action:
            self._prev_action = torch.zeros((self._env_conf.action_space.shape[0],), dtype=torch.float32)
        if self._use_phase_features:
            self._phase = 0.0
            self._phase_omega = 0.12

    def __call__(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        if self.in_norm is not None:
            state = self.in_norm(state)

        with torch.inference_mode():
            action = self.forward(state)
            if self._use_prev_action:
                self._prev_action = action.detach()
        return action.detach().cpu().numpy()
