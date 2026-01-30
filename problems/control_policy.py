import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from problems.policy_mixin import PolicyParamsMixin


class ControlPolicyFactory:
    def __init__(self, *, use_layer_norm=True):
        self._use_layer_norm = bool(use_layer_norm)

    def __call__(self, env_conf):
        return ControlPolicy(env_conf, use_layer_norm=self._use_layer_norm)


# Reactor
class ControlPolicy(PolicyParamsMixin, nn.Module):
    def __init__(self, env_conf, *, use_layer_norm=True):
        super().__init__()
        self.problem_seed = env_conf.problem_seed
        self._env_conf = env_conf

        num_state = env_conf.gym_conf.state_space.shape[0]
        num_action = env_conf.action_space.shape[0]
        self._k = 2

        self._const_scale = 0.5
        self.in_norm = (
            nn.LayerNorm(num_state, elementwise_affine=True)
            if bool(use_layer_norm)
            else None
        )

        self.w_state = nn.Parameter(
            torch.empty((self._k, num_action, num_state), dtype=torch.float32)
        )
        self.w_phase = nn.Parameter(
            torch.empty((self._k, num_action, 2), dtype=torch.float32)
        )
        self.b_err = nn.Parameter(
            torch.zeros((self._k, num_action), dtype=torch.float32)
        )

        self.kp_raw = nn.Parameter(
            torch.full((self._k, num_action), -2.0, dtype=torch.float32)
        )
        self.ki_raw = nn.Parameter(
            torch.full((self._k, num_action), -4.0, dtype=torch.float32)
        )
        self.kd_raw = nn.Parameter(
            torch.full((self._k, num_action), -3.0, dtype=torch.float32)
        )

        self.gate_w = nn.Parameter(
            torch.zeros((self._k, num_state), dtype=torch.float32)
        )
        self.gate_b = nn.Parameter(torch.zeros((self._k,), dtype=torch.float32))

        self.filter_alpha_logit = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.int_leak_logit = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.act_beta_logit = nn.Parameter(torch.tensor(-1.0, dtype=torch.float32))
        self.rate_limit_raw = nn.Parameter(torch.tensor(-1.5, dtype=torch.float32))
        self.aw_gain_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))

        self.phase_omega_raw = nn.Parameter(torch.tensor(-2.0, dtype=torch.float32))
        self.phase_omega_state = nn.Parameter(
            torch.zeros((num_state,), dtype=torch.float32)
        )

        for k in range(self._k):
            nn.init.orthogonal_(self.w_state[k], gain=0.1)
            nn.init.orthogonal_(self.w_phase[k], gain=0.1)

        self.reset_state()

        with torch.inference_mode():
            self._flat_params_init = np.concatenate(
                [p.data.detach().cpu().numpy().reshape(-1) for p in self.parameters()]
            )

    def reset_state(self):
        num_state = self._env_conf.gym_conf.state_space.shape[0]
        num_action = self._env_conf.action_space.shape[0]
        self._z_filt = torch.zeros((num_state,), dtype=torch.float32)
        self._i = torch.zeros((self._k, num_action), dtype=torch.float32)
        self._e_prev = torch.zeros((self._k, num_action), dtype=torch.float32)
        self._a_prev = torch.zeros((num_action,), dtype=torch.float32)
        self._phi = torch.zeros((), dtype=torch.float32)

    def forward(self, state):
        x = state
        if self.in_norm is not None:
            x = self.in_norm(x)

        alpha = torch.sigmoid(self.filter_alpha_logit)
        self._z_filt = (1.0 - alpha) * self._z_filt + alpha * x

        omega = F.softplus(self.phase_omega_raw) + 1e-3
        omega = omega + 0.02 * torch.tanh(
            torch.dot(self.phase_omega_state, self._z_filt)
        )
        omega = torch.clamp(omega, 1e-3, 1.0)
        self._phi = torch.remainder(self._phi + omega, 2.0 * math.pi)
        phase_feat = torch.stack([torch.sin(self._phi), torch.cos(self._phi)], dim=0)

        w_logits = self.gate_w @ self._z_filt + self.gate_b
        w = torch.softmax(w_logits, dim=0)

        leak = torch.sigmoid(self.int_leak_logit)
        e = (
            torch.einsum("kan,n->ka", self.w_state, self._z_filt)
            + torch.einsum("kaj,j->ka", self.w_phase, phase_feat)
            + self.b_err
        )
        self._i = (1.0 - leak) * self._i + e
        de = e - self._e_prev
        self._e_prev = e

        kp = F.softplus(self.kp_raw)
        ki = F.softplus(self.ki_raw)
        kd = F.softplus(self.kd_raw)
        u_k = kp * e + ki * self._i + kd * de
        u = torch.einsum("k,ka->a", w, u_k)

        beta = torch.sigmoid(self.act_beta_logit)
        u_sat = torch.tanh(u)
        a_lp = (1.0 - beta) * self._a_prev + beta * u_sat
        delta = a_lp - self._a_prev
        rate = 0.25 * torch.sigmoid(self.rate_limit_raw)
        delta = rate * torch.tanh(delta / (rate + 1e-6))
        a = self._a_prev + delta
        self._a_prev = a

        aw = F.softplus(self.aw_gain_raw)
        self._i = self._i - aw * (a - a_lp)[None, :]
        return a

    def __call__(self, state):
        state = torch.as_tensor(state, dtype=torch.float32)
        with torch.inference_mode():
            a = self.forward(state)
        return a.detach().cpu().numpy()
