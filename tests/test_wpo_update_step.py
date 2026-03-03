import copy

import torch
import torch.nn as nn
import torch.optim as optim
from torch.distributions import Normal

from rl.core.wpo_update import WPOUpdateBatch, WPOUpdateHyperParams, WPOUpdateModules, WPOUpdateOptimizers, wpo_update_step


class _TinyActor(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.act_dim = int(act_dim)
        self.net = nn.Linear(obs_dim, 2 * act_dim)

    def mean_log_std(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        out = self.net(obs)
        mean = out[..., : self.act_dim]
        log_std = out[..., self.act_dim :].clamp(-3.0, 1.0)
        return (mean, log_std)

    def sample(self, obs: torch.Tensor, *, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        mean, log_std = self.mean_log_std(obs)
        if deterministic:
            action = torch.tanh(mean)
            return (action, torch.zeros(action.shape[0], dtype=action.dtype, device=action.device))
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        z = dist.rsample()
        action = torch.tanh(z)
        log_prob = dist.log_prob(z) - torch.log(1.0 - action.pow(2) + 1e-6)
        return (action, log_prob.sum(dim=-1))

    def log_prob_from_action(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        mean, log_std = self.mean_log_std(obs)
        std = torch.exp(log_std)
        dist = Normal(mean, std)
        action_clamped = action.clamp(-1.0 + 1e-6, 1.0 - 1e-6)
        pre_tanh = torch.atanh(action_clamped)
        log_prob = dist.log_prob(pre_tanh) - torch.log(1.0 - action_clamped.pow(2) + 1e-6)
        return log_prob.sum(dim=-1)


class _TinyQ(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.net = nn.Linear(obs_dim + act_dim, 1)

    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, act], dim=-1)).squeeze(-1)


def test_wpo_update_step_smoke():
    obs_dim = 5
    act_dim = 3
    actor = _TinyActor(obs_dim, act_dim)
    actor_target = copy.deepcopy(actor).eval()
    q1 = _TinyQ(obs_dim, act_dim)
    q2 = _TinyQ(obs_dim, act_dim)
    q1_target = copy.deepcopy(q1).eval()
    q2_target = copy.deepcopy(q2).eval()
    for module in (q1_target, q2_target, actor_target):
        for param in module.parameters():
            param.requires_grad_(False)

    log_alpha_mean = nn.Parameter(torch.zeros(act_dim))
    log_alpha_stddev = nn.Parameter(torch.zeros(act_dim))
    modules = WPOUpdateModules(
        actor=actor,
        actor_target=actor_target,
        q1=q1,
        q2=q2,
        q1_target=q1_target,
        q2_target=q2_target,
        log_alpha_mean=log_alpha_mean,
        log_alpha_stddev=log_alpha_stddev,
    )
    optimizers = WPOUpdateOptimizers(
        actor=optim.Adam(actor.parameters(), lr=1e-3),
        critic=optim.Adam(list(q1.parameters()) + list(q2.parameters()), lr=1e-3),
        dual=optim.Adam([log_alpha_mean, log_alpha_stddev], lr=1e-3),
    )
    batch = WPOUpdateBatch(
        obs=torch.randn(16, obs_dim),
        act=torch.randn(16, act_dim).tanh(),
        rew=torch.randn(16),
        nxt=torch.randn(16, obs_dim),
        done=torch.zeros(16),
    )
    hyper = WPOUpdateHyperParams(
        gamma=0.99,
        tau=0.01,
        num_samples=4,
        epsilon_mean=0.0025,
        epsilon_stddev=1e-6,
        policy_loss_scale=1.0,
        kl_loss_scale=1.0,
        dual_loss_scale=1.0,
        per_dim_constraining=True,
        squashing_type="identity",
    )
    actor_loss, critic_loss, dual_loss, alpha_mean, alpha_stddev = wpo_update_step(modules, optimizers, batch, hyper)
    assert torch.isfinite(torch.tensor(actor_loss))
    assert torch.isfinite(torch.tensor(critic_loss))
    assert torch.isfinite(torch.tensor(dual_loss))
    assert alpha_mean > 0.0
    assert alpha_stddev > 0.0
