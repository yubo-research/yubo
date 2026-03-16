"""Tests for GACActorNet."""

import numpy as np
import torch
import torch.nn as nn

from rl.core.runtime import ObsScaler
from rl.gac_actor import GACActorNet


def test_gac_actor_act_deterministic():
    obs_dim, act_dim = 4, 2
    backbone = nn.Linear(obs_dim, 8)
    direction_head = nn.Linear(8, act_dim)
    concentration_head = nn.Linear(8, 1)
    obs_scaler = ObsScaler(
        lb=np.zeros(obs_dim, dtype=np.float32),
        width=np.ones(obs_dim, dtype=np.float32),
    )
    actor = GACActorNet(
        backbone=backbone,
        direction_head=direction_head,
        concentration_head=concentration_head,
        obs_scaler=obs_scaler,
        act_dim=act_dim,
        action_radius=2.5,
    )
    obs = torch.randn(3, obs_dim, dtype=torch.float32)
    action = actor.act(obs)
    assert action.shape == (3, act_dim)
    assert torch.all((action >= -1.0) & (action <= 1.0))


def test_gac_actor_sample_stochastic():
    obs_dim, act_dim = 4, 2
    backbone = nn.Linear(obs_dim, 8)
    direction_head = nn.Linear(8, act_dim)
    concentration_head = nn.Linear(8, 1)
    obs_scaler = ObsScaler(
        lb=np.zeros(obs_dim, dtype=np.float32),
        width=np.ones(obs_dim, dtype=np.float32),
    )
    actor = GACActorNet(
        backbone=backbone,
        direction_head=direction_head,
        concentration_head=concentration_head,
        obs_scaler=obs_scaler,
        act_dim=act_dim,
    )
    obs = torch.randn(2, obs_dim, dtype=torch.float32)
    action, kappa = actor.sample(obs, deterministic=False)
    assert action.shape == (2, act_dim)
    assert kappa.shape == (2,)
    assert torch.all((action >= -1.0) & (action <= 1.0))
