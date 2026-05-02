from __future__ import annotations

import torch.nn as nn


def pm_backbone_head_init(self):
    nn.Module.__init__(self)
    self.actor_backbone = nn.Linear(1, 1)
    self.actor_head = nn.Linear(1, 1)
    self.critic_backbone = nn.Linear(1, 1)
    self.critic_head = nn.Linear(1, 1)


def make_pm_module_type(name: str = "PM"):
    return type(name, (nn.Module,), {"__init__": pm_backbone_head_init})
