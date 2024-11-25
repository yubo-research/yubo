import numpy as np
import torch
import torch.nn as nn

#src = "https://openreview.net/pdf?id=lsFa23pHCH"


class NNDraw:
    def __init__(self, dim=200, seed=0):
        self.dim = dim
        torch.manual_seed(seed)
        self.model = nn.Sequential(
            nn.Linear(self.dim, 50),
            nn.ReLU(),
            nn.Linear(50, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, mean=0, std=1)
                nn.init.normal_(layer.bias, mean=0, std=1)
    
    def __call__(self, x):
        x = torch.tensor(x, dtype=torch.float32).view(1, -1)
        if x.size(1) != self.dim:
            raise ValueError(f"Input dimension must be {self.dim}")     
        with torch.no_grad():
            return self.model(x).item()

class PestControl:
    def __init__(self, stages=25, categories=5, seed=0):
        self.stages = stages
        self.categories = categories
        np.random.seed(seed)
        self.cost_matrix = np.random.rand(stages, categories)
    
    def __call__(self, interventions):
        if len(interventions) != self.stages:
            raise ValueError(f"Input must have {self.stages} stages.")
        if any(i >= self.categories or i < 0 for i in interventions):
            raise ValueError(f"Each intervention must be between 0 and {self.categories - 1}.")
        total_cost = sum(self.cost_matrix[i, intervention] for i, intervention in enumerate(interventions))
        spread_penalty = sum(interventions) * 0.1
        return total_cost + spread_penalty
