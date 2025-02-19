import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

class PestControl:
    def __init__(self, stages=25, categories=5, seed=None, dim=200):
        self.stages = stages
        self.categories = categories
        self.seed = seed
        self.dim = dim

        # Set the initial random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize the state
        self.state = np.random.uniform(0, 1, size=(dim,))

        # Define observation space
        self.observation_space = Box(
            low=0.0, 
            high=1.0, 
            shape=(dim,), 
            dtype=np.float32
        )

        # Define action space
        self.action_space = Box(
            low=0.0,
            high=1.0,
            shape=(categories,),
            dtype=np.float32
        )
        print(f"Initialized PestControl with stages={stages}, categories={categories}, seed={seed}")

    def reset(self, seed=None):
        """Reset the environment to an initial state."""
        if seed is not None:
            np.random.seed(seed)
        self.state = np.random.uniform(0, 1, size=(self.dim,))
        return self.state, {}

    def step(self, action):
        """Step the environment with the given action."""
        reward = -np.sum((self.state[:self.categories] - action) ** 2)
        done = bool(np.random.rand() < 0.05)
        self.state = np.clip(self.state - 0.05, 0, 1)
        return self.state, reward, done, {}
    
    def close(self):
        # Perform any cleanup if needed (e.g., closing files, releasing resources)
        pass