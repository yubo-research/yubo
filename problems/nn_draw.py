import gymnasium as gym
from gymnasium.spaces import Box
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn


class NNDraw:
    def __init__(self, dim=200, seed=None):
        self.dim = dim
        self.seed = seed

        # Set random seed
        if seed is not None:
            np.random.seed(seed)

        # Initialize state
        self.state = None  # Will be initialized in reset
        print(f"NNDraw: Initialized with dim={dim}, seed={seed}")

        # Define observation space
        self.observation_space = Box(
            low=0.0,
            high=1.0,
            shape=(dim,),
            dtype=np.float32
        )
        self.action_space = Box(
            low=-1.0,
            high=1.0,
            shape=(dim,),
            dtype=np.float32
        )

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.state = np.random.uniform(0, 1, size=(self.dim,))
        print(f"NNDraw: Reset with seed={seed}, state={self.state}")
        return self.state, {}

    def step(self, action):
        action = np.clip(action, self.action_space.low, self.action_space.high)
        self.state = np.clip(self.state + action, 0, 1)

        # Example reward: sum of state elements close to 0.5
        reward = -np.sum((self.state - 0.5) ** 2)

        # Termination condition: state within a threshold
        done = bool(np.random.rand() < 0.05)

        print(f"NNDraw: Step -> state={self.state}, reward={reward}, done={done}")
        return self.state, reward, done, {}

    def close(self):
        # Perform any cleanup if needed (e.g., closing files, releasing resources)
        pass