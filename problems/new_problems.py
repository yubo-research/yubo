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
        done = np.all(np.abs(self.state - 0.5) < 0.1)

        print(f"NNDraw: Step -> state={self.state}, reward={reward}, done={done}")
        return self.state, reward, done, {}

    def close(self):
        # Perform any cleanup if needed (e.g., closing files, releasing resources)
        pass

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
        done = np.all(self.state < 0.1)
        self.state = np.clip(self.state - 0.05, 0, 1)
        return self.state, reward, done, {}
    
    def close(self):
        # Perform any cleanup if needed (e.g., closing files, releasing resources)
        pass
       
'''env = NNDraw(dim=200)
state, _ = env.reset()

for i in range(10):
    action = np.random.uniform(-0.1, 0.1, size=state.shape)
    state, reward, done, _ = env.step(action)
    print(f"Step {i}: State = {state}, Reward = {reward}")

class Mopta08:


    def __init__(self):
        super(Mopta08, self).__init__()
        self.dims = 124
        self.lb = np.zeros(self.dims)
        self.ub = np.ones(self.dims)

        # Define observation and action spaces
        self.observation_space = spaces.Box(low=self.lb, high=self.ub, dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.dims,), dtype=np.float32)

        # Initialize random state
        self.state = self.reset()

    def reset(self):
        self.state = np.random.uniform(self.lb, self.ub)
        return self.state

    def step(self, action):
        # Ensure action is within the allowed range
        action = np.clip(action, -1.0, 1.0)

        # Apply action as an update to the current state
        self.state = np.clip(self.state + 0.1 * action, self.lb, self.ub)

        # Evaluate the function
        reward = self._evaluate(self.state)

        # Mopta08 is a static optimization problem, so it never "ends"
        done = False
        return self.state, reward, done, {}

    def _evaluate(self, x):
        # Example nonlinear objective function (replace with actual equation if available)
        objective = np.sum((x - 0.5) ** 2)

        # Example soft constraints (penalized if exceeded)
        constraints = np.array([
            np.sum(x) - 50,  # Constraint: Sum of inputs should be <= 50
            30 - np.sum(x[:10]),  # Constraint: First 10 parameters sum should be >= 30
        ])

        # Apply penalties for constraint violations
        penalty = 10 * np.sum(np.clip(constraints, a_min=0, a_max=None))
        return -(objective + penalty)'''