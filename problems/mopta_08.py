import numpy as np
from gymnasium.spaces import Box

class Mopta08:

    def __init__(self, num_params = 124):
        self.num_params = num_params
        self.state = None  

        # Define lower and upper bounds
        self.lb = np.zeros(self.num_params)  # Lower bound of state
        self.ub = np.ones(self.num_params)  # Upper bound of state

        # Define observation space
        self.observation_space = type('', (), {})()  
        self.observation_space.low = self.lb  
        self.observation_space.high = self.ub  

        # Define action space
        self.action_space = Box(low=-0.1, high=0.1, shape=(self.num_params,), dtype=np.float32)  # Define action space
    
    def reset(self, seed=None):
        """Resets the environment and returns the initial state"""
        if seed is not None:
            np.random.seed(seed)  
        self.state = np.random.uniform(self.lb, self.ub)  
        return self.state, {}

    def step(self, action):
        """Applies action and returns next state, reward, and done flag"""
        print(f"DEBUG: action = {action}")  # Print action values
        action = np.clip(action, self.action_space.low, self.action_space.high)  # Ensure action is valid
        self.state = np.clip(self.state + 0.5 * action, self.lb, self.ub)  # Change 0.1 to 0.5       
        optimal_solution = np.full(self.num_params, 0.5)  # Define an arbitrary optimal solution
        reward = -np.sum((self.state - optimal_solution) ** 2)
        done = bool(np.random.rand() < 0.05)        
        print(f"DEBUG: state = {self.state}, reward = {reward}, done = {done}")  # Print debug info
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
        return -(objective + penalty)
    
    def close(self):
        # Perform any cleanup if needed (e.g., closing files, releasing resources)
        pass