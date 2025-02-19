import numpy as np
from problems.pest_control import PestControl

def test_pest_control():
    env = PestControl()

    # Test reset method
    state, _ = env.reset()
    assert isinstance(state, np.ndarray), "State should be a numpy array"
    assert state.shape == (200,), "State should have 200 dimensions"

    # Test step method
    action = np.random.uniform(-1, 1, size=env.action_space.shape)
    next_state, reward, done, _ = env.step(action)
    
    assert isinstance(next_state, np.ndarray), "Next state should be a numpy array"
    assert next_state.shape == (200,), "Next state should have 200 dimensions"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done flag should be a boolean"
    
    # Ensure state updates correctly
    assert not np.array_equal(state, next_state), "State should change after taking an action"