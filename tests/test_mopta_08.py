import numpy as np
from problems.mopta_08 import Mopta08

def test_mopta08():
    env = Mopta08()

    # Test reset method
    state, _ = env.reset()
    assert isinstance(state, np.ndarray), "State should be a numpy array"
    assert state.shape == (124,), "State should have 124 dimensions"

    # Test step method
    action = np.random.uniform(-1, 1, size=env.num_params)
    next_state, reward, done, _ = env.step(action)

    assert isinstance(next_state, np.ndarray), "Next state should be a numpy array"
    assert next_state.shape == (124,), "Next state should have 124 dimensions"
    assert isinstance(reward, float), "Reward should be a float"
    assert isinstance(done, bool), "Done flag should be a boolean"

    # Ensure step updates state meaningfully
    assert not np.array_equal(state, next_state), "State should change after an action"
    
    # Check termination conditions
    assert done in [True, False], "Done should be a boolean"