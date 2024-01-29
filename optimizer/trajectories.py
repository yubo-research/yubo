from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from IPython.display import clear_output


@dataclass
class Trajectory:
    rreturn: float
    states: np.ndarray
    actions: np.ndarray


def collect_trajectory(env_conf, policy, noise_seed=None, show_frames=False):
    b_gym = env_conf.gym_conf is not None
    if b_gym and show_frames:
        num_frames_skip = env_conf.gym_conf.num_frames_skip

    render_mode = "rgb_array" if show_frames else None
    env = env_conf.make(render_mode=render_mode)
    done = False
    return_trajectory = 0
    traj_states, traj_actions = [], []
    lb = env.observation_space.low
    if np.all(np.isinf(lb)):
        # lb = -1 * np.ones(shape=lb.shape)
        lb = np.zeros(shape=lb.shape)
    assert not np.any(np.isinf(lb)), lb

    width = env.observation_space.high - lb
    if np.all(np.isinf(width)):
        # width = 2 * np.ones(shape=width.shape)
        width = np.ones(shape=width.shape)
    assert not np.any(np.isinf(width)), width

    def draw():
        clear_output(wait=True)
        plt.imshow(env.render())
        plt.title(f"i_iter = {i_iter} return = {return_trajectory:.2f}")
        plt.show()

    state, _ = env.reset(seed=noise_seed)
    if b_gym:
        max_steps = env_conf.gym_conf.max_steps
    else:
        max_steps = 99999

    for i_iter in range(max_steps):
        # assert np.all(state >= lb), (state, lb)

        state_p = (state - lb) / width
        if b_gym and env_conf.gym_conf.transform_state:
            action_p = policy(state_p)  # in [-1,1]
        else:
            action_p = policy(state)

        if hasattr(env.action_space, "low"):
            action = env.action_space.low + (env.action_space.high - env.action_space.low) * (1 + action_p) / 2
        else:
            action = action_p

        traj_states.append(state_p)
        traj_actions.append(action_p)

        state, reward, done = env.step(action)[:3]
        return_trajectory += reward
        if show_frames and i_iter % max(1, (max_steps // num_frames_skip)) == 0:
            draw()
        if done:
            break
    if show_frames:
        draw()
    env.close()
    return Trajectory(return_trajectory, np.array(traj_states).T, np.array(traj_actions).T)
