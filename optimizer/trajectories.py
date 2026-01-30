from dataclasses import dataclass

import numpy as np


@dataclass
class Trajectory:
    rreturn: float

    states: np.ndarray
    actions: np.ndarray
    rreturn_se: float = None
    rreturn_est: float = None

    def get_decision_rreturn(self) -> float:
        if self.rreturn_est is None:
            return float(self.rreturn)
        return float(self.rreturn_est)


def _get_bounds_safe(arr, default_fn):
    return default_fn(arr.shape) if np.all(np.isinf(arr)) else arr


def _get_max_steps(env_conf, b_gym):
    if b_gym:
        return env_conf.gym_conf.max_steps
    return getattr(env_conf, "max_steps", 99999)


def _transform_action(action_p, action_space):
    if not hasattr(action_space, "low"):
        return action_p
    return action_space.low + (action_space.high - action_space.low) * (1 + action_p) / 2


def _make_return(policy, return_trajectory):
    if not (hasattr(policy, "wants_vector_return") and bool(policy.wants_vector_return())):
        return return_trajectory
    assert hasattr(policy, "metrics")
    mets = np.asarray(policy.metrics(), dtype=np.float64)
    return np.concatenate([np.asarray([float(return_trajectory)], dtype=np.float64), mets], axis=0)


def collect_trajectory(env_conf, policy, noise_seed=None, show_frames=False):
    b_gym = env_conf.gym_conf is not None
    num_frames_skip = env_conf.gym_conf.num_frames_skip if b_gym and show_frames else 1

    render_mode = "rgb_array" if show_frames else None
    env = env_conf.make(render_mode=render_mode)
    return_trajectory = 0
    traj_states, traj_actions = [], []
    lb = _get_bounds_safe(env.observation_space.low, np.zeros)
    width = _get_bounds_safe(env.observation_space.high - lb, np.ones)
    assert not np.any(np.isinf(lb)), lb
    assert not np.any(np.isinf(width)), width

    def draw():
        import matplotlib.pyplot as plt
        from IPython.display import clear_output

        clear_output(wait=True)
        plt.imshow(env.render())
        plt.title(f"i_iter = {i_iter} return = {return_trajectory:.2f}")
        plt.show()

    if hasattr(policy, "reset_state"):
        policy.reset_state()

    state, _ = env.reset(seed=noise_seed)
    max_steps = _get_max_steps(env_conf, b_gym)
    transform_state = b_gym and env_conf.gym_conf.transform_state

    for i_iter in range(max_steps):
        state_p = (state - lb) / width
        action_p = policy(state_p if transform_state else state)
        action = _transform_action(action_p, env.action_space)
        traj_states.append(state_p)
        traj_actions.append(action_p)
        state, reward, done = env.step(action)[:3]
        return_trajectory += reward
        if show_frames and i_iter % max(1, max_steps // num_frames_skip) == 0:
            draw()
        if done:
            break

    if show_frames:
        draw()
    env.close()
    rreturn = _make_return(policy, return_trajectory)
    return Trajectory(rreturn, np.array(traj_states).T, np.array(traj_actions).T, rreturn_se=None)
