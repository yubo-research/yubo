import matplotlib.pyplot as plt
import numpy as np

from problems.mlp_policy import MLPPolicyFactory


class MockActionSpace:
    def __init__(self, action_size):
        self.action_size = action_size
        self.low = np.array([-1.0] * action_size)
        self.high = np.array([1.0] * action_size)

    @property
    def shape(self):
        return (self.action_size,)


class MockObservationSpace:
    def __init__(self, obs_size):
        self.obs_size = obs_size
        self.low = np.array([0.0] * obs_size)
        self.high = np.array([1.0] * obs_size)


class MockStateSpace:
    def __init__(self, obs_size):
        self.shape = (obs_size,)


class MockGymConf:
    def __init__(self, obs_size, max_steps=250, transform_state=True):
        self.state_space = MockStateSpace(obs_size)
        self.max_steps = max_steps
        self.transform_state = transform_state
        self.num_frames_skip = 10


def khepera_maze_conf():
    class KheperaMazeEnvConf:
        def __init__(self):
            self.env_name = "khepera-maze"
            self.gym_conf = MockGymConf(5, max_steps=250, transform_state=True)
            self.policy_class = MLPPolicyFactory((8,))
            self.action_space = MockActionSpace(2)
            self.problem_seed = None

        def make(self, **kwargs):
            return KheperaMazeEnv()

    return KheperaMazeEnvConf()


_max = -1e99


class KheperaMazeEnv:
    def __init__(self):
        self.width = 1.0
        self.height = 1.0
        self.robot_radius = 0.02
        self.dt = 0.05
        self.wheel_base = 0.05
        self.max_speed = 0.1
        self.goal = np.array([0.15, 0.9])
        self.goal_radius = 0.04
        self.max_steps = 250
        self.laser_angles = np.array([-np.pi / 4, 0, np.pi / 4])
        self.laser_range = 0.2
        self._define_maze()
        self.observation_space = MockObservationSpace(5)
        self.action_space = MockActionSpace(2)
        self.reset()

    def close(self):
        pass

    def _define_maze(self):
        self.walls = [
            # Border walls
            (0, 0, 1, 0),  # Bottom
            (1, 0, 1, 1),  # Right
            (1, 1, 0, 1),  # Top
            (0, 1, 0, 0),  # Left
            # Internal walls (reference standard maze)
            (0.25, 0.25, 0.25, 0.75),  # Vertical middle
            (0.14, 0.45, 0.0, 0.65),  # Obstacle top left
            (0.25, 0.75, 0.0, 0.8),  # Wall to target
            (0.25, 0.75, 0.66, 0.875),  # Wall top right
            (0.355, 0.0, 0.525, 0.185),  # Obstacle bottom right
            (0.25, 0.5, 0.75, 0.215),  # Funnel bottom
            (1.0, 0.25, 0.435, 0.55),  # Funnel top
            (0.0, 0.8, 0.0, 1.0),  # Wall top left
            (0.355, 0.0, 1.0, 0.0),  # Wall bottom right
        ]

    def reset(self, seed=None):
        # ok to ignore seed
        self.state = np.array([0.15, 0.15, np.pi / 2])
        self.steps = 0
        return self._get_obs(), {}

    def step(self, action):
        l_speed = np.clip(action[0], -self.max_speed, self.max_speed)
        r_speed = np.clip(action[1], -self.max_speed, self.max_speed)
        x, y, theta = self.state
        v = (l_speed + r_speed) / 2
        omega = (r_speed - l_speed) / self.wheel_base
        nx = x + v * np.cos(theta) * self.dt
        ny = y + v * np.sin(theta) * self.dt
        ntheta = (theta + omega * self.dt) % (2 * np.pi)
        next_state = np.array([nx, ny, ntheta])
        if self._collides(next_state):
            next_state[:2] = self.state[:2]
        self.state = next_state
        self.steps += 1
        done = self._at_goal() or self.steps >= self.max_steps
        reward = -np.linalg.norm(self.state[:2] - self.goal)
        global _max
        if reward > _max:
            _max = reward
            print("R:", _max, reward)
        return self._get_obs(), reward, done, {}

    def _get_obs(self):
        lasers = self._laser_measures()
        bumpers = self._bumper_measures()
        return np.concatenate([lasers, bumpers])

    def _laser_measures(self):
        x, y, theta = self.state
        measures = []
        for angle in self.laser_angles:
            laser_theta = theta + angle

            for d in np.linspace(0, self.laser_range, 20):
                tx = x + d * np.cos(laser_theta)
                ty = y + d * np.sin(laser_theta)
                if self._point_collides(tx, ty):
                    measures.append(d)
                    break
            else:
                measures.append(-1.0)
        return np.array(measures)

    def _bumper_measures(self):
        x, y, theta = self.state
        left_angle = theta - np.pi / 2
        right_angle = theta + np.pi / 2
        left_x = x + self.robot_radius * np.cos(left_angle)
        left_y = y + self.robot_radius * np.sin(left_angle)
        right_x = x + self.robot_radius * np.cos(right_angle)
        right_y = y + self.robot_radius * np.sin(right_angle)
        left = 1.0 if self._point_collides(left_x, left_y) else -1.0
        right = 1.0 if self._point_collides(right_x, right_y) else -1.0
        return np.array([left, right])

    def _point_collides(self, x, y):
        for x1, y1, x2, y2 in self.walls:
            if self._point_line_dist(x, y, x1, y1, x2, y2) < self.robot_radius:
                return True
        return False

    def _point_line_dist(self, px, py, x1, y1, x2, y2):
        line_mag = np.hypot(x2 - x1, y2 - y1)
        if line_mag < 1e-8:
            return np.hypot(px - x1, py - y1)
        u = ((px - x1) * (x2 - x1) + (py - y1) * (y2 - y1)) / (line_mag**2)
        u = np.clip(u, 0, 1)
        ix = x1 + u * (x2 - x1)
        iy = y1 + u * (y2 - y1)
        return np.hypot(px - ix, py - iy)

    def _collides(self, state):
        x, y, _ = state
        return self._point_collides(x, y)

    def _at_goal(self):
        return np.linalg.norm(self.state[:2] - self.goal) < self.goal_radius

    def render(self, ax=None):
        import matplotlib.pyplot as plt

        close_fig = False
        if ax is None:
            fig, ax = plt.subplots(figsize=(5, 5))
            close_fig = True
        ax.clear()
        for x1, y1, x2, y2 in self.walls:
            ax.plot([x1, x2], [y1, y2], "k-", lw=3)
        ax.plot(self.goal[0], self.goal[1], "g+", markersize=16, markeredgewidth=3)
        circle = plt.Circle(self.goal, self.goal_radius, color="g", fill=False, linestyle="--")
        ax.add_patch(circle)
        x, y, theta = self.state
        robot = plt.Circle((x, y), self.robot_radius, color="b")
        ax.add_patch(robot)
        hx = x + self.robot_radius * np.cos(theta)
        hy = y + self.robot_radius * np.sin(theta)
        ax.plot([x, hx], [y, hy], "r-", lw=2)
        ax.set_xlim(0, self.width)
        ax.set_ylim(0, self.height)
        ax.set_aspect("equal")
        ax.axis("off")
        if close_fig:
            import io

            buf = io.BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
            buf.seek(0)
            import matplotlib.pyplot as plt

            img = plt.imread(buf)
            plt.close(fig)
            return (img[:, :, :3] * 255).astype("uint8") if img.dtype == float else img[:, :, :3]
        return None
