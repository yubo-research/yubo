import numpy as np

from problems.push_utils import b2WorldInterface, create_body, end_effector, make_base, run_simulation


class Push:
    num_dim = 14

    def __init__(self):
        self.xmin = np.array([-5.0, -5.0, -10.0, -10.0, 2.0, 0.0, -5.0, -5.0, -10.0, -10.0, 2.0, 0.0, -5.0, -5.0])
        self.xmax = np.array([5.0, 5.0, 10.0, 10.0, 30.0, 2.0 * np.pi, 5.0, 5.0, 10.0, 10.0, 30.0, 2.0 * np.pi, 5.0, 5.0])

        self.sxy = (0, 2)
        self.sxy2 = (0, -2)
        self.gxy = [4, 3.5]
        self.gxy2 = [-4, 3.5]
        self._rng = None

    def reset(self, noise_seed=None):
        self._rng = np.random.default_rng(noise_seed)

    @property
    def f_max(self):
        return np.linalg.norm(np.array(self.gxy) - np.array(self.sxy)) + np.linalg.norm(np.array(self.gxy2) - np.array(self.sxy2))

    def __call__(self, x):
        assert x.shape == (Push.num_dim,), x.shape
        assert np.all(x >= -1 - 1e-9) and np.all(x <= 1 + 1e-9), x

        argv = self.xmin + (self.xmax - self.xmin) * (x + 1) / 2

        rx = float(argv[0])
        ry = float(argv[1])
        xvel = float(argv[2])
        yvel = float(argv[3])
        simu_steps = int(float(argv[4]) * 10)
        init_angle = float(argv[5])
        rx2 = float(argv[6])
        ry2 = float(argv[7])
        xvel2 = float(argv[8])
        yvel2 = float(argv[9])
        simu_steps2 = int(float(argv[10]) * 10)
        init_angle2 = float(argv[11])
        rtor = float(argv[12])
        rtor2 = float(argv[13])

        initial_dist = self.f_max

        world = b2WorldInterface(False)
        ofriction, odensity, hand_shape, hand_size = 0.01, 0.05, "rectangle", (1, 0.3)

        base = make_base(500, 500, world)
        body = create_body(base, world, "rectangle", (0.5, 0.5), ofriction, odensity, self.sxy)
        body2 = create_body(base, world, "circle", 1, ofriction, odensity, self.sxy2)

        robot = end_effector(world, (rx, ry), base, init_angle, hand_shape, hand_size)
        robot2 = end_effector(world, (rx2, ry2), base, init_angle2, hand_shape, hand_size)
        (ret1, ret2) = run_simulation(world, body, body2, robot, robot2, xvel, yvel, xvel2, yvel2, rtor, rtor2, simu_steps, simu_steps2, rng=self._rng)

        ret1 = np.linalg.norm(np.array(self.gxy) - ret1)
        ret2 = np.linalg.norm(np.array(self.gxy2) - ret2)
        reward = initial_dist - ret1 - ret2
        return -reward
