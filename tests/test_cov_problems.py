import numpy as np

from problems.benchmark_functions_1c import Schaffer2, Schaffer4
from problems.benchmark_util import mk_4d
from problems.np_policy_util import set_params_pm1
from problems.push import Push
from problems.push_utils import (
    b2WorldInterface,
    create_body,
    end_effector,
    make_base,
    run_simulation,
)


def test_schaffer2():
    f = Schaffer2()
    x = np.array([0.5, 0.3, 0.1])
    y = f(x)
    assert np.isfinite(y)


def test_schaffer4():
    f = Schaffer4()
    x = np.array([0.2, 0.4, 0.6, 0.8])
    y = f(x)
    assert np.isfinite(y)


def test_mk_4d():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    y = mk_4d(x)
    assert y.shape == (4,)
    assert np.isfinite(y).all()


class _MockPolicy:
    _num_p = 3
    _x_orig = None

    def _set_derived(self, x):
        pass


def test_set_params_pm1():
    policy = _MockPolicy()
    x = np.array([0.5, -0.3, 0.8])
    set_params_pm1(policy, x)
    assert policy._x_orig is not None
    assert np.array_equal(policy._x_orig, x)


def test_push_f_max():
    push = Push()
    f_max = push.f_max
    assert np.isfinite(f_max)
    assert f_max > 0


def test_b2worldinterface_init():
    world = b2WorldInterface(False)
    assert world.world is not None
    assert world.do_gui is False


def test_b2worldinterface_add_bodies():
    world = b2WorldInterface(False)
    body1 = world.world.CreateStaticBody(position=(0, 0))
    body2 = world.world.CreateStaticBody(position=(1, 1))
    world.add_bodies([body1, body2])
    assert len(world.bodies) == 2
    world.add_bodies(body1)
    assert len(world.bodies) == 3


def test_end_effector_init():
    world = b2WorldInterface(False)
    base = make_base(10, 10, world)
    ee = end_effector(world, (0, 0), base, 0.0, "rectangle", (0.3, 1))
    assert ee.hand is not None
    assert ee.hand_shape == "rectangle"
    assert ee.hand_size == (0.3, 1)


def test_end_effector_set_pos():
    world = b2WorldInterface(False)
    base = make_base(10, 10, world)
    ee = end_effector(world, (0, 0), base, 0.0, "rectangle", (0.3, 1))
    ee.set_pos((1.0, 2.0), 0.5)
    assert ee.hand.position == (1.0, 2.0)
    assert ee.hand.angle == 0.5


def test_end_effector_apply_wrench():
    world = b2WorldInterface(False)
    base = make_base(10, 10, world)
    ee = end_effector(world, (0, 0), base, 0.0, "rectangle", (0.3, 1))
    ee.apply_wrench((1.0, 0.0), 0.1)
    # Just verify it doesn't raise an exception
    assert ee.hand is not None


def test_end_effector_get_state():
    world = b2WorldInterface(False)
    base = make_base(10, 10, world)
    ee = end_effector(world, (0, 0), base, 0.0, "rectangle", (0.3, 1))
    state = ee.get_state()
    assert len(state) == 6  # position (2) + angle (1) + linearVelocity (2) + angularVelocity (1)
    assert all(np.isfinite(s) for s in state)


def test_create_body():
    world = b2WorldInterface(False)
    base = make_base(10, 10, world)
    body = create_body(base, world, "rectangle", (0.5, 0.5), 0.01, 0.05, (1, 1))
    assert body is not None
    assert body.position == (1, 1)


def test_make_base():
    world = b2WorldInterface(False)
    base = make_base(500, 500, world)
    assert base is not None
    assert base.type == 0  # static body


def test_run_simulation():
    world = b2WorldInterface(False)
    base = make_base(10, 10, world)
    body = create_body(base, world, "rectangle", (0.5, 0.5), 0.01, 0.05, (0, 0))
    body2 = create_body(base, world, "circle", 0.5, 0.01, 0.05, (1, 1))
    robot = end_effector(world, (0, 0), base, 0.0, "rectangle", (0.3, 1))
    robot2 = end_effector(world, (1, 1), base, 0.0, "rectangle", (0.3, 1))
    rng = np.random.default_rng(42)
    ret1, ret2 = run_simulation(world, body, body2, robot, robot2, 0.1, 0.1, 0.1, 0.1, 0.0, 0.0, 10, 10, rng)
    assert len(ret1) == 2
    assert len(ret2) == 2
    assert all(np.isfinite(x) for x in ret1)
    assert all(np.isfinite(x) for x in ret2)
