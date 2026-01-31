import numpy as np
from Box2D import b2CircleShape, b2PolygonShape, b2Vec2, b2World


class b2WorldInterface:
    def __init__(self, do_gui=False):
        self.world = b2World(gravity=(0.0, 0.0), doSleep=True)
        self.do_gui = do_gui
        self.TARGET_FPS = 100
        self.TIME_STEP = 1.0 / self.TARGET_FPS
        self.VEL_ITERS, self.POS_ITERS = 10, 10
        self.bodies = []
        self.gui_world = None

    def add_bodies(self, new_bodies):
        if isinstance(new_bodies, list):
            self.bodies += new_bodies
        else:
            self.bodies.append(new_bodies)

    def step(self, show_display=True, idx=0):
        self.world.Step(self.TIME_STEP, self.VEL_ITERS, self.POS_ITERS)


class end_effector:
    def __init__(
        self,
        b2world_interface,
        init_pos,
        base,
        init_angle,
        hand_shape="rectangle",
        hand_size=(0.3, 1),
    ):
        world = b2world_interface.world
        self.hand = world.CreateDynamicBody(position=init_pos, angle=init_angle)
        self.hand_shape = hand_shape
        self.hand_size = hand_size
        if hand_shape == "rectangle":
            rshape = b2PolygonShape(box=hand_size)
            self.forceunit = 30.0
        elif hand_shape == "circle":
            rshape = b2CircleShape(radius=hand_size)
            self.forceunit = 100.0
        elif hand_shape == "polygon":
            rshape = b2PolygonShape(vertices=hand_size)
        else:
            raise Exception("%s is not a correct shape" % hand_shape)

        self.hand.CreateFixture(shape=rshape, density=0.1, friction=0.1)
        self.hand.userData = "hand"

        world.CreateFrictionJoint(
            bodyA=base,
            bodyB=self.hand,
            maxForce=2,
            maxTorque=2,
        )
        b2world_interface.add_bodies(self.hand)

    def set_pos(self, pos, angle):
        self.hand.position = pos
        self.hand.angle = angle

    def apply_wrench(self, rlvel=(0, 0), ravel=0):
        avel = self.hand.angularVelocity
        delta_avel = ravel - avel
        torque = self.hand.mass * delta_avel * 30.0
        self.hand.ApplyTorque(torque, wake=True)

        lvel = self.hand.linearVelocity
        delta_lvel = b2Vec2(rlvel) - b2Vec2(lvel)
        force = self.hand.mass * delta_lvel * self.forceunit
        self.hand.ApplyForce(force, self.hand.position, wake=True)

    def get_state(self, verbose=False):
        state = list(self.hand.position) + [self.hand.angle] + list(self.hand.linearVelocity) + [self.hand.angularVelocity]
        return state


def create_body(base, b2world_interface, body_shape, body_size, body_friction, body_density, obj_loc):
    world = b2world_interface.world

    link = world.CreateDynamicBody(position=obj_loc)
    if body_shape == "rectangle":
        linkshape = b2PolygonShape(box=body_size)
    elif body_shape == "circle":
        linkshape = b2CircleShape(radius=body_size)
    elif body_shape == "polygon":
        linkshape = b2PolygonShape(vertices=body_size)
    else:
        raise Exception("%s is not a correct shape" % body_shape)

    link.CreateFixture(
        shape=linkshape,
        density=body_density,
        friction=body_friction,
    )
    world.CreateFrictionJoint(
        bodyA=base,
        bodyB=link,
        maxForce=5,
        maxTorque=2,
    )

    b2world_interface.add_bodies([link])
    return link


def make_base(table_width, table_length, b2world_interface):
    world = b2world_interface.world
    base = world.CreateStaticBody(
        position=(0, 0),
        shapes=b2PolygonShape(box=(table_length, table_width)),
    )

    b2world_interface.add_bodies([base])
    return base


def run_simulation(
    world,
    body,
    body2,
    robot,
    robot2,
    xvel,
    yvel,
    xvel2,
    yvel2,
    rtor,
    rtor2,
    simulation_steps,
    simulation_steps2,
    rng,
):
    assert rng is not None
    desired_vel = np.array([xvel, yvel])
    rvel = b2Vec2(desired_vel[0] + rng.normal(0, 0.01), desired_vel[1] + rng.normal(0, 0.01))

    desired_vel2 = np.array([xvel2, yvel2])
    rvel2 = b2Vec2(desired_vel2[0] + rng.normal(0, 0.01), desired_vel2[1] + rng.normal(0, 0.01))

    tmax = np.max([simulation_steps, simulation_steps2])
    for t in range(tmax + 100):
        if t < simulation_steps:
            robot.apply_wrench(rvel, rtor)
        if t < simulation_steps2:
            robot2.apply_wrench(rvel2, rtor2)
        world.step()

    return (list(body.position), list(body2.position))
