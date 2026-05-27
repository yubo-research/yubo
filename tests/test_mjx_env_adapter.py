from __future__ import annotations

import sys
from types import ModuleType, SimpleNamespace

import pytest


def test_gymnasium_tag_parse_and_registry_surface() -> None:
    from problems.jax_env_core import supported_jax_env_tags, supports_jax_env_tag
    from problems.mjx_env import is_gymnasium_env_tag, parse_gymnasium_env_id

    assert parse_gymnasium_env_id("gymnasium:HalfCheetah-v5") == "HalfCheetah-v5"
    assert is_gymnasium_env_tag("gymnasium:Ant-v5")
    assert supports_jax_env_tag("gymnasium:HalfCheetah-v5")
    assert "gymnasium:HalfCheetah-v5" in supported_jax_env_tags()
    assert "mjx:xml:/path/to/model.xml" not in supported_jax_env_tags()


def test_gymnasium_tag_parse_rejects_empty_id() -> None:
    from problems.mjx_env import parse_gymnasium_env_id

    with pytest.raises(ValueError, match="missing an env id"):
        parse_gymnasium_env_id("gymnasium:")


def test_jax_env_factory_routes_gymnasium_to_mjx(monkeypatch) -> None:
    import problems.jax_env_factory as factory

    calls = []

    def fake_gymnasium_mjx_adapter(env_name, *, jax, jnp):
        calls.append((env_name, jax, jnp))
        return SimpleNamespace()

    monkeypatch.setattr(factory, "GymnasiumMJXAdapter", fake_gymnasium_mjx_adapter)

    adapter = factory.make_jax_env_adapter("gymnasium:HalfCheetah-v5", jax="jax", jnp="jnp")

    assert isinstance(adapter, SimpleNamespace)
    assert calls == [("gymnasium:HalfCheetah-v5", "jax", "jnp")]


def test_mjx_adapter_loads_verified_gymnasium_model(monkeypatch) -> None:
    import problems.mjx_env as mjx_env

    closed = []
    model = SimpleNamespace()
    unwrapped = SimpleNamespace(
        model=model,
        spec=SimpleNamespace(id="HalfCheetah-v5", max_episode_steps=1000),
        _exclude_current_positions_from_observation=True,
        _forward_reward_weight=1.0,
        _ctrl_cost_weight=0.1,
        _reset_noise_scale=0.1,
        frame_skip=5,
    )
    env = SimpleNamespace(unwrapped=unwrapped, close=lambda: closed.append(True))

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: env
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    loaded_model, spec = mjx_env._load_gymnasium_env_spec("HalfCheetah-v5")
    assert loaded_model is model
    assert spec.env_id == "HalfCheetah-v5"
    assert spec.frame_skip == 5
    assert closed == [True]


def test_mjx_adapter_rejects_unverified_gymnasium_semantics(monkeypatch) -> None:
    import problems.mjx_env as mjx_env

    closed = []
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(model=SimpleNamespace(), spec=SimpleNamespace(id="Other-v0")),
        close=lambda: closed.append(True),
    )

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: env
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)

    with pytest.raises(ValueError, match="no verified MuJoCo semantics"):
        mjx_env._load_gymnasium_env_spec("Other-v0")
    assert closed == [True]


def test_halfcheetah_spec_matches_gymnasium_v5_defaults() -> None:
    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    env = SimpleNamespace(
        spec=SimpleNamespace(id="HalfCheetah-v5"),
        _exclude_current_positions_from_observation=True,
        _forward_reward_weight=1.0,
        _ctrl_cost_weight=0.1,
        _reset_noise_scale=0.1,
        frame_skip=5,
    )

    spec = resolve_gymnasium_mujoco_spec(env)

    assert spec is not None
    assert spec.bindings["obs_qpos_start"] == 1
    assert spec.bindings["forward_reward_weight"] == 1.0
    assert spec.bindings["ctrl_cost_weight"] == 0.1
    assert spec.reset_noise_scale == 0.1
    assert spec.frame_skip == 5


def test_halfcheetah_mjx_reward_uses_gymnasium_formula() -> None:
    import numpy as np

    from problems.gymnasium_mujoco_specs import (
        resolve_gymnasium_mujoco_spec,
        supported_gymnasium_mujoco_specs,
    )

    env = SimpleNamespace(
        spec=SimpleNamespace(id="HalfCheetah-v5"),
        _exclude_current_positions_from_observation=True,
        _forward_reward_weight=1.0,
        _ctrl_cost_weight=0.1,
        _reset_noise_scale=0.1,
        frame_skip=5,
    )
    spec = resolve_gymnasium_mujoco_spec(env)
    state = SimpleNamespace(qpos=np.array([1.0], dtype=np.float32), time=np.float32(0.0))
    next_state = SimpleNamespace(qpos=np.array([1.5], dtype=np.float32), time=np.float32(0.05))
    action = np.array([1.0, -2.0], dtype=np.float32)

    reward, info = spec.reward_info(state, next_state, action, SimpleNamespace(), np)

    assert reward == pytest.approx(9.5)
    assert info["x_position"] == pytest.approx(1.5)
    assert info["x_velocity"] == pytest.approx(10.0)
    assert info["reward_forward"] == pytest.approx(10.0)
    assert info["reward_ctrl"] == pytest.approx(-0.5)
    assert supported_gymnasium_mujoco_specs() == ("HalfCheetah-v5",)


def test_gymnasium_mujoco_spec_wrapper_methods() -> None:
    import numpy as np

    from problems.gymnasium_mujoco_specs import resolve_gymnasium_mujoco_spec

    env = SimpleNamespace(
        spec=SimpleNamespace(id="HalfCheetah-v5"),
        _exclude_current_positions_from_observation=False,
        _forward_reward_weight=1.0,
        _ctrl_cost_weight=0.1,
        _reset_noise_scale=0.1,
        frame_skip=5,
    )
    spec = resolve_gymnasium_mujoco_spec(env)
    model = SimpleNamespace(nq=2, nv=1)
    data = SimpleNamespace(
        qpos=np.asarray([1.0, 2.0], dtype=np.float32),
        qvel=np.asarray([3.0], dtype=np.float32),
    )

    assert spec.obs_dim(model) == 3
    np.testing.assert_array_equal(
        spec.obs(data, model, np),
        np.asarray([1.0, 2.0, 3.0], dtype=np.float32),
    )
    assert spec.terminated(data, np) == np.asarray(False, dtype=bool)


def test_gymnasium_mjx_state_is_named_tuple() -> None:
    from problems.mjx_env import GymnasiumMJXState

    state = GymnasiumMJXState(data="data", steps=3)

    assert state.data == "data"
    assert state.steps == 3
    assert tuple(state) == ("data", 3)


def test_gymnasium_mjx_adapter_uses_specs_frame_skip_and_time_limit(
    monkeypatch,
) -> None:
    import numpy as np

    from problems.mjx_env import GymnasiumMJXAdapter

    def make_fake_data(*, qpos=None, qvel=None, ctrl=None, time=0.0):
        data = SimpleNamespace(
            qpos=np.asarray([0.0, 0.0] if qpos is None else qpos, dtype=np.float32),
            qvel=np.asarray([0.0] if qvel is None else qvel, dtype=np.float32),
            ctrl=np.asarray([0.0] if ctrl is None else ctrl, dtype=np.float32),
            time=np.float32(time),
        )
        data.replace = lambda **kwargs: make_fake_data(
            qpos=kwargs.get("qpos", data.qpos),
            qvel=kwargs.get("qvel", data.qvel),
            ctrl=kwargs.get("ctrl", data.ctrl),
            time=kwargs.get("time", data.time),
        )
        return data

    def fake_scan(fn, carry, xs=None, length=1):
        del xs
        for _ in range(int(length)):
            carry, _out = fn(carry, None)
        return carry, None

    def fake_tree_map(fn, left, right):
        if hasattr(left, "qpos"):
            return make_fake_data(
                qpos=fn(left.qpos, right.qpos),
                qvel=fn(left.qvel, right.qvel),
                ctrl=fn(left.ctrl, right.ctrl),
                time=fn(left.time, right.time),
            )
        return fn(left, right)

    model = SimpleNamespace(
        nq=2,
        nv=1,
        nu=1,
        qpos0=np.asarray([0.0, 0.0], dtype=np.float32),
        actuator_ctrlrange=np.asarray([[-2.0, 2.0]], dtype=np.float32),
        actuator_ctrllimited=np.asarray([True]),
    )
    unwrapped = SimpleNamespace(
        model=model,
        spec=SimpleNamespace(id="HalfCheetah-v5", max_episode_steps=1),
        _exclude_current_positions_from_observation=True,
        _forward_reward_weight=1.0,
        _ctrl_cost_weight=0.1,
        _reset_noise_scale=0.1,
        frame_skip=5,
    )
    gymnasium = ModuleType("gymnasium")
    gymnasium.make = lambda _env_id: SimpleNamespace(unwrapped=unwrapped, close=lambda: None)
    spaces_mod = ModuleType("gymnax.environments.spaces")
    spaces_mod.Box = lambda *, low, high, shape, dtype: SimpleNamespace(low=low, high=high, shape=shape, dtype=dtype)
    environments_mod = ModuleType("gymnax.environments")
    environments_mod.spaces = spaces_mod
    gymnax_mod = ModuleType("gymnax")
    gymnax_mod.environments = environments_mod
    mujoco_mod = ModuleType("mujoco")
    mujoco_mod.mjx = SimpleNamespace(
        put_model=lambda model: model,
        make_data=lambda _model: make_fake_data(),
        forward=lambda _model, data: data,
        step=lambda _model, data: data.replace(
            qpos=data.qpos + np.asarray([0.1, 0.0], dtype=np.float32),
            time=data.time + np.float32(0.01),
        ),
    )
    fake_jax = SimpleNamespace(
        random=SimpleNamespace(
            split=lambda _key: ("left", "right"),
            uniform=lambda _key, shape, *, minval, maxval, dtype: np.zeros(shape, dtype=dtype),
            normal=lambda _key, shape, *, dtype: np.zeros(shape, dtype=dtype),
        ),
        lax=SimpleNamespace(scan=fake_scan),
        tree_util=SimpleNamespace(tree_map=fake_tree_map),
    )
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    monkeypatch.setitem(sys.modules, "gymnax", gymnax_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments.spaces", spaces_mod)
    monkeypatch.setitem(sys.modules, "mujoco", mujoco_mod)

    adapter = GymnasiumMJXAdapter("gymnasium:HalfCheetah-v5", jax=fake_jax, jnp=np)
    obs, state = adapter.reset("reset-key")
    action = adapter.clip_action(np.asarray([4.0], dtype=np.float32))
    next_obs, next_state, reward, done, info = adapter.step("step-key", state, action)

    np.testing.assert_array_equal(obs, np.asarray([0.0, 0.0], dtype=np.float32))
    np.testing.assert_array_equal(action, np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(next_obs, np.asarray([0.0, 0.0], dtype=np.float32))
    assert next_state.steps == np.asarray(0, dtype=np.int32)
    assert done == pytest.approx(1.0)
    assert reward == pytest.approx(9.6)
    assert info["x_velocity"] == pytest.approx(10.0)


def test_mujoco_playground_impl_uses_valid_mjx_impl(monkeypatch) -> None:
    import problems.mujoco_playground_env as playground

    monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
    monkeypatch.delitem(sys.modules, "warp", raising=False)
    assert playground._mjx_impl() == "jax"

    monkeypatch.setitem(sys.modules, "mujoco_warp", ModuleType("mujoco_warp"))
    monkeypatch.setitem(sys.modules, "warp", SimpleNamespace(types=SimpleNamespace()))
    assert playground._mjx_impl() == "jax"

    monkeypatch.setitem(
        sys.modules,
        "warp",
        SimpleNamespace(types=SimpleNamespace(warp_type_to_np_dtype={})),
    )
    assert playground._mjx_impl() == "warp"


def test_mujoco_playground_adapter_passes_valid_impl(monkeypatch) -> None:
    import numpy as np

    import problems.mujoco_playground_env as playground

    calls = []

    def make_state(obs, *, reward=0.0, done=0.0):
        return SimpleNamespace(
            obs={
                "state": np.asarray(obs, dtype=np.float32),
                "privileged_state": np.asarray([9.0, 9.0], dtype=np.float32),
            },
            reward=np.asarray(reward, dtype=np.float32),
            done=np.asarray(done, dtype=np.float32),
            metrics={"metric": np.asarray(1.0, dtype=np.float32)},
        )

    fake_env = SimpleNamespace(
        _mj_model=SimpleNamespace(nu=2),
        reset=lambda _key: make_state([1.0, 2.0, 3.0]),
        step=lambda _state, _action: make_state([4.0, 5.0, 6.0], reward=7.0, done=1.0),
    )

    def fake_load(env_name, config_overrides):
        calls.append((env_name, config_overrides))
        return fake_env

    registry_mod = SimpleNamespace(load=fake_load)
    playground_mod = ModuleType("mujoco_playground")
    playground_mod.registry = registry_mod
    spaces_mod = ModuleType("gymnax.environments.spaces")
    spaces_mod.Box = lambda *, low, high, shape, dtype: SimpleNamespace(low=low, high=high, shape=shape, dtype=dtype)
    environments_mod = ModuleType("gymnax.environments")
    environments_mod.spaces = spaces_mod
    gymnax_mod = ModuleType("gymnax")
    gymnax_mod.environments = environments_mod
    fake_jax = SimpleNamespace(
        random=SimpleNamespace(
            key=lambda seed: ("key", seed),
            split=lambda key: ((key, "step"), (key, "reset")),
        ),
        lax=SimpleNamespace(
            cond=lambda pred, true_fn, false_fn, operand: (true_fn(operand) if pred else false_fn(operand)),
        ),
        tree_util=SimpleNamespace(tree_leaves=lambda value: [value]),
    )
    monkeypatch.setitem(sys.modules, "mujoco_playground", playground_mod)
    monkeypatch.setitem(sys.modules, "gymnax", gymnax_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments", environments_mod)
    monkeypatch.setitem(sys.modules, "gymnax.environments.spaces", spaces_mod)
    monkeypatch.delitem(sys.modules, "mujoco_warp", raising=False)
    monkeypatch.delitem(sys.modules, "warp", raising=False)

    adapter = playground.MujocoPlaygroundAdapter("mujoco_playground:G1JoystickRoughTerrain", jax=fake_jax, jnp=np)
    obs, state = adapter.reset("reset-key")
    action = adapter.clip_action(np.asarray([2.0, -3.0], dtype=np.float32))
    next_obs, next_state, reward, done, metrics = adapter.step("step-key", state, action)

    assert calls == [("G1JoystickRoughTerrain", {"impl": "jax"})]
    assert adapter.observation_space.shape == (3,)
    assert adapter.action_space.shape == (2,)
    np.testing.assert_array_equal(obs, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    np.testing.assert_array_equal(action, np.asarray([1.0, -1.0], dtype=np.float32))
    np.testing.assert_array_equal(next_obs, np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
    assert next_state is not state
    assert reward == pytest.approx(7.0)
    assert done == pytest.approx(1.0)
    assert metrics["metric"] == pytest.approx(1.0)
