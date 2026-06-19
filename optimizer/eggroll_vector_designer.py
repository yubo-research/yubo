from __future__ import annotations

import time

import numpy as np

from optimizer.datum import Datum
from optimizer.designer_asserts import assert_scalar_rreturn
from optimizer.designer_errors import NoSuchDesignerError
from optimizer.eggroll_runtime import EggRollJAXRuntime
from optimizer.optimizer_types import IterateResult
from optimizer.trajectory import Trajectory
from optimizer.turbo_enn_designer import TurboENNDesigner


class EggRollJAXVectorDesigner(TurboENNDesigner):
    """Use Turbo-ENN proposals while evaluating EggRoll policies in JAX."""

    def __init__(
        self,
        policy,
        env_conf,
        *,
        steps_per_episode: int = 200,
        num_envs: int = 1,
        deterministic_policy: bool = False,
        param_scale: float = 0.5,
        seed_offset: int = 0,
        **turbo_kwargs,
    ) -> None:
        steps_per_episode = int(steps_per_episode)
        num_envs = int(num_envs)
        param_scale = float(param_scale)
        if steps_per_episode < 1:
            raise NoSuchDesignerError("EggRoll Turbo-ENN option 'steps_per_episode' must be >= 1.")
        if num_envs < 1:
            raise NoSuchDesignerError("EggRoll Turbo-ENN option 'num_envs' must be >= 1.")
        if param_scale <= 0.0:
            raise NoSuchDesignerError("EggRoll Turbo-ENN option 'param_scale' must be > 0.")

        super().__init__(policy, **turbo_kwargs)
        if self._tr_type == "morbo":
            raise NoSuchDesignerError("MORBO Turbo-ENN variants require multi-objective returns; EggRoll RL/MARL returns are scalar.")

        self._runtime = EggRollJAXRuntime(
            policy,
            env_conf,
            steps_per_episode=steps_per_episode,
            num_envs=num_envs,
            deterministic_policy=deterministic_policy,
            seed_offset=seed_offset,
            vector_mode="offset",
            param_scale=param_scale,
            es_key_fold=11,
            eval_key_fold=12,
            error_cls=NoSuchDesignerError,
            option_label="EggRoll Turbo-ENN option",
            stack_error_message=(
                "EggRoll Turbo-ENN requires the separate HyperscaleES environment plus ENN. Run the Pixi setup task first, then use that Pixi environment."
            ),
        )

    def _tell_new_data(self, new_data):
        assert_scalar_rreturn(new_data)
        if len(new_data) == 0:
            return
        x_list = []
        for datum in new_data:
            x = getattr(datum.policy, "_eggroll_bo_x", None)
            if x is None:
                raise NoSuchDesignerError("EggRoll Turbo-ENN datum is missing its proposal vector.")
            x_list.append(np.asarray(x, dtype=np.float64))
        y_list = [d.trajectory.rreturn for d in new_data]
        y_se_list = [d.trajectory.rreturn_se for d in new_data] if self._use_y_var else []
        if self._use_y_var:
            assert all(se is not None for se in y_se_list)
        x = np.asarray(x_list, dtype=np.float64)
        y_obs = np.asarray(y_list, dtype=np.float64)
        y_obs = y_obs[:, None] if y_obs.ndim == 1 else y_obs
        y_est = self._turbo.tell(x, y_obs, y_var=np.asarray(y_se_list) ** 2) if y_se_list else self._turbo.tell(x, y_obs)
        assert y_obs.shape == y_est.shape and y_obs.shape[0] == len(new_data)
        if y_est.shape[1] == 1:
            self._update_best_estimate(new_data, y_est[:, 0])

    def iterate(self, data, num_arms: int, *, telemetry=None) -> IterateResult:
        if self._num_arms is None:
            self._num_arms = int(num_arms)
            self._init_optimizer(data, int(num_arms))

        t_prop = time.time()
        if len(data) > self._num_told:
            self._tell_new_data(data[self._num_told :])
            self._num_told = len(data)
        x_new = np.asarray(self._turbo.ask(int(num_arms)), dtype=np.float64)
        if telemetry is not None:
            t = self._turbo.telemetry()
            telemetry.set_dt_fit(t.dt_fit)
            telemetry.set_dt_select(t.dt_sel)
        dt_prop = time.time() - t_prop

        t_eval = time.time()
        scores = self._runtime.evaluate_values_with_keys(x_new, self._runtime.next_eval_keys(int(num_arms)))
        dt_eval = time.time() - t_eval

        out = []
        num_steps = int(self._runtime.steps_per_episode * self._runtime.num_envs)
        for x, score in zip(x_new, scores, strict=True):
            policy = self._runtime.make_policy(x, attr_name="_eggroll_bo_x")
            out.append(
                Datum(
                    self,
                    policy,
                    None,
                    Trajectory(
                        rreturn=float(score),
                        states=np.empty((0,)),
                        actions=np.empty((0,)),
                        num_steps=num_steps,
                    ),
                )
            )
        return IterateResult(data=out, dt_prop=float(dt_prop), dt_eval=float(dt_eval))
