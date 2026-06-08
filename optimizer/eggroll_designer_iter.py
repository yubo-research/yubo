from __future__ import annotations

import time
from contextlib import nullcontext, redirect_stdout
from io import StringIO

import numpy as np

from optimizer.datum import Datum
from optimizer.designer_errors import NoSuchDesignerError
from optimizer.eggroll_designer_nanoegg import update_best_and_telemetry
from optimizer.optimizer_types import IterateResult
from optimizer.trajectory import Trajectory


def iterate_jax(designer, state, _data, num_arms: int, *, telemetry=None) -> IterateResult:
    _ = _data
    _validate_population(int(num_arms))
    raw_scores, train_eval_dt = _evaluate_training_population(designer, state, int(num_arms))
    prop_dt = _update_jax_params(designer, state, raw_scores)
    datum, policy_eval_dt = _evaluate_current_policy(designer, state, int(num_arms))
    update_best_and_telemetry(designer, state, datum, prop_dt, telemetry)
    state.epoch += 1
    return IterateResult(
        data=[datum],
        dt_prop=float(prop_dt),
        dt_eval=float(train_eval_dt + policy_eval_dt),
    )


def _validate_population(num_arms: int) -> None:
    if num_arms < 2 or num_arms % 2 != 0:
        raise NoSuchDesignerError("EggRoll requires an even population >= 2 because HyperscaleES noisers use mirrored thread pairs.")


def _evaluate_training_population(designer, state, num_arms: int):
    t_eval = time.time()
    state.noiser_params = _scheduled_noiser_params(designer, state)
    state.train_key, batch_key = designer._jax.random.split(state.train_key)
    batch_keys = designer._jax.random.split(batch_key, num_arms)
    raw_scores = designer._evaluate_population(
        state.params,
        state.noiser_params,
        designer._jnp.asarray(state.epoch, dtype=designer._jnp.int32),
        batch_keys,
    )
    return designer._jax.block_until_ready(raw_scores), time.time() - t_eval


def _update_jax_params(designer, state, raw_scores) -> float:
    t_prop = time.time()
    stdout_ctx = redirect_stdout(StringIO()) if designer._suppress_noiser_stdout else nullcontext()
    with stdout_ctx:
        state.noiser_params, state.params = designer._update_params(
            state.noiser_params,
            state.params,
            raw_scores,
            designer._jnp.asarray(state.epoch, dtype=designer._jnp.int32),
        )
    _block_tree(designer, state.params)
    return time.time() - t_prop


def _evaluate_current_policy(designer, state, num_arms: int):
    t_eval = time.time()
    state.eval_key, eval_batch_key = designer._jax.random.split(state.eval_key)
    eval_scores = designer._evaluate_policy(
        state.params,
        state.noiser_params,
        designer._jax.random.split(eval_batch_key, designer._num_envs),
    )
    eval_scores = designer._jax.block_until_ready(eval_scores)
    datum = _make_datum(designer, state, eval_scores, num_arms)
    return datum, time.time() - t_eval


def _make_datum(designer, state, eval_scores, num_arms: int) -> Datum:
    return Datum(
        designer,
        designer._policy.with_params(state.params),
        None,
        Trajectory(
            rreturn=float(designer._jnp.mean(eval_scores)),
            states=np.empty((0,)),
            actions=np.empty((0,)),
            num_steps=int((num_arms + designer._num_envs) * designer._steps_per_episode),
        ),
    )


def _block_tree(designer, tree):
    for leaf in designer._jax.tree_util.tree_leaves(tree):
        leaf.block_until_ready()
    return tree


def _scheduled_noiser_params(designer, state):
    if not isinstance(state.noiser_params, dict) or "sigma" not in state.noiser_params:
        if designer._sigma_decay != 1.0:
            raise NoSuchDesignerError("EggRoll sigma_decay requires noiser_params to expose a 'sigma' field.")
        return state.noiser_params
    return state.noiser_params | {"sigma": designer._current_sigma()}
