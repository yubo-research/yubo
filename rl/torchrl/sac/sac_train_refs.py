from __future__ import annotations

import importlib
from types import SimpleNamespace


def _sac_training_refs() -> SimpleNamespace:
    m = importlib.import_module
    r = SimpleNamespace()
    r.episode_rollout = m("rl.core.episode_rollout")
    r.eval_noise = m("rl.eval_noise")
    rl_helpers = m("common.rl_helpers")
    r.seed_all = rl_helpers.seed_all
    r.video = rl_helpers.video
    r.get_env_conf = m("problems.env_conf").get_env_conf
    r.experiment_seeds = m("common.experiment_seeds")
    r.torchrl_common = m("rl.core.runtime")
    r.torchrl_sac_actor_eval = m("rl.torchrl.sac.actor_eval")
    r.torchrl_sac_loop = m("rl.torchrl.sac.loop")
    r.cfg_mod = m("rl.torchrl.sac.config")
    r.phase_a = m("rl.torchrl.sac.sac_trainer_phase_a")
    r.phase_b = m("rl.torchrl.sac.sac_trainer_phase_b")
    r.setup = m("rl.torchrl.sac.setup")
    r.rl_logger = m("rl.logger")
    return r
