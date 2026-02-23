"""TorchRL TD-MPC2 trainer.

TODO(tdmpc2): align this implementation with full TD-MPC2 training targets.
"""

from __future__ import annotations

import dataclasses
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis.data_io import write_config
from problems.env_conf import get_env_conf
from rl import logger as rl_logger
from rl.backends.torchrl.common.common import select_device
from rl.checkpointing import CheckpointManager
from rl.seed_util import global_seed_for_run, resolve_problem_seed

from .config import TDMPC2Config


class _ReplayBuffer:
    def __init__(self, obs_dim: int, act_dim: int, capacity: int):
        self.obs = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.act = np.zeros((capacity, act_dim), dtype=np.float32)
        self.rew = np.zeros((capacity,), dtype=np.float32)
        self.nxt = np.zeros((capacity, obs_dim), dtype=np.float32)
        self.done = np.zeros((capacity,), dtype=np.float32)
        self.capacity = int(capacity)
        self.size = 0
        self.ptr = 0

    def add(self, obs: np.ndarray, act: np.ndarray, rew: float, nxt: np.ndarray, done: bool) -> None:
        i = int(self.ptr)
        self.obs[i] = np.asarray(obs, dtype=np.float32).reshape(-1)
        self.act[i] = np.asarray(act, dtype=np.float32).reshape(-1)
        self.rew[i] = float(rew)
        self.nxt[i] = np.asarray(nxt, dtype=np.float32).reshape(-1)
        self.done[i] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, self.size, size=int(batch_size))
        obs = torch.as_tensor(self.obs[idx], device=device)
        act = torch.as_tensor(self.act[idx], device=device)
        rew = torch.as_tensor(self.rew[idx], device=device)
        nxt = torch.as_tensor(self.nxt[idx], device=device)
        done = torch.as_tensor(self.done[idx], device=device)
        return obs, act, rew, nxt, done


class _LatentModel(nn.Module):
    def __init__(self, obs_dim: int, act_dim: int, hidden_dim: int, latent_dim: int):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
            nn.LayerNorm(latent_dim),
            nn.Tanh(),
        )
        self.dynamics = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.reward = nn.Sequential(
            nn.Linear(latent_dim + act_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.value = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.actor = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, act_dim),
            nn.Tanh(),
        )

    def encode(self, obs: torch.Tensor) -> torch.Tensor:
        return self.encoder(obs)

    def next_latent(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.dynamics(torch.cat([z, a], dim=-1))

    def reward_pred(self, z: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        return self.reward(torch.cat([z, a], dim=-1)).squeeze(-1)

    def value_pred(self, z: torch.Tensor) -> torch.Tensor:
        return self.value(z).squeeze(-1)

    def actor_action(self, z: torch.Tensor) -> torch.Tensor:
        return self.actor(z)


@dataclasses.dataclass
class TrainResult:
    best_return: float
    last_eval_return: float
    num_updates: int
    total_steps: int


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _to_env_action(a_norm: np.ndarray, *, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.clip(low + 0.5 * (a_norm + 1.0) * (high - low), low, high)


def _to_norm_action(a_env: np.ndarray, *, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    width = np.maximum(high - low, 1e-6)
    return np.clip(2.0 * (np.asarray(a_env, dtype=np.float32) - low) / width - 1.0, -1.0, 1.0)


def _plan_action(
    model: _LatentModel,
    obs: np.ndarray,
    *,
    device: torch.device,
    horizon: int,
    gamma: float,
    act_dim: int,
    n_samples: int,
    n_elites: int,
    n_iters: int,
    explore_std: float,
    add_exploration: bool,
) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        obs_t = torch.as_tensor(np.asarray(obs, dtype=np.float32).reshape(1, -1), device=device)
        z0 = model.encode(obs_t).repeat(int(n_samples), 1)
        mean = torch.zeros((int(horizon), int(act_dim)), device=device)
        std = torch.ones((int(horizon), int(act_dim)), device=device)
        for _ in range(int(n_iters)):
            seq = torch.clamp(mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn((int(n_samples), int(horizon), int(act_dim)), device=device), -1.0, 1.0)
            z = z0
            ret = torch.zeros((int(n_samples),), device=device)
            discount = 1.0
            for t in range(int(horizon)):
                a_t = seq[:, t, :]
                r_t = model.reward_pred(z, a_t)
                z = model.next_latent(z, a_t)
                ret = ret + discount * r_t
                discount *= float(gamma)
            ret = ret + discount * model.value_pred(z)
            elite_idx = torch.topk(ret, k=min(int(n_elites), int(n_samples))).indices
            elite = seq[elite_idx]
            mean = elite.mean(dim=0)
            std = elite.std(dim=0).clamp_min(1e-3)
        a = mean[0]
        if add_exploration:
            a = torch.clamp(a + float(explore_std) * torch.randn_like(a), -1.0, 1.0)
    return np.asarray(a.detach().cpu().numpy(), dtype=np.float32)


def _evaluate(
    model: _LatentModel,
    env_conf,
    *,
    device: torch.device,
    episodes: int,
    seed_base: int,
    low: np.ndarray,
    high: np.ndarray,
    cfg: TDMPC2Config,
) -> float:
    returns = []
    for ep in range(int(max(1, episodes))):
        env = env_conf.make()
        obs, _ = env.reset(seed=int(seed_base + ep))
        done = False
        total = 0.0
        while not done:
            a_norm = _plan_action(
                model,
                np.asarray(obs, dtype=np.float32).reshape(-1),
                device=device,
                horizon=int(cfg.horizon),
                gamma=float(cfg.gamma),
                act_dim=int(low.shape[0]),
                n_samples=int(cfg.plan_samples),
                n_elites=int(cfg.plan_elites),
                n_iters=int(cfg.plan_iters),
                explore_std=0.0,
                add_exploration=False,
            )
            action = _to_env_action(a_norm, low=low, high=high)
            obs, reward, terminated, truncated, _ = env.step(action)
            total += float(reward)
            done = bool(terminated or truncated)
        env.close()
        returns.append(float(total))
    return float(np.mean(returns)) if returns else float("nan")


def _train_step(
    model: _LatentModel,
    target_value: nn.Module,
    replay: _ReplayBuffer,
    *,
    cfg: TDMPC2Config,
    device: torch.device,
    model_opt: optim.Optimizer,
    actor_opt: optim.Optimizer,
    value_opt: optim.Optimizer,
) -> dict[str, float]:
    obs, act, rew, nxt, done = replay.sample(int(cfg.rollout_batch_size), device=device)

    z = model.encode(obs)
    with torch.no_grad():
        z_next = model.encode(nxt)

    # TODO(tdmpc2): switch to multi-step latent/reward consistency with sequence replay.
    pred_z_next = model.next_latent(z, act)
    pred_rew = model.reward_pred(z, act)
    model_loss = torch.mean((pred_z_next - z_next) ** 2) + torch.mean((pred_rew - rew) ** 2)
    model_opt.zero_grad(set_to_none=True)
    model_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
    model_opt.step()

    v = model.value_pred(z.detach())
    with torch.no_grad():
        v_tgt = rew + float(cfg.gamma) * (1.0 - done) * target_value(z_next).squeeze(-1)
    value_loss = torch.mean((v - v_tgt) ** 2)
    value_opt.zero_grad(set_to_none=True)
    value_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.value.parameters(), 10.0)
    value_opt.step()

    # TODO(tdmpc2): replace behavior-cloning actor loss with value-max objective from imagined rollouts.
    z_actor = model.encode(obs).detach()
    a_actor = model.actor_action(z_actor)
    actor_loss = torch.mean((a_actor - act) ** 2)
    actor_opt.zero_grad(set_to_none=True)
    actor_loss.backward()
    torch.nn.utils.clip_grad_norm_(model.actor.parameters(), 10.0)
    actor_opt.step()

    tau = float(cfg.tau)
    with torch.no_grad():
        for p_tgt, p in zip(target_value.parameters(), model.value.parameters(), strict=True):
            p_tgt.data.mul_(1.0 - tau).add_(tau * p.data)

    return {
        "model": float(model_loss.item()),
        "actor": float(actor_loss.item()),
        "value": float(value_loss.item()),
    }


@dataclasses.dataclass
class _TDMPC2Runtime:
    env_conf: Any
    env: Any
    model: _LatentModel
    target_value: nn.Module
    replay: _ReplayBuffer
    model_opt: optim.Optimizer
    actor_opt: optim.Optimizer
    value_opt: optim.Optimizer
    ckpt: CheckpointManager
    metrics_path: Path
    device: torch.device
    act_low: np.ndarray
    act_high: np.ndarray
    act_dim: int
    run_seed: int
    state: np.ndarray
    best_return: float = -float("inf")
    last_eval_return: float = float("nan")
    num_updates: int = 0
    start_time: float = 0.0


@dataclasses.dataclass(frozen=True)
class _TDMPC2EnvInit:
    env_conf: Any
    env: Any
    state: np.ndarray
    obs_dim: int
    act_low: np.ndarray
    act_high: np.ndarray
    act_dim: int


@dataclasses.dataclass(frozen=True)
class _TDMPC2CoreInit:
    device: torch.device
    model: _LatentModel
    target_value: nn.Module
    model_opt: optim.Optimizer
    actor_opt: optim.Optimizer
    value_opt: optim.Optimizer
    replay: _ReplayBuffer


def _init_tdmpc2_env(config: TDMPC2Config, *, problem_seed: int, run_seed: int) -> _TDMPC2EnvInit:
    env_conf = get_env_conf(str(config.env_tag), problem_seed=int(problem_seed), from_pixels=False)
    env = env_conf.make()
    obs, _ = env.reset(seed=int(run_seed))
    if not hasattr(env.action_space, "low") or not hasattr(env.action_space, "high"):
        env.close()
        raise ValueError("tdmpc2 requires continuous (Box) action space.")
    obs_dim = int(np.asarray(obs, dtype=np.float32).reshape(-1).shape[0])
    act_low = np.asarray(env.action_space.low, dtype=np.float32).reshape(-1)
    act_high = np.asarray(env.action_space.high, dtype=np.float32).reshape(-1)
    act_dim = int(act_low.shape[0])
    return _TDMPC2EnvInit(
        env_conf=env_conf,
        env=env,
        state=np.asarray(obs, dtype=np.float32).reshape(-1),
        obs_dim=obs_dim,
        act_low=act_low,
        act_high=act_high,
        act_dim=act_dim,
    )


def _init_tdmpc2_components(
    config: TDMPC2Config,
    *,
    obs_dim: int,
    act_dim: int,
) -> _TDMPC2CoreInit:
    device = select_device(str(config.device))
    model = _LatentModel(obs_dim, act_dim, int(config.hidden_dim), int(config.latent_dim)).to(device)
    target_value = nn.Sequential(
        nn.Linear(int(config.latent_dim), int(config.hidden_dim)),
        nn.SiLU(),
        nn.Linear(int(config.hidden_dim), 1),
    ).to(device)
    target_value.load_state_dict(model.value.state_dict())
    for p in target_value.parameters():
        p.requires_grad_(False)
    model_params = list(model.encoder.parameters()) + list(model.dynamics.parameters()) + list(model.reward.parameters())
    model_opt = optim.Adam(model_params, lr=float(config.model_lr))
    actor_opt = optim.Adam(model.actor.parameters(), lr=float(config.actor_lr))
    value_opt = optim.Adam(model.value.parameters(), lr=float(config.value_lr))
    replay = _ReplayBuffer(obs_dim, act_dim, int(config.replay_capacity))
    return _TDMPC2CoreInit(
        device=device,
        model=model,
        target_value=target_value,
        model_opt=model_opt,
        actor_opt=actor_opt,
        value_opt=value_opt,
        replay=replay,
    )


def _log_tdmpc2_header(config: TDMPC2Config, *, obs_dim: int, act_dim: int, device: torch.device) -> None:
    rl_logger.register_algo_metrics("tdmpc2", [("model", 7, ".4f"), ("actor", 7, ".4f"), ("value", 7, ".4f")])
    rl_logger.log_run_header_basic(
        algo_name="tdmpc2",
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        backbone_name="latent_mlp",
        from_pixels=False,
        obs_dim=obs_dim,
        act_dim=act_dim,
        frames_per_batch=1,
        num_iterations=int(config.total_timesteps),
        device_type=device.type,
    )


def _make_tdmpc2_runtime(
    *,
    exp_dir: Path,
    run_seed: int,
    env_init: _TDMPC2EnvInit,
    core_init: _TDMPC2CoreInit,
    ckpt: CheckpointManager,
) -> _TDMPC2Runtime:
    return _TDMPC2Runtime(
        env_conf=env_init.env_conf,
        env=env_init.env,
        model=core_init.model,
        target_value=core_init.target_value,
        replay=core_init.replay,
        model_opt=core_init.model_opt,
        actor_opt=core_init.actor_opt,
        value_opt=core_init.value_opt,
        ckpt=ckpt,
        metrics_path=exp_dir / "metrics.jsonl",
        device=core_init.device,
        act_low=env_init.act_low,
        act_high=env_init.act_high,
        act_dim=env_init.act_dim,
        run_seed=int(run_seed),
        state=env_init.state,
        start_time=float(time.time()),
    )


def _build_tdmpc2_runtime(config: TDMPC2Config, *, exp_dir: Path, run_seed: int, problem_seed: int) -> _TDMPC2Runtime:
    env_init = _init_tdmpc2_env(
        config,
        problem_seed=int(problem_seed),
        run_seed=int(run_seed),
    )
    core_init = _init_tdmpc2_components(
        config,
        obs_dim=env_init.obs_dim,
        act_dim=env_init.act_dim,
    )
    ckpt = CheckpointManager(exp_dir=exp_dir)
    _log_tdmpc2_header(config, obs_dim=env_init.obs_dim, act_dim=env_init.act_dim, device=core_init.device)
    return _make_tdmpc2_runtime(
        exp_dir=exp_dir,
        run_seed=run_seed,
        env_init=env_init,
        core_init=core_init,
        ckpt=ckpt,
    )


def _tdmpc2_action(config: TDMPC2Config, runtime: _TDMPC2Runtime, *, step: int) -> np.ndarray:
    if int(step) <= int(config.warmup_steps):
        return np.random.uniform(-1.0, 1.0, size=(runtime.act_dim,)).astype(np.float32)
    return _plan_action(
        runtime.model,
        runtime.state,
        device=runtime.device,
        horizon=int(config.horizon),
        gamma=float(config.gamma),
        act_dim=runtime.act_dim,
        n_samples=int(config.plan_samples),
        n_elites=int(config.plan_elites),
        n_iters=int(config.plan_iters),
        explore_std=float(config.exploration_std),
        add_exploration=True,
    )


def _tdmpc2_rollout_step(config: TDMPC2Config, runtime: _TDMPC2Runtime, *, action_norm: np.ndarray) -> None:
    env_action = _to_env_action(action_norm, low=runtime.act_low, high=runtime.act_high)
    next_obs, reward, terminated, truncated, _ = runtime.env.step(env_action)
    done = bool(terminated or truncated)
    next_state = np.asarray(next_obs, dtype=np.float32).reshape(-1)
    runtime.replay.add(
        runtime.state,
        _to_norm_action(env_action, low=runtime.act_low, high=runtime.act_high),
        float(reward),
        next_state,
        done,
    )
    runtime.state = next_state
    if done:
        reset_obs, _ = runtime.env.reset()
        runtime.state = np.asarray(reset_obs, dtype=np.float32).reshape(-1)


def _tdmpc2_optimizer_updates(config: TDMPC2Config, runtime: _TDMPC2Runtime) -> dict[str, float]:
    losses = {"model": float("nan"), "actor": float("nan"), "value": float("nan")}
    if runtime.replay.size < int(config.rollout_batch_size):
        return losses
    for _ in range(int(config.updates_per_step)):
        losses = _train_step(
            runtime.model,
            runtime.target_value,
            runtime.replay,
            cfg=config,
            device=runtime.device,
            model_opt=runtime.model_opt,
            actor_opt=runtime.actor_opt,
            value_opt=runtime.value_opt,
        )
        runtime.num_updates += 1
    return losses


def _maybe_log_tdmpc2(config: TDMPC2Config, runtime: _TDMPC2Runtime, *, step: int, losses: dict[str, float], elapsed: float) -> None:
    should_eval = int(config.eval_interval) > 0 and (step % int(config.eval_interval) == 0 or step == int(config.total_timesteps))
    if should_eval:
        runtime.last_eval_return = _evaluate(
            runtime.model,
            runtime.env_conf,
            device=runtime.device,
            episodes=int(config.eval_episodes),
            seed_base=int(runtime.run_seed + 1000),
            low=runtime.act_low,
            high=runtime.act_high,
            cfg=config,
        )
        runtime.best_return = max(runtime.best_return, float(runtime.last_eval_return))
        rl_logger.append_metrics(
            runtime.metrics_path,
            {
                "step": int(step),
                "eval_return": float(runtime.last_eval_return),
                "best_return": float(runtime.best_return),
                "model_loss": float(losses["model"]),
                "actor_loss": float(losses["actor"]),
                "value_loss": float(losses["value"]),
                "elapsed": float(elapsed),
                "num_updates": int(runtime.num_updates),
            },
        )
        rl_logger.log_eval_iteration(
            iteration=int(step),
            num_iterations=int(config.total_timesteps),
            frames_per_batch=1,
            eval_return=float(runtime.last_eval_return),
            heldout_return=None,
            best_return=float(runtime.best_return),
            algo_metrics={"model": losses["model"], "actor": losses["actor"], "value": losses["value"]},
            algo_name="tdmpc2",
            elapsed=float(elapsed),
            step_override=int(step),
        )
        return
    if int(config.log_interval) > 0 and step % int(config.log_interval) == 0:
        rl_logger.log_progress_iteration(
            iteration=int(step),
            num_iterations=int(config.total_timesteps),
            frames_per_batch=1,
            elapsed=float(elapsed),
            algo_name="tdmpc2",
            step_override=int(step),
        )


def _maybe_save_tdmpc2_checkpoint(config: TDMPC2Config, runtime: _TDMPC2Runtime, *, step: int) -> None:
    if config.checkpoint_interval is None or int(config.checkpoint_interval) <= 0:
        return
    if step % int(config.checkpoint_interval) != 0:
        return
    runtime.ckpt.save_both(
        {
            "step": int(step),
            "model": runtime.model.state_dict(),
            "target_value": runtime.target_value.state_dict(),
            "model_opt": runtime.model_opt.state_dict(),
            "actor_opt": runtime.actor_opt.state_dict(),
            "value_opt": runtime.value_opt.state_dict(),
            "best_return": float(runtime.best_return),
            "last_eval_return": float(runtime.last_eval_return),
            "num_updates": int(runtime.num_updates),
        },
        iteration=int(step),
    )


def _finalize_tdmpc2(config: TDMPC2Config, runtime: _TDMPC2Runtime) -> TrainResult:
    runtime.env.close()
    total_time = float(time.time() - runtime.start_time)
    best = runtime.best_return if runtime.best_return == runtime.best_return else runtime.last_eval_return
    rl_logger.log_run_footer(
        best_return=float(best),
        total_iters_or_steps=int(config.total_timesteps),
        total_time=total_time,
        algo_name="tdmpc2",
        step_label="steps",
    )
    return TrainResult(
        best_return=float(runtime.best_return),
        last_eval_return=float(runtime.last_eval_return),
        num_updates=int(runtime.num_updates),
        total_steps=int(config.total_timesteps),
    )


def train_tdmpc2(config: TDMPC2Config) -> TrainResult:
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())
    problem_seed = resolve_problem_seed(seed=int(config.seed), problem_seed=config.problem_seed)
    run_seed = global_seed_for_run(int(problem_seed))
    _seed_everything(run_seed)
    runtime = _build_tdmpc2_runtime(config, exp_dir=exp_dir, run_seed=int(run_seed), problem_seed=int(problem_seed))
    for step in range(1, int(config.total_timesteps) + 1):
        action_norm = _tdmpc2_action(config, runtime, step=step)
        _tdmpc2_rollout_step(config, runtime, action_norm=action_norm)
        losses = _tdmpc2_optimizer_updates(config, runtime)
        elapsed = float(time.time() - runtime.start_time)
        _maybe_log_tdmpc2(config, runtime, step=step, losses=losses, elapsed=elapsed)
        _maybe_save_tdmpc2_checkpoint(config, runtime, step=step)
    return _finalize_tdmpc2(config, runtime)
