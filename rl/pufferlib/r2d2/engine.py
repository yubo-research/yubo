"""PufferLib R2D2 trainer."""

from __future__ import annotations

import dataclasses
import random
import time
from collections import deque
from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from analysis.data_io import write_config
from problems.env_conf import get_env_conf
from rl import logger as rl_logger
from rl.checkpointing import CheckpointManager
from rl.pufferlib.r2d2.config import R2D2Config
from rl.seed_util import global_seed_for_run, resolve_problem_seed
from rl.torchrl.common.common import select_device


class _SequenceReplay:
    def __init__(self, capacity: int):
        self.obs = deque(maxlen=int(capacity))
        self.actions = deque(maxlen=int(capacity))
        self.rewards = deque(maxlen=int(capacity))
        self.dones = deque(maxlen=int(capacity))

    def __len__(self) -> int:
        return len(self.obs)

    def add(self, obs_seq: np.ndarray, actions: np.ndarray, rewards: np.ndarray, dones: np.ndarray) -> None:
        self.obs.append(np.asarray(obs_seq, dtype=np.uint8 if obs_seq.dtype == np.uint8 else np.float32))
        self.actions.append(np.asarray(actions, dtype=np.int64))
        self.rewards.append(np.asarray(rewards, dtype=np.float32))
        self.dones.append(np.asarray(dones, dtype=np.float32))

    def sample(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, ...]:
        idx = np.random.randint(0, len(self.obs), size=int(batch_size))
        obs = torch.as_tensor(np.stack([self.obs[i] for i in idx], axis=0), device=device)
        actions = torch.as_tensor(np.stack([self.actions[i] for i in idx], axis=0), device=device)
        rewards = torch.as_tensor(np.stack([self.rewards[i] for i in idx], axis=0), device=device)
        dones = torch.as_tensor(np.stack([self.dones[i] for i in idx], axis=0), device=device)
        return obs, actions, rewards, dones


class _R2D2QNet(nn.Module):
    def __init__(self, obs_shape: tuple[int, ...], action_n: int, hidden_dim: int):
        super().__init__()
        self.obs_shape = tuple(int(x) for x in obs_shape)
        self.action_n = int(action_n)
        self.hidden_dim = int(hidden_dim)
        self.is_image = len(self.obs_shape) >= 3
        if self.is_image:
            with torch.no_grad():
                dummy = torch.zeros((1, *self.obs_shape))
                dummy_chw = self._to_chw(dummy)
                in_ch = int(dummy_chw.shape[1])
            self.encoder = nn.Sequential(
                nn.Conv2d(in_ch, 32, kernel_size=8, stride=4),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1),
                nn.ReLU(),
            )
            with torch.no_grad():
                feat_dim = int(np.prod(self.encoder(dummy_chw).shape[1:]))
            self.fc = nn.Sequential(nn.Linear(feat_dim, 512), nn.ReLU())
            lstm_in = 512
        else:
            in_dim = int(np.prod(self.obs_shape))
            self.fc = nn.Sequential(
                nn.Linear(in_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
            )
            self.encoder = None
            lstm_in = 256
        self.lstm = nn.LSTM(input_size=lstm_in, hidden_size=int(hidden_dim), num_layers=1, batch_first=True)
        self.q_head = nn.Linear(int(hidden_dim), int(action_n))

    def _to_chw(self, x: torch.Tensor) -> torch.Tensor:
        # Accept:
        # - [N,H,W,C]
        # - [N,C,H,W]
        # - [N,stack,H,W,1] (Atari FrameStack + grayscale_newaxis)
        if x.ndim == 5 and int(x.shape[-1]) == 1:
            x = x.squeeze(-1)
        if x.ndim != 4:
            raise ValueError(f"Expected image tensor with 4 dims after normalization, got shape={tuple(x.shape)}")
        if int(x.shape[1]) in (1, 3, 4):
            return x
        if int(x.shape[-1]) in (1, 3, 4):
            return x.permute(0, 3, 1, 2)
        raise ValueError(f"Unable to infer channel axis for image shape={tuple(x.shape)}")

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h = torch.zeros((1, int(batch_size), int(self.hidden_dim)), device=device)
        c = torch.zeros((1, int(batch_size), int(self.hidden_dim)), device=device)
        return h, c

    def _encode(self, obs_bt: torch.Tensor) -> torch.Tensor:
        if self.is_image:
            x = self._to_chw(obs_bt).float() / 255.0
            x = self.encoder(x)
            x = x.reshape(x.shape[0], -1)
            return self.fc(x)
        x = obs_bt.float().reshape(obs_bt.shape[0], -1)
        return self.fc(x)

    def forward(
        self,
        obs: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        # obs: [B, T, ...]
        b, t = int(obs.shape[0]), int(obs.shape[1])
        feats = self._encode(obs.reshape(b * t, *obs.shape[2:])).reshape(b, t, -1)
        if hidden is None:
            hidden = self.init_hidden(batch_size=b, device=obs.device)
        out, hidden_next = self.lstm(feats, hidden)
        q = self.q_head(out)
        return q, hidden_next

    def act(
        self,
        obs_batch: np.ndarray,
        hidden: tuple[torch.Tensor, torch.Tensor],
        *,
        eps: float,
        device: torch.device,
    ) -> tuple[np.ndarray, tuple[torch.Tensor, torch.Tensor], float]:
        obs_t = torch.as_tensor(obs_batch, device=device).unsqueeze(1)
        q, hidden_next = self.forward(obs_t, hidden)
        q_step = q[:, 0, :]
        greedy = torch.argmax(q_step, dim=-1)
        if float(eps) > 0.0:
            rand_mask = torch.rand_like(greedy.float()) < float(eps)
            random_actions = torch.randint(0, int(self.action_n), greedy.shape, device=device)
            actions = torch.where(rand_mask, random_actions, greedy)
        else:
            actions = greedy
        return (
            np.asarray(actions.detach().cpu().numpy(), dtype=np.int64),
            (hidden_next[0].detach(), hidden_next[1].detach()),
            float(q_step.mean().item()),
        )


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


def _epsilon(step: int, cfg: R2D2Config) -> float:
    frac = min(1.0, float(step) / float(max(1, int(cfg.eps_decay_steps))))
    return float(cfg.eps_start + frac * (cfg.eps_end - cfg.eps_start))


def _make_vector_env(env_conf, *, num_envs: int):
    env_fns = [lambda: env_conf.make() for _ in range(int(num_envs))]
    return gym.vector.SyncVectorEnv(env_fns)


def _append_sequence_from_history(history: deque, replay: _SequenceReplay, seq_len: int) -> None:
    if len(history) < int(seq_len):
        return
    obs = [history[i][0] for i in range(int(seq_len))]
    obs.append(history[-1][4])
    actions = [history[i][1] for i in range(int(seq_len))]
    rewards = [history[i][2] for i in range(int(seq_len))]
    dones = [history[i][3] for i in range(int(seq_len))]
    replay.add(np.stack(obs, axis=0), np.asarray(actions), np.asarray(rewards), np.asarray(dones))


def _compute_loss(
    online: _R2D2QNet,
    target: _R2D2QNet,
    obs: torch.Tensor,
    actions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    *,
    cfg: R2D2Config,
) -> torch.Tensor:
    q_curr, _ = online(obs[:, :-1, ...])
    q_next_online, _ = online(obs[:, 1:, ...])
    with torch.no_grad():
        q_next_target, _ = target(obs[:, 1:, ...])
    start = int(cfg.burn_in)
    act = actions[:, start:].unsqueeze(-1)
    q_sa = torch.gather(q_curr[:, start:, :], dim=-1, index=act).squeeze(-1)
    next_act = torch.argmax(q_next_online[:, start:, :], dim=-1, keepdim=True)
    next_q = torch.gather(q_next_target[:, start:, :], dim=-1, index=next_act).squeeze(-1)
    td_target = rewards[:, start:] + float(cfg.gamma) * (1.0 - dones[:, start:]) * next_q
    return nn.functional.smooth_l1_loss(q_sa, td_target)


def _evaluate(online: _R2D2QNet, env_conf, *, cfg: R2D2Config, device: torch.device, seed_base: int) -> float:
    returns = []
    for ep in range(int(max(1, cfg.eval_episodes))):
        env = env_conf.make()
        obs, _ = env.reset(seed=int(seed_base + ep))
        hidden = online.init_hidden(batch_size=1, device=device)
        done = False
        total = 0.0
        while not done:
            action, hidden, _ = online.act(np.expand_dims(obs, axis=0), hidden, eps=0.0, device=device)
            obs, reward, terminated, truncated, _ = env.step(int(action[0]))
            total += float(reward)
            done = bool(terminated or truncated)
        env.close()
        returns.append(float(total))
    return float(np.mean(returns))


@dataclasses.dataclass
class _R2D2Runtime:
    env_conf: Any
    vec_env: Any
    obs: np.ndarray
    online: _R2D2QNet
    target: _R2D2QNet
    optimizer: optim.Optimizer
    replay: _SequenceReplay
    seq_len: int
    histories: list[deque]
    hidden: tuple[torch.Tensor, torch.Tensor]
    device: torch.device
    action_n: int
    obs_shape: tuple[int, ...]
    ckpt: CheckpointManager
    metrics_path: Path
    run_seed: int
    best_return: float = -float("inf")
    last_eval_return: float = float("nan")
    num_updates: int = 0
    start_time: float = 0.0
    global_step: int = 0
    last_loss: float = float("nan")
    last_q: float = float("nan")


def _build_r2d2_runtime(config: R2D2Config, *, exp_dir: Path, run_seed: int, problem_seed: int) -> _R2D2Runtime:
    env_conf = get_env_conf(str(config.env_tag), problem_seed=int(problem_seed))
    vec_env = _make_vector_env(env_conf, num_envs=int(config.num_envs))
    obs, _ = vec_env.reset(seed=[int(run_seed + i) for i in range(int(config.num_envs))])
    obs_shape = tuple(int(x) for x in obs.shape[1:])
    action_n = int(vec_env.single_action_space.n)
    device = select_device(str(config.device))
    online = _R2D2QNet(obs_shape, action_n, int(config.recurrent_hidden_dim)).to(device)
    target = _R2D2QNet(obs_shape, action_n, int(config.recurrent_hidden_dim)).to(device)
    target.load_state_dict(online.state_dict())
    optimizer = optim.Adam(online.parameters(), lr=float(config.learning_rate))
    replay = _SequenceReplay(capacity=int(config.replay_capacity))
    seq_len = int(config.burn_in + config.unroll_length)
    histories = [deque(maxlen=seq_len) for _ in range(int(config.num_envs))]
    hidden = online.init_hidden(batch_size=int(config.num_envs), device=device)
    ckpt = CheckpointManager(exp_dir=exp_dir)
    rl_logger.register_algo_metrics("r2d2", [("loss", 7, ".4f"), ("eps", 7, ".4f"), ("q", 7, ".3f")])
    rl_logger.log_run_header_basic(
        algo_name="r2d2",
        env_tag=str(config.env_tag),
        seed=int(config.seed),
        backbone_name="recurrent_q",
        from_pixels=len(obs_shape) >= 3,
        obs_dim=int(np.prod(obs_shape)),
        act_dim=action_n,
        frames_per_batch=1,
        num_iterations=int(config.total_timesteps),
        device_type=device.type,
    )
    return _R2D2Runtime(
        env_conf=env_conf,
        vec_env=vec_env,
        obs=obs,
        online=online,
        target=target,
        optimizer=optimizer,
        replay=replay,
        seq_len=seq_len,
        histories=histories,
        hidden=hidden,
        device=device,
        action_n=action_n,
        obs_shape=obs_shape,
        ckpt=ckpt,
        metrics_path=exp_dir / "metrics.jsonl",
        run_seed=int(run_seed),
        start_time=float(time.time()),
    )


def _run_r2d2_env_step(runtime: _R2D2Runtime, config: R2D2Config) -> float:
    eps = _epsilon(runtime.global_step, config)
    actions, hidden_next, q_mean = runtime.online.act(runtime.obs, runtime.hidden, eps=eps, device=runtime.device)
    nxt_obs, rewards, terms, truncs, _ = runtime.vec_env.step(actions)
    done = np.logical_or(terms, truncs)
    for i in range(int(config.num_envs)):
        runtime.histories[i].append((runtime.obs[i], int(actions[i]), float(rewards[i]), bool(done[i]), nxt_obs[i]))
        _append_sequence_from_history(runtime.histories[i], runtime.replay, seq_len=runtime.seq_len)
        if bool(done[i]):
            runtime.histories[i].clear()
            hidden_next[0][:, i, :] = 0.0
            hidden_next[1][:, i, :] = 0.0
    runtime.obs = nxt_obs
    runtime.hidden = hidden_next
    runtime.global_step += int(config.num_envs)
    runtime.last_q = float(q_mean)
    return float(eps)


def _run_r2d2_optimizer_updates(runtime: _R2D2Runtime, config: R2D2Config) -> None:
    ready = len(runtime.replay) >= int(config.batch_size) and runtime.global_step >= int(config.learning_starts)
    if not ready:
        return
    for _ in range(int(config.updates_per_step)):
        batch = runtime.replay.sample(int(config.batch_size), device=runtime.device)
        loss = _compute_loss(runtime.online, runtime.target, *batch, cfg=config)
        runtime.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(runtime.online.parameters(), 10.0)
        runtime.optimizer.step()
        runtime.num_updates += 1
        runtime.last_loss = float(loss.item())
        if runtime.num_updates % int(config.target_update_interval) == 0:
            runtime.target.load_state_dict(runtime.online.state_dict())


def _maybe_log_r2d2_progress(runtime: _R2D2Runtime, config: R2D2Config, *, eps: float, elapsed: float) -> None:
    do_eval = int(config.eval_interval) > 0 and (
        runtime.global_step % int(config.eval_interval) < int(config.num_envs) or runtime.global_step >= int(config.total_timesteps)
    )
    if do_eval:
        runtime.last_eval_return = _evaluate(
            runtime.online,
            runtime.env_conf,
            cfg=config,
            device=runtime.device,
            seed_base=int(runtime.run_seed + 1000),
        )
        runtime.best_return = max(runtime.best_return, float(runtime.last_eval_return))
        rl_logger.append_metrics(
            runtime.metrics_path,
            {
                "step": int(runtime.global_step),
                "eval_return": float(runtime.last_eval_return),
                "best_return": float(runtime.best_return),
                "loss": float(runtime.last_loss),
                "epsilon": float(eps),
                "q_mean": float(runtime.last_q),
                "elapsed": float(elapsed),
                "num_updates": int(runtime.num_updates),
            },
        )
        rl_logger.log_eval_iteration(
            iteration=int(runtime.global_step),
            num_iterations=int(config.total_timesteps),
            frames_per_batch=1,
            eval_return=float(runtime.last_eval_return),
            heldout_return=None,
            best_return=float(runtime.best_return),
            algo_metrics={"loss": runtime.last_loss, "eps": eps, "q": runtime.last_q},
            algo_name="r2d2",
            elapsed=float(elapsed),
            step_override=int(runtime.global_step),
        )
        return
    if int(config.log_interval) > 0 and runtime.global_step % int(config.log_interval) < int(config.num_envs):
        rl_logger.log_progress_iteration(
            iteration=int(runtime.global_step),
            num_iterations=int(config.total_timesteps),
            frames_per_batch=1,
            elapsed=float(elapsed),
            algo_name="r2d2",
            step_override=int(runtime.global_step),
        )


def _maybe_save_r2d2_checkpoint(runtime: _R2D2Runtime, config: R2D2Config) -> None:
    if config.checkpoint_interval is None or int(config.checkpoint_interval) <= 0:
        return
    if runtime.global_step % int(config.checkpoint_interval) >= int(config.num_envs):
        return
    runtime.ckpt.save_both(
        {
            "step": int(runtime.global_step),
            "online": runtime.online.state_dict(),
            "target": runtime.target.state_dict(),
            "optimizer": runtime.optimizer.state_dict(),
            "best_return": float(runtime.best_return),
            "last_eval_return": float(runtime.last_eval_return),
            "num_updates": int(runtime.num_updates),
        },
        iteration=int(runtime.global_step),
    )


def _finalize_r2d2(runtime: _R2D2Runtime, config: R2D2Config) -> TrainResult:
    runtime.vec_env.close()
    total_time = float(time.time() - runtime.start_time)
    best = runtime.best_return if runtime.best_return == runtime.best_return else runtime.last_eval_return
    rl_logger.log_run_footer(
        best_return=float(best),
        total_iters_or_steps=int(runtime.global_step),
        total_time=total_time,
        algo_name="r2d2",
        step_label="steps",
    )
    return TrainResult(
        best_return=float(runtime.best_return),
        last_eval_return=float(runtime.last_eval_return),
        num_updates=int(runtime.num_updates),
        total_steps=int(runtime.global_step),
    )


def train_r2d2(config: R2D2Config) -> TrainResult:
    exp_dir = Path(config.exp_dir)
    exp_dir.mkdir(parents=True, exist_ok=True)
    write_config(str(exp_dir), config.to_dict())
    problem_seed = resolve_problem_seed(seed=int(config.seed), problem_seed=config.problem_seed)
    run_seed = global_seed_for_run(int(problem_seed))
    _seed_everything(run_seed)
    runtime = _build_r2d2_runtime(config, exp_dir=exp_dir, run_seed=int(run_seed), problem_seed=int(problem_seed))
    while runtime.global_step < int(config.total_timesteps):
        eps = _run_r2d2_env_step(runtime, config)
        _run_r2d2_optimizer_updates(runtime, config)
        elapsed = float(time.time() - runtime.start_time)
        _maybe_log_r2d2_progress(runtime, config, eps=float(eps), elapsed=elapsed)
        _maybe_save_r2d2_checkpoint(runtime, config)
    return _finalize_r2d2(runtime, config)
