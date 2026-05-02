import hashlib
import importlib
from dataclasses import asdict, dataclass
from typing import Any, NamedTuple, Optional

from analysis.data_io import TraceRecord
from common.collector import Collector

# Cumulative proposal-time budget for Modal timing sweep (``Optimizer._cum_dt_proposing``).
TIMING_SWEEP_MAX_CUMULATIVE_PROPOSAL_SECONDS = 5 * 60 * 60


class _SampleResult(NamedTuple):
    collector_log: Collector
    collector_trace: Collector
    trace_records: list[TraceRecord]
    stop_reason: Optional[str] = None


def _load_attr(module_parts: tuple[str, ...], attr_name: str):
    module_name = ".".join(module_parts)
    module = importlib.import_module(module_name)
    return getattr(module, attr_name)


@dataclass
class ExperimentConfig:
    exp_dir: str
    env_tag: str
    opt_name: str
    num_arms: int
    num_reps: int
    num_rounds: Optional[int] = None
    total_timesteps: Optional[int] = None
    num_denoise: Optional[int] = None
    num_denoise_passive: Optional[int] = None
    max_proposal_seconds: Optional[float] = None
    max_total_seconds: Optional[float] = None
    b_trace: bool = True
    video_enable: bool = False
    scale: Optional[str] = None  # "auto" | "low" | "medium" | "high" | "huge" for dim-based scaling
    video_num_episodes: int = 8
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None
    video_prefix: str = "bo"
    runtime_device: str = "auto"
    local_workers: int = 1
    policy_tag: Optional[str] = None

    def to_dir_name(self) -> str:
        budget_key = "num_rounds" if self.num_rounds is not None else "total_timesteps"
        budget_val = self.num_rounds if self.num_rounds is not None else self.total_timesteps
        config_str = (
            f"env={self.env_tag}--opt_name={self.opt_name}"
            f"--num_arms={self.num_arms}--{budget_key}={budget_val}"
            f"--num_reps={self.num_reps}--num_denoise={self.num_denoise}"
            f"--max_proposal_seconds={self.max_proposal_seconds}"
            f"--video_enable={self.video_enable}"
        )
        short_hash = hashlib.md5(config_str.encode()).hexdigest()[:8]
        return f"{self.exp_dir}/{short_hash}"

    def to_dir_name_legacy(self) -> str:
        budget_key = "num_rounds" if self.num_rounds is not None else "total_timesteps"
        budget_val = self.num_rounds if self.num_rounds is not None else self.total_timesteps
        return (
            f"{self.exp_dir}/env={self.env_tag}--opt_name={self.opt_name}"
            f"--num_arms={self.num_arms}--{budget_key}={budget_val}"
            f"--num_reps={self.num_reps}--num_denoise={self.num_denoise}"
        )

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExperimentConfig":
        max_prop = d.get("max_proposal_seconds")
        if max_prop in (None, "None"):
            max_prop = None
        else:
            max_prop = float(max_prop)
        max_total = d.get("max_total_seconds")
        if max_total in (None, "None"):
            max_total = None
        else:
            max_total = float(max_total)
        runtime_device = str(d.get("runtime_device", "auto")).strip().lower()
        if runtime_device not in {"auto", "cpu", "cuda"}:
            raise ValueError(f"runtime_device must be one of: auto, cpu, cuda (got: {runtime_device})")
        local_workers = int(d.get("local_workers", 1))
        if local_workers < 1:
            raise ValueError(f"local_workers must be >= 1 (got: {local_workers})")
        num_rounds = None if d.get("num_rounds") in (None, "None") else int(d["num_rounds"])
        total_timesteps = None if d.get("total_timesteps") in (None, "None") else int(d["total_timesteps"])
        if num_rounds is None and total_timesteps is None:
            raise ValueError("Either num_rounds or total_timesteps must be provided.")
        if num_rounds is not None and num_rounds < 1:
            raise ValueError(f"num_rounds must be >= 1 (got: {num_rounds})")
        if total_timesteps is not None and total_timesteps < 1:
            raise ValueError(f"total_timesteps must be >= 1 (got: {total_timesteps})")

        policy_tag = d.get("policy_tag")
        if policy_tag in (None, "None", ""):
            raise ValueError(
                "Missing required field 'policy_tag'. Set [experiment].policy_tag in TOML "
                "(e.g. policy_tag='pure-function'). Policy inference from env_tag is disabled."
            )
        policy_tag = str(policy_tag)

        return cls(
            exp_dir=d["exp_dir"],
            env_tag=d["env_tag"],
            opt_name=d["opt_name"],
            num_arms=int(d["num_arms"]),
            num_rounds=num_rounds,
            total_timesteps=total_timesteps,
            num_reps=int(d["num_reps"]),
            num_denoise=None if d.get("num_denoise") in (None, "None") else int(d["num_denoise"]),
            num_denoise_passive=None if d.get("num_denoise_passive") in (None, "None") else int(d["num_denoise_passive"]),
            max_proposal_seconds=max_prop,
            max_total_seconds=max_total,
            b_trace=true_false(d.get("b_trace", True)),
            video_enable=true_false(d.get("video_enable", False)),
            video_num_video_episodes=3,
            video_episode_selection="best",
            video_seed_base=None,
            video_prefix="bo",
            runtime_device=runtime_device,
            local_workers=local_workers,
            policy_tag=policy_tag,
        )


@dataclass
class RunConfig:
    opt_name: str
    num_rounds: Optional[int]
    num_arms: int
    num_denoise: Optional[int]
    num_denoise_passive: Optional[int]
    max_proposal_seconds: Optional[float]
    b_trace: bool
    trace_fn: str
    total_timesteps: Optional[int] = None
    bo_console: bool = True
    deadline: Optional[float] = None
    video_enable: bool = False
    video_num_episodes: int = 8
    video_num_video_episodes: int = 3
    video_episode_selection: str = "best"
    video_seed_base: Optional[int] = None
    video_prefix: str = "bo"
    runtime_device: str = "auto"
    problem: Any | None = None
    env_conf: Any | None = None

    def to_dict(self) -> dict:
        return asdict(self)


def true_false(string_bool):
    string_bool = str(string_bool).lower()
    if string_bool in ["false", "f"]:
        return False
    if string_bool in ["true", "t"]:
        return True
    assert False, string_bool
