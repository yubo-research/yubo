from __future__ import annotations

import zipfile
from pathlib import Path
from typing import Any


def _torch():
    import torch

    return torch


def _actor_snapshot_from_payload(payload: dict[str, Any]) -> dict[str, Any]:
    best_actor = payload.get("best_actor_state")
    if best_actor is not None:
        if not isinstance(best_actor, dict):
            raise TypeError("Checkpoint field 'best_actor_state' must be a dict when present.")
        return dict(best_actor)

    if "actor_backbone" in payload and "actor_head" in payload:
        snapshot = {
            "backbone": payload["actor_backbone"],
            "head": payload["actor_head"],
        }
        if "log_std" in payload:
            snapshot["log_std"] = payload["log_std"]
        return snapshot

    if "backbone" in payload and "head" in payload:
        return dict(payload)

    raise ValueError("Checkpoint does not contain a PPO actor snapshot.")


def _actor_snapshot_from_sb3_policy(policy_state: dict[str, Any]) -> dict[str, Any]:
    required = (
        "log_std",
        "mlp_extractor.policy_net.0.weight",
        "mlp_extractor.policy_net.0.bias",
        "mlp_extractor.policy_net.2.weight",
        "mlp_extractor.policy_net.2.bias",
        "action_net.weight",
        "action_net.bias",
    )
    missing = [key for key in required if key not in policy_state]
    if missing:
        raise ValueError(f"SB3 PPO checkpoint is missing required actor key(s): {', '.join(missing)}.")
    return {
        "backbone": {
            "0.weight": policy_state["mlp_extractor.policy_net.0.weight"],
            "0.bias": policy_state["mlp_extractor.policy_net.0.bias"],
            "2.weight": policy_state["mlp_extractor.policy_net.2.weight"],
            "2.bias": policy_state["mlp_extractor.policy_net.2.bias"],
        },
        "head": {
            "weight": policy_state["action_net.weight"],
            "bias": policy_state["action_net.bias"],
        },
        "log_std": policy_state["log_std"],
    }


def _load_actor_snapshot(path: Path) -> dict[str, Any]:
    torch = _torch()
    if path.suffix.lower() == ".zip":
        if not zipfile.is_zipfile(path):
            raise ValueError(f"Zip checkpoint is not a valid zip file: {path}")
        with zipfile.ZipFile(path) as zf:
            if "policy.pth" not in zf.namelist():
                raise ValueError(f"Zip checkpoint does not contain policy.pth: {path}")
            with zf.open("policy.pth") as f:
                policy_state = torch.load(f, map_location="cpu", weights_only=False)
        if not isinstance(policy_state, dict):
            raise TypeError(f"SB3 policy.pth must contain a dict, got {type(policy_state).__name__}.")
        return _actor_snapshot_from_sb3_policy(policy_state)

    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise TypeError(f"Checkpoint payload must be a dict, got {type(payload).__name__}.")
    return _actor_snapshot_from_payload(payload)


def _restore_actor_snapshot(policy: Any, snapshot: dict[str, Any]) -> None:
    missing = [name for name in ("actor_backbone", "actor_head") if not hasattr(policy, name)]
    if missing:
        raise ValueError(f"initial_policy_checkpoint requires a policy with actor_backbone and actor_head; missing: {', '.join(missing)}.")
    if hasattr(policy, "log_std") and "log_std" not in snapshot:
        raise ValueError("Checkpoint actor snapshot is missing log_std required by this policy.")

    from rl.core.actor_state import restore_backbone_head_snapshot

    torch = _torch()
    try:
        first_param = next(policy.parameters())
        device = first_param.device
    except StopIteration:
        device = torch.device("cpu")
    restore_backbone_head_snapshot(
        policy.actor_backbone,
        policy.actor_head,
        snapshot,
        log_std=getattr(policy, "log_std", None),
        device=device,
    )


def apply_policy_checkpoint(policy: Any, checkpoint_path: str | Path) -> Any:
    """Load a PPO actor checkpoint into a BO policy and make it the search origin."""
    path = Path(checkpoint_path).expanduser()
    from rl.checkpointing import resolve_checkpoint_path

    path = resolve_checkpoint_path(path)
    snapshot = _load_actor_snapshot(path)
    _restore_actor_snapshot(policy, snapshot)
    if not hasattr(policy, "_cache_flat_params_init") or not callable(policy._cache_flat_params_init):
        raise ValueError("initial_policy_checkpoint requires a policy that can rebase flat BO parameters.")
    policy._cache_flat_params_init()
    return policy


def load_policy_from_checkpoint(policy: Any, checkpoint_path: str | Path) -> Any:
    """Return a policy clone whose BO origin is the PPO actor checkpoint."""
    loaded_policy = policy.clone()
    apply_policy_checkpoint(loaded_policy, checkpoint_path)
    return loaded_policy
