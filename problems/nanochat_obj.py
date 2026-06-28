from __future__ import annotations

import logging
import os
import threading

import numpy as np
import torch

import third_party.nanochat.core_eval  # noqa: F401
from ops.uhd_config import UHDConfig
from problems.nanochat_lora import _NanochatSubspaceCodec
from problems.uhd_obj_types import UHDVectorObjectiveMixin

_log = logging.getLogger(__name__).info


_NANOCHAT_POLICIES = {
    "nanochat:tiny": {"n_layer": 4, "n_head": 4, "n_embd": 128, "synthetic_vocab_size": 8192},
    "nanochat:d12": {"n_layer": 12, "n_head": 6, "n_embd": 768, "synthetic_vocab_size": 32768},
    "nanochat:d24": {"n_layer": 24, "n_head": 6, "n_embd": 768, "synthetic_vocab_size": 32768},
}


class NanochatUHDVectorObjective(UHDVectorObjectiveMixin):
    """Bridge for nanochat GPT models as a UHD vector objective."""

    def __init__(self, cfg: UHDConfig) -> None:
        from third_party.nanochat.gpt import GPT, GPTConfig
        from third_party.nanochat.tokenizer import get_token_bytes

        self.cfg = cfg
        self._eval_count = 0
        self._lock = threading.Lock()

        policy_tag = str(cfg.policy_tag or "nanochat:d12")
        policy = _resolve_nanochat_policy(policy_tag)
        vocab_size = _nanochat_vocab_size(cfg, policy)

        gpt_cfg = GPTConfig(
            n_layer=policy["n_layer"],
            n_head=policy["n_head"],
            n_kv_head=policy["n_head"],
            n_embd=policy["n_embd"],
            sequence_len=int(getattr(cfg, "max_tokens", 256)),
            vocab_size=vocab_size,
        )

        device = "cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = GPT(gpt_cfg).to(device)
        self.model.init_weights()
        self.model.eval()
        self.num_model_params = int(sum(param.numel() for param in self.model.parameters()))

        self._codec = _NanochatSubspaceCodec(
            self.model,
            dim=int(getattr(cfg, "text_search_dim", 128)),
            delta_scale=float(getattr(cfg, "text_delta_scale", 1.0)),
            seed=int(getattr(cfg, "problem_seed", 0)),
        )

        self._x0 = self._codec.x0

        # Real-world data setup (BPB objective)
        tag_parts = str(cfg.env_tag).split(":")
        dataset_name = tag_parts[1] if len(tag_parts) > 1 else "synthetic"
        self._bin_path = f"data/{dataset_name}.bin"
        self._has_real_data = os.path.exists(self._bin_path)

        # Try to load token_bytes if they exist; otherwise fallback
        try:
            self._token_bytes = get_token_bytes(device=device)
        except Exception:
            # Fallback: assume each token is 1 byte for synthetic/simple testing
            self._token_bytes = torch.ones(gpt_cfg.vocab_size, dtype=torch.int64, device=device)

        _log(
            "NanochatUHD: "
            f"policy={policy_tag} layers={gpt_cfg.n_layer} embd={gpt_cfg.n_embd} vocab={gpt_cfg.vocab_size} "
            f"dim={self.dim} dataset={dataset_name} real_data={self._has_real_data} "
            f"model_params={self.num_model_params:,} search_params={self._codec.num_total_params:,}"
        )

        if not self._has_real_data and dataset_name != "synthetic":
            raise RuntimeError(f"Real dataset '{dataset_name}' requested but {self._bin_path} not found. Run scripts/prepare_tinystories.py first.")

        # Behavioral Embedder Setup
        # We need a fixed set of tokens to use as "probes" for the behavioral signature
        self._probe_tokens = torch.randint(0, gpt_cfg.vocab_size, (1, 8), device=device)
        self._embed_num_probes = 0

    @property
    def dim(self) -> int:
        return self._codec.dim

    @property
    def x0(self) -> np.ndarray:
        return self._x0.copy()

    @property
    def steps_per_episode(self) -> int:
        return 1

    @property
    def num_envs(self) -> int:
        return int(getattr(self.cfg, "num_envs", 1))

    def make_policy(self, x: np.ndarray):
        return self.model

    @torch.no_grad()
    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        """Evaluates model using nanochat's BPB (Bits Per Byte) metric."""
        from third_party.nanochat.loss_eval import evaluate_bpb

        with self._lock:
            self._codec.apply(x)
            try:
                device = self.model.get_device()
                batch_size = self.num_envs
                seq_len = self.model.config.sequence_len

                # Setup data stream
                if self._has_real_data:
                    # In a real run, this would be a shard of a dataset
                    # For this bridge, we maintain the ability to run on Modal
                    batches = self._get_data_iterator(batch_size, seq_len, seed, device)
                else:
                    # Fallback to synthetic if data not available
                    batches = self._get_synthetic_iterator(batch_size, seq_len, seed, device)

                # Evaluate using nanochat's BPB function
                bpb = evaluate_bpb(self.model, batches, steps=1, token_bytes=self._token_bytes)

                # UHD optimizer maximizes, so return negative BPB
                y = -float(bpb)

                self._eval_count += 1
                return y, 0.0
            finally:
                self._codec.revert(x)

    def _get_synthetic_iterator(self, b, t, seed, device):
        rng = torch.Generator(device=device)
        rng.manual_seed(int(seed))
        while True:
            idx = torch.randint(0, self.model.config.vocab_size, (b, t), generator=rng, device=device)
            targets = torch.roll(idx, -1, dims=1)
            targets[:, -1] = -1
            yield idx, targets

    def _get_data_iterator(self, b, t, seed, device):
        from problems.nanochat_dataloader import BinDataLoader

        # Map env_tag to data file
        # e.g., 'nanochat:tinystories' -> 'data/tinystories.bin'
        tag_parts = str(self.cfg.env_tag).split(":")
        dataset_name = tag_parts[1] if len(tag_parts) > 1 else "synthetic"

        bin_path = f"data/{dataset_name}.bin"
        if os.path.exists(bin_path):
            loader = BinDataLoader(bin_path, b, t)
            return loader.iterator(seed, device)

        # Fallback to synthetic if bin not found
        return self._get_synthetic_iterator(b, t, seed, device)

    def configure_embedding(self, num_probes: int) -> None:
        self._embed_num_probes = int(num_probes)

    @torch.no_grad()
    def embed(self, x: np.ndarray) -> np.ndarray:
        if self._embed_num_probes <= 0:
            return np.zeros((0,), dtype=np.float32)

        with self._lock:
            self._codec.apply(x)
            try:
                # Fast Behavioral Signature: only run the first 4 layers
                # This is much faster than a full 24-layer forward pass
                # while still capturing the behavioral shift in the weights.
                h = self.model(self._probe_tokens, up_to_layer=4)  # (1, T, n_embd)
                # Take first token's first N embedding dims as the signature
                # Cast to float32 before numpy conversion (BFloat16 not supported by numpy)
                sig = h[0, 0, : self._embed_num_probes].float().cpu().numpy()
                return sig.astype(np.float32)
            finally:
                self._codec.revert(x)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        # Serial embedding loop
        rows = np.asarray(x_batch, dtype=np.float64)
        zs = [self.embed(x) for x in rows]
        if not zs:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(zs).astype(np.float32)

    def sample_eggroll_noiser_noise(
        self,
        _x: np.ndarray,
        *,
        seed: int,
        noiser_name: str = "eggroll",
        rank: int = 1,
        group_size: int = 0,
        freeze_nonlora: bool = False,
    ) -> np.ndarray:
        if noiser_name != "eggroll":
            raise ValueError(f"Nanochat only supports eggroll perturb materialization, got {noiser_name!r}.")
        return self._codec.sample_eggroll_direction(
            seed=int(seed),
            rank=int(rank),
            freeze_nonlora=bool(freeze_nonlora),
        )


def is_nanochat_env(env_tag: str) -> bool:
    return str(env_tag).startswith("nanochat:")


def _resolve_nanochat_policy(policy_tag: str) -> dict[str, int]:
    try:
        return dict(_NANOCHAT_POLICIES[str(policy_tag)])
    except KeyError as exc:
        supported = ", ".join(sorted(_NANOCHAT_POLICIES))
        raise KeyError(f"Unknown nanochat policy_tag {policy_tag!r}. Supported policies: {supported}") from exc


def _nanochat_vocab_size(cfg: UHDConfig, policy: dict[str, int]) -> int:
    if "tinystories" in str(cfg.env_tag):
        return 50257
    return int(policy["synthetic_vocab_size"])
