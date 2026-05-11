from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ops.uhd_config import UHDConfig
from problems import pre_obj_vector_helpers as vector_helpers
from problems.nanoegg_subspace import _NanoEggSubspaceCodec


_DEFAULT_TOKENS_PER_EVAL = 100
_DEFAULT_SYNTHETIC_BYTES = 4096
_FIXED_POINT = 4
_FBIT = 4
_MAX_INT8 = 127
_PARAM_STANDARD = 0
_PARAM_MM = 1
_PARAM_EMB = 2


@dataclass(frozen=True)
class NanoEggObjectiveSpec:
    env_tag: str
    dataset: str
    policy_tag: str
    dtype: str
    n_layer: int
    n_embd: int
    vocab_size: int = 256


@dataclass(frozen=True)
class NanoEggPolicySnapshot:
    x: np.ndarray
    params: Any


def _require_jax():
    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError("NanoEgg UHD requires JAX. Run admin/setup-hyperscalees.sh first, then use ./ops/exp_uhd.py from that environment.") from exc
    return jax, jnp


def _as_int(value: int | None, default: int) -> int:
    if value is None:
        return int(default)
    return int(value)


def _quantized_normal(jax, jnp, key, shape: tuple[int, ...]):
    return jnp.round(jax.random.normal(key, shape) * (2**_FIXED_POINT)).astype(jnp.int8)


def _init_params(jax, jnp, *, seed: int, vocab_size: int, n_layer: int, n_embd: int):
    master_key = jax.random.key(int(seed) & 0xFFFFFFFF)
    model_key = jax.random.fold_in(master_key, 0)
    key_emb, key_block, key_head = jax.random.split(model_key, 3)
    d = int(n_embd)
    h = 4 * d
    block_keys = jax.random.split(key_block, int(n_layer))
    att_keys = jax.vmap(lambda key: jax.random.split(key, 2)[0])(block_keys)
    mlp_keys = jax.vmap(lambda key: jax.random.split(key, 2)[1])(block_keys)
    gru_keys = jax.vmap(lambda key: jax.random.split(key, 4))(att_keys)
    mlp_linear_keys = jax.vmap(lambda key: jax.random.split(key, 2))(mlp_keys)
    init_dd = jax.vmap(lambda key: _quantized_normal(jax, jnp, key, (d, d)))
    init_hd = jax.vmap(lambda key: _quantized_normal(jax, jnp, key, (h, d)))
    init_dh = jax.vmap(lambda key: _quantized_normal(jax, jnp, key, (d, h)))

    params = {
        "emb": _quantized_normal(jax, jnp, key_emb, (int(vocab_size), d)),
        "blocks": {
            "ln1": {"weight": jnp.ones((int(n_layer), d), dtype=jnp.int8) * (2**_FIXED_POINT)},
            "att": {
                "Wf": init_dd(gru_keys[:, 0]),
                "Uf": init_dd(gru_keys[:, 1]),
                "bf": jnp.zeros((int(n_layer), d), dtype=jnp.int8),
                "Wh": init_dd(gru_keys[:, 2]),
                "Uh": init_dd(gru_keys[:, 3]),
                "bh": jnp.zeros((int(n_layer), d), dtype=jnp.int8),
            },
            "ln2": {"weight": jnp.ones((int(n_layer), d), dtype=jnp.int8) * (2**_FIXED_POINT)},
            "mlp": {
                "0": {"weight": init_hd(mlp_linear_keys[:, 0])},
                "1": {"weight": init_dh(mlp_linear_keys[:, 1])},
            },
        },
        "ln_out": {"weight": jnp.ones((d,), dtype=jnp.int8) * (2**_FIXED_POINT)},
        "head": _quantized_normal(jax, jnp, key_head, (int(vocab_size), d)),
    }
    es_map = {
        "emb": _PARAM_EMB,
        "blocks": {
            "ln1": {"weight": _PARAM_STANDARD},
            "att": {
                "Wf": _PARAM_MM,
                "Uf": _PARAM_MM,
                "bf": _PARAM_STANDARD,
                "Wh": _PARAM_MM,
                "Uh": _PARAM_MM,
                "bh": _PARAM_STANDARD,
            },
            "ln2": {"weight": _PARAM_STANDARD},
            "mlp": {
                "0": {"weight": _PARAM_MM},
                "1": {"weight": _PARAM_MM},
            },
        },
        "ln_out": {"weight": _PARAM_STANDARD},
        "head": _PARAM_MM,
    }
    return params, es_map


def _clip_int8(jnp, x):
    return jnp.clip(x, -_MAX_INT8, _MAX_INT8).astype(jnp.int8)


def _clipped_add(jnp, *values):
    total = sum(value.astype(jnp.int32) for value in values)
    return _clip_int8(jnp, total)


def _mm(jnp, weight, x):
    base = jnp.dot(x, weight.T, preferred_element_type=jnp.int32)
    scale = (2**_FIXED_POINT) * int(np.sqrt(int(weight.shape[-1])))
    return _clip_int8(jnp, base // int(scale))


def _layer_norm(jnp, x, weight):
    weight_i32 = weight.astype(jnp.int32)
    abs_sum = jnp.maximum(jnp.sum(jnp.abs(x).astype(jnp.int32)) // int(x.size), 1)
    numerator = (x.astype(jnp.int32) * weight_i32).astype(jnp.int16).astype(jnp.int32)
    return _clip_int8(jnp, numerator // abs_sum)


def _egg_gru(jnp, att, x, state):
    ft = _clipped_add(jnp, _mm(jnp, att["Wf"], x), _mm(jnp, att["Uf"], state), att["bf"])
    gated_past = (((ft.astype(jnp.int32) + _MAX_INT8) * state.astype(jnp.int32)) >> (_FBIT + 4)).astype(jnp.int8)
    ht = _clipped_add(jnp, _mm(jnp, att["Wh"], x), _mm(jnp, att["Uh"], gated_past), att["bh"])
    next_state = state + (((ft.astype(jnp.int32) + _MAX_INT8) * (ht.astype(jnp.int32) - state.astype(jnp.int32))) >> (_FBIT + 4)).astype(jnp.int8)
    return next_state, next_state


def _layer_egg(jnp, block, x, state):
    residual = x
    x = _layer_norm(jnp, x, block["ln1"]["weight"])
    x, state = _egg_gru(jnp, block["att"], x, state)
    x = _clipped_add(jnp, x, residual)

    residual = x
    x = _layer_norm(jnp, x, block["ln2"]["weight"])
    x = _mm(jnp, block["mlp"]["0"]["weight"], x)
    x = _mm(jnp, block["mlp"]["1"]["weight"], x)
    x = _clipped_add(jnp, x, residual)
    return x, state


def _int_log_likelihood(jnp, logits, target):
    logits_i32 = logits.astype(jnp.int32) + 128
    target_logits = logits_i32[target]
    exp2 = jnp.asarray((np.exp2(np.arange(256) / (2**_FBIT)) * (2**_FBIT)).astype(np.int32))
    exp_sum = jnp.sum(exp2[logits_i32])
    logsumexp = (jnp.log2(exp_sum.astype(jnp.float32) / float(2**_FBIT)) * float(2**_FBIT)).astype(jnp.int32)
    return target_logits - logsumexp


def _score_sequence(jax, jnp, params, tokens):
    inputs = tokens[:-1].astype(jnp.int32)
    targets = tokens[1:].astype(jnp.int32)
    init_state = jnp.zeros_like(params["blocks"]["att"]["bh"], dtype=jnp.int8)

    def step(state, pair):
        token, target = pair
        state = jax.lax.select(token == 0, jnp.zeros_like(state), state)
        x = params["emb"][token]

        def block_loop(x_inner, inputs):
            block, state_inner = inputs
            x_inner, state_i = _layer_egg(jnp, block, x_inner, state_inner)
            return x_inner, state_i

        x, next_state = jax.lax.scan(block_loop, x, (params["blocks"], state), unroll=True)
        logits = _mm(jnp, params["head"], _layer_norm(jnp, x, params["ln_out"]["weight"]))
        ll = _int_log_likelihood(jnp, logits, target)
        return next_state, ll

    _state, log_likelihood = jax.lax.scan(step, init_state, (inputs, targets))
    return jnp.sum(log_likelihood).astype(jnp.float32) / (jnp.asarray(inputs.size, dtype=jnp.float32) * float(2**_FBIT))


def _synthetic_bytes(*, seed: int, size: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    data = rng.integers(1, 256, size=max(int(size), 2), dtype=np.uint8)
    data[::97] = 0
    return data


def _minipile_cache_path(cfg: UHDConfig) -> Path:
    root = Path(cfg.hf_home).expanduser() if cfg.hf_home else Path.home() / ".cache" / "yubo" / "nanoegg"
    suffix = "full" if cfg.sub_dataset_size is None else f"{int(cfg.sub_dataset_size)}"
    return root / f"minipile_validation_bytes_{suffix}.npy"


def _load_minipile_bytes(cfg: UHDConfig) -> np.ndarray:
    cache_path = _minipile_cache_path(cfg)
    if cache_path.is_file():
        data = np.asarray(np.load(cache_path), dtype=np.uint8)
        print(f"NANOEGG: loaded cached MiniPile bytes path={cache_path} bytes={data.size}", flush=True)
        return data
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError("NanoEgg MiniPile objective requires the 'datasets' package.") from exc
    print(f"NANOEGG: loading MiniPile validation split cache_path={cache_path}", flush=True)
    ds = load_dataset("JeanKaddour/minipile", split="validation")
    arrays = []
    limit = cfg.sub_dataset_size
    total = 0
    next_report = 16 * 1024 * 1024
    for row in ds:
        encoded = np.frombuffer(("\0" + str(row["text"])).encode("utf-8"), dtype=np.uint8)
        arrays.append(encoded)
        total += int(encoded.size)
        if total >= next_report:
            print(f"NANOEGG: loaded MiniPile bytes={total}", flush=True)
            next_report += 16 * 1024 * 1024
        if limit is not None and total >= int(limit):
            break
    data = np.concatenate(arrays).astype(np.uint8)
    if limit is not None:
        data = data[: int(limit)]
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache_path, data)
    print(f"NANOEGG: cached MiniPile bytes path={cache_path} bytes={data.size}", flush=True)
    return data


def _load_objective_bytes(cfg: UHDConfig, spec: NanoEggObjectiveSpec, *, tokens_per_eval: int) -> np.ndarray:
    min_size = max(_DEFAULT_SYNTHETIC_BYTES, int(tokens_per_eval) * max(2, int(cfg.num_envs) + 1) + 1)
    if spec.dataset == "synthetic":
        return _synthetic_bytes(seed=0 if cfg.problem_seed is None else int(cfg.problem_seed), size=max(min_size, _as_int(cfg.sub_dataset_size, min_size)))
    if spec.dataset == "minipile":
        data = _load_minipile_bytes(cfg)
        if data.size < min_size:
            reps = int(np.ceil(min_size / max(int(data.size), 1)))
            data = np.tile(data, reps)
        return data.astype(np.uint8)
    raise ValueError(f"Unsupported NanoEgg dataset {spec.dataset!r}; expected 'minipile' or 'synthetic'.")


class NanoEggUHDObjective:
    def __init__(self, cfg: UHDConfig, spec: NanoEggObjectiveSpec) -> None:
        if spec.dtype != "int8":
            raise ValueError(f"NanoEgg currently supports dtype='int8' only, got {spec.dtype!r}.")
        self.cfg = cfg
        self.spec = spec
        self.jax, self.jnp = _require_jax()
        self.vocab_size = int(spec.vocab_size)
        self.tokens_per_eval = int(cfg.pretrain_generation_length or min(int(cfg.max_tokens), _DEFAULT_TOKENS_PER_EVAL))
        if self.tokens_per_eval < 2:
            raise ValueError("NanoEgg tokens_per_eval must be >= 2.")
        seed = 0 if cfg.problem_seed is None else int(cfg.problem_seed)
        print(
            f"NANOEGG: initializing int8 EGG n_layer={int(spec.n_layer)} n_embd={int(spec.n_embd)} vocab_size={self.vocab_size}",
            flush=True,
        )
        base_params, es_map = _init_params(
            self.jax,
            self.jnp,
            seed=seed,
            vocab_size=self.vocab_size,
            n_layer=int(spec.n_layer),
            n_embd=int(spec.n_embd),
        )
        self._codec = _NanoEggSubspaceCodec(
            self.jax,
            self.jnp,
            base_params,
            es_map,
            dim=int(cfg.pretrain_search_dim),
            delta_scale=float(cfg.pretrain_delta_scale),
            seed=seed,
            lora_only=bool(cfg.pretrain_lora_only),
            basis_max_leaves=cfg.pretrain_basis_max_leaves,
        )
        self.dim = int(self._codec.dim)
        self.x0 = self._codec.x0.copy()
        self.steps_per_episode = int(cfg.steps_per_episode)
        self.num_envs = int(cfg.num_envs)
        self._tokens = _load_objective_bytes(cfg, spec, tokens_per_eval=self.tokens_per_eval)
        self._embed_indices: np.ndarray | None = None
        self._score_jit = self.jax.jit(lambda params, tokens: _score_sequence(self.jax, self.jnp, params, tokens))
        print(
            f"NANOEGG: objective ready env={spec.env_tag} policy={spec.policy_tag} dim={self.dim} "
            f"tokens={self._tokens.size} tokens_per_eval={self.tokens_per_eval} num_envs={self.num_envs}",
            flush=True,
        )

    def _segment(self, seed: int) -> np.ndarray:
        total = int(self._tokens.size)
        length = int(self.tokens_per_eval) + 1
        if total <= length:
            return np.resize(self._tokens, length).astype(np.uint8)
        rng = np.random.default_rng(int(seed))
        start = int(rng.integers(0, total - length))
        return self._tokens[start : start + length].astype(np.uint8)

    def _decode(self, x: np.ndarray):
        return self._codec.decode(np.asarray(x, dtype=np.float64))

    def make_policy(self, x: np.ndarray):
        x_arr = np.asarray(x, dtype=np.float64).copy()
        return NanoEggPolicySnapshot(x=x_arr, params=self._decode(x_arr))

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        params = self._decode(x)
        scores = []
        for i in range(int(self.num_envs)):
            tokens = self.jnp.asarray(self._segment(int(seed) + i), dtype=self.jnp.uint8)
            scores.append(float(self._score_jit(params, tokens)))
        values = np.asarray(scores, dtype=np.float64)
        se = 0.0 if values.size <= 1 else float(np.std(values, ddof=0) / np.sqrt(values.size))
        return float(np.mean(values)), se

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        return vector_helpers.evaluate_many_serial(self.evaluate, x_batch, seed=seed)

    def configure_embedding(self, num_probes: int) -> None:
        rng = np.random.default_rng(123)
        n = min(max(int(num_probes), 1), self.dim)
        self._embed_indices = rng.choice(self.dim, size=n, replace=False)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        if self._embed_indices is None:
            self.configure_embedding(64)
        assert self._embed_indices is not None
        return np.asarray(x_batch, dtype=np.float64)[:, self._embed_indices]

    def embed(self, x: np.ndarray) -> np.ndarray:
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        rng = np.random.default_rng(int(seed))
        if num_module_target is not None:
            num_dim_target = num_module_target
        if num_dim_target is None:
            return rng.standard_normal(self.dim).astype(np.float64)
        target = float(num_dim_target)
        if target <= 0:
            raise ValueError("NanoEgg perturb target must be > 0.")
        noise = np.zeros(self.dim, dtype=np.float64)
        if 0 < target < 1:
            mask = rng.random(self.dim) < target
            if not np.any(mask):
                mask[int(rng.integers(self.dim))] = True
            noise[mask] = rng.standard_normal(int(np.sum(mask)))
            return noise
        k = min(max(int(target), 1), self.dim)
        idx = rng.choice(self.dim, size=k, replace=False)
        noise[idx] = rng.standard_normal(k)
        return noise

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
            raise ValueError(f"NanoEgg only supports eggroll perturb materialization, got {noiser_name!r}.")
        return self._codec.sample_eggroll_direction(
            seed=int(seed),
            rank=int(rank),
            group_size=int(group_size),
            freeze_nonlora=bool(freeze_nonlora),
        )

    def close(self) -> None:
        return None


def build_nanoegg_uhd_objective(*, cfg: UHDConfig, spec) -> NanoEggUHDObjective:
    local_spec = NanoEggObjectiveSpec(
        env_tag=str(spec.env_tag),
        dataset=str(spec.dataset),
        policy_tag=str(spec.policy_tag),
        dtype=str(spec.dtype),
        n_layer=int(spec.n_layer),
        n_embd=int(spec.n_embd),
        vocab_size=int(getattr(spec, "vocab_size", 256)),
    )
    return NanoEggUHDObjective(cfg, local_spec)
