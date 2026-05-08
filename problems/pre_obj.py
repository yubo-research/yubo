from __future__ import annotations

import importlib
import pickle
import re
import time
import warnings
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any

import numpy as np

from ops.uhd_config import UHDConfig


_STACK_ERROR = (
    "Real HyperscaleES pretraining UHD requires the separate HyperscaleES environment. "
    "Run admin/setup-hyperscalees.sh first, then use the plain python CLI from that environment."
)
_NANOEGG_STACK_ERROR = (
    "Real NanoEgg pretraining UHD requires the installed nanoegg.uhd module to expose "
    "build_uhd_objective(cfg=..., spec=...). The old pretrain:nanoegg surrogate path has been removed."
)


@dataclass(frozen=True)
class HyperscaleESPretrainSpec:
    env_tag: str
    task: str
    model_choice: str
    thinking_length: int
    answer_length: int
    rwkv_type: str = "BaseRWKV"
    dtype: str | None = None

    @property
    def generation_length(self) -> int:
        return int(self.thinking_length) + int(self.answer_length)


@dataclass(frozen=True)
class NanoEggPretrainSpec:
    env_tag: str
    dataset: str
    dtype: str
    n_layer: int
    n_embd: int


@dataclass(frozen=True)
class _HyperscaleESStack:
    jax: Any
    jnp: Any
    simple_es_tree_key: Any
    get_model: Any
    legacy_tokenizer_cls: Any
    noiser_cls: Any
    all_tasks: Any
    validation_tasks: Any


_PRETRAIN_SPECS: dict[str, HyperscaleESPretrainSpec] = {
    "pretrain:hyperscalees:gsm8k-7w3b": HyperscaleESPretrainSpec(
        env_tag="pretrain:hyperscalees:gsm8k-7w3b",
        task="gsm8k",
        model_choice="7w3B",
        thinking_length=256,
        answer_length=256,
    ),
    "pretrain:hyperscalees:countdownn-7w1p5b": HyperscaleESPretrainSpec(
        env_tag="pretrain:hyperscalees:countdownn-7w1p5b",
        task="countdownn",
        model_choice="7w1.5B",
        thinking_length=100,
        answer_length=100,
    ),
}

_PRETRAIN_TAG_PREFIX = "pretrain:hyperscalees:"
_NANOEGG_TAG_PREFIX = "pretrain:nanoegg:"

_HYPERSCALEES_MODEL_CHOICES_BY_TAG = {
    "7w0p1b": "7w0.1B",
    "7w0p4b": "7w0.4B",
    "7w1p5b": "7w1.5B",
    "7w3b": "7w3B",
    "7n0p1b": "7n0.1B",
    "7n0p4b": "7n0.4B",
    "7n1p5b": "7n1.5B",
    "7g0p1b": "7g0.1B",
    "7g0p4b": "7g0.4B",
    "7g1p5b": "7g1.5B",
    "7g2p9b": "7g2.9B",
    "7g7b": "7g7B",
    "7g14b": "7g14B",
}

_BASE_LLM_BANDIT_TASKS = frozenset(
    {
        "fastzero",
        "uniquetok",
        "reptok",
        "digits",
        "gsm8k",
        "gsm8ksft",
        "countdownn",
        "aime24",
        "aime25",
    }
)

_REASONING_GYM_TASKS = frozenset(
    {
        "ab",
        "acre",
        "advanced_geometry",
        "aiw",
        "arc_1d",
        "arc_agi",
        "base_conversion",
        "basic_arithmetic",
        "bf",
        "binary_alternation",
        "binary_matrix",
        "bitwise_arithmetic",
        "boxnet",
        "caesar_cipher",
        "calendar_arithmetic",
        "chain_sum",
        "circuit_logic",
        "codeio",
        "coin_flip",
        "color_cube_rotation",
        "complex_arithmetic",
        "composite",
        "count_bits",
        "count_primes",
        "course_schedule",
        "cryptarithm",
        "decimal_arithmetic",
        "decimal_chain_sum",
        "dice",
        "emoji_mystery",
        "family_relationships",
        "figlet_font",
        "fraction_simplification",
        "futoshiki",
        "game_of_life",
        "game_of_life_halting",
        "gcd",
        "graph_color",
        "group_anagrams",
        "gsm_symbolic",
        "intermediate_integration",
        "isomorphic_strings",
        "jugs",
        "kakurasu",
        "knight_swap",
        "knights_knaves",
        "largest_island",
        "lcm",
        "leg_counting",
        "letter_counting",
        "letter_jumble",
        "list_functions",
        "mahjong_puzzle",
        "manipulate_matrix",
        "maze",
        "mini_sudoku",
        "modulo_grid",
        "n_queens",
        "needle_haystack",
        "number_filtering",
        "number_format",
        "number_sequence",
        "number_sorting",
        "palindrome_generation",
        "palindrome_partitioning",
        "polynomial_equations",
        "polynomial_multiplication",
        "pool_matrix",
        "power_function",
        "prime_factorization",
        "products",
        "propositional_logic",
        "puzzle24",
        "quantum_lock",
        "ransom_note",
        "rearc",
        "rectangle_count",
        "rotate_matrix",
        "rotten_oranges",
        "rubiks_cube",
        "rush_hour",
        "self_reference",
        "sentence_reordering",
        "shortest_path",
        "simple_equations",
        "simple_geometry",
        "simple_integration",
        "sokoban",
        "spell_backward",
        "spiral_matrix",
        "string_insertion",
        "string_manipulation",
        "string_splitting",
        "string_synthesis",
        "sudoku",
        "survo",
        "syllogism",
        "time_intervals",
        "tower_of_hanoi",
        "tsumego",
        "word_ladder",
        "word_sequence_reversal",
        "word_sorting",
        "zebra_puzzles",
    }
)

_HYPERSCALEES_LLM_BANDIT_TASKS = _BASE_LLM_BANDIT_TASKS | _REASONING_GYM_TASKS


def _model_choice_to_tag(model_choice: str) -> str:
    return str(model_choice).replace(".", "p").lower()


def _hyperscalees_model_choices_by_tag() -> dict[str, str]:
    try:
        from hyperscalees.models.llm.auto import models
    except Exception:
        return dict(_HYPERSCALEES_MODEL_CHOICES_BY_TAG)
    return {_model_choice_to_tag(str(model_choice)): str(model_choice) for model_choice in models}


def _hyperscalees_llm_bandit_tasks() -> frozenset[str]:
    try:
        from hyperscalees.environments.llm_bandits import all_tasks
    except Exception:
        return _HYPERSCALEES_LLM_BANDIT_TASKS
    return frozenset(str(task) for task in all_tasks)


def supported_hyperscalees_llm_bandit_tasks() -> tuple[str, ...]:
    return tuple(sorted(_hyperscalees_llm_bandit_tasks()))


def is_hyperscalees_pretrain_env(env_tag: str) -> bool:
    return str(env_tag).startswith(_PRETRAIN_TAG_PREFIX)


def is_nanoegg_pretrain_env(env_tag: str) -> bool:
    return str(env_tag).startswith(_NANOEGG_TAG_PREFIX)


def resolve_nanoegg_pretrain_spec(env_tag: str) -> NanoEggPretrainSpec:
    tag = str(env_tag)
    if not tag.startswith(_NANOEGG_TAG_PREFIX):
        raise ValueError(f"Unsupported NanoEgg pretraining env_tag: {tag!r}. Expected prefix {_NANOEGG_TAG_PREFIX!r}.")
    suffix = tag.removeprefix(_NANOEGG_TAG_PREFIX)
    match = re.fullmatch(r"(?P<dataset>[a-z0-9_+-]+)-(?P<dtype>[a-z0-9]+)-(?P<layers>[1-9][0-9]*)l(?P<embd>[1-9][0-9]*)d", suffix)
    if match is None:
        raise ValueError(
            f"Unsupported NanoEgg pretraining env_tag: {tag!r}. "
            "Use 'pretrain:nanoegg:<dataset>-<dtype>-<layers>l<embd>d', "
            "e.g. 'pretrain:nanoegg:minipile-int8-6l256d'."
        )
    return NanoEggPretrainSpec(
        env_tag=tag,
        dataset=match.group("dataset"),
        dtype=match.group("dtype"),
        n_layer=int(match.group("layers")),
        n_embd=int(match.group("embd")),
    )


def resolve_hyperscalees_pretrain_spec(env_tag: str) -> HyperscaleESPretrainSpec:
    tag = str(env_tag)
    if tag in _PRETRAIN_SPECS:
        return _PRETRAIN_SPECS[tag]
    if not tag.startswith(_PRETRAIN_TAG_PREFIX):
        raise ValueError(f"Unsupported real HyperscaleES pretraining env_tag: {tag!r}. Expected prefix {_PRETRAIN_TAG_PREFIX!r}.")

    suffix = tag.removeprefix(_PRETRAIN_TAG_PREFIX)
    if "-" not in suffix:
        raise ValueError(
            f"Unsupported real HyperscaleES pretraining env_tag: {tag!r}. "
            "Use 'pretrain:hyperscalees:<task>-<model>', e.g. 'pretrain:hyperscalees:basic_arithmetic-7w1p5b'."
        )
    task, model_tag = suffix.rsplit("-", 1)
    model_choices = _hyperscalees_model_choices_by_tag()
    model_choice = model_choices.get(model_tag.lower())
    if model_choice is None:
        known_models = ", ".join(sorted(model_choices))
        raise ValueError(f"Unsupported HyperscaleES model tag {model_tag!r} in env_tag {tag!r}. Known model tags: {known_models}.")
    tasks = _hyperscalees_llm_bandit_tasks()
    if task not in tasks:
        known_tasks = ", ".join(sorted(tasks))
        raise ValueError(f"Unsupported HyperscaleES LLM bandit task {task!r} in env_tag {tag!r}. Known tasks: {known_tasks}.")
    return HyperscaleESPretrainSpec(
        env_tag=tag,
        task=task,
        model_choice=model_choice,
        thinking_length=100,
        answer_length=100,
    )


def _require_stack() -> _HyperscaleESStack:
    try:
        import jax
        import jax.numpy as jnp
        from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks
        from hyperscalees.models.common import simple_es_tree_key
        from hyperscalees.models.llm.auto import get_model
        from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer
        from hyperscalees.noiser.base_noiser import Noiser
    except ImportError as exc:
        raise ImportError(_STACK_ERROR) from exc
    return _HyperscaleESStack(
        jax=jax,
        jnp=jnp,
        simple_es_tree_key=simple_es_tree_key,
        get_model=get_model,
        legacy_tokenizer_cls=LegacyWorldTokenizer,
        noiser_cls=Noiser,
        all_tasks=all_tasks,
        validation_tasks=validation_tasks,
    )


def _load_nanoegg_uhd_module():
    try:
        return importlib.import_module("nanoegg.uhd")
    except ImportError as exc:
        raise ImportError(_NANOEGG_STACK_ERROR) from exc


def _build_external_nanoegg_objective(cfg: UHDConfig, spec: NanoEggPretrainSpec):
    module = _load_nanoegg_uhd_module()
    build = getattr(module, "build_uhd_objective", None) or getattr(module, "build_objective", None)
    if not callable(build):
        raise RuntimeError(f"{_NANOEGG_STACK_ERROR} Module {module.__name__!r} does not expose build_uhd_objective or build_objective.")
    try:
        return build(cfg=cfg, spec=spec)
    except TypeError:
        return build(cfg, spec)


def _log(message: str) -> None:
    print(f"UHD-HyperscaleES: {message}", flush=True)


def _load_hyperscalees_model(get_model, spec: HyperscaleESPretrainSpec):
    try:
        return get_model(
            spec.model_choice,
            rwkv_type=spec.rwkv_type,
            verbose=True,
            dtype=spec.dtype,
        )
    except (pickle.UnpicklingError, EOFError):
        warnings.warn(
            "HyperscaleES converted model cache appears corrupt; retrying with reload_cache=True. "
            "If this repeats, remove ~/.cache/yubo/hyperscalees/hyperscalees_cache for this model and check disk space.",
            RuntimeWarning,
            stacklevel=2,
        )
        return get_model(
            spec.model_choice,
            rwkv_type=spec.rwkv_type,
            verbose=True,
            dtype=spec.dtype,
            reload_cache=True,
        )


def _fold_in_thread_key(jax, key, epoch, thread_idx):
    return jax.random.fold_in(jax.random.fold_in(key, epoch), thread_idx)


def _build_generate_thread(stack: _HyperscaleESStack, model, noiser, frozen_noiser_params, config, base_evo_keys, master_gen_key, temperature: float):
    jax = stack.jax
    jnp = stack.jnp

    def forward_and_sample(noiser_params, params, input_token, input_state, generation_key, iterinfo):
        gen_key, sample_key = jax.random.split(generation_key)
        logits, generated_state = model.forward(
            noiser,
            frozen_noiser_params,
            noiser_params,
            config,
            params,
            base_evo_keys,
            iterinfo,
            input_token,
            input_state,
        )
        if float(temperature) != 0.0:
            sampled_token = jax.random.categorical(sample_key, logits[-1] / float(temperature))
        else:
            sampled_token = jnp.argmax(logits[-1])
        return sampled_token, generated_state, gen_key

    def generate_thread(noiser_params, params, prompt, thread_idx, epoch):
        gen_key = _fold_in_thread_key(jax, master_gen_key, epoch, thread_idx)
        iterinfo = (epoch, thread_idx)

        def step(carry, input_token):
            token, state, key = carry
            true_input = jnp.where(input_token == 0, token, input_token)
            next_token, next_state, next_key = forward_and_sample(noiser_params, params, true_input, state, key, iterinfo)
            return (next_token, next_state, next_key), true_input

        init_token = jnp.asarray(0, dtype=jnp.int32)
        init_state = model.default_state(params, config)
        _, output_tokens = jax.lax.scan(step, (init_token, init_state, gen_key), prompt)
        return output_tokens

    return generate_thread


def _build_validate(
    stack: _HyperscaleESStack,
    model,
    config,
    params_example,
    base_evo_keys,
    master_gen_key,
    tokenizer,
    legacy_tokenizer,
    args,
    *,
    temperature: float = 0.0,
    use_validation_set: bool = True,
    noiser_cls,
    sigma: float = 0.0,
):
    jax = stack.jax
    jnp = stack.jnp
    frozen_noiser_params, noiser_params = noiser_cls.init_noiser(params_example, float(sigma), 0.0)
    task_registry = stack.validation_tasks if use_validation_set else stack.all_tasks
    task = task_registry[args.task](tokenizer, legacy_tokenizer, int(args.generation_length))
    generate_thread = _build_generate_thread(
        stack,
        model,
        noiser_cls,
        frozen_noiser_params,
        config,
        base_evo_keys,
        master_gen_key,
        float(temperature),
    )

    parallel_validations = int(args.parallel_validations)
    generation_length = int(args.generation_length)
    _log("compiling validation generator")
    t0 = time.perf_counter()
    generate_batch = (
        jax.jit(jax.vmap(generate_thread, in_axes=(None, None, 0, 0, None)))
        .lower(
            noiser_params,
            params_example,
            jax.ShapeDtypeStruct((parallel_validations, generation_length), jnp.dtype("int32")),
            jnp.arange(parallel_validations),
            0,
        )
        .compile()
    )
    _log(f"validation generator ready dt={time.perf_counter() - t0:.2f}s")

    def validate(params, epoch: int):
        total_score = 0.0
        cpu = jax.local_devices(backend="cpu")[0]
        for i in range(int(args.validation_iterations)):
            indices = jnp.arange(parallel_validations) + (i * parallel_validations)
            prompts = task.get_input(indices)
            output_batch = jax.block_until_ready(generate_batch(noiser_params, params, prompts, indices, int(epoch)))
            scores = task.get_batch_fitness(jax.device_put(indices, cpu), jax.device_put(output_batch, cpu))
            total_score += jnp.sum(jnp.asarray(scores))
        return total_score / (parallel_validations * int(args.validation_iterations))

    return validate


class _SubspaceParamCodec:
    """Map a small UHD vector into deterministic sparse deltas on a real param tree."""

    def __init__(
        self,
        jax,
        jnp,
        params,
        *,
        es_map=None,
        dim: int,
        delta_scale: float,
        seed: int,
        lora_only: bool,
        basis_max_leaves: int | None,
    ) -> None:
        self._jax = jax
        self._jnp = jnp
        self._params = params
        self.dim = int(dim)
        self.delta_scale = float(delta_scale)
        leaves, treedef = jax.tree_util.tree_flatten(params)
        self._leaves = tuple(leaves)
        self._treedef = treedef
        sizes = np.asarray([int(leaf.size) for leaf in leaves], dtype=np.int64)
        eligible = np.ones(len(leaves), dtype=bool)
        if lora_only:
            map_leaves, map_treedef = jax.tree_util.tree_flatten(es_map)
            if map_treedef != treedef:
                raise ValueError("HyperscaleES es_map tree does not match params tree; cannot build LoRA-only UHD subspace.")
            eligible = np.asarray([_is_lora_leaf(map_leaf) for map_leaf in map_leaves], dtype=bool)

        valid = np.flatnonzero((sizes > 0) & eligible)
        if valid.size == 0:
            raise ValueError("HyperscaleES params tree has no eligible trainable leaves for the UHD pretraining subspace.")
        if basis_max_leaves is not None and int(basis_max_leaves) < valid.size:
            rng_for_leaves = np.random.default_rng(int(seed) ^ 0x5F3759DF)
            leaf_probs = sizes[valid].astype(np.float64)
            leaf_probs = leaf_probs / leaf_probs.sum()
            valid = np.sort(rng_for_leaves.choice(valid, size=int(basis_max_leaves), replace=False, p=leaf_probs).astype(np.int64))
        probs = sizes[valid].astype(np.float64)
        probs = probs / probs.sum()
        rng = np.random.default_rng(int(seed))
        self._basis_leaf = rng.choice(valid, size=self.dim, replace=True, p=probs).astype(np.int64)
        self._basis_index = np.asarray([rng.integers(sizes[leaf]) for leaf in self._basis_leaf], dtype=np.int64)
        self._basis_sign = rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=self.dim).astype(np.float32)
        self.num_total_leaves = int(len(leaves))
        self.num_candidate_leaves = int(valid.size)
        self.num_candidate_params = int(sizes[valid].sum())
        self.num_basis_leaves = int(np.unique(self._basis_leaf).size)
        self.x0 = np.zeros((self.dim,), dtype=np.float64)

    def decode(self, x: np.ndarray):
        coeffs = np.asarray(x, dtype=np.float32).reshape(-1)
        if coeffs.shape[0] != self.dim:
            raise ValueError(f"x must have shape ({self.dim},), got {coeffs.shape}.")
        active = np.flatnonzero(coeffs != 0.0)
        if active.size == 0:
            return self._params
        leaves = list(self._leaves)
        active_basis_leaf = self._basis_leaf[active]
        for leaf_idx in np.unique(active_basis_leaf):
            positions = active[np.flatnonzero(active_basis_leaf == leaf_idx)]
            leaf = leaves[int(leaf_idx)]
            flat = self._jnp.reshape(leaf, (-1,))
            idx = self._jnp.asarray(self._basis_index[positions], dtype=self._jnp.int32)
            values = self._jnp.asarray(coeffs[positions] * self._basis_sign[positions] * self.delta_scale, dtype=flat.dtype)
            leaves[int(leaf_idx)] = flat.at[idx].add(values).reshape(leaf.shape)
        return self._jax.tree_util.tree_unflatten(self._treedef, leaves)


def _is_lora_leaf(map_leaf: Any) -> bool:
    arr = np.asarray(map_leaf)
    return bool(np.any(arr == 1))


def _sample_vector_noise(*, dim: int, seed: int, num_dim_target: float | None = None) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    dim = int(dim)
    if num_dim_target is None:
        return rng.standard_normal(dim).astype(np.float64)
    target = float(num_dim_target)
    if target <= 0:
        raise ValueError("perturb target must be > 0.")
    if 0 < target < 1:
        mask = rng.random(dim) < target
        if not np.any(mask):
            mask[int(rng.integers(dim))] = True
        noise = rng.standard_normal(dim).astype(np.float64)
        noise[~mask] = 0.0
        return noise
    k = min(max(int(target), 1), dim)
    idx = rng.choice(dim, size=k, replace=False)
    noise = np.zeros(dim, dtype=np.float64)
    noise[idx] = rng.standard_normal(k)
    return noise


class NanoEggPretrainVectorObjective:
    """Real NanoEgg objective adapter for UHD optimizers.

    The actual model/data runtime lives in nano-egg. This wrapper only defines
    the Yubo-side UHD contract and refuses to use the old surrogate adapter.
    """

    def __init__(self, cfg: UHDConfig) -> None:
        self.cfg = cfg
        self.spec = resolve_nanoegg_pretrain_spec(cfg.env_tag)
        self._objective = _build_external_nanoegg_objective(cfg, self.spec)
        self._embed_indices: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return int(getattr(self._objective, "dim"))

    @property
    def x0(self) -> np.ndarray:
        return np.asarray(getattr(self._objective, "x0"), dtype=np.float64)

    @property
    def steps_per_episode(self) -> int:
        return int(getattr(self._objective, "steps_per_episode", 1))

    @property
    def eval_episodes(self) -> int:
        return int(getattr(self._objective, "eval_episodes", 1))

    def make_policy(self, x: np.ndarray):
        make_policy = getattr(self._objective, "make_policy", None)
        if callable(make_policy):
            return make_policy(x)
        return SimpleNamespace(_nanoegg_pretrain_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        result = self._objective.evaluate(np.asarray(x, dtype=np.float64), seed=int(seed))
        if isinstance(result, tuple) and len(result) == 2:
            return float(result[0]), float(result[1])
        return float(result), 0.0

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        evaluate_many = getattr(self._objective, "evaluate_many", None)
        if callable(evaluate_many):
            mus, ses = evaluate_many(np.asarray(x_batch, dtype=np.float64), seed=int(seed))
            return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)
        mus, ses = [], []
        for i, x in enumerate(np.asarray(x_batch, dtype=np.float64)):
            mu, se = self.evaluate(x, seed=int(seed) + i)
            mus.append(mu)
            ses.append(se)
        return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)

    def configure_embedding(self, num_probes: int) -> None:
        configure_embedding = getattr(self._objective, "configure_embedding", None)
        if callable(configure_embedding):
            configure_embedding(int(num_probes))
            return
        rng = np.random.default_rng(123)
        n = min(max(int(num_probes), 1), self.dim)
        self._embed_indices = rng.choice(self.dim, size=n, replace=False)

    def embed_many(self, x_batch: np.ndarray) -> np.ndarray:
        embed_many = getattr(self._objective, "embed_many", None)
        if callable(embed_many):
            return np.asarray(embed_many(np.asarray(x_batch, dtype=np.float64)), dtype=np.float64)
        if self._embed_indices is None:
            self.configure_embedding(64)
        assert self._embed_indices is not None
        return np.asarray(x_batch, dtype=np.float64)[:, self._embed_indices]

    def embed(self, x: np.ndarray) -> np.ndarray:
        embed = getattr(self._objective, "embed", None)
        if callable(embed):
            return np.asarray(embed(np.asarray(x, dtype=np.float64)), dtype=np.float64)
        return self.embed_many(np.asarray([x], dtype=np.float64))[0]

    def sample_noise(
        self,
        *,
        seed: int,
        num_dim_target: float | None = None,
        num_module_target: float | None = None,
    ) -> np.ndarray:
        sample_noise = getattr(self._objective, "sample_noise", None)
        if callable(sample_noise):
            return np.asarray(
                sample_noise(seed=int(seed), num_dim_target=num_dim_target, num_module_target=num_module_target),
                dtype=np.float64,
            )
        return _sample_vector_noise(
            dim=self.dim,
            seed=int(seed),
            num_dim_target=num_module_target if num_module_target is not None else num_dim_target,
        )

    def close(self) -> None:
        close = getattr(self._objective, "close", None)
        if callable(close):
            close()


class HyperscaleESLLMVectorObjective:
    """Real HyperscaleES LLM generation/scoring objective for UHD optimizers.

    UHD optimizes a low-dimensional coefficient vector. The coefficients are
    decoded into sparse deltas on the real RWKV parameter tree before calling
    upstream HyperscaleES validation generation and task scoring.
    """

    def __init__(self, cfg: UHDConfig) -> None:
        self.cfg = cfg
        self.spec = resolve_hyperscalees_pretrain_spec(cfg.env_tag)
        stack = _require_stack()
        self.jax = stack.jax
        self.jnp = stack.jnp
        self._eval_count = 0

        if cfg.pretrain_rwkv_type is not None:
            self.spec = HyperscaleESPretrainSpec(
                env_tag=self.spec.env_tag,
                task=self.spec.task,
                model_choice=self.spec.model_choice,
                thinking_length=self.spec.thinking_length,
                answer_length=self.spec.answer_length,
                rwkv_type=str(cfg.pretrain_rwkv_type),
                dtype=self.spec.dtype,
            )
        generation_length = self.spec.generation_length
        if cfg.pretrain_generation_length is not None:
            generation_length = int(cfg.pretrain_generation_length)

        _log(f"init start env_tag={cfg.env_tag} model={self.spec.model_choice} rwkv_type={self.spec.rwkv_type} generation_length={generation_length}")
        t0 = time.perf_counter()
        rwkv, full_params, tokenizer = _load_hyperscalees_model(stack.get_model, self.spec)
        _log(f"model loaded dt={time.perf_counter() - t0:.2f}s")
        config, params, scan_map, es_map = full_params
        seed = (0 if cfg.problem_seed is None else int(cfg.problem_seed)) + int(cfg.seed_offset)
        key = stack.jax.random.key(seed & 0xFFFFFFFF)
        _log("building ES key tree")
        t0 = time.perf_counter()
        base_evo_keys = stack.simple_es_tree_key(params, stack.jax.random.fold_in(key, 1), scan_map)
        _log(f"ES key tree ready dt={time.perf_counter() - t0:.2f}s")
        legacy_tokenizer = stack.legacy_tokenizer_cls() if self.spec.model_choice.startswith("7") else tokenizer

        t0 = time.perf_counter()
        self._codec = _SubspaceParamCodec(
            stack.jax,
            stack.jnp,
            params,
            es_map=es_map,
            dim=int(cfg.pretrain_search_dim),
            delta_scale=float(cfg.pretrain_delta_scale),
            seed=seed,
            lora_only=bool(cfg.pretrain_lora_only),
            basis_max_leaves=cfg.pretrain_basis_max_leaves,
        )
        _log(
            "subspace ready "
            f"dt={time.perf_counter() - t0:.2f}s dim={self._codec.dim} "
            f"candidate_leaves={self._codec.num_candidate_leaves}/{self._codec.num_total_leaves} "
            f"basis_leaves={self._codec.num_basis_leaves} "
            f"candidate_params={self._codec.num_candidate_params}"
        )
        args = SimpleNamespace(
            task=self.spec.task,
            generation_length=generation_length,
            parallel_validations=max(1, int(cfg.eval_episodes)),
            validation_iterations=1,
        )
        self._num_validation_samples = int(args.parallel_validations) * int(args.validation_iterations)
        _log(f"building validation task={args.task} parallel_validations={args.parallel_validations} validation_iterations={args.validation_iterations}")
        t0 = time.perf_counter()
        self._validate = _build_validate(
            stack,
            rwkv,
            config,
            params,
            base_evo_keys,
            stack.jax.random.fold_in(key, 2),
            tokenizer,
            legacy_tokenizer,
            args,
            temperature=0.0,
            use_validation_set=True,
            noiser_cls=stack.noiser_cls,
            sigma=0.0,
        )
        _log(f"validation ready dt={time.perf_counter() - t0:.2f}s")
        self._embed_indices: np.ndarray | None = None

    @property
    def dim(self) -> int:
        return self._codec.dim

    @property
    def x0(self) -> np.ndarray:
        return self._codec.x0

    @property
    def steps_per_episode(self) -> int:
        return 1

    @property
    def eval_episodes(self) -> int:
        return self._num_validation_samples

    def make_policy(self, x: np.ndarray):
        return SimpleNamespace(_hyperscalees_pretrain_x=np.asarray(x, dtype=np.float64).copy())

    def evaluate(self, x: np.ndarray, *, seed: int) -> tuple[float, float]:
        x_arr = np.asarray(x, dtype=np.float64)
        trace_eval = self._eval_count < 5
        if trace_eval:
            _log(f"eval start eval={self._eval_count} seed={int(seed)} active_coeffs={int(np.count_nonzero(x_arr))}")
        t0 = time.perf_counter()
        params = self._codec.decode(x_arr)
        score = float(self._validate(params, int(seed)))
        if trace_eval:
            _log(f"eval done eval={self._eval_count} dt={time.perf_counter() - t0:.2f}s score={score:.6f}")
        self._eval_count += 1
        if 0.0 <= score <= 1.0 and self._num_validation_samples > 0:
            se = float((score * (1.0 - score) / self._num_validation_samples) ** 0.5)
        else:
            se = 0.0
        return score, se

    def evaluate_many(self, x_batch: np.ndarray, *, seed: int) -> tuple[np.ndarray, np.ndarray]:
        mus, ses = [], []
        for i, x in enumerate(np.asarray(x_batch, dtype=np.float64)):
            mu, se = self.evaluate(x, seed=int(seed) + i)
            mus.append(mu)
            ses.append(se)
        return np.asarray(mus, dtype=np.float64), np.asarray(ses, dtype=np.float64)

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
        if num_module_target is not None:
            num_dim_target = num_module_target
        rng = np.random.default_rng(int(seed))
        if num_dim_target is None:
            return rng.standard_normal(self.dim).astype(np.float64)
        target = float(num_dim_target)
        if target <= 0:
            raise ValueError("perturb target must be > 0.")
        if 0 < target < 1:
            mask = rng.random(self.dim) < target
            if not np.any(mask):
                mask[int(rng.integers(self.dim))] = True
            noise = rng.standard_normal(self.dim).astype(np.float64)
            noise[~mask] = 0.0
            return noise
        k = min(max(int(target), 1), self.dim)
        idx = rng.choice(self.dim, size=k, replace=False)
        noise = np.zeros(self.dim, dtype=np.float64)
        noise[idx] = rng.standard_normal(k)
        return noise
