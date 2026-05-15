from __future__ import annotations

import re
from dataclasses import dataclass


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
    policy_tag: str
    dtype: str
    n_layer: int
    n_embd: int
    vocab_size: int = 256


@dataclass(frozen=True)
class NanoEggPolicySpec:
    policy_tag: str
    dtype: str
    n_layer: int
    n_embd: int
    vocab_size: int = 256


_PRETRAIN_TAG_PREFIX = "pretrain:hyperscalees:"
_NANOEGG_TAG_PREFIX = "pretrain:nanoegg:"

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


def supported_hyperscalees_pretrain_env_tags() -> tuple[str, ...]:
    return tuple(sorted(_PRETRAIN_SPECS))


def supported_nanoegg_pretrain_examples() -> tuple[tuple[str, str], ...]:
    return (
        ("pretrain:nanoegg:minipile", "nanoegg:int8:6l:256d"),
        ("pretrain:nanoegg:minipile-int8-6l256d", "nanoegg:int8:6l:256d"),
    )


def is_hyperscalees_pretrain_env(env_tag: str) -> bool:
    return str(env_tag).startswith(_PRETRAIN_TAG_PREFIX)


def is_nanoegg_pretrain_env(env_tag: str) -> bool:
    return str(env_tag).startswith(_NANOEGG_TAG_PREFIX)


def resolve_nanoegg_policy_spec(policy_tag: str | None) -> NanoEggPolicySpec:
    tag = "" if policy_tag is None else str(policy_tag)
    match = re.fullmatch(
        r"nanoegg:(?P<dtype>[a-z0-9]+):(?P<layers>[1-9][0-9]*)l:(?P<embd>[1-9][0-9]*)d",
        tag,
    )
    if match is None:
        match = re.fullmatch(
            r"nanoegg-(?P<dtype>[a-z0-9]+)-(?P<layers>[1-9][0-9]*)l-(?P<embd>[1-9][0-9]*)d",
            tag,
        )
    if match is None:
        raise ValueError(f"Unsupported NanoEgg policy_tag: {tag!r}. Use 'nanoegg:<dtype>:<layers>l:<embd>d', e.g. 'nanoegg:int8:6l:256d'.")
    return NanoEggPolicySpec(
        policy_tag=tag,
        dtype=match.group("dtype"),
        n_layer=int(match.group("layers")),
        n_embd=int(match.group("embd")),
    )


def _nanoegg_policy_signature(spec: NanoEggPolicySpec) -> str:
    return f"{spec.dtype}:{int(spec.n_layer)}:{int(spec.n_embd)}:{int(spec.vocab_size)}"


def resolve_nanoegg_pretrain_spec(env_tag: str, policy_tag: str | None = None) -> NanoEggPretrainSpec:
    tag = str(env_tag)
    if not tag.startswith(_NANOEGG_TAG_PREFIX):
        raise ValueError(f"Unsupported NanoEgg pretraining env_tag: {tag!r}. Expected prefix {_NANOEGG_TAG_PREFIX!r}.")
    suffix = tag.removeprefix(_NANOEGG_TAG_PREFIX)
    legacy_match = re.fullmatch(
        r"(?P<dataset>[a-z0-9_+-]+)-(?P<dtype>[a-z0-9]+)-(?P<layers>[1-9][0-9]*)l(?P<embd>[1-9][0-9]*)d",
        suffix,
    )
    if legacy_match is not None:
        env_policy = NanoEggPolicySpec(
            policy_tag=f"nanoegg:{legacy_match.group('dtype')}:{int(legacy_match.group('layers'))}l:{int(legacy_match.group('embd'))}d",
            dtype=legacy_match.group("dtype"),
            n_layer=int(legacy_match.group("layers")),
            n_embd=int(legacy_match.group("embd")),
        )
        if policy_tag is not None:
            explicit_policy = resolve_nanoegg_policy_spec(policy_tag)
            if _nanoegg_policy_signature(explicit_policy) != _nanoegg_policy_signature(env_policy):
                raise ValueError(
                    f"NanoEgg env_tag {tag!r} encodes model {env_policy.policy_tag!r}, "
                    f"but policy_tag is {policy_tag!r}. Use env_tag 'pretrain:nanoegg:{legacy_match.group('dataset')}' "
                    "when selecting the model via policy_tag."
                )
            env_policy = explicit_policy
        return NanoEggPretrainSpec(
            env_tag=tag,
            dataset=legacy_match.group("dataset"),
            policy_tag=env_policy.policy_tag,
            dtype=env_policy.dtype,
            n_layer=env_policy.n_layer,
            n_embd=env_policy.n_embd,
            vocab_size=env_policy.vocab_size,
        )

    if re.fullmatch(r"[a-z0-9_+-]+", suffix) is None:
        raise ValueError(
            f"Unsupported NanoEgg pretraining env_tag: {tag!r}. "
            "Use 'pretrain:nanoegg:<dataset>' with policy_tag 'nanoegg:<dtype>:<layers>l:<embd>d', "
            "e.g. env_tag='pretrain:nanoegg:minipile' and policy_tag='nanoegg:int8:6l:256d'."
        )
    policy = resolve_nanoegg_policy_spec(policy_tag)
    return NanoEggPretrainSpec(
        env_tag=tag,
        dataset=suffix,
        policy_tag=policy.policy_tag,
        dtype=policy.dtype,
        n_layer=policy.n_layer,
        n_embd=policy.n_embd,
        vocab_size=policy.vocab_size,
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
