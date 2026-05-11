from __future__ import annotations

import llm.tasks_base as _base
import llm.tasks_countdown as _countdown
import llm.tasks_factory as _factory
import llm.tasks_math as _math
import llm.tasks_static as _static


CountdownTask = _countdown.CountdownTask
LLMTask = _base.LLMTask
MathTask = _math.MathTask
MathTaskConfig = _math.MathTaskConfig
RandomTask = _static.RandomTask
ZerosTask = _static.ZerosTask
_extract_math_answer = _math._extract_math_answer
_first_nonempty_match = _base._first_nonempty_match
_grade_math_answer = _math._grade_math_answer
_grade_math_answer_with_math_verify = _math._grade_math_answer_with_math_verify
_has_explicit_math_answer = _math._has_explicit_math_answer
_last_boxed_answer = _math._last_boxed_answer
_load_countdown_dataset = _countdown._load_countdown_dataset
_load_math_dataset = _math._load_math_dataset
_normalize_answer = _math._normalize_answer
_safe_arithmetic_eval = _countdown._safe_arithmetic_eval
_synthetic_countdown_dataset = _countdown._synthetic_countdown_dataset
build_task = _factory.build_task
check_math_correct = _math.check_math_correct
countdown_answer_reward = _countdown.countdown_answer_reward
countdown_format_reward = _countdown.countdown_format_reward
extract_model_answer = _base.extract_model_answer
score_generations = _base.score_generations


__all__ = [
    "CountdownTask",
    "LLMTask",
    "MathTask",
    "MathTaskConfig",
    "RandomTask",
    "ZerosTask",
    "_extract_math_answer",
    "_first_nonempty_match",
    "_grade_math_answer",
    "_grade_math_answer_with_math_verify",
    "_has_explicit_math_answer",
    "_last_boxed_answer",
    "_load_countdown_dataset",
    "_load_math_dataset",
    "_normalize_answer",
    "_safe_arithmetic_eval",
    "_synthetic_countdown_dataset",
    "build_task",
    "check_math_correct",
    "countdown_answer_reward",
    "countdown_format_reward",
    "extract_model_answer",
    "score_generations",
]
