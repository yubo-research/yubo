"""Normalize (exp_dir, env_tag[, regex, param]) panel specs and resolve paths."""

from pathlib import Path


def _normalize_single_panel_source(
    source: tuple[str, str | None] | tuple[str, str | None, str, str],
    *,
    regex_pattern: str,
    param_name_for_print: str,
) -> tuple[str, str | None, str, str]:
    if len(source) == 2:
        exp_dir, env_tag = source
        return exp_dir, env_tag, regex_pattern, param_name_for_print
    if len(source) == 4:
        exp_dir, env_tag, panel_regex_pattern, panel_param_name = source
        return exp_dir, env_tag, panel_regex_pattern, panel_param_name
    raise ValueError("Each panel source must be (exp_dir, env_tag) or (exp_dir, env_tag, regex_pattern, param_name_for_print)")


def _normalize_panel_sources(
    panel_sources: tuple[
        tuple[str, str | None] | tuple[str, str | None, str, str],
        ...,
    ],
    *,
    regex_pattern: str,
    param_name_for_print: str,
) -> list[tuple[str, str | None, str, str]]:
    return [
        _normalize_single_panel_source(
            source,
            regex_pattern=regex_pattern,
            param_name_for_print=param_name_for_print,
        )
        for source in panel_sources
    ]


def _resolve_panel_exp_paths(
    results_dir: Path,
    normalized_sources: list[tuple[str, str | None, str, str]],
) -> list[Path] | None:
    exp_paths: list[Path] = []
    for exp_dir, _, _, _ in normalized_sources:
        exp_path = results_dir / exp_dir
        if not exp_path.exists():
            print(f"Directory not found: {exp_path}")
            print(f"Run prep for {exp_dir} first.")
            return None
        exp_paths.append(exp_path)
    return exp_paths
