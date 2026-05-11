from __future__ import annotations


def missing_runtime_message(runtime: str, missing: list[str], launcher: str) -> str:
    packages = ", ".join(sorted(set(missing)))
    return (
        f"The LLM {runtime} runtime is missing {packages}. "
        "Run `bash admin/setup-hyperscalees.sh` on the CUDA machine, then run "
        f"`micromamba activate yubo-hyperscalees` before launching `{launcher}`."
    )


__all__ = ["missing_runtime_message"]
