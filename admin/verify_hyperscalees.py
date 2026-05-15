import importlib
import importlib.metadata
import importlib.util
import sys


def log(msg):
    print(f"[verify-hyperscalees] {msg}")


def die(msg):
    print(f"[verify-hyperscalees] error: {msg}")
    sys.exit(1)


def check_imports():
    log("checking ENN/FAISS NumPy ABI compatibility")
    import enn
    import faiss
    import h5py
    import numba
    import numpy

    assert int(numpy.__version__.split(".", 1)[0]) >= 2, f"Expected NumPy >= 2, got {numpy.__version__}"
    assert numpy.__version__.startswith("2.2"), f"Expected NumPy 2.2.x, got {numpy.__version__}"
    log(f"imports OK: numpy={numpy.__version__} faiss={faiss.__version__} enn={enn.__name__} h5py={h5py.__version__} numba={numba.__version__}")

    log("checking HyperscaleES package metadata")
    try:
        version = importlib.metadata.version("hyperscalees")
        log(f"hyperscalees package OK: {version}")
    except importlib.metadata.PackageNotFoundError:
        die("hyperscalees package not found")

    log("checking HyperscaleES LLM/pretrain imports")
    from hyperscalees.environments.llm_bandits import all_tasks, validation_tasks
    from hyperscalees.models.llm.tokenizer import LegacyWorldTokenizer

    tok = LegacyWorldTokenizer()
    assert len(tok.encode("1 + 1")) > 0
    assert "gsm8k" in all_tasks
    assert "gsm8k" in validation_tasks
    log(f"HyperscaleES LLM imports OK: tasks={len(all_tasks)} tokenizer={type(tok).__name__}")

    log("checking owned vLLM LoRA runtime imports")
    import jax
    import ray
    import vllm

    antlr_version = importlib.metadata.version("antlr4-python3-runtime")
    assert antlr_version.startswith("4.9."), f"Expected antlr 4.9.x, got {antlr_version}"

    log(f"vllm import OK: {vllm.__version__}")
    log(f"ray import OK: {ray.__version__}")
    log(f"jax import OK after vLLM install: {jax.__version__}")

    log("checking Prime Intellect verifiers import")
    log(f"verifiers import OK: {importlib.metadata.version('verifiers')}")

    log("checking BO/RL optional extras")

    from rl.pufferlib_compat import import_pufferlib_modules

    import_pufferlib_modules()

    mods = ["smac", "LassoBench", "pyvecch", "warp", "mujoco_warp"]
    for name in mods:
        mod = importlib.import_module(name)
        log(f"import OK: {name} -> {getattr(mod, '__file__', 'unknown')}")


def check_isaaclab():
    log("checking Isaac Lab / Newton imports")
    import torch

    isaacsim_spec = importlib.util.find_spec("isaacsim")
    if isaacsim_spec is None:
        log("warning: isaacsim package is not installed")
        return

    log(f"isaacsim package OK: {isaacsim_spec.origin}")
    mods = ["isaaclab", "isaaclab_tasks", "isaaclab_newton"]
    for name in mods:
        mod = importlib.import_module(name)
        log(f"import OK: {name} -> {getattr(mod, '__file__', 'unknown')}")
    log(f"torch import OK for Isaac stack: {torch.__version__}")


if __name__ == "__main__":
    try:
        check_imports()
        check_isaaclab()
        log("All verifications passed!")
    except Exception as e:
        die(f"Verification failed: {e}")
