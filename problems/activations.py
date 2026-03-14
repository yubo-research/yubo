import torch.nn as nn


def _normalize(name: str) -> str:
    return "".join(ch for ch in str(name).strip().lower() if ch.isalnum())


_ACTIVATIONS = {_normalize(name): mod for name, mod in vars(nn.modules.activation).items() if isinstance(mod, type) and issubclass(mod, nn.Module)} | {
    "swish": nn.SiLU
}


def activation(name: str):
    key = _normalize(name)
    if key not in _ACTIVATIONS:
        raise ValueError(f"Unsupported activation '{name}'.")
    return _ACTIVATIONS[key]
