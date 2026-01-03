from typing import Any, Iterable, List, Tuple

import torch
import torch.nn as nn


class SingleTensorAccessor:
    def __init__(self, tensor: torch.Tensor) -> None:
        assert isinstance(tensor, torch.Tensor)
        self._tensor = tensor
        self.device = tensor.device
        self.dtype = tensor.dtype
        self.lb = 0.0
        self.ub = 1.0
        self.num_dims = int(tensor.numel())

    def numel(self) -> int:
        return int(self._tensor.numel())

    def clone_flat(self) -> torch.Tensor:
        return self._tensor.view(-1).clone()

    def clone(self) -> torch.Tensor:
        return self.clone_flat()

    def add_inplace_(self, delta_flat: torch.Tensor) -> None:
        assert isinstance(delta_flat, torch.Tensor) and delta_flat.numel() == self.numel()
        with torch.no_grad():
            self._tensor.add_(delta_flat.view_as(self._tensor))

    def add_(self, delta_flat: torch.Tensor) -> None:
        self.add_inplace_(delta_flat)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        assert isinstance(indices, torch.Tensor) and indices.dtype == torch.long
        flat = self._tensor.view(-1)
        return flat.index_select(0, indices)

    def scatter_(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        assert isinstance(indices, torch.Tensor) and indices.dtype == torch.long
        assert isinstance(values, torch.Tensor) and values.numel() == indices.numel()
        with torch.no_grad():
            flat = self._tensor.view(-1)
            flat.index_copy_(0, indices, values)

    def mul_(self, scalar: float) -> None:
        assert isinstance(scalar, (int, float))
        with torch.no_grad():
            self._tensor.mul_(float(scalar))

    def clamp_(self, min_value: float, max_value: float) -> None:
        assert isinstance(min_value, (int, float))
        assert isinstance(max_value, (int, float))
        with torch.no_grad():
            self._tensor.clamp_(float(min_value), float(max_value))

    def copy_(self, src: torch.Tensor) -> None:
        assert isinstance(src, torch.Tensor) and src.numel() == self.numel()
        with torch.no_grad():
            self._tensor.copy_(src.view_as(self._tensor))

    def __torch_function__(self, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        if func is torch.randn_like:
            return torch.randn_like(self._tensor, **kwargs)
        if func is torch.rand_like:
            return torch.rand_like(self._tensor, **kwargs)
        raise TypeError(f"{type(self).__name__} does not support torch function {func.__name__}")

    def backup(self) -> torch.Tensor:
        return self.clone_flat()

    def restore(self, backup_flat: torch.Tensor) -> None:
        assert isinstance(backup_flat, torch.Tensor) and backup_flat.numel() == self.numel()
        with torch.no_grad():
            self._tensor.copy_(backup_flat.view_as(self._tensor))


class ModuleParamAccessor:
    def __init__(self, params: Iterable[nn.Parameter]) -> None:
        self._params: List[nn.Parameter] = [p for p in params]
        sizes = [int(p.numel()) for p in self._params]
        self._splits: List[Tuple[int, int]] = []
        start = 0
        for s in sizes:
            self._splits.append((start, start + s))
            start += s
        self._total = start
        self.num_dims = int(self._total)
        if self._total > 0:
            self.device = self._params[0].device
            self.dtype = self._params[0].dtype
        else:
            self.device = torch.device("cpu")
            self.dtype = torch.float32

    def numel(self) -> int:
        return int(self._total)

    def clone_flat(self) -> torch.Tensor:
        out = torch.empty(self._total, dtype=self._params[0].dtype, device=self._params[0].device) if self._total > 0 else torch.tensor(0.0)
        if self._total == 0:
            return out
        i = 0
        for p in self._params:
            n = p.numel()
            out[i : i + n] = p.view(-1).detach()
            i += n
        return out

    def add_inplace_(self, delta_flat: torch.Tensor) -> None:
        assert isinstance(delta_flat, torch.Tensor) and delta_flat.numel() == self._total
        with torch.no_grad():
            for (start, end), p in zip(self._splits, self._params):
                p.view(-1).add_(delta_flat[start:end])

    def add_(self, delta_flat: torch.Tensor) -> None:
        self.add_inplace_(delta_flat)

    def gather(self, indices: torch.Tensor) -> torch.Tensor:
        assert isinstance(indices, torch.Tensor) and indices.dtype == torch.long
        if indices.numel() == 0 or self._total == 0:
            return torch.empty(indices.numel(), dtype=self.dtype, device=indices.device)
        out = torch.empty(indices.numel(), dtype=self.dtype, device=self.device)
        for (start, end), p in zip(self._splits, self._params):
            mask = (indices >= start) & (indices < end)
            if mask.any():
                idx_pos = torch.nonzero(mask, as_tuple=False).view(-1)
                rel = indices.index_select(0, idx_pos) - start
                vals = p.view(-1).index_select(0, rel)
                out.index_copy_(0, idx_pos, vals)
        return out

    def scatter_(self, indices: torch.Tensor, values: torch.Tensor) -> None:
        assert isinstance(indices, torch.Tensor) and indices.dtype == torch.long
        assert isinstance(values, torch.Tensor) and values.numel() == indices.numel()
        if indices.numel() == 0 or self._total == 0:
            return
        with torch.no_grad():
            for (start, end), p in zip(self._splits, self._params):
                mask = (indices >= start) & (indices < end)
                if mask.any():
                    idx_pos = torch.nonzero(mask, as_tuple=False).view(-1)
                    rel = indices.index_select(0, idx_pos) - start
                    src = values.index_select(0, idx_pos)
                    p.view(-1).index_copy_(0, rel, src)

    def mul_(self, scalar: float) -> None:
        assert isinstance(scalar, (int, float))
        with torch.no_grad():
            for p in self._params:
                p.mul_(float(scalar))

    def clamp_(self, min_value: float, max_value: float) -> None:
        assert isinstance(min_value, (int, float))
        assert isinstance(max_value, (int, float))
        with torch.no_grad():
            for p in self._params:
                p.clamp_(float(min_value), float(max_value))

    def backup(self) -> torch.Tensor:
        return self.clone_flat()

    def restore(self, backup_flat: torch.Tensor) -> None:
        assert isinstance(backup_flat, torch.Tensor) and backup_flat.numel() == self._total
        with torch.no_grad():
            for (start, end), p in zip(self._splits, self._params):
                p.view(-1).copy_(backup_flat[start:end])


def make_param_accessor(obj: Any):
    if isinstance(obj, (torch.Tensor, nn.Parameter)):
        acc = SingleTensorAccessor(obj)  # type: ignore[arg-type]
        return acc
    if hasattr(obj, "parameters_"):
        t = getattr(obj, "parameters_")
        if isinstance(t, torch.Tensor):
            acc = SingleTensorAccessor(t)
            if hasattr(obj, "lb"):
                setattr(acc, "lb", float(getattr(obj, "lb")))
            if hasattr(obj, "ub"):
                setattr(acc, "ub", float(getattr(obj, "ub")))
            return acc
    assert isinstance(obj, nn.Module)
    acc = ModuleParamAccessor(obj.parameters())
    if hasattr(obj, "lb"):
        setattr(acc, "lb", float(getattr(obj, "lb")))
    if hasattr(obj, "ub"):
        setattr(acc, "ub", float(getattr(obj, "ub")))
    return acc
