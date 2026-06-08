from __future__ import annotations

from collections import OrderedDict
from typing import Any


class _PromptBatchCache:
    def __init__(self, max_size: int = 64) -> None:
        self._max_size = max(1, int(max_size))
        self._items: OrderedDict[int, tuple[list[str], list[Any]]] = OrderedDict()

    def get_or_create(self, key: int, create) -> tuple[list[str], list[Any]]:
        key = int(key)
        if key in self._items:
            self._items.move_to_end(key)
            prompts, answers = self._items[key]
            return list(prompts), list(answers)
        prompts, answers = create()
        self._items[key] = (list(prompts), list(answers))
        if len(self._items) > self._max_size:
            self._items.popitem(last=False)
        return list(prompts), list(answers)
