from collections import defaultdict
from functools import singledispatchmethod
from typing import MutableMapping

class CoverageTable:
    _coverage: MutableMapping[int, int]
    _size: int

    def __init__(self):
        super().__init__()
        self._coverage = defaultdict(int)
        self._size = 0

    def __len__(self) -> int:
        return self._size
    
    def __repr__(self) -> str:
        return "Coverage(" + ", ".join([f"{k}: {v}" for k, v in self._coverage.items() if v > 0]) + ")"
    
    def __getitem__(self, node: int) -> int:
        return self._coverage[node]

    @singledispatchmethod
    def incr(self, data) -> None:
        """
        Increments the coverage of the provided node(s).
        """

    @incr.register
    def _(self, data: int) -> None:
        if self._coverage[data] == 0:
            self._size += 1
        self._coverage[data] += 1

    @incr.register
    def _(self, data: list) -> int:
        for node in data:
            self.incr(node)

    @singledispatchmethod
    def decr(self, data) -> None:
        """
        Decrements the coverage of the provided node(s).
        """

    @decr.register
    def _(self, data: int) -> None:
        self._coverage[data] -= 1
        if self._coverage[data] == 0:
            self._size -= 1
    
    @decr.register
    def _(self, data: list) -> None:
        for node in data:
            self.decr(node)

    def clear(self) -> None:
        self._coverage.clear()
        self._size = 0
