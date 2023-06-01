from collections import defaultdict
from functools import singledispatchmethod
from typing import Self

class CoverageTable:
    """
    A dynamic data structure for managing the coverage of some graph throughout
    a VF2++ matching run. The coverage level of a node is defined to be nonzero
    if and only if that node is either mapped or is a candidate to be mapped in
    the current state. The size of a CoverageTable is defined as the size of the
    candidate set, i.e., the number of nodes with nonzero coverage levels that
    are not currently mapped.
    """
    _level: defaultdict[int, int]
    _size: int

    def __init__(self) -> Self:
        """
        CoverageTable constructor.
        """
        super().__init__()
        self._level = defaultdict(int)
        self._size = 0

    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, node: int) -> int:
        return self._level[node]
    
    def __repr__(self) -> str:
        return "Coverage(" + ", ".join([f"{k}: {v}" for k, v in self._level.items() if v > 0]) + ")"
    
    def map(self, node: int) -> None:
        """
        Sets a unmapped node as mapped. The node must already be covered,
        i.e., have a nonzero coverage level.
        """
        if not self._level[node]:
            raise ValueError("Attempting to map an uncovered node")
        if self._level[node] < 0:
            raise ValueError("Attempting to map an already mapped node")
        self._level[node] *= -1
        self._size -= 1
    
    def unmap(self, node: int) -> None:
        """
        Sets a mapped node as unmapped. The node must already be covered,
        i.e., have a nonzero coverage level.
        """
        if not self._level[node]:
            raise ValueError("Attempting to unmap an uncovered node")
        if self._level[node] > 0:
            raise ValueError("Attempting to unmap an already unmapped node")
        self._level[node] *= -1
        self._size += 1

    @singledispatchmethod
    def incr(self, _) -> None:
        """
        Increments the coverage level of the provided node(s).
        """

    @incr.register
    def _(self, node: int) -> None:
        if self._level[node] == 0:
            self._size += 1
        self._level[node] = max(self._level[node] + 1, self._level[node] - 1)

    @incr.register
    def _(self, nodes: list) -> int:
        for node in nodes:
            self.incr(node)

    @singledispatchmethod
    def decr(self, _) -> None:
        """
        Decrements the coverage level of the provided node(s).
        """

    @decr.register
    def _(self, node: int) -> None:
        self._level[node] = min(self._level[node] + 1, self._level[node] - 1)
        if self._level[node] == 0:
            self._size -= 1
    
    @decr.register
    def _(self, nodes: list) -> None:
        for node in nodes:
            self.decr(node)

    def clear(self) -> None:
        self._level.clear()
        self._size = 0
