from collections import defaultdict
from functools import singledispatchmethod
from graphmap import GraphMap
from qiskit.transpiler import Layout
from typing import Generator, Optional

class CoverageTable:
    _level: defaultdict[int, int]
    _num_nodes: int
    _size: int

    def __init__(self, num_nodes: int):
        super().__init__()
        self._level = defaultdict(int)
        self._num_nodes = num_nodes
        self._size = 0

    def __len__(self) -> int:
        return self._size
    
    def __repr__(self) -> str:
        return "Coverage(" + ", ".join([f"{k}: {v}" for k, v in self._level.items() if v > 0]) + ")"
    
    def __getitem__(self, node: int) -> int:
        return self._level[node]
    
    def cover(self) -> None:
        """
        Sets a node as covered, i.e., added to the mapping.
        """
        self._size -= 1
    
    def uncover(self) -> None:
        """
        Unsets a covered node, i.e., removed from the mapping.
        """
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
        self._level[node] += 1

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
        self._level[node] -= 1
        if self._level[node] == 0:
            self._size -= 1
    
    @decr.register
    def _(self, nodes: list) -> None:
        for node in nodes:
            self.decr(node)
    
    def is_cand(self, node: int, gmap: GraphMap, qumap: Optional[Layout] = None) -> bool:
        """
        Returns True if the given node is a candidate.
        :return: True if the given node is a candidate.
        """
        is_uncovered = gmap[node] is None
        if qumap:
            is_uncovered = gmap[qumap[node]] is None
        return self._level[node] and is_uncovered
    
    def cands(self, gmap: GraphMap, qumap: Optional[Layout] = None) -> Generator[int, None, None]:
        """
        Returns an iterator through each candidate.
        :return: An iterator through each candidate.
        """
        for node in range(self._num_nodes):
            if not self._size and not self._level[node]:
                yield node
            if self.is_cand(node, gmap, qumap):
                yield node

    def clear(self) -> None:
        self._level.clear()
        self._size = 0
