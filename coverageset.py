from collections import defaultdict
from typing import Generator, Iterable, Self, DefaultDict

class CoverageSet:
    """
    A dynamic data structure for managing the coverage of a graph. The coverage of a
    node is defined to be:
    - > 0 if the node is currently an unmapped neighbor;
    - < 0 if the node is currently mapped; and
    - = 0 if the node is neither mapped nor an unmapped neighbor.
    """
    _coverage: DefaultDict[int, int]
    _size: int

    def __init__(self) -> Self:
        """
        CandidateSet constructor.
        """
        super().__init__()
        self._coverage = defaultdict(int)
        self._size = 0

    def __len__(self) -> int:
        return self._size
    
    def __getitem__(self, node: int) -> int:
        return self._coverage[node]
    
    def __repr__(self) -> str:
        return f"CandidateSet{{{', '.join([str(k) for k, v in self._coverage.items() if v > 0])}}}"
    
    def map(self, node: int) -> None:
        """
        Sets a unmapped node as mapped. The node must be a currently unmapped candidate.
        """
        self._coverage[node] *= -1
        self._size -= 1
    
    def unmap(self, node: int) -> None:
        """
        Sets a mapped node as unmapped. The node must be a currently mapped candidate.
        """
        self._coverage[node] *= -1
        self._size += 1
    
    def cover(self, nodes: Iterable[int]) -> None:
        """
        Increments the coverage of the provided nodes.
        """
        for node in nodes:
            if not self._coverage[node]:
                self._size += 1
            self._coverage[node] = max(self._coverage[node] + 1, self._coverage[node] - 1)
    
    def uncover(self, nodes: Iterable[int]) -> None:
        """
        Decrements the coverage of the provided nodes.
        """
        for node in nodes:
            self._coverage[node] = min(self._coverage[node] + 1, self._coverage[node] - 1)
            if not self._coverage[node]:
                self._size -= 1

    def is_unmapped_neighbor(self, node: int) -> bool:
        """
        Returns True if the given node is currently an umapped neighbor.
        :return: True if the above holds.
        """
        return self._coverage[node] > 0
    
    def is_mapped(self, node: int) -> bool:
        """
        Returns True if the given node is currently mapped.
        :return: True if the above holds.
        """
        return self._coverage[node] < 0

    def clear(self) -> None:
        self._coverage.clear()
        self._size = 0
