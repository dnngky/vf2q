from collections import defaultdict
from typing import Iterable, Self

class CandidateSet:
    """
    A dynamic data structure for managing the candidates of a graph. The coverage of a node
    is defined to be:
    - = 0 if it is not mapped nor a candidate;
    - < 0 if it is currently mapped, and
    - > 0 if it is currently a candidate.
    """
    _coverage: defaultdict[int, int]
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
        return "CandidateSet{" + ", ".join([f"{k}: {v}" for k, v in self._coverage.items() if v > 0]) + "}"
    
    def map(self, node: int) -> None:
        """
        Sets a unmapped node as mapped. The node must already be covered, i.e., has a
        nonzero coverage value.
        """
        if not self._coverage[node]:
            raise ValueError("Attempting to map an uncovered node")
        if self._coverage[node] < 0:
            raise ValueError("Attempting to map an already mapped node")
        self._coverage[node] *= -1
        self._size -= 1
    
    def unmap(self, node: int) -> None:
        """
        Sets a mapped node as unmapped. The node must already be covered, i.e., has a
        nonzero coverage value.
        """
        if not self._coverage[node]:
            raise ValueError("Attempting to unmap an uncovered node")
        if self._coverage[node] > 0:
            raise ValueError("Attempting to unmap an already unmapped node")
        self._coverage[node] *= -1
        self._size += 1
    
    def cover(self, cands: Iterable[int]) -> None:
        """
        Increments the coverage value of the provided candidates.
        """
        for cand in cands:
            if not self._coverage[cand]:
                self._size += 1
            self._coverage[cand] = max(self._coverage[cand] + 1, self._coverage[cand] - 1)
    
    def uncover(self, cands: Iterable[int]) -> None:
        """
        Decrements the coverage value of the provided candidates.
        """
        for cand in cands:
            self._coverage[cand] = min(self._coverage[cand] + 1, self._coverage[cand] - 1)
            if not self._coverage[cand]:
                self._size -= 1

    def iscand(self, node: int) -> bool:
        """
        Returns True if the given node is currently a candidate.
        :return: True if the above holds.
        """
        return self._coverage[node] > 0
    
    def ismapped(self, node: int) -> bool:
        """
        Returns True if the given node is currently mapped.
        :return: True if the above holds.
        """
        return self._coverage[node] < 0

    def clear(self) -> None:
        self._coverage.clear()
        self._size = 0
