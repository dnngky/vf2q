from typing import Dict, Iterable, Self
import rustworkx as rx

class Coverage:
    """
    A dynamic data structure for managing the coverage of a graph. The coverage of a
    node is defined to be:
    - > 0 if the node is currently an unmapped neighbor;
    - < 0 if the node is currently mapped; and
    - = 0 if the node is neither mapped nor an unmapped neighbor.
    """
    _graph: rx.PyGraph
    _coverage: Dict[int, int]
    _num_unmapped_neighbors: int

    def __init__(self, graph: rx.PyGraph) -> Self:
        """
        Coverage constructor.
        """
        self._graph = graph
        self.clear()
    
    def __repr__(self) -> str:
        return "Coverage{\n" + "\n".join([f"cov({k}): {v}" for k, v in self._coverage.items()]) + "\n}"
    
    @property
    def num_unmapped_neighbors(self) -> int:
        """
        The number of unmapped neighbors.
        """
        return self._num_unmapped_neighbors
    
    def map(self, node: int) -> None:
        """
        Sets a unmapped node as mapped. The node must be a currently unmapped candidate.
        """
        self._coverage[node] *= -1
        self._num_unmapped_neighbors -= 1
    
    def unmap(self, node: int) -> None:
        """
        Sets a mapped node as unmapped. The node must be a currently mapped candidate.
        """
        self._coverage[node] *= -1
        self._num_unmapped_neighbors += 1
    
    def cover(self, nodes: Iterable[int]) -> None:
        """
        Increments the coverage of the provided nodes.
        """
        for node in nodes:
            if not self._coverage[node]:
                self._num_unmapped_neighbors += 1
            self._coverage[node] = max(self._coverage[node] + 1, self._coverage[node] - 1)
    
    def uncover(self, nodes: Iterable[int]) -> None:
        """
        Decrements the coverage of the provided nodes.
        """
        for node in nodes:
            self._coverage[node] = min(self._coverage[node] + 1, self._coverage[node] - 1)
            if not self._coverage[node]:
                self._num_unmapped_neighbors -= 1

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
        self._coverage = dict.fromkeys(self._graph.node_indices(), 0)
        self._num_unmapped_neighbors = 0
