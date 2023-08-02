from coverage import Coverage
from typing import Dict, Self, Tuple
from rustworkx import connected_components, PyGraph

class Connectivity:
    """
    A dynamic data structure for managing the connectivity of nodes. The connectivity of a node is
    defined as the number of unmapped neighbours it has. A node with zero connectivity is considered
    a sink node.
    """
    _graph: PyGraph
    _connectivity: Dict[int, int]
    _num_sinks: int

    def __init__(self, graph: PyGraph) -> Self:
        """
        Connectivity constructor.
        """
        self._graph = graph
        self.clear()

    def __repr__(self) -> str:
        return "Connectivity{\n" + "\n".join([f"deg({k}): {v}" for k, v in self._connectivity.items()]) + "\n}"
    
    @property
    def num_sinks(self) -> Tuple[int]:
        return self._num_sinks

    def disconnect_neighbors(self, node: int, covg: Coverage) -> None:
        """
        Decrements the connectivity of each neighbor of the given node.
        """
        for neighbor in self._graph.neighbors(node):
            self._connectivity[neighbor] -= 1
            if self._connectivity[neighbor] == 0 and not covg.is_mapped(neighbor):
                self._num_sinks += 1
        if self._connectivity[node] == 0:
            self._num_sinks -= 1

    def reconnect_neighbors(self, node: int, covg: Coverage) -> None:
        """
        Increments the connectivity of each neighbor of the given node.
        """
        for neighbor in self._graph.neighbors(node):
            if self._connectivity[neighbor] == 0 and not covg.is_mapped(neighbor):
                self._num_sinks -= 1
            self._connectivity[neighbor] += 1
        if self._connectivity[node] == 0:
            self._num_sinks += 1

    def num_non_sinks_after_mapping(self, node: int, covg: Coverage) -> Tuple[int, int]:
        """
        Computes the number of non-sink nodes after the given node is mapped.
        :return: The number of non-sink nodes.
        """
        self.disconnect_neighbors(node, covg)
        num_sinks = self._num_sinks
        self.reconnect_neighbors(node, covg)
        return len(self._graph) - num_sinks
    
    def clear(self) -> None:
        self._connectivity = dict([(u, self._graph.degree(u)) for u in self._graph.node_indices()])
        self._num_sinks = len([cc for cc in connected_components(self._graph) if len(cc) == 1])
