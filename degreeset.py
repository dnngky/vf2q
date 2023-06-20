from coverageset import CoverageSet
from typing import Dict, Self, Tuple
import rustworkx as rx

class DegreeSet:
    """
    A dynamic data structure for managing the degrees of nodes. A node with degree zero
    is considered a sink node.
    """
    _graph: rx.PyGraph
    _degree: Dict[int, int]
    _num_sinks: int

    def __init__(self, graph: rx.PyGraph) -> Self:
        """
        DegreeSet constructor.
        """
        self._graph = graph
        self.clear()

    def __repr__(self) -> str:
        return "DegreeSet(" + ', '.join([f"deg({k}): v" for k, v in self._degree.items()]) + ")"
    
    @property
    def num_sinks(self) -> Tuple[int]:
        """
        The number of sink nodes.
        """
        return self._num_sinks

    def incr_neighbors(self, node: int, candset: CoverageSet) -> None:
        """
        Increments the degree of each neighbor of the given node.
        """
        for neighbor in self._graph.neighbors(node):
            if self._degree[neighbor] == 0 and not candset.is_mapped(neighbor):
                self._num_sinks -= 1
            self._degree[neighbor] += 1
        if self._degree[node] == 0:
            self._num_sinks += 1

    def decr_neighbors(self, node: int, candset: CoverageSet) -> None:
        """
        Decrements the degree of each neighbor of the given node.
        """
        for neighbor in self._graph.neighbors(node):
            self._degree[neighbor] -= 1
            if self._degree[neighbor] == 0 and not candset.is_mapped(neighbor):
                self._num_sinks += 1
        if self._degree[node] == 0:
            self._num_sinks -= 1

    def num_sinks_after_mapping(self, node: int, candset: CoverageSet) -> Tuple[int, int]:
        """
        Computes the number of sink nodes if the given node gets mapped.
        :return: The number of sink nodes.
        """
        self.decr_neighbors(node, candset)
        num_sinks = self._num_sinks
        self.incr_neighbors(node, candset)
        return num_sinks
    
    def clear(self) -> None:
        self._degree = dict([(u, self._graph.degree(u)) for u in self._graph.node_indices()])
        self._num_sinks = len([cc for cc in rx.connected_components(self._graph) if len(cc) == 1])
