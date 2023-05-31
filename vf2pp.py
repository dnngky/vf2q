from collections import defaultdict
from typing import List, Generator, MutableMapping, Optional, Self, Sequence, Tuple

from graphmap import GraphMap
from coveragetable import CoverageTable

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Layout

import rustworkx as rx
import time

class VF2PP:
    
    _G1: rx.PyGraph
    _G2: rx.PyGraph
    _qumap: Layout
    _embeddable: Optional[bool]

    _gmaps: List[GraphMap]
    _node_order: List[int]
    _covg1: CoverageTable
    _covg2: CoverageTable
    _state: int
    _call_limit: int
    _nmap_limit: int

    def __init__(
        self,
        circuit: QuantumCircuit,
        archgraph: rx.PyGraph
    ) -> Self:
        """
        VF2++ initialiser.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Circuit is not a QuantumCircuit")
        if not isinstance(archgraph, rx.PyGraph):
            raise TypeError("Archgraph is not a PyGraph")
        
        self._G1, self._qumap = VF2PP._graphify(circuit)
        self._G2 = archgraph
        self._embeddable = None
        self._gmaps = list()
        self._node_order = list()
        self._covg1 = CoverageTable()
        self._covg2 = CoverageTable()
        self._state = 0
        self._call_limit = -1
        self._nmap_limit = -1

        if len(self._G1) > len(self._G2):
            raise ValueError("Circuit width exceeds archgraph width")

    @staticmethod
    def _graphify(circuit: QuantumCircuit) -> Tuple[rx.PyGraph, Layout]:
        """
        Converts the given quantum circuit to a graph G = (V, E), where V comprises the qubits in the
        circuit, and for every (q0, q1) in E, there exists some CNOT gate operating on qubits q0 and q1.
        :return: The graph corresponding to the given quantum circuit, and its qubit-to-index mapping.
        """
        circgraph = rx.PyGraph()
        qumap = Layout()

        for qubit in circuit.qubits:
            index = circgraph.add_node(qubit)
            qumap[qubit] = index

        for gate in circuit.data:
            if gate.operation.name != "cx":
                continue
            q0, q1 = gate.qubits
            if circgraph.has_edge(qumap[q0], qumap[q1]):
                continue
            circgraph.add_edge(qumap[q0], qumap[q1], (qumap[q0], qumap[q1]))
        
        return circgraph, qumap
    
    def _reset(self) -> None:
        """
        Clears all dynamic attributes and data structures.
        """
        self._gmaps.clear()
        self._node_order.clear()
        self._covg1.clear()
        self._covg2.clear()
        self._state = 0
        self._call_limit = -1
        self._nmap_limit = -1
    
    @property
    def circgraph(self) -> rx.PyGraph:
        return self._G1
    
    @property
    def archgraph(self) -> rx.PyGraph:
        return self._G2

    @property
    def all_maps(self) -> List[GraphMap]:
        return self._gmaps
    
    def mappings(self) -> Generator[GraphMap, None, None]:
        for map in self._gmaps:
            yield map

    def is_embeddable(self) -> bool:
        """
        Returns True if G1 is embeddable onto, i.e., subgraph isomorphic to, G2.
        :return: True if G1 is embeddable onto G2.
        """
        if not self._embeddable:
            self.run(self.matching_order(), call_limit=1)
        return self._embeddable
    
    def matching_order(self) -> List[int]:
        """
        Computes the matching order for G1, based on VF2++.
        :return: The matching order for G1.
        """
        node_order = list()
        visited = defaultdict(bool)
        max_deg = max([self._G1.degree(u) for u in self._G1.node_indices()])
        
        # VF2++ Algorithm 2
        while len(node_order) < len(self._G1.node_indices()):
            node_list = set(self._G1.node_indices()).difference(set(node_order))
            root = max(node_list, key=lambda u: self._G1.degree(u))
            self._process_level(root, node_order, visited, max_deg)
        
        return node_order
    
    def _process_level(
        self,
        root: int,
        node_order: Sequence[int],
        visited: MutableMapping[int, bool],
        max_deg: int
    ) -> None:
        """
        Performs BFS on the given root and process each level of the BFS tree, based on VF2++.
        """
        curr_layer = [root]
        visited[root] = True

        # Breadth-First Search
        while curr_layer:
            next_layer = list()
            
            for u in curr_layer:
                for v in self._G1.neighbors(u):
                    if visited[v]:
                        continue
                    next_layer.append(v)
                    visited[v] = True
            
            # VF2++ Algorithm 3
            while curr_layer:
                priority = lambda u, d=max_deg, n=node_order: (self._G1.degree(u) / d + self._conn(u, n))
                node = max(curr_layer, key=priority)
                curr_layer.remove(node)
                node_order.append(node)

            curr_layer = next_layer

    def _conn(self, node: int, node_order: Sequence[int]) -> int:
        """
        Computes the number of neighbours of the given node that are also in the matching order.
        :return: The number of neighbours of the given node that are also in the matching order.
        """
        return len(set(self._G1.neighbors(node)).intersection(set(node_order)))

    def run(
        self,
        node_order: Optional[Sequence[int]] = None,
        call_limit: int = -1,
        nmap_limit: int = -1
    ) -> int:
        """
        Performs VF2++ on G1 and G2 according to the matching order. If no matching order is
        provided, `matching_order()` is implictly called. `call_limit` specifies the maximum
        number of states to visit before halting, and `nmap_limit` specifies the maximum
        number of complete mappings to search for. If set to -1, no limit is imposed.
        :return: The number of complete mappings found.
        """
        self._reset() # Clean everything up

        if call_limit <= 0 and call_limit != -1:
            raise ValueError("call_limit must be -1 or a nonzero integer")
        if nmap_limit <= 0 and nmap_limit != -1:
            raise ValueError("nmap_limit must be -1 or a nonzero integer")
        
        self._node_order = (node_order if node_order else self.matching_order())
        self._call_limit = call_limit
        self._nmap_limit = nmap_limit
        num_maps = self._match() # Run VF2++
        self._embeddable = True if num_maps else False # Record embeddable status

        return num_maps

    def _match(self, gmap: GraphMap = GraphMap(), depth: int = 0) -> int:
        
        self._state += 1

        if depth == len(self._G1):
            self._gmaps.append(gmap.copy())
            return 1
        if self._state == self._call_limit:
            return 0
        
        num_maps = 0 # Number of complete mappings found

        for cand1, cand2 in self._candidates(gmap, depth):
            
            # Filter out infeasible candidates: O(deg(cand1) * deg(cand2))
            if self._cons(gmap, cand1, cand2) and not self._cut(gmap, cand1, cand2):

                # Obtain uncovered neighbours of cand1 and cand2: O(deg(cand1) + deg(cand2))
                uncovered_neighbors1 = [v for v in self._G1.neighbors(cand1) if gmap[self._qumap[v]] is None]
                uncovered_neighbors2 = [v for v in self._G2.neighbors(cand2) if gmap[v] is None]

                # Extend mapping and coverages: O(deg(cand1) + deg(cand2))
                gmap[self._qumap[cand1]] = cand2
                self._covg1.incr([cand1] + uncovered_neighbors1)
                self._covg2.incr([cand2] + uncovered_neighbors2)
                self._covg1.cover()
                self._covg2.cover()

                num_maps += self._match(gmap, depth + 1)
                
                # Restore mapping and coverages: O(deg(cand1) + deg(cand2))
                del gmap[self._qumap[cand1]]
                self._covg1.decr([cand1] + uncovered_neighbors1)
                self._covg2.decr([cand2] + uncovered_neighbors2)
                self._covg1.uncover()
                self._covg2.uncover()

                # If limits are set to -1, these will never be true
                if len(self._gmaps) >= (self._nmap_limit if self._nmap_limit != -1 else len(self._gmaps) + 1):
                    return num_maps
                if self._state >= (self._call_limit if self._call_limit != -1 else self._state + 1):
                    return num_maps
        
        return num_maps
    
    def _candidates(self, gmap: GraphMap, depth: int) -> Generator[Tuple[int], None, None]:
        """
        Returns an iterator through each candidate pair.
        :return: An iterator through each candidate pair.
        """
        for node2 in range(len(self._G2)):
            
            # If either covg1 or covg2 is empty, then every uncovered node is a candidate
            if not (self._covg1 and self._covg2 or self._covg2[node2]):
                yield (self._node_order[depth], node2)
                continue
            
            if self._covg2[node2] and gmap[node2] is None:
                yield (self._node_order[depth], node2)
    
    def _cons(self, gmap: GraphMap, cand1: int, cand2: int) -> bool:
        """
        Returns True if for all covered neighbours of `cand1`, the nodes they map to are precisely
        the covered neighbours of `cand2`.
        :return: True if the above holds.
        """
        for neighbor in map(lambda n: self._qumap[n], self._G1.neighbors(cand1)):
            if gmap[neighbor] is None: # If this neighbor is not yet mapped, we skip it
                continue
            if not self._G2.has_edge(cand2, gmap[neighbor]):
                return False
        
        return True
    
    def _cut(self, gmap: GraphMap, cand1: int, cand2: int) -> bool:
        """
        Returns True if:
        1. `cand2` has fewer neighbors which are also in the candidate set than `cand1`.
        2. `cand2` has fewer neighbors which are neither mapped nor in the candidate set than `cand1`.

        :return: True if the above holds.
        """
        inner_neighbors1 = [v for v in self._G1.neighbors(cand1) if gmap[self._qumap[v]] is None and self._covg1[v]]
        inner_neighbors2 = [v for v in self._G2.neighbors(cand2) if gmap[v] is None and self._covg2[v]]
        if len(inner_neighbors2) < len(inner_neighbors1):
            return True

        outer_neighbors1 = [v for v in self._G1.neighbors(cand1) if gmap[self._qumap[v]] is None]
        outer_neighbors2 = [v for v in self._G2.neighbors(cand2) if gmap[v] is None]
        if len(outer_neighbors2) < len(outer_neighbors1):
            return True

        return False

    def verify(self, gmap: GraphMap) -> bool:
        """
        Verifies a mapping.
        :return: True if the mapping is complete and consistent.
        """
        visited = defaultdict(bool)

        for node in self._G1.node_indices():
            if not (visited[node] or self._check(gmap, visited, node)):
                return False
        
        return True
    
    def _check(self, gmap: GraphMap, visited: MutableMapping[int, bool], node: int) -> bool:
        
        visited[node] = True

        for neighbor in self._G1.neighbors(node):
            if gmap[self._qumap[neighbor]] is None:
                return False
            if not self._G2.has_edge(gmap[self._qumap[node]], gmap[self._qumap[neighbor]]):
                return False
            if not (visited[neighbor] or self._check(gmap, visited, neighbor)):
                return False
        
        return True
