from collections import defaultdict
from typing import List, Generator, MutableMapping, Optional, Sequence, Tuple

from graphmap import GraphMap
from coveragetable import CoverageTable

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Layout

import itertools as it
import rustworkx as rx

class VF2PP:
    _G1: rx.PyGraph
    _G2: rx.PyGraph
    _gmaps: List[GraphMap]
    _qumap: Layout
    _covg1: CoverageTable
    _covg2: CoverageTable

    def __init__(self, circuit: QuantumCircuit, archgraph: rx.PyGraph):
        """
        VF2++ initialiser.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Circuit is not a QuantumCircuit")
        if not isinstance(archgraph, rx.PyGraph):
            raise TypeError("Archgraph is not a PyGraph")
        
        self._G1, self._qumap = VF2PP._graphify(circuit)
        self._G2 = archgraph
        
        if len(self._G1) > len(self._G2):
            raise ValueError("Circuit width exceeds archgraph width")
        
        self._gmaps = list()
        self._covg1 = CoverageTable()
        self._covg2 = CoverageTable()
    
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

    @staticmethod
    def _graphify(circuit: QuantumCircuit) -> Tuple[rx.PyGraph, Layout]:
        """
        Converts the given quantum circuit to a graph G = (V, E), where V comprises the qubits in the
        circuit, and for every (q0, q1) in E, there exists some CNOT gate operating on qubits q0 and q1.
        :return: The graph corresponding to the given quantum circuit, and its node-to-index mapping.
        """
        circgraph = rx.PyGraph()
        qubitmap = Layout()

        for qubit in circuit.qubits:
            index = circgraph.add_node(qubit)
            qubitmap[qubit] = index

        for gate in circuit.data:
            if gate.operation.name != "cx":
                continue
            q0, q1 = gate.qubits
            if circgraph.has_edge(qubitmap[q0], qubitmap[q1]):
                continue
            circgraph.add_edge(qubitmap[q0], qubitmap[q1], (qubitmap[q0], qubitmap[q1]))
        
        return circgraph, qubitmap
    
    def matching_order(self) -> List[int]:
        """
        Computes the matching order for G1, based on VF2++.
        :return: The matching order for G1.
        """
        node_order = list()
        visited = defaultdict(lambda: False)
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
        call_limit: int = 1
    ) -> int:
        """
        Performs VF2 on G1 and G2 according to the matching order, if provided. Halts
        after the specified (`call_limit`) number of whole mappings have been found.
        If `limit` is set to -1, then all complete mappings are searched (generally
        not recommended).
        :return: The number of consistent whole mappings found.
        """
        if call_limit <= 0 and call_limit != -1:
            raise ValueError("Limit must be -1 or a nonzero integer")
        
        # Clear data structures
        self._gmaps.clear()
        self._covg1.clear()
        self._covg2.clear()

        return self._match(GraphMap(), node_order, 0, call_limit)

    def _match(
        self,
        gmap: GraphMap,
        node_order: Optional[Sequence[int]] = None,
        depth: int = 0,
        call_limit: int = 1
    ) -> int:

        if depth == len(self._G1):
            self._gmaps.append(gmap.copy())
            return 1
        
        n_maps = 0 # number of complete mappings found
        
        if node_order is not None:
            cands1 = [node_order[depth]]
        else:
            cands1 = [
                u for u in self._G1.node_indices() if \
                    (self._covg1[u] and gmap[self._qumap[u]] is None) or not self._covg1
            ]
            cands1.sort(key=lambda u: self._G1.degree(u), reverse=True) # sort by degree
        
        cands2 = [
            u for u in self._G2.node_indices() if \
                (self._covg2[u] and gmap[u] is None) or not self._covg2
        ]
        for cand1, cand2 in it.product(cands1, cands2):
            
            # Filter out infeasible candidates: O(deg(cand1) * deg(cand2))
            if self._cons(gmap, cand1, cand2) and not self._cut(gmap, cand1, cand2):
                
                assert (gmap[self._qumap[cand1]] is None) and (gmap[cand2] is None), \
                    "Attempt to map to an already-mapped node!"

                # Obtain uncovered neighbours of cand1 and cand2: O(deg(cand1) + deg(cand2))
                uncovered_neighbors1 = [v for v in self._G1.neighbors(cand1) if gmap[self._qumap[v]] is None]
                uncovered_neighbors2 = [v for v in self._G2.neighbors(cand2) if gmap[v] is None]

                # Extend mapping and coverages: O(deg(cand1) + deg(cand2))
                gmap[self._qumap[cand1]] = cand2
                self._covg1.incr([cand1] + uncovered_neighbors1)
                self._covg2.incr([cand2] + uncovered_neighbors2)

                n_maps += self._match(gmap, node_order, depth + 1, call_limit)
                
                # Restore mapping and coverages: O(deg(cand1) + deg(cand2))
                del gmap[self._qumap[cand1]]
                self._covg1.decr([cand1] + uncovered_neighbors1)
                self._covg2.decr([cand2] + uncovered_neighbors2)

                # If call_limit == -1, this will never be true
                if len(self._gmaps) >= (call_limit if call_limit != -1 else len(self._gmaps) + 1):
                    return n_maps
        
        return n_maps
    
    def _cons(self, gmap: GraphMap, cand1: int, cand2: int) -> bool:
        """
        Returns True if for all covered neighbours of cand1, the corresponding mapped nodes of
        those neighbours are precisely the covered neighbours of cand2.
        :return: True if the above holds.
        """
        for neighbor in map(lambda n: self._qumap[n], self._G1.neighbors(cand1)):
            if gmap[neighbor] is None:
                continue
            if not self._G2.has_edge(cand2, gmap[neighbor]):
                return False
        
        return True
    
    def _cut(self, gmap: GraphMap, cand1: int, cand2: int) -> bool:
        """
        Returns True if cand2 has fewer neighbours within G2's coverage than those of cand1, or
        cand2 has fewer neighbours beyond G2's coverage than those of cand1.
        :return: True if the above holds.
        """
        within_neighbors1 = [v for v in self._G1.neighbors(cand1) if self._covg1[v] > 0 and gmap[self._qumap[v]] is None]
        within_neighbors2 = [v for v in self._G2.neighbors(cand2) if self._covg2[v] > 0 and gmap[v] is None]
        if len(within_neighbors2) < len(within_neighbors1):
            return True

        # NOTE: For some reason, the 2-look-ahead rule prevents the algorithm from working...
        # beyond_neighbors1 = [v for v in self._G1.neighbors(cand1) if self._covg1[v] == 0]
        # beyond_neighbors2 = [v for v in self._G2.neighbors(cand2) if self._covg2[v] == 0]
        # if len(beyond_neighbors2) < len(beyond_neighbors1):
        #     return True

        return False

    def verify(self, gmap: GraphMap) -> bool:
        """
        Verifies a mapping.
        :return: True if the mapping is complete and consistent.
        """
        visited = defaultdict(lambda: False)

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
