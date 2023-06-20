from coverageset import CoverageSet
from collections import defaultdict
from degreeset import DegreeSet
from heapq import heapify, heappop
from typing import Dict, List, Generator, MutableMapping, Optional, Self, Sequence, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit.transpiler import Layout

import math
import rustworkx as rx
import time

class VF2Q:
    """
    Qubit mapping using subgraph isomorphism, based on VF2 and VF2++.
    """
    _G1: rx.PyGraph                 # Interaction graph of quantum circuit (to be embedded)
    _G2: rx.PyGraph                 # Architecture graph of quantum device (to embed onto)
    _num_nonsinks1: int             # Number of non-sink nodes (i.e., nodes with non-zero degree) in G1
    _centrality2: Dict[int, float]  # Betweeness centrality of nodes in G2
    _qmap: Layout                   # Qubit-to-index mapping
    _embeddable: Optional[bool]     # True if G1 is embeddable onto G2

    _gmaps: List[Layout]            # All complete mappings found
    _node_order: List[int]          # Matching order
    _cc_endpts: List[int]           # Cumulative ending indexes of connected components
    _covset1: CoverageSet           # Coverage set of G1
    _covset2: CoverageSet           # Coverage set of G2
    _degset2: DegreeSet             # Degree set of G2
    
    _state: int                     # Number of states visited during
    _call_limit: int                # Limit for number of states to visit
    _time_limit: int                # Limit the period of time (s) to spend on searching
    _nmap_limit: int                # Limit for number of complete mappings to search for
    _min_cost: int                  # Cost of current minimum-cost mapping

    def __init__(
        self,
        circuit: QuantumCircuit,
        archgraph: rx.PyGraph
    ) -> Self:
        """
        VF2++ constructor.
        """
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Circuit is not a QuantumCircuit")
        if not isinstance(archgraph, rx.PyGraph):
            raise TypeError("Archgraph is not a PyGraph")
        
        self._G1, self._qmap = VF2Q._graphify(circuit)
        self._G2 = archgraph
        self._num_nonsinks1 = len(self._G1) - len([cc for cc in rx.connected_components(self._G1) if len(cc) == 1])
        self._centrality2 = rx.betweenness_centrality(self._G2)
        self._embeddable = None

        self._gmaps = list()
        self._node_order = list()
        self._cc_endpts = list()
        self._covset1 = CoverageSet()
        self._covset2 = CoverageSet()
        self._degset2 = DegreeSet(self._G2)

        if len(self._G1) > len(self._G2):
            raise ValueError("Circuit width exceeds archgraph width")

    @staticmethod
    def _graphify(circuit: QuantumCircuit) -> Tuple[rx.PyGraph, Layout]:
        """
        Converts the given quantum circuit to a graph G = (V, E), where V comprises
        the qubits in the circuit, and for every (p, q) in E, there exists some
        CNOT gate operating on qubits p and q.
        :return: The graph corresponding to the given quantum circuit, and its
        qubit-to-index mapping.
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
        self._cc_endpts.clear()
        self._covset1.clear()
        self._covset2.clear()
        self._degset2.clear()

        self._state = 0
        self._call_limit = -1
        self._time_limit = -1
        self._nmap_limit = -1
        self._min_cost = math.inf
    
    @property
    def circgraph(self) -> rx.PyGraph:
        return self._G1
    
    @property
    def archgraph(self) -> rx.PyGraph:
        return self._G2
    
    @property
    def node_order(self) -> List[int]:
        if not self._node_order:
            return self._matching_order()[0]
        return self._node_order
    
    @property
    def cc_cum_len(self) -> List[int]:
        if not self._cc_endpts:
            return self._matching_order()[1]
        return self._cc_endpts

    @property
    def all_maps(self) -> List[Layout]:
        return self._gmaps
    
    def mappings(self) -> Generator[Layout, None, None]:
        for map in self._gmaps:
            yield map

    def is_embeddable(self) -> bool:
        """
        Returns True if G1 is embeddable onto, i.e., subgraph isomorphic to, G2.
        :return: True if G1 is embeddable onto G2.
        """
        if self._embeddable is None:
            self.match_all(self.matching_priority(), self.matching_order(), nmap_limit=1)
        return self._embeddable
    
    def _matching_order(self) -> Tuple[List[int], List[int]]:
        """
        Computes the matching order for G1, based on VF2++.
        :return: The matching order for G1.
        """
        node_orders = list()
        nodes_left = list(self._G1.node_indices())
        visited = defaultdict(bool)
        max_deg = max([self._G1.degree(u) for u in self._G1.node_indices()])
        
        # VF2++ Algorithm 2
        while nodes_left:
            root = min(nodes_left, key=lambda u: self._G1.degree(u))
            node_order = self._process_level(root, visited, nodes_left, max_deg)
            node_orders.append(node_order)
        
        merged_node_orders = list()
        cc_cum_len = list()
        for node_order in sorted(node_orders, key=lambda n: len(n), reverse=True):
            merged_node_orders.extend(node_order)
            cc_cum_len.append(len(merged_node_orders))
        cc_cum_len = list(map(lambda x: x - 1, cc_cum_len))
        
        return merged_node_orders, cc_cum_len
    
    def _process_level(
        self,
        root: int,
        visited: MutableMapping[int, bool],
        nodes_left: Sequence[int],
        max_deg: int
    ) -> List[int]:
        """
        Performs BFS on the given root and process each level of the BFS tree, based on VF2++.
        :return: The matching order for the connected component containing `root`.
        """
        node_order = list()
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
                priority = lambda u, d=max_deg, n=node_order: self._conn(u, n) + (1 - self._G1.degree(u) / d)
                node = min(curr_layer, key=priority)
                curr_layer.remove(node)
                nodes_left.remove(node)
                node_order.append(node)

            curr_layer = next_layer

        return node_order

    def _conn(self, node: int, node_order: Sequence[int]) -> int:
        """
        Computes the number of neighbours of the given node that are also in the matching order.
        :return: The number indicating the above.
        """
        return len(set(self._G1.neighbors(node)).intersection(set(node_order)))

    def match_all(
        self,
        call_limit: int = -1,
        time_limit: int = -1,
        nmap_limit: int = -1
    ) -> int:
        """
        Performs VF2++ on G1 and G2 according to the matching order. If no matching order is
        provided, `matching_order()` is implictly called. `call_limit` specifies the maximum
        number of states to visit before halting, `time_limit` specifies the maximum period
        of time (in seconds) to run for, and `nmap_limit` specifies the maximum number of
        complete mappings to search for (whichever limit is reached first). If set to -1, no
        limit is imposed.
        :return: The number of complete mappings found.
        """
        self._reset() # Clean everything up

        if call_limit <= 0 and call_limit != -1:
            raise ValueError("call_limit must be -1 or a nonzero integer")
        if time_limit <= 0 and time_limit != -1:
            raise ValueError("time_limit must be -1 or a nonzero integer")
        if nmap_limit <= 0 and nmap_limit != -1:
            raise ValueError("nmap_limit must be -1 or a nonzero integer")
        
        self._node_order, self._cc_endpts = self._matching_order()
        self._call_limit = call_limit if call_limit != -1 else math.inf
        self._time_limit = time_limit if time_limit != -1 else math.inf
        self._nmap_limit = nmap_limit if nmap_limit != -1 else math.inf

        num_maps = self._run(time.time()) # Run VF2++
        self._embeddable = True if num_maps else False # Record embeddable status
        
        return num_maps
    
    def match_best(
        self,
        prev_gmap: Layout,
        call_limit: int = -1,
        time_limit: int = -1,
        nmap_limit: int = -1
    ) -> int:
        """
        Similar to `match_all()`, except only the minimum-cost mapping is saved out of all
        complete mappings discovered. The cost of a mapping is the total distance from nodes
        in the image of `prev_map` to their corresponding nodes in the image of the current
        mapping.
        :return: The number of complete mappings found.
        """
        self._reset()

        if call_limit <= 0 and call_limit != -1:
            raise ValueError("call_limit must be -1 or a nonzero integer")
        if time_limit <= 0 and time_limit != -1:
            raise ValueError("time_limit must be -1 or a nonzero integer")
        
        self._node_order, self._cc_endpts = self._matching_order()
        self._call_limit = call_limit if call_limit != -1 else math.inf
        self._time_limit = time_limit if time_limit != -1 else math.inf
        self._nmap_limit = nmap_limit if nmap_limit != -1 else math.inf

        num_maps = self._run(time.time(), prev_gmap=prev_gmap)
        self._embeddable = True if num_maps else False
        
        return num_maps

    def _run(
        self,
        start: float,
        gmap: Layout = Layout(),
        depth: int = 0,
        prev_gmap: Optional[Layout] = None
    ) -> int:
        
        self._state += 1

        # Check for complete mapping and limits reached
        if depth == len(self._G1):
            if prev_gmap:
                cost = self._cost(gmap, prev_gmap)
                if cost < self._min_cost:
                    self._gmaps = [gmap]
                    self._min_cost = cost
            else:
                self._gmaps.append(gmap.copy())
            return 1
        if time.time() >= start + self._time_limit:
            return 0
        if self._state >= self._call_limit:
            return 0
        
        num_maps = 0 # Number of complete mappings found

        for cand1, cand2 in self._candidates(depth):
            
            # Filter out infeasible candidates: O(deg(cand1) * deg(cand2))
            if self._consistent(gmap, cand1, cand2) and not self._cut(depth, cand1, cand2):

                # Obtain unmapped neighbours of cand1 and cand2
                unmapped_neighbors1 = [v for v in self._G1.neighbors(cand1) if not self._covset1.is_mapped(v)]
                unmapped_neighbors2 = [v for v in self._G2.neighbors(cand2) if not self._covset2.is_mapped(v)]

                # Extend mapping
                gmap[self._qmap[cand1]] = cand2

                # Update neighbor and degree sets
                self._covset1.cover([cand1] + unmapped_neighbors1)
                self._covset2.cover([cand2] + unmapped_neighbors2)
                self._covset1.map(cand1)
                self._covset2.map(cand2)
                self._degset2.decr_neighbors(cand2, self._covset2)
                
                num_maps += self._run(start, gmap, depth + 1)
                
                # Restore neighbor and degree sets
                self._covset1.unmap(cand1)
                self._covset2.unmap(cand2)
                self._covset1.uncover([cand1] + unmapped_neighbors1)
                self._covset2.uncover([cand2] + unmapped_neighbors2)
                self._degset2.incr_neighbors(cand2, self._covset2)

                # If at least one limit has been reached, return early
                if len(self._gmaps) >= self._nmap_limit:
                    return num_maps
                if time.time() >= start + self._time_limit:
                    return num_maps
                if self._state >= self._call_limit:
                    return num_maps
        
        return num_maps
    
    def _candidates(self, depth: int) -> Generator[Tuple[int, int], None, None]:
        """
        Returns an iterator through each candidate pair.
        :return: An iterator through each candidate pair.
        """
        node1 = self._node_order[depth]

        if not (self._covset1 and self._covset2):
            cand_nodes2 = list()
            for node2 in self._G2.node_indices():
                if not self._covset2.is_mapped(node2):
                    cand_nodes2.append((self._centrality2[node2], node2))
            heapify(cand_nodes2)
            while cand_nodes2:
                yield (node1, heappop(cand_nodes2)[1])
        
        else:
            cand_nodes2 = list()
            for node2 in self._G2.node_indices():
                if self._covset2.is_unmapped_neighbor(node2):
                    cand_nodes2.append((self._G2.degree(node2), node2))
            heapify(cand_nodes2)
            while cand_nodes2:
                yield (node1, heappop(cand_nodes2)[1])
    
    def _consistent(self, gmap: Layout, cand1: int, cand2: int) -> bool:
        """
        Returns True if for all mapped neighbours of `cand1`, the nodes they map to are precisely
        the mapped neighbours of `cand2`.
        :return: True if the above holds.
        """
        for neighbor in self._G1.neighbors(cand1):
            if not self._covset1.is_mapped(neighbor):
                continue
            if not self._G2.has_edge(cand2, gmap[self._qmap[neighbor]]):
                return False
        
        return True
    
    def _cut(self, depth: int, cand1: int, cand2: int) -> bool:
        """
        Returns True if
        1. `cand2` has fewer candidate neighbors
        2. `cand2` has fewer unmapped neighbors than `cand1`.
        3. Mapping `cand2` leaves G2 with insufficient non-sink nodes.

        :return: True if the above holds.
        """
        num_cand_neighbors1 = len([v for v in self._G1.neighbors(cand1) if self._covset1.is_unmapped_neighbor(v)])
        num_cand_neighbors2 = len([v for v in self._G2.neighbors(cand2) if self._covset2.is_unmapped_neighbor(v)])
        if num_cand_neighbors2 < num_cand_neighbors1:
            return True

        num_unmapped_neighbors1 = len([v for v in self._G1.neighbors(cand1) if not self._covset1.is_mapped(v)])
        num_unmapped_neighbors2 = len([v for v in self._G2.neighbors(cand2) if not self._covset2.is_mapped(v)])
        if num_unmapped_neighbors2 < num_unmapped_neighbors1:
            return True
        
        num_nonsinks2 = len(self._G2) - self._degset2.num_sinks_after_mapping(cand2, self._covset2)
        if depth in self._cc_endpts and num_nonsinks2 < self._num_nonsinks1:
            return True

        return False
    
    def _cost(self, gmap: Layout, prev_gmap: Layout) -> int:
        """
        Computes the cost of the current mapping, with respect to the previous mapping. The cost of a single
        logical qubit is defined as the shortest path between its mapped physical qubit on the previous and
        on current mapping. The cost of the current mapping is the sum of costs, for each logical qubit.
        :return: The cost of the current mapping.
        """
        total_cost = 0

        for node in self._G1.node_indices():
            p = prev_gmap[self._qmap[node]]
            q = gmap[self._qmap[node]]
            if p is None or q is None:
                continue
            total_cost += rx.dijkstra_shortest_path_lengths(self._G2, p, lambda _: 1, q)[p]
        
        return total_cost

    def verify(self, gmap: Layout) -> bool:
        """
        Verifies the given mapping.
        :return: True if the mapping is complete and consistent.
        """
        visited = defaultdict(bool)

        for node in self._G1.node_indices():
            if not (visited[node] or self._check(gmap, visited, node)):
                return False
        
        return True
    
    def _check(self, gmap: Layout, visited: MutableMapping[int, bool], node: int) -> bool:
        
        visited[node] = True

        for neighbor in self._G1.neighbors(node):
            p = self._qmap[node]
            q = self._qmap[neighbor]
            if q not in gmap:
                return False
            if not self._G2.has_edge(gmap[p], gmap[q]):
                return False
            if not (visited[neighbor] or self._check(gmap, visited, neighbor)):
                return False
        
        return True
    