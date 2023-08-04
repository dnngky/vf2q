from coverage import Coverage
from collections import defaultdict
from connectivity import Connectivity
from heapq import heapify, heappop
from math import inf
from time import time, sleep
from typing import List, Generator, MutableMapping, Optional, Self, Sequence, Tuple

from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Layout
from qiskit.transpiler.passes.routing.algorithms.token_swapper import ApproximateTokenSwapper

from rustworkx import (
    all_pairs_dijkstra_path_lengths,
    betweenness_centrality,
    connected_components,
    dijkstra_shortest_path_lengths,
    distance_matrix,
    is_subgraph_isomorphic,
    PyGraph,
)

class VF2QMeta(type):
    """
    Metaclass for VF2Q.

    Attributes:
        archgraph (PyGraph): architecture graph of the quantum device
        circuit (QuantumCircuit): quantum circuit to be mapped
        subcircs (List[QuantumCircuit]): list of embeddable sub-circuits of circuit
        matchers (VF2Q): list of VF2Q instances for each sub-circuit
        maps (List[Layout]): list of mappings for each sub-circuit
        swaps (List[List]): list of swap sequences to transform each mapping to its successor
        cost (int): total cost across all mappings
        transformed_circ (QuantumCircuit): transformed quantum circuit on archgraph, including swaps
    """
    def __init__(cls, *_) -> Self:
        cls._archgraph = None
        cls._circuit = None
        cls._subcircs = list()
        cls._matchers = list()
        cls._maps = list()
        cls._swaps = list()
        cls._cost = 0
        cls._transformed_circ = None
    
    @property
    def archgraph(cls) -> PyGraph:
        return cls._archgraph
    
    @archgraph.setter
    def archgraph(cls, archgraph: PyGraph) -> None:
        if not isinstance(archgraph, PyGraph):
            raise TypeError("Archgraph is not a PyGraph")
        cls._archgraph = archgraph

    @property
    def circuit(cls) -> QuantumCircuit:
        return cls._circuit
    
    @circuit.setter
    def circuit(cls, circuit: QuantumCircuit) -> None:
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Circuit is not a QuantumCircuit")
        cls._circuit = circuit

    @property
    def circgraph(cls) -> PyGraph:
        return VF2Q.graphify(cls._circuit)

    @property
    def subcircs(cls) -> List[QuantumCircuit]:
        return cls._subcircs
    
    @property
    def matchers(cls) -> List[Self]:
        return cls._matchers
    
    @property
    def maps(cls) -> List[Layout]:
        return cls._maps
    
    @property
    def swaps(cls) -> List[List[Tuple]]:
        return cls._swaps
    
    @property
    def cost(cls) -> int:
        return cls._cost
    
    @property
    def transformed_circ(cls) -> QuantumCircuit:
        return cls._transformed_circ

class VF2Q(metaclass=VF2QMeta):
    """
    Qubit mapping using subgraph isomorphism, based on VF2(++).
    
    Attributes:
        G1 (PyGraph): interaction graph of quantum circuit
        G2 (PyGraph): architecture graph of the quantum device
        qmap (Layout): qubit-to-index mapping
        num_non_singletons1: number of nodes which are not themselves connected components of G1
        dist2 (dict): shortest path lengths between all pairs of nodes in G2
        deg2 (dict): normalized degree of nodes in G2
        centrality2 (dict): normalized betweenness centrality of nodes in G2
        node_order (list): matching order of G1
        cc_end_idxs (list): end indexes of connected components of G1 in the matching order

        prev_gmap (Layout): previous mapping
        gmap_min_cost (int): cost of the current minimum-cost mapping
        num_states (int): number of states visited
        covg1 (Coverage): coverage of G1
        covg2 (Coverage): coverage of G2
        conn2 (Connectivity): connectivity of G2

        call_limit (int): upperbound for number of visited states
        time_limit (int): upperbound for period of time (s) spent
        nmap_limit (int): upperbound for number of complete mappings to search for
        w1 (float): weight factor for prioritised candidate selection
        w2 (float): weight factor for prioritised candidate selection

        gmaps (list): list of complete mappings found
        is_embeddable (bool): True if G1 is (directly) embeddable onto G2
    """
    def __init__(self, circuit: QuantumCircuit) -> Self:
        """
        VF2Q constructor.
        
        Args:
            circuit (QuantumCircuit): quantum circuit to be embedded
            archgraph (PyGraph): architecture graph to embed onto
        """
        if VF2Q._archgraph is None:
            raise AttributeError("Archgraph has not been set")
        if not isinstance(circuit, QuantumCircuit):
            raise TypeError("Circuit is not a QuantumCircuit")
        if circuit.num_qubits > len(VF2Q._archgraph):
            raise ValueError("Circuit width exceeds archgraph width")
        
        self._G1, self._qmap = VF2Q.graphify(circuit)
        self._G2 = VF2Q._archgraph
        self._num_non_singletons1 = \
            len(self._G1) - len([cc for cc in connected_components(self._G1) if len(cc) == 1])
        self._dist2 = all_pairs_dijkstra_path_lengths(self._G2, lambda _: 1)
        self._centrality2 = betweenness_centrality(self._G2, normalized=False)
        self._node_order = list()
        self._cc_end_idxs = list()
        
        # DYNAMIC ATTRIBUTES
        self._prev_gmap = None
        self._gmap_min_cost = inf
        self._num_states = 0
        self._covg1 = Coverage(self._G2)
        self._covg2 = Coverage(self._G2)
        self._conn2 = Connectivity(self._G2)
        
        # LIMITS AND WEIGHTS
        self._call_limit = -1
        self._time_limit = -1
        self._nmap_limit = -1
        self._w1 = -1
        self._w2 = -1

        # RESULTS
        self._gmaps = list()
        self._is_embeddable = None

    @classmethod
    def partition_circuit(
        cls,
        max_iter: int = inf,
        max_calls: Optional[int] = None
    ) -> None:
        """
        Computes the minimal set of embeddable sub-circuits from the given circuit.
        """
        cls._subcircs.clear()
        subcircs = [dag_to_circuit(layer["graph"]) for layer in circuit_to_dag(cls._circuit).layers()]
        n = 0

        while subcircs and n < max_iter:
            if len(subcircs) > 1:
                subcircs = cls._partition_subcircs(subcircs, len(subcircs), len(subcircs), max_calls)
            cls._subcircs.append(subcircs.pop(0))
            n += 1
        
        if subcircs:
            cls._subcircs.extend(subcircs)

    @classmethod
    def _partition_subcircs(
        cls,
        subcircs: Sequence[QuantumCircuit],
        split: int,
        size: int,
        max_calls: Optional[int] = None,
    ) -> List[QuantumCircuit]:
        """
        Given an ordered set of sub-circuits P, computes P' such that the leftmost sub-circuit of P' is
        a maximal embeddable composition of the first n sub-circuits of P, i.e., composing it with the
        (n+1)th sub-circuit will make it non-embeddable.
        :return: List of sub-circuits with a maximal leftmost embeddable sub-circuit.
        """
        if size <= 1:
            return subcircs
        
        circ = subcircs[0].copy_empty_like()
        for i in range(split):
            circ.compose(subcircs[i], inplace=True)

        is_embeddable = is_subgraph_isomorphic(
            cls._archgraph, cls.graphify(circ)[0], induced=False, call_limit=max_calls
        )
        if is_embeddable and size == len(subcircs):
            return [circ]
        elif is_embeddable:
            return cls._partition_subcircs([circ] + subcircs[split:], 1 + size // 2, size // 2, max_calls)
        else:
            return cls._partition_subcircs(subcircs, split - size // 2, size // 2 + size % 2, max_calls)

    @classmethod
    def match_subcircs(
        cls,
        w1: float = 200,
        w2: float = 200
    ) -> None:
        """
        Match each and every sub-circuit.
        """
        if cls._archgraph is None:
            raise AttributeError("Archgraph has not been set")
        
        cls._matchers.clear()
        cls._cost = 0

        if len(cls._subcircs) == 1:
            matcher = cls(cls._subcircs[0])
            matcher.match(nmap_limit=1)
            cls._matchers = [matcher]
            cls._maps = matcher.maps

        fwd_cost = 0
        fwd_matchers = list()
        fwd_gmaps = list()
        prev_gmap = None

        for i in range(len(cls._subcircs)):
            matcher = cls(cls._subcircs[i])
            matcher.match_best(prev_gmap, w1=w1, w2=w2)
            if prev_gmap == matcher.maps[0]:
                continue
            prev_gmap = matcher.maps[0]
            fwd_gmaps.append(prev_gmap)
            fwd_cost += matcher.compute_cost(matcher.maps[0])
            fwd_matchers.append(matcher)
        
        bwd_cost = 0
        bwd_matchers = list()
        bwd_gmaps = list()
        prev_gmap = None

        for i in reversed(range(len(cls._subcircs))):
            matcher = cls(cls._subcircs[i])
            matcher.match_best(prev_gmap, w1=w1, w2=w2)
            if prev_gmap == matcher.maps[0]:
                continue
            prev_gmap = matcher.maps[0]
            bwd_gmaps.append(prev_gmap)
            bwd_cost += matcher.compute_cost(matcher.maps[0])
            bwd_matchers.append(matcher)
        
        if fwd_cost <= bwd_cost:
            cls._matchers = fwd_matchers
            cls._maps = fwd_gmaps
            cls._cost = fwd_cost
        else:
            cls._matchers = list(reversed(bwd_matchers))
            cls._maps = list(reversed(bwd_gmaps))
            cls._cost = bwd_cost
        
    @classmethod
    def apply_swaps(cls, trials: int = 4) -> None:
        """
        Apply swaps to transform the mapping of some sub-circuit to that of the successive
        sub-circuit, for all sub-circuits.
        """
        cls._swaps.clear()
        for i in range(1, len(cls._maps)):
            mapping = dict()
            for q in cls._maps[i - 1].get_physical_bits():
                mapping[q] = cls._maps[i][cls._maps[i - 1][q]]
            swapper = ApproximateTokenSwapper(cls._archgraph)
            cls._swaps.append(swapper.map(mapping, trials))
    
    @classmethod
    def transform_circuit(cls) -> None:

        cls._transformed_circ = QuantumCircuit(len(cls._archgraph))
        for i in range(len(cls._subcircs)):
            gmap = cls._maps[i]
            for gate in cls._subcircs[i].data:
                cls._transformed_circ.append(gate.replace(
                    qubits=[cls._transformed_circ.qubits[gmap[q]] for q in gate.qubits]
                ))
            if i < len(cls._swaps):
                for q0, q1 in cls._swaps[i]:
                    cls._transformed_circ.swap(q0, q1)

    @staticmethod
    def graphify(circuit: QuantumCircuit) -> Tuple[PyGraph, Layout]:
        """
        Converts the given quantum circuit to a graph G = (V, E), where V comprises
        the qubits in the circuit, and for every (p, q) in E, there exists some
        CNOT gate operating on qubits p and q.
        :return: The graph corresponding to the given quantum circuit, and its
        qubit-to-index mapping.
        """
        circgraph = PyGraph()
        qumap = Layout()

        for qubit in circuit.qubits:
            index = circgraph.add_node(qubit)
            qumap[index] = qubit

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
        Resets all dynamic attributes and data structures.
        """
        self._gmap_min_cost = inf
        self._num_states = 0
        self._covg1.clear()
        self._covg2.clear()
        self._conn2.clear()
        self._gmaps.clear()
        self._is_embeddable = None
    
    @property
    def circgraph(self) -> PyGraph:
        return self._G1
    
    @property
    def maps(self) -> List[Layout]:
        """
        List of complete mappings found.
        """
        return self._gmaps
    
    def is_embeddable(self, call_limit: int = -1, time_limit: float = -1) -> bool:
        """
        Returns True if the given circuit is embeddable onto, i.e., subgraph isomorphic
        to, the architecture graph.
        :return: True if the circuit is embeddable onto the architecture.
        """
        if self._is_embeddable is None:
            self.match(call_limit, time_limit, nmap_limit=1)
        return self._is_embeddable
    
    def matching_order(self) -> Tuple[List[int], List[int]]:
        """
        Computes the matching order for G1, based on VF2++.
        :return: The matching order for G1.
        """
        node_orders = list()
        nodes_left = list(self._G1.node_indices())
        visited = defaultdict(bool)
        max_deg = max([self._G1.degree(u) for u in self._G1.node_indices()])
        
        while nodes_left:
            root = min(nodes_left, key=lambda u: self._G1.degree(u))
            node_order = self._process_level(root, visited, nodes_left, max_deg)
            node_orders.append(node_order)
        
        merged_node_orders = list()
        cc_end_idxs = list()
        for node_order in sorted(node_orders, key=lambda n: len(n), reverse=True):
            merged_node_orders.extend(node_order)
            cc_end_idxs.append(len(merged_node_orders))
        cc_end_idxs = list(map(lambda x: x - 1, cc_end_idxs))
        
        return merged_node_orders, cc_end_idxs
    
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

        while curr_layer:
            next_layer = list()
            
            for u in curr_layer:
                for v in self._G1.neighbors(u):
                    if visited[v]:
                        continue
                    next_layer.append(v)
                    visited[v] = True
            
            while curr_layer:
                if max_deg > 0:
                    priority = lambda u, n=node_order, d=max_deg: \
                        len(set(self._G1.neighbors(u)).intersection(set(n))) + \
                        (1 - self._G1.degree(u) / d)
                else: # if G1 is edgeless
                    priority = lambda u, n=node_order: \
                        len(set(self._G1.neighbors(u)).intersection(set(n)))
                node = min(curr_layer, key=priority)
                curr_layer.remove(node)
                nodes_left.remove(node)
                node_order.append(node)

            curr_layer = next_layer

        return node_order

    def match(
        self,
        call_limit: int = -1,
        time_limit: float = -1,
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
        self._reset()

        if call_limit <= 0 and call_limit != -1:
            raise ValueError("call_limit must be -1 or a nonzero integer")
        if time_limit <= 0 and time_limit != -1:
            raise ValueError("time_limit must be -1 or a nonzero integer")
        if nmap_limit <= 0 and nmap_limit != -1:
            raise ValueError("nmap_limit must be -1 or a nonzero integer")
        
        self._node_order, self._cc_end_idxs = self.matching_order()
        self._call_limit = call_limit if call_limit != -1 else inf
        self._time_limit = time_limit if time_limit != -1 else inf
        self._nmap_limit = nmap_limit if nmap_limit != -1 else inf

        num_maps = self._run(time()) # Run VF2Q
        self._is_embeddable = True if num_maps else False # Record embeddable status
        
        return num_maps
    
    def match_best(
        self,
        prev_gmap: Layout,
        call_limit: int = -1,
        time_limit: float = -1,
        w1: float = 200,
        w2: float = 200
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
        
        self._node_order, self._cc_end_idxs = self.matching_order()
        self._prev_gmap = prev_gmap
        self._call_limit = call_limit if call_limit != -1 else inf
        self._time_limit = time_limit if time_limit != -1 else inf
        self._nmap_limit = 1
        self._w1 = w1
        self._w2 = w2

        num_maps = self._run(time()) # Run VF2Q
        self._is_embeddable = True if num_maps else False # Record embeddable status
        
        return num_maps
    
    def _run(
        self,
        start: float,
        gmap: Layout = Layout(),
        depth: int = 0
    ) -> int:
        
        self._num_states += 1

        # Check for complete mapping and limits reached
        if depth == len(self._G1):
            self._gmaps.append(gmap.copy())
            return 1
        if time() >= start + self._time_limit:
            return 0
        if self._num_states >= self._call_limit:
            return 0
        
        num_maps = 0 # Number of complete mappings found

        for cand1, cand2 in self._candidates(depth):
            
            # Filter out infeasible candidates
            if self._consistent(gmap, cand1, cand2) and not self._cut(depth, cand1, cand2):

                # Obtain unmapped neighbours of cand1 and cand2
                unmapped_neighbors1 = [v for v in self._G1.neighbors(cand1) if not self._covg1.is_mapped(v)]
                unmapped_neighbors2 = [v for v in self._G2.neighbors(cand2) if not self._covg2.is_mapped(v)]

                # Extend mapping
                gmap[self._qmap[cand1]] = cand2

                # Update coverage and connectivity
                self._covg1.cover([cand1] + unmapped_neighbors1)
                self._covg2.cover([cand2] + unmapped_neighbors2)
                self._covg1.map(cand1)
                self._covg2.map(cand2)
                self._conn2.disconnect_neighbors(cand2, self._covg2)
                
                # Descend to the next state
                num_maps += self._run(start, gmap, depth + 1)
                
                # Restore coverage and connectivity
                self._covg1.unmap(cand1)
                self._covg2.unmap(cand2)
                self._covg1.uncover([cand1] + unmapped_neighbors1)
                self._covg2.uncover([cand2] + unmapped_neighbors2)
                self._conn2.reconnect_neighbors(cand2, self._covg2)

                # Unextend mapping
                del gmap[self._qmap[cand1]]

                # If at least one limit has been reached, return early
                if len(self._gmaps) >= self._nmap_limit:
                    return num_maps
                if time() >= start + self._time_limit:
                    return num_maps
                if self._num_states >= self._call_limit:
                    return num_maps
        
        return num_maps
    
    def _candidates(
        self,
        depth: int
    ) -> Generator[Tuple[int, int], None, None]:
        """
        Returns an iterator through each candidate pair.
        :return: An iterator through each candidate pair.
        """
        cand_nodes2 = list()

        if not (self._covg1.num_unmapped_neighbors and self._covg2.num_unmapped_neighbors):
            for node2 in self._G2.node_indices():
                if self._covg2.is_mapped(node2):
                    continue
                
                priority2 = self._centrality2[node2]
                if self._prev_gmap:
                    cc_end_idx = [i for i in self._cc_end_idxs if i >= depth][0]
                    cc_dist_sum = 0
                    for i in range(depth, cc_end_idx + 1):
                        node1 = self._qmap[self._node_order[i]]
                        if self._prev_gmap[node1] == node2:
                            continue
                        cc_dist_sum += self._dist2[self._prev_gmap[node1]][node2]
                    priority2 +=  self._w1 * cc_dist_sum / (cc_end_idx + 1 - depth)
                cand_nodes2.append((round(priority2, 10), node2))
        
        else:
            for node2 in self._G2.node_indices():
                if not self._covg2.is_unmapped_neighbor(node2):
                    continue
                
                priority2 = self._G2.degree(node2)
                node1 = self._qmap[self._node_order[depth]]
                if self._prev_gmap and self._prev_gmap[node1] != node2:
                    priority2 += self._w2 * self._dist2[self._prev_gmap[node1]][node2]                
                cand_nodes2.append((priority2, node2))
            
        heapify(cand_nodes2)
        while cand_nodes2:
            yield (self._node_order[depth], heappop(cand_nodes2)[1])
    
    def _consistent(self, gmap: Layout, cand1: int, cand2: int) -> bool:
        """
        Returns True if every mapped neighbor of `cand1` maps to a distinct mapped neighbor of `cand1`.
        :return: True if the above holds.
        """
        for neighbor in self._G1.neighbors(cand1):
            if not self._covg1.is_mapped(neighbor):
                continue
            if not self._G2.has_edge(cand2, gmap[self._qmap[neighbor]]):
                return False
        
        return True
    
    def _cut(self, depth: int, cand1: int, cand2: int) -> bool:
        """
        Returns True if
        1. `cand2` has fewer candidate neighbors than `cand1`.
        2. `cand2` has fewer unmapped neighbors than `cand1`.
        3. `cand1` is the last unmapped node of the current connected component, and mapping it to `cand2`
        leaves G2 with insufficient non-sink nodes with respect to the number of non-singletons in G1.
        :return: True if the above holds.
        """
        # Cut rule 1
        num_cand_neighbors1 = len([v for v in self._G1.neighbors(cand1) if self._covg1.is_unmapped_neighbor(v)])
        num_cand_neighbors2 = len([v for v in self._G2.neighbors(cand2) if self._covg2.is_unmapped_neighbor(v)])
        if num_cand_neighbors2 < num_cand_neighbors1:
            return True

        # Cut rule 2
        num_unmapped_neighbors1 = len([v for v in self._G1.neighbors(cand1) if not self._covg1.is_mapped(v)])
        num_unmapped_neighbors2 = len([v for v in self._G2.neighbors(cand2) if not self._covg2.is_mapped(v)])
        if num_unmapped_neighbors2 < num_unmapped_neighbors1:
            return True
        
        # Cut rule 3
        if depth in self._cc_end_idxs and \
            self._conn2.num_non_sinks_after_mapping(cand2, self._covg2) < self._num_non_singletons1:
            return True

        return False
    
    def compute_cost(self, gmap: Layout) -> int:
        """
        Computes the cost of the current mapping, with respect to the previous mapping. The cost of a single
        logical qubit is defined as the shortest path between its mapped physical qubit on the previous and
        on current mapping. The cost of the current mapping is the sum of costs, for each logical qubit.
        :return: The cost of the current mapping.
        """
        if not self._prev_gmap:
            return 0
        
        total_cost = 0
        for node in self._G1.node_indices():
            p = self._prev_gmap[self._qmap[node]]
            q = gmap[self._qmap[node]]
            if p is None or q is None:
                continue
            dist = int(dijkstra_shortest_path_lengths(self._G2, p, lambda _: 1, q)[q])
            total_cost += max(dist * 2 - 1, 0)
        # print()
        return total_cost

    def verify(self, gmap: Layout) -> bool:
        """
        Verifies the given mapping using DFS.
        :return: True if the mapping is complete and consistent, False otherwise.
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
