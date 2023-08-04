from ag import *
from math import inf
from qiskit.circuit import QuantumCircuit
from qiskit.converters import circuit_to_dag, dag_to_circuit
from qiskit.transpiler import Layout
from typing import Optional
from vf2q import VF2Q

import os
import random
import time

def test_match(
    path: str,
    include_vf2q: bool = True,
    include_rxvf2: bool = True,
    verify_mapping: bool = True,
    display_mapping: bool = True
) -> None:
    """
    Noticeable findings:
    o------CIRCUIT------o-------VF2Q-------o------RX-VF2------o
    | 54QBT_05CYC_QSE_0 | pass (< 5s)      | pass (< 1s)      |
    | 54QBT_05CYC_QSE_2 | pass (< 1s)      | pass (< 1s)      |
    | 54QBT_05CYC_QSE_3 | pass (< 1s)      | timeout^         |
    | 54QBT_05CYC_QSE_5 | pass (< 1s)      | pass (< 100s)    |
    | 54QBT_05CYC_QSE_7 | pass (< 5s)      | pass (< 1s)      |
    | 54QBT_05CYC_QSE_9 | pass (< 1s)      | pass (< 300s)    |
    | 54QBT_10CYC_QSE_0 | pass (< 10s)     | pass (< 1s)      |
    o-------------------o------------------o------------------o
    ^did not halt within 600 seconds.
    """
    max_vf2_runtime = [-1., ""]
    max_vf2pp_runtime = [-1., ""]
    total_vf2_runtime = 0.
    total_vf2pp_runtime = 0.
    num_files = 0
    
    for filename in os.listdir(path):

        print(filename)
        
        circuit = QuantumCircuit.from_qasm_file(path + filename)
        vf2q = VF2Q(circuit)

        if include_rxvf2:
            
            start = time.time()
            is_embeddable = rx.is_subgraph_isomorphic(vf2q._archgraph, vf2q.circgraph, induced=False)
            end = time.time()

            if (end - start) > max_vf2_runtime[0]:
                max_vf2_runtime[0] = (end - start)
                max_vf2_runtime[1] = filename
            total_vf2_runtime += (end - start)
            print(f"rx-vf2_runtime: {end - start}")

            if is_embeddable:
                print("rx-vf2_embeddable: True")

                vf2_map = rx.vf2_mapping(vf2q._archgraph, vf2q.circgraph, subgraph=True, induced=False)
                if display_mapping:
                    print(f"rx-vf2_mapping: {next(vf2_map)}")
            
            else:
                print("rx-vf2_embeddable: False")

        if include_vf2q:

            start = time.time()
            vf2q.match(nmap_limit=1)
            end = time.time()

            if (end - start) > max_vf2pp_runtime[0]:
                max_vf2pp_runtime[0] = (end - start)
                max_vf2pp_runtime[1] = filename
            total_vf2pp_runtime += (end - start)
            print(f"vf2-pp_runtime: {end - start}")

            if vf2q.is_embeddable():
                print("vf2-pp_embeddable: True")

                vf2pp_map = next(vf2q.maps())
                if display_mapping:
                    print(f"vf2pp_mapping: {vf2pp_map}")
                if verify_mapping and not vf2q.verify(vf2pp_map):
                    raise ValueError("vf2-pp verify: Invalid")
            
            else:
                print("vf2-pp_embeddable: False")

            if include_rxvf2:
                assert vf2q.is_embeddable() == is_embeddable, "Conflicting results"
        
        num_files += 1
        print()
    
    if include_rxvf2:
        print(f"rx-vf2_max_runtime: {max_vf2_runtime[0]} ({max_vf2_runtime[1]})")
        print(f"rx-vf2_ttl_runtime: {total_vf2_runtime}")
        print(f"rx-vf2_avg_runtime: {total_vf2_runtime / num_files}")
        print()
    
    print(f"vf2-pp_max_runtime: {max_vf2pp_runtime[0]} ({max_vf2pp_runtime[1]})")
    print(f"vf2-pp_ttl_runtime: {total_vf2pp_runtime}")
    print(f"vf2-pp_avg_runtime: {total_vf2pp_runtime / num_files}")

def test_match_best(
    path: str,
    w1: float,
    w2: float,
    verify_mapping: bool = True,
    display_mapping: bool = True
) -> None:
    
    max_runtime = [-1., ""]
    max_cost = [-1, ""]
    total_runtime = 0.
    total_cost = 0
    num_files = 0

    for filename in os.listdir(path):

        if filename.startswith("54QBT"): continue

        print(filename)

        circuit = QuantumCircuit.from_qasm_file(path + filename)
        vf2q = VF2Q(circuit)

        # Generate random mapping
        prev_map = Layout()
        nodes2 = list(vf2q._archgraph.nodes())
        for node1 in vf2q.circgraph.nodes():
            node2 = random.choice(nodes2)
            prev_map[node2] = node1
            nodes2.remove(node2)

        start = time.time()
        vf2q.match_best(prev_map, w1=w1, w2=w2)
        end = time.time()

        if (end - start) > max_runtime[0]:
            max_runtime[0] = (end - start)
            max_runtime[1] = filename
        total_runtime += (end - start)
        print(f"runtime: {end - start}")

        if vf2q.is_embeddable():
            print("embeddable: True")

            mapping = vf2q.maps[0]
            cost = vf2q.compute_cost(mapping)
            if cost > max_cost[0]:
                max_cost[0] = cost
                max_cost[1] = filename
            total_cost += cost
            
            print(f"cost: {cost}")
            if display_mapping:
                print(f"mapping: {mapping}")
            if verify_mapping and not vf2q.verify(mapping):
                raise ValueError("verify: Invalid")
            
        else:
            print("embeddable: False")

        num_files += 1
        print()

    print(f"max_runtime: {max_runtime[0]} ({max_runtime[1]})")
    print(f"ttl_runtime: {total_runtime}")
    print(f"avg_runtime: {total_runtime / num_files}")

    print(f"max_cost: {max_cost[0]} ({max_cost[1]})")
    print(f"ttl_cost: {total_cost}")
    print(f"avg_cost: {total_cost / num_files}")

def test_partition(
    path: str,
    max_iter: int = inf,
    max_calls: Optional[int] = None
) -> None:

    max_reduction = [0, ""]
    total_reduction = 0
    num_files = 0

    for filename in os.listdir(path):

        circuit = QuantumCircuit.from_qasm_file(path + filename)
        dag = circuit_to_dag(circuit)
        VF2Q.partition_circuit(circuit, max_iter, max_calls)
        reduction = len(list(dag.layers())) - len(VF2Q.subcircs)
        if reduction > max_reduction[0]:
            max_reduction[0] = reduction
            max_reduction[1] = filename
        total_reduction += reduction
        print(f"{filename}: {len(list(dag.layers()))} -> {len(VF2Q.subcircs)} (-{reduction})")

        num_files += 1

    print(f"max_reduction: {max_reduction[0]} ({max_reduction[1]})")
    print(f"ttl_reduction: {total_reduction}")
    print(f"avg_reduction: {total_reduction / num_files}")

def test_match_subcircs(
    path: str,
    max_iter: int,
    max_calls: int,
    w1: float,
    w2: float
) -> None:

    max_cost = [-1, ""]
    total_cost = 0
    num_files = 0

    for filename in os.listdir(path):

        circuit = QuantumCircuit.from_qasm_file(path + filename)
        VF2Q.partition_circuit(circuit, max_iter, max_calls)
        VF2Q.match_subcircs(w1, w2)
        if VF2Q.cost > max_cost[0]:
            max_cost[0] = VF2Q.cost
            max_cost[1] = filename
        total_cost += VF2Q.cost
        print(f"{filename}: {VF2Q.cost}")

        num_files += 1

    print(f"max_cost: {max_cost[0]} ({max_cost[1]})")
    print(f"ttl_cost: {total_cost}")
    print(f"avg_cost: {total_cost / num_files}")

def test_all(
    path: str,
    max_iter: int,
    max_calls: int,
    w1: float,
    w2: float
) -> None:
    
    max_swaps = [-1, ""]
    total_swaps = 0
    max_depth_diff = [-1, ""]
    total_depth_diff = 0
    max_runtime = [-1., ""]
    total_runtime = 0.
    num_files = 0
    
    for filename in os.listdir(path):
        
        VF2Q.circuit = QuantumCircuit.from_qasm_file(path + filename)
        start = time.time()
        VF2Q.partition_circuit(max_iter, max_calls)
        VF2Q.match_subcircs(w1, w2)
        VF2Q.apply_swaps()
        VF2Q.transform_circuit()
        end = time.time()

        swaps = len(VF2Q.transformed_circ.data) - len(VF2Q.circuit)
        if swaps > max_swaps[0]:
            max_swaps[0] = swaps
            max_swaps[1] = filename
        total_swaps += swaps
        depth_diff = VF2Q.transformed_circ.depth() - VF2Q.circuit.depth()
        if depth_diff > max_depth_diff[0]:
            max_depth_diff[0] = depth_diff
            max_depth_diff[1] = filename
        total_depth_diff += depth_diff
        if end - start > max_runtime[0]:
            max_runtime[0] = end - start
            max_runtime[1] = filename
        total_runtime += end - start
        print(f"{filename}: swaps {swaps}, depth_diff {depth_diff}")
        
        num_files += 1
    
    print(f"max_swaps: {max_swaps[0]} ({max_swaps[1]})")
    print(f"ttl_swaps: {total_swaps}")
    print(f"avg_swaps: {total_swaps / num_files}")
    print(f"max_depth_diff: {max_depth_diff[0]} ({max_depth_diff[1]})")
    print(f"ttl_depth_diff: {total_depth_diff}")
    print(f"avg_depth_diff: {total_depth_diff / num_files}")
    print(f"max_runtime: {max_runtime[0]} ({max_runtime[1]})")
    print(f"ttl_runtime: {total_runtime}")
    print(f"avg_runtime: {total_runtime / num_files}")

if __name__ == "__main__":

    VF2Q.archgraph = sycamore54()
    path = "./benchmark/20Q_depth_Tokyo/"

    # test_match(
    #     path,
    #     include_vf2q=True,
    #     include_rxvf2=False,
    #     verify_mapping=True,
    #     display_mapping=False
    # )
    # test_match_best(
    #     path,
    #     w1=0,
    #     w2=0,
    #     verify_mapping=True,
    #     display_mapping=False
    # )
    # test_partition(
    #     path,
    #     max_iter=20,
    #     max_calls=2000
    # )
    # test_match_subcircs(
    #     path,
    #     max_iter=20,
    #     max_calls=2000,
    #     w1=200,
    #     w2=200
    # )
    test_all(
        path,
        max_iter=20,
        max_calls=2000,
        w1=200,
        w2=200
    )
