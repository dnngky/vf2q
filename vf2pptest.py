from ag import *
from qiskit.circuit import QuantumCircuit
from vf2pp import VF2PP

import os
import time

if __name__ == "__main__":

    path = "./benchmark/BNTF/"

    """
    VF2PP has trouble with:
    - 54QBT_05CYC_QSE_1
    - 54QBT_05CYC_QSE_3
    - 54QBT_05CYC_QSE_9
    
    rustworkx's VF2 has trouble with (crashes the terminal):
    - 54QBT_05CYC_QSE_2

    In general, rustwork's VF2 is much faster (0s for most circuits,
    with the exception of 54QBT_15+).
    """

    max_vf2_runtime = [0., None]
    max_vf2pp_runtime = [0., None]
    total_vf2_runtime = 0.
    total_vf2pp_runtime = 0.
    num_files = 0

    VERIFY_MAPPING = True # Toggle this to enable/disable VF2PP mapping verification
    INCLUDE_VF2 = True # Toggle this to include rustworkx's vf2_mapping() algorithm
    
    for filename in os.listdir(path):

        if filename == "54QBT_05CYC_QSE_1.qasm": continue
        if filename == "54QBT_05CYC_QSE_3.qasm": continue
        if filename == "54QBT_05CYC_QSE_9.qasm": continue
        if filename == "54QBT_05CYC_QSE_2.qasm": continue

        print(filename)
        
        circuit = QuantumCircuit.from_qasm_file(path + filename)
        vf2pp = VF2PP(circuit, sycamore54())

        if INCLUDE_VF2:
        
            start = time.time()
            vf2_map = rx.vf2_mapping(vf2pp.archgraph, vf2pp.circgraph, subgraph=True, induced=False)
            end = time.time()
            if (end - start) > max_vf2_runtime[0]:
                max_vf2_runtime[0] = (end - start)
                max_vf2_runtime[1] = filename
            total_vf2_runtime += (end - start)

            print(f"vf2_mapping: {next(vf2_map)}")
            print(f"vf2_runtime: {end - start}")

        start = time.time()
        n_maps = vf2pp.run(vf2pp.matching_order(), call_limit=1)
        end = time.time()
        if (end - start) > max_vf2pp_runtime[0]:
            max_vf2pp_runtime[0] = (end - start)
            max_vf2pp_runtime[1] = filename
        total_vf2pp_runtime += (end - start)
        vf2pp_map = next(vf2pp.mappings())

        print(f"vf2pp_mapping: {vf2pp_map}")
        print(f"vf2pp_runtime: {end - start}")

        if VERIFY_MAPPING:
            if not vf2pp.verify(vf2pp_map):
                raise ValueError("Mapping is not correct!")
            print("[MAPPING VERIFIED]")
        
        num_files += 1
        print()
    
    if INCLUDE_VF2:
        print(f"max_vf2_runtime: {max_vf2_runtime[0]} ({max_vf2_runtime[1]})")
        print(f"total_vf2_runtime: {total_vf2_runtime}")
        print(f"avg_vf2_runtime: {total_vf2_runtime / num_files}")
        print()
    
    print(f"max_vf2pp_runtime: {max_vf2pp_runtime[0]} ({max_vf2pp_runtime[1]})")
    print(f"total_vf2pp_runtime: {total_vf2pp_runtime}")
    print(f"avg_vf2pp_runtime: {total_vf2pp_runtime / num_files}")
