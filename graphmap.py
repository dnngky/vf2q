from math import ceil, log10
from qiskit.circuit import Qubit
from qiskit.transpiler import Layout
from typing import Optional, Union

class GraphMap(Layout):

    def __init__(self):
        """
        Construct a graph map.
        """
        super().__init__()

    def __getitem__(self, item: Union[Qubit, int]) -> Optional[Union[int, Qubit]]:
        if item in self._p2v:
            return self._p2v[item]
        if item in self._v2p:
            return self._v2p[item]
        return None
    
    def __delitem__(self, key: Union[Qubit, int]):
        if key in self._p2v or key in self._v2p:
            super().__delitem__(key)
    
    def __repr__(self) -> str:
        entries = ", ".join([f"{v.index}: {p}" for v, p in self._v2p.items()])
        return f"GraphMap{{{entries}}}"