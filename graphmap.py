from qiskit.circuit import Qubit
from qiskit.transpiler import Layout
from typing import Optional, Self, Union

class GraphMap(Layout):
    """
    A dynamic data structure representing the isomorphism between two graphs, if
    any. Its structure is analogous to Qiskit's Layout module, with a few exceptions:
    1. Attempting to access a non-existent item (in both v2p and p2v mappings)
    returns None instead of raising a KeyError.
    2. Attempting to delete a non-existent item (in both v2p and p2v mappings)
    returns None instead of raising a LayoutError.
    """

    def __init__(self) -> Self:
        """
        GraphMap constructor.
        """
        super().__init__()

    def __getitem__(self, item: Union[Qubit, int]) -> Optional[Union[int, Qubit]]:
        if item in self._p2v:
            return self._p2v[item]
        if item in self._v2p:
            return self._v2p[item]
        return None
    
    def __delitem__(self, key: Union[Qubit, int]) -> None:
        if key in self._p2v or key in self._v2p:
            super().__delitem__(key)
    
    def __repr__(self) -> str:
        entries = ", ".join([f"{v.index}: {p}" for v, p in self._v2p.items()])
        return f"GraphMap{{{entries}}}"
    
    def clear(self) -> None:
        self._v2p.clear()
        self._p2v.clear()
