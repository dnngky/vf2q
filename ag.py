import rustworkx as rx

# Q-grid
def qgrid(m,n) -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(0,m*n-1)))
    for i in range(0,m):
        for j in range(0,n):
            if i < m-1:
                g.add_edge(j*m+i, j*m+i+1, (j*m+i, j*m+i+1))
            if j < n-1:
                g.add_edge(j*m+i, (j+1)*m+i, (j*m+i, (j+1)*m+i))
    return g

# IBM Q Ourense (5 qubits)
def ourense() -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(5)))
    g.add_edge(0, 1, (0, 1))
    g.add_edge(0, 2, (0, 2))
    g.add_edge(0, 3, (0, 3))
    g.add_edge(3, 4, (3, 4))
    return g

# IBM Q Melbourne (14 qubits)
def melbourne() -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(14)))
    for i in range(6):
        g.add_edge(i, i+1, (i, i+1))
        g.add_edge(i+1, i+7, (i+1, i+7))
        g.add_edge(i+7, i+8, (i+7, i+8))
    return g

# IBM Q Tokyo (20 qubits)
def tokyo() -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(20)))
    for i in range(0, 4):
        g.add_edge(i, i+1, (i, i+1))
    for i in range(5, 9):
        g.add_edge(i, i+1, (i, i+1))
    for i in range(10, 14):
        g.add_edge(i, i+1, (i, i+1))
    for i in range(15, 19):
        g.add_edge(i, i+1, (i, i+1))
    for i in range(0, 15):
        g.add_edge(i, i+5, (i, i+5))
    for i in [1, 3, 5, 7, 11, 13]:
        g.add_edge(i, i+6, (i, i+6))
    for i in [2, 4, 6, 8, 12, 14]:
        g.add_edge(i, i+4, (i, i+4))
    return g

# IBM Q Rochester (53 qubits)
def rochester() -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(53)))
    I = list(range(4)) + list(range(7,15)) +\
        list(range(19,27)) + list(range(30,38)) +\
            list(range(42,50))
    for i in I:
        g.add_edge(i, i+1, (i, i+1))
    E = [
            (0, 5), (5, 9), (4, 6), (6, 13), (7, 16), (16, 19),
            (11, 17), (17, 23), (15, 18), (18, 27), (21, 28), (28, 32),
            (25, 29), (29, 36), (30, 39), (39, 42), (34, 40), (40, 46),
            (38, 41), (41, 50), (44, 51), (48, 52)
        ]
    g.add_edges_from([(i, j, (i, j)) for i, j in E])
    return g

# Google Sycamore (53 qubits)
def sycamore53() -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(54))) 
    I = list(range(6, 12)) + list(range(18, 24)) + list(range(30,36))+ list(range(42,48))
    for i in I:
        for j in g.nodes():
            if j in I: continue
            if i - j in [5,6] or j - i in [6,7]:
                g.add_edge(i, j, (i, j))
    g.remove_node(3)
    assert 3 not in g.node_indices(), "Node error"
    for i in range(4, len(g) + 1):
        g[i] = i - 1
    return g

# Google Sycamore (54 qubits)
def sycamore54() -> rx.PyGraph:
    g = rx.PyGraph()
    g.add_nodes_from(list(range(54)))
    I = list(range(6, 12)) + list(range(18, 24)) + list(range(30, 36)) + list(range(42, 48))
    for i in I:
        for j in g.nodes():
            if j in I:
                continue
            if i-j in [5, 6] or j-i in [6, 7]:
                g.add_edge(i, j, (i, j))
    return g


if __name__=='__main__':
    AG = sycamore54()
    print(f"Nodes ({len(AG)}): {AG.nodes()}")
    print(f"Edges ({len(AG.edges())}): {AG.edges()}")