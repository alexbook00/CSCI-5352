import numpy as np
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import random

def ccdf(n, c, r):
    G = nx.DiGraph()

    edges = [(0, 1), (0, 2), (0, 3),
             (1, 0), (1, 2), (1, 3),
             (2, 0), (2, 1), (2, 3),
             (3, 0), (3, 1), (3, 2)]

    G.add_nodes_from([0, 1, 2, 3])
    G.add_edges_from(edges)

    for new_node in range(4, n):
        print(new_node)
        G.add_node(new_node)
        while G.out_degree[new_node] < c:
            random_node = random.choice(list(G.nodes()))
            q_i = (G.degree[random_node] + r)/((new_node - 1) * (r + c))
            if (np.random.uniform() <= q_i) and (new_node != random_node) and not (G.has_edge(new_node, random_node)):
               G.add_edge(new_node, random_node)

    occurences = 0
    for node in G.nodes():
        if G.degree[node] > c:
            occurences += 1

    return occurences/n

if __name__ == '__main__':
    print(ccdf(100, 3, 1))
