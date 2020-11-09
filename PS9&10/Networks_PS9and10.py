import numpy as np
from pprint import pprint
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

def MVR(G):
    n = G.number_of_nodes()
    iterSinceImprovement = 0
    total_iter = 0
    violations_vs_iter = {}
    while iterSinceImprovement < np.math.factorial(n)/(2*np.math.factorial(n-2)):
        # pick two random nodes
        a, b = np.random.randint(n), np.random.randint(n)
        # propose swap
        swapped_G = nx.relabel.relabel_nodes(G, {a:b, b:a})
        original_V = V(G)
        swap_V = V(swapped_G)
        # if the swap decreases V, keep it and reset iterSinceImprovement back to zero
        if swap_V < original_V:
            G = swapped_G.copy()
            iterSinceImprovement = 0
        # if the swap keeps V the same, keep it and increment iterSinceImprovement
        elif swap_V == original_V:
            G = swapped_G.copy()
            iterSinceImprovement += 1
        # if the swap increases V, keep the original graph and increment iterSinceImprovement
        else:
            iterSinceImprovement += 1
        violations_vs_iter[total_iter] = V(G)
        total_iter += 1
    return violations_vs_iter

def V(G):
    count = 0
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            if node < neighbor:
                count += 1
    return count

def createLinearGraph(n):
    G = nx.DiGraph()
    arr = np.arange(n)
    np.random.shuffle(arr)
    for i in range(n):
        G.add_node(i)
    for i in range(len(arr)-1):
        G.add_edge(arr[i], arr[i+1])
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G

def G_n_p(n, p):
    # G = nx.gnp_random_graph(n, p, directed=True)
    G = nx.DiGraph()
    arr = np.arange(n)
    for i in range(n):
        G.add_node(i)

    for node1 in G.nodes():
        for node2 in G.nodes():
            if np.random.uniform() <= p:
                if node1 <= node2:
                    G.add_edge(node1, node2)
                else:
                    G.add_edge(node2, node1)
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G


if __name__ == '__main__':
    dict_arr1 = []
    len_arr1 = []
    for i in range(50):
        print(i)
        G = createLinearGraph(50)
        dict_arr1.append(MVR(G))

    df = pd.DataFrame(dict_arr1[0], index=[0]).transpose()
    ax = df.plot()
    len_arr1.append(len(dict_arr1[0]))

    for dict in dict_arr1[1:]:
        df = pd.DataFrame(dict, index=[0]).transpose()
        df.plot(ax=ax)
        len_arr1.append(len(dict))

    ax.get_legend().remove()
    ax.set_xlabel('Number of timesteps since conception')
    ax.set_ylabel('Number of violations')
    plt.show()

    plt.hist(len_arr1, bins=15)
    plt.xlabel('Timesteps elapsed when algorithm stops')
    plt.ylabel('Occurences')
    plt.show()

    # graph_arr2 = []
    # for i in range(50):
    #     print(i)
    #     G = G_n_p(50)
    #     graph_arr2.append(MVR(G))
