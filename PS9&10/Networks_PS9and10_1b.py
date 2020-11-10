import numpy as np
from pprint import pprint
import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
from matplotlib import cm

def MVR(G):
    n = G.number_of_nodes()
    iterSinceImprovement = 0
    total_iter = 0
    # violations_vs_iter = {}
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
        # violations_vs_iter[total_iter] = V(G)
        total_iter += 1
    return V(G)

def V(G):
    count = 0
    for node in G.nodes():
        for neighbor in G.neighbors(node):
            if node < neighbor:
                count += 1
    return count

def G_n_p(n, p):
    G = nx.DiGraph()
    arr = np.arange(n)
    for i in range(n):
        G.add_node(i)

    for node1 in G.nodes():
        for node2 in G.nodes():
            if np.random.uniform() <= p and node1 != node2:
                G.add_edge(node1, node2)
    # nx.draw(G, with_labels=True)
    # plt.show()
    return G

def make_plots(filename, a, b):
    dfNew = pd.read_csv(filename)

    X = list(dfNew['n'])
    Y = list(dfNew['p'])
    Z = list(dfNew['violations after algorithm run'])
    x = np.reshape(X, (a, b))
    y = np.reshape(Y, (a, b))
    z = np.reshape(Z, (a, b))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = ax.plot_surface(x, y, z, cmap = cm.coolwarm)
    fig.colorbar(q)
    ax.set_xlabel('Number of nodes')
    ax.set_ylabel('Probability of edge existing between two nodes')
    ax.set_zlabel('Violations after MVR algorithm runs')
    plt.show()

if __name__ == '__main__':
    V_n_p_dict = {}
    n_arr = [10, 20, 30, 40, 50]
    p_arr = [.1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    for n in n_arr:
        for p in p_arr:
            print(n, p)
            V_arr = []
            for i in range(10):
                G = G_n_p(n, p)
                V_arr.append(MVR(G))
            V_n_p_dict[(n,p)] = np.mean(V_arr)

    print(V_n_p_dict)
    df = pd.DataFrame(columns = ['n', 'p', 'violations after algorithm run'])
    for key, value in V_n_p_dict.items():
        df.loc[-1] = [key[0], key[1], value]
        df.index += 1
    df.to_csv('data2.csv')

    a = len(n_arr)
    b = len(p_arr)
    # make_plots('data.csv', a, b)
