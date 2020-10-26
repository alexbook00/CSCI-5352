import numpy as np
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd

def generate_omega_and_groupid():
    B = 2
    n1 = 500
    n2 = 500
    n = n1 + n2
    group_id = np.array([0]*n1 + [1]*n2)
    epsilon = 0
    c = 8
    within = (2*c + epsilon)/(2*n)
    between = (2*c - epsilon)/(2*n)
    omega = [[within,between,0],
             [between,within,between],
             [0,between,within]]

    return omega, group_id

def generate_graph(omega, group_id):
    edge_list = []
    for i in range(1000):
        for j in range(i):
            if np.random.rand() < omega[group_id[i]][group_id[j]]:
                edge_list.append([i,j])

    G = nx.Graph()
    G.add_edges_from(edge_list)

    return G

def pandemic_simulation(G):
    spread = True
    S_I_list = np.array(['S']*1000)
    length = 0
    num_infected = 0

    while spread == True:
        spread = False
        if length == 0:
            S_I_list[np.random.randint(1000)] = 'I'
            num_infected += 1
        for node in G.nodes():
            if S_I_list[node] =='I':
                for neighbor in G.neighbors(node):
                    if S_I_list[neighbor] == 'S' and np.random.uniform() <= p:
                        S_I_list[neighbor] = 'I'
                        spread = True
                        num_infected += 1
        length += 1
    return length, num_infected/1000

def make_plots(filename):
    dfNew = pd.read_csv(filename)
    dfNew.columns = ['p', 'length', 'size']

    dfNew.plot(x = 'p', y = 'length')
    plt.xlabel('Probability of spreading')
    plt.ylabel('Length of pandemic (days)')
    plt.hlines(np.log(1000), 0, 1, colors='purple', linestyles='dotted')
    plt.vlines(.03, 0, 12, colors='red')
    plt.show()

    dfNew.plot(x = 'p', y = 'size')
    plt.xlabel('Probability of spreading')
    plt.ylabel('Size of pandemic (fraction of infected individuals)')
    plt.vlines(.03, 0, 1, colors='red')
    plt.show()

if __name__ == '__main__':
    omega, group_id = generate_omega_and_groupid()
    length_size_dict = {}
    p_array = [0, .0025, .005, .0075, .01, .0125, .015, .0175,
               .02, .0225, .025, .0275, .03, .0325, .035, .0375,
               .04, .0425, .045, .0475, .05, .0525, .055, .0575,
               .06, .0625, .065, .0675, .07, .0725, .075, .0775,
               .08, .0825, .085, .0875, .09, .0925, .095, .0975,
               .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375,
               .4, .5, .6, .7, .8, .9, 1]

    for p in p_array:
        print(p)
        length_array = []
        size_array = []
        for i in range(100):
            G = generate_graph(omega, group_id)
            length, size = pandemic_simulation(G)
            length_array.append(length)
            size_array.append(size)
        length_size_dict[p] = (np.average(length_array), np.average(size_array))

    df = pd.DataFrame.from_dict(length_size_dict)
    df.T.to_csv('data.csv')

    make_plots('data.csv')
