import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import networkx.algorithms.community as nx_comm
from pprint import pprint

def e_r(G, r, p):
    ret = 0
    for i in G.nodes():
        for j in G.nodes():
            if G.has_edge(i,j):
                if groupDict[i] == r and groupDict[j] == p:
                    ret += 1
    return ret/(2*len(G.edges()))

def a_r(G, r):
    ret = 0
    for i in G.nodes():
        if groupDict[i] == r:
            ret += G.degree[i]
    return ret/(2*len(G.edges()))

def modularity(G):
    ret = 0
    for group in set(groupDict.values()):
        ret += e_r(G, group, group) - a_r(G, group)**2
    return ret

def prob_x_y(x, y, partition1, partition2):
    # x and and y are ints representing groups in p1 and p2, partition1 and partition2 are dicts of node,group pairs in p1 and p2
    # probability of a given node being in group x of one partition and group y of another
    matches = 0
    for node in partition1.keys():
        if partition1[node] == x and partition2[node] == y:
            matches += 1
    return matches/len(partition1)

def prob_x(x, partition):
    # probability of a given node being in group x of a partition
    size_x = 0
    for node,group in partition.items():
        if group == x:
            size_x += 1
    return size_x/len(partition)

def entropy(partition):
    ret = 0
    for g in set(partition.values()):
        size = 0
        for node,group in partition.items():
            if group == g:
                size += 1
        p_i = size/len(partition)
        ret += -p_i*np.log(p_i)
    return ret

def nmi(c, cprime):
    i_c_cprime = 0
    for x in set(c.values()):
        for y in set(cprime.values()):
            numerator = prob_x_y(x, y, c, cprime)
            denominator = prob_x(x, c)*prob_x(y, cprime)
            # print('#######################')
            # print(numerator, denominator)
            if denominator != 0 and numerator != 0:
                i_c_cprime += prob_x_y(x, y, c, cprime)*np.log(numerator/denominator)
    return (2*i_c_cprime)/(entropy(c)+entropy(cprime))

if __name__ == '__main__':
    with open('karate_edges_77.txt') as f:
        G = nx.read_edgelist(f)

    # key is vertex, value is what group that vertex is part of
    groupDict = {
        '1' : '1',
        '2' : '2',
        '3' : '3',
        '4' : '4',
        '5' : '5',
        '6' : '6',
        '7' : '7',
        '8' : '8',
        '9' : '9',
        '10' : '10',
        '11' : '11',
        '12' : '12',
        '13' : '13',
        '14' : '14',
        '15' : '15',
        '16' : '16',
        '17' : '17',
        '18' : '18',
        '19' : '19',
        '20' : '20',
        '21' : '21',
        '22' : '22',
        '23' : '23',
        '24' : '24',
        '25' : '25',
        '26' : '26',
        '27' : '27',
        '28' : '28',
        '29' : '29',
        '30' : '30',
        '31' : '31',
        '32' : '32',
        '33' : '33',
        '34' : '34'
    }

    Q = modularity(G)

    # key is number of merges, value is modularity
    modDict = {
        0 : Q
    }
    merges = 0

    Q_max = -np.inf
    Q_max_groups = None

    while len(set(groupDict.values())) > 1:
        merges += 1
        # analyze all possible merges, perform best one, add new modularity to modDict
        delta_Q = -np.inf
        for u in set(groupDict.values()):
            for v in set(groupDict.values()):
                if u != v:
                    currentDelta = 2*(e_r(G, u, v) - a_r(G, u)*a_r(G, v))
                    if currentDelta > delta_Q:
                        delta_Q = currentDelta
                        lowerMerged = min(int(u), int(v))
                        upperMerged = max(int(u), int(v))

        for key in groupDict.keys():
            if int(groupDict[key]) == upperMerged:
                groupDict[key] = str(lowerMerged)
            if int(groupDict[key]) > upperMerged:
                groupDict[key] = str(int(groupDict[key])-1)

        Q += delta_Q
        modDict[merges] = Q

        if Q > Q_max:
            Q_max = Q
            Q_max_groups = dict(groupDict)

    # plot showing the modularity score as a function of the number of merges
    lists = sorted(modDict.items())
    x, y = zip(*lists)
    plt.plot(x, y)
    plt.xlabel("Number of merges")
    plt.ylabel("Modularity score")
    plt.show()

    # visualization showing the grouping with max modularity
    color_map = []
    for i in sorted(Q_max_groups):
        if Q_max_groups[i] == '1':
            color_map.append('red')
        elif Q_max_groups[i] == '2':
            color_map.append('green')
        elif Q_max_groups[i] == '3':
            color_map.append('blue')
    nx.draw(G, node_color=color_map, with_labels=True)
    plt.show()

    karate_groups_dict = {}
    with open('karate_groups.txt') as f:
        for line in f:
            arr = line.strip().split('\t')
            karate_groups_dict[arr[0]] = arr[1]

    print(nmi(Q_max_groups, karate_groups_dict))
