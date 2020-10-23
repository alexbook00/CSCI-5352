import numpy as np
from webweb import Web
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd

# choose the number of groups
B = 2
# choose the sizes of groups
n1 = 500
n2 = 500
n = n1+n2
# group memberships b
group_id = np.array([0]*n1 + [1]*n2)
# block matrix, omega
epsilon = 0
c = 8
within = (2*c + epsilon)/(2*n)
between = (2*c - epsilon)/(2*n)
omega = [
    [within,between,0],
    [between,within,between],
    [0,between,within]]
# # make that happy little SBM!
# edge_list = []
# for i in range(1,n):
#     for j in range(i):
#         if np.random.rand() < omega[group_id[i]][group_id[j]]:
#             edge_list.append([i,j])
#
# G = nx.Graph()
# G.add_edges_from(edge_list)

p_array = [.01, .02, .03, .04, .05, .06, .07, .08, .09, .1,
           .11, .12, .13, .14, .15, .16, .17, .18, .19, .2,
           .21, .22, .23, .24, .25, .26, .27, .28, .29, .3,
           .4, .5, .6, .7, .8, .9]

length_size_dict = {} # key is p, value is (average length, average size) tuple

for p in p_array:

    # each iteration is one simulation of a pandemic
    length_array = []
    size_array = []

    for i in range(100):
        # make that happy little SBM!
        edge_list = []
        for i in range(1,n):
            for j in range(i):
                if np.random.rand() < omega[group_id[i]][group_id[j]]:
                    edge_list.append([i,j])

        G = nx.Graph()
        G.add_edges_from(edge_list)

        spread = True
        S_I_list = np.array(['S']*1000)
        l = 0
        num_infected = 0

        while spread == True:
            spread = False
            if l == 0:
                S_I_list[np.random.randint(1000)] = 'I'
                num_infected += 1

            for node in G.nodes():
                for neighbor in G.neighbors(node):
                    if (S_I_list[node] == 'I') and (S_I_list[neighbor] == 'S') and (np.random.uniform() < p):
                        S_I_list[neighbor] = 'I'
                        spread = True
                        num_infected += 1

            l += 1

        length_array.append(l)
        size_array.append(num_infected/len(S_I_list))

    length_size_dict[p] = (np.average(length_array), np.average(size_array))

# print('Average pandemic length for p={p}: {avg}'.format(p=p, avg=np.average(length_array)))
# pprint(length_size_dict)
df = pd.DataFrame.from_dict(length_size_dict)
df.T.to_csv('pandemic_data.csv')
