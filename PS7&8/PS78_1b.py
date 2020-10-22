import numpy as np
from webweb import Web
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt

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
# make that happy little SBM!
edge_list = []
for i in range(1,n):
    for j in range(i):
        if np.random.rand() < omega[group_id[i]][group_id[j]]:
            edge_list.append([i,j])

G = nx.Graph()
G.add_edges_from(edge_list)


length_array = []
p = 1

for i in range(100):
    spread = True
    S_I_list = np.array(['S']*1000)
    l = 0

    while spread == True:
        spread = False
        if l == 0:
            S_I_list[np.random.randint(1000)] = 'I'

        for node in G.nodes():
            for neighbor in G.neighbors(node):
                if (S_I_list[node] == 'I') and (S_I_list[neighbor] == 'S') and (np.random.uniform() < p):
                    S_I_list[neighbor] = 'I'
                    spread = True

        l += 1

    length_array.append(l)

print('Average pandemic length for p={p}: {avg}'.format(p=p, avg=np.average(length_array)))
