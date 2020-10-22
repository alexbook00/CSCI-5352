import numpy as np
from webweb import Web
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt

# choose the number of groups
B = 2
# choose the sizes of groups
n1 = 25
n2 = 25
n = n1+n2
# group memberships b
group_id = np.array([0]*n1 + [1]*n2)
# block matrix, omega
epsilon_list = [0, 4, 8]
for epsilon in epsilon_list:
    within = (10 + epsilon)/(2*n)
    between = (10 - epsilon)/(2*n)
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
    pos_dict = nx.drawing.layout.spring_layout(G)

    nx.draw_networkx_nodes(G, pos_dict, nodelist=np.arange(25), node_color='r')
    nx.draw_networkx_nodes(G, pos_dict, nodelist=np.arange(25, 50), node_color='b')
    nx.draw_networkx_edges(G, pos_dict)

    title = 'epsilon = ' + str(epsilon)
    plt.title(title)
    plt.show()
