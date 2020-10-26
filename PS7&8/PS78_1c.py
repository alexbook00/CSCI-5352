import numpy as np
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def pandemic_simulation(G):
    spread = True
    S_I_list = np.array(['S']*200)
    length = 0
    num_infected = 0

    while spread == True:
        spread = False
        if length == 0:
            S_I_list[np.random.randint(200)] = 'I'
            num_infected += 1
        for node in G.nodes():
            if S_I_list[node] =='I':
                for neighbor in G.neighbors(node):
                    if S_I_list[neighbor] == 'S' and np.random.uniform() <= p:
                        S_I_list[neighbor] = 'I'
                        spread = True
                        num_infected += 1
        length += 1
    return length, num_infected/200

def make_plots(filename, a, b):
    dfNew = pd.read_csv(filename)

    X = list(dfNew['epsilon'])
    Y = list(dfNew['p'])
    Z = list(dfNew['length'])
    x = np.reshape(X, (a, b))
    y = np.reshape(Y, (a, b))
    z = np.reshape(Z, (a, b))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = ax.plot_surface(x, y, z, cmap = cm.coolwarm)
    fig.colorbar(q)
    ax.set_xlabel('epsilon')
    ax.set_ylabel('Probability of spreading')
    ax.set_zlabel('Length of pandemic (days)')
    ax.view_init(15, 15)
    plt.show()

    X = list(dfNew['epsilon'])
    Y = list(dfNew['p'])
    Z = list(dfNew['size'])
    x = np.reshape(X, (a, b))
    y = np.reshape(Y, (a, b))
    z = np.reshape(Z, (a, b))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = ax.plot_surface(x, y, z, cmap = cm.coolwarm)
    fig.colorbar(q)
    ax.set_xlabel('epsilon')
    ax.set_ylabel('Probability of spreading')
    ax.set_zlabel('Size of pandemic (fraction of infected individuals)')
    ax.view_init(15, -15)
    plt.show()

if __name__ == '__main__':
    B = 2
    n1 = 100
    n2 = 100
    n = n1 + n2
    group_id = np.array([0]*n1 + [1]*n2)
    c = 8
    epsilon_array = [0, .025*c, .05*c, .075*c, 0.1*c, .125*c, .15*c, .175*c,
                  .2*c, .225*c, .25*c, .275*c, 0.3*c, .325*c, .35*c, .375*c,
                  .4*c, .425*c, .45*c, .475*c, 0.5*c, .525*c, .55*c, .575*c,
                  .6*c, .625*c, .65*c, .675*c, 0.7*c, .725*c, .75*c, .775*c,
                  .8*c, .825*c, .85*c, .875*c, 0.9*c, .925*c, .95*c, .975*c,
                 1.0*c, 1.025*c, 1.05*c, 1.075*c, 1.1*c, 1.125*c, 1.15*c, 1.175*c,
                 1.2*c, 1.225*c, 1.25*c, 1.275*c, 1.3*c, 1.325*c, 1.35*c, 1.375*c,
                 1.4*c, 1.425*c, 1.45*c, 1.475*c, 1.5*c, 1.525*c, 1.55*c, 1.575*c,
                 1.6*c, 1.625*c, 1.65*c, 1.675*c, 1.7*c, 1.725*c, 1.75*c, 1.775*c,
                 1.8*c, 1.825*c, 1.85*c, 1.875*c, 1.9*c, 1.925*c, 1.95*c, 1.975*c, 2*c]

    p_array = [0, .005, .01, .015, .02, .025, .03, .035, .04, .045,
             .05, .055, .06, .065, .07, .075, .08, .085, .09, .095,
              .1, .125, .15, .175, .2, .225, .25, .275, .3, .325, .35, .375,
              .4, .425, .45, .475, .5, .525, .55, .575, .6, .625, .65, .675,
              .7, .725, .75, .775, .8, .825, .85, .875, .9, .925, .95, .975, 1]

    length_size_dict = {}
    for epsilon in epsilon_array:
        print(epsilon)
        for p in p_array:
            length_array = []
            size_array = []
            for i in range(100):

                within = (2*c + epsilon)/(2*n)
                between = (2*c - epsilon)/(2*n)
                omega = [[within,between,0],
                         [between,within,between],
                         [0,between,within]]

                edge_list = []
                for i in range(200):
                    for j in range(i):
                        if np.random.rand() < omega[group_id[i]][group_id[j]]:
                            edge_list.append([i,j])

                G = nx.Graph()
                G.add_edges_from(edge_list)

                length, size = pandemic_simulation(G)
                length_array.append(length)
                size_array.append(size)
            length_size_dict[(epsilon, p)] = (np.average(length_array), np.average(size_array))

    df = pd.DataFrame(columns = ['epsilon', 'p', 'length', 'size'])
    for key, value in length_size_dict.items():
        df.loc[-1] = [key[0], key[1], value[0], value[1]]
        df.index += 1
    df.to_csv('overnight_data.csv')

    # a = len(epsilon_array)
    # b = len(p_array)
    # make_plots('almost_there.csv', a, b)
