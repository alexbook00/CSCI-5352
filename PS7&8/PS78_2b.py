import numpy as np
import networkx as nx
from pprint import pprint
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib import cm

def get_neighbors(A, x, y):
    changes = [(1, 0), (0, 1), (-1, 0), (0, -1)]
    reachable = []
    for pair in changes:
        if (x+pair[0] < len(A)) and (x+pair[0] >= 0) and (y+pair[1] < len(A)) and (y+pair[1] >= 0):
            reachable.append((x+pair[0], y+pair[1]))
    return reachable

def get_random_connections(q):
    A = [[0 for x in range(100)] for y in range(100)]

    for i in range(100):
        for j in range(100):
            if np.random.uniform() <= q:
                A[i][j] = 1

    return A

def pandemic_simulation(p, connections):
    S_I_grid = [['S' for x in range(10)] for y in range(10)]

    spread = True
    length = 0
    num_infected = 0

    while spread is True:
        if num_infected == 100:
            break
        copy_grid = S_I_grid.copy()
        spread = False
        if length == 0:
            copy_grid[np.random.randint(10)][np.random.randint(10)] = 'I'
            num_infected += 1
        for i in range(10):
            for j in range(10):
                if S_I_grid[i][j] == 'I':
                    for coords in get_neighbors(S_I_grid, i, j):
                        if S_I_grid[coords[0]][coords[1]] == 'S' and np.random.uniform() <= p:
                            copy_grid[coords[0]][coords[1]] = 'I'
                            spread = True
                            num_infected += 1
                    for x in range(len(connections[10*i + j])):
                        if connections[10*i + j] == 1:
                            a, b = str(x).split()
                            if S_I_grid[int(a)][int(b)] == 'S' and np.random.uniform() <= p:
                                copy_grid[int(a)][int(b)] = 'I'
                                spread = True
                                num_infected += 1

        length += 1
        S_I_grid = copy_grid.copy()
    return length, num_infected/2500

def make_plots(filename, a, b):
    dfNew = pd.read_csv(filename)

    X = list(dfNew['p'])
    Y = list(dfNew['q'])
    Z = list(dfNew['length'])
    x = np.reshape(X, (a, b))
    y = np.reshape(Y, (a, b))
    z = np.reshape(Z, (a, b))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = ax.plot_surface(x, y, z, cmap = cm.coolwarm)
    fig.colorbar(q)
    ax.set_xlabel('Probability of spreading')
    ax.set_ylabel('q (chance of random connections)')
    ax.set_zlabel('Length of pandemic (days)')
    ax.view_init(15, -15)
    plt.show()

    X = list(dfNew['p'])
    Y = list(dfNew['q'])
    Z = list(dfNew['size'])
    x = np.reshape(X, (a, b))
    y = np.reshape(Y, (a, b))
    z = np.reshape(Z, (a, b))
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    q = ax.plot_surface(x, y, z, cmap = cm.coolwarm)
    fig.colorbar(q)
    ax.set_xlabel('Probability of spreading')
    ax.set_ylabel('q (chance of random connections)')
    ax.set_zlabel('Size of pandemic (fraction of infected individuals)')
    ax.view_init(15, -15)
    plt.show()

if __name__ == '__main__':
    p_array = [0, .0025, .005, .0075, .01, .0125, .015, .0175, .02,
                  .0225, .025, .0275, .03, .0325, .035, .0375, .04,
                  .0425, .045, .0475, .05, .0525, .055, .0575, .06,
                  .0625, .065, .0675, .07, .0725, .075, .0775, .08,
                  .0825, .085, .0875, .09, .0925, .095, .0975, .1,
                  .125, .15, .175, .2, .225, .25, .275, .3,
                  .325, .35, .375, .4, .425, .45, .475, .5,
                  .525, .55, .575, .6, .625, .65, .675, .7,
                  .725, .75, .775, .8, .825, .85, .875, .9,
                  .925, .95, .975, 1]

    q_array = [0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1]

    length_size_dict = {}

    for p in p_array:
        for q in q_array:
            print(p)
            length_array = []
            size_array = []
            for i in range(50):
                connections = get_random_connections(q)
                length, size = pandemic_simulation(p, connections)
                length_array.append(length)
                size_array.append(size)
            length_size_dict[(p, q)] = (np.average(length_array), np.average(size_array))

    df = pd.DataFrame(columns = ['p', 'q', 'length', 'size'])
    for key, value in length_size_dict.items():
        df.loc[-1] = [key[0], key[1], value[0], value[1]]
        df.index += 1
    df.to_csv('TENBYTEN_0025UpToPoint1_025UpTo1_50iter.csv')

    a = len(p_array)
    b = len(q_array)
    make_plots('TENBYTEN_0025UpToPoint1_025UpTo1_50iter.csv', a, b)
