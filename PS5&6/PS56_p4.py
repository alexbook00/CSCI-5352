import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import networkx.algorithms.community.quality as nx_qual
from pprint import pprint
import os
from os import listdir

# Using  the  FB100  networks,  investigate  the  assortativity  patterns  for  three  vertexattributes:
# (i) student/faculty status, (ii) major, and (iii) vertex degree.
# Treat these networks as simple graphs in your analysis.
if __name__ == '__main__':

    path = 'facebook100txt'
    edge_list_filepaths = []
    for f in listdir(path):
        if ('_attr' not in f) and ('readme' not in f) and ('pdf' not in f) and ('.DS_Store' not in f):
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                edge_list_filepaths.append(fp)
    edge_list_filepaths = sorted(edge_list_filepaths)

    path = 'facebook100txt'
    attr_filepaths = []
    for f in listdir(path):
        if '_attr' in f:
            fp = os.path.join(path, f)
            if os.path.isfile(fp):
                attr_filepaths.append(fp)
    attr_filepaths = sorted(attr_filepaths)

    modList = []

    for i in range(len(edge_list_filepaths)):
        print(attr_filepaths[i])

        with open(edge_list_filepaths[i]) as f:
            G = nx.read_edgelist(f, nodetype=int)

        attrDF = pd.read_csv(attr_filepaths[i], sep='\t')
        attrDF.index += 1

        degreeList = sorted(list(G.degree()))
        degreeList = [i for _,i in degreeList]
        attrDF['degree'] = degreeList

        statuses = attrDF['status'].unique()
        status_set = []
        for j in statuses:
            status_set.append(attrDF.index[attrDF['status'] == j].tolist())
        Q_status = nx_qual.modularity(G, status_set)

        majors = attrDF['major'].unique()
        major_set = []
        for j in majors:
            major_set.append(attrDF.index[attrDF['major'] == j].tolist())
        Q_major = nx_qual.modularity(G, major_set)

        degrees = attrDF['degree'].unique()
        degree_set = []
        for j in degrees:
            degree_set.append(attrDF.index[attrDF['degree'] == j].tolist())
        Q_degree = nx_qual.modularity(G, degree_set)

        network_size = len(attrDF.index)

        modList.append([attr_filepaths[i].split('_')[0].split('\\')[1], Q_status, Q_major, Q_degree, network_size])

    df = pd.DataFrame(modList, columns=['School', 'Modularity by status', 'Modularity by major', 'Modularity by degree', 'Network size'])
    df.to_csv('df_out.csv')

    dfNew = pd.read_csv('df_out.csv')

    fig, ax = plt.subplots()
    ax.scatter(dfNew['Network size'], dfNew['Modularity by status'], alpha = .9)
    ax.hlines(0, 1e2, 1e5, alpha = .4)
    ax.grid(True)
    ax.set_xlim(1e2, 1e5)
    ax.set_xlabel('Network size, n', fontsize=18)
    ax.set_ylabel('Modularity by status, Q', fontsize=18)
    ax.set_xscale('log')
    plt.show()

    fig, ax = plt.subplots()
    dfNew['Modularity by status'].plot(kind='density')
    ax.grid(True)
    ax.set_xlabel('Modularity by status, Q', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.vlines(0, -1000, 10000, alpha = .4)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(dfNew['Network size'], dfNew['Modularity by major'], alpha = .9)
    ax.hlines(0, 1e2, 1e5, alpha = .4)
    ax.grid(True)
    ax.set_xlim(1e2, 1e5)
    ax.set_xlabel('Network size, n', fontsize=18)
    ax.set_ylabel('Modularity by major, Q', fontsize=18)
    ax.set_xscale('log')
    plt.show()

    fig, ax = plt.subplots()
    dfNew['Modularity by major'].plot(kind='density')
    ax.grid(True)
    ax.set_xlabel('Modularity by major, Q', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.vlines(0, -1000, 10000, alpha = .4)
    plt.show()

    fig, ax = plt.subplots()
    ax.scatter(dfNew['Network size'], dfNew['Modularity by degree'], alpha = .9)
    ax.hlines(0, 1e2, 1e5, alpha = .4)
    ax.grid(True)
    ax.set_xlim(1e2, 1e5)
    ax.set_xlabel('Network size, n', fontsize=18)
    ax.set_ylabel('Modularity by node degree, Q', fontsize=18)
    ax.set_xscale('log')
    plt.show()

    fig, ax = plt.subplots()
    dfNew['Modularity by degree'].plot(kind='density')
    ax.grid(True)
    ax.set_xlabel('Modularity by node degree, Q', fontsize=18)
    ax.set_ylabel('Density', fontsize=18)
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.vlines(0, -1000, 10000, alpha = .4)
    plt.show()
