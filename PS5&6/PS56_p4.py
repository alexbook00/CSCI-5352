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

    # get list of edge list files
    path = 'facebook100txt'
    edge_list_filepaths = []
    for f in listdir(path):
        # only select edge list files
        if ('_attr' not in f) and ('readme' not in f) and ('pdf' not in f) and ('.DS_Store' not in f):
            fp = os.path.join(path, f)
            # test if file actually exists
            if os.path.isfile(fp):
                edge_list_filepaths.append(fp)
    edge_list_filepaths = sorted(edge_list_filepaths)

    # get list of attribute files
    path = 'facebook100txt'
    attr_filepaths = []
    for f in listdir(path):
        # only select attribute files
        if '_attr' in f:
            fp = os.path.join(path, f)
            # test if file actually exists
            if os.path.isfile(fp):
                attr_filepaths.append(fp)
    attr_filepaths = sorted(attr_filepaths)

    modList = []

    # print(len(edge_list_filepaths), len(attr_filepaths))
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

        modList.append([attr_filepaths[i].split('_')[0].split('\\')[1], Q_status, Q_major, Q_degree])

    # pprint(modList)
    df = pd.DataFrame(modList, columns=['School', 'Modularity by status', 'Modularity by major', 'Modularity by degree'])
    df.to_csv('df_out.csv')
