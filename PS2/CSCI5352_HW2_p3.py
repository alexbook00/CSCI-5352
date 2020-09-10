import numpy as np
import networkx as nx
from os import listdir
from pprint import pprint
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd

def fillDictFromFiles(fileList):
    dict = {}

    for file in filepaths:
        G = nx.read_edgelist(file)

        totalDegree = 0
        nodes = 0
        for i in G.degree:
            totalDegree += i[1]
        averageDegree = totalDegree/len(G)

        neighborDict = nx.average_neighbor_degree(G)
        totalNeighborDegree = 0
        for key in neighborDict:
            totalNeighborDegree += neighborDict[key]
        averageNeighborDegree = totalNeighborDegree/len(neighborDict)

        school = file.split('/')[1].split('.')[0]
        print(school)

        dict[school] = (averageDegree, averageNeighborDegree)

    return dict

def plot_csv(file):
    df = pd.read_csv(file)
    df.columns = ['School', 'Average Degree', 'Average Neighbor Degree']
    df['Ratio'] = df['Average Neighbor Degree']/df['Average Degree']

    fig, ax = plt.subplots()
    ax.scatter(df['Average Degree'], df['Ratio'])
    ax.hlines(1, min(df['Average Degree'])-5, max(df['Average Degree'])+5, color='red', label='No Paradox')
    ax.grid(True)
    ax.set_xlim(min(df['Average Degree'])-5, max(df['Average Degree'])+5)
    ax.set_xlabel(r'$\langle k_{u}\rangle$', fontsize=18)
    ax.set_ylabel(r'$\langle k_{v}\rangle / \langle k_{u}\rangle$', fontsize=18)

    # only label the desired schools
    labeldf = df.loc[df['School'].isin(['Reed98', 'Bucknell39', 'Mississippi66', 'Virginia63', 'Berkeley13'])]
    for i in labeldf.index:
        ax.annotate(df['School'][i], (df['Average Degree'][i], df['Ratio'][i]))

    ax.legend()
    plt.show()

if __name__ == '__main__':
    path = 'facebook100txt'

    filepaths = []

    for f in listdir(path):

        # only select the desired files
        if ('_attr' not in f) and ('Traud' not in f) and ('readme' not in f):
            fp = path + '/' + f
            filepaths.append(fp)

    # dict = fillDictFromFiles(filepaths)
    # df = pd.DataFrame.from_dict(data=dict, orient='index')
    # df.to_csv('dict_file.csv', header=['Average Degree', 'Average Neighbor Degree'])

    plot_csv('dict_file.csv')
