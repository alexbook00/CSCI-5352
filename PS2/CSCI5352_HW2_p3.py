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

        # finds the mean degree and mean neighbor degree of each network
        totalDegree = 0
        totalDegreeSquared = 0
        for i in G.degree:
            totalDegree += i[1]
            totalDegreeSquared += i[1]**2
        averageDegree = totalDegree/len(G)
        averageNeighborDegree = (totalDegreeSquared/len(G))/averageDegree

        # neighborDict = nx.average_neighbor_degree(G)
        # totalNeighborDegree = 0
        # for key in neighborDict:
        #     totalNeighborDegree += neighborDict[key]
        # averageNeighborDegree = totalNeighborDegree/len(neighborDict)

        school = file.split('/')[1].split('.')[0]
        print(school)

        dict[school] = (averageDegree, averageNeighborDegree)

    return dict

def plot_csv(file):
    df = pd.read_csv(file)
    df.columns = ['School', 'Mean Degree', 'Mean Neighbor Degree']
    df['Ratio'] = df['Mean Neighbor Degree']/df['Mean Degree']

    fig, ax = plt.subplots()
    ax.scatter(df['Mean Degree'], df['Ratio'], alpha = .5)
    ax.hlines(1, min(df['Mean Degree'])-5, max(df['Mean Degree'])+5, color='red', label='No Paradox')
    ax.grid(True)
    ax.set_xlim(min(df['Mean Degree'])-5, max(df['Mean Degree'])+5)
    ax.set_xlabel(r'$\langle k_{u}\rangle$ (Mean Degree)', fontsize=18)
    ax.set_ylabel(r'$\langle k_{v}\rangle / \langle k_{u}\rangle$ (MND / Mean Degree)', fontsize=18)

    # only label the desired schools
    labeldf = df.loc[df['School'].isin(['Reed98', 'Bucknell39', 'Mississippi66', 'Virginia63', 'Berkeley13'])]
    for i in labeldf.index:
        ax.annotate(df['School'][i] + '\n(' + str(round(df['Mean Degree'][i],3)) + ',' + str(round(df['Ratio'][i],3)) + ')',
                    (df['Mean Degree'][i], df['Ratio'][i]))

    ax.legend()
    plt.show()

if __name__ == '__main__':
    path = 'facebook100txt'

    filepaths = []

    for f in listdir(path):
        # only select the desired files
        if ('_attr' not in f) and ('Traud' not in f) and ('readme' not in f):
            filepath = path + '/' + f
            filepaths.append(filepath)

    dict = fillDictFromFiles(filepaths)
    df = pd.DataFrame.from_dict(data=dict, orient='index')
    df.to_csv('dict_file.csv', header=['Average Degree', 'Average Neighbor Degree'])

    plot_csv('dict_file.csv')
