import networkx as nx
import matplotlib
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

if __name__ == '__main__':
    with open('medici_edge_list.txt') as f:
        G = nx.read_edgelist(f, delimiter=',')
    G.remove_edge('11','11')

    relabelDict = {
        '0' : 'Acciaiuoli',
        '1' : 'Albizzi',
        '2' : 'Barbadori',
        '3' : 'Bischeri',
        '4' : 'Castellani',
        '5' : 'Ginori',
        '6' : 'Guadagni',
        '7' : 'Lamberteschi',
        '8' : 'Medici',
        '9' : 'Pazzi',
        '10' : 'Peruzzi',
        '11' : 'Pucci',
        '12' : 'Ridolfi',
        '13' : 'Salviati',
        '14' : 'Strozzi',
        '15' : 'Tornabuoni'
    }
    G = nx.relabel_nodes(G, relabelDict)
    # nx.draw_networkx(G, with_labels = True)
    # plt.show()

    # degree centrality
    degreeDict = {}
    for node in G.nodes:
        degreeDict[node] = round(G.degree(node)/(len(G.nodes)-1),3)

    # harmonic centrality
    harmonicDict = {}
    for node1 in G.nodes:
        sum = 0
        for node2 in G.nodes:
            if (node1 != node2) and nx.has_path(G, node1, node2):
                sum += 1/nx.algorithms.shortest_paths.shortest_path_length(G, source=node1, target=node2)
        harmonicDict[node1] = round(sum/(len(G.nodes)-1),3)

    # eigenvector centrality
    eigenvectorDict = nx.eigenvector_centrality(G)
    for key in eigenvectorDict.keys():
        eigenvectorDict[key] = round(eigenvectorDict[key], 3)

    # betweenness centrality
    betweennessDict = nx.betweenness_centrality(G, k=len(G.nodes))
    for key in betweennessDict.keys():
        betweennessDict[key] = round(betweennessDict[key], 3)

    dfD = pd.DataFrame.from_dict(degreeDict, orient='index', columns = ['Degree Centrality']).sort_values('Degree Centrality', ascending=False)
    dfH = pd.DataFrame.from_dict(harmonicDict, orient='index', columns = ['Harmonic Centrality']).sort_values('Harmonic Centrality', ascending=False)
    dfE = pd.DataFrame.from_dict(eigenvectorDict, orient='index', columns = ['Eigenvector Centrality']).sort_values('Eigenvector Centrality', ascending=False)
    dfB = pd.DataFrame.from_dict(betweennessDict, orient='index', columns = ['Betweenness Centrality']).sort_values('Betweenness Centrality', ascending=False)
    df = pd.concat([dfD, dfH, dfE, dfB], axis=1).sort_values('Degree Centrality', ascending=False)

    # config model stuff
    thousandDict = {
        'Acciaiuoli' : [],
        'Albizzi' : [],
        'Barbadori' : [],
        'Bischeri' : [],
        'Castellani' : [],
        'Ginori' : [],
        'Guadagni' : [],
        'Lamberteschi' : [],
        'Medici' : [],
        'Pazzi' : [],
        'Peruzzi' : [],
        'Pucci' : [],
        'Ridolfi' : [],
        'Salviati' : [],
        'Strozzi' : [],
        'Tornabuoni' : []
    }
    for i in range(1000):
        G = nx.Graph(nx.configuration_model([1,3,2,3,3,1,4,1,6,1,3,0,3,2,4,3]))
        relabelDict = {
            0 : 'Acciaiuoli',
            1 : 'Albizzi',
            2 : 'Barbadori',
            3 : 'Bischeri',
            4 : 'Castellani',
            5 : 'Ginori',
            6 : 'Guadagni',
            7 : 'Lamberteschi',
            8 : 'Medici',
            9 : 'Pazzi',
            10 : 'Peruzzi',
            11 : 'Pucci',
            12 : 'Ridolfi',
            13 : 'Salviati',
            14 : 'Strozzi',
            15 : 'Tornabuoni'
        }
        G = nx.relabel_nodes(G, relabelDict)
        for node1 in G.nodes:
            sum = 0
            for node2 in G.nodes:
                if (node1 != node2) and nx.has_path(G, node1, node2):
                    sum += 1/nx.algorithms.shortest_paths.shortest_path_length(G, source=node1, target=node2)
            thousandDict[node1].append(round(sum/(len(G.nodes)-1),3))
    percentile25dict = {}
    percentile75dict = {}
    diffDict = {}
    for key, value in thousandDict.items():
        avgEnsembleCentrality = np.average(value)
        diff = harmonicDict[key] - avgEnsembleCentrality
        diffDict[key] = diff

        diffList = []
        for i in range(len(value)):
            diffList.append(harmonicDict[key]-value[i])
        percentile25dict[key] = np.percentile(diffList, 25)
        percentile75dict[key] = np.percentile(diffList, 75)

    harmonicDict = dict(sorted(harmonicDict.items()))
    fig, ax = plt.subplots()
    ax.plot(list(diffDict.keys()), list(diffDict.values()), color='blue')
    ax.plot(list(percentile25dict.keys()), list(percentile25dict.values()), 'r-', alpha=0)
    ax.plot(list(percentile75dict.keys()), list(percentile75dict.values()), 'r-', alpha=0)
    ax.fill_between(range(len(list(diffDict.values()))), list(percentile25dict.values()),list(percentile75dict.values()), alpha=.3, color='gray')
    fig.autofmt_xdate(rotation=45)
    ax.set_xlabel('Family')
    ax.set_ylabel('Difference Between Harmonic Centrality on G\n and Avg. Harmonic Centrality in Ensemble')
    ax.set_xlim(0, len(list(diffDict.values()))-1)
    plt.show()
