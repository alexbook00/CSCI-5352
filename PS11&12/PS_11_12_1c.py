import numpy as np
import pandas as pd
from pprint import pprint

def nodes_to_dict(filename):
    file = open(filename, 'r')

    node_dict = {}

    for line in file:
        arr = line.split()
        if '#' not in arr:
            node, date = arr
            # cleaning cross-listed papers
            if node[:2] == '11':
                node = node[2:]
            node = int(node)
            if node not in node_dict.keys():
                node_dict[node] = [date, 0, 0] # [date, out-degree, in-degree]
    file.close()
    return node_dict

def edge_list_to_array(filename, node_dict):
    file = open(filename, 'r')

    for line in file:
        arr = line.split()
        if '#' not in arr:
            out_node, in_node = arr
            out_node, in_node = int(out_node), int(in_node)
            if (out_node in node_dict.keys()) and (in_node in node_dict.keys()):
                node_dict[out_node][1] += 1
                node_dict[in_node][2] += 1

    return node_dict

if __name__ == '__main__':
    node_dict = nodes_to_dict('partc_dates')
    node_dict = edge_list_to_array('partc_edges', node_dict)

    sorted_nodes = [(k,v) for (k, v) in sorted(node_dict.items(), key=lambda x: x[1])]
    # pprint(sorted_nodes)

    # print(len(sorted_nodes))
    for tuple in sorted_nodes:
        # if both out-degree and in-degree are zero, remove the node
        if tuple[1][1] == 0 and tuple[1][2] == 0:
            sorted_nodes.remove(tuple)
    # print(len(sorted_nodes))

    degrees = 0
    count = 0
    for tuple in sorted_nodes[:len(sorted_nodes)//10]:
        degrees += tuple[1][2]
        count += 1
    first10_avg = degrees/count

    degrees = 0
    count = 0
    for tuple in sorted_nodes[len(sorted_nodes)-len(sorted_nodes)//10:]:
        degrees += tuple[1][2]
        count += 1
    last10_avg = degrees/count

    print(first10_avg, last10_avg)
