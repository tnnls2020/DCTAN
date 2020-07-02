import numpy as np
import scipy.sparse as sp
import pandas as pd
from collections import Counter
import load_data


def trans_to_can(dataset):
    adjs, features = load_data.load_tn_wt(dataset)
    edge_file = "data_can/{}_can.edge".format(dataset)
    attr_file = "data_can/{}_can.node".format(dataset)
    node_num = adjs.shape[1]

    # sum the adjs
    adj_sum = np.zeros([node_num, node_num])
    for adj in adjs:
        adj_sum += adj

    with open(edge_file, "w") as fp:
        edges = np.nonzero(adj_sum)
        for index in range(len(edges[0])):
            # print(edge)
            line = str(edges[0][index]) + "\t" + str(edges[1][index]) + "\n"
            fp.write(line)

    with open(attr_file, "w") as fp:
        edges = np.nonzero(features)
        for index in range(len(edges[0])):
            line = str(edges[1][index]) + "\t" + str(edges[0][index]) + "\n"
            fp.write(line)
    return True


def trans_to_gae(dataset):
    adjs, features = load_data.load_tn_wt(dataset)
    edge_file = "data_gae/{}_gae.edge".format(dataset)
    attr_file = "data_gae/{}_gae.node".format(dataset)
    node_num = adjs.shape[1]

    # sum the adjs
    adj_sum = np.zeros([node_num, node_num])
    for adj in adjs:
        adj_sum += adj

    with open(edge_file, "w") as fp:
        edges = np.nonzero(adj_sum)
        for index in range(len(edges[0])):
            # print(edge)
            line = str(edges[0][index]) + "\t" + str(edges[1][index]) + "\n"
            fp.write(line)

    with open(attr_file, "w") as fp:
        edges = np.nonzero(features)
        for index in range(len(edges[0])):
            line = str(edges[1][index]) + "\t" + str(edges[1][index]) + "\n"
            fp.write(line)
    return True


trans_to_can("email")
trans_to_gae("email")



