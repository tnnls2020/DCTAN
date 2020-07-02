import torch as t
import numpy as np
import utils
import networkx as nx
import scipy.sparse as sp


def view_network(adjs):
    """ print the details of a dynamic network """
    for index, adj in enumerate(adjs):
        print(index,np.array(np.array(adj).nonzero()).shape)

    temp = np.full(adjs[0].shape, 0)
    out = []
    for i in range(len(adjs)):
        temp += adjs[i].astype(int)
    for i in range(len(adjs)):
        out.append((np.array(np.where(temp==i)).shape))
    print(out)


def view_features(features):
    for i, feature in enumerate(features):
        print(i, len(np.nonzero(features[i])[0]))
        if i != 0:
            print(i, len(np.nonzero(features[i] + features[i - 1])[0]))


def load_tn_wt(dataset):

    edge_file = open("data/{}.edge".format(dataset), 'r')
    edges = edge_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    print("node number:{} , edge_number:{}".format(node_num, edge_num))
    edges.pop(0)
    edges.pop(0)

    max_time = 0
    max_node = 0
    min_time = float("inf")
    for line in edges:
        timestamp = int(line.split(' ')[2].strip())
        node = int(line.split(' ')[1].strip())
        if timestamp > max_time:
            max_time = timestamp
        if timestamp < min_time:
            min_time = timestamp
        if node > max_node:
            max_node = node

    period = 20
    print("max_node:"+str(max_node))
    if dataset == "email1" or dataset == "email" or dataset == "email_all":
        period = 40

    gap = float((max_time+1-min_time)/period)

    print("min_time:{},max_time{},period:{},step_length:{}".format(str(min_time),
                                                           str(max_time),
                                                           str(period),
                                                           str(gap)))

    adjs_shape = [period, node_num, node_num]
    features_shape = [period, node_num]
    adjs = np.zeros(adjs_shape)
    features = np.zeros(features_shape)
    index = 0
    for line in edges:
        node1 = int(line.split(' ')[0].strip())-1
        node2 = int(line.split(' ')[1].strip())-1
        timestamp = int(line.split(' ')[2].strip())
        time = int((timestamp-min_time)/gap)
        adjs[time][node1][node2] = 1.0
        features[time][node1] = 1.0
        features[time][node2] = 1.0
        # print(index, time)
        index = index + 1

    save_file_name = dataset+'.edge'
    # with open(save_file_name, 'wb') as fp:
    #     pickle.dump(adjs, fp)
    if dataset == "email":
        adjs = adjs[0:20]
        features = features[0:20]
        # adjs = adjs
    view_network(adjs)
    return adjs, features


def load_adjs_and_attrs(dataset):
    """ initialize return tensors"""
    adjs = []
    attributes = []

    """ set the start and end of the period """
    start = 0
    end = 0

    if "DBLP" in dataset:
        start = 2005
        end = 2019

    for time in range(start, end):
        print("loading time {}".format(time))
        edge_path = "data/{}/{}.edge.{}".format(dataset, dataset, time)
        attri_path = "data/{}/{}.attr.{}".format(dataset, dataset, time)

        """ file_path """
        adj, attribute = load_AN(dataset, edge_path, attri_path)
        adjs.append(adj.todense())
        attributes.append(attribute.todense())

    adjs = np.array(adjs)
    attributes = np.array(attributes)
    return adjs, attributes


def load_AN(dataset, edge_path, attri_file):
    edge_file = open(edge_path, 'r')
    attri_file = open(attri_file, 'r')
    edges = edge_file.readlines()
    attributes = attri_file.readlines()
    node_num = int(edges[0].split('\t')[1].strip())
    edge_num = int(edges[1].split('\t')[1].strip())
    attribute_number = int(attributes[0].split('\t')[1].strip())
    association_number = int(attributes[1].split('\t')[1].strip())
    print("dataset:{}, node_num:{},edge_num:{},attribute_num:{},association_num:{}". \
          format(dataset, node_num, edge_num, attribute_number, association_number))
    edges.pop(0)
    edges.pop(0)
    attributes.pop(0)
    attributes.pop(0)
    adj_row = []
    adj_col = []

    for line in edges:
        node1 = int(line.split('\t')[0].strip())
        node2 = int(line.split('\t')[1].strip())
        adj_row.append(node1)
        adj_col.append(node2)
    adj = sp.csc_matrix((np.ones(edge_num), (adj_row, adj_col)), shape=(node_num, node_num))

    att_row = []
    att_col = []
    for line in attributes:
        node1 = int(line.split('\t')[0].strip())
        attribute1 = int(line.split('\t')[1].strip())
        att_row.append(node1)
        att_col.append(attribute1)
    attribute = sp.csc_matrix((np.ones(len(att_row)), (att_row, att_col)), shape=(node_num, attribute_number))
    return adj, attribute


def load_data(dataset):
    print("load {}".format(dataset))
    adjs, attributes = None, None
    if dataset == "email1" or dataset == "email" or dataset == "email_all" or dataset == "collegeMsg":
        adjs, attributes = load_tn_wt(dataset)
    elif "DBLP" in dataset:
        adjs, attributes = load_adjs_and_attrs(dataset)
    return adjs, attributes


# adjs, attributes = load_data("DBLP_sub")
# print(attributes.shape)
# adjs_train, val_adjs, val_adjs_negative, test_adjs, test_adjs_negative = utils.mask_adjs_test(adjs)
# attributes_train, val_attributes, val_attributes_negative, \
# test_attributes, test_attributes_negative = utils.mask_attributes_test(attributes)

# for time in range(0, 14):
#     print("time:{}".format(time))
#     print(len(val_adjs[time]), len(val_adjs_negative[time]), len(test_adjs[time]), len(test_adjs_negative[time]))
#     print(len(val_attributes[time]), len(val_attributes_negative[time]), len(test_attributes[time]), len(test_attributes_negative[time]))


