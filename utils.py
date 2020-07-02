import torch as t
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
import scipy.sparse as sp


def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""

    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = np.diag(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)


def preprocess_adj(adj):
    """Preprocessing of adjacency matrix for simple GCN model and conversion to tuple representation."""
    adj_normalized = normalize_adj(adj + np.eye(adj.shape[0]) + adj.T)
    return adj_normalized


def preprocess_adjs(adjs):
    adjs_normalized = []
    for adj in adjs:
        adjs_normalized.append(preprocess_adj(adj))
    return np.array(adjs_normalized)


def get_roc_score_adj(edges_pos, edges_neg, adj_rec, t_2, adjs):
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))

    # Predict on test set of edges
    # print(adj_rec)

    preds = []
    pos = []
    adj = adjs[t_2]
    # print(adj)
    for e in edges_pos:
        preds.append(sigmoid(adj_rec[e[0], e[1]]))
        pos.append(adj[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in edges_neg:
        preds_neg.append(sigmoid(adj_rec[e[0], e[1]]))
        neg.append(adj[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    # print(labels_all)
    # print(preds_all)
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def get_roc_score_feat(feas_pos, feas_neg, fea_rec, t_2, features):
    def sigmoid(x):
        x = np.clip(x, -500, 500)
        return 1.0 / (1 + np.exp(-x))
    # Predict on test set of edges
    preds = []
    pos = []
    features_orig = np.squeeze(features)[t_2]
    fea_rec = fea_rec.T
    for e in feas_pos:
        preds.append(sigmoid(fea_rec[e[0], e[1]]))
        pos.append(features_orig[e[0], e[1]])

    preds_neg = []
    neg = []
    for e in feas_neg:
        preds_neg.append(sigmoid(fea_rec[e[0], e[1]]))
        neg.append(features_orig[e[0], e[1]])

    preds_all = np.hstack([preds, preds_neg])
    labels_all = np.hstack([np.ones(len(preds)), np.zeros(len(preds_neg))])
    roc_score = roc_auc_score(labels_all, preds_all)
    ap_score = average_precision_score(labels_all, preds_all)

    return roc_score, ap_score


def mask_test_edges(adj):
    adj_row = adj.nonzero()[0]
    adj_col = adj.nonzero()[1]
    edges = []
    edges_dic = {}
    for i in range(len(adj_row)):
        edges.append([adj_row[i], adj_col[i]])
        edges_dic[(adj_row[i], adj_col[i])] = 1
    false_edges_dic = {}
    num_test = int(np.floor(len(edges) / 10.))
    num_val = int(np.floor(len(edges) / 20.))
    all_edge_idx = np.arange(len(edges))
    np.random.shuffle(all_edge_idx)
    val_edge_idx = all_edge_idx[:num_val]
    test_edge_idx = all_edge_idx[num_val:(num_val + num_test)]
    edges = np.array(edges)
    test_edges = edges[test_edge_idx]
    val_edges = edges[val_edge_idx]
    train_edges = np.delete(edges, np.hstack([test_edge_idx, val_edge_idx]), axis=0)
    # train_edges = edges
    test_edges_false = []
    val_edges_false = []
    while len(test_edges_false) < num_test or len(val_edges_false) < num_val:
        i = np.random.randint(0, adj.shape[0])
        j = np.random.randint(0, adj.shape[0])
        if (i, j) in edges_dic:
            continue
        if (j, i) in edges_dic:
            continue
        if (i, j) in false_edges_dic:
            continue
        if (j, i) in false_edges_dic:
            continue
        else:
            false_edges_dic[(i, j)] = 1
            false_edges_dic[(j, i)] = 1
        if np.random.random_sample() > 0.333:
            if len(test_edges_false) < num_test:
                test_edges_false.append((i, j))
            else:
                if len(val_edges_false) < num_val:
                    val_edges_false.append([i, j])
        else:
            if len(val_edges_false) < num_val:
                val_edges_false.append([i, j])
            else:
                if len(test_edges_false) < num_test:
                    test_edges_false.append([i, j])

    data = np.ones(train_edges.shape[0])
    adj_train = sp.csr_matrix((data, (train_edges[:, 0], train_edges[:, 1])), shape=adj.shape)
    adj_train = adj_train + adj_train.T
    return adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false


def mask_test_feas(features):
    fea_row = features.nonzero()[0]
    fea_col = features.nonzero()[1]
    feas = []
    feas_dic = {}
    for i in range(len(fea_row)):
        feas.append([fea_row[i], fea_col[i]])
        feas_dic[(fea_row[i], fea_col[i])] = 1
    false_feas_dic = {}
    num_test = int(np.floor(len(feas) / 10.))
    num_val = int(np.floor(len(feas) / 20.))
    all_fea_idx = np.arange(len(feas))
    np.random.shuffle(all_fea_idx)
    val_fea_idx = all_fea_idx[:num_val]
    test_fea_idx = all_fea_idx[num_val:(num_val + num_test)]
    feas = np.array(feas)
    test_feas = feas[test_fea_idx]
    val_feas = feas[val_fea_idx]
    train_feas = np.delete(feas, np.hstack([test_fea_idx, val_fea_idx]), axis=0)
    test_feas_false = []
    val_feas_false = []
    while len(test_feas_false) < num_test or len(val_feas_false) < num_val:
        i = np.random.randint(0, features.shape[0])
        j = np.random.randint(0, features.shape[1])
        if (i, j) in feas_dic:
            continue
        if (i, j) in false_feas_dic:
            continue
        else:
            false_feas_dic[(i, j)] = 1
        if np.random.random_sample() > 0.333:
            if len(test_feas_false) < num_test:
                test_feas_false.append([i, j])
            else:
                if len(val_feas_false) < num_val:
                    val_feas_false.append([i, j])
        else:
            if len(val_feas_false) < num_val:
                val_feas_false.append([i, j])
            else:
                if len(test_feas_false) < num_test:
                    test_feas_false.append([i, j])
    data = np.ones(train_feas.shape[0])
    fea_train = sp.csr_matrix((data, (train_feas[:, 0], train_feas[:, 1])), shape=features.shape)
    val_feas_false = np.array(val_feas_false)
    return fea_train, train_feas, val_feas, val_feas_false, test_feas, test_feas_false


def mask_adjs_test(adjs):
    adjs_train = []
    test_adjs = []
    test_adjs_negative = []
    val_adjs = []
    val_adjs_negative = []
    for adj in adjs:
        out = mask_test_edges(adj)  # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
        adjs_train.append(out[0].toarray())
        val_adjs.append(out[2])
        val_adjs_negative.append(out[3])
        test_adjs.append(out[4])
        test_adjs_negative.append(out[5])
    adjs_train = np.array(adjs_train)
    return adjs_train, val_adjs, val_adjs_negative, test_adjs, test_adjs_negative


def mask_attributes_test(attributes):
    attributes_train = []
    test_attributes = []
    test_attributes_negative = []
    val_attributes = []
    val_attributes_negative = []
    for attribute in attributes:
        out = mask_test_feas(attribute)  # adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false
        attributes_train.append(out[0].toarray())
        val_attributes.append(out[2])
        val_attributes_negative.append(out[3])
        test_attributes.append(out[4])
        test_attributes_negative.append(out[5])
    attributes_train = np.array(attributes_train)
    return attributes_train, val_attributes, val_attributes_negative, test_attributes, test_attributes_negative
