import scipy.sparse as sp
import numpy as np


def sample_sub_snapshots(dataset="DBLP"):
    # set the number of nodes and attributes
    node_num = 10000
    attr_num = 5000

    node_ids = np.arange(0, node_num)
    attr_ids = np.arange(0, attr_num)

    dataset_new = dataset + "_10000"

    attidx_path = "data/{}/{}.attidx".format(dataset, dataset)
    nodidx_path = "data/{}/{}.nodidx".format(dataset, dataset)
    attidx_outpath = "data/{}/{}.attidx".format(dataset_new, dataset_new)
    nodidx_outpath = "data/{}/{}.nodidx".format(dataset_new, dataset_new)
    attidx_file = open(attidx_path, "rb")
    nodidx_file = open(nodidx_path, "rb")
    attidx_outfile = open(attidx_outpath, "wb")
    nodidx_outfile = open(nodidx_outpath, "wb")
    attidx = attidx_file.readlines()
    nodidx = nodidx_file.readlines()

    attidx = np.array(attidx)
    nodidx = np.array(nodidx)
    attidx = attidx[attr_ids]
    nodidx = nodidx[node_ids]

    # write sub index file
    for line in attidx:
        attidx_outfile.write(line)

    for line in nodidx:
        nodidx_outfile.write(line)

    start = 2005
    end = 2018
    for time in range(start, end+1):
        print("Process time {}".format(time))
        edge_path = "data/{}/{}.edge.{}".format(dataset, dataset, time)
        edge_file = open(edge_path, "r")
        edge_outpath = "data/{}/{}.edge.{}".format(dataset_new, dataset_new, time)
        edge_outfile = open(edge_outpath, "w")
        edges = edge_file.readlines()
        edges = edges[2:]
        out_edges = []
        for line in edges:
            node1 = int(line.split('\t')[0].strip())
            node2 = int(line.split('\t')[1].strip())
            if node1 in node_ids and node2 in node_ids:
                out_edges.append(line)
        edge_num = len(out_edges)
        edge_outfile.write("#Nodes\t{}\n".format(node_num))
        edge_outfile.write("#Edges\t{}\n".format(edge_num))
        edge_outfile.writelines(out_edges)

        attr_path = "data/{}/{}.attr.{}".format(dataset, dataset, time)
        attr_file = open(attr_path, 'r')
        attr_outpath = "data/{}/{}.attr.{}".format(dataset_new, dataset_new, time)
        attr_outfile = open(attr_outpath, "w")
        attributes = attr_file.readlines()
        out_attributes = []
        attributes = attributes[2:]
        for line in attributes:
            node1 = int(line.split('\t')[0].strip())
            attribute1 = int(line.split('\t')[1].strip())
            if node1 in node_ids and attribute1 in attr_ids:
                out_attributes.append(line)
        assc_num = len(out_attributes)
        attr_outfile.write("#Attributes\t{}\n".format(attr_num))
        attr_outfile.write("#Associations\t{}\n".format(assc_num))
        attr_outfile.writelines(out_attributes)

# sample_sub_snapshots("DBLP")
# adjs, attributes = load_adjs_and_attrs("DBLP_sub")
# print(adjs.shape, attributes.shape)
