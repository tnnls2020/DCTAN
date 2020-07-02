import utils
import load_data
import numpy as np
import torch

adjs, attributes = load_data.load_data("DBLP_sub")
adj = adjs[-2:].sum(0)
print(adj.shape)
for val in range(0, 15):
    print(val, len(np.where(adj==val)[0]))
